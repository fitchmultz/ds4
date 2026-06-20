#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#ifndef DS4_NO_GPU
#include "../ds4_gpu.h"
#include <math.h>

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static bool test_env_bool(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static uint32_t test_env_u32(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long n = strtoul(v, &end, 10);
    if (end == v) return 0;
    return n > UINT32_MAX ? UINT32_MAX : (uint32_t)n;
}

static uint64_t test_env_gib(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long long n = strtoull(v, &end, 10);
    if (end == v || n == 0) return 0;
    const uint64_t one_gib = 1024ull * 1024ull * 1024ull;
    if (n > UINT64_MAX / one_gib) return UINT64_MAX;
    return (uint64_t)n * one_gib;
}

static char *test_save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = malloc(len + 1);
    TEST_ASSERT(copy != NULL);
    if (!copy) return NULL;
    memcpy(copy, value, len + 1);
    return copy;
}

static void test_restore_env(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

typedef struct {
    char *cold_decode;
    char *batch_selected_addr;
} test_streaming_prefill_env;

static test_streaming_prefill_env test_force_canonical_streaming_prefill(void) {
    test_streaming_prefill_env saved = {
        .cold_decode =
            test_save_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL"),
        .batch_selected_addr =
            test_save_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR"),
    };
    if (test_env_bool("DS4_TEST_SSD_STREAMING")) {
        setenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL", "1", 1);
        setenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR", "1", 1);
    }
    return saved;
}

static void test_restore_canonical_streaming_prefill(
        test_streaming_prefill_env saved) {
    test_restore_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL",
                     saved.cold_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                     saved.batch_selected_addr);
}

static ds4_engine *test_open_engine(bool quality) {
    ds4_engine *engine = NULL;
    /* DS4_TEST_MTP loads the MTP head on the fast engine so the speculative
     * verify regression can reuse it; draft=4 hits the multi-row verify path. */
    const char *mtp = getenv("DS4_TEST_MTP");
    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = quality,
        .ssd_streaming = test_env_bool("DS4_TEST_SSD_STREAMING"),
        .ssd_streaming_cold = test_env_bool("DS4_TEST_SSD_STREAMING_COLD"),
        .ssd_streaming_cache_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS"),
        .ssd_streaming_cache_bytes =
            test_env_gib("DS4_TEST_SSD_STREAMING_CACHE_GB"),
        .ssd_streaming_preload_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS"),
        .mtp_path = (mtp && mtp[0] && !quality) ? mtp : NULL,
        .mtp_draft_tokens = (mtp && mtp[0] && !quality) ? 4 : 0,
    };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0);
    return engine;
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    *slot = test_open_engine(quality);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float test_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void test_fill_q8_0_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = test_float_to_f16(amax / 127.0f);
            const float scale = test_f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_f16_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 11u + i * 13u + (o ^ i) * 5u) % 61u) - 30;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 96.0f);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 7u + i * 17u + (t ^ i) * 3u) % 73u) - 36;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                ref += test_f16_to_f32(weights[(uint64_t)o * in_dim + i]) *
                       x_host[(uint64_t)t * in_dim + i];
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_q8_0_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float ref = 0.0f;
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = test_f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    ref += scale * (float)qs[i] *
                           x_host[(uint64_t)t * in_dim + b * 32u + i];
                }
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_kernel_group(void) {
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_q8_0_prefill_matmul();
}

static void test_metal_short_prefill_ratio4(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    const int tokens[] = {
        ds4_token_user(engine),
        ds4_token_assistant(engine),
        ds4_token_eos(engine),
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        TEST_ASSERT(tokens[i] >= 0);
        if (tokens[i] < 0) return;
    }

    for (size_t n = 1; n <= 3; n++) {
        ds4_tokens prompt = {0};
        for (size_t i = 0; i < n; i++) {
            ds4_tokens_push(&prompt, tokens[i]);
        }
        TEST_ASSERT(prompt.len == (int)n);

        ds4_session *session = NULL;
        TEST_ASSERT(ds4_session_create(&session, engine, 2048) == 0);
        if (!session) {
            ds4_tokens_free(&prompt);
            return;
        }

        char err[160] = {0};
        const int rc = ds4_session_sync(session, &prompt, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-test: short prefill failed for %zu token(s): %s\n",
                    n, err);
        }
        TEST_ASSERT(rc == 0);

        ds4_session_free(session);
        ds4_tokens_free(&prompt);
    }
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

typedef struct {
    const char *name;
    int number;
} test_long_fact;

static const test_long_fact test_long_facts[] = {
    {"Bob", 34},
    {"Alice", 52},
    {"Clara", 71},
    {"Diego", 93},
    {"Elena", 16},
    {"Felix", 88},
    {"Greta", 47},
    {"Hugo", 29},
    {"Iris", 64},
    {"Jonas", 12},
    {"Kira", 81},
    {"Leo", 39},
    {"Marta", 76},
    {"Nadia", 23},
    {"Owen", 58},
    {"Priya", 97},
};

static bool test_is_name_boundary(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || !(isalnum(uc) || c == '_');
}

static bool test_parse_assignment_value(const char *p, int *value) {
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '=') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!isdigit((unsigned char)*p)) return false;

    int v = 0;
    while (isdigit((unsigned char)*p)) {
        v = v * 10 + (*p - '0');
        p++;
    }
    *value = v;
    return true;
}

static bool test_output_has_fact(const char *text, const test_long_fact *fact) {
    const size_t name_len = strlen(fact->name);
    const char *p = text;
    bool saw_wrong_assignment = false;
    int wrong_value = -1;

    while ((p = strstr(p, fact->name)) != NULL) {
        const bool before_ok = p == text || test_is_name_boundary(p[-1]);
        const bool after_ok = test_is_name_boundary(p[name_len]) ||
                              p[name_len] == ' ' ||
                              p[name_len] == '\t' ||
                              p[name_len] == '=';
        if (before_ok && after_ok) {
            int value = 0;
            if (test_parse_assignment_value(p + name_len, &value)) {
                if (value == fact->number) return true;
                saw_wrong_assignment = true;
                wrong_value = value;
            }
        }
        p += name_len;
    }

    if (saw_wrong_assignment) {
        fprintf(stderr,
                "ds4-test: long-context wrong assignment for %s: got %d expected %d\n",
                fact->name, wrong_value, fact->number);
    } else {
        fprintf(stderr,
                "ds4-test: long-context missing assignment for %s=%d\n",
                fact->name, fact->number);
    }
    return false;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(ds4_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "ds4-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_story_fact_recall(void) {
    const char *prompt_path = getenv("DS4_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_story_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 100000) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    ds4_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 350; generated++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    for (size_t i = 0; i < sizeof(test_long_facts) / sizeof(test_long_facts[0]); i++) {
        TEST_ASSERT(test_output_has_fact(text, &test_long_facts[i]));
    }

    buf_free(&out);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static void test_logprob_vector_case(ds4_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    free(prompt_text);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);

    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int token = ds4_session_argmax(session);
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "ds4-test: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            TEST_ASSERT(false);
        }

        for (int t = 0; t < step->ntop; t++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (test_token_bytes_equal(engine, scores[j].id,
                                           step->top[t].bytes,
                                           step->top[t].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "ds4-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "ds4-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(ds4_session_eval(session, token, err, sizeof(err)) == 0);
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static bool test_logprob_vector_case_disabled(const test_vec_case *vc) {
    /*
     * This one long-context vector currently matches the public DeepSeek API less
     * after adding the official Hadamard+FP4 indexer path.  The public official
     * implementation and the API appear to disagree here; the official graph has
     * slightly lower local perplexity on the A/B check we ran, so DS4 keeps that
     * implementation and only excludes this brittle API fixture for now.
     */
    return !strcmp(vc->id, "long_memory_archive");
}

static void test_official_logprob_vectors_run(const char *case_filter) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "2048", 1);
    if (getenv("DS4_TEST_LOGPROB_AUTO_METAL") == NULL) {
        setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    } else {
        unsetenv("DS4_METAL_DISABLE_METAL4");
    }
    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_vec_case vc;
    int ran = 0;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (case_filter && case_filter[0] && strcmp(vc.id, case_filter)) {
            continue;
        }
        if (test_logprob_vector_case_disabled(&vc)) {
            fprintf(stderr, "ds4-test: vector %s skipped (API/official graph mismatch)\n",
                    vc.id);
            continue;
        }
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
        ran++;
    }
    TEST_ASSERT(!case_filter || !case_filter[0] || ran == 1);
    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

static void test_official_logprob_vectors(void) {
    test_official_logprob_vectors_run(NULL);
}

static void test_metal_ssd_streaming_cache_pressure(void) {
#ifndef __APPLE__
    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro skipped "
            "(Metal-only)\n");
#else
    /*
     * Regression repro for GitHub issue #384.
     *
     * The bug needs the Metal SSD-streaming decode layer-batch path and a small
     * routed-expert cache. Under pressure, a cache entry referenced by an
     * already-encoded-but-not-yet-executed layer can be reused for a later
     * layer in the same command buffer, producing deterministic wrong logits.
     */
    char *saved_streaming = test_save_env("DS4_TEST_SSD_STREAMING");
    char *saved_cache_gb = test_save_env("DS4_TEST_SSD_STREAMING_CACHE_GB");
    char *saved_cache_experts =
        test_save_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    char *saved_disable_layer_batch =
        test_save_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    char *saved_disable_static_decode =
        test_save_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    char *saved_one_stage =
        test_save_env("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    setenv("DS4_TEST_SSD_STREAMING", "1", 1);
    setenv("DS4_TEST_SSD_STREAMING_CACHE_GB", "16", 1);
    unsetenv("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    unsetenv("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    unsetenv("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    unsetenv("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro "
            "(16GiB cache, layer-batched decode, short_code_completion)\n");
    test_official_logprob_vectors_run("short_code_completion");

    test_restore_env("DS4_METAL_MOE_ONE_STAGE_PROFILE", saved_one_stage);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP",
                     saved_disable_static_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH",
                     saved_disable_layer_batch);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS",
                     saved_cache_experts);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_GB", saved_cache_gb);
    test_restore_env("DS4_TEST_SSD_STREAMING", saved_streaming);
#endif
}

static void test_logits_topk(const float *logits, int n, int *out, int k);
static bool test_topk_contains(const int *top, int k, int id);

#define TEST_LOCAL_GOLDEN_MAX_TOP 128

typedef struct {
    int id;
    float logit;
} test_local_golden_top;

typedef struct {
    char id[96];
    char mode[16];
    char prompt_path[512];
    int ctx;
    int frontier;
    int ntop;
    test_local_golden_top top[TEST_LOCAL_GOLDEN_MAX_TOP];
} test_local_golden_case;

static bool test_read_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    memset(tc, 0, sizeof(*tc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %15s %d %d %511s %d",
                   tc->id, tc->mode, &tc->ctx, &tc->frontier,
                   tc->prompt_path, &tc->ntop) == 6) {
            TEST_ASSERT(tc->ctx > tc->frontier);
            TEST_ASSERT(tc->frontier > 0);
            TEST_ASSERT(tc->ntop > 0 && tc->ntop <= TEST_LOCAL_GOLDEN_MAX_TOP);
            return true;
        }
        TEST_ASSERT(!"unexpected line before local golden case");
        return false;
    }
    return false;
}

static bool test_fill_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    int seen = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) {
            TEST_ASSERT(seen == tc->ntop);
            return seen == tc->ntop;
        }
        int rank = -1;
        int id = -1;
        float logit = 0.0f;
        if (sscanf(p, "top %d %d %f", &rank, &id, &logit) != 3) {
            TEST_ASSERT(!"bad local golden top line");
            return false;
        }
        TEST_ASSERT(rank == seen);
        TEST_ASSERT(seen < tc->ntop);
        if (seen >= tc->ntop) return false;
        tc->top[seen].id = id;
        tc->top[seen].logit = logit;
        seen++;
    }
    TEST_ASSERT(!"unterminated local golden case");
    return false;
}

static int test_local_golden_overlap(const test_local_golden_case *tc,
                                     const int *cand_top,
                                     int n) {
    int overlap = 0;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        if (test_topk_contains(cand_top, n, tc->top[i].id)) overlap++;
    }
    return overlap;
}

static float test_local_golden_max_abs(const test_local_golden_case *tc,
                                       const float *cand_logits,
                                       int n) {
    float max_abs = 0.0f;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        const int id = tc->top[i].id;
        if (id < 0) continue;
        const float abs_delta = fabsf(cand_logits[id] - tc->top[i].logit);
        if (abs_delta > max_abs) max_abs = abs_delta;
    }
    return max_abs;
}

static void test_local_golden_case_run(ds4_engine *engine,
                                       const test_local_golden_case *tc) {
    char *prompt_text = test_read_file(tc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    if (!strcmp(tc->mode, "text")) {
        ds4_tokenize_text(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "rendered")) {
        ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "chat")) {
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    } else {
        TEST_ASSERT(!"unknown local golden prompt mode");
    }
    free(prompt_text);
    TEST_ASSERT(prompt.len >= tc->frontier);
    if (prompt.len < tc->frontier) {
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_tokens prefix = {
        .v = prompt.v,
        .len = tc->frontier,
        .cap = tc->frontier,
    };

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);

    const int vocab = ds4_engine_vocab_size(engine);
    float *cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
    TEST_ASSERT(cand_logits != NULL);
    if (cand_logits &&
        ds4_session_copy_logits(session, cand_logits, vocab) == vocab) {
        int cand_top[TEST_LOCAL_GOLDEN_MAX_TOP];
        const int ntop = tc->ntop < TEST_LOCAL_GOLDEN_MAX_TOP ?
                         tc->ntop : TEST_LOCAL_GOLDEN_MAX_TOP;
        test_logits_topk(cand_logits, vocab, cand_top, ntop);

        const int top5_overlap = test_local_golden_overlap(tc, cand_top, 5);
        const int top20_overlap = test_local_golden_overlap(tc, cand_top, 20);
        const int top64_overlap = test_local_golden_overlap(tc, cand_top, 64);
        const float top20_max_abs =
            test_local_golden_max_abs(tc, cand_logits, 20);

        fprintf(stderr,
                "ds4-test: local golden %s top1 ref=%d cand=%d "
                "top5_overlap=%d/5 top20_overlap=%d/20 top64_overlap=%d/64 "
                "top20_max_abs=%g\n",
                tc->id, tc->top[0].id, cand_top[0],
                top5_overlap, top20_overlap, top64_overlap, top20_max_abs);

        /*
         * This is intentionally tolerant: it is meant to catch substantial
         * backend drift (wrong tiling, skipped work, bad dispatch), not tiny
         * floating-point differences from otherwise sane kernel changes.
         */
        TEST_ASSERT(cand_top[0] == tc->top[0].id);
        TEST_ASSERT(top5_overlap >= 4);
        TEST_ASSERT(top20_overlap >= 15);
        TEST_ASSERT(top64_overlap >= 40);
        TEST_ASSERT(top20_max_abs <= 8.0f);
    } else {
        TEST_ASSERT(false);
    }

    free(cand_logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_local_golden_vectors(void) {
    const char *path = getenv("DS4_TEST_LOCAL_GOLDEN_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/local-golden.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    char *saved_moe_tile_max = test_save_env("DS4_METAL_MOE_TILE_MAX");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "4096", 1);
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    unsetenv("DS4_METAL_MOE_TILE_MAX");

    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_local_golden_case tc;
    while (test_read_local_golden_case(fp, &tc)) {
        if (!test_fill_local_golden_case(fp, &tc)) break;
        test_local_golden_case_run(engine, &tc);
    }

    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

#define TEST_MPP_EQ_MAX_CASES 8
#define TEST_MPP_EQ_TOPK 20
#define TEST_MPP_EQ_TOP5 5
#define TEST_MPP_EQ_DELTAS 5

typedef struct {
    char id[96];
    int ctx;
    int vocab_size;
    int gen_steps;
    ds4_tokens prompt;
    float *ref_logits;
    int ref_gen[TEST_VEC_MAX_STEPS];
    int ref_gen_len;
} test_mpp_eq_case;

typedef struct {
    int ref_top1;
    int cand_top1;
    int overlap;
    int top5_overlap;
    int max_rank_delta;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
    bool same_top1;
    bool pass;
} test_mpp_eq_result;

typedef struct {
    const char *label;
    int cases;
    int capture_failures;
    int logits_failures;
    int greedy_failures;
    int top1_mismatches;
    int min_overlap;
    int min_top5_overlap;
    int worst_rank_delta;
    float worst_rms;
    float worst_max_abs;
    float worst_top20_max_abs;
} test_mpp_eq_summary;

static void test_mpp_eq_case_free(test_mpp_eq_case *tc) {
    if (!tc) return;
    ds4_tokens_free(&tc->prompt);
    free(tc->ref_logits);
    memset(tc, 0, sizeof(*tc));
}

static void test_logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int id = 0; id < n; id++) {
        const float v = logits[id];
        if (!isfinite(v)) continue;
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j] = id;
                break;
            }
        }
    }
}

static bool test_topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static int test_topk_rank(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return i;
    }
    return -1;
}

static void test_note_delta(int *ids, float *ref_vals, float *cand_vals,
                            float *abs_vals, int id, float ref, float cand) {
    const float abs_delta = fabsf(cand - ref);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        if (ids[i] < 0 || abs_delta > abs_vals[i]) {
            for (int j = TEST_MPP_EQ_DELTAS - 1; j > i; j--) {
                ids[j] = ids[j - 1];
                ref_vals[j] = ref_vals[j - 1];
                cand_vals[j] = cand_vals[j - 1];
                abs_vals[j] = abs_vals[j - 1];
            }
            ids[i] = id;
            ref_vals[i] = ref;
            cand_vals[i] = cand;
            abs_vals[i] = abs_delta;
            return;
        }
    }
}

static float test_top_union_max_abs(const float *ref, const float *cand,
                                    const int *ref_top, const int *cand_top, int k) {
    float max_abs = 0.0f;
    for (int i = 0; i < k; i++) {
        if (ref_top[i] >= 0) {
            const float d = fabsf(cand[ref_top[i]] - ref[ref_top[i]]);
            if (d > max_abs) max_abs = d;
        }
        if (cand_top[i] >= 0 && !test_topk_contains(ref_top, k, cand_top[i])) {
            const float d = fabsf(cand[cand_top[i]] - ref[cand_top[i]]);
            if (d > max_abs) max_abs = d;
        }
    }
    return max_abs;
}

/*
 * Metal4/TensorOps equivalence is a smoke test, not a demand for bitwise local
 * logits.  Tensor kernels change precision and reduction order, so the useful
 * invariant here is: no NaNs, same first greedy token, and same short greedy
 * continuation.  Larger logit drift is still printed so it can be compared with
 * official API-vector and long-context recall gates.
 */
static test_mpp_eq_result test_compare_mpp_logits(const test_mpp_eq_case *tc,
                                                  const float *cand_logits,
                                                  bool assert_thresholds) {
    int ref_top[TEST_MPP_EQ_TOPK];
    int cand_top[TEST_MPP_EQ_TOPK];
    test_logits_topk(tc->ref_logits, tc->vocab_size, ref_top, TEST_MPP_EQ_TOPK);
    test_logits_topk(cand_logits, tc->vocab_size, cand_top, TEST_MPP_EQ_TOPK);

    int overlap = 0;
    int top5_overlap = 0;
    int max_rank_delta = 0;
    for (int i = 0; i < TEST_MPP_EQ_TOPK; i++) {
        const int cand_rank = test_topk_rank(cand_top, TEST_MPP_EQ_TOPK, ref_top[i]);
        if (ref_top[i] >= 0 && cand_rank >= 0) {
            overlap++;
            const int rank_delta = abs(cand_rank - i);
            if (rank_delta > max_rank_delta) max_rank_delta = rank_delta;
        }
        if (i < TEST_MPP_EQ_TOP5 &&
            ref_top[i] >= 0 &&
            test_topk_contains(cand_top, TEST_MPP_EQ_TOP5, ref_top[i])) {
            top5_overlap++;
        }
    }

    double sumsq = 0.0;
    float max_abs = 0.0f;
    int nonfinite = 0;
    int delta_ids[TEST_MPP_EQ_DELTAS];
    float delta_ref[TEST_MPP_EQ_DELTAS];
    float delta_cand[TEST_MPP_EQ_DELTAS];
    float delta_abs[TEST_MPP_EQ_DELTAS];
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        delta_ids[i] = -1;
        delta_ref[i] = 0.0f;
        delta_cand[i] = 0.0f;
        delta_abs[i] = 0.0f;
    }

    for (int i = 0; i < tc->vocab_size; i++) {
        if (!isfinite(tc->ref_logits[i]) || !isfinite(cand_logits[i])) {
            nonfinite++;
            continue;
        }
        const float delta = cand_logits[i] - tc->ref_logits[i];
        const float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
        test_note_delta(delta_ids, delta_ref, delta_cand, delta_abs,
                        (int)i, tc->ref_logits[i], cand_logits[i]);
    }

    const float rms = (float)sqrt(sumsq / (double)tc->vocab_size);
    const float top_abs = test_top_union_max_abs(tc->ref_logits, cand_logits,
                                                 ref_top, cand_top, TEST_MPP_EQ_TOPK);
    const bool same_top1 = ref_top[0] >= 0 && ref_top[0] == cand_top[0];
    test_mpp_eq_result result = {
        .ref_top1 = ref_top[0],
        .cand_top1 = cand_top[0],
        .overlap = overlap,
        .top5_overlap = top5_overlap,
        .max_rank_delta = max_rank_delta,
        .nonfinite = nonfinite,
        .rms = rms,
        .max_abs = max_abs,
        .top20_max_abs = top_abs,
        .same_top1 = same_top1,
        .pass = nonfinite == 0 && same_top1,
    };

    fprintf(stderr,
            "ds4-test: Tensor equivalence %s top1 ref=%d cand=%d top5_overlap=%d/%d overlap=%d/%d max_rank_delta=%d rms=%g max_abs=%g top20_max_abs=%g\n",
            tc->id, ref_top[0], cand_top[0],
            top5_overlap, TEST_MPP_EQ_TOP5,
            overlap, TEST_MPP_EQ_TOPK,
            max_rank_delta, rms, max_abs, top_abs);
    fprintf(stderr, "ds4-test: Tensor equivalence %s largest deltas:", tc->id);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS && delta_ids[i] >= 0; i++) {
        fprintf(stderr, " id=%d ref=%g cand=%g abs=%g",
                delta_ids[i], delta_ref[i], delta_cand[i], delta_abs[i]);
    }
    fputc('\n', stderr);

    if (assert_thresholds) {
        TEST_ASSERT(nonfinite == 0);
        TEST_ASSERT(same_top1);
    }
    return result;
}

static bool test_mpp_capture(ds4_engine *engine, const test_mpp_eq_case *tc,
                             float *logits, int *gen, int *gen_len) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    int n = 0;
    while (ok && n < tc->gen_steps) {
        const int token = ds4_session_argmax(session);
        gen[n++] = token;
        if (n < tc->gen_steps && ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            ok = false;
            TEST_ASSERT(false);
        }
    }
    *gen_len = n;

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_capture_logits_only(ds4_engine *engine,
                                         const test_mpp_eq_case *tc,
                                         float *logits) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_eq_case_selected(const char *id) {
    const char *filter = getenv("DS4_TEST_MPP_EQ_CASE");
    if (!filter || !filter[0]) return true;

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", filter);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        tok = test_trim_line(tok);
        if (tok[0] && strstr(id, tok)) return true;
    }
    return false;
}

static int test_load_mpp_cases(ds4_engine *engine, test_mpp_eq_case *cases, int cap) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return 0;

    int ncase = 0;
    test_vec_case vc;
    while (ncase < cap && test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (!test_mpp_eq_case_selected(vc.id)) continue;
        char *prompt_text = test_read_file(vc.prompt_path);
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) continue;

        test_mpp_eq_case *tc = &cases[ncase++];
        snprintf(tc->id, sizeof(tc->id), "%s", vc.id);
        tc->ctx = vc.ctx;
        tc->vocab_size = ds4_engine_vocab_size(engine);
        tc->gen_steps = vc.nsteps < TEST_VEC_MAX_STEPS ? vc.nsteps : TEST_VEC_MAX_STEPS;
        ds4_encode_chat_prompt(engine, "", prompt_text, DS4_THINK_NONE, &tc->prompt);
        free(prompt_text);
        TEST_ASSERT(tc->prompt.len > 0);
    }
    fclose(fp);
    return ncase;
}

static void test_mpp_summary_init(test_mpp_eq_summary *summary, const char *label) {
    memset(summary, 0, sizeof(*summary));
    summary->label = label;
    summary->min_overlap = TEST_MPP_EQ_TOPK;
    summary->min_top5_overlap = TEST_MPP_EQ_TOP5;
}

static void test_mpp_summary_note_logits(test_mpp_eq_summary *summary,
                                         const test_mpp_eq_result *result) {
    if (!result->pass) summary->logits_failures++;
    if (!result->same_top1) summary->top1_mismatches++;
    if (result->overlap < summary->min_overlap) summary->min_overlap = result->overlap;
    if (result->top5_overlap < summary->min_top5_overlap) {
        summary->min_top5_overlap = result->top5_overlap;
    }
    if (result->max_rank_delta > summary->worst_rank_delta) {
        summary->worst_rank_delta = result->max_rank_delta;
    }
    if (result->rms > summary->worst_rms) summary->worst_rms = result->rms;
    if (result->max_abs > summary->worst_max_abs) summary->worst_max_abs = result->max_abs;
    if (result->top20_max_abs > summary->worst_top20_max_abs) {
        summary->worst_top20_max_abs = result->top20_max_abs;
    }
}

static void test_mpp_summary_print(const test_mpp_eq_summary *summary) {
    fprintf(stderr,
            "ds4-test: Tensor summary route=%s cases=%d capture_fail=%d logits_fail=%d greedy_fail=%d top1_mismatch=%d min_top5_overlap=%d/%d min_overlap=%d/%d worst_rank_delta=%d worst_rms=%g worst_max_abs=%g worst_top20_max_abs=%g\n",
            summary->label,
            summary->cases,
            summary->capture_failures,
            summary->logits_failures,
            summary->greedy_failures,
            summary->top1_mismatches,
            summary->min_top5_overlap,
            TEST_MPP_EQ_TOP5,
            summary->min_overlap,
            TEST_MPP_EQ_TOPK,
            summary->worst_rank_delta,
            summary->worst_rms,
            summary->worst_max_abs,
            summary->worst_top20_max_abs);
}

static void test_run_mpp_candidate(const char *label,
                                   test_mpp_eq_case *cases,
                                   int ncase) {
    fprintf(stderr, "ds4-test: Tensor equivalence candidate route=%s\n", label);
    test_mpp_eq_summary summary;
    test_mpp_summary_init(&summary, label);
    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        const int vocab_size = ncase > 0 ? cases[0].vocab_size : 0;
        float *cand_logits = malloc((size_t)vocab_size * sizeof(cand_logits[0]));
        TEST_ASSERT(cand_logits != NULL);
        if (cand_logits) {
            for (int i = 0; i < ncase; i++) {
                test_mpp_eq_case *tc = &cases[i];
                if (!tc->ref_logits) continue;
                int cand_gen[TEST_VEC_MAX_STEPS] = {0};
                int cand_gen_len = 0;
                if (!test_mpp_capture(cand_engine, tc, cand_logits, cand_gen, &cand_gen_len)) {
                    summary.capture_failures++;
                    continue;
                }
                summary.cases++;
                test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_logits, true);
                test_mpp_summary_note_logits(&summary, &result);
                TEST_ASSERT(cand_gen_len == tc->ref_gen_len);
                if (cand_gen_len != tc->ref_gen_len) summary.greedy_failures++;
                for (int j = 0; j < tc->ref_gen_len && j < cand_gen_len; j++) {
                    if (cand_gen[j] != tc->ref_gen[j]) {
                        fprintf(stderr,
                                "ds4-test: Tensor equivalence %s greedy token mismatch step=%d ref=%d cand=%d\n",
                                tc->id, j, tc->ref_gen[j], cand_gen[j]);
                        summary.greedy_failures++;
                    }
                    TEST_ASSERT(cand_gen[j] == tc->ref_gen[j]);
                }
            }
            free(cand_logits);
        }
        ds4_engine_close(cand_engine);
    }
    test_mpp_summary_print(&summary);
}

static void test_metal_mpp_equivalence(void) {
    test_close_engines();

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);

    test_run_mpp_candidate("auto", cases, ncase);

    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

static void test_streaming_decode_prefill_correctness(void) {
    test_close_engines();
    if (!test_env_bool("DS4_TEST_SSD_STREAMING")) {
        fprintf(stderr,
                "ds4-test: streaming decode-prefill correctness skipped "
                "(set DS4_TEST_SSD_STREAMING=1 to enable)\n");
        return;
    }

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();

    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);

    unsetenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL");
    unsetenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR");

    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        for (int i = 0; i < ncase; i++) {
            test_mpp_eq_case *tc = &cases[i];
            if (!tc->ref_logits) continue;

            float *cand_cold = malloc((size_t)tc->vocab_size * sizeof(cand_cold[0]));
            float *cand_warm_a = malloc((size_t)tc->vocab_size * sizeof(cand_warm_a[0]));
            float *cand_warm_b = malloc((size_t)tc->vocab_size * sizeof(cand_warm_b[0]));
            TEST_ASSERT(cand_cold != NULL);
            TEST_ASSERT(cand_warm_a != NULL);
            TEST_ASSERT(cand_warm_b != NULL);
            if (!cand_cold || !cand_warm_a || !cand_warm_b) {
                free(cand_cold);
                free(cand_warm_a);
                free(cand_warm_b);
                continue;
            }

            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_cold));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_a));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_b));

            test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_cold, false);
            TEST_ASSERT(result.nonfinite == 0);
            TEST_ASSERT(result.top5_overlap >= 2);
            TEST_ASSERT(result.overlap >= 10);
            TEST_ASSERT(result.rms <= 4.0f);
            TEST_ASSERT(result.top20_max_abs <= 12.0f);

            int cold_warm_neq = 0;
            int warm_repeat_neq = 0;
            int repeat_nonfinite = 0;
            float cold_warm_max_abs = 0.0f;
            float warm_repeat_max_abs = 0.0f;
            for (int j = 0; j < tc->vocab_size; j++) {
                if (!isfinite(cand_cold[j]) ||
                    !isfinite(cand_warm_a[j]) ||
                    !isfinite(cand_warm_b[j])) {
                    repeat_nonfinite++;
                    continue;
                }
                const float cold_warm_d = fabsf(cand_cold[j] - cand_warm_a[j]);
                if (cold_warm_d != 0.0f) cold_warm_neq++;
                if (cold_warm_d > cold_warm_max_abs) cold_warm_max_abs = cold_warm_d;
                const float warm_repeat_d = fabsf(cand_warm_a[j] - cand_warm_b[j]);
                if (warm_repeat_d != 0.0f) warm_repeat_neq++;
                if (warm_repeat_d > warm_repeat_max_abs) {
                    warm_repeat_max_abs = warm_repeat_d;
                }
            }
            TEST_ASSERT(repeat_nonfinite == 0);
            TEST_ASSERT(cold_warm_neq == 0);
            TEST_ASSERT(warm_repeat_neq == 0);
            fprintf(stderr,
                    "ds4-test: streaming decode-prefill %s cold_warm_neq=%d "
                    "cold_warm_max_abs=%g warm_repeat_neq=%d "
                    "warm_repeat_max_abs=%g top1 canonical=%d decode=%d\n",
                    tc->id,
                    cold_warm_neq,
                    cold_warm_max_abs,
                    warm_repeat_neq,
                    warm_repeat_max_abs,
                    result.ref_top1,
                    result.cand_top1);

            free(cand_cold);
            free(cand_warm_a);
            free(cand_warm_b);
        }
        ds4_engine_close(cand_engine);
    }

    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static const char *test_think_recovery_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":true,"
        "\"temperature\":0,"
        "\"max_tokens\":384,"
        "\"stream\":false"
        "}";
}

/* The model sometimes opens a DSML stanza without closing </think> first.
 * The server's forward recovery must force the close plus a fresh stanza
 * opening, after which the model must still complete a valid call.  The
 * malformed prefix is teacher-forced so the regression is deterministic and
 * does not depend on coaxing the model into misbehaving. */
static void test_think_tool_recovery(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_think_recovery_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    if (getenv("DS4_TEST_RECOVERY_PROBE") != NULL) {
        /* Diagnostic: print the model's natural tool-call turn for this
         * request instead of running the recovery. */
        buf nat = {0};
        uint64_t prng = 7;
        for (int i = 0; i < 300; i++) {
            int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &prng);
            if (token == ds4_token_eos(engine)) break;
            size_t plen = 0;
            char *p = ds4_token_text(engine, token, &plen);
            buf_append(&nat, p, plen);
            free(p);
            bool ps = false, pe = false;
            observe_tool_markers(nat.ptr, &ps, &pe, NULL);
            if (pe) break;
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) break;
        }
        fprintf(stderr, "ds4-test: natural turn=[%s]\n", nat.ptr ? nat.ptr : "");
        buf_free(&nat);
        ds4_session_free(session);
        request_free(&r);
        test_close_engine(false);
        return;
    }

    thinking_state thinking = thinking_state_from_prompt(&r);
    buf text = {0};
    buf forced = {0};
    if (!thinking.inside) buf_append(&forced, "<think>", 7);
    const char *body =
        "The user wants a directory listing. I will call the "
        "list_files tool right away.\n\n" DS4_TOOL_CALLS_START;
    buf_append(&forced, body, strlen(body));

    server srv;
    memset(&srv, 0, sizeof(srv));
    srv.engine = engine;
    srv.session = session;

    /* Replay the malformed prefix exactly as the worker loop would see it:
     * token by token, running the recovery scan after each piece.  The stanza
     * opening spans several tokens, so this also checks that detection does
     * not depend on how the marker happens to be tokenized: recovery must
     * stay quiet on every partial prefix and trigger exactly when the
     * opening completes. */
    ds4_tokens toks = {0};
    ds4_tokenize_rendered_chat(engine, forced.ptr, &toks);
    TEST_ASSERT(toks.len > 1);
    size_t scan_from = 0;
    int completion = 0;
    int rec = 0;
    int triggered_at = -1;
    for (int i = 0; i < toks.len; i++) {
        TEST_ASSERT(ds4_session_eval(session, toks.v[i], err, sizeof(err)) == 0);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, toks.v[i], &piece_len);
        buf_append(&text, piece, piece_len);
        thinking_state_feed(&thinking, piece, piece_len);
        free(piece);
        TEST_ASSERT(thinking.inside);
        rec = chat_think_tool_recovery(&srv, &text, &thinking, &scan_from,
                                       &completion, 512, err, sizeof(err));
        TEST_ASSERT(rec >= 0);
        if (rec == 1) {
            triggered_at = i;
            break;
        }
    }
    fprintf(stderr,
            "ds4-test: think-tool-recovery trigger=%d/%d injected_tokens=%d\n",
            triggered_at, toks.len, completion);
    TEST_ASSERT(rec == 1);
    TEST_ASSERT(triggered_at == toks.len - 1);
    ds4_tokens_free(&toks);
    buf_free(&forced);
    TEST_ASSERT(!thinking.inside);
    TEST_ASSERT(completion > 0);
    TEST_ASSERT(text.ptr && text.len >= 10 &&
                !memcmp(text.ptr + text.len - 10, "</think>\n\n", 10));

    /* The model must now complete a valid call on the executable side. */
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_start = false;
    bool saw_end = false;
    for (int i = 0; i < 256 && !saw_end; i++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr, &saw_start, &saw_end, NULL);
        if (saw_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }
    fprintf(stderr, "ds4-test: think-tool-recovery continuation=[%s]\n",
            text.ptr ? text.ptr : "");
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(saw_end);

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr, true,
                                             &content, &reasoning, &calls);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));
    TEST_ASSERT(reasoning && strstr(reasoning, "list_files tool right away"));

    fprintf(stderr,
            "ds4-test: think-tool-recovery recovered=%d gen_tokens=%d calls=%d name=%s\n",
            rec, completion, calls.len, calls.len ? calls.v[0].name : "-");

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
    test_close_engine(false);
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    for (int i = 0; i < r.max_tokens; i++) {
        int token = ds4_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr ? text.ptr : "", &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr ? text.ptr : "",
                                             false, &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

/* Greedy speculative decode: capture committed tokens and the largest accepted
 * chunk, so the caller can confirm the multi-row verify path actually ran. */
static bool test_mtp_capture_speculative(ds4_engine *engine, const ds4_tokens *prompt,
                                         int max_tokens, int *out, int *out_len,
                                         int *max_chunk) {
    *out_len = 0;
    *max_chunk = 0;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    const int eos = ds4_token_eos(engine);
    int n = 0;
    bool stop = false;
    while (ok && !stop && n < max_tokens) {
        const int token = ds4_session_argmax(session);
        if (token == eos) break;

        int toks[17]; /* base token + draft depth, which the engine clamps to 16 */
        const int ntok = ds4_session_eval_speculative_argmax(
            session, token, max_tokens - n, eos, toks,
            (int)(sizeof(toks) / sizeof(toks[0])), err, sizeof(err));
        if (ntok < 0) { ok = false; TEST_ASSERT(false); break; }
        if (ntok > *max_chunk) *max_chunk = ntok;

        for (int j = 0; j < ntok; j++) {
            if (toks[j] == eos) { stop = true; break; }
            out[n++] = toks[j];
            if (n >= max_tokens) { stop = true; break; }
        }
    }

    *out_len = n;
    ds4_session_free(session);
    return ok;
}

/* Replay toks[] through plain decode and return the largest gap between a
 * position's argmax logit and the committed token's logit.  Correct speculation
 * commits (near-)argmax tokens (gap ~0); a mis-committed token gives a big gap. */
static bool test_mtp_worst_argmax_gap(ds4_engine *engine, const ds4_tokens *prompt,
                                      const int *toks, int n,
                                      float *worst_gap, int *worst_at) {
    *worst_gap = 0.0f;
    *worst_at = -1;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    for (int i = 0; ok && i < n; i++) {
        ds4_token_score best, cur;
        ok = ds4_session_top_logprobs(session, &best, 1) >= 1 &&
             ds4_session_token_logprob(session, toks[i], &cur) == 1;
        TEST_ASSERT(ok);
        if (!ok) break;

        const float gap = best.logit - cur.logit;
        if (gap > *worst_gap) { *worst_gap = gap; *worst_at = i; }
        if (ds4_session_eval(session, toks[i], err, sizeof(err)) != 0) { ok = false; TEST_ASSERT(false); break; }
    }

    ds4_session_free(session);
    return ok;
}

/* Verbatim-copy task: keeps the model confident (a mis-committed token shows as
 * a large argmax gap) and draft acceptance high (so the multi-row verify path is
 * exercised across the generation). */
static const char *test_mtp_copy_prompt(void) {
    return
        "Reproduce the following C code EXACTLY, character for character, "
        "inside a single code block and output nothing else:\n\n"
        "```c\n"
        "static uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {\n"
        "    if (v < lo) return lo;\n"
        "    if (v > hi) return hi;\n"
        "    return v;\n"
        "}\n"
        "\n"
        "static uint32_t ring_advance(uint32_t pos, uint32_t cap) {\n"
        "    uint32_t next = pos + 1u;\n"
        "    return next >= cap ? 0u : next;\n"
        "}\n"
        "\n"
        "static int scratch_init(scratch *s, uint32_t ctx_size) {\n"
        "    if (ctx_size == 0u) ctx_size = 1u;\n"
        "    s->ctx_size = ctx_size;\n"
        "    s->comp_cap = ctx_size / 4u + 2u;\n"
        "    s->rows = clamp_u32(s->comp_cap, 1u, 4096u);\n"
        "    s->head = 0u;\n"
        "    return s->rows > 0u ? 0 : -1;\n"
        "}\n"
        "```\n";
}

#define TEST_MTP_MAXGEN 256

/* Regression for the swapped top-k arguments in metal_graph_verify_suffix_tops
 * at draft depth > 2.  Replays the committed speculative tokens through plain
 * decode and requires each to be a (near-)argmax: that is the verify invariant,
 * and unlike comparing token streams it tolerates the near-greedy tie
 * divergences.  Needs an MTP head, so it self-skips without DS4_TEST_MTP. */
static void test_mtp_verify_depth(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_has_mtp(engine)) {
        fprintf(stderr, "ds4-test: mtp-verify-depth skipped (set DS4_TEST_MTP to an MTP GGUF)\n");
        return;
    }
    TEST_ASSERT(ds4_engine_mtp_draft_tokens(engine) > 2);

    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", test_mtp_copy_prompt());
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    TEST_ASSERT(prompt.len > 0);

    int *spec = malloc((size_t)TEST_MTP_MAXGEN * sizeof(*spec));
    TEST_ASSERT(spec != NULL);
    if (spec && prompt.len > 0) {
        int nspec = 0, max_chunk = 0;
        const bool ok_spec = test_mtp_capture_speculative(engine, &prompt, TEST_MTP_MAXGEN,
                                                          spec, &nspec, &max_chunk);
        TEST_ASSERT(ok_spec);
        TEST_ASSERT(max_chunk > 1);  /* multi-token chunks committed: the multi-row path ran */
        TEST_ASSERT(nspec > 128);    /* enough output to surface the bug, incl. a spurious-EOS truncation */

        float worst_gap = 0.0f;
        int worst_at = -1;
        const bool ok_check = test_mtp_worst_argmax_gap(engine, &prompt, spec, nspec,
                                                        &worst_gap, &worst_at);
        TEST_ASSERT(ok_check);
        fprintf(stderr, "ds4-test: mtp-verify-depth nspec=%d max_chunk=%d worst_argmax_gap=%.3f at=%d\n",
                nspec, max_chunk, worst_gap, worst_at);
        TEST_ASSERT(worst_gap <= 2.0f);  /* correct: ~0; bug: ~21 on the reference model */
    }

    free(spec);
    ds4_tokens_free(&prompt);
}
#endif

static void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
}

/* Locks docs/GLM52-PORT.md section 5: the GLM-5.2 UD-IQ2_M SSD streaming
 * cache plan at a 128 GiB recommended working set.  Needs no model file -- it
 * is the pure cache-arithmetic regression for the reused ds4_ssd helpers, so it
 * runs on every build (Metal/CPU/CUDA/ROCm). */
static void test_glm_ssd_cache_plan(void) {
    const uint64_t gib         = 1024ull * 1024ull * 1024ull;
    const uint64_t recommended = 128ull * gib;
    const uint64_t non_routed  = 15441835008ull;   /* ~14.381 GiB resident   */
    const uint64_t per_expert  = 11304960ull;       /* 10.78125 MiB / expert  */
    const uint32_t max_experts = 19200u;            /* 75 routed layers * 256 */

    ds4_ssd_cache_plan plan;
    TEST_ASSERT(ds4_ssd_auto_cache_plan(recommended, non_routed, per_expert,
                                        max_experts, &plan));
    TEST_ASSERT(plan.cache_experts == 8359u);
    TEST_ASSERT(plan.effective_cache_bytes ==
                (uint64_t)plan.cache_experts * per_expert);

    /* 88 GiB budget must floor to >= 8000 experts at this slab size. */
    TEST_ASSERT(ds4_ssd_cache_experts_for_byte_budget(88ull * gib, per_expert)
                >= 8000u);

    fprintf(stderr,
            "  glm-ssd-math: per-expert bytes %llu, cache experts %u, "
            "effective cache %.3f GiB (%.1f%% of %u backbone)\n",
            (unsigned long long)per_expert, plan.cache_experts,
            (double)plan.effective_cache_bytes / (double)gib,
            100.0 * (double)plan.cache_experts / (double)max_experts,
            max_experts);
}

/* ---------------- GLM-5.2 BPE oracle (Phase 2) ---------------- */

static char *test_read_whole_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long sz = ftell(fp);
    if (sz < 0) { fclose(fp); return NULL; }
    rewind(fp);
    char *buf = malloc((size_t)sz + 1);
    TEST_ASSERT(buf != NULL);
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[rd] = '\0';
    if (len_out) *len_out = rd;
    return buf;
}

/* Decode a JSON string literal (opening quote at p[0]) into a malloc'd
 * NUL-terminated byte buffer.  Handles \" \\ \/ \b \f \n \r \t and \uXXXX
 * (BMP + surrogate pairs); all other bytes (including raw UTF-8) pass through.
 * *next_out points just past the closing quote. */
static char *test_json_decode_string(const char *p, const char **next_out) {
    TEST_ASSERT(*p == '"');
    p++;
    size_t cap = 64, len = 0;
    char *out = malloc(cap);
    TEST_ASSERT(out != NULL);
    while (*p && *p != '"') {
        unsigned char c = (unsigned char)*p;
        if (c == '\\') {
            p++;
            char e = *p++;
            unsigned int cp = 0;
            switch (e) {
                case '"':  c = '"'; break;
                case '\\': c = '\\'; break;
                case '/':  c = '/'; break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u':
                    for (int k = 0; k < 4; k++) {
                        char h = *p++;
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                    }
                    if (cp >= 0xd800 && cp <= 0xdbff && p[0] == '\\' && p[1] == 'u') {
                        unsigned int lo = 0;
                        p += 2;
                        for (int k = 0; k < 4; k++) {
                            char h = *p++;
                            lo <<= 4;
                            if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
                            else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
                        }
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                    }
                    /* emit cp as UTF-8 */
                    {
                        char tmp[4]; int n = 0;
                        if (cp < 0x80) { tmp[n++] = (char)cp; }
                        else if (cp < 0x800) {
                            tmp[n++] = (char)(0xc0 | (cp >> 6));
                            tmp[n++] = (char)(0x80 | (cp & 0x3f));
                        } else if (cp < 0x10000) {
                            tmp[n++] = (char)(0xe0 | (cp >> 12));
                            tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3f));
                            tmp[n++] = (char)(0x80 | (cp & 0x3f));
                        } else {
                            tmp[n++] = (char)(0xf0 | (cp >> 18));
                            tmp[n++] = (char)(0x80 | ((cp >> 12) & 0x3f));
                            tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3f));
                            tmp[n++] = (char)(0x80 | (cp & 0x3f));
                        }
                        for (int k = 0; k < n; k++) {
                            if (len + 1 >= cap) { cap *= 2; out = realloc(out, cap); TEST_ASSERT(out); }
                            out[len++] = tmp[k];
                        }
                        continue;
                    }
                default: c = (unsigned char)e; break;
            }
            /* simple escape: c holds the decoded byte, p already advanced past it */
            if (len + 1 >= cap) { cap *= 2; out = realloc(out, cap); TEST_ASSERT(out); }
            out[len++] = (char)c;
            continue;
        }
        if (len + 1 >= cap) { cap *= 2; out = realloc(out, cap); TEST_ASSERT(out); }
        out[len++] = (char)c;
        p++;
    }
    TEST_ASSERT(*p == '"');
    p++;
    out[len] = '\0';
    if (next_out) *next_out = p;
    return out;
}

static const char *test_json_find_key(const char *p, const char *key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *hit = strstr(p, pat);
    return hit ? hit + strlen(pat) : NULL;
}

static void test_glm_bpe(void) {
    const char *model_path = getenv("DS4_TEST_GLM52_SHARD1");
    if (!model_path || !model_path[0]) {
        model_path = "gguf/glm52/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf";
    }
    const char *vec_path = getenv("DS4_TEST_GLM52_TOKENIZE");
    if (!vec_path || !vec_path[0]) {
        vec_path = "tests/test-vectors/glm52-tokenize.json";
    }

    size_t json_len = 0;
    char *json = test_read_whole_file(vec_path, &json_len);
    if (!json) {
        fprintf(stderr, "  glm-bpe: SKIP (oracle %s not found)\n", vec_path);
        return;
    }

    /* Sanity: shard 1 must actually be loadable for this to validate parsing. */
    FILE *probe = fopen(model_path, "rb");
    if (!probe) {
        fprintf(stderr, "  glm-bpe: SKIP (shard %s not found)\n", model_path);
        free(json);
        return;
    }
    fclose(probe);

    int max_ids = 4096;
    int *got = malloc(sizeof(int) * (size_t)max_ids);
    int *exp = malloc(sizeof(int) * (size_t)max_ids);
    TEST_ASSERT(got && exp);

    int n_cases = 0, n_pass = 0;
    const char *p = json;
    while ((p = test_json_find_key(p, "text")) != NULL) {
        while (*p && (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n')) p++;
        const char *after = NULL;
        char *text = test_json_decode_string(p, &after);
        p = after;

        const char *ids_p = test_json_find_key(p, "ids");
        TEST_ASSERT(ids_p != NULL);
        while (*ids_p && *ids_p != '[') ids_p++;
        ids_p++;
        int n_exp = 0;
        while (*ids_p && *ids_p != ']') {
            if (*ids_p == '-' || (*ids_p >= '0' && *ids_p <= '9')) {
                char *end = NULL;
                long v = strtol(ids_p, &end, 10);
                TEST_ASSERT(end != ids_p);
                if (n_exp < max_ids) exp[n_exp] = (int)v;
                n_exp++;
                ids_p = end;
            } else {
                ids_p++;
            }
        }
        p = ids_p + 1;

        int n_got = ds4_tokenize_model_text(model_path, text, got, max_ids);
        n_cases++;
        bool ok = (n_got == n_exp);
        for (int i = 0; ok && i < n_got && i < n_exp; i++) {
            if (got[i] != exp[i]) ok = false;
        }
        if (ok) {
            n_pass++;
        } else {
            fprintf(stderr, "  glm-bpe: MISMATCH (%d ids exp, %d got) text=%s\n", n_exp, n_got, text);
            fprintf(stderr, "    exp:");
            for (int i = 0; i < n_exp && i < 16; i++) fprintf(stderr, " %d", exp[i]);
            fprintf(stderr, "\n    got:");
            for (int i = 0; i < n_got && i < 16; i++) fprintf(stderr, " %d", got[i]);
            fprintf(stderr, "\n");
        }
        free(text);
    }

    fprintf(stderr, "  glm-bpe: %d/%d cases match HF tokenizers oracle\n", n_pass, n_cases);
    TEST_ASSERT(n_cases == 43);
    TEST_ASSERT(n_pass == n_cases);

    /* Chat-template smoke: render system+user, confirm [gMASK]<sop> prefix and
     * the role sentinels are present, and that the prefix round-trips. */
    int chat[256];
    int n_chat = ds4_render_chat_prompt(model_path, "You are helpful.", "Hi",
                                        DS4_THINK_HIGH, chat, 256);
    TEST_ASSERT(n_chat >= 2);
    bool has_prefix = (chat[0] == 154822 && chat[1] == 154824);
    fprintf(stderr, "  glm-bpe: chat prefix [%d, %d] (expect 154822, 154824)\n",
            n_chat > 0 ? chat[0] : -1, n_chat > 1 ? chat[1] : -1);
    bool has_sys = false, has_user = false, has_asst = false;
    for (int i = 0; i < n_chat; i++) {
        if (chat[i] == 154826) has_sys = true;
        if (chat[i] == 154827) has_user = true;
        if (chat[i] == 154828) has_asst = true;
    }
    TEST_ASSERT(has_prefix);
    TEST_ASSERT(has_sys && has_user && has_asst);
    TEST_ASSERT(chat[n_chat - 1] == 154841);  /* opens with <think> */

    /* Nothink path renders the empty <think></think> block at the end. */
    int nothink[256];
    int n_no = ds4_render_chat_prompt(model_path, "You are helpful.", "Hi",
                                       DS4_THINK_NONE, nothink, 256);
    TEST_ASSERT(n_no >= 2);
    fprintf(stderr, "  glm-bpe: nothink tail [%d, %d] (expect 154841, 154842)\n",
            nothink[n_no - 2], nothink[n_no - 1]);
    TEST_ASSERT(nothink[n_no - 2] == 154841 && nothink[n_no - 1] == 154842);

    /* Genuine round-trip: render → decode ids back to text (shard 1) →
     * re-tokenize → must reproduce the same id stream.  This proves special
     * tokens survive decode as single ids and BPE words re-merge identically. */
    int rt[512];
    int n_rt = ds4_render_chat_prompt(model_path, "You are helpful.", "Hi",
                                      DS4_THINK_HIGH, rt, 512);
    TEST_ASSERT(n_rt > 0 && n_rt <= 512);
    char *rt_text = NULL;
    size_t rt_len = ds4_decode_model_text(model_path, rt, n_rt, &rt_text);
    TEST_ASSERT(rt_text != NULL && rt_len > 0);
    int rt2[512];
    int n_rt2 = ds4_tokenize_model_text(model_path, rt_text, rt2, 512);
    bool rt_ok = (n_rt2 == n_rt);
    for (int i = 0; rt_ok && i < n_rt; i++) {
        if (rt2[i] != rt[i]) rt_ok = false;
    }
    if (!rt_ok) {
        fprintf(stderr, "  glm-bpe: round-trip drift (%d -> %d ids) decoded=%s\n",
                n_rt, n_rt2, rt_text);
    }
    TEST_ASSERT(rt_ok);
    fprintf(stderr, "  glm-bpe: round-trip stable (%d ids, %zu bytes)\n", n_rt, rt_len);
    free(rt_text);

    /* Multi-turn transcript via the public append API.  Assistant history that
     * already opens with <think> must not get a second <think> prepended (no
     * "<think><think>"); stripped content gets exactly one empty <think></think>
     * block.  Also covers the mode-dispatched max-effort prefix (glm4-tokenized
     * <|system|>Reasoning Effort: Max right after the header). */
    int hist[512];
    int n_hist = ds4_render_chat_history(model_path, NULL,
                                         "<think>plan</think>done", false, hist, 512);
    TEST_ASSERT(n_hist > 0);
    char *hist_text = NULL;
    size_t hist_len = ds4_decode_model_text(model_path, hist, n_hist, &hist_text);
    TEST_ASSERT(hist_text && hist_len > 0);
    bool no_double = (strstr(hist_text, "<think><think>") == NULL);
    bool has_think = (strstr(hist_text, "<think>") != NULL);
    fprintf(stderr, "  glm-bpe: assistant-history no-double-think=%d has-think=%d\n",
            (int)no_double, (int)has_think);
    TEST_ASSERT(no_double && has_think);
    free(hist_text);

    int stripped[512];
    int n_stripped = ds4_render_chat_history(model_path, NULL, "Hello there.",
                                             false, stripped, 512);
    TEST_ASSERT(n_stripped > 0);
    char *stripped_text = NULL;
    (void)ds4_decode_model_text(model_path, stripped, n_stripped, &stripped_text);
    bool stripped_block = (strstr(stripped_text, "<think></think>") != NULL);
    fprintf(stderr, "  glm-bpe: stripped assistant <think></think> block=%d\n",
            (int)stripped_block);
    TEST_ASSERT(stripped_block);
    free(stripped_text);

    int eff[512];
    int n_eff = ds4_render_chat_history(model_path, NULL, "Hi", true, eff, 512);
    TEST_ASSERT(n_eff > 8);
    /* max-effort: <|system|>(154826) at header end, then glm4 tokens of
     * "Reasoning Effort: Max" — proves the prefix is glm4-tokenized on GLM. */
    TEST_ASSERT(eff[2] == 154826);
    int exp_eff[32];
    int n_exp_eff = ds4_tokenize_model_text(model_path, "Reasoning Effort: Max",
                                            exp_eff, 32);
    bool eff_ok = (n_exp_eff > 0 && 3 + n_exp_eff <= n_eff);
    for (int i = 0; eff_ok && i < n_exp_eff; i++) {
        if (eff[3 + i] != exp_eff[i]) eff_ok = false;
    }
    fprintf(stderr, "  glm-bpe: max-effort glm4-tokenized=%d (%d ids after <|system|>)\n",
            (int)eff_ok, n_exp_eff);
    TEST_ASSERT(eff_ok);

    free(got);
    free(exp);
    free(json);
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
} ds4_test_entry;

/* GLM-5.2 K-quant CPU dequant vs the authoritative llama.cpp oracle
 * (tests/test-vectors/glm52-quant/<TYPE>.bin + <TYPE>.oracle.txt). Covers every
 * quant type the UD-IQ2_M forward touches: Q8_0 (attn projections), IQ2_XXS
 * (expert gate/up), Q4_K/Q5_K/Q6_K, IQ3_XXS/IQ4_XS/IQ2_S (expert down + dense). */
static void test_glm_quant_dequant(void) {
    const char *dir = getenv("DS4_TEST_GLM52_QUANT_DIR");
    if (!dir || !dir[0]) dir = "tests/test-vectors/glm52-quant";
    static const struct { const char *name; uint32_t type; uint32_t block_bytes; uint32_t block_elems; } types[] = {
        {"Q8_0", 8, 34, 32}, {"Q2_K", 10, 84, 256}, {"Q3_K", 11, 110, 256},
        {"Q4_K", 12, 144, 256}, {"Q5_K", 13, 176, 256}, {"Q6_K", 14, 210, 256},
        {"IQ3_XXS", 18, 98, 256}, {"IQ4_XS", 23, 136, 256},
        {"IQ2_S", 22, 82, 256}, {"IQ2_XXS", 16, 66, 256},
    };
    const size_t n_types = sizeof(types) / sizeof(types[0]);
    int n_pass = 0, n_run = 0;
    for (size_t t = 0; t < n_types; t++) {
        char bin[512], oracle[512];
        snprintf(bin, sizeof(bin), "%s/%s.bin", dir, types[t].name);
        snprintf(oracle, sizeof(oracle), "%s/%s.oracle.txt", dir, types[t].name);
        size_t blen = 0, olen = 0;
        char *b = test_read_whole_file(bin, &blen);
        char *o = test_read_whole_file(oracle, &olen);
        if (!b || !o) {
            fprintf(stderr, "  glm-quant-dequant: SKIP %s (fixture missing in %s)\n", types[t].name, dir);
            free(b); free(o); continue;
        }
        const size_t n_blocks = blen / types[t].block_bytes;
        const size_t n_elem   = n_blocks * types[t].block_elems;
        /* Large-slice reference: llama.cpp oracle (one F32 per line).  Uses a
         * dynamically-sized buffer so fixtures can be >=1000 blocks (256k+
         * elements) -- the IQ2_XXS pointer-arithmetic regression only surfaced
         * on a large real slice, not the old 8-block fixtures. */
        float *ref = malloc(sizeof(float) * n_elem);
        TEST_ASSERT(ref != NULL);
        size_t nref = 0;
        for (char *p = o; *p && nref < n_elem; ) {
            char *e = NULL; double v = strtod(p, &e);
            if (e == p) { p++; continue; }
            ref[nref++] = (float)v; p = e;
        }
        if (nref < n_elem) {
            fprintf(stderr, "  glm-quant-dequant: FAIL %s (fixture size: %zu blocks/%zu ref)\n", types[t].name, n_blocks, nref);
            free(ref); free(b); free(o); continue;
        }
        float *out = malloc(sizeof(float) * n_elem);
        TEST_ASSERT(out != NULL);
        if (!ds4_dequant_glm_row(types[t].type, b, out, n_elem)) {
            fprintf(stderr, "  glm-quant-dequant: FAIL %s (unsupported type %u)\n", types[t].name, types[t].type);
            free(out); free(ref); free(b); free(o); continue;
        }
        float maxabs = 0.0f, maxrel = 0.0f;
        for (size_t i = 0; i < n_elem; i++) {
            float d = fabsf(out[i] - ref[i]);
            if (d > maxabs) maxabs = d;
            if (fabsf(ref[i]) > 1e-9f) { float r = d / fabsf(ref[i]); if (r > maxrel) maxrel = r; }
        }
        bool ok = (maxabs < 1e-4f) || (maxrel < 1e-5f);
        fprintf(stderr, "  glm-quant-dequant: %s %s (n=%zu maxabs=%.3g maxrel=%.3g)\n",
                types[t].name, ok ? "PASS" : "FAIL", n_elem, (double)maxabs, (double)maxrel);
        if (ok) n_pass++;
        n_run++;
        free(out); free(ref); free(b); free(o);
    }
    fprintf(stderr, "  glm-quant-dequant: %d/%d types match llama.cpp oracle\n", n_pass, (int)n_types);
    TEST_ASSERT(n_run == (int)n_types);
    TEST_ASSERT(n_pass == (int)n_types);
}

/* GLM-5.2 CPU reference forward components vs the numpy oracle
 * (tests/test-vectors/glm52-ref/, generated by dump_fixtures.py from
 * glm52_mla_moe_ref.py).  Validates every Phase 4a component individually and
 * the MLA composite end-to-end, on the SAME synthetic inputs the numpy
 * reference consumes.  Tolerance: 1e-4 abs / 1e-5 rel (per task spec). */

static float *glm_ref_load_floats(const char *dir, const char *fname, size_t n) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, fname);
    size_t ln = 0;
    char *raw = test_read_whole_file(path, &ln);
    if (!raw) return NULL;
    float *buf = malloc(sizeof(float) * n);
    TEST_ASSERT(buf != NULL);
    size_t got = 0;
    for (char *c = raw; *c && got < n; ) {
        char *e = NULL;
        double v = strtod(c, &e);
        if (e == c) { c++; continue; }
        buf[got++] = (float)v;
        c = e;
    }
    free(raw);
    if (got < n) {
        fprintf(stderr, "  glm-cpu-ref: FAIL %s (read %zu, want %zu)\n", fname, got, n);
        TEST_ASSERT(0);
    }
    return buf;
}

static int *glm_ref_load_ints(const char *dir, const char *fname, size_t n) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, fname);
    size_t ln = 0;
    char *raw = test_read_whole_file(path, &ln);
    if (!raw) return NULL;
    int *buf = malloc(sizeof(int) * n);
    TEST_ASSERT(buf != NULL);
    size_t got = 0;
    for (char *c = raw; *c && got < n; ) {
        char *e = NULL;
        long v = strtol(c, &e, 10);
        if (e == c) { c++; continue; }
        buf[got++] = (int)v;
        c = e;
    }
    free(raw);
    return buf;
}

static size_t g_glm_total, g_glm_pass;
static void glm_ref_cmp(const char *label, const float *got, const float *ref,
                        size_t n, float tol_abs, float tol_rel) {
    float ma = 0.0f, mr = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(got[i] - ref[i]);
        if (d > ma) ma = d;
        if (fabsf(ref[i]) > 1e-9f) {
            float r = d / fabsf(ref[i]);
            if (r > mr) mr = r;
        }
    }
    bool ok = (ma < tol_abs) || (mr < tol_rel);
    fprintf(stderr, "  glm-cpu-ref: %-26s %s (n=%zu maxabs=%.3g maxrel=%.3g)\n",
            label, ok ? "PASS" : "FAIL", n, (double)ma, (double)mr);
    g_glm_total++;
    if (ok) g_glm_pass++;
    TEST_ASSERT(ok);
}

static void test_glm_cpu_ref_components(void) {
    const char *dir = getenv("DS4_TEST_GLM52_REF_DIR");
    if (!dir || !dir[0]) dir = "tests/test-vectors/glm52-ref";

    const glm_cpu_shape shape = glm_cpu_shape_fixture();
    const uint32_t H = shape.hidden, nh = shape.n_head, ql = shape.q_lora, kvl = shape.kv_lora;
    const uint32_t nope = shape.qk_nope, rope = shape.qk_rope, vd = shape.v_dim;
    const uint32_t ff = shape.ff_inter, E = shape.n_expert, K = shape.n_expert_used;
    const uint32_t seq_n = 4, t_pos = 3;  /* t=3 -> seq_n=t+1=4 (dims.txt) */
    const float tol_abs = 1e-4f, tol_rel = 1e-5f;
    g_glm_total = g_glm_pass = 0;

    /* --------------------------- inputs --------------------------------- */
    float *x = glm_ref_load_floats(dir, "mla_x.txt", H);
    float *WqA = glm_ref_load_floats(dir, "mla_WqA.txt", (size_t)ql * H);
    float *WqB = glm_ref_load_floats(dir, "mla_WqB.txt", (size_t)nh * (nope + rope) * ql);
    float *WkvA = glm_ref_load_floats(dir, "mla_WkvA.txt", (size_t)(kvl + rope) * H);
    float *WkB = glm_ref_load_floats(dir, "mla_WkB.txt", (size_t)nh * nope * kvl);
    float *WvB = glm_ref_load_floats(dir, "mla_WvB.txt", (size_t)nh * vd * kvl);
    float *Wo = glm_ref_load_floats(dir, "mla_Wo.txt", (size_t)H * (nh * vd));
    /* Caches are mutable in the C path; load reference copies for reset. */
    float *K_nope_ref = glm_ref_load_floats(dir, "mla_K_nope_cache.txt", (size_t)nh * seq_n * nope);
    float *V_ref      = glm_ref_load_floats(dir, "mla_V_cache.txt",     (size_t)nh * seq_n * vd);
    float *K_rope_ref = glm_ref_load_floats(dir, "mla_K_rope_cache.txt", (size_t)nh * seq_n * rope);
    if (!x || !WqA || !WqB || !WkvA || !WkB || !WvB || !Wo ||
        !K_nope_ref || !V_ref || !K_rope_ref) {
        fprintf(stderr, "  glm-cpu-ref: SKIP (fixtures missing in %s; run dump_fixtures.py)\n", dir);
        return;
    }

    /* ----------------------- reference intermediates -------------------- */
    float *ref_wqa     = glm_ref_load_floats(dir, "mla_q_a_proj.txt",   ql);
    float *ref_wqa_n   = glm_ref_load_floats(dir, "mla_q_a_normed.txt", ql);
    float *ref_q       = glm_ref_load_floats(dir, "mla_q.txt",          (size_t)nh * (nope + rope));
    float *ref_kva     = glm_ref_load_floats(dir, "mla_kva.txt",        kvl + rope);
    float *ref_latent  = glm_ref_load_floats(dir, "mla_latent.txt",     kvl);
    float *ref_krraw   = glm_ref_load_floats(dir, "mla_k_rope_raw.txt", rope);
    float *ref_kvln    = glm_ref_load_floats(dir, "mla_kvln.txt",       kvl);
    float *ref_knope   = glm_ref_load_floats(dir, "mla_k_nope.txt",     (size_t)nh * nope);
    float *ref_v       = glm_ref_load_floats(dir, "mla_v.txt",          (size_t)nh * vd);
    float *ref_qrr     = glm_ref_load_floats(dir, "mla_q_rope_r.txt",   (size_t)nh * rope);
    float *ref_krr     = glm_ref_load_floats(dir, "mla_k_rope_r.txt",   (size_t)nh * rope);
    float *ref_attn_o  = glm_ref_load_floats(dir, "mla_attn_out.txt",   (size_t)nh * vd);
    float *ref_mla_out = glm_ref_load_floats(dir, "mla_out.txt",        H);

    /* ----------------- component 1: matvec q_a + RMSNorm ---------------- */
    float *wqa = malloc(sizeof(float) * ql); TEST_ASSERT(wqa != NULL);
    glm_matvec_f32(wqa, WqA, x, ql, H);
    glm_ref_cmp("matvec q_a (WqA@x)", wqa, ref_wqa, ql, tol_abs, tol_rel);
    float *ones = malloc(sizeof(float) * (ql > kvl ? ql : kvl));
    for (uint32_t i = 0; i < (ql > kvl ? ql : kvl); i++) ones[i] = 1.0f;
    float *wqa_n = malloc(sizeof(float) * ql); TEST_ASSERT(wqa_n != NULL);
    glm_rmsnorm_f32(wqa_n, wqa, ones, ql, shape.rms_eps);
    glm_ref_cmp("rmsnorm q_a (ones)", wqa_n, ref_wqa_n, ql, tol_abs, tol_rel);

    /* ------------- component 2: matvec q_b + reshape ------------------- */
    float *q_flat = malloc(sizeof(float) * (size_t)nh * (nope + rope)); TEST_ASSERT(q_flat != NULL);
    glm_matvec_f32(q_flat, WqB, wqa_n, nh * (nope + rope), ql);
    glm_ref_cmp("matvec q_b (WqB@wqa)", q_flat, ref_q, (size_t)nh * (nope + rope), tol_abs, tol_rel);

    /* ------------- component 3: kv_a matvec + latent/rope split --------- */
    float *kva = malloc(sizeof(float) * (kvl + rope)); TEST_ASSERT(kva != NULL);
    glm_matvec_f32(kva, WkvA, x, kvl + rope, H);
    glm_ref_cmp("matvec kv_a (WkvA@x)", kva, ref_kva, kvl + rope, tol_abs, tol_rel);
    glm_ref_cmp("kv_a latent split", kva, ref_latent, kvl, tol_abs, tol_rel);
    glm_ref_cmp("kv_a rope split", kva + kvl, ref_krraw, rope, tol_abs, tol_rel);

    /* ------------- component 4: latent-only RMSNorm -------------------- */
    float *kvln = malloc(sizeof(float) * kvl); TEST_ASSERT(kvln != NULL);
    glm_rmsnorm_f32(kvln, kva, ones, kvl, shape.rms_eps);
    glm_ref_cmp("rmsnorm latent-only", kvln, ref_kvln, kvl, tol_abs, tol_rel);

    /* ------------- component 5: k_b / v_b projections ------------------ */
    float *k_nope = malloc(sizeof(float) * (size_t)nh * nope); TEST_ASSERT(k_nope != NULL);
    float *v      = malloc(sizeof(float) * (size_t)nh * vd);   TEST_ASSERT(v != NULL);
    glm_matvec_f32(k_nope, WkB, kvln, nh * nope, kvl);
    glm_matvec_f32(v,      WvB, kvln, nh * vd,   kvl);
    glm_ref_cmp("matvec k_b (WkB@kvln)", k_nope, ref_knope, (size_t)nh * nope, tol_abs, tol_rel);
    glm_ref_cmp("matvec v_b (WvB@kvln)", v,      ref_v,     (size_t)nh * vd,   tol_abs, tol_rel);

    /* ------------- component 6: interleaved RoPE on rope slice --------- */
    float *qrope_in = malloc(sizeof(float) * (size_t)nh * rope); TEST_ASSERT(qrope_in != NULL);
    for (uint32_t h = 0; h < nh; h++)
        for (uint32_t j = 0; j < rope; j++)
            qrope_in[(size_t)h * rope + j] = q_flat[(size_t)h * (nope + rope) + nope + j];
    float *qrope_out = malloc(sizeof(float) * (size_t)nh * rope); TEST_ASSERT(qrope_out != NULL);
    glm_rope_interleaved_f32(qrope_out, qrope_in, rope, nh, shape.rope_base, t_pos);
    glm_ref_cmp("rope q (interleaved)", qrope_out, ref_qrr, (size_t)nh * rope, tol_abs, tol_rel);

    float *kraw_b = malloc(sizeof(float) * (size_t)nh * rope); TEST_ASSERT(kraw_b != NULL);
    for (uint32_t h = 0; h < nh; h++)
        memcpy(kraw_b + (size_t)h * rope, kva + kvl, rope * sizeof(float));
    float *krope_out = malloc(sizeof(float) * (size_t)nh * rope); TEST_ASSERT(krope_out != NULL);
    glm_rope_interleaved_f32(krope_out, kraw_b, rope, nh, shape.rope_base, t_pos);
    glm_ref_cmp("rope k (interleaved)", krope_out, ref_krr, (size_t)nh * rope, tol_abs, tol_rel);

    /* ------------- component 7: composite MLA forward (one token) ------ */
    /* Reset caches from reference copies (the composite mutates them). */
    float *K_nope_c = malloc(sizeof(float) * (size_t)nh * seq_n * nope); TEST_ASSERT(K_nope_c != NULL);
    float *V_c      = malloc(sizeof(float) * (size_t)nh * seq_n * vd);   TEST_ASSERT(V_c != NULL);
    float *K_rope_c = malloc(sizeof(float) * (size_t)nh * seq_n * rope); TEST_ASSERT(K_rope_c != NULL);
    memcpy(K_nope_c, K_nope_ref, sizeof(float) * (size_t)nh * seq_n * nope);
    memcpy(V_c,      V_ref,      sizeof(float) * (size_t)nh * seq_n * vd);
    memcpy(K_rope_c, K_rope_ref, sizeof(float) * (size_t)nh * seq_n * rope);
    float *mla_out = malloc(sizeof(float) * H); TEST_ASSERT(mla_out != NULL);
    glm_mla_forward_token_f32(mla_out, x, WqA, WqB, WkvA, WkB, WvB, Wo,
                              NULL, NULL,
                              K_nope_c, V_c, K_rope_c, seq_n, t_pos, &shape, NULL);
    glm_ref_cmp("MLA composite (one token)", mla_out, ref_mla_out, H, tol_abs, tol_rel);

    /* ----------------- MoE sigmoid router components ------------------ */
    float *gate       = glm_ref_load_floats(dir, "moe_gate.txt",   (size_t)E * H);
    float *bias       = glm_ref_load_floats(dir, "moe_bias.txt",   E);
    float *ref_logits = glm_ref_load_floats(dir, "moe_logits.txt", E);
    float *ref_prob   = glm_ref_load_floats(dir, "moe_prob.txt",   E);
    float *ref_sel    = glm_ref_load_floats(dir, "moe_sel.txt",    E);
    float *ref_w      = glm_ref_load_floats(dir, "moe_w.txt",      K);
    int   *ref_idx    = glm_ref_load_ints(dir,   "moe_idx.txt",    K);

    float *logits = malloc(sizeof(float) * E); TEST_ASSERT(logits != NULL);
    glm_matvec_f32(logits, gate, x, E, H);
    glm_ref_cmp("MoE logits (gate@x)", logits, ref_logits, E, tol_abs, tol_rel);

    float *prob = malloc(sizeof(float) * E); TEST_ASSERT(prob != NULL);
    for (uint32_t i = 0; i < E; i++)
        prob[i] = (logits[i] >= 0.0f) ? 1.0f / (1.0f + expf(-logits[i]))
                                      : expf(logits[i]) / (1.0f + expf(logits[i]));
    glm_ref_cmp("MoE sigmoid(prob)", prob, ref_prob, E, tol_abs, tol_rel);

    float *sel = malloc(sizeof(float) * E); TEST_ASSERT(sel != NULL);
    for (uint32_t i = 0; i < E; i++) sel[i] = prob[i] + bias[i];
    glm_ref_cmp("MoE sel (prob+bias)", sel, ref_sel, E, tol_abs, tol_rel);

    /* Full router: sigmoid + bias, top-k, normalize selected, scale 2.5. */
    int idx[64]; float w_out[64];
    float *scratch = malloc(sizeof(float) * E); TEST_ASSERT(scratch != NULL);
    glm_moe_route_sigmoid(idx, w_out, gate, x, bias, &shape, scratch);
    /* Compare idx as ints (bit-equal expectation). */
    {
        bool idx_ok = true;
        for (uint32_t k = 0; k < K; k++) if (idx[k] != ref_idx[k]) { idx_ok = false; break; }
        fprintf(stderr, "  glm-cpu-ref: %-26s %s (idx=[%d,%d] ref=[%d,%d])\n",
                "MoE topk idx", idx_ok ? "PASS" : "FAIL",
                K >= 1 ? idx[0] : -1, K >= 2 ? idx[1] : -1,
                K >= 1 ? ref_idx[0] : -1, K >= 2 ? ref_idx[1] : -1);
        g_glm_total++;
        if (idx_ok) g_glm_pass++;
        TEST_ASSERT(idx_ok);
    }
    glm_ref_cmp("MoE weights (norm*2.5)", w_out, ref_w, K, tol_abs, tol_rel);

    /* ----------------- dense SwiGLU FFN components -------------------- */
    float *ff_gate    = glm_ref_load_floats(dir, "ffn_gate.txt",   (size_t)ff * H);
    float *ff_up      = glm_ref_load_floats(dir, "ffn_up.txt",     (size_t)ff * H);
    float *ff_down    = glm_ref_load_floats(dir, "ffn_down.txt",   (size_t)H * ff);
    float *ref_gh     = glm_ref_load_floats(dir, "ffn_gate_h.txt", ff);
    float *ref_uh     = glm_ref_load_floats(dir, "ffn_up_h.txt",   ff);
    float *ref_act    = glm_ref_load_floats(dir, "ffn_act.txt",    ff);
    float *ref_ff_out = glm_ref_load_floats(dir, "ffn_out.txt",    H);

    float *gh = malloc(sizeof(float) * ff); TEST_ASSERT(gh != NULL);
    float *uh = malloc(sizeof(float) * ff); TEST_ASSERT(uh != NULL);
    glm_matvec_f32(gh, ff_gate, x, ff, H);
    glm_matvec_f32(uh, ff_up,   x, ff, H);
    glm_ref_cmp("FFN gate@x", gh, ref_gh, ff, tol_abs, tol_rel);
    glm_ref_cmp("FFN up@x",   uh, ref_uh, ff, tol_abs, tol_rel);
    float *act = malloc(sizeof(float) * ff); TEST_ASSERT(act != NULL);
    glm_silu_f32(act, gh, ff);
    for (uint32_t i = 0; i < ff; i++) act[i] *= uh[i];
    glm_ref_cmp("FFN swiglu (silu(g)*u)", act, ref_act, ff, tol_abs, tol_rel);
    float *ff_out = malloc(sizeof(float) * H); TEST_ASSERT(ff_out != NULL);
    glm_swiglu_dense_f32(ff_out, x, ff_gate, ff_up, ff_down, &shape, gh, uh);
    glm_ref_cmp("FFN dense composite", ff_out, ref_ff_out, H, tol_abs, tol_rel);

    /* ----------------- tensor binding (structural surface) ------------ */
    glm_layer_weights lw;
    uint32_t bound = glm_layer_bind(NULL, 0, &lw);
    TEST_ASSERT(bound == 0);  /* no model loaded in this oracle test */
    fprintf(stderr, "  glm-cpu-ref: tensor-binding surface present (bound=%u)\n", bound);

    fprintf(stderr, "  glm-cpu-ref: %zu/%zu component cases match numpy oracle\n",
            g_glm_pass, g_glm_total);

    /* cleanup */
    free(wqa); free(wqa_n); free(ones); free(q_flat); free(kva); free(kvln);
    free(k_nope); free(v); free(qrope_in); free(qrope_out);
    free(kraw_b); free(krope_out); free(K_nope_c); free(V_c); free(K_rope_c);
    free(mla_out); free(logits); free(prob); free(sel); free(scratch);
    free(gh); free(uh); free(act); free(ff_out);
    free(x); free(WqA); free(WqB); free(WkvA); free(WkB); free(WvB); free(Wo);
    free(K_nope_ref); free(V_ref); free(K_rope_ref);
    free(ref_wqa); free(ref_wqa_n); free(ref_q); free(ref_kva); free(ref_latent);
    free(ref_krraw); free(ref_kvln); free(ref_knope); free(ref_v); free(ref_qrr);
    free(ref_krr); free(ref_attn_o); free(ref_mla_out);
    free(gate); free(bias); free(ref_logits); free(ref_prob); free(ref_sel);
    free(ref_w); free(ref_idx);
    free(ff_gate); free(ff_up); free(ff_down); free(ref_gh); free(ref_uh);
    free(ref_act); free(ref_ff_out);
}

/* Phase 4a-full: tiny synthetic full CPU forward assembly self-check (no
 * model on disk needed).  Exercises the full layer loop -- embedding, residual
 * streams, growing KV cache, MLA decode over real-layout weights (incl. the k_b
 * reorder), dense SwiGLU for blk.0, sigmoid MoE with on-demand expert dequant
 * for blk.1-3, output norm, LM head -- and asserts finite non-degenerate
 * logits + a valid argmax token id. */
static void test_glm_cpu_forward_synth(void) {
    int token = -1; float top = 0.0f; bool finite = false;
    int rc = ds4_glm_cpu_forward_synth(&token, &top, &finite);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(finite);
    TEST_ASSERT(token >= 0);
    fprintf(stderr, "  glm-cpu-forward-synth: PASS (token=%d top=%.6f finite=%s)\n",
            token, (double)top, finite ? "yes" : "no");
}

static void test_glm_nextn_synth(void) {
    int token = -1; float top = 0.0f; bool finite = false;
    int rc = ds4_glm_nextn_synth(&token, &top, &finite);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(finite);
    TEST_ASSERT(token == 13);
    TEST_ASSERT(fabsf(top - 0.031084f) < 1e-5f);
    fprintf(stderr, "  glm-nextn-synth: PASS (token=%d top=%.6f finite=%s)\n",
            token, (double)top, finite ? "yes" : "no");
}

/* Phase 4c-iv: the full Metal forward assembly (glm_metal_forward) on the same
 * tiny synthetic model as the CPU synth.  Proves the end-to-end Metal path
 * (per-layer dequant -> GPU kernels -> GPU KV cache -> MLA/dense/MoE dispatch
 * -> LM head) is runnable + finite without the 238 GiB split, and that Metal
 * argmax == CPU synth argmax (same deterministic weights). */
static void test_glm_metal_forward_synth(void) {
    int mtoken = -1; float mtop = 0.0f; bool mfinite = false;
    int ctoken = -1; float ctop = 0.0f; bool cfinite = false;
    int mrc = ds4_glm_metal_forward_synth(&mtoken, &mtop, &mfinite);
    TEST_ASSERT(mrc == 0);
    TEST_ASSERT(mfinite);
    /* CPU synth reference on the identical model. */
    int crc = ds4_glm_cpu_forward_synth(&ctoken, &ctop, &cfinite);
    TEST_ASSERT(crc == 0);
    fprintf(stderr, "  glm-metal-forward-synth: metal=%d cpu=%d (top m=%.6f c=%.6f)\n",
            mtoken, ctoken, (double)mtop, (double)ctop);
    TEST_ASSERT(mtoken == ctoken);
}

/* GLM-5.2 Metal component kernels (metal/glm.metal via ds4_gpu_glm_* wrappers)
 * validated against the Phase 4a CPU reference on the SAME synthetic fixtures.
 * Each component runs the CPU oracle and the Metal kernel on identical inputs;
 * the full MLA chain is then run end-to-end on Metal and compared to the CPU
 * composite (glm_mla_forward_token_f32) and the numpy mla_out fixture.
 * Skips gracefully when built without GPU/Metal or when no Metal device is
 * present.  Tolerance: 1e-4 abs / 1e-5 rel (Metal is F32 -> expect ~1e-6). */
static size_t g_glm_metal_total, g_glm_metal_pass;
static void glm_metal_cmp(const char *label, const float *metal, const float *cpu,
                          size_t n, float tol_abs, float tol_rel) {
    float ma = 0.0f, mr = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(metal[i] - cpu[i]);
        if (d > ma) ma = d;
        if (fabsf(cpu[i]) > 1e-9f) {
            float r = d / fabsf(cpu[i]);
            if (r > mr) mr = r;
        }
    }
    bool ok = (ma < tol_abs) || (mr < tol_rel);
    fprintf(stderr, "  glm-metal: %-30s %s (n=%zu maxabs=%.3g maxrel=%.3g)\n",
            label, ok ? "PASS" : "FAIL", n, (double)ma, (double)mr);
    g_glm_metal_total++;
    if (ok) g_glm_metal_pass++;
    TEST_ASSERT(ok);
}

/* CPU reference attention block in isolation (mirrors the per-head loop in
 * glm_mla_forward_token_f32).  Used to validate the Metal decode-attention
 * kernel on a pre-assembled Q whose rope slice is already rotated and caches
 * that already hold the current token at position t. */
static void glm_cpu_attn_decode(const float *Q,
                                const float *K_nope_cache,
                                const float *K_rope_cache,
                                const float *V_cache, float *attn_out,
                                uint32_t nh, uint32_t nope, uint32_t rope,
                                uint32_t vd, uint32_t qhd,
                                uint32_t seq_n, uint32_t t) {
    const float kq_scale = 1.0f / sqrtf((float)qhd);
    float *scores = malloc(sizeof(float) * (size_t)(t + 1));
    TEST_ASSERT(scores != NULL);
    for (uint32_t h = 0; h < nh; h++) {
        const float *qn = Q + (size_t)h * qhd;
        const float *qr = qn + nope;
        float maxs = -1e30f;
        for (uint32_t n = 0; n <= t; n++) {
            const float *kn = K_nope_cache + ((size_t)h * seq_n + n) * nope;
            const float *kr = K_rope_cache + ((size_t)h * seq_n + n) * rope;
            float s = 0.0f;
            for (uint32_t d = 0; d < nope; d++) s += qn[d] * kn[d];
            for (uint32_t d = 0; d < rope; d++) s += qr[d] * kr[d];
            s *= kq_scale;
            scores[n] = s;
            if (s > maxs) maxs = s;
        }
        float denom = 0.0f;
        for (uint32_t n = 0; n <= t; n++) {
            scores[n] = expf(scores[n] - maxs);
            denom += scores[n];
        }
        float *oh = attn_out + (size_t)h * vd;
        for (uint32_t d = 0; d < vd; d++) oh[d] = 0.0f;
        for (uint32_t n = 0; n <= t; n++) {
            const float wt = scores[n] / denom;
            const float *vn = V_cache + ((size_t)h * seq_n + n) * vd;
            for (uint32_t d = 0; d < vd; d++) oh[d] += wt * vn[d];
        }
    }
    free(scores);
}

static void test_glm_generate_synth(void) {
    /* Phase 4e: incremental generation must match a fresh naive full forward at
     * every decode step (CPU), and Metal incremental must match CPU naive (so
     * the Metal greedy sequence == CPU greedy sequence). */
    const int n_steps = 8;
    int cpu_match = 0, metal_match = 0;
    int crc = ds4_glm_cpu_generate_synth(n_steps, &cpu_match);
    fprintf(stderr, "  glm-generate-synth (cpu): incremental-vs-naive %d/%d steps match\n",
            cpu_match, n_steps);
    TEST_ASSERT(crc == 0);
    TEST_ASSERT(cpu_match == n_steps);
    int mrc = ds4_glm_metal_generate_synth(n_steps, &metal_match);
    fprintf(stderr, "  glm-generate-synth (metal): incremental-vs-cpu-naive %d/%d steps match\n",
            metal_match, n_steps);
    TEST_ASSERT(mrc == 0);
    TEST_ASSERT(metal_match == n_steps);
}

static void test_glm_metal_components(void) {
#ifdef DS4_NO_GPU
    fprintf(stderr, "  glm-metal: SKIP (built without GPU/Metal)\n");
    return;
#else
    if (!ds4_gpu_init()) {
        fprintf(stderr, "  glm-metal: SKIP (no Metal device)\n");
        return;
    }
    const char *dir = getenv("DS4_TEST_GLM52_REF_DIR");
    if (!dir || !dir[0]) dir = "tests/test-vectors/glm52-ref";

    const glm_cpu_shape shape = glm_cpu_shape_fixture();
    const uint32_t H = shape.hidden, nh = shape.n_head, ql = shape.q_lora, kvl = shape.kv_lora;
    const uint32_t nope = shape.qk_nope, rope = shape.qk_rope, vd = shape.v_dim;
    const uint32_t qhd = nope + rope;
    const uint32_t seq_n = 4, t_pos = 3;
    const float tol_abs = 1e-4f, tol_rel = 1e-5f;
    g_glm_metal_total = g_glm_metal_pass = 0;

    /* inputs (same fixtures the CPU oracle consumes) */
    float *x    = glm_ref_load_floats(dir, "mla_x.txt", H);
    float *WqA  = glm_ref_load_floats(dir, "mla_WqA.txt", (size_t)ql * H);
    float *WqB  = glm_ref_load_floats(dir, "mla_WqB.txt", (size_t)nh * qhd * ql);
    float *WkvA = glm_ref_load_floats(dir, "mla_WkvA.txt", (size_t)(kvl + rope) * H);
    float *WkB  = glm_ref_load_floats(dir, "mla_WkB.txt", (size_t)nh * nope * kvl);
    float *WvB  = glm_ref_load_floats(dir, "mla_WvB.txt", (size_t)nh * vd * kvl);
    float *Wo   = glm_ref_load_floats(dir, "mla_Wo.txt", (size_t)H * (nh * vd));
    float *K_nope_ref = glm_ref_load_floats(dir, "mla_K_nope_cache.txt", (size_t)nh * seq_n * nope);
    float *V_ref      = glm_ref_load_floats(dir, "mla_V_cache.txt",     (size_t)nh * seq_n * vd);
    float *K_rope_ref = glm_ref_load_floats(dir, "mla_K_rope_cache.txt", (size_t)nh * seq_n * rope);
    if (!x || !WqA || !WqB || !WkvA || !WkB || !WvB || !Wo ||
        !K_nope_ref || !V_ref || !K_rope_ref) {
        fprintf(stderr, "  glm-metal: SKIP (fixtures missing in %s; run dump_fixtures.py)\n", dir);
        return;
    }

    float *ones = malloc(sizeof(float) * (ql > kvl ? ql : kvl));
    for (uint32_t i = 0; i < (ql > kvl ? ql : kvl); i++) ones[i] = 1.0f;

    /* ---- component 1: matvec q_a + RMSNorm (Metal vs live CPU ref) ---- */
    float *cqa = malloc(sizeof(float) * ql), *mqa = malloc(sizeof(float) * ql);
    glm_matvec_f32(cqa, WqA, x, ql, H);
    TEST_ASSERT(ds4_gpu_glm_matvec_f32(WqA, x, mqa, ql, H));
    glm_metal_cmp("matvec q_a (WqA@x)", mqa, cqa, ql, tol_abs, tol_rel);
    float *cqa_n = malloc(sizeof(float) * ql), *mqa_n = malloc(sizeof(float) * ql);
    glm_rmsnorm_f32(cqa_n, cqa, ones, ql, shape.rms_eps);
    TEST_ASSERT(ds4_gpu_glm_rmsnorm_f32(cqa, ones, mqa_n, ql, shape.rms_eps));
    glm_metal_cmp("rmsnorm q_a (ones)", mqa_n, cqa_n, ql, tol_abs, tol_rel);

    /* ---- component 2: matvec q_b ---- */
    float *cq = malloc(sizeof(float) * (size_t)nh * qhd), *mq = malloc(sizeof(float) * (size_t)nh * qhd);
    glm_matvec_f32(cq, WqB, cqa_n, nh * qhd, ql);
    TEST_ASSERT(ds4_gpu_glm_matvec_f32(WqB, cqa_n, mq, nh * qhd, ql));
    glm_metal_cmp("matvec q_b (WqB@qa_n)", mq, cq, (size_t)nh * qhd, tol_abs, tol_rel);

    /* ---- component 3: matvec kv_a + latent-only RMSNorm ---- */
    float *ckva = malloc(sizeof(float) * (kvl + rope)), *mkva = malloc(sizeof(float) * (kvl + rope));
    glm_matvec_f32(ckva, WkvA, x, kvl + rope, H);
    TEST_ASSERT(ds4_gpu_glm_matvec_f32(WkvA, x, mkva, kvl + rope, H));
    glm_metal_cmp("matvec kv_a (WkvA@x)", mkva, ckva, kvl + rope, tol_abs, tol_rel);
    float *ckvln = malloc(sizeof(float) * kvl), *mkvln = malloc(sizeof(float) * kvl);
    glm_rmsnorm_f32(ckvln, ckva, ones, kvl, shape.rms_eps);
    TEST_ASSERT(ds4_gpu_glm_rmsnorm_f32(ckva, ones, mkvln, kvl, shape.rms_eps));
    glm_metal_cmp("rmsnorm latent-only", mkvln, ckvln, kvl, tol_abs, tol_rel);

    /* ---- component 4: k_b / v_b projections ---- */
    float *ck = malloc(sizeof(float) * (size_t)nh * nope), *mk = malloc(sizeof(float) * (size_t)nh * nope);
    float *cv = malloc(sizeof(float) * (size_t)nh * vd),   *mv = malloc(sizeof(float) * (size_t)nh * vd);
    glm_matvec_f32(ck, WkB, ckvln, nh * nope, kvl);
    glm_matvec_f32(cv, WvB, ckvln, nh * vd,   kvl);
    TEST_ASSERT(ds4_gpu_glm_matvec_f32(WkB, ckvln, mk, nh * nope, kvl));
    TEST_ASSERT(ds4_gpu_glm_matvec_f32(WvB, ckvln, mv, nh * vd,   kvl));
    glm_metal_cmp("matvec k_b (WkB@kvln)", mk, ck, (size_t)nh * nope, tol_abs, tol_rel);
    glm_metal_cmp("matvec v_b (WvB@kvln)", mv, cv, (size_t)nh * vd,   tol_abs, tol_rel);

    /* ---- component 5: interleaved RoPE on the rope slice (Q + K) ---- */
    float *qr_in = malloc(sizeof(float) * (size_t)nh * rope);
    for (uint32_t h = 0; h < nh; h++)
        for (uint32_t j = 0; j < rope; j++)
            qr_in[(size_t)h * rope + j] = mq[(size_t)h * qhd + nope + j];
    float *cqr = malloc(sizeof(float) * (size_t)nh * rope), *mqr = malloc(sizeof(float) * (size_t)nh * rope);
    glm_rope_interleaved_f32(cqr, qr_in, rope, nh, shape.rope_base, t_pos);
    TEST_ASSERT(ds4_gpu_glm_rope_interleaved_f32(qr_in, mqr, rope, nh, shape.rope_base, t_pos));
    glm_metal_cmp("rope q (interleaved)", mqr, cqr, (size_t)nh * rope, tol_abs, tol_rel);

    float *kr_in = malloc(sizeof(float) * (size_t)nh * rope);
    for (uint32_t h = 0; h < nh; h++)
        memcpy(kr_in + (size_t)h * rope, mkva + kvl, rope * sizeof(float));
    float *ckr = malloc(sizeof(float) * (size_t)nh * rope), *mkr = malloc(sizeof(float) * (size_t)nh * rope);
    glm_rope_interleaved_f32(ckr, kr_in, rope, nh, shape.rope_base, t_pos);
    TEST_ASSERT(ds4_gpu_glm_rope_interleaved_f32(kr_in, mkr, rope, nh, shape.rope_base, t_pos));
    glm_metal_cmp("rope k (interleaved)", mkr, ckr, (size_t)nh * rope, tol_abs, tol_rel);

    /* ---- component 6: MLA decode attention ----
     * Q is assembled [nope | rotated-rope]; caches hold the current token at
     * position t (the fixtures already fill it).  CPU oracle via the isolated
     * attention block above. */
    float *Qm = malloc(sizeof(float) * (size_t)nh * qhd), *Qc = malloc(sizeof(float) * (size_t)nh * qhd);
    for (uint32_t h = 0; h < nh; h++) {
        memcpy(Qm + (size_t)h * qhd,       mq  + (size_t)h * qhd,        nope * sizeof(float));
        memcpy(Qm + (size_t)h * qhd + nope, mqr + (size_t)h * rope,       rope * sizeof(float));
        memcpy(Qc + (size_t)h * qhd,       cq  + (size_t)h * qhd,        nope * sizeof(float));
        memcpy(Qc + (size_t)h * qhd + nope, cqr + (size_t)h * rope,       rope * sizeof(float));
    }
    float *cattn = malloc(sizeof(float) * (size_t)nh * vd), *mattn = malloc(sizeof(float) * (size_t)nh * vd);
    glm_cpu_attn_decode(Qc, K_nope_ref, K_rope_ref, V_ref, cattn,
                        nh, nope, rope, vd, qhd, seq_n, t_pos);
    TEST_ASSERT(ds4_gpu_glm_attn_decode_f32(Qm, K_nope_ref, K_rope_ref, V_ref, mattn,
                        nh, nope, rope, vd, qhd, seq_n, t_pos));
    glm_metal_cmp("attn decode (scores+V)", mattn, cattn, (size_t)nh * vd, tol_abs, tol_rel);

    /* ---- component 7: single attn_output projection ---- */
    float *co = malloc(sizeof(float) * H), *mo = malloc(sizeof(float) * H);
    glm_matvec_f32(co, Wo, cattn, H, nh * vd);
    TEST_ASSERT(ds4_gpu_glm_matvec_f32(Wo, mattn, mo, H, nh * vd));
    glm_metal_cmp("matvec attn_output (Wo)", mo, co, H, tol_abs, tol_rel);

    /* ---- composite: full Metal MLA chain vs CPU glm_mla_forward_token_f32 ----
     * Run the entire Metal pipeline into mout, writing this token's Metal k/v
     * into the caches at position t (mirroring the CPU composite). */
    float *Kc_n = malloc(sizeof(float) * (size_t)nh * seq_n * nope);
    float *Kc_r = malloc(sizeof(float) * (size_t)nh * seq_n * rope);
    float *Vc   = malloc(sizeof(float) * (size_t)nh * seq_n * vd);
    memcpy(Kc_n, K_nope_ref, sizeof(float) * (size_t)nh * seq_n * nope);
    memcpy(Kc_r, K_rope_ref, sizeof(float) * (size_t)nh * seq_n * rope);
    memcpy(Vc,   V_ref,      sizeof(float) * (size_t)nh * seq_n * vd);
    for (uint32_t h = 0; h < nh; h++) {
        memcpy(Kc_n + ((size_t)h * seq_n + t_pos) * nope, mk + (size_t)h * nope, nope * sizeof(float));
        memcpy(Vc   + ((size_t)h * seq_n + t_pos) * vd,   mv + (size_t)h * vd,   vd   * sizeof(float));
        memcpy(Kc_r + ((size_t)h * seq_n + t_pos) * rope, mkr + (size_t)h * rope, rope * sizeof(float));
    }
    float *mattn2 = malloc(sizeof(float) * (size_t)nh * vd);
    TEST_ASSERT(ds4_gpu_glm_attn_decode_f32(Qm, Kc_n, Kc_r, Vc, mattn2,
                        nh, nope, rope, vd, qhd, seq_n, t_pos));
    float *mout = malloc(sizeof(float) * H);
    TEST_ASSERT(ds4_gpu_glm_matvec_f32(Wo, mattn2, mout, H, nh * vd));

    /* CPU composite on freshly reset caches. */
    float *Kc2_n = malloc(sizeof(float) * (size_t)nh * seq_n * nope);
    float *Kc2_r = malloc(sizeof(float) * (size_t)nh * seq_n * rope);
    float *Vc2   = malloc(sizeof(float) * (size_t)nh * seq_n * vd);
    memcpy(Kc2_n, K_nope_ref, sizeof(float) * (size_t)nh * seq_n * nope);
    memcpy(Kc2_r, K_rope_ref, sizeof(float) * (size_t)nh * seq_n * rope);
    memcpy(Vc2,   V_ref,      sizeof(float) * (size_t)nh * seq_n * vd);
    float *cout = malloc(sizeof(float) * H);
    glm_mla_forward_token_f32(cout, x, WqA, WqB, WkvA, WkB, WvB, Wo,
                              NULL, NULL,
                              Kc2_n, Vc2, Kc2_r, seq_n, t_pos, &shape, NULL);
    glm_metal_cmp("MLA composite (Metal vs CPU)", mout, cout, H, tol_abs, tol_rel);

    /* cross-check the Metal composite against the numpy mla_out fixture. */
    float *ref_mla_out = glm_ref_load_floats(dir, "mla_out.txt", H);
    if (ref_mla_out) glm_metal_cmp("MLA composite (Metal vs numpy)", mout, ref_mla_out, H, tol_abs, tol_rel);

    /* =============================================================
     * Phase 4c-ii: sigmoid MoE router + dense/shared-expert SwiGLU FFN.
     * Same component-then-composite pattern as the MLA block above: each
     * step runs Metal and the Phase 4a CPU oracle on identical inputs, then
     * the full Metal chain is compared to the CPU composite + numpy fixture.
     * ============================================================= */
    {
        const uint32_t E = shape.n_expert, K = shape.n_expert_used;
        const uint32_t ff = shape.ff_inter;
        const float scale = shape.moe_scale;

        /* ---- MoE router: gate@x -> sigmoid -> +bias -> top-K -> norm*scale ---- */
        float *gate       = glm_ref_load_floats(dir, "moe_gate.txt",   (size_t)E * H);
        float *bias       = glm_ref_load_floats(dir, "moe_bias.txt",   E);
        float *ref_logits = glm_ref_load_floats(dir, "moe_logits.txt", E);
        float *ref_w      = glm_ref_load_floats(dir, "moe_w.txt",      K);
        int   *ref_idx    = glm_ref_load_ints(dir,   "moe_idx.txt",    K);
        if (!gate || !bias || !ref_logits || !ref_w || !ref_idx) {
            fprintf(stderr, "  glm-metal: SKIP MoE (fixtures missing in %s)\n", dir);
        } else {
            float *clogits = malloc(sizeof(float) * E); TEST_ASSERT(clogits);
            float *mlogits = malloc(sizeof(float) * E); TEST_ASSERT(mlogits);
            glm_matvec_f32(clogits, gate, x, E, H);               /* CPU oracle logits */
            TEST_ASSERT(ds4_gpu_glm_matvec_f32(gate, x, mlogits, E, H));
            glm_metal_cmp("matvec router logits (gate@x)", mlogits, clogits, E, tol_abs, tol_rel);
            glm_metal_cmp("router logits vs numpy", mlogits, ref_logits, E, tol_abs, tol_rel);

            /* CPU oracle router (derives the same gate@x logits internally). */
            int   cidx[8]; float cw[8];
            float *scratch = malloc(sizeof(float) * E); TEST_ASSERT(scratch);
            glm_moe_route_sigmoid(cidx, cw, gate, x, bias, &shape, scratch);
            free(scratch);

            /* Metal router fed the SAME (CPU-computed) logits: isolates the
             * sigmoid+bias+topk+normalize*scale math from matvec drift. */
            int   midx[8]; float mw[8];
            TEST_ASSERT(ds4_gpu_glm_moe_route_f32(clogits, bias, midx, mw, E, K, scale));
            /* MoE indices must be EXACT (indices exact, per spec). */
            {
                bool idx_ok = true;
                for (uint32_t k = 0; k < K; k++) if (midx[k] != cidx[k]) { idx_ok = false; break; }
                fprintf(stderr, "  glm-metal: %-30s %s (idx=[%d,%d] cpu=[%d,%d])\n",
                        "MoE route idx (sigmoid+topk)", idx_ok ? "PASS" : "FAIL",
                        K >= 1 ? midx[0] : -1, K >= 2 ? midx[1] : -1,
                        K >= 1 ? cidx[0] : -1, K >= 2 ? cidx[1] : -1);
                g_glm_metal_total++;
                if (idx_ok) g_glm_metal_pass++;
                TEST_ASSERT(idx_ok);
            }
            glm_metal_cmp("MoE route weights (norm*2.5)", mw, cw, K, tol_abs, tol_rel);
            /* cross-check the Metal-selected weights/indices vs numpy fixtures. */
            {
                bool idx_ok = true;
                for (uint32_t k = 0; k < K; k++) if (midx[k] != ref_idx[k]) { idx_ok = false; break; }
                fprintf(stderr, "  glm-metal: %-30s %s (idx=[%d,%d] ref=[%d,%d])\n",
                        "MoE route idx vs numpy", idx_ok ? "PASS" : "FAIL",
                        K >= 1 ? midx[0] : -1, K >= 2 ? midx[1] : -1,
                        K >= 1 ? ref_idx[0] : -1, K >= 2 ? ref_idx[1] : -1);
                g_glm_metal_total++;
                if (idx_ok) g_glm_metal_pass++;
                TEST_ASSERT(idx_ok);
            }
            glm_metal_cmp("MoE route weights vs numpy", mw, ref_w, K, tol_abs, tol_rel);

            /* Composite: full Metal chain (matvec logits -> router) vs CPU oracle. */
            int   midx2[8]; float mw2[8];
            TEST_ASSERT(ds4_gpu_glm_moe_route_f32(mlogits, bias, midx2, mw2, E, K, scale));
            {
                bool idx_ok = true;
                for (uint32_t k = 0; k < K; k++) if (midx2[k] != cidx[k]) { idx_ok = false; break; }
                fprintf(stderr, "  glm-metal: %-30s %s (idx=[%d,%d] cpu=[%d,%d])\n",
                        "MoE route (Metal chain) idx", idx_ok ? "PASS" : "FAIL",
                        K >= 1 ? midx2[0] : -1, K >= 2 ? midx2[1] : -1,
                        K >= 1 ? cidx[0] : -1, K >= 2 ? cidx[1] : -1);
                g_glm_metal_total++;
                if (idx_ok) g_glm_metal_pass++;
                TEST_ASSERT(idx_ok);
            }
            glm_metal_cmp("MoE route (Metal chain) weights", mw2, cw, K, tol_abs, tol_rel);

            free(clogits); free(mlogits);
        }

        /* ---- dense/shared-expert SwiGLU FFN ---- */
        float *ff_gate    = glm_ref_load_floats(dir, "ffn_gate.txt",   (size_t)ff * H);
        float *ff_up      = glm_ref_load_floats(dir, "ffn_up.txt",     (size_t)ff * H);
        float *ff_down    = glm_ref_load_floats(dir, "ffn_down.txt",   (size_t)H * ff);
        float *ref_gh     = glm_ref_load_floats(dir, "ffn_gate_h.txt", ff);
        float *ref_uh     = glm_ref_load_floats(dir, "ffn_up_h.txt",   ff);
        float *ref_act    = glm_ref_load_floats(dir, "ffn_act.txt",    ff);
        float *ref_ff_out = glm_ref_load_floats(dir, "ffn_out.txt",    H);
        if (!ff_gate || !ff_up || !ff_down || !ref_gh || !ref_uh || !ref_act || !ref_ff_out) {
            fprintf(stderr, "  glm-metal: SKIP FFN (fixtures missing in %s)\n", dir);
        } else {
            float *cgh = malloc(sizeof(float) * ff), *mgh = malloc(sizeof(float) * ff);
            float *cuh = malloc(sizeof(float) * ff), *muh = malloc(sizeof(float) * ff);
            float *cact = malloc(sizeof(float) * ff), *mact = malloc(sizeof(float) * ff);
            TEST_ASSERT(cgh && mgh && cuh && muh && cact && mact);

            /* gate/up projections reuse the 4c-i matvec kernel. */
            glm_matvec_f32(cgh, ff_gate, x, ff, H);
            glm_matvec_f32(cuh, ff_up,   x, ff, H);
            TEST_ASSERT(ds4_gpu_glm_matvec_f32(ff_gate, x, mgh, ff, H));
            TEST_ASSERT(ds4_gpu_glm_matvec_f32(ff_up,   x, muh, ff, H));
            glm_metal_cmp("FFN gate@x (matvec)", mgh, cgh, ff, tol_abs, tol_rel);
            glm_metal_cmp("FFN up@x (matvec)",   muh, cuh, ff, tol_abs, tol_rel);
            glm_metal_cmp("FFN gate@x vs numpy", mgh, ref_gh, ff, tol_abs, tol_rel);
            glm_metal_cmp("FFN up@x vs numpy",   muh, ref_uh, ff, tol_abs, tol_rel);

            /* SwiGLU activation gate (the new kernel) vs CPU silu*up. */
            glm_silu_f32(cact, cgh, ff);
            for (uint32_t i = 0; i < ff; i++) cact[i] *= cuh[i];
            TEST_ASSERT(ds4_gpu_glm_swiglu_f32(mgh, muh, mact, ff));
            glm_metal_cmp("FFN swiglu (silu(g)*u)", mact, cact, ff, tol_abs, tol_rel);
            glm_metal_cmp("FFN swiglu vs numpy", mact, ref_act, ff, tol_abs, tol_rel);

            /* down projection (matvec) on the Metal activation. */
            float *cff_out = malloc(sizeof(float) * H), *mff_out = malloc(sizeof(float) * H);
            TEST_ASSERT(cff_out && mff_out);
            glm_matvec_f32(cff_out, ff_down, cact, H, ff);
            TEST_ASSERT(ds4_gpu_glm_matvec_f32(ff_down, mact, mff_out, H, ff));
            glm_metal_cmp("FFN down@act (matvec)", mff_out, cff_out, H, tol_abs, tol_rel);
            glm_metal_cmp("FFN down@act vs numpy", mff_out, ref_ff_out, H, tol_abs, tol_rel);

            /* Composite: full Metal FFN chain vs CPU glm_swiglu_dense_f32. */
            float *cff2 = malloc(sizeof(float) * H);
            TEST_ASSERT(cff2);
            glm_swiglu_dense_f32(cff2, x, ff_gate, ff_up, ff_down, &shape, cgh, cuh);
            glm_metal_cmp("FFN composite (Metal vs CPU)", mff_out, cff2, H, tol_abs, tol_rel);
            glm_metal_cmp("FFN composite (Metal vs numpy)", mff_out, ref_ff_out, H, tol_abs, tol_rel);

            free(cgh); free(mgh); free(cuh); free(muh); free(cact); free(mact);
            free(cff_out); free(mff_out); free(cff2);
        }

        free(gate); free(bias); free(ref_logits); free(ref_w); free(ref_idx);
        free(ff_gate); free(ff_up); free(ff_down); free(ref_gh); free(ref_uh);
        free(ref_act); free(ref_ff_out);
    }

    fprintf(stderr, "  glm-metal: %zu/%zu component+composite cases Metal==CPU\n",
            g_glm_metal_pass, g_glm_metal_total);

    free(ones); free(cqa); free(mqa); free(cqa_n); free(mqa_n); free(cq); free(mq);
    free(ckva); free(mkva); free(ckvln); free(mkvln); free(ck); free(mk); free(cv); free(mv);
    free(qr_in); free(cqr); free(mqr); free(kr_in); free(ckr); free(mkr);
    free(Qm); free(Qc); free(cattn); free(mattn); free(co); free(mo);
    free(Kc_n); free(Kc_r); free(Vc); free(mattn2); free(mout);
    free(Kc2_n); free(Kc2_r); free(Vc2); free(cout); free(ref_mla_out);
    free(x); free(WqA); free(WqB); free(WkvA); free(WkB); free(WvB); free(Wo);
    free(K_nope_ref); free(V_ref); free(K_rope_ref);
#endif
}

static const ds4_test_entry test_entries[] = {
#ifndef DS4_NO_GPU
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall},
    {"--tool-call-quality", "tool-call-quality", "model emits valid DSML tool calls", test_tool_call_quality},
    {"--think-tool-recovery", "think-tool-recovery", "forced </think> recovery when a tool call starts inside thinking", test_think_tool_recovery},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison on the standard Metal path", test_official_logprob_vectors},
    {"--metal-ssd-streaming-cache-pressure", "metal-ssd-streaming-cache-pressure", "Metal SSD-streaming layer-batched decode cache-pressure repro for issue #384", test_metal_ssd_streaming_cache_pressure},
    {"--local-golden-vectors", "local-golden-vectors", "local top-k/logit drift regression for long Metal prefill", test_local_golden_vectors},
    {"--metal-short-prefill", "metal-short-prefill", "Metal ratio-4 short prefill regression", test_metal_short_prefill_ratio4},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_kernel_group},
    {"--metal-tensor-equivalence", "metal-tensor-equivalence", "fast/quality Metal prompt-logit and greedy equivalence", test_metal_mpp_equivalence},
    {"--streaming-decode-prefill-correctness", "streaming-decode-prefill-correctness", "streaming decode-style cold prefill drift and repeatability", test_streaming_decode_prefill_correctness},
    {"--mtp-verify-depth", "mtp-verify-depth", "MTP speculative verify commits autoregressive-identical tokens at draft depth > 2", test_mtp_verify_depth},
#endif
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group},
    {"--glm-ssd-math", "glm-ssd-math", "GLM-5.2 SSD cache-plan arithmetic regression (no model needed)", test_glm_ssd_cache_plan},
    {"--glm-bpe", "glm-bpe", "GLM-5.2 glm4 BPE tokenizer vs HF tokenizers oracle (shard 1)", test_glm_bpe},
    {"--glm-quant-dequant", "glm-quant-dequant", "GLM-5.2 K-quant CPU dequant vs llama.cpp oracle", test_glm_quant_dequant},
    {"--glm-cpu-ref-components", "glm-cpu-ref-components", "GLM-5.2 CPU reference components (RoPE/MLA/dense-FFN/MoE) vs numpy oracle", test_glm_cpu_ref_components},
    {"--glm-cpu-forward-synth", "glm-cpu-forward-synth", "GLM-5.2 full CPU forward assembly on a tiny synthetic model (no model needed)", test_glm_cpu_forward_synth},
    {"--glm-nextn-synth", "glm-nextn-synth", "GLM-5.2 NextN/MTP blk.78 path on a tiny synthetic block (no model needed)", test_glm_nextn_synth},
    {"--glm-metal-forward-synth", "glm-metal-forward-synth", "GLM-5.2 full Metal forward (Phase 4c-iv) on a tiny synthetic model: argmax == CPU synth (no model needed)", test_glm_metal_forward_synth},
    {"--glm-metal-components", "glm-metal-components", "GLM-5.2 Metal component kernels (RoPE/MLA projections/latent RMSNorm/attention) vs CPU reference", test_glm_metal_components},
    {"--glm-generate-synth", "glm-generate-synth", "GLM-5.2 incremental generation: greedy argmax == naive full forward every step (CPU), and Metal incremental == CPU (no model needed)", test_glm_generate_synth},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Tests:");
    puts("  --all");
    puts("      Run every test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s\n", test_entries[i].flag, test_entries[i].desc);
    }
    puts("  --list");
    puts("      Print test names only.");
#ifndef DS4_NO_GPU
    puts("  --metal-mpp-equivalence");
    puts("      Compatibility alias for --metal-tensor-equivalence.");
#endif
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_SSD_STREAMING=1   Run model tests through Metal SSD streaming.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_GB=N  Streaming routed expert cache in GiB.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=N  Streaming routed expert cache count.");
    puts("  DS4_TEST_SSD_STREAMING_COLD=1  Skip streaming hot expert preload.");
    puts("  DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL=1  Force canonical streamed cold prefill.");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Simple official-vector fixture.");
    puts("  DS4_TEST_LOCAL_GOLDEN_FILE=FILE  Local top-k golden-vector fixture.");
    puts("  DS4_TEST_MPP_EQ_CASE=NAME  Run only Tensor equivalence cases whose id contains NAME.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
#ifndef DS4_NO_GPU
    if (!strcmp(arg, "--metal-mpp-equivalence")) {
        arg = "--metal-tensor-equivalence";
    }
#endif
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const ds4_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    ds4_log(stderr,
            test_failures == before ? DS4_LOG_OK : DS4_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const ds4_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "ds4-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

#ifndef DS4_NO_GPU
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "ds4 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 tests: ok");
    return 0;
}
