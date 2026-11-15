/* Internal V4.1 drafter graph. Included after the target graph in ds4.c. */
#ifndef DS4_DEEPSEEK41_DSPARK_H
#define DS4_DEEPSEEK41_DSPARK_H

#define DS41_DRAFT_WINDOW 128u
#define DS41_DRAFT_ROWS 5u
#define DS41_SUFFIX_DRAFT_ROWS 3u
#define DS41_VERIFY_ROWS (DS41_DRAFT_ROWS + 1u)
#define DS41_SUFFIX_BUCKETS 65536u
#define DS41_SUFFIX_VERIFIER_BYTES (UINT64_C(8) << 20)

struct ds41_dspark {
    const ds4_dspark_weights *weights;
    ds41_prefill_row batch;
    ds4_gpu_tensor *target[3], *window[3];
    uint32_t positions[3][DS41_DRAFT_WINDOW];
    uint32_t kv_end;
    bool kv_valid;
    ds4_gpu_tensor *mean_weights, *mean, *input, *projected, *main_x, *main_kv;
    ds4_gpu_tensor *logits, *tokens, *markov_embed, *markov_logits, *confidence_input, *confidence;
    float *host_logits, *draft_probs;
    ds4_gpu_tensor *verify_residual, *verify_pre, *verify_logits, *verify_pool;
    ds4_gpu_tensor *old_raw, *old_features;
    uint32_t old_positions[3][DS41_VERIFY_ROWS];
    ds41_gpu_graph verified[DS41_VERIFY_ROWS + 1u];
    float *host_verified;
    uint32_t verify_start, verify_count, rewind_end;
    uint32_t suffix_cap, suffix_indexed_len, zero_streak;
    float accept_ema;
    float v0_ms, vk_ms[DS41_VERIFY_ROWS];
    int32_t *suffix_tokens, *suffix_prev3;
    int32_t *suffix_head3, *recycle_top1;
};

static void ds41_dspark_reset(ds41_dspark *d) {
    if (!d) return;
    memset(d->positions, 0xff, sizeof(d->positions));
    d->kv_valid = false;
    d->kv_end = 0;
    d->verify_count = 0;
    d->rewind_end = 0;
    d->suffix_indexed_len = 0;
    d->zero_streak = 0;
    d->accept_ema = 0.5f;
    d->v0_ms = 26.2f;
    static const float vk_init[DS41_VERIFY_ROWS] = {
        26.2f, 40.8f, 55.5f, 70.5f, 108.0f, 135.0f
    };
    for (unsigned i = 0; i < DS41_VERIFY_ROWS; i++) d->vk_ms[i] = vk_init[i];
    if (d->suffix_head3) memset(d->suffix_head3, 0xff, DS41_SUFFIX_BUCKETS * sizeof(int32_t));
    if (d->recycle_top1) memset(d->recycle_top1, 0xff, DS4_N_VOCAB * sizeof(int32_t));
}

static void ds41_dspark_free(ds41_dspark *d) {
    if (!d) return;
#define DS41_DRAFT_FREE(name, count) ds4_gpu_tensor_free(d->batch.name);
    DS41_PREFILL_ROWS(DS41_DRAFT_FREE)
#undef DS41_DRAFT_FREE
    for (unsigned i = 0; i < 3; i++) {
        ds4_gpu_tensor_free(d->target[i]);
        ds4_gpu_tensor_free(d->window[i]);
    }
    ds4_gpu_tensor_free(d->mean_weights); ds4_gpu_tensor_free(d->mean);
    ds4_gpu_tensor_free(d->input); ds4_gpu_tensor_free(d->projected);
    ds4_gpu_tensor_free(d->main_x); ds4_gpu_tensor_free(d->main_kv);
    ds4_gpu_tensor_free(d->logits); ds4_gpu_tensor_free(d->markov_embed);
    ds4_gpu_tensor_free(d->tokens);
    ds4_gpu_tensor_free(d->markov_logits); ds4_gpu_tensor_free(d->confidence_input);
    ds4_gpu_tensor_free(d->confidence);
    for (unsigned i = 0; i <= DS41_VERIFY_ROWS; i++) {
        ds41_gpu_graph *g = &d->verified[i];
        ds4_gpu_tensor_free(g->residual); ds4_gpu_tensor_free(g->pre);
        ds4_gpu_tensor_free(g->logits);
        ds4_gpu_tensor_free(g->tp_logits_half);
        for (unsigned j = 0; j < 4; j++) {
            ds4_gpu_tensor_free(g->verify_previous_kv[j]);
            ds4_gpu_tensor_free(g->verify_previous_score[j]);
        }
    }
    ds4_gpu_tensor_free(d->verify_residual); ds4_gpu_tensor_free(d->verify_pre);
    ds4_gpu_tensor_free(d->verify_logits); ds4_gpu_tensor_free(d->verify_pool);
    ds4_gpu_tensor_free(d->old_raw); ds4_gpu_tensor_free(d->old_features);
    free(d->host_verified);
    free(d->host_logits);
    free(d->draft_probs);
    free(d->suffix_tokens);
    free(d->suffix_prev3);
    free(d->suffix_head3);
    free(d->recycle_top1);
    free(d);
}

static DS4_MAYBE_UNUSED ds41_dspark *ds41_dspark_alloc_verifier(uint32_t ctx_cap) {
    if (!ctx_cap) return NULL;
    ds41_dspark *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->suffix_cap = ctx_cap;
#define DS41_VERIFY_A(name, count) \
    if (!(d->name = ds4_gpu_tensor_alloc((uint64_t)(count) * sizeof(float)))) goto fail;
    DS41_VERIFY_A(verify_residual, (DS41_VERIFY_ROWS + 1u) * DS4_N_HC * DS4_N_EMBD)
    DS41_VERIFY_A(verify_pre, (DS41_VERIFY_ROWS + 1u) * DS4_N_HC)
    DS41_VERIFY_A(verify_logits, (DS41_VERIFY_ROWS + 1u) * DS4_N_VOCAB)
    DS41_VERIFY_A(verify_pool, (DS41_VERIFY_ROWS + 1u) * 8u * DS4_N_HEAD_DIM)
    DS41_VERIFY_A(old_raw, 40u * DS41_VERIFY_ROWS * DS4_N_HEAD_DIM)
#undef DS41_VERIFY_A
    d->host_verified = malloc((uint64_t)(DS41_VERIFY_ROWS + 1u) * DS4_N_VOCAB * sizeof(float));
    d->suffix_tokens = malloc((size_t)ctx_cap * sizeof(int32_t));
    d->suffix_prev3 = malloc((size_t)ctx_cap * sizeof(int32_t));
    d->suffix_head3 = malloc(DS41_SUFFIX_BUCKETS * sizeof(int32_t));
    d->recycle_top1 = malloc(DS4_N_VOCAB * sizeof(int32_t));
    if (!d->host_verified || !d->suffix_tokens || !d->suffix_prev3 ||
        !d->suffix_head3 || !d->recycle_top1)
        goto fail;
    ds41_dspark_reset(d);
#ifdef __APPLE__
    ds4_gpu_warm_v41_verify_pipelines();
#endif
    return d;
fail:
    ds41_dspark_free(d);
    return NULL;
}

static inline uint32_t ds41_suffix_hash3(int a, int b, int c) {
    uint32_t h = (uint32_t)a * 3266489917u ^ ((uint32_t)b * 668265263u) ^ ((uint32_t)c * 374761393u);
    return (h ^ (h >> 16)) & (DS41_SUFFIX_BUCKETS - 1u);
}

static inline void ds41_suffix_note_timing(ds41_dspark *d, uint32_t verify_rows, double elapsed_ms) {
    if (!d || !verify_rows || verify_rows > DS41_VERIFY_ROWS || elapsed_ms <= 1.0) return;
    const float ms = (float)elapsed_ms;
    if (verify_rows == 1u) {
        d->v0_ms = 0.88f * d->v0_ms + 0.12f * ms;
    } else {
        d->vk_ms[verify_rows - 1u] = 0.88f * d->vk_ms[verify_rows - 1u] + 0.12f * ms;
    }
}

static inline float ds41_suffix_marginal_cost(const ds41_dspark *d, uint32_t k) {
    if (!d || k == 0u || k >= DS41_VERIFY_ROWS || d->v0_ms <= 1.0f) return 0.50f;
    const float prev = (k == 1u) ? d->v0_ms : d->vk_ms[k - 1u];
    float ck = (d->vk_ms[k] - prev) / d->v0_ms;
    if (ck < 0.12f) ck = 0.12f;
    if (ck > 0.85f) ck = 0.85f;
    return ck;
}

static inline void ds41_suffix_note_recycle(ds41_dspark *d, int prev_token, int next_top1) {
    if (d && d->recycle_top1 && prev_token >= 0 && (uint32_t)prev_token < DS4_N_VOCAB &&
        next_top1 >= 0 && (uint32_t)next_top1 < DS4_N_VOCAB) {
        d->recycle_top1[prev_token] = next_top1;
    }
}

static inline void ds41_suffix_note_outcome(ds41_dspark *d, uint32_t proposed, uint32_t accepted) {
    if (!d || !proposed) return;
    if (accepted == 0) d->zero_streak++;
    else d->zero_streak = 0;
    d->accept_ema = 0.85f * d->accept_ema + 0.15f * ((float)accepted / (float)proposed);
}

static void ds41_suffix_sync_index(ds41_dspark *d, const int *v, uint32_t len) {
    if (!d || !d->suffix_tokens) return;
    if (len > d->suffix_cap) len = d->suffix_cap;
    bool rebuild = len < d->suffix_indexed_len;
    if (!rebuild && d->suffix_indexed_len > 0) {
        const uint32_t last = d->suffix_indexed_len - 1u;
        const uint32_t mid = d->suffix_indexed_len >> 1;
        if (d->suffix_tokens[last] != v[last] ||
            d->suffix_tokens[0] != v[0] ||
            d->suffix_tokens[mid] != v[mid] ||
            memcmp(d->suffix_tokens, v, (size_t)d->suffix_indexed_len * sizeof(int32_t)) != 0) {
            rebuild = true;
        }
    }
    if (rebuild) {
        memset(d->suffix_head3, 0xff, DS41_SUFFIX_BUCKETS * sizeof(int32_t));
        d->suffix_indexed_len = 0;
    }
    for (uint32_t i = d->suffix_indexed_len; i < len; i++) {
        d->suffix_tokens[i] = v[i];
        if (i >= 2u) {
            const uint32_t h3 = ds41_suffix_hash3(v[i - 2u], v[i - 1u], v[i]);
            d->suffix_prev3[i] = d->suffix_head3[h3];
            d->suffix_head3[h3] = (int32_t)i;
        } else {
            d->suffix_prev3[i] = -1;
        }
    }
    d->suffix_indexed_len = len;
}

static inline int ds41_suffix_seq_at(const int *v, uint32_t n, const int *tokens, uint32_t idx) {
    return idx < n ? v[idx] : tokens[idx - n];
}

/* Lan-staircase (arXiv:2610.34440) live-calibrated 3-gram SuffixDecoding +
 * Token Recycling proposer. Extends a draft to step k only while cumulative
 * empirical continuation probability P_cum exceeds 1.32 * c_k, where
 * c_k = (V(k) - V(k-1)) / V(0) is calibrated online from actual Metal verify
 * latencies under the active thermal/power envelope. */
static DS4_MAYBE_UNUSED uint32_t ds41_suffix_propose(
        ds41_dspark *d, const int *v, uint32_t n, int first_token,
        uint32_t cap, int eos, bool in_thinking, int *tokens) {
    tokens[0] = first_token;
    /* Inside <think> free-form reasoning, IQ1_S MoE 8-expert fanout makes
     * multi-row verify cost ~0.56 * V(0) per draft row while prose paraphrases
     * prompt phrases; restrict speculation to post-</think> structured output
     * (code, tables, patches, and <｜DSML｜...> tool calls). */
    if (!d || !d->suffix_tokens || in_thinking || cap < 2u || n < 1u) return 1u;
    if (cap > DS41_VERIFY_ROWS) cap = DS41_VERIFY_ROWS;
    if (n > d->suffix_cap) n = d->suffix_cap;
    ds41_suffix_sync_index(d, v, n);
    /* Keep verify batches within count <= 4 (max_drafts <= 3) so every dense
     * Q8_0 projection stays on the single-pass seq4 kernel family. */
    const uint32_t max_drafts = (cap - 1u) < 3u ? (cap - 1u) : 3u;

    int32_t active_p[48];
    uint32_t active_n = 0;
    uint32_t best_l = 0;
    if (n >= 2u) {
        const int t2 = v[n - 2u], t1 = v[n - 1u], t0 = first_token;
        const uint32_t h3 = ds41_suffix_hash3(t2, t1, t0);
        uint32_t scanned = 0;
        for (int32_t p = d->suffix_head3[h3]; p >= 2 && scanned < 96u; p = d->suffix_prev3[p]) {
            if (v[p] != t0 || v[p - 1] != t1 || v[p - 2] != t2) continue;
            scanned++;
            uint32_t l = 3u;
            const uint32_t max_l = p + 1 < (int32_t)(n + 1u) ? (uint32_t)(p + 1) : (n + 1u);
            const uint32_t lim = max_l < 64u ? max_l : 64u;
            while (l < lim && v[(uint32_t)p - l] == ds41_suffix_seq_at(v, n, tokens, n - l)) l++;
            if (l > best_l) {
                best_l = l;
                active_p[0] = p;
                active_n = 1u;
            } else if (l == best_l && active_n < 48u) {
                active_p[active_n++] = p;
            }
        }
    }

    if (best_l >= 6u && active_n > 0u) {
        const int recycled0 = (first_token >= 0 && (uint32_t)first_token < DS4_N_VOCAB && d->recycle_top1) ?
            d->recycle_top1[first_token] : -1;
        const float streak_scale = d->zero_streak >= 2u ? 0.80f : 1.0f;
        float p_cum = 1.0f;
        uint32_t count = 1u;
        for (uint32_t k = 1u; k <= max_drafts && active_n > 0u; k++) {
            int cand_tok[48];
            int32_t cand_pos[48];
            uint32_t valid_n = 0;
            for (uint32_t i = 0; i < active_n; i++) {
                const uint32_t src_idx = (uint32_t)active_p[i] + k;
                if (src_idx >= n + count) continue;
                const int tok = ds41_suffix_seq_at(v, n, tokens, src_idx);
                if (tok < 0 || (uint32_t)tok >= DS4_N_VOCAB) continue;
                cand_pos[valid_n] = active_p[i];
                cand_tok[valid_n] = tok;
                valid_n++;
            }
            if (!valid_n) break;

            int modal_tok = cand_tok[0];
            uint32_t modal_agree = 0;
            for (uint32_t i = 0; i < valid_n; i++) {
                uint32_t c = 0;
                for (uint32_t j = 0; j < valid_n; j++) {
                    if (cand_tok[j] == cand_tok[i]) c++;
                }
                if (c > modal_agree) {
                    modal_agree = c;
                    modal_tok = cand_tok[i];
                }
            }

            float p_step = 0.0f;
            if (valid_n == 1u) {
                if (best_l >= 12u) p_step = 0.92f;
                else if (best_l >= 8u) p_step = 0.85f;
                else p_step = 0.74f;
            } else {
                float prior = 0.72f;
                float psu = 1.0f;
                if (best_l >= 8u) { prior = 0.85f; psu = 1.0f; }
                p_step = ((float)modal_agree + psu * prior) / ((float)valid_n + psu);
                if (k == 1u && modal_tok == recycled0 && p_step > 0.0f) {
                    p_step += 0.06f;
                    if (p_step > 0.96f) p_step = 0.96f;
                }
            }
            p_cum *= p_step * streak_scale;
            if (p_cum <= 1.32f * ds41_suffix_marginal_cost(d, k)) break;

            tokens[count++] = modal_tok;
            if (modal_tok == eos) break;

            uint32_t next_n = 0;
            for (uint32_t i = 0; i < valid_n; i++) {
                if (cand_tok[i] == modal_tok) active_p[next_n++] = cand_pos[i];
            }
            active_n = next_n;
        }
        if (count > 1u) return count;
    }
    return 1u;
}

static DS4_MAYBE_UNUSED ds41_dspark *ds41_dspark_alloc(const ds4_dspark_weights *dw) {
    if (!dw || !dw->v41 || dw->n_stages != 3 || dw->block_size != DS41_DRAFT_ROWS ||
        dw->missing_tensors || dw->invalid_tensors || dw->metadata_errors) return NULL;
    ds41_dspark *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->weights = dw;
    const ds41_gpu_graph shape = {.ctx = DS41_DRAFT_WINDOW}, *g = &shape;
#define DS41_DRAFT_ALLOC(name, width) \
    if (!(d->batch.name = ds4_gpu_tensor_alloc((uint64_t)(width) * DS41_DRAFT_ROWS * sizeof(float)))) goto fail;
    DS41_PREFILL_ROWS(DS41_DRAFT_ALLOC)
#undef DS41_DRAFT_ALLOC
    for (unsigned i = 0; i < 3; i++) {
        d->target[i] = ds4_gpu_tensor_alloc((uint64_t)DS41_DRAFT_WINDOW * DS4_N_EMBD * 4u);
        d->window[i] = ds4_gpu_tensor_alloc((uint64_t)(DS41_DRAFT_WINDOW + DS41_DRAFT_ROWS) * DS4_N_HEAD_DIM * 4u);
        if (!d->target[i] || !d->window[i]) goto fail;
    }
#define DS41_DRAFT_A(name, count) \
    if (!(d->name = ds4_gpu_tensor_alloc((uint64_t)(count) * sizeof(float)))) goto fail;
    DS41_DRAFT_A(mean_weights, DS41_DRAFT_WINDOW * DS4_N_HC)
    DS41_DRAFT_A(mean, DS41_DRAFT_WINDOW * DS4_N_EMBD)
    DS41_DRAFT_A(input, DS41_DRAFT_WINDOW * 3u * DS4_N_EMBD)
    DS41_DRAFT_A(projected, DS41_DRAFT_WINDOW * DS4_N_EMBD)
    DS41_DRAFT_A(main_x, DS41_DRAFT_WINDOW * DS4_N_EMBD)
    DS41_DRAFT_A(main_kv, DS41_DRAFT_WINDOW * DS4_N_HEAD_DIM)
    DS41_DRAFT_A(logits, DS41_DRAFT_ROWS * DS4_N_VOCAB)
    DS41_DRAFT_A(tokens, DS41_DRAFT_ROWS)
    DS41_DRAFT_A(markov_embed, dw->markov_rank)
    DS41_DRAFT_A(markov_logits, DS4_N_VOCAB)
    DS41_DRAFT_A(confidence_input, DS4_N_EMBD + dw->markov_rank)
    DS41_DRAFT_A(confidence, 1)
    DS41_DRAFT_A(verify_residual, (DS41_VERIFY_ROWS + 1u) * DS4_N_HC * DS4_N_EMBD)
    DS41_DRAFT_A(verify_pre, (DS41_VERIFY_ROWS + 1u) * DS4_N_HC)
    DS41_DRAFT_A(verify_logits, (DS41_VERIFY_ROWS + 1u) * DS4_N_VOCAB)
    DS41_DRAFT_A(verify_pool, (DS41_VERIFY_ROWS + 1u) * 8u * DS4_N_HEAD_DIM)
    DS41_DRAFT_A(old_raw, 40u * DS41_VERIFY_ROWS * DS4_N_HEAD_DIM)
    DS41_DRAFT_A(old_features, 3u * DS41_VERIFY_ROWS * DS4_N_EMBD)
#undef DS41_DRAFT_A
    d->host_logits = malloc((uint64_t)DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float));
    d->draft_probs = malloc((uint64_t)DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float));
    d->host_verified = malloc((uint64_t)(DS41_VERIFY_ROWS + 1u) * DS4_N_VOCAB * sizeof(float));
    if (!d->host_logits || !d->draft_probs || !d->host_verified || !ds4_gpu_tensor_fill_f32(d->mean_weights,
        1.0f / DS4_N_HC, DS41_DRAFT_WINDOW * DS4_N_HC) ||
        !ds4_gpu_tensor_fill_f32(d->tokens, 0, DS41_DRAFT_ROWS)) goto fail;
    ds41_dspark_reset(d);
#ifdef __APPLE__
    ds4_gpu_warm_v41_verify_pipelines();
#endif
    return d;
fail:
    ds41_dspark_free(d);
    return NULL;
}

static bool ds41_dspark_capture(ds41_dspark *d, uint32_t layer, uint32_t pos,
                                const ds4_gpu_tensor *residual, uint32_t rows) {
    if (!d) return true;
    d->rewind_end = 0;
    if (!d->weights) return true;
    unsigned stage = 0;
    while (stage < 3 && d->weights->target_layers[stage] != layer) stage++;
    if (stage == 3) return true;
    if (!rows || rows > UINT32_MAX - pos) return false;
    const uint32_t count = rows < DS41_DRAFT_WINDOW ? rows : DS41_DRAFT_WINDOW;
    const uint32_t first = rows - count, start = pos + first;
    const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *src = ds4_gpu_tensor_view((ds4_gpu_tensor *)residual,
        first * DS4_N_HC * bytes, count * DS4_N_HC * bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_view(d->mean, 0, count * bytes);
    ds4_gpu_tensor *mix = ds4_gpu_tensor_view(d->mean_weights, 0, count * DS4_N_HC * sizeof(float));
    bool ok = src && out && mix && ds4_gpu_hc_weighted_sum_tensor(out, src, mix, DS4_N_EMBD, DS4_N_HC) &&
        ds4_gpu_dsv41_quantize(out, DS4_N_EMBD, count, DS4_V41_BF16);
    const uint32_t slot = start % DS41_DRAFT_WINDOW;
    const uint32_t part = count < DS41_DRAFT_WINDOW - slot ? count : DS41_DRAFT_WINDOW - slot;
    if (ok) ok = ds4_gpu_tensor_copy(d->target[stage], slot * bytes, out, 0, part * bytes) &&
        (part == count || ds4_gpu_tensor_copy(d->target[stage], 0, out, part * bytes, (count - part) * bytes));
    ds4_gpu_tensor_free(mix); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(src);
    if (ok) for (uint32_t i = 0; i < count; i++)
        d->positions[stage][(start + i) % DS41_DRAFT_WINDOW] = start + i;
    return ok;
}

static bool ds41_dspark_matmul_batch(ds4_gpu_tensor *out, const ds4_model *m,
        const ds4_tensor *weight, const ds4_gpu_tensor *in, uint32_t count, bool round) {
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    /* Draft rows share a GEMM. Target verification retains its scalar-reduction
     * kernels; proposal arithmetic need not reproduce their reduction order. */
    if (count > 1 && weight->type == DS4_TENSOR_F16 &&
        !getenv("DS4_DISABLE_V41_DRAFT_GEMM"))
        return metal_graph_matmul_plain_tensor(out, m, weight, weight->dim[0],
            weight->dim[1], in, count) && (!round ||
            ds4_gpu_dsv41_quantize(out, (uint32_t)weight->dim[1], count, DS4_V41_BF16));
#endif
    return ds41_matmul_batch(out, m, weight, in, count, round);
}

/* Only accepted target features enter these rings. A fresh prefill seeds the
 * last window once; subsequent calls project only the newly accepted rows. */
static bool ds41_dspark_seed(ds41_dspark *d, const ds4_model *m, uint32_t pos) {
    if (!pos) return false;
    const uint32_t oldest = pos > DS41_DRAFT_WINDOW ? pos - DS41_DRAFT_WINDOW : 0;
    const uint32_t start = d->kv_valid && d->kv_end >= oldest && d->kv_end <= pos ? d->kv_end : oldest;
    const uint32_t count = pos - start;
    if (!count) return true;
    const uint64_t row_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t slot = (start + i) % DS41_DRAFT_WINDOW;
        for (unsigned stage = 0; stage < 3; stage++) {
            if (d->positions[stage][slot] != start + i ||
                !ds4_gpu_tensor_copy(d->input, (i * 3u + stage) * row_bytes,
                    d->target[stage], slot * row_bytes, row_bytes)) return false;
        }
    }
    const ds4_dspark_stage_weights *first = &d->weights->stage[0];
    if (!ds41_dspark_matmul_batch(d->projected, m, first->main_proj, d->input, count, true) ||
        !ds41_norm_batch(d->main_x, d->projected, m, first->main_norm, count)) return false;
    const uint64_t kv_bytes = DS4_N_HEAD_DIM * sizeof(float);
    for (unsigned stage = 0; stage < 3; stage++) {
        const ds4_layer_weights *l = &d->weights->stage[stage].block;
        if (!ds41_dspark_matmul_batch(d->main_kv, m, l->attn_kv, d->main_x, count, true) ||
            !ds41_norm_batch(d->main_kv, d->main_kv, m, l->attn_kv_a_norm, count) ||
            !ds4_gpu_dsv41_rope(d->main_kv, DS4_N_HEAD_DIM, 1, count, start, false, false) ||
            !ds4_gpu_dsv41_quantize(d->main_kv, DS4_N_HEAD_DIM, count, DS4_V41_FP8_E8M0)) return false;
        const uint32_t slot = start % DS41_DRAFT_WINDOW;
        const uint32_t part = count < DS41_DRAFT_WINDOW - slot ? count : DS41_DRAFT_WINDOW - slot;
        if (!ds4_gpu_tensor_copy(d->window[stage], slot * kv_bytes, d->main_kv, 0, part * kv_bytes) ||
            (part < count && !ds4_gpu_tensor_copy(d->window[stage], 0, d->main_kv,
                part * kv_bytes, (count - part) * kv_bytes))) return false;
    }
    d->kv_end = pos;
    d->kv_valid = true;
    return true;
}

static bool ds41_dspark_attention_output(ds41_prefill_row *b, const ds4_model *m,
                                          const ds4_layer_weights *l) {
    const uint32_t groups = DS4_N_OUT_GROUP, width = DS4_N_HEAD * DS4_N_HEAD_DIM / groups;
    uint64_t weight_row;
    if (!tensor_nbytes(l->attn_output_a->type, width, &weight_row)) return false;
    for (uint32_t row = 0; row < DS41_DRAFT_ROWS; row++) {
        for (uint32_t group = 0; group < groups; group++) {
            ds4_tensor slice = *l->attn_output_a;
            slice.abs_offset += group * DS4_N_LORA_O * weight_row;
            slice.dim[1] = DS4_N_LORA_O;
            ds4_gpu_tensor *x = ds4_gpu_tensor_view(b->heads,
                ((uint64_t)row * groups + group) * width * 4u, width * 4u);
            ds4_gpu_tensor *y = ds4_gpu_tensor_view(b->low,
                ((uint64_t)row * groups + group) * DS4_N_LORA_O * 4u, DS4_N_LORA_O * 4u);
            const bool ok = x && y && ds41_matmul(y, m, &slice, x, true);
            ds4_gpu_tensor_free(y); ds4_gpu_tensor_free(x);
            if (!ok) return false;
        }
    }
    return ds41_dspark_matmul_batch(b->block, m, l->attn_output_b, b->low, DS41_DRAFT_ROWS, true);
}

static bool ds41_dspark_stage(ds41_dspark *d, const ds4_model *m, uint32_t stage, uint32_t pos) {
    ds41_prefill_row *b = &d->batch;
    const ds4_layer_weights *l = &d->weights->stage[stage].block;
    const uint32_t n = DS41_DRAFT_ROWS, past = pos < DS41_DRAFT_WINDOW ? pos : DS41_DRAFT_WINDOW;
    if (!ds41_hc_mix_batch(b, m, l, false, n) ||
        !(stage ? ds4_gpu_hc_weighted_sum_split_tensor(b->x, b->residual, b->ffn_split, DS4_N_EMBD, DS4_N_HC) :
                  ds4_gpu_hc_weighted_sum_tensor(b->x, b->residual, b->pre, DS4_N_EMBD, DS4_N_HC)) ||
        !ds4_gpu_dsv41_quantize(b->x, DS4_N_EMBD, n, DS4_V41_BF16) ||
        !ds41_norm_batch(b->norm, b->x, m, l->attn_norm, n) ||
        !ds41_dspark_matmul_batch(b->qr, m, l->attn_q_a, b->norm, n, true) ||
        !ds41_norm_batch(b->qr, b->qr, m, l->attn_q_a_norm, n) ||
        !ds41_dspark_matmul_batch(b->q, m, l->attn_q_b, b->qr, n, true) ||
        !ds4_gpu_dsv41_rope(b->q, DS4_N_HEAD_DIM, DS4_N_HEAD, n, pos, false, false) ||
        !ds41_dspark_matmul_batch(b->kv, m, l->attn_kv, b->norm, n, true) ||
        !ds41_norm_batch(b->kv, b->kv, m, l->attn_kv_a_norm, n) ||
        !ds4_gpu_dsv41_rope(b->kv, DS4_N_HEAD_DIM, 1, n, pos, false, false) ||
        !ds4_gpu_dsv41_quantize(b->kv, DS4_N_HEAD_DIM, n, DS4_V41_FP8_E8M0) ||
        !ds4_gpu_tensor_copy(d->window[stage], (uint64_t)past * DS4_N_HEAD_DIM * 4u,
            b->kv, 0, (uint64_t)n * DS4_N_HEAD_DIM * 4u) ||
        !ds4_gpu_attention_noncausal_raw_batch_heads_tensor(b->heads, m->map, m->size,
            l->attn_sinks->abs_offset, b->q, d->window[stage], n, past + n,
            DS41_DRAFT_WINDOW + n, 0, DS4_N_HEAD, DS4_N_HEAD_DIM) ||
        !ds4_gpu_dsv41_quantize(b->heads, DS4_N_HEAD * DS4_N_HEAD_DIM, n, DS4_V41_BF16) ||
        !ds4_gpu_dsv41_rope(b->heads, DS4_N_HEAD_DIM, DS4_N_HEAD, n, pos, false, true) ||
        !ds41_dspark_attention_output(b, m, l) ||
        !ds41_after_attention_batch(b, m, l, n, false)) {
        fprintf(stderr, "ds4: V4.1 draft attention failed\n");
        return false;
    }
    const uint32_t experts = d->weights->n_experts, active = d->weights->n_experts_used;
    const uint64_t gate_row = routed_expert_row_bytes(l->ffn_gate_exps);
    const uint64_t down_row = routed_expert_row_bytes(l->ffn_down_exps);
    bool mid_f16 = false;
    if (!(ds41_dspark_matmul_batch(b->route_logits, m, l->ffn_gate_inp, b->norm, n, false) &&
        ds4_gpu_router_select_batch_tensor(b->selected, b->route_weights, b->route_probs,
            m->map, m->size, l->ffn_exp_probs_b->abs_offset, 0, 0, 0, 0, true, false,
            b->route_logits, d->tokens, experts, active, DS4_EXPERT_WEIGHT_SCALE, n))) {
        fprintf(stderr, "ds4: V4.1 draft routing failed\n"); return false;
    }
    if (!(ds41_dspark_matmul_batch(b->shared_gate, m, l->ffn_gate_shexp, b->norm, n, true) &&
        ds41_dspark_matmul_batch(b->shared_up, m, l->ffn_up_shexp, b->norm, n, true) &&
        ds4_gpu_swiglu_tensor(b->shared_mid, b->shared_gate, b->shared_up,
            n * DS4_N_FF_EXP, DS4_SWIGLU_CLAMP_EXP, 1.0f) &&
        ds4_gpu_dsv41_quantize(b->shared_mid, DS4_N_FF_EXP, n, DS4_V41_BF16) &&
        ds41_dspark_matmul_batch(b->shared, m, l->ffn_down_shexp, b->shared_mid, n, true))) {
        fprintf(stderr, "ds4: V4.1 draft shared expert failed\n"); return false;
    }
    if (!ds4_gpu_routed_moe_batch_tensor(b->routed, b->gate, b->up, b->mid, b->experts,
            m->map, m->size, l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset,
            l->ffn_down_exps->abs_offset, l->ffn_gate_exps->type, l->ffn_down_exps->type,
            gate_row * DS4_N_FF_EXP, gate_row, down_row * DS4_N_EMBD, down_row,
            DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD, b->selected, b->route_weights,
            experts, active, DS4_SWIGLU_CLAMP_EXP, b->norm, stage, n, &mid_f16, true)) {
        fprintf(stderr, "ds4: V4.1 draft routed experts failed\n"); return false;
    }
    return ds4_gpu_add_tensor(b->block, b->routed, b->shared, n * DS4_N_EMBD) &&
        ds4_gpu_dsv41_quantize(b->block, DS4_N_EMBD, n, DS4_V41_BF16) &&
        ds4_gpu_hc_expand_split_tensor(b->residual, b->block, b->after_attn,
            b->ffn_split, DS4_N_EMBD, DS4_N_HC) &&
        ds4_gpu_dsv41_quantize(b->residual, DS4_N_HC * DS4_N_EMBD, n, DS4_V41_BF16);
}

/* Leaves the five base-logit rows and unnormalized collapsed features on the
 * GPU. Markov sampling is sequential and performed after this stage chain. */
static DS4_MAYBE_UNUSED bool ds41_dspark_forward(ds41_dspark *d, const ds4_model *target,
                                 const ds4_weights *tw, const ds4_model *draft,
                                 uint32_t pos, int token, ds41_gpu_graph *target_graph) {
    if (!d || !pos || token < 0 || (uint32_t)token >= DS4_N_VOCAB) return false;
    ds41_prefill_row *b = &d->batch;
    const bool local_experts = target_graph && target_graph->tp_world == 2;
    if (local_experts) ds4_gpu_tp_suspend_expert_sharding(1);
    bool ok = ds4_gpu_begin_commands() && ds41_dspark_seed(d, draft, pos);
    if (!ok) fprintf(stderr, "ds4: V4.1 DSpark seed failed\n");
    const float pre[DS41_DRAFT_ROWS][4] = {{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}};
    if (ok) ok = ds4_gpu_tensor_write(b->pre, 0, pre, sizeof(pre));
    for (uint32_t row = 0; ok && row < DS41_DRAFT_ROWS; row++) {
        const uint64_t bytes = (uint64_t)DS4_N_EMBD * DS4_N_HC * sizeof(float);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(b->residual, row * bytes, bytes);
        ok = out && ds4_gpu_embed_token_hc_tensor(out, target->map, target->size,
            tw->token_embd->abs_offset, DS4_N_VOCAB, row ? d->weights->noise_token_id : (uint32_t)token,
            DS4_N_EMBD, DS4_N_HC);
        ds4_gpu_tensor_free(out);
    }
    for (uint32_t stage = 0; ok && stage < 3; stage++) {
        ok = ds41_dspark_stage(d, draft, stage, pos);
        if (!ok) fprintf(stderr, "ds4: V4.1 DSpark stage %u failed\n", stage);
    }
    if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(b->x, b->residual,
        b->ffn_split, DS4_N_EMBD, DS4_N_HC) &&
        ds4_gpu_dsv41_quantize(b->x, DS4_N_EMBD, DS41_DRAFT_ROWS, DS4_V41_BF16) &&
        ds41_norm_batch(b->norm, b->x, draft, d->weights->stage[2].norm, DS41_DRAFT_ROWS);
    if (ok) ok = target_graph ?
        ds41_output_projection(target_graph, d->logits, target, tw, b->norm, DS41_DRAFT_ROWS) :
        ds41_matmul_batch(d->logits, target, tw->output, b->norm, DS41_DRAFT_ROWS, false);
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = false;
    if (local_experts) ds4_gpu_tp_suspend_expert_sharding(0);
    if (!ok) d->kv_valid = false;
    return ok;
}

static DS4_MAYBE_UNUSED bool ds41_dspark_markov(ds41_dspark *d, const ds4_model *m,
                               uint32_t row, int previous, float *confidence) {
    if (!d || row >= DS41_DRAFT_ROWS || previous < 0 ||
        (uint32_t)previous >= DS4_N_VOCAB || !confidence) return false;
    const ds4_dspark_stage_weights *last = &d->weights->stage[2];
    float embedding[256];
    const uint32_t rank = d->weights->markov_rank;
    if (rank != 256 || !dspark_dense_row_to_f32(embedding, m, last->markov_w1, (uint32_t)previous))
        return false;
    const uint64_t bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_view(d->logits, row * bytes, bytes);
    bool ok = logits && ds4_gpu_begin_commands() &&
        ds4_gpu_tensor_write(d->markov_embed, 0, embedding, sizeof(embedding)) &&
        ds41_matmul(d->markov_logits, m, last->markov_w2, d->markov_embed, false) &&
        ds4_gpu_add_tensor(logits, logits, d->markov_logits, DS4_N_VOCAB) &&
        ds4_gpu_tensor_copy(d->confidence_input, 0, d->batch.x,
            (uint64_t)row * DS4_N_EMBD * 4u, (uint64_t)DS4_N_EMBD * 4u) &&
        ds4_gpu_tensor_copy(d->confidence_input, (uint64_t)DS4_N_EMBD * 4u,
            d->markov_embed, 0, rank * 4u) &&
        ds41_matmul(d->confidence, m, last->confidence_proj, d->confidence_input, false);
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = false;
    if (ok) ok = ds4_gpu_tensor_read(logits, 0, d->host_logits + (uint64_t)row * DS4_N_VOCAB, bytes) &&
        ds4_gpu_tensor_read(d->confidence, 0, confidence, sizeof(*confidence));
    ds4_gpu_tensor_free(logits);
    return ok;
}

/* Cache appends beyond the accepted frontier are invisible. Only the six
 * overwritten raw-ring slots and pooling carries need rollback; no compressed
 * cache proportional to the context is copied. */
static DS4_MAYBE_UNUSED bool ds41_dspark_verify(ds41_dspark *d, ds41_gpu_graph *g,
        const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t count) {
    if (!d || !g->valid || count < 2 || count > DS41_VERIFY_ROWS ||
        g->pos > g->ctx || count > g->ctx - g->pos || !ds41_tp_batch_enabled(g)) return false;
    /* Prepost the complete small-row exchange window, as in V4.0 verification.
     * Bulk-prefill handshakes at every attention/FFN gate are too expensive. */
    const bool tp_block = g->tp_world == 2;
    uint32_t gates = 2;
    if (tp_block && (!g->tp_batch_out || !g->tp_batch_in || !g_tp_block_ctx ||
        !ds4_tp_batch_block_begin_gates(g_tp_block_ctx, count, DS4_N_LAYER, &gates) ||
        gates != 2)) return false;
    const uint64_t kv = DS4_N_HEAD_DIM * 4u, feature = DS4_N_EMBD * 4u;
    ds41_gpu_graph *graphs[DS41_VERIFY_ROWS];
    bool ok = ds4_gpu_begin_commands();
    d->verify_start = g->pos;
    d->verify_count = 0;
    d->rewind_end = 0;
    for (uint32_t i = 0; ok && i <= DS41_VERIFY_ROWS; i++) {
        if (i >= count && i != DS41_VERIFY_ROWS) continue;
        const uint32_t slot = (g->pos + i) % DS41_DRAFT_WINDOW;
        for (uint32_t il = 0; ok && i < count && il < 40; il++)
            ok = ds4_gpu_tensor_copy(d->old_raw, (il * DS41_VERIFY_ROWS + i) * kv,
                g->window[il], slot * kv, kv);
        for (uint32_t stage = 0; ok && d->weights && i < count && stage < 3; stage++) {
            d->old_positions[stage][i] = d->positions[stage][slot];
            ok = ds4_gpu_tensor_copy(d->old_features, (stage * DS41_VERIFY_ROWS + i) * feature,
                d->target[stage], slot * feature, feature);
        }
        ds41_gpu_graph *row = &d->verified[i];
        ds4_gpu_tensor *res = row->residual, *pre = row->pre, *logits = row->logits;
        ds4_gpu_tensor *half = row->tp_logits_half;
        ds4_gpu_tensor *pool_k[4], *pool_s[4];
        memcpy(pool_k, row->verify_previous_kv, sizeof(pool_k));
        memcpy(pool_s, row->verify_previous_score, sizeof(pool_s));
        *row = *g;
        row->residual = res ? res : ds4_gpu_tensor_view(d->verify_residual, i * DS4_N_HC * feature, DS4_N_HC * feature);
        row->pre = pre ? pre : ds4_gpu_tensor_view(d->verify_pre, i * DS4_N_HC * 4u, DS4_N_HC * 4u);
        row->logits = logits ? logits : ds4_gpu_tensor_view(d->verify_logits,
            (uint64_t)i * DS4_N_VOCAB * 4u, (uint64_t)DS4_N_VOCAB * 4u);
        row->tp_logits_half = NULL;
        if (g->tp_logits_half) {
            const uint64_t bytes = (uint64_t)DS4_N_VOCAB * 2u;
            row->tp_logits_half = half ? half : ds4_gpu_tensor_view(row->logits,
                g->tp_rank * bytes, bytes);
            ok = ok && row->tp_logits_half;
        }
        row->verify_prefix = true;
        for (uint32_t owner = 0; owner < 4; owner++) {
            const uint64_t off = (i * 8u + owner * 2u) * kv;
            row->verify_previous_kv[owner] = pool_k[owner] ? pool_k[owner] :
                ds4_gpu_tensor_view(d->verify_pool, off, kv);
            row->verify_previous_score[owner] = pool_s[owner] ? pool_s[owner] :
                ds4_gpu_tensor_view(d->verify_pool, off + kv, kv);
            ok = ok && row->verify_previous_kv[owner] && row->verify_previous_score[owner];
        }
        ok = ok && row->residual && row->pre && row->logits;
        if (i < count) graphs[i] = row;
        else {
            /* Keep the block start as well: exact sampling can change mode
             * at the first token, before any verified token is consumed. */
            ok = ok && ds4_gpu_tensor_copy(row->residual, 0, g->residual, 0, DS4_N_HC * feature) &&
                ds4_gpu_tensor_copy(row->pre, 0, g->pre, 0, DS4_N_HC * 4u) &&
                ds4_gpu_tensor_copy(row->logits, 0, g->logits, 0, (uint64_t)DS4_N_VOCAB * 4u);
            for (uint32_t owner = 0; ok && owner < 4; owner++)
                ok = ds4_gpu_tensor_copy(row->verify_previous_kv[owner], 0, g->previous_kv[owner], 0, kv) &&
                    ds4_gpu_tensor_copy(row->verify_previous_score[owner], 0, g->previous_score[owner], 0, kv);
        }
    }
    if (ok) ok = ds41_graph_step_batch(graphs, tokens, (int)count, count, m, w);
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = false;
    if (tp_block && (!ok || !ds4_tp_batch_block_end(g_tp_block_ctx))) {
        ds4_tp_mark_failed(g_tp_block_ctx);
        ok = false;
    }
    if (ok) ok = ds4_gpu_tensor_read(d->verify_logits, 0, d->host_verified,
        (uint64_t)count * DS4_N_VOCAB * 4u) &&
        ds4_gpu_tensor_read(d->verify_logits, (uint64_t)DS41_VERIFY_ROWS * DS4_N_VOCAB * 4u,
            d->host_verified + (uint64_t)DS41_VERIFY_ROWS * DS4_N_VOCAB, DS4_N_VOCAB * 4u);
    if (ok) d->verify_count = count;
    else g->valid = false;
    return ok;
}

static DS4_MAYBE_UNUSED bool ds41_dspark_commit(ds41_dspark *d, ds41_gpu_graph *g,
                                               uint32_t keep, float *logits) {
    if (!d || !d->verify_count || keep > d->verify_count ||
        (g->pos != d->verify_start && g->pos != d->rewind_end) ||
        (d->rewind_end && d->verify_start + keep > d->rewind_end)) return false;
    const uint64_t kv = DS4_N_HEAD_DIM * 4u, feature = DS4_N_EMBD * 4u;
    const uint32_t row = keep ? keep - 1u : DS41_VERIFY_ROWS;
    ds41_gpu_graph *last = &d->verified[row];
    bool ok = ds4_gpu_begin_commands();
    for (uint32_t i = keep; ok && i < d->verify_count; i++) {
        const uint32_t slot = (d->verify_start + i) % DS41_DRAFT_WINDOW;
        for (uint32_t il = 0; ok && il < 40; il++)
            ok = ds4_gpu_tensor_copy(g->window[il], slot * kv,
                d->old_raw, (il * DS41_VERIFY_ROWS + i) * kv, kv);
        for (uint32_t stage = 0; ok && d->weights && stage < 3; stage++) {
            ok = ds4_gpu_tensor_copy(d->target[stage], slot * feature,
                d->old_features, (stage * DS41_VERIFY_ROWS + i) * feature, feature);
            d->positions[stage][slot] = d->old_positions[stage][i];
        }
    }
    for (uint32_t owner = 0; ok && owner < 4; owner++)
        ok = ds4_gpu_tensor_copy(g->previous_kv[owner], 0, last->verify_previous_kv[owner], 0, kv) &&
            ds4_gpu_tensor_copy(g->previous_score[owner], 0, last->verify_previous_score[owner], 0, kv);
    if (ok) ok = ds4_gpu_tensor_copy(g->residual, 0, last->residual, 0, DS4_N_HC * feature) &&
        ds4_gpu_tensor_copy(g->pre, 0, last->pre, 0, DS4_N_HC * 4u) &&
        ds4_gpu_tensor_copy(g->logits, 0, last->logits, 0, (uint64_t)DS4_N_VOCAB * 4u);
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) ok = false;
    if (ok) {
        g->pos = last->pos;
        g->history = last->history;
        memcpy(logits, d->host_verified + (uint64_t)row * DS4_N_VOCAB, DS4_N_VOCAB * 4u);
    } else g->valid = false;
    d->rewind_end = ok ? g->pos : 0;
    return ok;
}

#endif
