// GLM-5.2 (glm-dsa) standalone component kernels — Phase 4c-i + 4c-ii.
//
// Small, flat-buffer kernels that mirror the Phase 4a CPU reference
// (ds4.c: glm_matvec_f32 / glm_rmsnorm_f32 / glm_rope_interleaved_f32 /
// the attention block of glm_mla_forward_token_f32, plus 4c-ii:
// glm_moe_route_sigmoid post-logits math and the SwiGLU activation gate)
// line-for-line on simple
// contiguous F32 arrays.  They exist ONLY to validate the Metal path against
// the CPU oracle on the tests/test-vectors/glm52-ref fixtures; the production
// GLM graph wiring (Phase 4c-iv) will reuse the engine's strided dense/norm
// matmul kernels.  These never enter the DeepSeek graph: they are dispatched
// solely from the standalone ds4_gpu_glm_* wrappers, which the test exercises
// directly.
//
// Each kernel uses one thread per work item (one thread per output row, per
// (head,pair), or per head) with a sequential inner loop.  This deliberately
// matches the CPU reference's F32 accumulation ORDER so Metal and CPU agree to
// ~1e-6, well inside the 1e-4 abs / 1e-5 rel oracle tolerance.  Performance is
// irrelevant here; the production matmul/attention kernels live in dense.metal
// and flash_attn.metal.

struct ds4_metal_args_glm_matvec {
    uint32_t rows;   // output rows
    uint32_t cols;   // input / reduction dim
};

struct ds4_metal_args_glm_matmul {
    uint32_t rows;   // output rows per token
    uint32_t cols;   // input / reduction dim
    uint32_t n_tok;  // token batch
};

// out[r] = sum_c W[r*cols + c] * x[c].  Mirrors glm_matvec_f32.
kernel void kernel_glm_matvec_f32(
        constant ds4_metal_args_glm_matvec & args,
        device const float * W,
        device const float * x,
        device       float * out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.rows) return;
    device const float * wr = W + (uint64_t)gid * args.cols;
    float acc = 0.0f;
    for (uint32_t c = 0; c < args.cols; c++) acc += wr[c] * x[c];
    out[gid] = acc;
}

#define DS4_GLM_QK_K 256

struct ds4_metal_args_glm_qk_matvec {
    uint32_t rows;
    uint32_t cols;
    uint64_t row_bytes;
};

struct ds4_metal_block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[DS4_GLM_QK_K / 8];
    uchar qs[DS4_GLM_QK_K / 2];
};

struct ds4_metal_block_q6_K {
    uchar ql[DS4_GLM_QK_K / 2];
    uchar qh[DS4_GLM_QK_K / 4];
    char  scales[DS4_GLM_QK_K / 16];
    half d;
};

kernel void kernel_glm_matvec_q5_k_f32(
        constant ds4_metal_args_glm_qk_matvec & args,
        device const char  * src0,
        device const float * yy,
        device       float * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 2;
    constexpr short nr0 = 1;
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const int nb = args.cols / DS4_GLM_QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    if (first_row >= (int)args.rows) return;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;
    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uchar hm1 = 1u << (2 * iq);
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    float sumf = 0.0f;
    float yl[16];
    float yh[16];
    ushort sc16[4];
    thread const uchar * sc8 = (thread const uchar *)sc16;
    device const float * y1 = yy + ix * DS4_GLM_QK_K + y_offset;
    device const ds4_metal_block_q5_K * x = (device const ds4_metal_block_q5_K *)(src0 + (uint64_t)first_row * args.row_bytes);

    for (int i = ix; i < nb; i += 4) {
        device const uchar * q1 = x[i].qs + q_offset;
        device const uchar * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const ushort * a = (device const ushort *)x[i].scales + iq;
        device const float * y2 = y1 + 128;
        float4 sumy = float4(0.0f);
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l +  0]; sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l +  0]; sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
        }

        device const uchar * q2 = q1 + 64;
        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = float4(0.0f);
        float4 acc2 = float4(0.0f);
        for (short l = 0; l < 8; ++l) {
            const uchar h = qh[l];
            acc1[0] += yl[l + 0] * (q1[l] & 0x0F);
            acc1[1] += yl[l + 8] * (q1[l] & 0xF0);
            acc1[2] += yh[l + 0] * (q2[l] & 0x0F);
            acc1[3] += yh[l + 8] * (q2[l] & 0xF0);
            acc2[0] += h & hm1 ? yl[l + 0] : 0.0f;
            acc2[1] += h & hm2 ? yl[l + 8] : 0.0f;
            acc2[2] += h & hm3 ? yh[l + 0] : 0.0f;
            acc2[3] += h & hm4 ? yh[l + 8] : 0.0f;
        }

        sumf += (float)dh[0] * (sc8[0] * (acc1[0]      + 16.0f * acc2[0]) +
                                sc8[1] * (acc1[1]/16.0f + 16.0f * acc2[1]) +
                                sc8[4] * (acc1[2]      + 16.0f * acc2[2]) +
                                sc8[5] * (acc1[3]/16.0f + 16.0f * acc2[3])) -
                (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        y1 += 4 * DS4_GLM_QK_K;
    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0) dst[first_row] = tot;
}

kernel void kernel_glm_matvec_q6_k_f32(
        constant ds4_metal_args_glm_qk_matvec & args,
        device const char  * src0,
        device const float * yy,
        device       float * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 2;
    constexpr short nr0 = 2;
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const int nb = args.cols / DS4_GLM_QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    if (first_row >= (int)args.rows) return;

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;
    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    float sumf[nr0] = { 0.0f, 0.0f };
    float yl[16];
    device const ds4_metal_block_q6_K * x0 = (device const ds4_metal_block_q6_K *)(src0 + (uint64_t)first_row * args.row_bytes);

    for (int i = ix; i < nb; i += 2) {
        device const float * y = yy + i * DS4_GLM_QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }
        for (short row = 0; row < nr0 && first_row + row < (int)args.rows; ++row) {
            device const ds4_metal_block_q6_K * x = (device const ds4_metal_block_q6_K *)((device const char *)x0 + (uint64_t)row * args.row_bytes);
            device const uchar * q1 = x[i].ql + q_offset_l;
            device const uchar * q2 = q1 + 32;
            device const uchar * qh = x[i].qh + q_offset_h;
            device const char  * sc = x[i].scales + is;
            const float d = (float)x[i].d;
            float4 sums = float4(0.0f);
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((char)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((char)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((char)((q1[l] >>  4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((char)((q2[l] >>  4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += d * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    for (short row = 0; row < nr0 && first_row + row < (int)args.rows; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) dst[first_row + row] = tot;
    }
}

// out[t*rows + r] = dot(W[r], x[t]).  Same accumulation order as matvec.
// This is the small GLM primitive needed by future layer-major verifier and
// prefill microbatches; it deliberately stays F32/simple like the component
// kernels instead of touching the DeepSeek graph.
kernel void kernel_glm_matmul_f32(
        constant ds4_metal_args_glm_matmul & args,
        device const float * W,
        device const float * x,
        device       float * out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint32_t r = gid.x;
    const uint32_t t = gid.y;
    if (r >= args.rows || t >= args.n_tok) return;
    device const float * wr = W + (uint64_t)r * args.cols;
    device const float * xt = x + (uint64_t)t * args.cols;
    float acc = 0.0f;
    for (uint32_t c = 0; c < args.cols; c++) acc += wr[c] * xt[c];
    out[(uint64_t)t * args.rows + r] = acc;
}

struct ds4_metal_args_glm_add_batch {
    uint32_t n;      // elements per token row
    uint32_t n_tok;  // token batch
};

// Elementwise residual add over [n_tok,n].  One thread per scalar; this tiny
// glue primitive lets layer-major verifier/prefill composites keep residual
// math on the GPU instead of round-tripping through CPU buffers.
kernel void kernel_glm_add_batch_f32(
        constant ds4_metal_args_glm_add_batch & args,
        device const float * a,
        device const float * b,
        device       float * out,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t total = (uint64_t)args.n_tok * args.n;
    if ((uint64_t)gid >= total) return;
    out[gid] = a[gid] + b[gid];
}

struct ds4_metal_args_glm_argmax_batch {
    uint32_t n_vocab; // row width
    uint32_t n_tok;   // row count
};

// Row-wise top-1 over [n_tok,n_vocab] logits.  One thread scans one row;
// lower-index tie break matches sample_argmax / CPU greedy verification.
kernel void kernel_glm_argmax_batch_f32(
        constant ds4_metal_args_glm_argmax_batch & args,
        device const float * logits,
        device       int   * out_idx,
        device       float * out_val,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_tok || args.n_vocab == 0) return;
    device const float * row = logits + (uint64_t)gid * args.n_vocab;
    float best = row[0];
    uint32_t best_i = 0;
    for (uint32_t i = 1; i < args.n_vocab; i++) {
        const float v = row[i];
        if (v > best) {
            best = v;
            best_i = i;
        }
    }
    out_idx[gid] = (int)best_i;
    out_val[gid] = best;
}

struct ds4_metal_args_glm_rmsnorm {
    uint32_t n;     // elements to normalize over
    float    eps;   // 1e-5
};

struct ds4_metal_args_glm_rmsnorm_batch {
    uint32_t n;      // elements per token row
    uint32_t n_tok;  // token batch
    float    eps;    // 1e-5
};

// out[i] = x[i] * (1/sqrt(mean(x^2)+eps)) * w[i], one row.
// Latent-only RMSNorm is just this kernel with n = kv_lora (not kv_lora+rope):
// the caller passes the latent slice and its width, so no separate masked
// variant is needed.  Mirrors glm_rmsnorm_f32 (Metal has no double, so the
// sum-of-squares accumulates in float; for n<=2048 this is well inside tol).
kernel void kernel_glm_rmsnorm_f32(
        constant ds4_metal_args_glm_rmsnorm & args,
        device const float * x,
        device const float * w,
        device       float * out) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < args.n; i++) ss += x[i] * x[i];
    const float scale = 1.0f / sqrt(ss / (float)args.n + args.eps);
    for (uint32_t i = 0; i < args.n; i++) out[i] = x[i] * scale * w[i];
}

// Batched RMSNorm: one sequential row per thread, same row-local accumulation
// order as kernel_glm_rmsnorm_f32. This is the paired primitive to
// kernel_glm_matmul_f32 for future GLM layer-major verifier/prefill batches.
kernel void kernel_glm_rmsnorm_batch_f32(
        constant ds4_metal_args_glm_rmsnorm_batch & args,
        device const float * x,
        device const float * w,
        device       float * out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_tok) return;
    device const float * xr = x + (uint64_t)gid * args.n;
    device       float * orow = out + (uint64_t)gid * args.n;
    float ss = 0.0f;
    for (uint32_t i = 0; i < args.n; i++) ss += xr[i] * xr[i];
    const float scale = 1.0f / sqrt(ss / (float)args.n + args.eps);
    for (uint32_t i = 0; i < args.n; i++) orow[i] = xr[i] * scale * w[i];
}

struct ds4_metal_args_glm_rope {
    uint32_t d;       // rope slice width (even)
    uint32_t n_head;  // independent rows
    float    base;    // theta base (8e6)
    uint32_t t;       // position
};

struct ds4_metal_args_glm_rope_batch {
    uint32_t d;       // rope slice width (even)
    uint32_t n_head;  // independent rows per token
    float    base;    // theta base (8e6)
    uint32_t pos0;    // first absolute position; row t uses pos0+t
    uint32_t n_tok;   // token batch
};

// Standard interleaved RoPE on n_head rows of width d.  Mirrors
// glm_rope_interleaved_f32: pairs (x[2i], x[2i+1]) rotate by
// theta_i = 1/base^(2i/d), ang = t*theta_i.  One thread per (head, pair).
kernel void kernel_glm_rope_interleaved_f32(
        constant ds4_metal_args_glm_rope & args,
        device const float * x,
        device       float * out,
        uint gid [[thread_position_in_grid]]) {
    const uint32_t npair  = args.d / 2u;
    const uint32_t total  = args.n_head * npair;
    if (gid >= total) return;
    const uint32_t h = gid / npair;
    const uint32_t i = gid - h * npair;
    const float theta = 1.0f / pow(args.base, (float)(2u * i) / (float)args.d);
    const float ang   = (float)args.t * theta;
    const float c = cos(ang);
    const float s = sin(ang);
    device const float * xh = x   + (uint64_t)h * args.d + 2u * i;
    device       float * oh = out + (uint64_t)h * args.d + 2u * i;
    const float a = xh[0];
    const float b = xh[1];
    oh[0] = a * c - b * s;
    oh[1] = a * s + b * c;
}

// Batched contiguous-position RoPE. Input/output layout is
// [n_tok, n_head, d]; row tok rotates at absolute position pos0+tok.
kernel void kernel_glm_rope_interleaved_batch_f32(
        constant ds4_metal_args_glm_rope_batch & args,
        device const float * x,
        device       float * out,
        uint gid [[thread_position_in_grid]]) {
    const uint32_t npair = args.d / 2u;
    const uint32_t per_tok = args.n_head * npair;
    const uint32_t total = args.n_tok * per_tok;
    if (gid >= total) return;
    const uint32_t tok = gid / per_tok;
    const uint32_t rem = gid - tok * per_tok;
    const uint32_t h = rem / npair;
    const uint32_t i = rem - h * npair;
    const float theta = 1.0f / pow(args.base, (float)(2u * i) / (float)args.d);
    const float ang = (float)(args.pos0 + tok) * theta;
    const float c = cos(ang);
    const float s = sin(ang);
    const uint64_t off = ((uint64_t)tok * args.n_head + h) * args.d + 2u * i;
    const float a = x[off];
    const float b = x[off + 1u];
    out[off] = a * c - b * s;
    out[off + 1u] = a * s + b * c;
}

struct ds4_metal_args_glm_attn {
    uint32_t nh;       // attention heads
    uint32_t nope;     // per-head no-RoPE q/k dim
    uint32_t rope;     // per-head RoPE slice dim
    uint32_t vd;       // per-head v dim
    uint32_t qhd;      // nope + rope (per-head q stride)
    uint32_t seq_n;    // cache seq stride
    uint32_t t;        // current position; score tokens n in [0, t]
    float    kq_scale; // 1/sqrt(qhd)
};

struct ds4_metal_args_glm_attn_batch {
    uint32_t nh;       // attention heads
    uint32_t nope;     // per-head no-RoPE q/k dim
    uint32_t rope;     // per-head RoPE slice dim
    uint32_t vd;       // per-head v dim
    uint32_t qhd;      // nope + rope (per-head q stride)
    uint32_t seq_n;    // cache seq stride
    uint32_t pos0;     // first query position; row tok attends [0,pos0+tok]
    uint32_t n_tok;    // token batch
    float    kq_scale; // 1/sqrt(qhd)
};

// Score one cached token n for head h: (q_nope . k_nope + q_rope . k_rope).
static inline float glm_attn_score_n(
        device const float * qn, device const float * qr,
        device const float * kn_base, device const float * kr_base,
        uint32_t nope, uint32_t rope, uint32_t seq_n, uint32_t n) {
    device const float * kn = kn_base + (uint64_t)n * nope;
    device const float * kr = kr_base + (uint64_t)n * rope;
    float s = 0.0f;
    for (uint32_t d = 0; d < nope; d++) s += qn[d] * kn[d];
    for (uint32_t d = 0; d < rope; d++) s += qr[d] * kr[d];
    return s;
}

// MLA decode attention for one query token.  One thread per head, sequential
// over cached tokens, mirroring glm_mla_forward_token_f32's per-head loop:
//   scores = (q_nope.K_nope + q_rope.K_rope) * kq_scale, n in [0, t]
//   softmax(scores) (max-subtract), sum V -> attn_out[h, vd]
// The score is recomputed per pass instead of stored, so no dynamic-size
// local array is needed; exp() is deterministic so the result matches the CPU
// oracle's stored-score path exactly.
kernel void kernel_glm_attn_decode_f32(
        constant ds4_metal_args_glm_attn & args,
        device const float * Q,            // [nh, qhd]: [nope | rope] per head
        device const float * K_nope_cache, // [nh, seq_n, nope]
        device const float * K_rope_cache, // [nh, seq_n, rope]
        device const float * V_cache,      // [nh, seq_n, vd]
        device       float * attn_out,     // [nh, vd]
        uint h [[thread_position_in_grid]]) {
    if (h >= args.nh) return;
    const uint32_t nt = args.t + 1u;   // tokens scored
    device const float * qn      = Q             + (uint64_t)h * args.qhd;
    device const float * qr      = qn           + args.nope;
    device const float * kn_base = K_nope_cache  + (uint64_t)h * args.seq_n * args.nope;
    device const float * kr_base = K_rope_cache  + (uint64_t)h * args.seq_n * args.rope;
    device const float * v_base  = V_cache       + (uint64_t)h * args.seq_n * args.vd;
    device       float * oh      = attn_out      + (uint64_t)h * args.vd;

    float maxs = -1e30f;
    for (uint32_t n = 0; n < nt; n++) {
        const float sc = glm_attn_score_n(qn, qr, kn_base, kr_base,
                                          args.nope, args.rope, args.seq_n, n)
                         * args.kq_scale;
        if (sc > maxs) maxs = sc;
    }
    float denom = 0.0f;
    for (uint32_t n = 0; n < nt; n++) {
        const float sc = glm_attn_score_n(qn, qr, kn_base, kr_base,
                                          args.nope, args.rope, args.seq_n, n)
                         * args.kq_scale;
        denom += exp(sc - maxs);
    }
    for (uint32_t d = 0; d < args.vd; d++) oh[d] = 0.0f;
    for (uint32_t n = 0; n < nt; n++) {
        const float sc = glm_attn_score_n(qn, qr, kn_base, kr_base,
                                          args.nope, args.rope, args.seq_n, n)
                         * args.kq_scale;
        const float w = exp(sc - maxs) / denom;
        device const float * vn = v_base + (uint64_t)n * args.vd;
        for (uint32_t d = 0; d < args.vd; d++) oh[d] += w * vn[d];
    }
}

// Batched decode attention. K/V caches are already populated for all queried
// positions; Q/out are [n_tok, nh, ...]. One thread handles one (token, head)
// and mirrors kernel_glm_attn_decode_f32 for absolute position pos0+token.
kernel void kernel_glm_attn_decode_batch_f32(
        constant ds4_metal_args_glm_attn_batch & args,
        device const float * Q,
        device const float * K_nope_cache,
        device const float * K_rope_cache,
        device const float * V_cache,
        device       float * attn_out,
        uint gid [[thread_position_in_grid]]) {
    const uint32_t total = args.n_tok * args.nh;
    if (gid >= total) return;
    const uint32_t tok = gid / args.nh;
    const uint32_t h = gid - tok * args.nh;
    const uint32_t nt = args.pos0 + tok + 1u;
    device const float * qn = Q + ((uint64_t)tok * args.nh + h) * args.qhd;
    device const float * qr = qn + args.nope;
    device const float * kn_base = K_nope_cache + (uint64_t)h * args.seq_n * args.nope;
    device const float * kr_base = K_rope_cache + (uint64_t)h * args.seq_n * args.rope;
    device const float * v_base = V_cache + (uint64_t)h * args.seq_n * args.vd;
    device       float * oh = attn_out + ((uint64_t)tok * args.nh + h) * args.vd;

    float maxs = -1e30f;
    for (uint32_t n = 0; n < nt; n++) {
        const float sc = glm_attn_score_n(qn, qr, kn_base, kr_base,
                                          args.nope, args.rope, args.seq_n, n)
                         * args.kq_scale;
        if (sc > maxs) maxs = sc;
    }
    float denom = 0.0f;
    for (uint32_t n = 0; n < nt; n++) {
        const float sc = glm_attn_score_n(qn, qr, kn_base, kr_base,
                                          args.nope, args.rope, args.seq_n, n)
                         * args.kq_scale;
        denom += exp(sc - maxs);
    }
    for (uint32_t d = 0; d < args.vd; d++) oh[d] = 0.0f;
    for (uint32_t n = 0; n < nt; n++) {
        const float sc = glm_attn_score_n(qn, qr, kn_base, kr_base,
                                          args.nope, args.rope, args.seq_n, n)
                         * args.kq_scale;
        const float w = exp(sc - maxs) / denom;
        device const float * vn = v_base + (uint64_t)n * args.vd;
        for (uint32_t d = 0; d < args.vd; d++) oh[d] += w * vn[d];
    }
}

// =======================================================================
// Phase 4c-ii: sigmoid MoE router + dense/shared-expert SwiGLU FFN.
// The router takes PRECOMPUTED logits (gate@x is done with
// kernel_glm_matvec_f32 above); the FFN activation gate is one elementwise
// kernel and its gate/up/down projections likewise reuse the matvec kernel.
// Same single-thread-F32, mirror-the-CPU-order discipline as 4c-i.
// =======================================================================

struct ds4_metal_args_glm_moe {
    uint32_t n_expert;   // E (256 real, 8 fixture)
    uint32_t top_k;      // K (8 real, 2 fixture); bounded by GLM_MOE_MAX_K
    uint32_t n_tok;      // token rows for batch router; ignored by single route
    float    scale;      // moe_scale (2.5)
};

#define GLM_MOE_MAX_EXPERT 1024   /* >= n_expert; real model is 256, CPU uses 1024 */
#define GLM_MOE_MAX_K      8      /* >= top_k (real 8, fixture 2) */

// Sigmoid MoE router on precomputed logits.  Mirrors the post-logits math of
// glm_moe_route_sigmoid exactly:
//   prob[i] = sigmoid(logits[i])     (stable: x>=0 -> 1/(1+exp(-x)))
//   sel[i]  = prob[i] + bias[i]
//   idx     = stable top-K by sel (strict >, lower index wins ties)
//   w[k]    = prob[idx[k]] / max(sum prob[idx], 6.1e-5) * scale
// One thread: E=256 is tiny; insertion sort + reduce is O(E*K).  The thread-
// local prob array covers the CPU's sel[1024] ceiling; top-K slots are fixed.
kernel void kernel_glm_moe_route_f32(
        constant ds4_metal_args_glm_moe & args,
        device const float * logits,   // [E]
        device const float * bias,     // [E]
        device int   * out_idx,        // [K]
        device float * out_w) {        // [K]
    const uint32_t E = args.n_expert;
    const uint32_t K = args.top_k;
    thread float prob[GLM_MOE_MAX_EXPERT];
    for (uint32_t i = 0; i < E; i++) {
        const float li = logits[i];
        prob[i] = (li >= 0.0f) ? 1.0f / (1.0f + exp(-li))
                               : exp(li) / (1.0f + exp(li));
    }
    /* Insertion sort into a descending-by-sel top-K with stable lower-index
     * tie-break: identical control flow to the CPU loop (strict >, so equal
     * sel never displaces an earlier/lower index). */
    int idx[GLM_MOE_MAX_K];
    for (uint32_t k = 0; k < K; k++) idx[k] = -1;
    for (uint32_t i = 0; i < E; i++) {
        const float sel_i = prob[i] + bias[i];
        for (uint32_t j = 0; j < K; j++) {
            if (idx[j] < 0 ||
                sel_i > (prob[(uint32_t)idx[j]] + bias[(uint32_t)idx[j]])) {
                for (uint32_t m = K - 1u; m > j; m--) idx[m] = idx[m - 1u];
                idx[j] = (int)i;
                break;
            }
        }
    }
    float sum = 0.0f;
    for (uint32_t k = 0; k < K; k++) sum += prob[(uint32_t)idx[k]];
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    const float inv = args.scale / sum;
    for (uint32_t k = 0; k < K; k++) {
        out_idx[k] = idx[k];
        out_w[k] = prob[(uint32_t)idx[k]] * inv;
    }
}

// Batched router: one sequential CPU-order route per token row. Logits layout
// is [n_tok,E], outputs are [n_tok,K].
kernel void kernel_glm_moe_route_batch_f32(
        constant ds4_metal_args_glm_moe & args,
        device const float * logits,
        device const float * bias,
        device int   * out_idx,
        device float * out_w,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n_tok) return;
    const uint32_t E = args.n_expert;
    const uint32_t K = args.top_k;
    device const float * row = logits + (uint64_t)gid * E;
    device int * idx_out = out_idx + (uint64_t)gid * K;
    device float * w_out = out_w + (uint64_t)gid * K;
    thread float prob[GLM_MOE_MAX_EXPERT];
    for (uint32_t i = 0; i < E; i++) {
        const float li = row[i];
        prob[i] = (li >= 0.0f) ? 1.0f / (1.0f + exp(-li))
                               : exp(li) / (1.0f + exp(li));
    }
    int idx[GLM_MOE_MAX_K];
    for (uint32_t k = 0; k < K; k++) idx[k] = -1;
    for (uint32_t i = 0; i < E; i++) {
        const float sel_i = prob[i] + bias[i];
        for (uint32_t j = 0; j < K; j++) {
            if (idx[j] < 0 ||
                sel_i > (prob[(uint32_t)idx[j]] + bias[(uint32_t)idx[j]])) {
                for (uint32_t m = K - 1u; m > j; m--) idx[m] = idx[m - 1u];
                idx[j] = (int)i;
                break;
            }
        }
    }
    float sum = 0.0f;
    for (uint32_t k = 0; k < K; k++) sum += prob[(uint32_t)idx[k]];
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    const float inv = args.scale / sum;
    for (uint32_t k = 0; k < K; k++) {
        idx_out[k] = idx[k];
        w_out[k] = prob[(uint32_t)idx[k]] * inv;
    }
}

// Elementwise dense/shared-expert SwiGLU: out[i] = silu(gate_x[i]) * up_x[i].
// Stable sigmoid mirrors glm_silu_f32; the gate/up/down projections reuse
// kernel_glm_matvec_f32.  One thread per element.
kernel void kernel_glm_swiglu_f32(
        device const float * gate_x,
        device const float * up_x,
        device       float * out,
        uint gid [[thread_position_in_grid]]) {
    const float g = gate_x[gid];
    const float u = up_x[gid];
    const float sig = (g >= 0.0f) ? 1.0f / (1.0f + exp(-g))
                                  : exp(g) / (1.0f + exp(g));
    out[gid] = g * sig * u;
}

// =======================================================================
// Phase 4c-fast-down: GLM routed-MoE DOWN projection, fused on the GPU.
//
// moe.metal #undefs QK_K at its end, so (re)define it locally for the block
// math below (all GLM-5.2 routed quants use 256-element blocks).
// =======================================================================
#ifndef QK_K
#define QK_K 256
#endif

// =======================================================================
// Replaces the F32 fallback that dequanted each selected down slab to F32 on
// the CPU (~50 MB/expert) and uploaded it for a per-expert F32 matvec.  The
// K selected experts' QUANTIZED down slabs are staged into one hot shared GPU
// buffer (as Step 4 did for gate/up); these kernels dequant ON the GPU and
// accumulate the K route-weighted matvecs into a single ffn_out[hidden] in one
// dispatch:
//
//     ffn_out[d] = sum_{k=0..K-1} sum_{i=0..inter-1}
//                      dequant(W_down[k, d, i]) * mid[k, i]
//
// W_down native layout is [inter, hidden, expert] (inter contiguous): within a
// staged expert slab the rows are hidden-major, each row `inter` elements =
// (inter/QK_K) quant blocks laid out block-contiguous.  mid[k] is the route-
// WEIGHTED SwiGLU mid from the fused gate/up path, so the down dot is
// accumulated UNWEIGHTED across k.  Behind DS4_GLM_FAST=1.
//
// Decomposition mirrors kernel_mul_mv_iq2_xxs_f32_impl: one simdgroup (32
// lanes) per OUTPUT ROW d; lanes stride over the row's (inter/32) 32-element
// sub-blocks (each lane handles inter/32/32 sub-blocks) and over the K experts.
// The lane-local dot is simd_sum-reduced; lane 0 writes the row.  Dequant math
// mirrors ds4_dequant_iq3_xxs / ds4_dequant_iq4_xs (ds4.c) element-for-element
// (same db/dl, same grid/sign/nibble order, same per-element weight fold) so the
// only divergence from the F32 oracle is the parallel reduction ORDER -- well
// inside the 1e-4 abs / 1e-5 rel tolerance used for the fused gate/up path.
// =======================================================================

struct ds4_metal_args_glm_moe_down {
    uint32_t hidden;        // H: output rows (6144)
    uint32_t inter;         // ei: input cols per expert (2048, multiple of QK_K)
    uint32_t K;             // selected experts (8)
    uint32_t block_bytes;   // bytes per quant block (98 IQ3_XXS / 136 IQ4_XS)
    uint64_t row_bytes;     // bytes per down weight row = (inter/QK_K)*block_bytes
    uint64_t expert_stride; // bytes per expert slab = hidden * row_bytes
};

// IQ3_XXS down experts (the common case: every MoE layer except blk.8 + blk.75..77).
// Block (98 B): half d | qs[64] (8 grid idx/sub-block) | scales[32] (4 B/sub-block).
// The 4-byte scales_and_signs packs a 4-bit scale (>>28) and four 7-bit sign
// indices; db = d*(0.5 + scale)*0.5; grid lookups via iq3xxs_grid, signs via
// ksigns_iq2xs (both shared with the IQ2_XXS path, defined in moe.metal).
kernel void kernel_glm_moe_down_iq3xxs_f32(
        constant ds4_metal_args_glm_moe_down & args [[buffer(0)]],
        device const char  * src0   [[buffer(1)]],   // staged down weights [K*expert_stride]
        device const float * mid    [[buffer(2)]],   // [K, inter] F32 (route-weighted)
        device       float * dst    [[buffer(3)]],   // [hidden] F32
        threadgroup  char  * shmem  [[threadgroup(0)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint32_t H = args.hidden, ei = args.inter, K = args.K;
    if (tgpig >= H) return;
    const uint32_t d = (uint32_t)tgpig;
    const uint32_t nb32 = (ei / QK_K) * (QK_K / 32);   // 32-elem sub-blocks per row

    // Stage iq3xxs_grid (256 uint32) + ksigns_iq2xs (128 uint8) into threadgroup
    // so the data-dependent grid/sign lookups hit fast shared memory (same
    // pattern as kernel_mul_mv_iq2_xxs_f32_impl).
    threadgroup uint32_t * sgrid  = (threadgroup uint32_t *)shmem;
    threadgroup uint8_t  * ssigns = (threadgroup uint8_t *)(sgrid + 256);
    for (uint32_t i = tiisg; i < 256; i += 32) sgrid[i]  = ds4_metal_iq3xxs_grid[i];
    for (uint32_t i = tiisg; i < 128; i += 32) ssigns[i] = ds4_metal_ksigns_iq2xs[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf = 0.0f;
    const uint32_t ix = tiisg;
    for (uint32_t k = 0; k < K; k++) {
        device const char  * row_base = src0 + (uint64_t)k * args.expert_stride
                                        + (uint64_t)d * args.row_bytes;
        device const float * y = mid + (uint64_t)k * ei + 32u * ix;
        for (uint32_t ib32 = ix; ib32 < nb32; ib32 += 32) {
            float yl[32];
            for (uint32_t i = 0; i < 32; ++i) yl[i] = y[i];
            const uint32_t ibl = ib32 / (QK_K / 32);   // block index in row
            const uint32_t ib  = ib32 % (QK_K / 32);   // sub-block within block
            device const char * blk = row_base + (uint64_t)ibl * args.block_bytes;
            const float d_block = (float)((device const half *)blk)[0];
            // qs (grid indices) at offset 2; scales_and_signs at offset 2+64.
            device const uint8_t * qs = (device const uint8_t *)(blk + 2) + ib * 8;
            device const uint8_t * sc = (device const uint8_t *)(blk + 2 + 64) + ib * 4;
            const uint32_t aux32 = (uint32_t)sc[0] | ((uint32_t)sc[1] << 8)
                                 | ((uint32_t)sc[2] << 16) | ((uint32_t)sc[3] << 24);
            const float db = d_block * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            float sum = 0.0f;
            for (uint32_t l = 0; l < 4; ++l) {
                const threadgroup uint8_t * g1 = (const threadgroup uint8_t *)(sgrid + qs[2*l]);
                const threadgroup uint8_t * g2 = (const threadgroup uint8_t *)(sgrid + qs[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> (7*l)) & 127];
                for (uint32_t j = 0; j < 4; ++j) {
                    // Fold db per element (matches the F32 oracle's dequant-then-matvec
                    // order: w = (db*grid)*sign, then += w * mid).
                    const float w1 = db * (float)g1[j] * (signs & ds4_metal_kmask_iq2xs[j]   ? -1.f : 1.f);
                    const float w2 = db * (float)g2[j] * (signs & ds4_metal_kmask_iq2xs[j+4] ? -1.f : 1.f);
                    sum += w1 * yl[8*l + j];
                    sum += w2 * yl[8*l + 4 + j];
                }
            }
            sumf += sum;
            y += 32u * 32u;
        }
    }

    const float total = simd_sum(sumf);
    if (ix == 0) dst[d] = total;
}

// IQ4_XS down experts (blk.8 + blk.75..77).  Block (136 B): half d | uint16
// scales_h | scales_l[4] | qs[128].  Per 32-elem sub-block ib: 16 qs bytes ->
// 32 values via kvalues_iq4nl[16] (low/high nibble); ls packs 4 bits from
// scales_l and 2 bits from scales_h; dl = d*(ls-32).
kernel void kernel_glm_moe_down_iq4xs_f32(
        constant ds4_metal_args_glm_moe_down & args [[buffer(0)]],
        device const char  * src0   [[buffer(1)]],   // staged down weights [K*expert_stride]
        device const float * mid    [[buffer(2)]],   // [K, inter] F32 (route-weighted)
        device       float * dst    [[buffer(3)]],   // [hidden] F32
        uint  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint32_t H = args.hidden, ei = args.inter, K = args.K;
    if (tgpig >= H) return;
    const uint32_t d = (uint32_t)tgpig;
    const uint32_t nb32 = (ei / QK_K) * (QK_K / 32);

    float sumf = 0.0f;
    const uint32_t ix = tiisg;
    for (uint32_t k = 0; k < K; k++) {
        device const char  * row_base = src0 + (uint64_t)k * args.expert_stride
                                        + (uint64_t)d * args.row_bytes;
        device const float * y = mid + (uint64_t)k * ei + 32u * ix;
        for (uint32_t ib32 = ix; ib32 < nb32; ib32 += 32) {
            float yl[32];
            for (uint32_t i = 0; i < 32; ++i) yl[i] = y[i];
            const uint32_t ibl = ib32 / (QK_K / 32);
            const uint32_t ib  = ib32 % (QK_K / 32);
            device const char * blk = row_base + (uint64_t)ibl * args.block_bytes;
            const float d_block = (float)((device const half *)blk)[0];
            // scales_h (uint16) at offset 2; scales_l[4] at offset 4; qs[128] at offset 8.
            const uint16_t scales_h = (uint16_t)((uint8_t)blk[2] | ((uint8_t)blk[3] << 8));
            device const uint8_t * scales_l = (device const uint8_t *)(blk + 4);
            device const uint8_t * qs = (device const uint8_t *)(blk + 8) + ib * 16;
            const uint32_t ls = ((uint32_t)(scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                              | (((uint32_t)(scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d_block * (float)((int32_t)ls - 32);
            float sum = 0.0f;
            for (uint32_t j = 0; j < 16; ++j) {
                const uint8_t qb = qs[j];
                const float w_lo = dl * (float)ds4_metal_kvalues_iq4nl[qb & 0xf];
                const float w_hi = dl * (float)ds4_metal_kvalues_iq4nl[qb >> 4];
                sum += w_lo * yl[j];
                sum += w_hi * yl[j + 16];
            }
            sumf += sum;
            y += 32u * 32u;
        }
    }

    const float total = simd_sum(sumf);
    if (ix == 0) dst[d] = total;
}
