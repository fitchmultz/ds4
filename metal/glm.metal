// GLM-5.2 (glm-dsa) standalone component kernels — Phase 4c-i.
//
// Small, flat-buffer kernels that mirror the Phase 4a CPU reference
// (ds4.c: glm_matvec_f32 / glm_rmsnorm_f32 / glm_rope_interleaved_f32 /
// the attention block of glm_mla_forward_token_f32) line-for-line on simple
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

struct ds4_metal_args_glm_rmsnorm {
    uint32_t n;     // elements to normalize over
    float    eps;   // 1e-5
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

struct ds4_metal_args_glm_rope {
    uint32_t d;       // rope slice width (even)
    uint32_t n_head;  // independent rows
    float    base;    // theta base (8e6)
    uint32_t t;       // position
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
