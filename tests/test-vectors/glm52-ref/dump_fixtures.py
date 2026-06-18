#!/usr/bin/env python3
# Dump GLM-5.2 Phase 4a CPU-reference fixtures.
#
# Reproduces the EXACT RNG sequence and math of glm52_mla_moe_ref.py (do not
# edit that file; this is the fixture exporter) and additionally emits:
#   - every component input  (x, weights, biases, fake KV cache)
#   - every component intermediate (rmsnorm, matvec, rope, attention out)
#   - the composite oracle outputs (MLA_OUT, MOE_IDX, MOE_W)
#   - a small dense SwiGLU FFN case (gate/up/down) the ref script does not cover
#
# Output format: one value per line, plain ASCII.  Matrices are row-major flat.
# The C --glm-cpu-ref-components test parses these with strtod and compares.
import sys
import os
import numpy as np

# Tiny synthetic config -- same STRUCTURE as the real GLM-5.2 model.
# Real: H=6144, nhead=64, q_lora=2048, kv_lora=512, nope=192, rope=64, v=256.
H, nhead = 16, 4
q_lora, kv_lora, nope, rope, vdim = 8, 6, 3, 2, 4
assert nope + rope == 5

# Dense SwiGLU FFN synthetic dims (the ref script does not cover dense FFN).
ff_inter = 8

# MoE synthetic dims (must match the ref script: E=8, topk=2, scale=2.5).
E, topk, scale = 8, 2, 2.5


def rmsnorm(x, w, eps=1e-5):
    return x * w / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rope_interleaved(x, base, t):
    # Exact GLM interleaved: pairs (x0,x1),(x2,x3)... rotate by theta_i.
    d = x.shape[-1]
    half = d // 2
    theta = 1.0 / (base ** ((np.arange(half) * 2) / d))
    ang = t * theta
    out = x.copy()
    out[..., 0::2] = x[..., 0::2] * np.cos(ang) - x[..., 1::2] * np.sin(ang)
    out[..., 1::2] = x[..., 0::2] * np.sin(ang) + x[..., 1::2] * np.cos(ang)
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def dump(path, arr):
    arr = np.asarray(arr).reshape(-1)
    with open(path, "w") as f:
        for v in arr:
            f.write(f"{float(v):.9g}\n")


def dump_int(path, arr):
    arr = np.asarray(arr).reshape(-1)
    with open(path, "w") as f:
        for v in arr:
            f.write(f"{int(v)}\n")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Reproduce the ref script's RNG draw order EXACTLY (seed=0, default_rng).
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)

    x = rng.standard_normal(H).astype(np.float32)
    WqA = rng.standard_normal((q_lora, H)).astype(np.float32) * 0.1
    WqB = rng.standard_normal((nhead * (nope + rope), q_lora)).astype(np.float32) * 0.1
    WkvA = rng.standard_normal((kv_lora + rope, H)).astype(np.float32) * 0.1
    WkB = rng.standard_normal((nhead * nope, kv_lora)).astype(np.float32) * 0.1
    WvB = rng.standard_normal((nhead * vdim, kv_lora)).astype(np.float32) * 0.1
    Wo = rng.standard_normal((H, nhead * vdim)).astype(np.float32) * 0.1

    wqa_norm_w = np.ones(q_lora, np.float32)
    kvln_norm_w = np.ones(kv_lora, np.float32)

    # MLA math, mirrored line-for-line from glm52_mla_moe_ref.py.
    wqa = rmsnorm(WqA @ x, wqa_norm_w)
    q = (WqB @ wqa).reshape(nhead, nope + rope)
    kva = WkvA @ x
    latent, k_rope_raw = kva[:kv_lora], kva[kv_lora:]
    kvln = rmsnorm(latent, kvln_norm_w)
    k_nope = (WkB @ kvln).reshape(nhead, nope)
    v = (WvB @ kvln).reshape(nhead, vdim)

    t = 3
    q_nope, q_rope = q[:, :nope], q[:, nope:]
    q_rope_r = rope_interleaved(q_rope, 8e6, t)
    k_rope_r = rope_interleaved(np.broadcast_to(k_rope_raw, (nhead, rope)).copy(), 8e6, t)

    seq_n = t + 1
    K_nope_cache = rng.standard_normal((nhead, seq_n, nope)).astype(np.float32)
    V_cache = rng.standard_normal((nhead, seq_n, vdim)).astype(np.float32)
    K_rope_cache = rng.standard_normal((nhead, seq_n, rope)).astype(np.float32)
    K_rope_cache[:, t] = k_rope_r
    K_nope_cache[:, t] = k_nope
    V_cache[:, t] = v

    scores = (np.einsum('hd,hnd->hn', q_nope, K_nope_cache)
              + np.einsum('hd,hnd->hn', q_rope_r, K_rope_cache))
    scores = scores / np.sqrt(nope + rope)
    mask = np.where(np.arange(seq_n) <= t, 0.0, -1e9)
    scores = scores + mask
    scores = scores - np.max(scores, axis=1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(1, keepdims=True)
    attn_out = np.einsum('hn,hnd->hd', attn, V_cache)   # [nhead, vdim]
    mla_out = Wo @ attn_out.reshape(-1)

    # MoE math, mirrored line-for-line.
    gate = rng.standard_normal((E, H)).astype(np.float32) * 0.1
    logits = gate @ x
    prob = 1.0 / (1.0 + np.exp(-logits))
    bias = rng.standard_normal(E).astype(np.float32) * 0.1
    sel = prob + bias
    idx = np.argsort(-sel)[:topk]
    w = prob[idx]
    w = w / w.sum() * scale

    # ------------------------------------------------------------------
    # Dense SwiGLU FFN case (ref script does not cover this; we add it).
    # ------------------------------------------------------------------
    ffn_gate = rng.standard_normal((ff_inter, H)).astype(np.float32) * 0.1
    ffn_up = rng.standard_normal((ff_inter, H)).astype(np.float32) * 0.1
    ffn_down = rng.standard_normal((H, ff_inter)).astype(np.float32) * 0.1
    gate_h = ffn_gate @ x
    up_h = ffn_up @ x
    act = silu(gate_h) * up_h
    ffn_out = ffn_down @ act

    # ------------------------------------------------------------------
    # Emit fixtures.
    # ------------------------------------------------------------------
    dump(f"{out_dir}/dims.txt", [
        H, nhead, q_lora, kv_lora, nope, rope, vdim, ff_inter, E, topk, t, seq_n,
    ])

    # MLA inputs.
    dump(f"{out_dir}/mla_x.txt", x)
    dump(f"{out_dir}/mla_WqA.txt", WqA)
    dump(f"{out_dir}/mla_WqB.txt", WqB)
    dump(f"{out_dir}/mla_WkvA.txt", WkvA)
    dump(f"{out_dir}/mla_WkB.txt", WkB)
    dump(f"{out_dir}/mla_WvB.txt", WvB)
    dump(f"{out_dir}/mla_Wo.txt", Wo)
    dump(f"{out_dir}/mla_wqa_norm_w.txt", wqa_norm_w)
    dump(f"{out_dir}/mla_kvln_norm_w.txt", kvln_norm_w)
    dump(f"{out_dir}/mla_K_nope_cache.txt", K_nope_cache)
    dump(f"{out_dir}/mla_V_cache.txt", V_cache)
    dump(f"{out_dir}/mla_K_rope_cache.txt", K_rope_cache)

    # MLA intermediates (per-component sub-validation).
    dump(f"{out_dir}/mla_q_a_proj.txt", WqA @ x)      # pre-norm (renamed: mla_wqa collides with mla_WqA on case-insensitive FS)
    dump(f"{out_dir}/mla_q_a_normed.txt", wqa)
    dump(f"{out_dir}/mla_q.txt", q)                   # [nhead, nope+rope]
    dump(f"{out_dir}/mla_kva.txt", kva)               # [kv_lora+rope]
    dump(f"{out_dir}/mla_latent.txt", latent)
    dump(f"{out_dir}/mla_k_rope_raw.txt", k_rope_raw)
    dump(f"{out_dir}/mla_kvln.txt", kvln)
    dump(f"{out_dir}/mla_k_nope.txt", k_nope)         # [nhead, nope]
    dump(f"{out_dir}/mla_v.txt", v)                   # [nhead, vdim]
    dump(f"{out_dir}/mla_q_rope_r.txt", q_rope_r)
    dump(f"{out_dir}/mla_k_rope_r.txt", k_rope_r)
    dump(f"{out_dir}/mla_attn_out.txt", attn_out)
    dump(f"{out_dir}/mla_out.txt", mla_out)

    # MoE inputs + intermediates + outputs.
    dump(f"{out_dir}/moe_gate.txt", gate)
    dump(f"{out_dir}/moe_bias.txt", bias)
    dump(f"{out_dir}/moe_logits.txt", logits)
    dump(f"{out_dir}/moe_prob.txt", prob)
    dump(f"{out_dir}/moe_sel.txt", sel)
    dump_int(f"{out_dir}/moe_idx.txt", idx)
    dump(f"{out_dir}/moe_w.txt", w)

    # Dense SwiGLU inputs + intermediates + output.
    dump(f"{out_dir}/ffn_gate.txt", ffn_gate)
    dump(f"{out_dir}/ffn_up.txt", ffn_up)
    dump(f"{out_dir}/ffn_down.txt", ffn_down)
    dump(f"{out_dir}/ffn_gate_h.txt", gate_h)
    dump(f"{out_dir}/ffn_up_h.txt", up_h)
    dump(f"{out_dir}/ffn_act.txt", act)
    dump(f"{out_dir}/ffn_out.txt", ffn_out)

    # Parity check vs the canonical ref outputs (regenerate .out live).
    print("MLA_OUT", __import__("json").dumps([round(float(z), 6) for z in mla_out]))
    print("MOE_IDX", list(map(int, idx)))
    print("MOE_W", __import__("json").dumps([round(float(z), 6) for z in w]))


if __name__ == "__main__":
    main()
