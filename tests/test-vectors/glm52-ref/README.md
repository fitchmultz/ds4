# GLM-5.2 reference (numpy) for Phase 4 CPU components

glm52_mla_moe_ref.py: numpy reference for GLM MLA (q_a/q_b/kv_a_mqa/kv_a_norm/
k_b/v_b projections, interleaved RoPE on the 64-dim rope slice theta=8e6, dense
causal MLA attention, single attn_output) and sigmoid MoE routing (sigmoid +
bias, top-8, normalize selected, scale 2.5) on a TINY synthetic config
(H=16, nhead=4, q_lora=8, kv_lora=6, nope=3, rope=2, v=4). Same STRUCTURE as
the real model (H=6144, 64 heads, q_lora=2048, kv_lora=512, nope=192, rope=64,
v=256) — validates algorithm correctness, not real weights.

glm52_mla_moe_ref.out: deterministic reference outputs (MLA_OUT, MOE_IDX, MOE_W)
the C CPU reference components must reproduce on the same synthetic inputs.

Use this to build per-component CPU reference unit tests (RoPE norm-preservation,
MLA projection shapes/values, MoE top-k selection + weights) before porting to
Metal.
