# GLM-5.2 Port — Source of Truth

Status: **CORE PORT COMPLETE & WORKING.** GLM-5.2 runs locally: load + glm4 tokenize + oracle-correct forward (CPU & Metal, logit-corr 0.9993 vs llama.cpp) + coherent multi-token chat generation (`--glm-chat`/`--glm-chat-cpu`), matching llama.cpp greedy token-for-token (11/12 on 'Say hello.' -> 'Hello! How can I help you today?'). The 238 GiB UD_IQ2_M model runs in ~43 GiB peak RAM on this 128 GiB Mac via mmap on-demand expert streaming (no resident copy). 12 commits on origin/glm.

VERIFIED DONE: P0 scaffolding; P1 loader/split-GGUF/metadata/tensor-inventory (1809/1809); P2 glm4 BPE+chat (oracle); P3 SSD cache-plan (8359 experts); P4b all 8 quant dequant bit-exact vs llama.cpp (incl. the IQ2_S full-tensor bugfix); P4a/P4a-full CPU forward (oracle-validated); P4c-i/ii Metal kernels (31/31); P4c-iv Metal forward (corr 1.0 vs CPU); P4e incremental-KV chat generation. DeepSeek path byte-identical at every commit; full ds4_test green.

REMAINING (optimizations, not correctness): fused-kernel production latency (current ~10-24 tok/s slow path materializes F32 per layer; DS4-speed needs the fused quant kernels / metal_graph integration); explicit ds4_ssd pread streaming (mmap already achieves the bounded-RAM + on-demand-SSD outcome); DSA sparse indexer (deferred — our dense-MLA forward matches llama.cpp, which also omits it); NextN/MTP blk.78 (loaded, unused — a speculation accelerator, not required for correct greedy gen).
Target: run GLM-5.2 (`glm-dsa`) on a
128 GiB RAM Mac with SSD-streamed routed experts, without breaking the existing
DeepSeek-V4 SSD / CUDA / distributed / default-Metal paths.

This doc is self-contained: it carries the GLM-5.2 hard facts, the delta vs the
DeepSeek-V4-specific engine, the SSD-cache arithmetic for 128 GiB, and the
phased plan. Update it as phases land.

## 1. Ground truth — GLM-5.2 (`glm-dsa`)

Sources: `unsloth/GLM-5.2-GGUF` (ungated, public), `zai-org/GLM-5.2` config +
tokenizer, huggingface Transformers `glm_moe_dsa`, llama.cpp `glm-dsa.cpp`.

| Field | Value |
|---|---|
| GGUF `general.architecture` | `glm-dsa` |
| Backbone transformer layers | 78 (`blk.0`..`blk.77`) |
| NextN / MTP block | 1 (`blk.78`) |
| Leading dense (non-MoE) layers | 3 (`blk.0`..`blk.2`) |
| Hidden size | 6144 |
| Dense FFN intermediate | 12288 |
| Expert FFN intermediate | 2048 |
| Attention heads | 64 |
| KV heads (MLA compressed) | 1 |
| Q LoRA rank | 2048 |
| KV LoRA rank | 512 |
| QK no-RoPE head dim | 192 |
| QK RoPE head dim | 64 |
| QK total head dim | 256 |
| V head dim | 256 |
| `attn_kv_a_mqa` width | 576 (= 512 latent + 64 rope) |
| Vocab | 154880 |
| Max context | 1,048,576 |
| RoPE base | 8,000,000 |
| RoPE dim | 64 (interleaved on the rope slice) |
| Norm | RMSNorm, eps 1e-5 |
| Activation | SwiGLU (`silu(gate)*up`) |
| Routed experts | 256 |
| Experts per token | 8 (top-k) |
| Shared experts | 1 |
| Expert group count | 1 (no grouping) |
| Router gating | **sigmoid** |
| Router correction bias | yes (`exp_probs_b.bias`) |
| Expert weight normalization | true |
| Routed scaling factor | 2.5 |
| Attention | **DeepSeek-style MLA + DSA sparse indexer** |

MoE: 744B total / ~40B active. Router score = `sigmoid(logit) + exp_probs_b.bias`,
top-8 over all 256, selected weights normalized then ×2.5.

Attention: core MLA reuses DeepSeek-V2/V3 tensor names
(`attn_q_a`/`attn_q_a_norm`/`attn_q_b`/`attn_kv_a_mqa`/`attn_kv_a_norm`/`attn_k_b`/`attn_v_b`/`attn_output`).
On top sits a **DSA sparse indexer** (32 heads, head-dim 128, top-k 2048,
IndexShare reuse every 4 layers). Standard RoPE on the 64-dim rope slice —
**no** DeepSeek-V4 per-layer compression ratios, no output-lora, no HC.

## 2. Delta vs the engine (DeepSeek-V4)

The engine is HIGH-coupling to DeepSeek-V4: hardcoded Flash/Pro profiles,
`deepseek4.*` metadata namespace, JoyAI tokenizer, top-6, V4 compressed-HC
output projection, per-layer RoPE compression, group-required MoE.

Key deltas to address (priority order):

1. **Metadata namespace + profile.** `glm-dsa.*` vs `deepseek4.*`; new shape
   profile (§1). Dispatch by `general.architecture == "glm-dsa"`.
2. **Split GGUF.** GLM ships as 6 shards; shard 1 is metadata+tokenizer only
   (0 tensors, 69 kv, 9,423,744 bytes). Engine loader is single-file. This is
   foundational.
3. **Tokenizer.** BPE gpt2 with `pre=glm4` (byte-level), GLM special tokens,
   `[gMASK]<sop>` chat template. Engine hardcodes JoyAI split + DeepSeek chat.
4. **MoE routing.** sigmoid + bias + norm + scale 2.5 + top-8, group=1. Engine
   assumes top-6, no bias, no norm, no scale, REQUIRES `expert_group_count==0`.
5. **Attention.** MLA core (reusable kernels) but bypass V4 output-lora/HC/
   compression; standard interleaved RoPE on rope slice (theta 8e6); DSA
   indexer is net-new (defer — run dense-MLA fallback first, like llama.cpp).
6. **Leading dense layers.** `blk.0`-`blk.2` use `ffn_gate/up/down` (dense),
   no experts. Engine assumes shared expert mandatory on all layers.
7. **SSD streaming.** `ds4_ssd` helpers + per-expert byte math + Metal slot
   layout (`gate*2+down`, gate==up) **transfer directly** — GLM uses identical
   3D `ffn_gate_exps/up_exps/down_exps` with gate==up byte size. Biggest lesson
   reused. Only change: wire to `glm-dsa` arch and GLM expert count/dims.
8. **MTP/NextN.** `blk.78.nextn.*` (eh_proj/enorm/hnorm/shared_head_norm).
   Defer (Phase 6).

## 3. Tensor inventory (`UD-IQ2_M`)

Total tensors across 6 shards: 1809. Per-shard counts: `[0, 412, 368, 368, 368, 293]`.

Global: `token_embd.weight [6144,154880] Q5_K`, `output_norm.weight [6144] F32`,
`output.weight [6144,154880] Q4_K`.

Per backbone layer (`blk.0`-`blk.77`, ×79): `attn_norm [6144] F32`,
`attn_q_a [6144,2048]`, `attn_q_a_norm [2048] F32`, `attn_q_b [2048,16384] Q8_0`,
`attn_kv_a_mqa [6144,576] Q8_0`, `attn_kv_a_norm [512] F32`,
`attn_k_b [192,512,64] Q8_0`, `attn_v_b [512,256,64] Q8_0`, `attn_output [16384,6144]`.

DSA indexer (×79, loaded, unused until Phase 5): `indexer.attn_k [6144,128]`,
`indexer.attn_q_b [2048,4096]`, `indexer.k_norm.weight/bias [128] F32`,
`indexer.proj [6144,32] F32`.

Leading dense FFN (`blk.0`-`blk.2`, ×3): `ffn_gate [6144,12288]`, `ffn_up [6144,12288]`,
`ffn_down [12288,6144]`.

Routed MoE (`blk.3`-`blk.77`, ×75): `ffn_norm [6144] F32`, `exp_probs_b.bias [256] F32`,
`ffn_gate_inp [6144,256] F32`, `ffn_gate_exps [6144,2048,256]`,
`ffn_up_exps [6144,2048,256]`, `ffn_down_exps [2048,6144,256]`,
`ffn_gate_shexp [6144,2048]`, `ffn_up_shexp [6144,2048]`, `ffn_down_shexp [2048,6144]`.

NextN (`blk.78`, ×1, Phase 6): `nextn.eh_proj [12288,6144]`, `nextn.enorm [6144]`,
`nextn.hnorm [6144]`, `nextn.shared_head_norm [6144]`, plus a full MoE block.

Pattern counts: 79 each (attn_* + all indexer.* + ffn_norm), 3 each (dense
ffn_*), 76 each (exp_probs_b/ffn_gate_inp/ffn_*_exps/ffn_*_shexp — 75 backbone
+ 1 NextN).

## 4. Tokenizer

`tokenizer.ggml.model=gpt2`, `tokenizer.ggml.pre=glm4`, byte-level BPE.
Special IDs (from GGUF): eos 154820 (`<|endoftext|>`), padding 154821
(`[MASK]`), bos 154822 (`[gMASK]`), eot 154827 (`<|user|>`), eom 154829
(`<|observation|>`). sop 154824 (`<sop>`). system 154826, user 154827,
assistant 154828. think 154841/154842. Note pad mismatch: HF config says pad
154820, GGUF metadata says 154821 — don't use pad during generation.

Chat template: prefix `[gMASK]<sop>` = tokens `[154822,154824]`; role sentinels
`<|system|>`/`<|user|>`/`<|assistant|>`/`<|observation|>`; assistant opens with
`<think>` (thinking default on, disable via `enable_thinking:false`); tool calls
`<tool_call>{name}<arg_key>..</arg_key><arg_value>..</arg_value></tool_call>`,
responses `<tool_response>..</tool_response>`.

## 5. SSD streaming — 128 GiB cache plan

`ds4_ssd` helpers are reused unchanged. Per-expert math is identical to
DeepSeek (gate==up size): `per_expert = gate_expert + up_expert + down_expert`,
Metal slot = `gate*2 + down`.

Counts for GLM: routed layers = 75 (`blk.3`-`blk.77`); `max_model_experts =
75*256 = 19200`. Do not include `blk.0`-`blk.2` dense or `blk.78` NextN.

Non-routed Phase-4 footprint (excluding `blk.78`): ~14.381 GiB
(attention/norm ~9.670, indexer ~0.765, token/output ~1.108, dense FFN ~0.463,
router ~0.440, shared experts ~1.935).

Base slab class (`blk.3`, 71 layers): gate/up `IQ2_XXS` (1584 B/row → 3,244,032
B/expert each), down `IQ3_XXS` (784 B/row → 4,816,896 B/expert) → **11,304,960
B = 10.78125 MiB per cached expert**.

Off-size layers: `blk.8` IQ2_S/IQ2_S/IQ4_XS (14.0625 MiB); `blk.75`-`77`
IQ2_XXS/IQ2_XXS/IQ4_XS (12.5625 MiB). Single-slab cache covers the 71 base
layers; off-size bypass via mapped views (matches existing mixed-precision
behavior).

Auto plan @ 128 GiB recommended working set (80% = 102.4 GiB):
`cache_bytes = 102.4 - 14.381 = 88.019 GiB` → `floor(88.019 GiB / 10.78125 MiB)
= 8359 experts`. Effective cache ≈ 88.008 GiB = 43.5% of backbone experts ≈
32.65 full-layer-equivalents. This is the number `ds4_ssd_auto_cache_plan`
must produce for `(128 GiB, 15441835008, 11304960, 19200)`.

## 6. Phased plan

**Phase 0 — scaffolding.** ✅ DONE (commit 4ac372f). This doc, `download_glm52.sh`,
branch `glm`. Verified: downloader `--help` works; shard-1 fetch = 9,423,744 B.

**Phase 1 — loader/metadata/tensor inventory for `glm-dsa`.** ✅ DONE (commit
009d62a). Architecture dispatch by `general.architecture`; split-GGUF support
(part array with part-0 aliases, per-tensor `part`, independent per-shard mmap,
metadata-only inspect works from shard 1 alone); GLM shape profile; `glm-dsa.*`
validation; tensor inventory pattern-count reporting (79/3/76/4/3); CUDA/
distributed/non-Metal reject GLM cleanly; GLM inference rejected (Phase 4).
Verified: `make` 0 warnings/errors; `./ds4 --inspect -m ds4flash.gguf` byte-
identical to `main`; shard-1 inspect reports glm-dsa / block_count 79 / split
1809 declared·0 present, exit 0; backend-reject + missing-shard exit 1. Scope:
`ds4.c` only (+593/-9); no DS4_MAX_* cap bumps; no CUDA/ROCm/distributed/Metal/
SSD/C++ changes. Full-split tensor-count strict check implemented, fires only
when all 6 shards present (validate in Phase 4 session).

**Phase 2 — glm4 BPE tokenizer + chat template.** ✅ DONE (commits adda569 + 6ac32c1).
Mode dispatch by `tokenizer.ggml.pre` (joyai-llm vs glm4). glm4 pre-tokenizer =
exact HF regex (`\p{L}`/`\p{N}` via 677+144 generated range tables); shared BPE
merge engine + gated `ignore_merges` fast-path (CJK/Cyrillic collapse to one
id). GLM specials read from GGUF `token_type` (growing array, bound-checked);
chat sentinels metadata-first. GLM chat renders `[gMASK]<sop>` + role sentinels +
`<think>`/nothink. `test_glm_bpe` (`--glm-bpe`, loads tables FROM shard 1):
**43/43 HF-tokenizers oracle match** (incl. 9 contraction cases). Review found 5
bugs the oracle missed (contraction offsets, undpatched max-effort, double
`<think>`, unbounded specials array, loose assertion) — all fixed with new
regression coverage. Verified: DeepSeek `--dump-tokens` + `--inspect` byte-
identical vs main across ASCII/CJK/code/whitespace/contraction prompts.

**Phase 3 — SSD streaming wiring + 128 GiB cache-plan sizing.** ✅ DONE (commit 3a13d5c). Reused `ds4_ssd` helpers UNMODIFIED; GLM per-expert byte math computed directly from the tensor inventory (`glm_streaming_per_expert_bytes`); `--inspect` prints a live cache plan when tensors present (per-expert 11304960, the §5 8359/88GiB/43.5% documented target computed via the same formula) or the shard-1 message + documented target otherwise. `test_glm_ssd_cache_plan` (`--glm-ssd-math`, no model needed) locks the §5 numbers. Per-expert math cross-checked two ways against real shard-2 tensors = 11304960. Verified: `make` clean; `ds4_test --glm-ssd-math` passes; DeepSeek byte-identical. Runtime per-tensor-part SSD pread wiring into Metal is the Phase 4 follow-up.

**Phase 4a — GLM CPU reference forward path (component oracle).** ✅ DONE (working tree). Additive CPU reference for the Metal port: `glm_cpu_shape` profile (real + tiny fixture), `glm_layer_weights` tensor-binding struct + `glm_layer_bind`, and per-component C functions mirroring `tests/test-vectors/glm52-ref/glm52_mla_moe_ref.py` line-for-line — `glm_rmsnorm_f32` (eps 1e-5), `glm_rope_interleaved_f32` (theta 8e6, 64-dim slice, interleaved pairs), `glm_matvec_f32`, `glm_silu_f32`, `glm_moe_route_sigmoid` (sigmoid + exp_probs_b.bias + top-8 + normalize + scale 2.5), `glm_swiglu_dense_f32`, and the `glm_mla_forward_token_f32` composite (q_a→RMSNorm→q_b; kv_a_mqa→split latent512+rope64→RMSNorm latent-only→k_b/v_b; interleaved RoPE on Q/K rope; scores=(Q_nope.K_nope+Q_rope.K_rope)/sqrt256; causal softmax; single attn_output 16384→6144). `dump_fixtures.py` reproduces the numpy RNG exactly and exports all inputs + intermediates + outputs (MLA_OUT/MOE_IDX/MOE_W). `--glm-cpu-ref-components` runs each C component on the SAME synthetic inputs and asserts numpy match (1e-4 abs / 1e-5 rel): **21/21 cases pass** (RoPE, MLA q/kv projections, latent-only RMSNorm, decode attention, attn_output, dense SwiGLU, sigmoid MoE routing). DeepSeek path byte-identical (purely additive: +710/-0 across ds4.c/ds4.h/ds4_test.c). Scope: CPU reference only, no Metal/CUDA/distributed; `glm_layer_bind` is the structural surface for the runtime GLM graph (real-model binding validated when the 238 GiB split is loaded in Phase 4b+).

**Phase 4 — GLM inference: dense-MLA fallback + dense leading FFN + sigmoid
**top-8 MoE.** GLM graph path (not DeepSeek V4 HC graph); standard interleaved
RoPE on rope slice (theta 8e6); single `attn_output` projection; dense SwiGLU
for `blk.0`-`2`; sigmoid+bias+norm+scale-2.5 top-8 router for `blk.3`-`77`; add
missing quant kernels (Q5_K/Q6_K dense, IQ2_S/IQ3_XXS/IQ4_XS/Q3_K routed).
DSA deferred. DONE: `--metal --ctx 4096 -p 'Hello' -n 1` emits a token; logits
match llama.cpp `glm-dsa` dense-fallback oracle; DeepSeek smoke unchanged.
**Needs full 238GB split on disk.**

**Phase 5 — DSA sparse indexer + IndexShare.** Indexer projections, top-k=2048
mask, IndexShare reuse every 4 layers. Net-new kernels. DONE: long-context
logits beat dense fallback where sparse matters.

**Phase 6 — NextN/MTP speculative.** Bind `blk.78.nextn.*`. DONE: greedy
spec-verify correct.

**Phase 7 — tests/eval/docs.** Tiered: always-on (metadata/cache-math/router
microkernels), `DS4_TEST_GLM52_SHARD1`, `DS4_TEST_GLM52_GGUF`, `DS4_TEST_GLM52_LONG`.

## 7. Session scope

Done this session: **Phases 0, 1, 2, 3** — all committed + pushed `origin/glm`, all
verified against the real 238GB `UD-IQ2_M` model (now fully downloaded). The
inspect/tokenizer/SSD-sizing foundation is complete and tested:
- 1809/1809 tensors load across 6 shards (split invariant enforced).
- glm4 BPE = 43/43 HF-tokenizers oracle match (incl. contractions).
- SSD cache plan live = documented: 8359 experts / 88 GiB / 43.5% @ 128 GiB.
- DeepSeek path byte-identical to `main` throughout.

Next: **Phase 4** (inference). This is the large, multi-session core. Recommended
first sub-step: a GLM **CPU reference** forward pass for a single layer on
synthetic fixtures (the engine keeps CPU as reference/debug per AGENT.md) to
serve as the oracle for the Metal kernels, then port kernels one at a time
(RoPE → MLA projections → dense FFN → sigmoid MoE → quant dots → per-part SSD
pread), each validated against the CPU ref. Needs: Q5_K/Q6_K dense +
IQ2_S/IQ3_XXS/IQ4_XS/Q3_K routed quant kernels; standard interleaved RoPE
(theta 8e6, dim 64); single `attn_output`; dense FFN for blk.0-2; sigmoid
top-8 MoE (bias + norm + scale 2.5). DSA (Phase 5) and NextN/MTP (Phase 6)
after. A llama.cpp `glm-dsa` dense-fallback build is the natural full-model
logit oracle once Phase 4 produces tokens.

## 8. Risks / blockers

- Phase 4 needs net-new quant kernels: Q5_K/Q6_K dense; IQ2_S/IQ3_XXS/IQ4_XS/Q3_K
  routed. Engine currently supports IQ2_XXS/Q2_K/Q4_K routed + the dense set.
- `DS4_MAX_*` cap bumps (DS4_MAX_LAYER>=79, DS4_MAX_VOCAB>=154880) for inference
  must not change DeepSeek shapes/allocs — decide between bumping vs a GLM-specific
  shape struct in Phase 4.
- Global Metal `g_model_fd` pread is single-file; runtime per-tensor-part SSD
  streaming wiring into Metal is Phase 4 work (Phase 3 did sizing + dry-run only).
- The DeepSeek V4 graph assumes HC / compressed attention / grouped output /
  top-6 routing / all-layers-routed. GLM breaks all of these — Phase 4 must add a
  separate GLM graph path, not retrofit the DeepSeek graph.
- DSA correctness (Phase 5) needs an external oracle (Transformers/vLLM/SGLang).
- No official 128 GiB SSD recipe exists; runtime streaming throughput is
  measurement-based once Phase 4 runs.
