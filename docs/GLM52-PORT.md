# GLM-5.2 Port — Source of Truth

Status: **Phase 0-1, 3 complete** (committed + pushed `origin/glm`; 238GB download running in background). Phase 2 (tokenizer) and Phases 4-7 remain. Target: run GLM-5.2 (`glm-dsa`) on a
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

**Phase 2 — glm4 BPE tokenizer + chat template.** Tokenizer mode from
`tokenizer.ggml.pre` (joyai vs glm4); GLM special tokens; `[gMASK]<sop>` chat
rendering. DeepSeek tokenization unchanged. DONE: `--dump-tokens` matches a HF
tokenizer oracle for ascii/CJK/chat/tool samples.

**Phase 3 — SSD streaming wiring + 128 GiB cache-plan sizing.** ✅ DONE (commit 3a13d5c). Reused `ds4_ssd` helpers UNMODIFIED; GLM per-expert byte math computed directly from the tensor inventory (`glm_streaming_per_expert_bytes`); `--inspect` prints a live cache plan when tensors present (per-expert 11304960, the §5 8359/88GiB/43.5% documented target computed via the same formula) or the shard-1 message + documented target otherwise. `test_glm_ssd_cache_plan` (`--glm-ssd-math`, no model needed) locks the §5 numbers. Per-expert math cross-checked two ways against real shard-2 tensors = 11304960. Verified: `make` clean; `ds4_test --glm-ssd-math` passes; DeepSeek byte-identical. Runtime per-tensor-part SSD pread wiring into Metal is the Phase 4 follow-up.

**Phase 4 — GLM inference: dense-MLA fallback + dense leading FFN + sigmoid
top-8 MoE.** GLM graph path (not DeepSeek V4 HC graph); standard interleaved
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

This session = **Phase 0 + Phase 1** (verifiable with shard-1 alone), plus
Phase 2-3 if time. Phases 4-7 require the full 238GB `UD-IQ2_M` split on disk
for per-kernel / full-model validation — separate sessions.

## 8. Risks / blockers

- Split GGUF is mandatory and foundational (single-file loader today).
- `UD-IQ2_M` uses quant types current kernels lack (Q5_K/Q6_K dense;
  IQ2_S/IQ3_XXS/IQ4_XS/Q3_K routed) — Phase 4 net-new kernels.
- `DS4_MAX_*` cap bumps must not change DeepSeek shapes/allocs.
- Global Metal `g_model_fd` pread is single-file — Phase 3 must make it
  per-tensor-part.
- DSA correctness needs an external oracle (Transformers/vLLM/SGLang).
- No official 128 GiB SSD recipe exists — cache sizing is measurement-based.
