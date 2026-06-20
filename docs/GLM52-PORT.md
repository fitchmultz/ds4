# GLM-5.2 Port — Source of Truth

Status: **CORE PORT COMPLETE & WORKING; FAST PATH IN PROGRESS.** GLM-5.2 runs locally: load + glm4 tokenize + oracle-correct forward (CPU & Metal, logit-corr 0.9993 vs llama.cpp) + coherent multi-token chat generation (`--glm-chat`/`--glm-chat-cpu`), matching llama.cpp greedy token-for-token (11/12 on 'Say hello.' -> 'Hello! How can I help you today?'). The 238 GiB UD_IQ2_M model runs in bounded RAM on this 128 GiB Mac via mmap/on-demand streaming plus resident fast-path caches.

VERIFIED DONE: P0 scaffolding; P1 loader/split-GGUF/metadata/tensor-inventory (1809/1809); P2 glm4 BPE+chat (oracle); P3 SSD cache-plan (8359 experts); P4b all 10 quant dequant types bit-exact vs llama.cpp, including blk.78 NextN-only Q2_K/Q3_K and the IQ2_S full-tensor bugfix; P4a/P4a-full CPU forward (oracle-validated); P4c-i/ii Metal kernels (40/40, including GLM F32 batch matmul/RMSNorm/RoPE/attention/MoE-router, layer-major MLA, dense-FFN, and F32 routed-MoE batch composites); P4c-iv Metal forward (corr 1.0 vs CPU); P4e incremental-KV chat generation; routed MoE fast path (IQ2_XXS gate/up + IQ3_XXS/IQ4_XS down + resident shared expert) with decode improving from ~27s/token to ~9–12s/token on warm runs; default-on selected-expert SSD pread staging in `DS4_GLM_FAST` (`DS4_GLM_EXPERT_PREAD=0/off` restores mmap A/B) validated on the real model; CLI GLM generation skips the unnecessary post-final-token target forward, so one-token raw completions return after prefill while preserving emitted tokens. DeepSeek path byte-identical at every commit; full ds4_test green.

REMAINING (usability/speed, not correctness): batched/persistent GLM graph or speculation to avoid one full model pass per token; smarter cache sizing/prefetch/concurrency tuning on top of the default selected-expert pread and opt-in LRU slab cache; MLA still uses the F32 materialization path on UD_IQ2_M because q_a/attn_output are Q5_K/Q6_K and tested one-off Q5/Q6/Q8 wrappers were slower; DSA sparse indexer (deferred — dense MLA matches llama.cpp); NextN/MTP blk.78 (loaded, Q2_K/Q3_K dequant covered, post-output-norm target hidden dump exposed by `DS4_GLM_H_NEXTN_OUT`, `nextn.eh_proj` CPU/Metal synth diagnostics, one-token blk.78 decoder diagnostic exposed by `DS4_GLM_NEXTN_H_OUT`/`DS4_GLM_NEXTN_LOGITS_OUT` and validated when fed by either CPU or Metal target hidden, no-model synthetic regressions `--glm-nextn-synth`/`--glm-nextn-metal-synth`, non-mutating one-draft acceptance probe exposed by `DS4_GLM_NEXTN_PROBE=1`, and an OPT-IN default-off correctness-first speculative greedy scaffold exposed by `--glm-nextn`/`--glm-nextn-draft` that is greedy-identical by construction but NOT yet a speed claim — it now uses the Metal NextN drafter by default, but still verifies one target token per emitted token, so batched target verification remains the speed work).
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
for `blk.0`-`2`; sigmoid+bias+norm+scale-2.5 top-8 router for `blk.3`-`77`.
All quant types present in the downloaded UD_IQ2_M split now have CPU dequant
coverage, including blk.78 NextN-only Q2_K/Q3_K. DSA deferred. DONE:
`--glm-metal-ref -p 'Hello' -n 1` emits the oracle token; logits match llama.cpp
`glm-dsa` dense-fallback oracle; DeepSeek smoke unchanged.
**Needs full 238GB split on disk.**

**Phase 5 — DSA sparse indexer + IndexShare.** Deferred. Dense MLA matches
llama.cpp on the current oracle prompts; sparse DSA/index sharing is a later
long-context quality/perf task.

**Phase 6 — NextN/MTP speculative.** Opt-in, default-off, correctness-first scaffold exists. `--glm-nextn` (Metal target only) + `--glm-nextn-draft N` (capped at 4, default 4) enable a top-k=1 greedy speculative decode for `--glm-raw`/`--glm-chat`: the current verified target argmax is emitted as the SGLang/EAGLE bonus token, then `blk.78` drafts up to N following tokens from `(bonus_token, post-output-norm target hidden that predicted the bonus)`, and the shared verifier (`glm_spec_decode`) emits only the contiguous accepted draft prefix, falling back to the authoritative target argmax on miss/partial. The recursive drafter now keeps a tiny draft decoder KV cache across draft steps, threads absolute RoPE positions through that local cache, carries full accepts into the next round's bonus token, and preserves verified full-accept draft prefix rows across rounds while resetting on fallback; target KV is still only mutated by verified target steps. Crucially, every emitted token is the verified target argmax and unverified drafts NEVER enter the target KV cache, so output is byte-for-byte identical to plain greedy GLM generation and there is NO target-cache rollback; a rejected draft tail is simply discarded. CPU target ignores the flag with a diagnostic. This is correctness-first, NOT a speed claim: the real-model drafter now uses the validated Metal NextN path by default (`DS4_GLM_NEXTN_DRAFT_BACKEND=cpu` is an A/B fallback), but verification still reuses the existing single-token target step, so each emitted token still costs one target forward; the genuine speedup still needs batched/prefill-style target verification (SGLang NEXTN/EAGLE V2 batched tree verify). Current SGLang source confirms the NextN layer contract in `python/sglang/srt/models/deepseek_nextn.py`: `eh_proj(concat(enorm(embed(input_ids)), hnorm(spec_info.hidden_states))) -> decoder -> shared_head.norm -> lm_head`. `python/sglang/srt/speculative/eagle_worker_v2.py` also shows the production loop shape: draft-extend fills/updates the draft KV from target hidden states and target next tokens, top-k=1 draft degenerates to a chain, target verify runs as a batch/tree, then a draft-extend step prepares the next iteration from verified target output. No-model coverage: `ds4_test --glm-spec-generate-synth` pins full-accept, partial-accept, and miss/rollback cases to the exact naive greedy sequence via a controllable mock drafter, `--glm-spec-metal-target-synth` runs those same cases through the real synthetic GLM Metal target step/verifier seam, `--glm-spec-batch-verify-synth` pins the same accept/rollback state machine when target verification is supplied as a batched after-bonus contract, `--glm-nextn-prefix-synth` pins the persistent draft-prefix commit/reset bookkeeping, while `--glm-nextn-metal-synth` now also checks a recursive 4-token Metal draft chain against the CPU chain with draft KV and asserts the synthetic decoder hidden changes when prior draft KV is present (`cache_delta`, so the fixture would catch accidentally reverting to reset-each-step draft attention). The prior diagnostics remain: `blk.78.nextn.*` loads/dequants/runs through the CPU diagnostic path; `ds4_test --glm-nextn-synth`/`--glm-nextn-metal-synth` pin the no-model control/tensor wiring; `DS4_GLM_NEXTN_PROBE=1` measures one-draft agreement during generation without changing output. The remaining speed direction is batched target verify/rollback (target tree verify), since single-token decode still costs ~9–12s/token.

**Phase 7 — tests/eval/docs.** Tiered: always-on (metadata/cache-math/router
microkernels), `DS4_TEST_GLM52_SHARD1`, `DS4_TEST_GLM52_GGUF`, `DS4_TEST_GLM52_LONG`.

## 7. Current optimization notes

The core GLM port is complete and verified. The active work is usability/speed:

- Current fastest chat path is `DS4_GLM_FAST=1 --glm-chat`: routed MoE gate/up/down
  fused plus resident shared expert. Warm decode is about **9–12s/token** on the
  128 GiB Mac; short chat prefill is still about one full pass per prompt token.
- The fused GLM MoE path now uses explicit `pread` from each split GGUF part fd
  by default inside `DS4_GLM_FAST=1`, so selected routed-expert staging avoids
  sparse mmap page faults without another env flag. `DS4_GLM_EXPERT_PREAD=0`
  (or `off`/`false`/`no`) restores the older mmap-fault staging path for A/B
  tests. The pread path is validated with the real model and keeps the same
  oracle token.
- `DS4_GLM_EXPERT_CACHE_MIB=<n>` or `DS4_GLM_EXPERT_CACHE_GIB=<n>` enables a
  bounded LRU CPU slab cache for the quantized routed-expert slabs staged by the
  fused GLM MoE gate/up/down kernels. `DS4_GLM_EXPERT_CACHE_PRESET=decode` (or
  `DS4_GLM_EXPERT_CACHE_AUTO=1`) picks a conservative RAM-derived decode cache
  capped at 8 GiB; `DS4_GLM_EXPERT_CACHE_PRESET=plan` uses the documented Phase
  3 8359-expert / ~88 GiB plan. Explicit MIB/GIB settings win over presets.
  The cache key is map identity + tensor offset + expert id + slab geometry, so
  gate/up/down tensors and split-GGUF parts stay isolated; defaults stay off.
  Validation: `DS4_GLM_FAST=1 DS4_GLM_EXPERT_CACHE_MIB=256
  --glm-metal-ref -p Hello` keeps token `154820` and reports cache
  stores/evictions (`1776` stores, `1719` evictions for this small smoke
  budget). With `DS4_GLM_EXPERT_CACHE_GIB=8`, `--glm-raw -p asdfqwer -n 2`
  emits `12345` and reports `2616/8880` slab-cache hits (29.5%), proving hot
  slabs are retained across decode steps. Further tuning is still needed for the
  production default/prefetch policy.
- `DS4_GLM_H_NEXTN_OUT=<path>` dumps the post-output-norm hidden state that feeds
  the LM head. SGLang's NextN path uses this vector as the target-model hidden
  input to `nextn.hnorm`; CPU-vs-Metal on `Hello` matches at maxabs ~3.4e-5,
  corr ~1.0.
- `DS4_GLM_NEXTN_EH_OUT=<path>` on `--glm-cpu-ref` computes the first NextN/MTP
  operation from the SGLang contract for the greedy accepted token:
  `eh_proj([enorm(token_embd), hnorm(target_hidden)])`. On `Hello` it writes a
  finite 6144-float seed vector (RMS ~0.127).
- `DS4_GLM_NEXTN_H_OUT=<path>` / `DS4_GLM_NEXTN_LOGITS_OUT=<path>` run a
  one-token CPU blk.78 decoder diagnostic after the eh_proj seed and dump the
  post-`nextn.shared_head_norm` hidden / logits. It can be driven from either
  `--glm-cpu-ref` or `--glm-metal-ref` target hidden. On `Hello`, this validates
  the real blk.78 attention + MoE tensor layout and Q2_K/Q3_K path, producing
  finite logits with diagnostic top token 2. CPU-fed vs Metal-fed diagnostic
  outputs match tightly (hidden maxabs ~8.2e-6; logits maxabs ~2.7e-5).
- `ds4_test --glm-nextn-synth` builds a tiny in-memory backbone plus one NextN
  block and runs the same `eh_proj -> decoder -> shared_head_norm -> LM-head`
  path without the 238 GiB model. `ds4_test --glm-nextn-metal-synth` runs the
  same synthetic path with Metal leaf kernels and asserts full-logit agreement
  with the CPU synthetic oracle (current maxabs ~7.5e-9). These pin Phase 6
  control/tensor wiring in the default no-model test suite.
- `DS4_GLM_NEXTN_PROBE=1` remains as an older non-mutating diagnostic probe:
  after a target token is accepted and the next target logits are known, blk.78
  drafts one token from `(accepted_token, target_hidden)` and compares it to the
  next target argmax. `DS4_GLM_NEXTN_PROBE_LOG=1` prints per-step evidence. The
  opt-in `--glm-nextn` scaffold below supersedes it for greedy-identical accept
  plumbing and CSV tracing; keep the probe for quick A/B diagnostics only.
- `--glm-nextn` + `--glm-nextn-draft N` is the opt-in, default-off,
  Metal-target-only NextN speculative greedy scaffold for `--glm-raw`/`--glm-chat`.
  The current verified target argmax is emitted as the SGLang/EAGLE bonus token;
  `blk.78` then drafts up to N following tokens (capped at 4, default 4) from
  `(bonus_token, post-output-norm target hidden that predicted the bonus)`. The
  recursive drafter keeps a bounded draft KV cache across those draft steps,
  runs draft RoPE at the EAGLE bonus token's absolute sequence position
  (`start_pos + j`) while storing into bounded local draft-cache slots, and
  preserves verified full-accept prefix rows across rounds. On a full active-
  depth accept, the already-computed next target argmax is carried into the next
  round as the next EAGLE bonus token instead of being emitted as a fallback;
  any true fallback resets the draft prefix because the fallback token's draft
  row was not produced. Target KV is still only written by verified target
  steps. The shared verifier
  emits only the contiguous accepted draft prefix and falls back to the
  authoritative target argmax on miss/partial. Every emitted token is the verified target
  argmax and drafts never enter the target KV cache, so output is byte-for-byte
  identical to plain greedy GLM and no target-cache rollback is needed. It is
  correctness-first, NOT a speed claim: the real-model drafter now defaults to
  the validated Metal NextN path (`DS4_GLM_NEXTN_DRAFT_BACKEND=cpu` keeps a CPU
  A/B fallback), but verification still reuses the single-token target step
  (one target forward per emitted token), so the speedup still needs batched
  target verification. `DS4_GLM_NEXTN_TRACE_OUT=<path>` writes a per-round CSV
  trace for acceptance-rate, semantic target-step accounting (`target_steps`),
  and physical verifier-launch accounting (`target_batches`) analysis.
  After correcting the scaffold to SGLang's
  bonus-token semantics (the current verified target token is always emitted,
  and drafts are checked against following target tokens), real `asdfqwer -n 2`
  emits identical ids `108714 100461` with `draft0=100461` accepted as a full
  one-token draft hit. A second raw smoke, `The answer is -n 4`, produces
  identical plain-vs-NextN ids `9829 13 758 2097` (` yes. In fact`) with
  one accepted draft (`13`) before fallback to `758` and final no-draft token
  `2097`. A small post-fix real sweep with `--glm-nextn-draft 4 -n 6` shows
  shallow but nonzero acceptance: `The answer is` accepted 2/6 active drafts,
  `Once upon a time` accepted 1/7, and `2+2=` accepted 2/6. All accepted real
  sweeps were partial rounds (mostly first-draft hits); this means target
  batched/tree verify can reduce overhead, but deeper speedup likely also needs
  better acceptance from fuller draft-worker state/prefix handling. Current
  SGLang `EagleDraftInputV2Mixin.prepare_for_v2_draft` seeds draft positions
  from `batch.seq_lens` and writes draft KV into request-token pools; ds4's
  drafter now has recursive local draft KV, absolute RoPE positions, full-accept
  carry to the next bonus token, and a conservative persistent full-accept
  prefix, but not SGLang's fuller request-token-pool state seeded from verified
  target hidden states. The single-step target verifier is now isolated in
  `glm_spec_verify_after_bonus_single`, giving the future Metal verifier
  microbatch a concrete contract to replace without changing accept/rollback
  semantics. `ds4_test --glm-spec-generate-synth` pins full/partial/miss cases
  to the exact naive greedy sequence, `--glm-spec-batch-verify-synth` pins the
  future batched-verifier contract to the same sequence,
  `--glm-nextn-prefix-synth` pins draft-prefix commit/reset bookkeeping, and
  `--glm-spec-trace-synth` pins the CSV trace shape, including the final
  no-draft row plus `target_steps`/`target_batches` accounting. `target_batches`
  is the physical verifier-launch counter reported by the verifier contract: the
  current single-step verifier counts one batch per target step, while the
  batched-verifier contract counts one batch per verify call.
- `--glm-raw` / `--glm-raw-cpu` bypass the GLM chat template and raw-tokenize
  `-p/--prompt`. This is a usability/benchmark mode, not chat: a one-token raw
  prompt avoids the 13-token chat-template prefill while batched prefill is
  pending. The CLI generation loop also skips the unnecessary target forward
  after the final requested token, so `-n 1` returns immediately after prefill
  plus text emission. `DS4_GLM_RAW_PROMPT_IDS_OUT=<path>` and
  `DS4_GLM_RAW_GEN_IDS_OUT=<path>` write raw-mode prompt/generated token ids for
  byte-independent oracle and plain-vs-NextN validation; current real-model
  smoke has plain and `--glm-nextn --glm-nextn-draft 1` both emitting ids
  `108714 100461` (`12345`) for `asdfqwer -n 2`.
- A critical probe showed the older “fused Q8 MLA” path is not active for the
  downloaded UD_IQ2_M quant: `attn_q_a` and `attn_output` are Q5_K (Q6_K on
  blk.8), so the all-Q8 eligibility is false for every backbone layer.
- Tested and rejected: resident reordered `attn_k_b` (correct, no speedup),
  naive Q5_K/Q6_K Metal row-dot for `q_a/out` (correct, slower), llama-style
  Q5_K/Q6_K decode kernels through the current wrapper (correct, slower), and
  per-projection Q8-only MLA fusion (slower). Do not re-chase these without a
  new benchmark reason.
- Next meaningful bar-movers: GLM-specific persistent/batched graph (especially
  batched prefill; the small GLM F32 batch matmul/RMSNorm/RoPE/attention Metal
  primitives plus layer-major MLA, dense-FFN, MoE-router, and F32 routed-MoE batch composites are now tested), NextN/MTP speculative decode,
  and controlled hot-expert SSD cache/prefetch.
  LM head is only ~0.5–0.6s/token and is not the next target.
  A measured top-1-only verifier readback experiment was discarded: on
  `asdfqwer -n 2` the full-logit path reported lm-head ~0.462s while the
  top-1-only path reported ~0.474s, so full-vocab readback is not the blocker.

## 8. Risks / blockers

- NextN/MTP has a correctness-first opt-in scaffold with Metal drafting and
  traceable greedy verification. Current recorded raw smokes now show real draft
  hits after the SGLang bonus-token semantics fix, but speed is still limited by
  one target forward per emitted token. Real speed still needs batched target
  verification and higher acceptance from the fuller SGLang-style draft-worker
  state; do not present the scaffold alone as a speed win.
- Batched/persistent GLM graph work remains the largest speed risk: prefill still
  runs one full per-token forward, and the NextN verifier still uses one target
  forward per emitted token.
- Runtime SSD streaming is validated through default-on selected-expert pread
  inside `DS4_GLM_FAST` and the opt-in routed-expert slab cache, but cache
  sizing/prefetch/concurrency policy still needs more measurement.
- DSA sparse-index correctness remains deferred because dense MLA matches the
  llama.cpp oracle; revisit only with a long-context quality/perf oracle.
- No official 128 GiB SSD recipe exists; throughput/prefetch policy remains
  measurement-based on this Mac.
