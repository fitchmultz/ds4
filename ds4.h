#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_ssd.h"

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*ds4_session_cancel_fn)(void *ud);

#define DS4_SESSION_SYNC_INTERRUPTED 2

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ds4_distributed_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    ds4_backend backend;
    int n_threads;
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    const char *expert_profile_path;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool inspect_only;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    bool glm_cpu_ref;            /* Phase 4a-full: GLM CPU reference forward */
    bool glm_metal_ref;          /* Phase 4c-iv: GLM Metal forward (F32 kernels) */
    bool glm_chat;               /* Phase 4e: GLM incremental chat generation    */
    ds4_distributed_options distributed;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);

/* GLM (glm-dsa) CPU reference dequant (Phase 4b). Dequantizes n elements (a
 * multiple of 256) of the given GGUF tensor type to F32. Returns true for the
 * supported K-quants (q4_k/q5_k/q6_k); false for unsupported types. Used by the
 * --glm-quant-dequant oracle test and the GLM CPU reference. */
bool ds4_dequant_glm_row(uint32_t gguf_type, const void *block_data,
                         float *out, size_t n);

/* === GLM-5.2 (glm-dsa) CPU reference forward path (Phase 4a) =============
 *
 * The Metal graph is the production path for GLM.  This CPU reference exists
 * only as a correctness oracle for the Metal port (per AGENT.md: "Keep the CPU
 * backend CPU-only and use it only as reference/debug code").  It is fully
 * additive, dispatched by arch == DS4_ARCH_GLM_DSA, and never touches the
 * DeepSeek path.  All math mirrors tests/test-vectors/glm52-ref/
 * glm52_mla_moe_ref.py line-for-line; the --glm-cpu-ref-components test pins it
 * to those numpy fixtures (1e-4 abs / 1e-5 rel). */

/* GLM-5.2 shape profile.  The synthetic fixture uses tiny dims with the same
 * STRUCTURE; the real model uses the §1 values from docs/GLM52-PORT.md. */
typedef struct {
    uint32_t hidden;           /* H                                              */
    uint32_t n_head;           /* attention heads                                */
    uint32_t q_lora;           /* q_a LoRA rank (2048 real, 8 fixture)           */
    uint32_t kv_lora;          /* kv latent rank (512 real, 6 fixture)           */
    uint32_t qk_nope;          /* per-head no-RoPE q/k dim (192 real, 3 fixture) */
    uint32_t qk_rope;          /* per-head RoPE slice dim (64 real, 2 fixture)   */
    uint32_t v_dim;            /* per-head v dim (256 real, 4 fixture)           */
    uint32_t ff_inter;         /* dense FFN intermediate (12288 real, 8 fixture) */
    uint32_t n_expert;         /* routed experts (256 real, 8 fixture)           */
    uint32_t n_expert_used;    /* top-k (8 real, 2 fixture)                      */
    float    rms_eps;          /* RMSNorm epsilon (1e-5)                         */
    float    rope_base;        /* RoPE theta (8e6)                               */
    float    moe_scale;        /* routed weight scale (2.5)                      */
} glm_cpu_shape;

/* Standard interleaved RoPE on a d-dim slice (d even).  Mirrors numpy:
 *   theta_i = 1 / base^(2i/d), ang = t*theta_i
 *   out[2i]   = x[2i]*cos(ang) - x[2i+1]*sin(ang)
 *   out[2i+1] = x[2i]*sin(ang) + x[2i+1]*cos(ang)
 * Rotates n_head independent rows of width d. */
void glm_rope_interleaved_f32(float *out, const float *x,
                              uint32_t d, uint32_t n_head,
                              float base, uint32_t t);

/* Standard RMSNorm with learned per-channel scale. */
void glm_rmsnorm_f32(float *out, const float *x, const float *weight,
                     uint32_t n, float eps);

/* F32 matvec: out[r] = sum_c W[r*cols + c] * x[c]  for r in [0,rows). */
void glm_matvec_f32(float *out, const float *W, const float *x,
                    uint32_t rows, uint32_t cols);

/* SiLU: out[i] = x[i] / (1 + exp(-x[i])). */
void glm_silu_f32(float *out, const float *x, uint32_t n);

/* Sigmoid MoE router: sigmoid(logits)+bias, top-k, normalize selected sigmoid
 * weights, scale by moe_scale.  Mirrors numpy exactly.
 *   logits[E] = gate[E][H] @ x[H]
 *   prob[i]   = sigmoid(logits[i])
 *   sel[i]    = prob[i] + bias[i]
 *   idx       = top-k of sel (descending; ties broken by lower index)
 *   w[j]      = prob[idx[j]] / sum_j(prob[idx[j]]) * moe_scale
 * Writes idx[k] and w[k].  scratch must hold >= E floats. */
void glm_moe_route_sigmoid(int *idx, float *w,
                           const float *gate, const float *x,
                           const float *bias,
                           const glm_cpu_shape *shape, float *scratch);

/* Dense SwiGLU FFN for blk.0-2: out = down @ (silu(gate @ x) * (up @ x)).
 * gate/up are [ff_inter, hidden]; down is [hidden, ff_inter]. */
void glm_swiglu_dense_f32(float *out, const float *x,
                          const float *gate, const float *up, const float *down,
                          const glm_cpu_shape *shape,
                          float *scratch_gate, float *scratch_up);

/* === Composite MLA decode attention for one token ======================
 *
 * Mirrors glm52_mla_moe_ref.py:
 *   q[h,:]  = (WqB[h*(nope+rope):, :] @ rmsnorm(WqA @ x, ones))
 *   kva     = WkvA @ x                      // [kv_lora + rope]
 *   latent  = kva[:kv_lora]; k_rope = kva[kv_lora:]
 *   kvln    = rmsnorm(latent, ones)         // latent-only norm
 *   k_nope[h,:] = WkB[h*nope:, :] @ kvln
 *   v[h,:]      = WvB[h*vdim:, :]  @ kvln
 *   q_rope, k_rope rotated by interleaved RoPE at position t
 *   scores[h,n] = (q_nope[h].K_nope[h,n] + q_rope[h].K_rope[h,n]) / sqrt(nope+rope)
 *   causal mask n<=t, softmax, sum V -> attn_out[h,:] (shape.v_dim per head)
 * K/V/R rope caches are [n_head, seq_n, <dim>] row-major; the current position
 * t row is overwritten by the freshly projected k/v before scoring.
 * out is the post-attention residual contribution: Wo @ attn_out.flatten().
 *
 * w_q_a_norm / w_kv_a_norm are the learned RMSNorm scale weights for the q_a
 * and kv_a latents ([q_lora] / [kv_lora]); pass NULL to use ones, which is the
 * synthetic-fixture behavior pinned by --glm-cpu-ref-components (21/21).  The
 * real model supplies learned F32 weights here (Phase 4a-full). */
void glm_mla_forward_token_f32(float *out,
                               const float *x,
                               const float *WqA, const float *WqB,
                               const float *WkvA, const float *WkB,
                               const float *WvB, const float *Wo,
                               const float *w_q_a_norm,
                               const float *w_kv_a_norm,
                               float *K_nope_cache, float *V_cache,
                               float *K_rope_cache,
                               uint32_t seq_n, uint32_t t,
                               const glm_cpu_shape *shape,
                               float *scratch);

/* Default GLM-5.2 (real model) shape profile from docs/GLM52-PORT.md §1. */
glm_cpu_shape glm_cpu_shape_real(void);

/* Synthetic fixture shape (tiny dims, same structure) used by the
 * --glm-cpu-ref-components oracle test. */
glm_cpu_shape glm_cpu_shape_fixture(void);

/* === Tensor binding (Phase 4a shape + binding) =========================
 *
 * Resolved pointers into the mmap'd split GGUF, mirroring the DeepSeek
 * ds4_layer_weights pattern but with GLM-5.2 tensor names/shapes.  Pointers
 * stay NULL when the corresponding tensor is absent (e.g. dense FFN on a MoE
 * layer, or MoE tensors on a dense layer).  Binding is arch-gated and only
 * resolves tensors for DS4_ARCH_GLM_DSA; it never touches the DeepSeek path. */
typedef struct {
    const void *token_embd;        /* [vocab, hidden]                       */
    const void *output_norm;       /* [hidden] F32                          */
    const void *output;            /* [vocab, hidden]                       */
    /* Per-layer attention (all 79 backbone blocks). */
    const void *attn_norm;         /* [hidden] F32                          */
    const void *attn_q_a;          /* [q_lora, hidden]                      */
    const void *attn_q_a_norm;     /* [q_lora] F32                          */
    const void *attn_q_b;          /* [n_head*(nope+rope), q_lora]           */
    const void *attn_kv_a_mqa;     /* [kv_lora+rope, hidden]                */
    const void *attn_kv_a_norm;    /* [kv_lora] F32                         */
    const void *attn_k_b;          /* [n_head*nope, kv_lora]                */
    const void *attn_v_b;          /* [n_head*v_dim, kv_lora]               */
    const void *attn_output;       /* [hidden, n_head*v_dim]                */
    /* Dense FFN (blk.0-2 only). */
    const void *ffn_gate;          /* [ff_inter, hidden]                    */
    const void *ffn_up;            /* [ff_inter, hidden]                    */
    const void *ffn_down;          /* [hidden, ff_inter]                    */
    /* MoE (blk.3-77 only). */
    const void *ffn_norm;          /* [hidden] F32                          */
    const void *exp_probs_b;       /* [n_expert] F32 bias                   */
    const void *ffn_gate_inp;      /* [n_expert, hidden] F32 router         */
    const void *ffn_gate_exps;     /* [n_expert, ff_expert_inter, hidden]   */
    const void *ffn_up_exps;       /* [n_expert, ff_expert_inter, hidden]   */
    const void *ffn_down_exps;     /* [n_expert, hidden, ff_expert_inter]   */
    const void *ffn_gate_shexp;    /* [ff_expert_inter, hidden]             */
    const void *ffn_up_shexp;      /* [ff_expert_inter, hidden]             */
    const void *ffn_down_shexp;    /* [hidden, ff_expert_inter]             */
} glm_layer_weights;

/* Bind one backbone layer's GLM tensors from the loaded split GGUF.  Returns
 * the number of tensors resolved (>0 on success).  layer_idx in [0,79).
 * Dense-layer tensors (ffn_gate/up/down) are left NULL for MoE layers and
 * vice-versa; the caller checks presence.  This is structural binding only;
 * materializing F32 from quant tensors uses ds4_dequant_glm_row. */
uint32_t glm_layer_bind(const void *engine_or_model, uint32_t layer_idx,
                        glm_layer_weights *out);

/* === Phase 4a-full: full per-token CPU reference forward (oracle) =======
 *
 * Assembles the validated Phase 4a component math + Phase 4b dequant into a
 * runnable forward over the REAL mmap'd split GGUF (via ds4_engine_glm_cpu_ref)
 * or a tiny in-memory model (ds4_glm_cpu_forward_synth).  CPU-only reference/
 * debug (AGENT.md); never the production Metal graph.  The core glm_cpu_forward
 * is file-local; these are the public entry points. */

/* Tiny synthetic self-check of the full layer-loop assembly (no model needed):
 * builds a few tiny F32 layers in memory and runs the full forward, also
 * pinning the real-tensor k_b layout reorder.  Sets *out_token (argmax),
 * *out_top_logit, *out_finite.  Returns 0 on success.  Used by the
 * --glm-cpu-forward-synth test so the assembly is provably runnable without
 * the 238 GiB split. */
int ds4_glm_cpu_forward_synth(int *out_token, float *out_top_logit,
                              bool *out_finite);

/* CLI driver: tokenize `prompt` with the loaded GLM vocab, run the full CPU
 * reference forward, and print the greedy token id + top logit + finiteness.
 * Returns 0 on success.  Slow (reference path). */
int ds4_engine_glm_cpu_ref(ds4_engine *e, const char *prompt, int n_predict);

/* Phase 4c-iv: full GLM forward on Metal, reusing the validated Phase 4c-i/ii
 * ds4_gpu_glm_* kernels.  Per-layer dequant -> GPU F32 buffers -> GPU KV cache
 * -> logits.  Correctness oracle = the CPU reference (ds4_engine_glm_cpu_ref).
 * Same tokenizer/embedding/LM-head/dequant policy as the CPU path; only the
 * leaf math runs on the validated Metal kernels.  Prints greedy token id +
 * top logit + finiteness, and optional DS4_GLM_LOGITS_OUT dump. */
int ds4_engine_glm_metal_ref(ds4_engine *e, const char *prompt, int n_predict);

/* Phase 4e: incremental GLM-5.2 multi-token generation (--glm-chat).  Applies
 * the GLM chat template ([gMASK]<sop> + user + <|assistant|> + nothink
 * <think></think>), prefills, then greedily decodes n_predict tokens with an
 * INCREMENTAL KV cache (one new K/V per step, reusing prior entries — constant
 * per-token cost, not O(n^2)).  use_metal selects the backend; both share the
 * validated per-token step so the greedy sequence is identical and matches
 * llama.cpp token-for-token.  GLM-5.2 only (rejects non-glm-dsa models). */
int ds4_engine_glm_chat(ds4_engine *e, const char *system, const char *prompt,
                        int n_predict, bool use_metal);

/* Phase 4e synth self-checks (no model on disk): incremental generation must
 * produce the SAME greedy argmax at every decode step as a fresh naive full
 * forward over the growing prefix.  ds4_glm_cpu_generate_synth pins the
 * incremental KV refactor vs the validated CPU forward; ds4_glm_metal_generate_synth
 * additionally pins Metal incremental == CPU naive (so Metal == CPU sequence).
 * *out_match receives the number of matching steps; return 0 if all n_steps match. */
int ds4_glm_cpu_generate_synth(int n_steps, int *out_match);
int ds4_glm_metal_generate_synth(int n_steps, int *out_match);

/* Phase 4c-iv: tiny synthetic Metal forward self-check (no model needed).
 * Builds the same in-memory F32 layers as ds4_glm_cpu_forward_synth and runs
 * them through ds4_engine_glm_metal_ref's Metal orchestration.  Proves the
 * end-to-end Metal assembly (layer loop, GPU KV cache, MLA, dense + MoE
 * dispatch, LM head) is runnable + finite without the 238 GiB split, and that
 * Metal argmax == CPU synth argmax.  Returns 0 on success. */
int ds4_glm_metal_forward_synth(int *out_token, float *out_top_logit, bool *out_finite);
int ds4_tokenize_model_text(const char *model_path, const char *text, int *out, int max_out);
int ds4_render_chat_prompt(const char *model_path, const char *system,
                           const char *prompt, ds4_think_mode think_mode,
                           int *out, int max_out);
int ds4_render_chat_history(const char *model_path, const char *system,
                            const char *assistant_content, bool max_effort,
                            int *out, int max_out);
size_t ds4_decode_model_text(const char *model_path, const int *ids, int n_ids,
                             char **out_text);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
bool ds4_session_is_distributed(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* Optional cooperative cancellation.  ds4_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns DS4_SESSION_SYNC_INTERRUPTED when it stops. */
void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_output_head(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif
