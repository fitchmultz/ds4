# GLM-5.2 quant dequant oracle fixtures

Each `.bin` is a block-aligned slice of a real GLM-5.2 UD-IQ2_M tensor, extracted
directly from the GGUF. Each `.oracle.txt` is the authoritative llama.cpp
`dequantize_row_*` output (one F32 per line). The original fixtures were
generated with `/tmp/gen_oracle` linking `libggml-base.dylib`; the Q2_K/Q3_K
NextN fixtures were generated with a standalone `/tmp/dump_ggml_q2q3_standalone.c`
that copies only llama.cpp's `dequantize_row_q2_K` / `dequantize_row_q3_K` code
and GGUF block structs.

**Large-slice policy:** every IQ fixture is >=1000 blocks (>=256000 elements).
The original 8-block (2048-element) fixtures hid a latent IQ2_XXS pointer-
arithmetic regression (only the first 32 elements of each block were read
correctly); the large real slices reproduce it and now pin every element
bit-exact against llama.cpp.

| type | source tensor | block bytes | blocks | slice bytes |
|---|---|---:|---:|---:|
| Q8_0    | blk.8.attn_q_b.weight       |  34 |    8 |    272 |
| Q2_K    | blk.78.ffn_gate_exps.weight |  84 |  512 |  43008 |
| Q3_K    | blk.78.ffn_down_exps.weight | 110 |  512 |  56320 |
| Q4_K    | output.weight (sanity)      | 144 |    8 |   1152 |
| Q5_K    | blk.3.attn_q_a.weight       | 176 |    8 |   1408 |
| Q6_K    | blk.0.ffn_down.weight       | 210 |    8 |   1680 |
| IQ3_XXS | blk.3.ffn_down_exps.weight  |  98 |    8 |    784 |
| IQ4_XS  | blk.8.ffn_down_exps.weight  | 136 | 1024 | 139264 |
| IQ2_S   | blk.8.ffn_gate_exps.weight  |  82 | 1024 |  83968 |
| IQ2_XXS | blk.7.ffn_gate_exps.weight  |  66 | 1024 |  67584 |

These cover every quant type the GLM-5.2 UD-IQ2_M CPU forward dequants or must
handle for blk.78 NextN/MTP: Q8_0 (attention projections), IQ2_XXS (most
routed-expert gate/up), IQ2_S (blk.8 gate/up), IQ3_XXS/IQ4_XS (routed-expert
down), Q2_K/Q3_K (blk.78 NextN gate/up/down), and the other K-quants. blk.8 is
the ONLY IQ2_S layer; blk.7 stands in for the IQ2_XXS layers.
