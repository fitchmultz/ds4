# GLM-5.2 quant dequant oracle fixtures

Each `.bin` is the first 8 blocks (2048 elements) of a real GLM-5.2 UD-IQ2_M
tensor, extracted block-aligned from the GGUF. Each `.oracle.txt` is the
authoritative llama.cpp `dequantize_row_*` output (one F32 per line, 2048 lines).

| type | source tensor | block bytes | 8-block slice bytes |
|---|---|---:|---:|
| Q5_K    | blk.3.attn_q_a.weight       | 176 | 1408 |
| Q6_K    | blk.0.ffn_down.weight       | 210 | 1680 |
| IQ3_XXS | blk.3.ffn_down_exps.weight  |  98 |  784 |
| IQ4_XS  | blk.75.ffn_down_exps.weight | 136 | 1088 |
| IQ2_S   | blk.8.ffn_gate_exps.weight  |  82 |  656 |
| Q4_K    | output.weight (sanity)      | 144 | 1152 |

Engine currently supports F32/F16/Q8_0/Q2_K/Q4_K/IQ2_XXS. These fixtures cover
the 5 quant types GLM-5.2 UD-IQ2_M needs that are MISSING (Q3_K only appears in
the deferred blk.78 NextN block, so it is excluded from this set / Phase 6).
Q4_K is included as a sanity case (engine already supports it).
