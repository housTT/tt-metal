# Phase 1 decode work (2026-09-10)

All switches are environment variables read at model construction; defaults are the
values validated here unless noted.

| switch | default | effect |
|---|---|---|
| `QWEN38_MOE_KERNEL` | `sparse_bank` (was `dispatch`) | `sparse_bank`: each rank folds its 128 resident experts into `[1,128,K,N]` banks (`Qwen38ResidentExperts._build_sparse_banks`) and evaluates them with `ttnn.sparse_matmul`; decode uses indexed mode over the ten selected experts (`MultichipDecoder._routed_experts_sparse_bank_indexed`), prefill the dense-scan mode with 4 row groups. Routed + shared partials are summed and reduce-scattered once. Removes dispatch, combine, bincount, cumsum, the counts all-gather and one reduce-scatter per layer. |
| `QWEN38_GDN_SPLIT_QKV` | `1` | Split the packed `[K, 10240+96]` GDN projection into `gdn_qkv` and `gdn_b_a` at setup (`_split_gdn_projection_weights`); removes a 90 us fp32 slice per GDN layer. BFP8 restored after the slice. |
| `QWEN38_GDN_RECOMMIT_TAP` | `0` | The second `gdn_qkv_b_a` matmul that re-wrote the newest conv tap is redundant with the fused decode body's copy; off. |
| `QWEN38_GDN_SOFTPLUS` | `kernel` | `composite` (log1p(exp(min(a,20)))) is 4 cheap ops but drifted the greedy sequence by token 3 on the reduced stack; kept opt-in. |
| `QWEN38_GDN_STATE_WORKSPACE` | `1` | `0` keeps GDN state in per-layer DRAM tensors (no L1 round trip); neutral on the reduced stack. |
| `QWEN38_LM_HEAD_COLUMNS_PER_RANK` | `62080` | One LM-head matmul per rank; `LMHead1D.forward` no longer concatenates a single split. Was two splits + a 401 us concat. |
| `QWEN38_FINAL_MIXER` | `dram_sharded` | Final hyper mixer: 1/4 folded into the down weight, DRAM-sharded down matmul. |
| `QWEN38_QSA_DENSE_K_CHUNK` / `QWEN38_QSA_DENSE_COMPUTE` | `128` / `default` | Paged flash-decode at head_dim=256 is numerically wrong across KV chunks with the model's fp32-accumulate compute config (first free-run token already diverged on a 201-token prompt); the op's default compute config with k_chunk 32 or 128 reproduces the selector path exactly. |
| `QWEN38_QSA_DENSE_BELOW_BUDGET` | `1` | While every active position is < 2040 the QSA layers run causal paged SDPA over the KV cache (`FusedDecoder._dense_qsa_attention`) instead of selector + gathers; the selector trace variant is captured on demand when a request crosses the budget (`Qwen38FullModel._switch_qsa_variant`, host position mirror `Qwen38BatchState.host_positions`). |
| `QWEN38_ARGMAX` | `gather` | `sharded` = per-rank argmax + gather of 4 pairs; measured slower on the reduced stack, left off. |
| `QWEN38_DECODE_RESIDUAL` | `fractured` | `replicated` keeps the full 10240-wide residual on every rank (0 collectives in the hyper mixers); correct but not faster, parked. |

Reduced-stack (layers 0,1,3) traced decode step incl. endpoints, golden tokens
`[99933, 106847, 176330, 171425, 236926]` reproduced in every row
(`tests/test_replicated_decode_residual.py::test_reduced_stack_golden_tokens`):

| configuration | step ms |
|---|---:|
| shipped build | 7.77 |
| sparse-bank MoE | 6.18 |
| + GDN split / no re-commit / softplus, single-split LM head, dense QSA | 5.51 |
| + BFP8 restored on split weights, DRAM-sharded final mixer | 5.44 |

48-layer AIME24 teacher-forcing gate (`RUN_QWEN38_ACCURACY=1`, evidence in this directory
and `../phase1_sparse_bank/`): sparse-bank only → top-1 95.96 %, top-5 100 %, top-100 100 %,
67.9 ms/token (baseline 97.98/100/100, 89.8 ms/token).

48-layer free-run (`test_full_model_aime24_autoregressive_quality`), config C = sparse-bank MoE +
GDN split/no re-commit + single-split LM head + DRAM-sharded final mixer + dense QSA below budget
(k_chunk 128, default compute): **53.7 ms/token (18.6 tok/s)**, TTFT 1.36 s, coherent output with the
baseline's 5-token matching prefix (`../phase1_cfgC/aime24_autoregressive_100_report_final.json`).
Baseline free-run: 87.4 ms/token.

48-layer teacher-forcing gate, config C: top-1 97.98 %, top-5 100 %, top-100 100 % (identical to the
shipped build's 97.98/100/100), teacher-forced decode 61.9 ms/token (baseline 89.8 ms/token);
`../phase1_cfgC/teacher_forcing_accuracy.json`.
