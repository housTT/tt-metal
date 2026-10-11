# AutoFix — stage03 optimization candidates

Status: in progress; no terminal failure or stage acceptance.

Contract: preserve >=0.995 per-token layer PCC, precise state/cache ownership,
active top8 execution and repeated traced replay while evaluating faster legal
configurations. Scope remains optimized_decoder.py and its tests/docs.

Diagnosis: AUTODEBUG.md independently reconstructs the BF8 native4x2 CB overlap:
actual end1559808, observed live frontier1556416,3392 bytes overlap.2x2 is a
separate DEST geometry limit (16 tiles/core versus8 supported), not L1 exhaustion.

Prioritized repairs and discriminators:

1. `expert_input_dtype=bfp8`: convert the existing padded DRAM row-major input
   to BF8 tiled once; preserve256 capacity rows,8 routes, BF16 output. Source
   predicts CB end1530176; measure same4x2 and4x4 candidates at BF8/BF4 weights.
   Matrix sparse_retry_matrix.json includes these trials. Execution pending.
2. Sparse scan candidates: rename helper config->program_config to remove the
   inherited MoEConfig collision; retry all affected geometry cells. Initial
   failures remain in expert_real, with exact commands and no fabricated metrics.
3. Split sparse gate/up: source factory builds user_fused_activation=None.
   The ignored SiLU explains the large PCC collapse, not reduced-weight precision.
   Apply SiLU through the supported binary input activation, preserving up/gate
   ordering. Independent source check requested; same-fixture retest pending.
4. Indexed sparse mode: existing in-tree tests prove its API contract. Decode
   uses compact top8 UINT16 indices, no nnz, placeholder sparsity, compact sparse
   down input and direct route-slot score weighting. Geometry/accuracy/perf trials
   pending; no speedup claimed yet.

Host validation: Python compilation passes. Runtime validation and final decisions
will be appended after serialized hardware runs.

## Verified repairs

`sparse_retry/commands.json` confirms all previously Python-blocked sparse
configs now execute. Native BF8 tiled inputs fit both4x2 and4x4 as predicted,
but their added input conversion loses warmed trace latency. Fixed separate
sparse gate/up with binary-consumer SiLU passes min decode PCC0.999394 for
both BFP8/BFP4 gate weights (old failure0.33266); packed remains faster.
Independent source review is `sparse_activation_diagnosis.md`.
Indexed active-expert mode now passes trace allocation audit and repeat
identity with min decode PCC0.999396. Final qualification remains pending.
