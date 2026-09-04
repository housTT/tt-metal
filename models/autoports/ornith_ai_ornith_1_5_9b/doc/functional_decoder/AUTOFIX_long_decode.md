# AutoFix: native-context decode accuracy

## Starting evidence

`AUTODEBUG_long_decode.md` inspected the original strict native-context failure:
`logs/native_decode_oracle.log` recorded real layer-3 HF/traced PCC
**0.9453528402993279** at position 262143. That log is preserved. The historical
input helper generated FP32 `randn` with seed 971, multiplied by .5, then cast
to BF16. The current calibrated helper instead uses .014285416342318058.

No `_attention_decode`, RoPE, HF reference, weight, cache-allocation, or precision
policy changes were made by this experiment. The stage coordinator independently
changed the unrelated DeltaNet gates and calibrated the input helper. The
original-scale controls below explicitly restore the historical helper formula.

## Precision and cache ledger

All runs use real layer-3 weights and real target dimensions: hidden 4096,
16 query heads, 4 KV heads, head dimension 256, context 262144, batch 1. Input and
ordinary weights are BF16; HF computations are FP32. K/V remain BF16 throughout.
Each physical cache is `[4096,4,64,256]`, using the model's allocation helper,
page-block size 64 and deterministic seed 970 shuffled INT32 table. Current
position is device INT32 262143; RoPE indices are device UINT32. No CCL is used.
The historical BF16 fixture is copied to physical pages, then the existing
`paged_update_cache` writes the current row. SDPA reads all 262144 entries;
rounded read coverage and physical allocation are exactly equal, so no
under-allocation cliff is present.

Ordinary projections use HiFi4, nonapproximate math, FP32 destination. The
unchanged baseline decode SDPA uses HiFi2, approximate math, BF16 destination,
no packer accumulation, K-chunk 64, 8x8 grid and `exp_approx_mode=False`. The
following controls alter only the named SDPA parameter(s), using the identical
query and actual post-update cache. BF16 intermediate/statistics storage remains
unchanged even with FP32 destination enabled.

## Hypothesis experiments

`probe_long_decode.py` runs two frozen input scales, six SDPA policies each.
Its host reads are named diagnostic boundaries, never part of a measured pass.
Every configuration includes an eager output and an uninstrumented captured
replay. `probe_long_decode.json` holds exact metrics, and
`logs/probe_long_decode.log` holds the command output.

| Policy | Calibrated full HF PCC | Original-scale full HF PCC | Same-cache attention relative L2, calibrated |
| --- | ---: | ---: | ---: |
| Unchanged default | .9996501206 | .9995616861 | .02528765 |
| Explicit default | .9996501206 | .9995616861 | .02528765 |
| FP32 destination only | .9996768537 | .9996095133 | .03405540 |
| Nonapproximate math only | .9996566655 | .9996708369 | .00371925 |
| K-chunk 256 only | .9996666958 | .9995589133 | .02531812 |
| K-chunk 512 only | .9996878365 | .9995600237 | .02533038 |

- **H1: default precision causes the recorded PCC failure.** Refuted as an
  explanation of the current-state failure: the unchanged default already
  passes both scales. Nonapproximate math lowers same-cache relative L2, while
  FP32 destination alone worsens it. This establishes sensitivity, not a fix
  for the historical failure. No precision change retained.
- **H2: long BF16 online accumulation requires a larger K chunk.** Refuted as
  a necessary intervention: 64, 256, 512 all pass the full-decoder gate, and the
  larger chunks do not improve the attention relative error. No chunk change
  retained.
- **H3: final-row/page mapping or trace replay corrupts data.** Controlled in
  the exact failing geometry: the mapped final row equals the actual TT update
  tensor bitwise for K and V, the preceding row remains bitwise identical to
  the fixture, and the FP32 oracle consumes the complete actual readback cache
  and TT query. Same-cache attention PCC is at least .9999958 across all 12
  rows. Eager and replay are bitwise equal in all 12 rows.
- **H4: diagnostic host reads, calibrated inputs, warm JIT, or watcher state
  mask the failure.** Controlled with the original uninstrumented test and
  exact original-scale helper, independently of the diagnostic monkeypatches.
  The default policy passes twice in each of a normal process, watcher process,
  and cold-JIT process. All 6 original-scale strict reruns report
  **.9995616861429337**. Cold-JIT telemetry reports **0/237 cache hits**. The
  calibrated original pytest also passes **.9996501206304318**.

Across the two-scale diagnostic and strict reruns there are 19 native-context
replay comparisons: 12 diagnostic policy/scale rows, 6 strict original-scale
reruns, and 1 strict calibrated pytest. All meet .995 without a decoder fix.
The watcher log has 4840 lines with attach/check/detach output and no matches for
`fatal|exception|overflow|invalid|sanitize` (case-insensitive inspection).

## Exact commands and artifacts

Run from the tt-metal root. The common environment is:

```bash
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8
export PYTHONPATH=.
export ORNITH_WEIGHTS=real
```

```bash
python_env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder/probe_long_decode.py
python_env/bin/pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_contract_extensions.py -k native_context_decode_oracle -x -v -s
python_env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder/recheck_long_decode_scale.py
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder/watcher_native_original python_env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder/recheck_long_decode_scale.py
TT_METAL_CACHE=/home/hous/dev/ornith-1.5-9b/state/tt-cache-native-control python_env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder/recheck_long_decode_scale.py
python_env/bin/pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_functional_decoder.py -k 'test_traced_decode_pcc and full_attention' -x -v -s
```

Stdout/stderr were redirected respectively to:

- `logs/probe_long_decode.log`
- `logs/native_decode_calibrated_rerun.log`
- `logs/native_decode_original_scale_rerun.log`
- `logs/native_decode_original_scale_watcher.log`
- `logs/native_decode_original_scale_cold_jit.log`
- `logs/autofix_full_attention_traced_regression.log`

Watcher artifacts are under
`watcher_native_original/generated/watcher/{watcher.log,kernel_names.txt,kernel_elf_paths.txt}`.
The separate cold cache is persistent and was not substituted for or used to
remove the original kernel cache.

The final short/batched regression passed 3 tests, batch 1/4/32 with three changed
input/position trace steps each. Nine HF/replay PCCs range **.998993–.999258**.
No hardware commands overlapped; every command closed its mesh before the next.

## Final status

**Current native-context failure not reproduced; no implementation change
retained.** Exact historical inputs, strict uninstrumented reruns, fresh JIT,
watcher, exact-cache readback, direct FP32 attention controls and batched traced
regressions all pass. These are controlled current-state results, not a claim
that a code fix explained the original .94535. Its historical cause remains
unknown, and the original failure artifact is preserved for independent stage
review. Any recurrence must trigger a fresh diagnosis with the failing output,
query and cache captured before changing source or numerical settings.

This investigation does not reduce context, weaken PCC, claim a performance
improvement, or establish another layer kind's correctness. The unchanged
production `_attention_decode` remains available for the coordinator's final
strict regression. Hardware ownership returned to the coordinator after the
last device closure at 21:05:01 UTC.
