# AutoFix: final public prefill continuation

## Final status

Fixed the observed continuation failures in the optimized model. Public
prefill now immediately copies each leading decode output to interleaved
DRAM and releases its L1 output allocation. It preserves the inherited
validation, positions, padding, cache updates, sequence lengths and borrowed
input ownership. Ordinary decode and every existing optimized block method
are unchanged. The native context remains 262144; no context gate was reduced.

The actual runtime passes 15 targeted contract/performance checks, and a
separate watcher run passes five capacity/no-host/ownership checks. The
hardware lane was released after devices closed normally. Remaining full
short-suite, native-context and stage acceptance gates belong to the parent.

## Starting evidence

- Diagnosis: [AUTOTRIAGE_final_continuation.md](AUTOTRIAGE_final_continuation.md).
- Original command and source archive:
  [final_default_short_v1.provenance.json](logs/final_default_short_v1.provenance.json).
- Failure log: [final_default_short_v1.log](logs/final_default_short_v1.log).
- Hardware: one Blackhole chip on physical P300c boards, serialized lane
  granted by the parent after its recorded list/reset/list recovery.
- Symptoms: a 67-input public concat request failed loading an NCRISC ELF
  with a 5232-byte local segment versus a 4832-byte limit; another continuation
  collided with L1 buffers at 636928 while static CBs ended at 641664. A later
  repeat hung on the host with idle devices, consistent with the LLRT binary
  cache's unfilled entry after an ELF-loading exception.

The independent verifier read the triage report, exact public orchestration,
optimized output memory policy and concat source, then tested each resource
boundary before changing runtime code. All controls ran in separate pytest
processes with `-x` and `timeout 300` so a failed ELF load could not poison a
later test in the same process.

## Hypothesis experiments

Every row below has `.log`, `.provenance.json` and `.sources.json.gz` artifacts
under `logs/`. The provenance includes the exact command, configuration,
environment, full model/test source snapshot, source hashes and exit status.
Test-only controls were removed after adjudication; their archived source
remains available with each run.

| Experiment record | Single intervention | Result and verdict |
| --- | --- | --- |
| `autofix_continuation_bounded_capacity_v1` | Cap concat groups at eight, leaving original output lifetimes/layouts | Failed with `bad optional access` when a group combined sharded and interleaved inputs. This attempted remedy did not satisfy the concat input-layout contract. |
| `autofix_continuation_bounded_capacity_v2` | Cap groups at eight and normalize sharded inputs at the final merge boundary; retain all leading outputs until then | Exact capacity 4096/split 63 passed, HF PCC 0.99871661. The final concat boundary is sufficient to address this case. |
| `autofix_continuation_bounded_nohost_v1` | Same final-merge intervention, no early spill | Reproduced the exact L1 collision, 636928 versus 641664. Bounding concat does not fix retained output pressure. |
| `autofix_continuation_drain_nohost_v1` | Move leading decode outputs to DRAM immediately; retain inherited concat | Passed the complete existing no-host guard, including start position 257 and both positive guard controls. Early spill fixes the L1 lifetime failure. |
| `autofix_continuation_drain_capacity_v1` | Same immediate spill only | Exact capacity 4096/split 63 passed. A separate Python concat tree is unnecessary for the observed ELF failure once inputs are normalized to DRAM. |
| `autofix_continuation_unbounded_dram257_v1` | Direct all-DRAM concat of 257 references to a model-width tensor | Passed. The prediction that 128 DRAM inputs must still fail was refuted. |
| `autofix_continuation_unbounded_dram1024_v1` | Direct all-DRAM concat of 1024 references | Passed. Confirmed the underlying all-interleaved API handles substantially larger public input lists. This isolated kernel specialization from tensor allocation pressure; it is not a full native-context run. |

All controls used BFLOAT4_B weights/LoFi runtime defaults where a model was
constructed, real checkpoint weights and recorded real layer-input rows. No
PCC threshold or no-host guard changed.

## Source refinement: existing concat batching

The initial triage's public input count is correct, but it is not necessarily
the instantiated reader's tensor count. In
`ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp`,
`calculate_max_tensors_per_concat` (line 240 in the inspected source) returns
47 for interleaved inputs and 56 for width/height-sharded inputs.
`concat_impl` recursively batches larger lists at line 335. Thus a public
67-input request can already be split before its reader is built; the exact
failing kernel's input count was not extracted from the rejected ELF.

Sharded inputs also carry different accessor metadata and follow different
layout transformations. Converting outputs to interleaved DRAM early both
releases scarce L1 and directs merging through the existing interleaved
batching contract. The 257/1024-input controls verify that contract. Adding a
second Python concat tree would duplicate functionality without fixing an
additional demonstrated failure, so it was not retained.

The LLRT exception-safety defect remains outside the authorized model-local
edit scope. This fix avoids triggering the rejected concat specialization;
it does not repair `get_risc_binary` for arbitrary future ELF-loading errors.

## Kept change

- [optimized_decoder.py](../../tt/optimized_decoder.py): add a public
  `prefill_forward` override with the inherited validation/orchestration and
  immediate DRAM collection of leading decode outputs. The buffer-address
  comparison avoids freeing an already-DRAM output that aliases its converted
  view. The original slice/pad ownership rules are retained.
- [test_optimized_continuation_resources.py](../../tests/test_optimized_continuation_resources.py):
  verify both layer kinds with a borrowed 127-token prefix and a borrowed
  one-token continuation, unchanged input addresses/contents, logical output
  length, DRAM output, HF output PCC and the subsequent decode's HF PCC.
- An AST comparison against the failing run's archived optimized source
  found zero changes to existing methods; `prefill_forward` is the only added
  method. No functional/fused decoder or C++ code changed.

## Runtime verification

The commands below ran from the checkout with `ORNITH_WEIGHTS=real`,
`OMP_NUM_THREADS=8`, and the recorded persistent torch/TT cache paths. The
recorder's exact command and environment are authoritative.

```bash
python_env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder/record_run.py \
  autofix_continuation_runtime_contract_v1 timeout 300 python_env/bin/python -m pytest \
  models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_decoder.py \
  -k 'unaligned_continuation_to_capacity or prefill_continuation or no_host_fallback_in_forward or test_perf_prefill or test_perf_decode_traced' \
  -x -v -s
```

Result: **15 passed**, 67 deselected, 36.72 seconds. Capacity continuation PCC
is 0.99871661. Both layer kinds pass split 63/65/128/129 and their complete
no-host guards. Measured outputs in every performance test satisfy HF PCC.

| Actual runtime measurement | Linear attention | Full attention |
| --- | ---: | ---: |
| Warm prefill, 2048 tokens | 8.12 ms | 6.40 ms |
| Prefill HF PCC | 0.99913254 | 0.99867025 |
| Warm traced decode, 32 iterations | 0.524 ms/token | 0.406 ms/token |
| Measured decode HF PCC | 0.99897083 | 0.99897343 |

These helper-test decode timings use a 128-token prefix and position128,
whereas the stage headline paired comparison uses2048/position2048. They
are not substituted for the headline number. These are observed ordinary-run
timings on a Blackhole chip on P300c boards,
not profiler/watcher timings or a claim of a new decode speedup.

```bash
TT_METAL_WATCHER=10 python_env/bin/python \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder/record_run.py \
  autofix_continuation_runtime_watcher_v1 timeout 300 python_env/bin/python -m pytest \
  models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_decoder.py \
  models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_continuation_resources.py \
  -k 'unaligned_continuation_to_capacity or no_host_fallback_in_forward or continuation_borrowed_input' \
  -x -v -s
```

Result: **5 passed**, 79 deselected, 38.77 seconds, with no watcher failure.
Profiler collection was disabled. Devices closed and the watcher thread
stopped normally before handing the hardware lane back to the parent.

Python formatting: Black with `--target-version py310` passed. No build was
needed for this Python-only change. No hardware reset or process kill was
needed during these bounded verification runs.
