# AutoFix: long-output replicas and native-capacity readback

The original failures did not reproduce after the root's recorded device
recovery. **No production repair was justified or retained.** The complete
original long suite and three capacity repetitions pass. Scoped worker-watcher verification also passes all three selected cases. This is recovery with controlled retesting,
not a claim that the original runtime anomaly has a proven source-level cause.

## Starting evidence

- `AUTODEBUG_long_replica.md` and `logs/final_long.log.gz`: linear 8001-token output
  failed exact residual replica equality after two native-context checks.
- `AUTOTRIAGE_capacity.md`, `logs/final_capacity.log.gz`, and
  `triage/capacity_host_pc.json`: device queues idle while a host helper polled
  buffer-read completion. Root captured two device snapshots before terminating
  only PID 450445 with SIGTERM (recorded return -15).
- Root then completed `reset_after_capacity`, `list_after_capacity`, and
  `mesh_after_capacity`, all exit 0; four Blackhole chips on two P300c boards
  were visible and the 1x4 ring opened/closed. Root explicitly handed this agent
  the serialized hardware lane. No additional reset was needed during these
  passing controls.
- Production SHA256 before and after these experiments remains
  `7b63bbb22ee64f76025d8363af0c60248296925609fc199f620007d12d8c7c0c`.
  Optimized baseline and C++ were not edited.

## Hypothesis experiments

| Hypothesis/control | Result | Verdict |
| --- | --- | --- |
| Replica assertion hides shared NaNs or finite rank differences; fresh 8001-token control without forward instrumentation | All four complete outputs finite and bitwise identical; small slices at 0/2048/4096/6144/7969 also identical | No isolated corruption reproduced |
| Physical chunks are correct but logical-tail concat changes data; retain device clones without forward host fences | Four physical chunks finite and replicated exactly; host physical concat trimmed to 8001 tokens equals final device concat on every rank | Concat corruption not reproduced; no speculative assembly fix |
| The preceding native checks cause the failure; exact original eight-long-test order | All 8 pass in 297.41 s | Original order alone does not reproduce failure after recovery |
| Original capacity read stalls with its reservations | Phase-marked original path passes in 8.78 s; each prefill/slice/read returns and eager/trace complete | No repeat stall; no aligned-envelope intervention used |
| Phase markers hide a capacity failure | Restored capacity test byte-for-byte from original run archive; two fresh-process repetitions pass | Three capacity successes total; marker-only healing unsupported |

The optional aligned-envelope capacity experiment was prepared but **not run**:
the original path passed first. Its temporary wrappers and untested alternative
were removed. The final capacity test is identical to the original failing
source. The production unaligned concat/slice path remains unchanged.

The long-suite measurements are:

| Layer kind | HF 8001-token prefill PCC | HF 8001-token decode PCC | Native 262144 chunk 2048 vs 1024 tail PCC | Native 262143 eager/trace PCC |
| --- | --- | --- | --- | --- |
| Linear attention | 0.999093 | 0.999200 | 0.999972 | 1.00000000 |
| Full attention | 0.998631 | 0.998408 | 0.999954 | 1.00000000 |

Both kinds pass native 262143 and 262144, preserving nonaligned logical lengths.
The capacity test holds 6,385,827,840 DRAM bytes/device plus 24 FP32 recurrent L1
reservations while executing native 262143 full attention and last-position
traced decode. Reservations represent planned stack memory, not execution of
a full model. All strict replica/PCC gates remain unchanged.

## Exact commands and artifacts

All runs use:

```bash
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8
```

Each command below is wrapped by
`python models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/record_run.py NAME`.
The corresponding `logs/NAME.provenance.json`, `.sources.json.gz`, and `.log.gz`
record exact command, source and environment provenance, and exit code.

```bash
# NAME=long_raw_v1
 timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_long_diagnostic --name long_raw_v1
# NAME=long_chunk_capture_v1
 timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_long_diagnostic --name long_chunk_capture_v1 --capture-chunks
# NAME=long_original_order_v1
 timeout 2400 python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_decoder.py -m long -x -q
# NAME=capacity_marked_original_v1 (temporary diagnostic wrappers archived)
 timeout 900 python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_capacity.py -x -q -s
# NAME=capacity_original_stress1, then capacity_original_stress2
 timeout 900 python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_capacity.py -x -q
# NAME=long_capacity_watcher_v1
 TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 timeout 900 python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_capacity.py models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_decoder.py -k 'native_context_with_stack_reservations or long_context_pcc' -x -q
```

Raw tensors remain local in `long_raw_v1.pt` and `long_chunk_capture_v1.pt`.
Committed compact evidence includes their `.json` summaries and
`.tensor_manifest.json` shape/dtype/SHA256 inventories;
`long_chunk_assembly_comparison.json` records exact assembly equality. These
large deterministic control tensors are excluded by the stage artifact policy.
The diagnostic script reconstructs the controls from pinned real weights and
recorded activations. The contract adapter now writes unique raw tensors and
finite/delta statistics **only on replica failure**, preserving its strict gate
and successful-forward timing.

## Final status

Controlled recovery succeeds without production changes. Original low-level
cause remains unproven; no claim of a concat, precision, page-table or capacity
fix is made. Worker watcher (`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`)
passes capacity plus 8001-token both kinds: 3 passed, 95 deselected in 187.43 s, clean
process exit 0 and no watcher errors. This checks worker instrumentation; ETH
checks are disabled for the previously established kernel-size/teardown issue.
The final raw watcher artifact is
`logs/long_capacity_watcher_v1_final_full_attention.watcher.log.gz`, with
`long_capacity_watcher_raw_provenance.json`. It covers the final full-attention
fixture only because the runtime overwrites the raw watcher file on each mesh
open; the complete console log records checks and passing outcomes for all 3
cases. Do not label this an all-features or all-fixtures raw-watcher capture. Black formatting and Python compile
checks pass for the added diagnostic and failure-evidence adapter. No C++ build
is required for these Python test-only changes.
