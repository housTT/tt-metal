# AutoFix: fully enabled ACTIVE_ETH watcher

## Final status

Fixed. The final watcher gate uses the repository-supported
`TT_METAL_FABRIC_OPT_LEVEL=Os` override to fit the two-ERISC fabric program in
the 26-KiB ACTIVE_ETH kernel-config region. All watcher features remain enabled.
A one-line router teardown repair clears sticky write-capable NoC packet tags
after draining writes and atomics, before the ERISCs return to firmware.

The final TP2/TP4 sliding/full matrix passed in one process and closed cleanly:

| Case | Prefill PCC | Decode PCC | Result |
| --- | ---: | ---: | --- |
| TP2 sliding | 0.9931029157 | 0.9991766884 | pass |
| TP2 full | 0.9920713329 | 0.9975173691 | pass |
| TP4 sliding | 0.9926647803 | 0.9986537098 | pass |
| TP4 full | 0.9924199023 | 0.9977844013 | pass |

The log records `Watcher server initialized, disabled features: None` on every
mesh reopen, four passing real-weight cases, normal watcher shutdown, normal
cluster destruction, and `SAFE_PYTEST_RESULT: PASS`.

## Starting evidence

- Source diagnosis:
  `autofix/AUTODEBUG_watcher_active_eth.md`.
- Original strict-watcher failure:
  `artifacts/20260829_watcher/default_release_watcher10_all.log.gz`.
  Normal inlining required 29,072 bytes in a 26,624-byte ACTIVE_ETH
  kernel-config buffer before model execution.
- `TT_METAL_WATCHER_DISABLE_ETH=1` passed the model matrix in
  `default_release_watcher10_noeth_all.log.gz`, but reported `disabled features:
  ETH`; it was rejected because it does not satisfy fully enabled watcher.

## Hypothesis experiments

### Global watcher no-inline

Hypothesis: `TT_METAL_WATCHER_NOINLINE=1` could shrink the fabric image without
disabling watcher.

Result: admission succeeded with `disabled features: None`, and model PCC
passed, but isolated TP2-full and post-reset TP4-sliding processes aborted at
teardown with a frozen ACTIVE_ETH heartbeat. Example:
`autofix_watcher_noinline_tp2_full_separate.log.gz`, exit 134.

Verdict: rejected. Global no-inline changes all `FORCE_INLINE` router helpers,
not only watcher diagnostics, and did not produce a clean real-model process.

### Selective router or watcher outlining

Focused normal-watcher mesh-open experiments gave:

| Candidate | Program/admission result | Verdict |
| --- | --- | --- |
| Speedy sender/receiver steps no-inline | 29,072 B, overflow | no effect; reverted |
| `run_routing` no-inline | 27,680 B, overflow | insufficient; reverted |
| Both routing helpers no-inline | 27,408 B, overflow | insufficient; reverted |
| `run_routing` + local-unicast helper no-inline | admitted | advanced to model test |
| Local-unicast only negative control | 26,672 B, overflow | proved both outlines needed |
| `debug_sanitize_noc_addr` no-inline | 29,216 B, overflow | grew due to specialization/clones; reverted |

The minimal two-router-helper candidate passed mesh admission, but a real
TP2-full model process passed PCC and then exited 134 during teardown
(`autofix_watcher_selective_tp2_full_separate.log.gz`). All outlining changes were
therefore reverted.

### Teardown localization

Hypothesis: the heartbeat failure was downstream of invalid router return
state, rather than an image-size or fabric-barrier deadlock.

Experiment: temporary bounded teardown waypoints were added to the admitted
two-helper candidate and TP2-full was rerun under fully enabled watcher.

Result: watcher reported on device 2, active-Ethernet core 29-25 subordinate
ERISC:

```text
invalid NOC command buffer state before starting the next kernel
(write-capable NOC packet tags must be zero ...)
Last waypoint: SEW,NKFW
```

Evidence: `autofix_watcher_selective_tp2_full_teardown_waypoints.log.gz`.

Verdict: verified. Blackhole `NOC_PACKET_TAG` selects transaction IDs;
`active_erisck.cc` asserts `ncrisc_noc_packet_tags_cleared` after the kernel
returns. Router teardown drained writes/atomics and reset counters but did not
clear this sticky state. The heartbeat timeout was a consequence of the
watcher assert.

Fix: after `noc_async_write_barrier()` and `noc_async_atomic_barrier()`, call
`noc_clear_packet_tags(noc_index)` in
`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`. This leaves
implicit transaction-ID users in the firmware/next kernel at the required ID
zero. Temporary waypoints were removed.

### Admission without outlining

Hypothesis: the existing fabric-only optimization override `Os` would shrink
the normally inlined router enough while preserving all watcher features.

Result: with `TT_METAL_WATCHER=10 TT_METAL_FABRIC_OPT_LEVEL=Os`, the focused
1x4 mesh-open smoke admitted, closed cleanly, reported `disabled features:
None`, and returned `SAFE_PYTEST_RESULT: PASS`
(`autofix_watcher_fabric_os_packet_tag_clear_mesh_smoke.log.gz`). An isolated
TP2-full model then passed PCC and closed cleanly
(`autofix_watcher_fabric_os_packet_tag_clear_tp2_full.log.gz`).

Verdict: verified. This avoids global or selective no-inline source changes and
uses the runtime option implemented by `rtoptions.cpp` and consumed by the
fabric datamover builder.

## Final verification

The final original-path matrix command was:

```bash
TT_METAL_WATCHER=10 TT_METAL_FABRIC_OPT_LEVEL=Os \
GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 \
GPT_OSS_120B_SNAPSHOT=<openai-gpt-oss-120b-snapshot> \
GPT_OSS_120B_MULTICHIP_ARTIFACT_DIR=<20260829-baseline-artifact-dir> \
GPT_OSS_120B_MULTICHIP_RUN_ID=20260829_optimized_multichip_baseline \
GPT_OSS_120B_MULTICHIP_TRACE_REPEATS=100 \
GPT_OSS_120B_MULTICHIP_TRACE_SAMPLES=1 \
scripts/run_safe_pytest.sh \
models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py::test_real_weight_multichip_against_baseline_artifact \
-q -s
```

Evidence: `artifacts/20260829_watcher/autofix_watcher_fabric_os_packet_tag_clear_all.log.gz`.
It reports `4 passed` and `SAFE_PYTEST_RESULT: PASS`. This single process
exercises fabric stop/reopen between all four cases and therefore covers the
original teardown/reinitialization risk.

After the matrix, `tt-smi -ls --local` showed all four p300c devices visible
and resettable (`autofix_watcher_final_post_matrix_tt_smi.log.gz`). No reset was
performed because the devices were healthy.

The retained kernel source was JIT-compiled and run on real hardware by the
focused smoke, isolated model, and four-case matrix. `git diff --check` passed.
A full CI-wrapper host build was not run: Garage credentials were unavailable
and the prescribed wrapper warned that this would be a cold build. The earlier
30-KiB HAL reservation candidate and its host build artifacts were reverted;
it is not part of the final repair.

## Retained and reverted changes

- Retained: packet-tag clear in fabric router teardown.
- Runtime watcher command requirement: `TT_METAL_FABRIC_OPT_LEVEL=Os`.
- Removed: diagnostic-only mesh-smoke test and teardown waypoints.
- Reverted: 30-KiB HAL reservation, global/selective no-inline candidates, and
  sanitizer outlining.
