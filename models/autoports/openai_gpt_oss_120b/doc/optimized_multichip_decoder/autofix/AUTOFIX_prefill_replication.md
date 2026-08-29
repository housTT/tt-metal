# AutoFix Report: TP4 Prefill Replica Divergence

## Starting Evidence

- Original failing check: `test_real_weight_multichip_against_baseline_artifact[blackhole-p150x4-parent-tp4-sliding]`.
- Failure log: `artifacts/20260829_candidates/final_default_tp4_sliding_check.log.gz`.
- Symptom: warmed prefill rank 2 differed from rank 0 in 24,134 elements, with maximum absolute difference 43.875 and inter-rank PCC 0.9951971191. The check failed before decode.
- The immediately preceding work exercised experimental async and persistent CCL paths. Earlier unchanged-source runs of the selected 45-core gate/up and 15-core down decode geometry had passed all TP2/TP4 layer-kind gates.

## Hypothesis Experiments

### Decode Geometry Was Unsafe for Prefill

- Hypothesis: `_ActiveExpertTPMLP` supplies the same sparse-matmul geometry to its decode and prefill program-config fields, so the selected M=1 decode grids might corrupt TP4 prefill.
- Experiment: split the prefill geometry from decode in a focused working-tree change. Prefill used the previously passing gate/up `(3, 4)`, `in0_block_w=30`, `subblock_w=2` and down `(5, 6)`, `in0_block_w=12`, TP4 `subblock_w=3`; decode retained gate/up `(5, 9)` and down `(5, 3)`.
- Command: the original TP4 sliding acceptance command recorded at the top of `artifacts/20260829_candidates/autofix_prefill_phase_split_tp4_sliding.log.gz`.
- Result: identical failure: rank 2 mismatches=24,134, max_abs=43.875, PCC=0.9951971191.
- Verdict: **refuted**. The geometry change had no effect on the failure signature, so it was reverted. No implementation or test change was kept.

### Stale Fabric/Device State After CCL Probes

- Hypothesis: experimental async/persistent CCL probes left stale device or fabric state that survived normal process teardown and contaminated later replicated collectives.
- Experiment and recovery:
  - `timeout 180 tt-smi -r` reset PCI devices 0, 1, 2, and 3 successfully in approximately 38.5 seconds.
  - `timeout 60 tt-smi -ls --local` showed all four p300c devices visible and resettable.
  - A bounded Python smoke opened and closed `ttnn.MeshShape(1, 4)` with `trace_region_size=0`; result: `MESH_SMOKE_OK`.
  - Reran the unchanged default TP4 sliding acceptance command.
- Result: pass. Prefill PCC was 0.9926647803 and traced decode was 0.40064329 ms with PCC 0.9815899522. Evidence: `artifacts/20260829_candidates/post_reset_final_default_tp4_sliding.log.gz`.
- Verdict: **verified as an infrastructure-state failure**. Resetting the four-device fabric restored exact replica agreement without any model-code change.

## Final Status

- Fixed by device/fabric reset; the optimized decoder implementation was unchanged.
- The speculative phase-specific geometry change and opt-in localization instrumentation were removed.
- Hardware remains usable: all four devices enumerate and a 1x4 mesh opens and closes.
- Risk: async/persistent CCL experiments can leave state that normal process teardown does not clear. Treat a deterministic replica-divergence appearing immediately after such probes as recoverable infrastructure first: preserve the failure, run the bounded reset/list/mesh-smoke sequence, then rerun the unchanged check before changing model code.
