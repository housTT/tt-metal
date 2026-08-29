# AutoDebug: peer-zero omission is not the layer-output failure

## Headline finding

The focused PCC failure is **pre-existing at the stage's starting HEAD and is not caused by skipping the peer-zero D2D copies**.

Two controlled artifacts are decisive:

- `autofix_force_peer_zero_control.xml` forced `reset_non_owner=True` for every prepared miss, restoring the four HEAD-era D2D copies (owner gate/down plus peer-zero gate/down). It still produced the exact same output PCC `0.8637402653694153` and the same displayed expected/actual values as `focused_owner_zero_tt.xml`.
- `autofix_clean_head_control.xml` ran the same test from a detached, clean worktree at HEAD `adcfaa2191584bdb6e56c1d2e8b49ecc278a9a51`; the worktree had no diff and its `host_weight_cache.py` was byte-identical to `git show HEAD:...`. It also produced the identical PCC and displayed values.

Therefore neither the new owner ledger nor the conditional peer reset is on the causal path for this observed mismatch. This stage uncovered a stale/pre-existing current-HEAD regression; it did not introduce it. Older committed artifacts (`expert_ep2_hardware_contracts.xml`, `baseline_real_pcc.xml`, and `candidate_stable_index_real_pcc.xml`) show this test passing in the earlier multichip stage, so the broader regression interval is after the last known passing multichip source (commit `5b8898664b3`) and no later than current HEAD.

No implementation fix is justified from the peer-zero evidence.

## Direct observations

1. The live production delta is narrow. `multichip_decoder.py` is unchanged. In `host_weight_cache.py`, it adds `_slot_last_owner`, derives `reset_non_owner`, conditionally omits only the non-owner `ttnn.copy` pair, and records ledger/metrics state (`host_weight_cache.py:623-628`, `819-875`, `892-968`). HEAD always enqueues the peer-zero pair.

2. The failing eager decode reads route IDs to the host, calls `ensure_indexed`, and immediately constructs/consumes the ten-slot bank (`multichip_decoder.py:1523-1547`, `1549-1620`, `1698-1703`). The consumer concatenates **all cache slots**, not only the first one (`1552-1556`).

3. The failing test checks only the first valid directory slot after the final device synchronization (`test_multichip_decoder.py:608-620`). Passing those four PCC checks proves that slot's two projections on both physical ranks eventually match the packed source. It says nothing about the other nine slots and, because it occurs after the layer computation and final synchronize, it does not by itself prove what the consumer observed at execution time.

4. The separate completed-cache test now closes most of that coverage gap. It runs twenty ten-expert same-owner waves with a synchronization per wave, checks all ten final same-owner slots on both ranks and both projections, flips ownership for all ten slots, and checks all ten again (`test_multichip_decoder.py:661-734`). `focused_same_owner_skip_flip_tt.xml` passes with 200 skips, 10 resets, and warm aggregate completed bandwidth `11.903243456796487 GB/s`. Thus all physical slots can retain exact owner data and exact-zero peers across the optimized reuse and owner-flip paths after completion.

5. Construction-time `ttnn.from_torch(torch.zeros(...), device=mesh_device)` is **not guaranteed device-complete when it returns**. `_replicated_device_zeros` uses `from_torch` (`host_weight_cache.py:562-581`); `from_torch` lowers through `Tensor` construction and `to_device` (`ttnn/ttnn/operations/core.py:370-383`, `ttnn/core/tensor/py_to_tt_tensor.cpp:458-489`). These BFP4 gate/down payloads are each below the 32 MiB pinned-write threshold, and the uniform write path uses `blocking=false` below that threshold (`tt_metal/impl/tensor/tensor_apis.cpp:40-47`, `174-219`). The new `None` ledger state therefore means "zero initialization was enqueued," not "a host completion fence proved zero."

6. That completion nuance does not explain this failure. Initialization, coordinate H2D, D2D, index publication, and eager consumption are submitted on the default parent mesh CQ; the test also synchronizes after prefill before decode (`test_multichip_decoder.py:590-608`). More decisively, restoring every peer-zero D2D and reverting all stage edits to clean HEAD leaves the result bit-for-bit unchanged, while the all-slot completed test validates the storage invariant.

## Ranked, falsifiable remaining hypotheses

1. **Very high confidence classification: a broader current-HEAD decode regression predates this stage.** This is proven by the clean-HEAD control, though the responsible earlier change is not isolated. The historical pass/current fail interval spans substantial full-model/vLLM changes, so assigning a specific source fix without a commit or stage split would be speculation.

2. **Moderate: a general cache-service-to-consumer ordering issue independent of peer-zero reset.** All inspected weights are correct after synchronization, but the focused test has no completion boundary between `ensure_indexed` and `_routed_experts_indexed_ready`. An owner H2D/D2D or slot-bank consumer edge could still be missing even though an extra peer-zero workload is irrelevant. This hypothesis predicts that one synchronize/event after `ensure_indexed` repairs the layer PCC while leaving final slot checks unchanged.

3. **Moderate if hypothesis 2 is false: a decode-only state, route-weight, indexed-MoE, or downstream reduction regression outside the cache bytes.** Prefill clears `0.995`, then decode deterministically fails with the same values under optimized, forced-reset, and clean-HEAD cache behavior. This predicts all ten active slots are exact immediately before consumption and the first divergence appears in route weights, indexed expert output, reduction, or final injection rather than in cache storage.

4. **Low: an unobserved bad slot in the exact failing route.** The original test checks one slot only, so this remains logically possible for that run. It is strongly demoted by the completed all-slot same-owner and owner-flip pass. It predicts at least one of the other nine active slots fails an immediate per-slot check in the focused test.

5. **Refuted for this symptom: construction-zero completion or the removed peer-zero D2D as a required fence.** Construction is enqueue-only, but forced peer-zero and clean-HEAD controls reproduce the identical failure. If the omitted copy were the necessary edge, forcing it on every miss would have changed or repaired the output; it did neither.

## Smallest discriminating follow-ups (not run here)

1. **Bisect the pre-existing regression:** run only the focused test in detached worktrees at `5b8898664b3` (last known passing artifact) and the subsequent model/full-model/vLLM commits through `281c0c1876f`. Prediction: the earlier multichip commit passes and the first failing commit narrows the real regression boundary; peer-zero stage changes are absent from every candidate.

2. **General ordering A/B:** monkeypatch `ensure_indexed` in the focused test to synchronize or wait on a CQ0 event immediately after service and before `_routed_experts_indexed_ready`. Prediction: recovery to `>=0.995` isolates a general owner-upload/D2D-to-consumer edge; unchanged `0.8637402653694153` refutes cache-service ordering.

3. **Exact failing-bank check:** after servicing the focused decode route, synchronize once and validate every active slot selected by `plan`, both ranks and both projections, against `host_expert_source.load`. Prediction: all ten pass based on the direct cache control; any failure instead identifies the exact slot/owner/generation and moves cache-byte corruption back up the ranking.

4. **Earliest-divergence stage split:** compare resident and host-backed route IDs, selected route weights, indexed expert output before shared-expert addition, reduce-scatter output, and final hyper-injection in that order. Prediction: the first PCC drop localizes the pre-existing HEAD regression without conflating it with post-compute slot inspection.

5. **Constructor fence A/B (lower priority):** synchronize immediately after `QwenDeviceExpertCache` construction. Prediction: no change, because the existing test already supplies a post-prefill completion boundary and both forced-reset and clean-HEAD runs fail identically. A recovery would contradict the current controls and warrant a smaller first-fill-only reproduction.

## Investigation constraint

The required fresh AutoDebug runner was launched first, but its isolated executor could not read or write because its `bubblewrap` setup failed with `RTM_NEWADDR: Operation not permitted`. The findings above are the required post-run verification performed source-only in the parent workspace. No TT device, pytest, vLLM, or hardware command was run during this investigation, and no implementation or test source was edited.
