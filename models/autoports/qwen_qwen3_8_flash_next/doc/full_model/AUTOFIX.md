# Full-model AutoFix record

Date: 2026-08-27

This stage invoked `$autofix` for the nontrivial lifetime and progressing-HF
failures. Proposed causes were tested in isolation; refuted hypotheses were
not retained.

## 1. Persistent expert slot destroyed by a singleton wave

Symptom: the first full 48-layer split-trace smoke failed during trace warm at
layer 5 with `expert slots require exactly two device shards`. Cache
construction had already proved that every slot was a two-shard mesh tensor.

Confirmed cause: the final prefill wave can contain one expert.
`ttnn.concat([slot], dim=1)` lowers to `to_memory_config(slot)`, which returns
the same shared tensor when it is already in DRAM. The old unconditional
`ttnn.deallocate(gate_up_bank/down_bank)` uses force deallocation and therefore
destroyed the persistent aliased slot. Later `get_device_tensors` correctly
reported a singleton because the mesh tensor was no longer allocated.

The ownership alternatives were refuted from source:

- rank-local shard handles share the parent `MeshTensorHolder` and ordinary
  destruction uses non-forced deallocation, which refuses to release shared
  storage;
- temporary rank-one model teardown completes before cache-slot allocation;
- construction itself asserts two shards for every new slot.

Fix: replace explicit bank deallocation in both routed-expert wave paths with
the existing alias-aware `_free(bank, *slots)` helper. The indexed path change
is defensive; the prefill wave change is causal.

Verification: `full48_slot_lifetime.xml` preserves the original failure and
`full48_after_singleton_fix.xml` passes the same 48-layer three-token
token-out trace. No mesh replication or host fallback was introduced.

## 2. Canonical fused recurrent-state lifetime corruption

Symptom: the missing optimized progressing-HF gate exposed an address-sensitive
standalone GDN trajectory. The retained live prefill output could corrupt L1
canonical recurrent/conv state even though no logical writer targeted it.

Confirmed repair: canonical GDN recurrent state, its three fused convolution
taps, and PLE convolution taps now reside in DRAM. Batch-1 host-backed
multichip execution keeps its existing declared L1 compute path by hydrating
and committing those canonical tensors through `MultichipDecodeStateWorkspace`;
the recurrent temporary memory follows the active bound state memory config.

Refuted causes:

- multichip workspace copy direction is correct (`canonical DRAM -> shared
  L1` on entry and `shared L1 -> canonical DRAM` on exit);
- queue order and the replay lock serialize hydrate/compute/commit;
- moving state alone did not change the original full-stack top-k failure, so
  it was retained for the independently proven lifetime defect, not claimed
  as the accuracy root.

Prevention: the functional progressing-HF test now returns and asserts its
PCC trajectory, and fused/optimized wrappers exercise the real 12-transition
path instead of checking only the first decode token.

## 3. Full-stack top-k loss from one invalid 1D projection role

Symptom: initial AIME24 teacher forcing passed prefill but decoded at
47.47/73.74/90.91% top-1/top-5/top-100. Trace and eager produced the same
first four-token trajectory, refuting an extra trace position transition.

The investigation also refuted these leading alternatives:

- GDN/PLE device recurrence and convolution taps are included in trace
  snapshot/restore; PLE host history is restored around warm/capture and
  advances exactly once;
- QSA top-k block membership cannot drop a visible block below 2,048 tokens,
  and its mod-4 ordering perturbation could not explain later top-100 loss;
- expert route-ID/weight ordering is aligned with the resident indexed
  kernel; ordered versus indexed cache publication did not identify a mapping
  defect;
- state-to-DRAM did not change the first 12 full-stack misses.

Confirmed cause: only the `gdn_qkv_b_a@0:55` decode 1D role. Removing that
role while preserving every other selected 1D role, BFP4 expert policy,
projection dtype/fidelity, collectives, fractured residual, and host stores
changed host-backed layer-0 progressing-HF PCC from approximately
0.96066->0.91035 to 0.999825->0.998188. The 12-row full-stack control changed
from 33.33/83.33/91.67% to 83.33/100/100%, and the final 99-row run passes at
91.92/100/100%.

Fix: remove `gdn_qkv_b_a@0:55` from
`OptimizedDecoder.DEFAULT_DECODE_1D_CONFIG`. The packed projection remains a
TT matmul at the inherited BFP8/HiFi2 policy; only the invalid width-sharded
geometry is rejected.

Evidence:

- `progressing_hf_optimized_layer0_fixed_default_policy.xml`
- `progressing_hf_multichip_host_layer0_no_gdn_qkv_1d.xml`
- `aime24_teacher_12_after_state_dram.xml` (rejected control)
- `aime24_teacher_12_no_gdn_qkv_1d.xml`
- `aime24_teacher_99_l1_workspace_final.xml`

## 4. Endpoint all-gather crash

The earlier reduced endpoint crash was handled through `$autodebug` and is
recorded separately in `AUTODEBUG.md`. The confirmed cause was a redundant
force deallocation after alias-aware `_free`; removing the second deallocation
made the real endpoint path and later full-stack gates pass.

## 5. Linear Sampling1D watcher assertion

Symptom: the first full-48 watcher run stopped device 0 on a BRISC assertion
in `minimal_default_writer.cpp` while the host was reading the sampled token.
The one-layer real-weight split-greedy watcher test reproduced the identical
core/kernel assertion, refuting decoder depth, expert/PLE service, and the
long full-stack CCL schedule.

AutoTriage resolved the reported line through the cached kernel DWARF to
`FabricConnectionManager::get_forward_connection()` and its
`ASSERT(has_forward_connection())`. Sampling1D force-argmax uses a Linear
experimental all-gather with one worker per direction. The non-mux writer
launched both directions and unconditionally requested the selected
connection before checking targets; a line endpoint legitimately has no
outward connection and zero outward targets.

Fix: the generic writer now retrieves a forward/backward connection only when
`has_*_connection()` is true and asserts that a null connection is permitted
only for a direction with no targets. Every later dereference remains guarded
by a valid target or nonzero `writes_expected`; valid sender paths and sampler
semantics are unchanged.

Verification:

- `reduced_split_trace_watcher.log` retains the pre-fix assertion;
- `reduced_split_trace_watcher_fixed.xml` passes eager argmax, capture/replay,
  direct feedback, positions, and page-table checks under watcher;
- `full48_tokenout_watcher_fixed.xml` passes the original all-48-layer
  token-out watcher test in 87.56 s with clean teardown;
- `split_greedy_sampler_strategy_ab_final.xml` keeps full-vocabulary argmax
  selected at 0.665837 ms versus 0.906958 ms for local-top32 k=1, with both
  matching host argmax.

The detailed triage ledger is in `AUTOTRIAGE.md`. No sampler downgrade or
host fallback was retained.

## 6. Active-trace allocation warning

Symptom: accepted untracked split-trace runs emitted TTNN's generic warning
that a device allocation created while a trace is active can be corrupted on
replay. The first stage review correctly rejected source-only dismissal.

AutoDebug ranked the decode-state snapshot clones allocated after ingress
capture as the likely source. Their ownership appeared bounded, but the report
required the runtime tracker as the decisive experiment. The reduced split
trace and original full-48 token-out test were rerun with
`TT_METAL_TRACE_ALLOC_TRACKING=1`, allocation tracebacks/depth 12, and watcher.
Both passed capture/replay and teardown with no tracker `RuntimeError`:

- `reduced_split_trace_alloc_tracker.xml`
- `full48_tokenout_trace_alloc_tracker.xml`
- `AUTODEBUG_TRACE_ALLOC.md`

Verdict: controlled. Every allocation younger than the active trace is freed
or marked safe before replay. No speculative state copy, trace disable, or
allocation suppression was retained.

## 7. Non-greedy seed-control hole

Symptom: the public generator accepted `seeds`, but device sampling did not
publish them to `Sampling1D`; only host compatibility sampling used them.

Fix: explicit non-greedy requests own deterministic per-lane Python RNGs.
Immediately before each non-greedy eager sample, sampling capture execution,
or sampling replay, the next bounded uint32 seed is copied into the existing
persistent device seed buffer. Trace warm does not consume a seed. This is
declared compact lookup/control H2D and is absent from canonical greedy decode.

Verification: `non_greedy_split_trace_final.xml` passes real reduced weights
with top-k 4, top-p 0.95, temperature 0.8, and seed 12345. It proves top-k
membership, changing seeds, direct `tt_out_tok` feedback, positions 4 through
7, changed/unchanged page tables, no token/position refresh after capture, and
sampled -> greedy -> sampled trace invalidation/rebuild. Greedy remains exact
full-vocabulary argmax.

## 8. Review evidence gaps

The first review also identified two evidence gaps rather than code failures.
They were closed rather than waived:

- `full48_batch32_eager_fixed_slots.xml` proves the real all-48 host-backed
  model/generator at batch 32, mixed prompt lengths 1/33, 30 inactive rows,
  distinct page tables, device feedback, reset, and PLE request isolation.
- `qualitative_shared_suite_final.xml/json` plus `QUALITATIVE_REVIEW.md` prove
  three exact chat-template prompts against a fresh 128-token HF control.
  Human review records coherent/non-degenerate TT output and the shared
  visible-reasoning truncation limit rather than claiming complete answers.

## Final verdict

AutoFix found and verified minimal repairs for every blocking model bug and
closed every first-review evidence gap. The final source passes the full top-k
bar, allocation-tracked and watcher-clean 48-layer token-out trace,
non-greedy split trace, and all-48 batch-32 eager gate. No
speculative precision blanket, single-chip fallback, CPU projection, cache
replication, activation round trip, or undeclared host work was retained.
