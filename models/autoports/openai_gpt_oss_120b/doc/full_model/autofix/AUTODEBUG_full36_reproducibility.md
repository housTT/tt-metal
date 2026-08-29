# AutoDebug: full-36 prefill/decode reproducibility

## Scope and headline

This was a fresh, source-only investigation. No TT hardware command was run
and no implementation or test file was changed.

This report records the pre-recovery diagnosis.  The prescribed cold-reset
experiment subsequently passed twice without a source change; the resolution
is recorded in `AUTOFIX_full36_reproducibility.md`.

**Headline:** the full artifact records genuine invocation-to-invocation
numeric nondeterminism, but not a wrong greedy result or gross corruption. The
bitwise gate is not shown to be invalid: identical reduced-stack execution is
bitwise exact, the selected layer tests require exact replica/replay behavior,
and no selected production op is documented as intentionally nondeterministic.
The highest-ranked cause is stale P150x4 fabric/collective state, for which this
same autoport already has an exact source-unchanged/reset/fixed precedent. If a
cold-reset unchanged rerun still fails, the next suspect is the shared CCL
manager's deep/repeated semaphore contract, not paged KV state or the recent
decode-only sub-tile compaction.

## Starting evidence

- Failing payload:
  `artifacts/logit_reproducibility.json`, full 36 layers, TP4, batch 2,
  configured context 131072, identical 214-token prompts.
- Passing control:
  `artifacts/logit_reproducibility_probe.json`, layers 0--1, TP4, batch 2,
  context 512. Prefill and decode are bitwise equal across both rows and both
  reset/reuse runs after the explicit decode-row compaction fix.
- The full prefill is four sequential batch-1 executions: two physical page
  rows, then `Generator.reset()`, then the same two rows again
  (`tests/test_full_model.py:588-609`; `tt/generator.py:216-226`).
- Every full prefill/decode logit tensor is finite. Prefill argmax is 200005 in
  all four cases and decode argmax is 35644 in all four cases. Those are also
  exactly the first two pinned HF reference tokens. Thus the failure has not
  changed the semantically greedy decision.
- Nevertheless, every raw full-logit hash and every top-100 ordering hash
  differs. Prefill has max row/run differences 1.75/1.8125 and decode has
  2.875/3.3125. This is real output nondeterminism, not a `torch.equal`/NaN
  artifact.

## What the evidence rules out

### The recent decode batch-row fix is not the first failure

The new slice -> row-major -> tile compaction is used only when
`is_decode=True` and logical batch exceeds one
(`tt/multichip_decoder.py:1524-1559`). Full prefill already differs before that
code can execute. The fix is also directly supported by the now-exact
two-layer prefill/decode gate.

### Paged-cache read/reset cannot explain the first prefill difference

Prefill computes attention from live Q/K/V and only writes its cast K/V into
the page range; it does not read the paged cache for its logits. Row 0 and row
1 differ before the first `Generator.reset()`. Therefore stale cache contents,
different physical page addresses, and the lack of an explicit synchronize
after reset cannot be the headline cause. An explicit reset synchronization is
still worth testing for the cross-run comparison, but it cannot repair the
same-run row comparison by itself.

The readback after every row is `logits.cpu(blocking=True)` before the next row
is dispatched (`tt/generator.py:166-192`). This also demotes an ordinary
unfinished-output host-read race.

### Layer type, terminal path, and prompt shape alone are insufficient

The exact two-layer control contains both GPT-OSS layer types (layer 0 sliding,
layer 1 full), uses the same non-aligned 214-token prompt, final norm, sharded
BFP8 LM head, host gather, and distinct physical pages. The new failure needs
either greater layer/collective depth, the full-context allocation/address
regime, or external device/fabric state.

## Ranked hypotheses

### 1. Stale external fabric/collective state

This is the strongest current hypothesis. The optimized-multichip stage's
`autofix/AUTOFIX_prefill_replication.md` records an unchanged TP4 prefill whose
replicated ranks diverged after async/persistent CCL experiments; normal
process teardown and healthy `tt-smi` status did not clear it. A bounded
physical reset plus mesh reopen made the unchanged check pass exactly. The
current failure followed many full-stack/trace/CCL processes, and the reported
post-test DRAM/error health does not inspect stale fabric-router or global
semaphore state.

Prediction: after a serialized physical reset and fresh mesh process, the
unchanged full-36 gate passes. In that outcome no model fix is warranted; the
failed artifact is retained as pre-recovery evidence and regenerated after
reset.

### 2. Deep/repeated per-layer CCL/fabric state

Each `MultichipDecoder.from_state_dict` constructs a layer-local `CCLManager`;
the separate manager built by the maintained zero-layer terminal base is not
passed into those decoders (`tt/model.py:405-414`, `tt/model.py:464-480`, and
`tt/multichip_decoder.py:1638-1642`). Each TP all-reduce still rotates two sets
of reduce-scatter, all-gather, and barrier semaphores on the same underlying
fabric
(`models/demos/gpt_oss/config.py:128-176` and
`models/demos/gpt_oss/tt/ccl.py:39-95`). The full stack exercises many more
reuses than the exact two-layer control.

The manager exposes `reset_global_semaphores`, but it resets only RS/AG
handles, not barrier handles or rotating indices, and `Generator.reset()` does
not call it. The closely related GPT-OSS DP manager explicitly documents that
its current barrier state is safe only because prefill is one-shot and lists
multi-run prefill reuse as follow-up work
(`models/demos/gpt_oss_d_p/tt/ccl.py:129-139`). This is source evidence of a
contract gap, not yet proof that it caused this run.

Prediction: if a cold-reset run still fails, the first bad layer boundary loses
TP-rank replication, and either correctly reset layer-local CCL state between
requests or a depth-localized collective A/B restores exactness.

### 3. Depth-amplified nondeterminism in a decoder op

The full stack can amplify a small first-boundary difference until most BFP8
logits change while preserving the top token. Candidate boundaries are the
ordinary Blackhole RS+AG path, packed sparse prefill MoE, router/top-k, or a
non-aligned pad/slice boundary. The known fused Blackhole prefill MM+RS race is
already gated off in `models/demos/gpt_oss/tt/attention/operations.py:128-138`,
so that specific fused op is not active.

Prediction: rank replicas remain exact but repeated logical hidden states first
diverge at one layer/subcomponent. No evidence currently identifies such a
component, so changing dtype, fidelity, padding, or sparse geometry now would
be speculative.

### 4. Terminal head or host gather

Demoted. The same final norm, LM head, and gather are exact in the reduced
control, and a blocking read occurs for each row. They remain a cheap final
localization boundary only if the 36-layer hidden state proves exact.

## Bitwise-invariant verdict

The artifact proves two distinct facts:

1. semantic greedy output is stable and correct for these two positions; and
2. the full logit function is not reproducible in the observed device state.

The first fact means this is not evidence of wrong autoregressive tokens. It
does not make the second fact acceptable. Top-100 order changes and nearly all
raw values change, while prior selected layer/trace evidence is bitwise exact.
There is no source contract establishing nondeterministic full inference as an
allowed property. Keep the exact gate until a cold-reset control and a first
differing boundary either prove recoverable external state or identify a
documented unavoidable kernel limitation. Do not replace it with a tolerance
or an argmax-only assertion on the present evidence.

## Smallest serialized hardware experiment

Run these steps under the device lock; stop as soon as one branch gives a
conclusive result.

1. Perform the `$tt-device-usage` bounded reset/list/1x4-mesh-smoke recovery,
   then run the **unchanged** full-36/full-context test in a fresh process. This
   is the cheapest decisive test because the same-repo precedent says external
   fabric state can survive ordinary teardown and health checks.
2. If it still fails, run a four-corner prefill-only matrix, one process at a
   time, always comparing two identical sequential calls before any reset and
   then one call after `reset(); ttnn.synchronize_device(mesh_device)`:

   | layers | context | purpose |
   | --- | ---: | --- |
   | 2 | 512 | already-passing environmental sentinel |
   | 2 | 131072 | isolates context/cache allocation and address pressure |
   | 36 | 512 | isolates layer/collective depth |
   | 36 | 131072 | original failure |

   The same-call comparison must use the same page-table row twice as well as
   two distinct rows. Prefill does not consume cached K/V, so this separates
   invocation state from physical page address. Record raw/top-100 hashes,
   argmax, max difference, and finite status; do not run decode until prefill is
   explained.
3. In the first failing corner, run prefill with `skip_lm_head=True` and hash
   the logical last-token hidden state **on every TP rank** after layer-prefix
   depths 2, 4, 8, 16, 24, 32, and 36 using one already-loaded model. Binary
   refine the first failing prefix. Also run final norm/head separately on one
   captured hidden state.

Interpretation is direct:

- only 2/131072 fails: allocation/address-pressure issue;
- 36/512 and 36/131072 fail: layer/collective-depth issue;
- only post-reset differs: cache-clear/request-reset synchronization issue;
- same-call repetitions differ: reset synchronization is refuted;
- TP ranks first differ: CCL/fabric boundary;
- ranks agree but repeated logical hidden differs: local decoder op;
- hidden is exact but logits differ: final norm/LM head/readback boundary.

## Conditional fixes

- Cold-reset pass: no source change. Regenerate the artifact, record the
  recovery, and rerun the full acceptance command once more in a clean fresh
  process.
- Verified request-boundary CCL drift: synchronize only at reset, reset every
  relevant RS/AG/barrier semaphore plus rotating index to the trace-compatible
  state, and prove both repeated eager prefill and warmed trace reuse. Do not
  reset semaphores while work is in flight.
- Verified deep CCL reuse: fix the first collective's semaphore/lifetime
  contract or give it correctly scoped persistent state. Do not serialize each
  layer or fall back to host/single-chip execution.
- Verified local op: change only the first failing batch/shape/program boundary
  and rerun the two-layer gate, the four-corner probe, the original full gate,
  and the optimized decoder policy/performance checks.

## Final status

**Still failing pending the cold-reset unchanged rerun.** Existing evidence is
enough to reject cache reset, physical page reads, the decode sub-tile fix, and
the two layer types as the first cause. It is not enough to keep an
implementation change. The recovery-first experiment above is the minimum
next action and preserves the optimized multichip policy.
