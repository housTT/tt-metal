# AutoDebug: optimized-full-model review blockers

Date: 2026-08-20 EDT

Scope: inspection only. No TT device command was run and no implementation file
was edited. The repo-local AutoDebug launcher was invoked first, but its nested
Codex process could not create its bubblewrap network namespace (`RTM_NEWADDR:
Operation not permitted`) and therefore could not read the checkout. The
findings below come from a fresh-context direct inspection of the same checkout
after that launcher exited.

## Headline findings

1. **The default greedy documentation and candidate ledger are false.** Qwen
   selects common `force_argmax`, and that branch gathers all 262,144 BF16
   logits on every TP device before `untilize` and `argmax`. It does not compute
   a local maximum or exchange compact candidates. The profile and context
   accounting agree with the code.

2. **A compact greedy family was measured, but was mislabeled and was not tested
   with the required physical-Ring candidate collective.** Common `Sampling1D`
   performs local `max_top_k=32`, gathers compact values and indices, then calls
   semantic sampling. The Qwen benchmark measured this at about 10.74--10.80 ms
   and the prior work log says its Linear collective failed safe Watcher. Thus
   the review is right that the selected path violates the topology contract,
   but too strong when it says no semantically greedy compact control was
   tested. The missing experiment is compact candidates over the proven TP4
   Ring protocol, not another comparison to the existing full-logit argmax.

3. **The 19.2209 to 15.6365 t/s/u difference is not a model-trace or sampler-trace
   regression.** Autonomous full64 timing is essentially unchanged across the
   two stages. Nearly the entire teacher-forcing delta localizes to the extra
   per-token host-feedback boundary: compact token readback, callback, creation
   of a new replicated host tensor, and ground-truth token copy to the device.
   The optimized source diff does not change this loop. The runs also used
   different installed TT runtimes, so the artifacts do not identify whether
   the added feedback cost is a runtime regression or transient host/runtime
   variance.

4. **The 14.62 to 23.84 second TTFT comparison is likewise confounded and is
   dominated by cold/setup behavior.** Readiness TTFT starts before the first
   `generate()` prefill and includes compilation/cache/setup for the 161-token
   prompt. The two artifacts use different TT runtime trees and do not record
   cache hit/miss state. In contrast, the same-process, exact-shape, warmed
   prompt-128 A/B shows the only optimized prefill change improving TTFT from
   667.532 to 646.847 ms. There is no code-backed causal path from removing 48
   allocate/concat/copy sequences to an additional 9.22 seconds. The underlying
   cold-start delta remains unresolved until controlled repeats are collected.

## 1. Greedy sampler: actual dataflow

### Direct observations

- `tt/model.py:328-357` builds `SamplingGenerator` with
  `allow_force_argmax=True`, a physical Ring topology, and small-Ring opt-ins.
- `tt/generator.py:154-194` normalizes scalar greedy parameters to 32 rows of
  `k=1, p=0, temperature=1`, making every row eligible for force argmax.
- `models/common/sampling/tt_sampling.py:53-75` documents and detects this mode.
- The executed branch at `tt_sampling.py:525-567` does:

  `TP-local [1,1,32,65536] logits -> Ring all_gather(dim=3) ->`
  `[1,1,32,262144] logits/device -> untilize -> argmax -> tt_out_tok`.

  The local tensor is never reduced before communication.
- `doc/context_contract.json` independently accounts for a 16,777,216-byte
  BF16 global-logit transient, exactly `1*1*32*262144*2` bytes.
- `evidence/final/profiler/token_out_summary.csv` reports about 0.885 ms of
  `AllGatherAsyncDeviceOperation` and 1.418 ms of `ArgMaxDeviceOperation`.
  These are approximately 52.8% of the reduced terminal profile.
- `README.md`, `runtime_fallback_audit.md`, and `candidates.csv` instead claim
  local maxima and compact candidate exchange. Those claims describe code that
  is not selected.

This is a documentation/selection bug, not an inference from timing. The full
vocabulary collective is explicit in the selected source branch.

### What the existing alternative really tests

`models/common/modules/sampling/sampling_1d.py:443-494` performs a local top-k
on each vocabulary shard and then gathers only `max_top_k` values and indices.
The Qwen benchmark calls it with the common parameter tensors and proves its
selected token equals host greedy (`tests/test_full_model.py:1204-1241`). With
four devices and `max_top_k=32`, its collective payload is 128 candidates per
row, not 262,144 logits.

The alternative is nevertheless not a valid final choice on this machine:

- its fallback collective is hard-coded Linear at
  `sampling_1d.py:498-523` (unless a `line_all_gather` implementation intercepts
  it);
- the prior full-model work log records 10.797 ms and a safe-Watcher writer
  stall for the common standard top-k Linear gather;
- it is 4.44x slower than full-logit force argmax in the recorded run.

Therefore `candidates.csv` is doubly misleading: it calls force argmax
`split_local_max_ring_winner`, and calls `Sampling1D` `full_vocab` even though
the source dataflow is the reverse.

### Correct compact split-greedy experiment

The least disruptive candidate is a greedy-only common-sampler branch:

1. Keep logits sharded as `[1,1,32,65536]` and BF16-convert locally if needed.
2. Run the established local `topk(k=32)` (or a proven local argmax/max pair).
3. Convert local indices and add each shard's `device_id * 65536` offset.
4. All-gather only the candidate values and global indices using a compact
   async Ring configuration validated for their dtypes and shapes.
5. Select semantic `k=1, p=0, temperature=1`, writing directly into the caller's
   persistent `tt_out_tok`.

Keeping the output tensor identity is sufficient for the existing split-trace
feedback contract: `SamplingGenerator.capture_trace()` records the output and
`sample()` replays it nonblocking (`models/common/sampling/generator.py:322-435`).
The trace key already separates force-argmax from stochastic configurations.

Do not silently route all stochastic traffic through this new branch. Qwen's
current non-force small-Ring mode deliberately gathers full logits and then
runs four 65,536-wide top-k kernels (`tt_sampling.py:573-610`), because the
older compact Linear candidate collective stalled. Preserve that known
top-k/top-p/temperature/seed/penalty path while landing greedy, or validate a
separate compact Ring stochastic migration with the full stochastic suite.

### Upstream commit 3497b22f6ae is the best blueprint, but not drop-in

Upstream commit `3497b22f6ae` (`BH decode perf: distributed force-argmax
sampling`) adds almost exactly the desired algorithm to common
`TTSampling._distributed_force_argmax`:

- BF16 local untilize plus local argmax/max;
- one-tile-per-device value and int32-index candidate gathers;
- deterministic two-level tie-breaking matching global first occurrence;
- exact global-token reconstruction despite TF32 integer precision limits;
- direct write to a persistent RM uint32 `[1,1,1,32]` `tt_out_tok`;
- eager and traced tests with fresh persistent input data.

Its commit message reports a Blackhole decode improvement from 50.7 to 52.8
t/s/u. Qwen's local logits and output buffer satisfy most guards: batch 32,
width 65,536 (tile aligned), sampling-DP 1, and RM uint32 32-element output.
The global ID formula `local_idx + local_width * winner_column` also matches
this TP4 contiguous vocabulary sharding.

It intentionally cannot run here unchanged:

1. `_use_distributed_argmax()` returns false when `cluster_axis is None`.
   Qwen's correct 1x4 physical-Ring contract deliberately resolves to
   `cluster_axis=None`; forcing axis 1 previously did not cure its CCL stop.
2. The upstream code derives `ncols = cluster_shape[cluster_axis]`. For this
   1D default-axis case it must use the actual TP group size (four), while also
   proving that the default-axis Ring gather concatenates candidates in the
   same logical device/vocabulary order used by global-ID reconstruction.
3. Its `tiny_gather()` always supplies a barrier semaphore. Qwen's proven
   small physical-Ring full-logit protocol specifically omits the Linear
   barrier. The adaptation must use the Qwen Ring completion policy for
   `cluster_axis=None`, not copy the Galaxy barrier policy.
4. The path is gated on `_force_argmax_sub_core_grids` and uses that grid for
   every operation, plus an optional worker sub-device ID. Qwen does not set a
   sampling sub-core grid or use the Galaxy split-senders/worker lifecycle.
   It needs an explicit, legal Qwen worker grid and must omit Galaxy-only
   subdevice pinning. The upstream `bitwise_or(..., output_tensor=tt_out_tok)`
   is still attractive because it preserves traced output identity, although
   its stated `ttnn.copy` race is specific to the split-subdevice flow.

This is still stronger than inventing a new local-top1 implementation. It
already handles subtle failures that the old Qwen candidate in
`tests/test_full_model.py:1382-1446` does not: argmax core-count underflow,
undefined tile pad lanes, global tie order, exact large integer reconstruction,
and trace-safe output placement.

However, prior Qwen evidence prevents calling the adaptation "proven" before a
Watcher run. `doc/full_model/AUTOFIX.md:124-136` records that (1) synchronous
Ring candidate gather stopped in its writer, (2) the exact then-proven async
Ring protocol also stopped on candidate payloads, and (3) payload-derived
cadence did not fix it. Only full-logit Ring gather was stable; a single
262,144-wide top-k then exceeded five minutes, leading to the retained
full-logit gather plus four local-width top-k workaround for stochastic
sampling. Earlier in the same report, standard compact top-k Linear stopped at
writer line 260, full-logit force Linear stopped at line 119, and explicit axis
1 was refuted (`AUTOFIX.md:20-43`).

The upstream implementation may cross the old failure boundary because it
packs exactly one tile page per device, uses different gather dimensions for
BF16 values and int32 indices, constrains grids, and has Blackhole traced test
coverage. That is a testable hypothesis, not transferable proof: its evidence
is for a non-`None` Galaxy column axis with a barrier and split-subdevice
ordering, whereas this target is a default-axis physical TP4 Ring without that
barrier.

### Decisive hardware experiments

Run these later on TT hardware; none was run by this inspection:

1. Port upstream `3497b22f6ae` narrowly behind Qwen's greedy opt-in, adapting
   only the default-axis group size, Ring barrier policy, and worker grid.
   First run isolated eager/traced candidate tests before integrating it with
   the full model.
2. Compare full64 warmed traces in one process: current force argmax, existing
   compact Linear `Sampling1D` as a control, and the adapted upstream
   distributed argmax. Report sampler and combined token-out distributions,
   not one sample.
3. Profile the selected candidate and require no `[...262144]` all-gather and no
   full-vocabulary ArgMax. Record collective input shape, dtype, topology,
   links, workers, buffers, and semaphore/barrier policy.
4. Run safe Watcher on eager setup, trace capture, at least 100 replays, reset,
   and teardown. A passing uninstrumented timing is insufficient given the
   prior asynchronous writer stalls.
5. Compare device output with host greedy for random logits, real checkpoint
   logits, cross-shard winners on every device, equal-value ties (lowest global
   token must win), all-negative logits, batch 32, inactive rows, and the
   real-vocabulary boundary. Decode masks IDs 248320--262143 at
   `tt/model.py:152-160,231-235,672`, but the candidate must preserve that fact.
6. Re-run seeded stochastic top-k/top-p, penalties, greedy-to-stochastic and
   stochastic-to-greedy trace transitions, reset, and direct `tt_out_tok`
   feedback. This proves the greedy optimization did not change stochastic
   semantics or trace-slot ownership.

## 2. Teacher forcing: causal localization

### Comparable device work is stable

The full-model and optimized-full-model token-out artifacts give:

| Component | Prior | Current | Delta |
|---|---:|---:|---:|
| model trace | 42.140081 ms | 42.013429 ms | -0.126652 ms |
| sampler trace | 2.416293 ms | 2.416484 ms | +0.000191 ms |
| combined trace | 44.552574 ms | 44.427468 ms | -0.125106 ms |
| caller-visible autonomous decode | 49.204778 ms/token | 49.246321 ms/token | +0.041542 ms |

There is no material autonomous decode regression.

Teacher forcing gives:

| Derived component | Prior | Current | Delta |
|---|---:|---:|---:|
| teacher decode | 52.0268 ms/token | 63.9529 ms/token | +11.9261 ms |
| premium over caller-visible autonomous decode | 2.8220 ms/token | 14.7066 ms/token | +11.8846 ms |

The change in the teacher-only premium accounts for approximately 99.65% of
the total per-token regression. Prompt lengths differ between the autonomous
and teacher workloads, so this is localization rather than a perfectly paired
A/B, but both execute the same fixed-batch 64-layer and sampler traces and the
cross-stage autonomous control is exceptionally stable.

### Code path responsible for the premium

Every teacher step supplies `next_input`. In `tt/generator.py:682-729` the loop:

1. replays model and sampler traces;
2. reads the compact sampled token to host at lines 695-696;
3. calls the Python observer/callback at line 717;
4. allocates a fresh `torch.zeros([1,1,1,32], int32)` at line 722;
5. creates a fresh host-side replicated TT tensor and submits
   `copy_host_to_device_tensor` through `_copy_replicated()` at lines 724-729
   and 313-327.

Autonomous decode performs steps 1-2 but not steps 3-5. The optimized source
diff adds a deferred-read API and changes batch-1 prefill state handling; it
does not alter this `generate()` loop. Therefore the optimized model change is
not a code-backed explanation for the teacher-only slowdown.

The evidence cannot distinguish a regression in host-to-device copy/runtime
behavior from transient process/host variance. The prior validated artifacts
used `/home/ttuser/.local/lib/model-bringup/tt-metal`; the optimized stage used
`/home/ttuser/dev/tt-metal` at runtime commit
`9b415f82002af5d9040eca389d703690e405d91f`. The JSON records Python and Torch
versions but no TTNN version or runtime commit. Calling the two results an
optimization A/B is therefore invalid.

### TTFT is a separate cold-start confounder

`run_teacher_forcing.py:92-115,169-187` starts TTFT timing immediately before
`generator.generate()` and stops on the first `next_input` callback. That
interval includes the 161-token prefill, lazy program compilation/cache lookup,
host logits transfer, and first host sampling. It is not the warmed prompt-128
TTFT benchmark.

The final source change in `tt/model.py:510-546` reuses already-owned recurrent
state for batch 1 and avoids allocation, one-element concat, and copy for 48
linear-attention layers. Its same-process exact-shape warmed control improves
667.532346 to 646.847323 ms. The readiness TTFT increase of 9,222.026 ms has the
opposite sign and is hundreds of times larger than that measured code effect.
Different runtime trees and unrecorded JIT/cache state are the strongest
confounders; they are not proof of a particular runtime bug.

### Decisive teacher-forcing experiments

1. First record provenance in every JSON: source SHA, imported `ttnn.__file__`,
   TT runtime SHA, firmware, environment, cache path, JIT hit/miss counts, and
   whether the process is cold or warmed.
2. Under one fixed runtime, run at least three fresh-process readiness trials
   and report TTFT/decode median and range. Then repeat with the prior installed
   runtime using the identical repo source and environment. This runtime matrix
   decides runtime regression versus run variance.
3. In one process, instrument callback-to-callback intervals around: model+
   sampler replay, compact readback, Python callback, and ground-truth H2D copy.
   Because host/device execution is asynchronous, add explicit synchronization
   only at attribution boundaries and keep that diagnostic timing separate from
   production throughput.
4. Run equal-length controls after identical warmup:
   autonomous decode; `next_input` returning the prediction; and
   `next_input` returning the teacher token. The second isolates the feedback
   machinery without changing token choice; the third detects any
   token/content-dependent effect.
5. Benchmark the existing allocate-every-step feedback against a preallocated
   persistent host/device source tensor updated in place. If the approximately
   11.9 ms delta follows allocation/copy, that supplies the causal fix target.
6. For TTFT, use prompt length 161 for both policies, warm the exact physical
   length, alternate `reuse_prefill_state=False/True` order across repeats, and
   report medians/ranges. Report cold first-run readiness TTFT separately from
   warmed model TTFT.

## Conclusion

The greedy blocker has a definite source-level cause: the selected branch is a
full-vocabulary Ring gather followed by global ArgMax, while stage documents
claim an unimplemented compact split reduction. A compact semantic control
already exists and was measured, but its Linear collective is unsuitable; the
decisive implementation experiment is compact candidate exchange over a
Watcher-safe physical Ring while retaining `tt_out_tok` trace feedback and the
unchanged stochastic path.

The teacher regression is not attributable to the optimized model trace or
sampler trace. Static code plus measured controls localize the decode delta to
teacher-only host feedback and show the TTFT numbers are cold/runtime-confounded.
The available evidence does not justify naming a lower-level runtime bug; the
runtime matrix and boundary instrumentation above are required to close it.
