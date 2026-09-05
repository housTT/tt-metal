# AutoFix: duplicate fresh-prefill state reset

## Evidence and hypothesis

On B1, decode and prefill layer objects share the same hybrid buffers. Each of
the24 linear-attention layers has one recurrent buffer and three convolution
buffers; full-attention reset is a no-op. Warm `generate` called96 in-place
multiplies in `reset(clear_kv=False)`, then repeated those same96 updates in the
eligible traced `_prefill` branch. A host execution of the actual reset methods
confirmed identical target buffers and order:192 calls over96 unique buffers.
No cache update occurs between them on the warmed request path.

The hypothesis was that skipping only the second reset preserves fresh-request
semantics and removes measurable host dispatch. Moving reset work into capture
was not needed for this repair.

## Focused experiments

[Probe source](probe_prefill_reset.py) wraps each actual layer's `reset_state`
for one warmed request. Control executes both calls; candidate skips only call2.
Both arms require exactly two invocations per layer, unchanged four-trace IDs
and key, zero captures/misses/eager-prefill calls, and one prefill/first-sampler
replay. Scoped wrappers restore the original methods even after exceptions.

[v1 failure](logs/prefill_reset_full_v1.log) occurred before either comparison
arm: scalar temperature combined with a32-element seed list caused the common
formatter to nest that seed list, and seed initialization rejected it. This was
an invalid mixed-format fixture, not evidence against the reset hypothesis.
The corrected fixture uses32 temperatures and32 explicit seeds. Host execution
of the actual common formatter verified flat integer seeds17..48 for both modes.
The original failure log and source receipt remain intact.

[v2 results](prefill_reset_full_v2.json) and [exact command/provenance](logs/prefill_reset_full_v2.provenance.json)
record all32 actual-weight layers, B1/native262144 cache, four Blackhole chips on
P300c, selected decoder/head policy, and watcher/profiler disabled for timing.
All12 correctness cases pass:128/131, greedy/seeded penalties, changed tokens,
reversed page tables and return to the original inputs. Tokens, all-rank hybrid
state, final decode logits, feedback, RNG, positions and penalty histories are
exact. Five alternating timing pairs per length measure:

| Logical length | Control median TTFT | Skip-second median TTFT | Reduction |
|---|---:|---:|---:|
|128|31.455467ms|28.718005ms|2.737462ms|
|131|38.061322ms|35.215203ms|2.846119ms|

The full request retains its necessary first reset. Actual mesh multiply calls
fall from192 to96. Correctness snapshots are outside the request timer.

## Promoted change and validation

[Generator](../../tt/generator.py) adds private `_prefill(...,
state_already_reset=False)`. Only its traced B1 branch skips the local reset when
that argument is true. `generate` supplies true after its existing
`reset(clear_kv=False)`. Public prefill uses the false default; explicit reset,
eager capability paths, seed/penalty preparation, graph capture and model code
are unchanged. No persistent reset flag or allocation-tracker exception exists.

[Targeted host test](test_prefill_trace_contract.py) exercises both flag values,
requires exactly one reset with or without a prior caller reset, then verifies
that a subsequent public prefill resets again. Together with
`test_trace_lifecycle.py` and `../full_model/test_generator_host_contract.py`,
39 tests pass under pytest with the autoport `doc` directory as `--confcutdir`.
Black with target py310, compileall and `git diff --check` pass.

The current probe remains runnable on the promoted runtime: a scoped private
prefill wrapper forces `state_already_reset=False` in both comparison arms, then
the candidate alone skips the second layer reset. Both wrappers restore prior
method ownership in `finally`; host checks verify argument forcing and normal/
exception restoration. Original v2 source receipts and timings remain immutable;
this compatibility adaptation does not create new timing evidence. Final promoted-runtime
[full32 watcher gate](prefill_integration_full32_v2/summary.json),
[long exact controls](prefill_integration_long_v2.json),
[warmed performance](perf_prefill_trace_release_v2.json),
[teacher forcing](teacher_prefill_trace_release_v2.json),
[selected qualitative review](qualitative_prefill_trace_release_v2/qualitative_review.json)
and [profiles](tracy/README.md) all pass. The parent's immutable
[host receipt](logs/host_prefill_reset_v1.log.gz) records39 checks with
`--noconftest` and disabled plugin autoload. Final stage verdict remains owned by
[the independent review](STAGE_REVIEW.md).
