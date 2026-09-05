# AutoFix: integrated prefill trace lifecycle

The integrated candidate passes exact whole-model and reduced-layer device
gates against explicit `use_prefill_trace=False`. Watcher and trace allocation
tracking pass with all 32 real layers; reduced-layer long generation also
passes through repeated output-history windows. This follows the measured
dispatch-gap hypothesis and standalone
exactness experiment in [AUTODEBUG_prefill_gaps.md](AUTODEBUG_prefill_gaps.md)
and [AUTOFIX_prefill_gaps.md](AUTOFIX_prefill_gaps.md).

The experiment tests the integrated four-trace lifetime rather than only the
standalone device body. It keeps the selected precision, common sampler,
logical prompt lengths, page-table contract, and native 262144-token B1 cache.
The candidate runs first, then a fresh owned-cache eager-control generator
shares the same model weights. No two generators or trace sets coexist.

## Completed quick and edge gate

Parent ran `prefill_integration_quick_v2`, exit 0, September 5, 2026,
20:58:14–20:58:56 UTC. It uses real layers 0/3 on four Blackhole chips,
`TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_DISABLE_ETH=1`, and
`TT_METAL_TRACE_ALLOC_TRACKING=1`; the device profiler is disabled.

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_TRACE_ALLOC_TRACKING=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.optimized_full_model.record_run \
prefill_integration_quick_v2 timeout 600 python_env/bin/python \
-m models.autoports.ornith_ai_ornith_1_5_9b.doc.optimized_full_model.probe_prefill_integration \
--quick --edges --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/prefill_integration_quick_v2.json
```

All 16 paired cases pass exact comparisons:

- Ten public prefill cases: changed tokens at 128, reversed physical pages,
  reset-driven 128→131→128 eviction, live 131 miss followed by resident 128
  reuse, exact lengths 1/2048, and return to 128. Complete logical logits,
  recurrent/convolution state hashes on every rank, next-decode logits/state,
  and sampled next tokens match the eager control. Each public device result
  owns a separate buffer; freeing it before decode preserves trace outputs.
- Two live `configure_sampling` cases enable seeded sampling and then penalties
  with explicit histories while the prefill trace remains resident. Model
  state, token feedback, positions, and prefill key survive recapture. The
  following decode scores/state/tokens match the eager control.
- Four four-token generation requests cover greedy, seeded sampling, penalties,
  and returning to greedy. Complete outputs, final logits, and hybrid state
  match. Each candidate request uses one prefill replay and one first-token
  sampler replay; the control uses eager prefill.

The assertions release and rebuild all four traces, then prove zero allocated
trace bytes after public teardown in both lanes. Plain candidate trace storage
is 2,424,832 bytes/device at 128 and 2,752,512 at 2048, compared with 1,703,936
for the eager control's decode/sampling traces. Changed tokens skip an unchanged
prefill page-table copy; reversed pages issue exactly one copy. Initial 128
first-use captures twice because seed preparation warms new programs; warmed
same-key requests reuse the inputs and trace spec. No tracker suppression is
used. Watcher timings are not performance evidence.

Artifacts: [full results](prefill_integration_quick_v2.json),
[compact summary](prefill_integration_summary.json),
[run provenance](logs/prefill_integration_quick_v2.provenance.json), and the
archived source snapshot beside the provenance. Both lanes retain full CPU
reference tensors in the `prefill_integration_quick_v2_*_expected.pt` artifacts.

## Completed whole-model gate

`prefill_integration_full32_v1` ran with `--quick --full32`, the same
Watcher/tracker environment and 600-second timeout, September 5, 2026,
20:59:21–21:01:18 UTC; exit 0. All **13 paired cases** pass: seven public
prefills at logical 128/131, two live sampling reconfigurations, and four
four-token generation requests. This repeats the changed-token/page,
128→131→128 eviction, live shape miss/reuse, owned output/deallocation,
and exact subsequent decode checks with every real layer and the native
262144-token B1 cache. Every rank's recurrent/convolution state hashes,
complete logical logits, and generated tokens match the eager control.

All four trace IDs release and rebuild correctly, and both lanes end with
zero allocated trace bytes. Candidate trace storage is 24,313,856 bytes/device
at 128 and 28,311,552 at 131; the plain eager control uses 15,073,280.
Every candidate generation request has one prefill replay, one first-token
sampler replay, and zero eager prefill calls. This run did not select `--edges`;
its exact whole-model prompt lengths are 128/131.

The checkpoint preserves the [whole-model summary](prefill_integration_full32_v1/summary.json),
complete [traced lane](prefill_integration_full32_v1/traced.json.gz), and
[eager-control lane](prefill_integration_full32_v1/eager_control.json.gz), plus
[provenance](logs/prefill_integration_full32_v1.provenance.json) and
[source snapshot](logs/prefill_integration_full32_v1.sources.json.gz).
The 3.2 MB complete raw JSON remains workspace-only; the compressed lanes and
summary reproduce its full contents without dropping per-rank state evidence.

## Completed long-generation gate

`prefill_integration_long_v1` ran without `--quick`, with the same cache and
600-second timeout, September 5, 2026, 21:03:40–21:04:05 UTC; exit 0.
Watcher, allocation tracking, and the device profiler were unset for this run.
All **16 paired cases** pass on real layers 0/3, including the ten public
prefill cases with exact lengths 1/2048 and the two live configurations.
The four generation requests produce 8 greedy tokens, 128 seeded-sampling
tokens, 128 penalty-sampling tokens, and 260 tokens after returning to greedy.
Complete output sequences, final logits, and hybrid state match exactly.

The 260-token request executes 259 history replays and three history readbacks
in each lane, crossing two full 128-row history windows. The 128-token sampled
and penalty requests each execute 127 history replays and one readback; their
first token comes from prefill. Each candidate request uses one prefill replay
and one first-token sampler replay, with zero eager prefill calls. Both lanes
again release all trace storage to zero. These are correctness and lifetime
checks; this report makes no performance claim from their timings.

Artifacts: [long results](prefill_integration_long_v1.json),
[provenance](logs/prefill_integration_long_v1.provenance.json), and
[source snapshot](logs/prefill_integration_long_v1.sources.json.gz).
The [compact summary](prefill_integration_summary.json) includes all three
runs, exact comparison counts, generation/history counters, trace storage,
commands, environments, and provenance verification.

## Provenance and scope

Host-only verification checked all 91 archived source hashes in each run,
each compressed snapshot hash, and each decompressed log hash against its
provenance. The current `model.py`, `generator.py`, integration probe, and
both native binaries match all three runs. The whole-model checkpoint's
raw-report hash is
`62d87e4765c5fb23e2c4988879e790109e0e38a73a7091f0e40580020e6174eb`;
its top-level summary and both compressed lanes match the workspace report
exactly.

These gates check the populated attention cache through the following exact
decode; they do not hash the entire native KV allocation. Exact lengths
1/2048 and 128/260-token generation are covered by the reduced-layer runs;
whole-model exactness here covers lengths 128/131 and four-token generation.
Parent owns separate native-capacity, broader model correctness, and
uninstrumented performance gates. No integration-gate evidence remains pending
within this report's stated scope.

Only the stage probe and this report were authored in this validation subtask.
Runtime integration belongs to the implementation agent. Black, Python
compilation, and deferred-import `--help` passed before hardware handoff; no
C++ build is required for these Python/documentation additions.
