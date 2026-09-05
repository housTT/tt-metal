# AutoFix: R22 QKV split watcher assertion

## Starting evidence and hypothesis

`AUTOTRIAGE.md` records the original default-R22 watcher failure, the repeated
failure after reset, and a passing residual-disabled control. The reported
kernel is the width-sharded QKV decode reader; it increments the input shard
coordinate after the final V tile and accesses a runtime coordinate outside
the exact coordinate table.

Source inspection confirmed that
`NLPCreateQKVHeadsDecodeDeviceOperation::select_program_factory` selects the
interleaved factory solely from `!input_tensor.is_sharded()`. The sharded reader
at `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:244` performs its next
coordinate lookup without a final-tile guard. The prediction was therefore
that an L1-interleaved QKV boundary would remove the watcher failure without
changing the sharded projection math or head output memory config.

## Scoped fix and regression

`OptimizedDecoder._attention_decode` now converts the residual-path QKV matmul
output with `ttnn.sharded_to_interleaved(..., ttnn.L1_MEMORY_CONFIG)` immediately
before head splitting. It immediately deallocates the consumed sharded QKV
allocation. The existing post-split release of the normalized projection input
and interleaved QKV allocation remains in place for batch-32 L1 capacity.
The head splitter still receives the original height-sharded output config.

The optimized oracle wrapper now intercepts the actual head-split call during
R11/R22 attention execution and asserts exactly one split call, an unsharded
input, and `ttnn.L1_MEMORY_CONFIG`. This regression runs within the existing
real-weight/HF, trace, and performance tests; it does not merely inspect source.
No C++ or files outside the optimized decoder/tests/docs scope were edited.

Frozen hashes for these focused experiments:

- Decoder: `2f12e858b7e0398aad0ade0571f1df3644aeea6647d1055d3d8be9b6eda05a5a`.
- Tests: `d51ac1637f73a4668ba9c8d74d03f82913b025f0cca240c7126ea123e39b4f34`.

## Recovery

No stale pytest/device process remained at experiment start; no processes were
killed and no locks were cleared. All hardware commands were serialized.

```bash
timeout 60 tt-smi -ls --local
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
timeout 60 python_env/bin/python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print('MESH_SMOKE_OK')
PY
```

All commands exited 0; both listings showed four Blackhole P300C devices and
the smoke printed `MESH_SMOKE_OK`. No second reset was needed. The final
`timeout 60 tt-smi -ls --local` after verification also exited 0 and showed
all four devices. These recovery checks are infrastructure evidence.

## Focused verification

The following commands are executable reproductions. The evidence commands
also set `GEMMA4_OPT_EXACT_COMMAND` to a descriptive label; its
`TTNN_CONFIG_OVERRIDES=fallback-exceptions` shorthand means the literal JSON
configuration shown here.

```bash
TT_METAL_WATCHER=10 GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_OPT_CANDIDATE_ID=autofix_qkv_interleaved_watcher_sliding \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'real_weights_prefill_decode and sliding_attention' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/candidate_runs/autofix_qkv_interleaved_watcher_sliding.xml

TT_METAL_WATCHER=10 GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_OPT_CANDIDATE_ID=autofix_qkv_interleaved_watcher_full \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'real_weights_prefill_decode and full_attention' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/candidate_runs/autofix_qkv_interleaved_watcher_full.xml
```

Results: sliding `1 passed in 6.92s`; full natural/shared cache
`2 passed in 12.96s`. Watcher and fallback exceptions were enabled, device
profiling was disabled, and all tests/process teardown completed cleanly.
The last watcher log had no assert/error/illegal/hang/timeout match.

| Layer/cache | Prefill PCC | Decode PCC | Required |
| --- | ---: | ---: | ---: |
| Sliding/shared | 0.9990979106 | 0.9995082026 | 0.995 |
| Full/natural | 0.9986184755 | 0.9997957033 | 0.995 |
| Full/shared view | 0.9986184755 | 0.9997957033 | 0.995 |

PCC is unchanged from the earlier R22 measurements. The candidate JSON/XML
pairs with the command candidate IDs above retain measurements and provenance.

## Performance cost and legal control

Watcher was disabled for both unprofiled timing commands:

```bash
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1000 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_OPT_CANDIDATE_ID=autofix_qkv_interleaved_perf_batch1 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'test_optimized_decoder_perf_profile and batch1' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/candidate_runs/autofix_qkv_interleaved_perf_batch1.xml

GEMMA4_OPT_RESIDUAL_SHARD_CORES=0 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1000 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_OPT_CANDIDATE_ID=autofix_qkv_nonresidual_perf_batch1 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'test_optimized_decoder_perf_profile and batch1' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/candidate_runs/autofix_qkv_nonresidual_perf_batch1.xml
```

Both commands passed both layer kinds, in 11.20s/11.16s respectively. Latencies
are mean warmed traced decode over 1,000 replays at sequence position 1024:

| Layer | Prior unsafe R22, ms | Fixed R22, ms | Added cost, µs | Fresh legal nonresidual, ms |
| --- | ---: | ---: | ---: | ---: |
| Sliding | 0.930676 | 0.940449 | 9.77 | 1.106792 |
| Full | 1.023211 | 1.032024 | 8.81 | 1.142292 |

The repaired path remains 15.0%/9.7% faster than the legal nonresidual candidate.
The prior unsafe R22 rows are historical cost controls from
`candidate_runs/final_frozen_r22_perf.json`, not passing final candidates.
New candidate JSON/XML pairs use the command IDs above. Fixed-path warmed
prefill was 96.711580 ms sliding and 108.053257 ms full, consistent with the
unchanged prefill path.

## Verdict and handoff

The scoped hypothesis is **verified**: the original watcher failure disappears
under the predicted factory selection; both attention kinds and both full
cache layouts retain their PCC. Keep the seven-line decoder workaround and
the runtime regression. `pre-commit run --files` on the decoder and test file
passed all applicable checks. Python-only changes require no C++ build.

The upstream sharded-reader defect is still present outside this stage's scope.
The extra conversion is necessary for a legal decoder path in this checkout.
The main stage must rerun its complete watcher/stress and batch-32 gates and
regenerate authoritative final profiler/provenance reports after this source
change. The final timing command was the nonresidual control, so its top-level
timing JSON is a candidate result until the main stage reruns default timings.
No commit or push was made by this hypothesis experiment.

## Context-capacity transport and allocation accounting — 2026-09-05

Starting evidence: the preserved [captured triage](triage/context_capacity_tt_triage.txt)
and its [compact summary](triage/context_capacity_summary.txt) describe the
262143-token capacity-prefill stall. `cq_prefetch` waited for tagged pinned-host
NoC reads while `cq_dispatch` waited for the DRAM-write payload; no active model
operation or allocator OOM was captured. This is separate from the historical
R22 watcher failure above. The original command was:

```bash
GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 \
GEMMA4_PREFILL_CAPACITY_LENGTH=262143 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'optimized_advertised_context_traced_decode or optimized_prefill_capacity_probe' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_context_capacity.xml
```

The main agent ran fresh-process sliding-capacity controls with the same
262143-token allocation policy. The unpinned control added
`TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`; the pinned repeat omitted it.
Both selected `-k 'optimized_prefill_capacity_probe and sliding_attention'`.
The saved [unpinned XML](candidate_runs/context_capacity_unpinned_control.xml)
reports **1 passed in 40.90s**; the
[pinned repeat XML](candidate_runs/context_capacity_pinned_repeat.xml) reports
**1 passed in 41.38s**. Matching JSON files retain candidate provenance.
The transport event was not reproducible in these controls. Pinning as a
repeatable cause is unproven, so this investigation made no model runtime fix
or pinned-memory workaround. These two passes do not validate the full-attention
capacity gate or explain the original transport event.

The independent accounting hypothesis was **verified** by an AST inspection of
the original test: its resource groups omitted
`decode_packed_expert_gate_up_batch32`, `batch32_expert_gate`,
`batch32_expert_up`, and `decode_attention_weights_batch32`. Constructor source
confirms that default B1/B32 packed weights use distinct BFP4/BFP8 allocations,
while B32 attention aliases `decoder.weights.qkv/o_proj`. The missing B32 packed
buffer is `128 * (2816 / 32) * (1536 / 32) * 1088 = 588251136` bytes (561 MiB).
Adding it to the recorded sliding/full totals yields 2116256768/2111524864
bytes. The later final device accounting run confirmed those values from live
buffer page counts and sizes.

The focused fix extracts the existing accounting loop into
`_persistent_weight_buffer_accounting`, includes all batch-specific groups,
and retains deduplication by `buffer_unique_id`. It skips unallocated tensors:
the shared BFP8 packed policy can retain wrappers for released gate/up source
buffers. Three host regression cases cover default distinct packed buffers,
shared BFP8 packing, and unpacked B32 gate/up. Separate attention wrappers
sharing buffer IDs contribute zero duplicate bytes, while a padded B1 QKV
copy is still counted. The device accounting test also asserts the actual
B1/B32 sharing policy and attention alias IDs when it next runs.

Host verification:

```bash
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'persistent_allocation_accounting_host or optimized_precision_defaults' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/candidate_runs/autofix_accounting_host.xml
pre-commit run --files \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py
```

Results: **4 passed in 1.67s**; all applicable pre-commit checks passed.
An in-memory AST mutation removing the newly enumerated groups made the default
host regression reject the missing B32 packed entry, confirming that it detects
the original omission. This accounting subtask ran no hardware commands and
did not edit `optimized_decoder.py`. Python/tests/docs only; no build needed.
The main stage subsequently ran the live accounting cases and the complete
context-capacity gate. Five selected accounting cases passed, and all four
context cases passed in 183.11s. Preserve the original transport evidence if
the stall recurs; the unpinned and fresh pinned controls did not reproduce it.

## Independent review remediation — 2026-09-05

The first independent stage review returned `more-work-needed`. The required
AutoFix loop attempted the skill's local fresh-context launcher; its bubblewrap
setup failed before analysis, so a fresh xhigh subagent performed the same
inspection and wrote the temporary repository-root `AUTODEBUG.md`. That report
identified five testable hypotheses: H1 actual BFP8 cache propagation and
numerical consumption, H2 missing expert block44/fair-separate coverage, H3
missing coherent R22 DRAM-sharded and packed-dense adaptations, H4 incorrect
prefill-route modeling plus untested attention/routing movement advice, and H5
stale evidence identity/prose. The report was integrated here and the temporary
root copy removed.

AutoFix tested each hypothesis in isolation before retaining changes:

| Hypothesis | Evidence | Result |
| --- | --- | --- |
| H1 BFP8 cache | nonaligned real-weight prefill followed by cache-consuming eager/captured/replayed decode; sliding 1025 and full 33 natural/shared | deterministic, but minimum PCC 0.994624887/0.988673202; reject BFP8, accept BF16 only |
| H2 expert geometry | packed block22/block44 and fair separate accurate-GeGLU correctness plus 1,000-replay timing | block44 ties inconsistently; separate 0.893834/0.915895 ms is about 5% slower; keep packed block22 |
| H3 R22 adaptations | Original `reviewfix_r22_*` matrix | invalidated by the final loader-dispatch audit below; the requested flags were discarded and counters prove these paths were unused |
| H4 prefill attention | G8x4/G8x8 role isolation, cumulative gates, and full O input-L1 | full QKV and selected sliding candidate fail downstream fused-equivalence PCC; correct O variants regress; reject |
| H4 routing movement | row-major zero base and scatter metadata with direct sparse consumption | correct and trace-stable; removes two untilizes and one unary; select |
| H5 evidence | regenerated canonical timing, full suite, context, watcher, serving, profiler, advice, allocation, and hashes | stale BFP8/review/timing prose replaced |

The selected routing change adds a 256-byte stable runtime buffer. The
allocation helper was extended to enumerate it rather than silently exclude
it; five accounting cases pass, and final per-layer totals are
2,116,257,024/2,111,525,120 bytes. This last test-only enumeration changed the
test hash after the long context/watcher/Tracy runs, but not the decoder source
or any tested runtime path. The complete suite and canonical performance were
rerun under the final test hash.

The candidate files beginning `reviewfix_` and the final `*_reviewfix_v2/v3`
XML/JSON files preserve failures as well as accepted results. The final
loader-dispatch audit below supersedes the original H3 conclusion. No other
AutoFix hypothesis remained untested, and no proposed change was retained
without both the relevant correctness gate and whole-layer evidence.

## Final loader-dispatch correction

Independent rereview showed that `FunctionalDecoder.from_state_dict` accepted
but did not retain the optimized-only `r22_dram_sharded` and
`r22_packed_dense_gate_up` flags. The original `reviewfix_r22_*` runs therefore
allocated candidate weights without executing the intended consumers; their
zero counters make them invalid as performance or correctness evidence. The
repair restores both requested flags after the fused loader returns, records
the public logical decode batch before internal tile padding, and makes tests
assert requested/default dispatch counters from environment intent.

The replacement `reviewfix2_r22_*` matrix proves real dispatch for
one/two/three-reader QKV, O, separate dense, and packed dense candidates. It
selects reader-1 DRAM-sharded O for sliding attention and reader-1
DRAM-sharded packed BFP8/LoFi dense gate/up plus down for both layer kinds. QKV
reader 1 was locally faster for sliding attention but fails direct fused decode
PCC at 0.987622. Full O reader 1 initially hit an L1/CB overlap; the adapted
legal block-8 version passes correctness but loses at 0.876961 ms. Readers 2
and 3 also lose. The selected sliding O plus packed-dense composition passes
direct fused decode at PCC 0.996685 and reproduces without candidate overrides
at 0.811773 ms; the full default reproduces at 0.859455 ms.

The setup-only interleaved packed-dense source is released after DRAM sharding.
Final live allocation is 2,158,283,008 bytes for sliding and 2,130,482,432
bytes for full attention, projecting to 64,609,487,360 bytes for the 25/5
layer mix. The exact-hash v5 complete, performance, context, watcher, serving,
allocation, direct-fused, and Tracy artifacts are indexed by
`final_manifest.json` and `work_log.md`. This correction is the authoritative
AutoFix closure for H3; the original unused-path files remain only as defect
evidence.
