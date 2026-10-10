# Stage Review

Verdict: clean-pass

Independent final review of Stage 01, functional decoder, for
`Qwen/Qwen3.5-35B-A3B`, completed on 2026-10-10 after the owner declared the
final evidence ready. No required correctness, capacity, trace, runtime-audit,
watcher, performance, or documentation gate remains open in the reviewed stage.

The reviewed live branch is `feat/qwen3-5-35b-a3b`, based on
`98dc4ae62ccfc32836ddcfad50f9fd97863612d4`. The checkpoint revision is
`59d61f3ce65a6d9863b86d2e96597125219dc754`. Stage files were untracked during
review; local checkpoint commits and their work-log SHAs follow this verdict,
as required by the stage-review workflow. This review authorizes no push.

## Required Work

- None.

## Other Concerns

- These are functional single-layer measurements. Eager prefill retains Python
  dispatch gaps, and the implementation executes batch rows separately. No
  optimized-decoder or full-model throughput claim is supported or made.
- Native-context HF prefill checks compare 16 selected output rows while retaining
  every recurrent or KV contribution. TT executes the complete layer for every
  token. The dense-HF control at length 1153 supports this reference decomposition;
  this is not exhaustive per-token numerical validation at native context.
- Roofline percentages use useful active-expert work and explicit compulsory
  traffic estimates. They are not measured physical DRAM utilization. The nominal
  120-core ASIC peak is disclosed alongside the actual 110 worker cores and mixed
  compute fidelity.

## Hard-Check Gaps

- No blocking gap remains for the stated stage contract. The final timing helper
  rejects incomplete device timestamps, and this reviewer independently counted
  all measured device rows and recomputed both whole-layer timing results.
- Future generator work must retain the documented preparation and stable-buffer
  contract when adding physical buckets, batch/slot bindings, or multiple live
  traces. The reviewed layer evidence does not stand in for later generator or
  serving trace-lifetime tests.

## Evidence Re-Derived

The checked-in target config exactly matches the pinned checkpoint config.
Source inspection of the installed HF implementation, remapping/loading boundary,
decoder, GDN, gated attention, router and sparse/shared experts confirms the real
2048-wide, 256-expert top-8 configuration and both layer kinds. The final router
selects real experts on device. The GDN recurrence follows HF's state-decay,
memory-read, delta, state-update and output-read order. Forward paths contain
device operations and host metadata planning, with tensor conversion confined to
construction and explicit harness boundaries.

| Evidence | Linear attention | Full attention |
|---|---:|---:|
| 4096-token prefill PCC | 0.999701798 | 0.998596847 |
| Minimum of 128 traced decode PCCs | 0.997864842 | 0.995969236 |
| 262144-token prefill PCC | 0.999611497 | 0.998373568 |
| Decode at position 262143 PCC | 0.999875247 | 0.998352349 |
| B32 heterogeneous-position minimum decode PCC | 0.998062253 | 0.996286869 |
| Whole warmed prefill, us | 1743870.348148 | 1720531.702222 |
| Mean whole traced decode, us | 2190.255232 | 1608.711806 |

The harness gates every individual decode token and row; the numbers above are
not only aggregate decode correlations. Both final profile runs reproduce the
headline PCC values and bit-identical repeated replay. Native-context and
262143-token nonaligned tests preserve the full advertised capability, with no
reduction. Real and statistics-matched synthetic device pytest cases pass for
both kinds; the saved weight-statistics files cover all 18/15 layer tensors.

The B32 test uses separate HF caches, distinct prefix lengths, reversed slots,
per-row positions, and eight traced steps. The B2 reuse matrices cover boundary
lengths, changed inputs/page ownership, multiple chunks plus tails, short-request
reuse, and continuation. Logical length 36 and continuation start 36 are withheld
from warmup in prepared physical buckets. The inspected runner captures once,
keeps that trace through all 17 requests, forbids new program-cache entries,
and releases only at teardown. The current-source full-attention reuse and exact
three-/nine-page pool regressions pass.

Only each trace's final output backing allocation is acknowledged as corruptible.
Its reshape views have no persistent-state consumers, and the output is consumed
before another replay. Persistent inputs, request state, caches, and snapshots
predate capture; temporary eager results are read and released before replay.
There is no broad allocation-scope exemption. Tracking-enabled acceptance commands
include program-cache allocations. Repeated replay, changed inputs, page-table
refresh, and eager/replay equality are backed by code and raw results.

Five watcher logs contain normal attach/dump/detach records and no matching
assertion, corruption, invalid-address, overflow, or error records. The four
device pytest cases and nine CPU contract checks pass. Watcher and profiling
were separate serialized runs.

Independent CSV inspection found 4866 measured prefill operations and 166 per
decode replay for linear attention; full attention has 867 and 101. Every one
of the 128 decode windows in each capture has complete firmware timestamps.
The reported interval is the earliest firmware start through the latest firmware
end, including all operations and internal gaps; decode is the mean of complete
replay intervals. It is not the sum of matmul durations.

Filtered report CSV row counts are exactly 4866/21248 and 867/12928 for
prefill/decode. Both human-readable tables are rendered tables, not export
boilerplate. Both lossless CSV archives decompress to the exact source CSV bytes,
and the hashes match `measurement.json`. Runtime rows confirm GDN BF16/HiFi4
projections, full-attention BFP8/LoFi projections, routed/shared BFP4 gate-up and
BFP8 down projections at LoFi, and the BF16/HiFi2 shared gate. The active-work
FLOP and compulsory-traffic formulas match those source policies. The physical
ASIC core/clock and per-card bandwidth basis agrees with the
[official P300c specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html)
and installed `tt_perf_report` phase rates.

All 20 final source/test hashes match `source_manifest.json`. The final profile
JSONs bind the current production source. Earlier native-context evidence remains
applicable: formatting is recorded separately, and the subsequent attention
adapter affects only odd tile capacities, which the native-context tables do not
use. The affected small/odd pools have current-source watcher regressions. The
final README, performance note, work log, context contract and telemetry packet
agree with the verified values and accurately leave only this review and local
commit bookkeeping pending.

## Anomaly Ledger

- Observed anomaly: Earlier GDN decode and unaligned-prefix cases failed PCC.
  Evidence: `linear_4096_128_per_token.json`, `positions_linear_b32.json`,
  `components_positions18*.json`, `AUTOFIX.md`.
  Affected path: Outer normalization and single-token recurrent state updates.
  Control or comparison: Exact-chunk, BF16-expert, router/norm/activation and
  native-recurrent controls; final SFPU recurrence versus the same HF inputs.
  Likely subsystem: Numerical precision at HF boundaries and carried recurrence.
  Investigation performed: Compared HF order with the final explicit FP32 SFPU
  recurrence and read the failed controls and passing headline/B32/native results.
  Resolution: fixed. The evidence supports the final path; it does not isolate
  every historical error to one instruction.

- Observed anomaly: An unseen logical length caused a live-trace program-cache miss.
  Evidence: `stage_review_initial.md`, `full_reuse_unseen.log`, final reuse JSONs.
  Affected path: Prefill slicing and continuation within a prepared input bucket.
  Control or comparison: Warmed 35 versus withheld 36 and withheld continuation
  start 36; retained trace ID through changed requests.
  Likely subsystem: Static slice/program signatures and persistent allocation safety.
  Investigation performed: Inspected resident indices, embedding reads, output
  writes, tensor chunk offsets and the cache-miss prohibition in the final runner.
  Resolution: fixed; both kinds pass the final 17-case reuse workload.

- Observed anomaly: Padded update tables rejected small pools; odd final pages
  could require an undefined extra page-table entry. The first adapter also used
  unsupported TTNN Shape slicing.
  Evidence: Initial review, `small_pool_before.log`,
  `small_pool_adapter_shape_failure.log`, `small_pool3_b1.json`,
  `small_pool3_b2.json`, `small_pool9_b2.json` and their watcher logs.
  Affected path: Full-attention tail prefill and paged decode.
  Control or comparison: Exact physical pools without guard/slack pages, covering
  positions through the last token of the odd final page.
  Likely subsystem: Cache ownership, SDPA read chunk size and Python Shape API.
  Investigation performed: Verified original tables drive updates, only chunk
  reads use padding, odd capacities select 32-token decode chunks, and dimensions
  are indexed individually; checked current-source PCC and watcher evidence.
  Resolution: fixed.

- Observed anomaly: The initial final linear profile dropped device markers.
  Evidence: `linear_profile_overflow.log`, rejected capture directory,
  `completion_commands.json`, accepted linear profile and processing provenance.
  Affected path: Profiler collection, not decoder mathematics.
  Control or comparison: Default 1000-program support versus 32768 with explicit
  between-pass drains, outside complete-layer windows and capture.
  Likely subsystem: Profiler buffer capacity.
  Investigation performed: Independently counted all accepted measured programs
  and timestamps, verified every replay window and lossless source archive.
  Resolution: fixed; no value is taken from the rejected capture.

- Observed anomaly: Full profiling rejected two active SDPA source markers sharing
  a 16-bit ID; the first runtime-root-only alias retained the original CWD path.
  Evidence: Both preserved collision logs/captures, `profiler_hash_diagnosis.md`,
  `profiler_path_alias.json`, final `profile_commands.json` and emitted zone log.
  Affected path: Profiling marker identities.
  Control or comparison: Final runtime/kernel path aliases and separate CWD.
  Likely subsystem: Source-location hashing and kernel path resolution.
  Investigation performed: Read the hashing/resolution source, verified the alias
  resolves to the exact checkout and unchanged kernel SHA/bytes, and independently
  rehashed all 31 actual final identities with zero collisions. Complete capture
  and postprocessing exit zero and preserve PCC.
  Resolution: controlled; the profiler-only path workaround removes no marker,
  disables no assertion, and changes no kernel math.

- Observed anomaly: All CSV `PROGRAM CACHE HIT` values are `False`.
  Evidence: Both accepted CSVs, `profile_integrity.json`,
  `op_profiler_json.cpp:174-180`, `op_profiler_serialize.hpp:380-384`.
  Affected path: Cached profiler metadata.
  Control or comparison: All 105 linear/84 full measured prefill program hashes
  already appear before the prefill start signpost; decode uses actual replay.
  Likely subsystem: A cached message retains its first call's hit flag.
  Investigation performed: Independently traced the stale-message source and
  checked both CSVs for new measured prefill hashes: none. Decode capture has
  program-cache misses forbidden.
  Resolution: controlled; this field is not used to establish cold execution,
  timing, or acceptance.

- Observed anomaly: Startup, shutdown and report-tool warnings remain in logs.
  Evidence: Accepted correctness/profile logs, watcher logs, pytest tail, and
  human-readable report headers; final README and performance note classify them.
  Affected path: Motherboard PCI fallback, subset-MMIO notice, memory-statistics
  shared memory, nanobind shutdown metadata, optional WASM trace copy, pandas
  mixed-column parsing, and report operation categorization.
  Control or comparison: Successful driver close and four passing pytest cases;
  five clean watcher logs; actual captured/exported files; complete measured row
  counts and hashes. The HF torch fallback message concerns the CPU reference.
  Likely subsystem: Environment discovery and optional diagnostics/reporting.
  Investigation performed: Read warning contexts, distinguished binding metadata
  from tensor/trace storage, verified raw exports and every measured timestamp.
  Resolution: controlled; no warning establishes an unclassified core-contract
  correctness or device-health failure in the accepted runs.

## Scope Inspected

- Goal/skill paths: Parent-provided original Stage 01 contract; workspace
  `../AGENTS.md`; installed `stage-review`, `functional-decoder`,
  `tt-enable-tracing`, `tt-device-usage`, and model-bringup startup/goal template
  under `/home/hous/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19`;
  checkout `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Artifact paths: This directory's README, contract, work log, initial review,
  AutoDebug/AutoFix, headline/native/position/reuse/small-pool JSON/log pairs,
  acceptance/completion/profile command records, four pytest results, five watcher
  logs, weight statistics, sparse-HF control, format/source manifests, profiler
  diagnoses/integrity, and both `tracy/{linear_attention,full_attention}` accepted
  CSVs, archives, measurements, rooflines, tables and processing provenance;
  `../context_contract.json`.
- Telemetry packet:
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/9c898890-7c79-49e8-a4b8-ff3c6515e5a9.json`.
  Both layer-kind metric dictionaries match the accepted whole-layer artifacts;
  workload is 4096/128, batch 1, concurrency 1, mesh 1x1.
- Code paths: All final autoport `tt/*.py`; layer/reference/position/reuse/audit,
  device/contract, profiling and roofline test helpers; installed HF Qwen3.5-MoE
  model source; inherited Qwen36 decoder, attention, MoE/shared-expert and weight
  loaders; relevant paged-update, attention, profiler and serialization source.
- Commands run: Read-only `cat`, `sed`, `rg`, `ls`, `git status/log/diff` and
  standard-library Python scripts for JSON/CSV comparisons, SHA-256, gzip identity,
  row/window counts, cycle-span calculations, warmup program-hash membership,
  source/config comparison and marker hashing; read of the official P300c page.
  This reviewer opened no TT device, imported no TTNN, ran no model/test/server/
  profiler/reset job, and changed no implementation file. The only review write
  is this report.

## Residual Risk

- Real target weights are exercised with seeded synthetic hidden activations;
  stacked-model routing/error accumulation and generated text are later-stage gates.
- Batch 32 and native context are validated separately, not their entire Cartesian
  product. New physical preparation signatures and downstream concurrent callers
  need their own applicable evidence.
- The accepted precision has finite PCC headroom, especially full-attention decode.
  Later fusion/precision changes must preserve these correctness gates.
- Timing is instrumented and includes whole eager-prefill dispatch gaps. The
  roofline traffic model omits extra reloads/temporary traffic and uses nominal
  hardware peaks; no full-model or serving extrapolation is established.

## Addendum: commit-hook formatting review

Date: 2026-10-10. Verdict remains **clean-pass**. Required work: none.

The commit hooks attempted to reformat historical source snapshots, trim raw
logs/tables, and reject required evidence larger than 500 KB. The parent restored
those artifacts and retained formatting/import cleanup only in current source.
I independently compared the staged reviewed source with the working files and
verified all 20 before/after SHA-256 entries in `commit_format_provenance.json`.
All four production `tt/*.py` files have identical ASTs. Across all 20 current
Python files, ASTs with import nodes removed are identical; inspection of the
remaining changes confirms import ordering and removal of unused imports only.
The target configuration differs only by its final newline and is JSON-identical.
No numerical operation, control flow, trace/cache behavior, or validation gate
changed.

I compared all 189 staged documentation/evidence files against their working
bytes. At the restoration check every file matched. The subsequent comparison
differs only in the disclosed README/work-log bookkeeping and updated
`source_manifest.json`; raw logs, historical snapshots, profile tables and
measurements remain unchanged. Both compressed accepted CSVs still decompress
to the recorded measurement hashes. The final source manifest matches all 20
current files, and the new provenance links them to the reviewed/measured bytes.

`precommit_source.log` records passing applicable repository hooks on current
source/configuration, including all six hooks excluded from the eventual
evidence-preserving commit invocation. `cpu_contract_commit_format.log` records
9 passing CPU contract tests. Its shutdown binding warnings are consistent with
the already classified nanobind metadata warnings in the anomaly ledger above.
The command-scoped exclusions for whitespace, EOF, Black, autoflake, isort and
file size preserve the original evidence; current source passed those checks
separately, and other applicable commit hooks remain enabled. This packaging
exception does not weaken a runtime or accuracy gate.

No hardware rerun is needed for this verified formatting/import-only change.
This addendum used read-only comparisons and standard-library inspection; this
reviewer ran no tests or hardware jobs and changed only this report. The original
scope limits and residual risks remain in force. Local checkpoint commits and
their recorded SHAs may now follow the clean-pass review.
