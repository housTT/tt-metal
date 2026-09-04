# Stage Review

Verdict: clean-pass

Independent reviewer, 2026-09-04. Scope: the initial functional-decoder stage for
`ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. Reviewed the live workspace on
`hous/ornith-1.5-9b`, based on `565b0aedf3`, before the stage checkpoint.
No devices, servers, resets, tests or hardware experiments were run by this
reviewer. The only reviewer-written artifact is this report.

## Required Work

- None. The two reporting/packaging findings raised during review were corrected
  by the stage owner and inspected again before this verdict.

## Other Concerns

- The four final filtered performance CSVs initially existed only as ignored
  local files. They are now staged, together with the rejected captures' filtered
  CSVs. Each final CSV's staged bytes match the SHA256 in `performance.json`.
  The human report links therefore survive the checkpoint. This finding is closed.
- The README initially claimed an unchanged allocation count across trace replay,
  although the tests establish stable buffers and replay correctness without
  asserting an allocation counter. The unsupported count claim was removed.
  The retained claim agrees with the inspected harness. This finding is closed.
- The default decoder/cache policy inspected and validated here is BF16, with
  FP32 recurrent state. Optional constructor dtype arguments are not evidence
  that lower-precision caches or weights have passed. Any later precision stage
  must validate its actual prefill-fill and decode-update dtype contracts.

## Hard-Check Gaps

- Native prefill evidence is a full decoder capacity/sanity run at 262143 and
  262144 tokens, plus last-128-token agreement between chunk sizes 2048 and 1024.
  It is not an HF comparison of every output at that length. Real-weight HF
  prefill/decode comparisons reach 8001 tokens; the separate native-position
  full-attention oracle uses exact synthetic BF16 historical KV and real layer-3
  projection/MLP weights. The README and context contract state these distinctions.
  No advertised native-context reduction is selected.
- Watcher reopens the device between cases. Its saved generated `watcher.log`
  therefore covers the last fixture, while suite stdout records earlier checks
  and the 23 passing outcomes. This limits the standalone generated log's scope;
  it does not contradict the clean watcher suite evidence.
- No independent runtime allocation counter or exhaustive long-context output
  readback is claimed. The code, guarded forwards, changed-input trace tests,
  complete replay records and saved correctness results satisfy this stage's
  evidence requirements without those additional instruments.

## Anomaly Ledger

- Observed anomaly: historical native-position HF/traced decode PCC 0.94535284.
  Evidence: `logs/native_decode_oracle.log.gz`, `AUTODEBUG_long_decode.md`,
  `AUTOFIX_long_decode.md`, `probe_long_decode.json`, diagnostic source and logs.
  Affected path: real-weight full-attention decode at position 262143.
  Control or comparison: six strict original-scale reruns in normal, watcher and
  cold-JIT processes report 0.99956169; calibrated final and watcher oracles
  report 0.99965012. The original RNG/scale/cast formula is restored explicitly
  in `recheck_long_decode_scale.py`. Twelve isolated scale/policy diagnostic rows
  pass, compare the actual TT query/cache with an FP32 attention oracle, check
  the final cache row and preceding row, and have bit-identical eager/replay output.
  Likely subsystem: historical cause remains unknown; the investigated candidates
  were SDPA precision, accumulation length, page/cache updates and trace lifecycle.
  Investigation performed: inspected both diagnostic scripts, raw strict logs,
  original failure, control metrics and the current unchanged attention path.
  Resolution: controlled in the current implementation, not a claimed numerical
  repair. The threshold and native context were not reduced. A recurrence still
  requires capture of the failing output/query/cache before changing the experiment.

- Observed anomaly: the full-extent DeltaNet mask-ramp slice freed its persistent
  source and broke prefill continuation.
  Evidence: `AUTODEBUG.md`, `AUTOFIX_ramp.md`, `logs/ramp_alias_probe.log.gz`,
  `logs/ramp_continuation_fix.log.gz` and the current `_gdn_gates` implementation.
  Affected path: padded linear-attention prefill with a reused 128-row ramp.
  Control or comparison: full-extent and partial slices had different ownership
  behavior; only the former deallocated the source. All four real-weight split
  continuations pass after the conditional-deallocation fix.
  Likely subsystem: TTNN tensor alias ownership.
  Investigation performed: checked the isolated address/allocation experiment,
  minimal retained fix and final continuation/watcher coverage.
  Resolution: fixed.

- Observed anomaly: continuation ending at an exactly allocated cache could
  request page-table end 65 when the physical table had 64 entries.
  Evidence: `continuation_capacity_geometry.json`,
  `logs/continuation_capacity_before.log.gz`,
  `logs/continuation_capacity_after.log.gz`, and
  `test_unaligned_continuation_to_capacity`.
  Affected path: full-attention prefill after an unaligned start offset.
  Control or comparison: context 4096, split 63 fails before the repair and
  passes afterwards at HF PCC 0.99951140; the same regression passes under watcher.
  Likely subsystem: 128-token physical prefill padding after only 64-token alignment.
  Investigation performed: re-derived the final padded extent and inspected the
  change that consumes a partial 128-token prefix through device decode operations.
  Resolution: fixed. Supplemental 11-test coverage includes both kinds' continuations
  and the expanded host-fallback guard; fresh-prefill/native-decode paths are unchanged.

- Observed anomaly: the earlier 32-replay decode profiler captures dropped markers,
  and the initial report command could reorder replay rows by capture timestamps.
  Evidence: `logs/profile_v2_*_decode.log.gz`, `performance_v2_rejected.json`,
  `tracy/*/rejected_v2/`, final v3 logs and report provenance.
  Affected path: measurement completeness and interpretation, not accepted latency.
  Control or comparison: final captures measure four warmed replays without dropped
  markers and use `--tracing-mode`. Sessions 5, 6, 7 and 8 each match the complete
  warm-template operation-ID sequence: 78 linear or 69 full-attention operations.
  Likely subsystem: profiler buffer capacity and report ordering.
  Investigation performed: independently parsed archived raw CSVs, signposts,
  replay sessions, filtered CSVs, source hashes and measured-output PCC logs.
  Resolution: fixed for accepted evidence; rejected timings remain explicitly rejected.

- Observed anomaly: environment/tool warnings include HF's missing fast-path
  libraries, unknown motherboard identification, opening a subset of MMIO devices,
  AICLK settling at 1343 MHz instead of requested 1350 MHz, and failure to copy the
  optional global Tracy GUI capture path.
  Evidence: final test/watcher/profile logs; local HF source; real per-run profiler
  paths in report provenance; complete ops CSVs and rendered tables.
  Affected path: HF reference selection, topology metadata, clocks and optional GUI export.
  Control or comparison: HF uses its inspected CPU oracle; guarded TT forwards pass;
  single-chip mesh runs and watcher pass; every accepted measurement has complete
  kernel records. All four accepted profiling runs record the same 1343 MHz warning.
  Likely subsystem: environment and collection tooling.
  Investigation performed: classified warning categories against code and artifacts,
  rather than treating the warning-free JSON summaries as sufficient.
  Resolution: controlled. These measurements describe one Blackhole chip on physical
  P300c hardware in the recorded clock regime; they establish no P150-board or
  remote-chip performance claim.

## Scope Inspected

- Goal/skill paths: model `AGENTS.md`; `.agents/skills/stage-review/SKILL.md`,
  `functional-decoder/SKILL.md` and `tt-device-usage/SKILL.md`; the supplied initial
  stage contract. Multi-chip, optimization, full-model, generation and serving
  stages are outside this verdict.
- Artifact paths: `README.md`, `work_log.md`, `../context_contract.json`,
  `hf_config.json`, layer-0/3 weight statistics, `provenance.json`,
  `performance.json`, `watcher_audit.json`, `evidence_integrity.json`, final and
  historical logs/provenance, diagnostic reports/scripts, raw/filtered profiler
  CSVs, human tables, and compressed watcher files under this evidence directory.
- Code paths: target `tt/functional_decoder.py`, `tt/model_config.py`, `tt/rope.py`,
  `reference/hf_reference.py`, both test modules and report/provenance tools;
  runtime `l2_norm_ttnn` and setup constant-tile helper; installed Transformers
  Qwen3.5 norms, DeltaNet, attention, MLP, rotary and decoder implementation.
  The copied 35B functional decoder's SHA256 matches its file at pinned commit
  `f7662055fe4ae3d66509335d96a7c74acd53911b`; its MoE path was replaced with the
  checkpoint's dense MLP in this implementation.
- Commands run: read-only `cat`, `sed`, `rg`, `wc`, `git diff`, `git show`,
  `git ls-files`, `git rev-parse`, `sha256sum`, and short standard-library Python
  scripts for JSON/CSV/gzip/hash analysis and reading the safetensors header.
  An attempted `git -C` lookup in the copied reference directory found no Git
  metadata; its content pin was instead checked against the fetched commit.
  No model/TTNN imports, pytest runs, hardware commands or implementation edits
  were performed by this reviewer.
- Source identity: `tt/functional_decoder.py` SHA256
  `bbc44e62d6045a15fb0c56060fffe91e264983874731f96ff1b187cce826bd30`;
  `tt/model_config.py`
  `678890d1334ad27808912dcf9b7cfe55704421d9a4ec6f53a21dcfb8abc21e77`;
  `tt/rope.py`
  `339780f27c5abfdc14224d1eed4b4ddb2e93dbfa1731a538a9dbbf6543d6fe70`.
  Current decoder hashes match the supplemental continuation, watcher and final
  profiler provenance. The earlier broad/long runs precede the documented narrow
  continuation fix; their passing results are supplemented rather than mislabelled
  as runs of an identical complete source snapshot.
- Re-derived checks: pinned snapshot/config JSON equality; all 14 layer-0 and 11
  layer-3 statistic keys/shapes match the real safetensors header; all 14 recorded
  run-log hashes match archived bytes; watcher hashes and suspicious-message scans
  match; final logs report 76 short, 9 long, 11 supplemental and 23 watcher passes.
  Both real and synthetic weight cases retain PCC >= 0.995. Changed inputs,
  positions and page tables are checked against per-user HF outputs from replay.
  Runtime guards cover fresh prefill, nonzero-start continuation and decode, with
  positive controls for Torch and transfer bans.
- Performance independently recalculated from accepted filtered CSVs and checked
  against raw nanoseconds: linear prefill 40.075804 ms, full prefill 35.104062 ms;
  linear traced decode 1.60966675 ms/replay, full traced decode 1.42231950 ms/replay.
  These are kernel sums, not dispatch-inclusive latency or generation throughput.
  Final measured-output HF PCC values are 0.99948959/0.99980134 for linear
  prefill/decode and 0.99949539/0.99902570 for full attention.

## Residual Risk

- The stage validates representative layers 0 and 3 at exact checkpoint dimensions;
  it does not validate stacked-model accuracy, every layer's weights or text quality.
- Native context is tested at batch 1; batch 4/32 coverage uses smaller per-user
  cache allocations. Aggregate batch/context capacity remains allocation-dependent.
- The layer caller owns valid disjoint page maps, in-range device positions and
  request-lane state lifetime. Scheduler isolation, cancellation and prefix caching
  require their later integration tests.
- Optional YaRN has HF frequency/table parity through position 999999. No
  million-token full-layer capacity, accuracy or serving capability is established.
- The original native-oracle failure is preserved and controlled by successful
  exact-input experiments, but its historical root cause was not recovered.
  This review does not convert that limitation into a claim of a proven repair.
- Local checkpoint commits and their recorded SHAs are the stage owner's next
  administrative step after this clean-pass. No push or PR is authorized by this report.
