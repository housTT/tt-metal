# Stage Review

Verdict: more-work-needed

Independent initial audit of stage 03, optimized-decoder, on 2026-09-05.
This is an inspection of a live worktree on `hous/ornith-1.5-9b`, starting
at `bc8f514f3000da7b24c4d2b289b0ea507674e999`. It is not final signoff.
`D` below means this documentation directory; `R` means the model directory.

## Required Work

- **P2: Revalidate packed decode MLP under the final residual and down-projection layout.**
  Evidence: Raw `OPTIMIZATION_PAIR` rows in
  [packed_mlp_aligned_slices.log](logs/packed_mlp_aligned_slices.log) give
  629.385/458.367 microseconds for linear/full attention. The corresponding
  [initial integrated control](logs/all_dram_best_geometry_c64.log) gives
  628.192/456.406 microseconds. The rejection margins are only 1.193/1.961
  microseconds. Both runs used residual64 and down64/block6/readers2, as
  recorded in their provenance. The selected final path uses residual32,
  down48/block8/readers2, shared gate/up input conversion, and lower output
  movement. `PackedAlignedMLPCandidate` explicitly slices into the down
  projection's input shards; this part of the candidate is coupled to the
  changed geometry. None of the 145 paired rows available at the initial
  inventory measures packed decode MLP on that final combination.
  Why this matters: The earlier legal comparison establishes a narrow loss
  for its own layout, but does not establish the winner after changing the
  slice/activation/down boundary. This is a concrete optimization gap under
  the goal's best-candidate and cumulative-contract requirements.
  Required next step: Compare an adapted packed gate/up candidate against
  the actual final default, preserving final precision, residual32, down48,
  state/cache policy, and attention path. Include slices, activation, layout
  movement, exact restored trace, real-input PCC, and whole-layer timing for
  both kinds. The stage owner began preparing this control during this audit;
  preparation alone does not close it.

- **P1: Finish the known continuation repair and final optimized correctness gates.**
  Evidence: [final_default_short_v1.log](logs/final_default_short_v1.log)
  contains the continuation ELF/L1 failures and ends without a completed
  suite. The source and triage evidence in
  [AUTOTRIAGE_final_continuation.md](AUTOTRIAGE_final_continuation.md)
  identify accumulated leading-token L1 outputs and concat resource pressure.
  The previously named [batch Watcher run](logs/autofix_batch_watcher_final.log)
  has **1 failed, 8 passed**, including B31 allocation failure; its name is
  not evidence of final success.
  Why this matters: Arbitrary logical continuation, borrowed-input resource
  safety, B1–32 behavior, and native context are explicit preserved contracts.
  Required next step: Complete the already active repair, rerun the final
  real-weight contract/long/stress/Watcher checks, and record the final code
  hashes. Do not count older repaired-failure logs as passing final evidence.

- **P1: Complete the known runtime precision/geometry profile and performance accounting gates.**
  Evidence: The historical
  [BFP4 interleaved report](tracy/full_attention/bfp4_interleaved/decode_perf_report.txt)
  covers a different projection layout and decode context129. It is useful
  historical evidence, but does not prove final DRAM-sharded geometry. The
  [reader capture](logs/profile_readers_linear_attention.log) drops profiler
  markers and fails report processing, while
  [reader measurements](readers_profile_linear_attention.json) contain caught
  reader3 runtime errors. [AUTODEBUG_reader_profiler.md](AUTODEBUG_reader_profiler.md)
  correctly distinguishes those failures from successful pytest exit.
  Why this matters: Policy dictionaries and microtest records alone do not
  meet the explicit requirement for measured BFP4/LoFi runtime rows across
  material geometry/reader candidates, or final warmed prefill/decode tables.
  Required next step: Finish the already active reduced captures and reader
  adaptations; preserve complete runtime rows, actual dtype/fidelity/config,
  advice dispositions, and same-run device/host/roofline reconciliation.
  Complete the final README/checklist and update `doc/context_contract.json`
  for BFP8 KV and changed persistent L1 allocations, then obtain final review
  and record stage-owned local commits. These were already declared pending.

## Other Concerns

- **No material default performance mismatch was found.** Parsing the raw
  logs independently yields 145 paired rows from 98 run logs. For
  [final_default_pair_v1](logs/final_default_pair_v1.log), recalculated
  medians match [final_default_measurements.json](final_default_measurements.json):

  | Kind | Fused → default prefill ms | Fused → default traced decode ms | Minimum HF output PCC |
  | --- | ---: | ---: | ---: |
  | Linear layer0 | 26.132228 → 8.103750 | 1.462276 → 0.524046 | 0.99912857 |
  | Full layer3 | 23.489991 → 6.403622 | 1.263083 → 0.427555 | 0.99870509 |

  Final-default decode differs from `final_candidate_kda1024` by approximately
  +0.025%/-0.019%. The absolute smallest candidate medians in the inventory
  are 0.523677/0.427468 ms; their differences from the default are also below
  0.1%. These differences do not establish a material regression. The selected
  candidate also has the strongest measured prefill combination. The paired
  harness really captures/replays the optimized path, compares an exact
  restored eager output, and checks real HF outputs; it does not time state
  restoration or transfers.

- **Geometry coverage is substantive.** The two main and two extra geometry
  files contain 888 role measurements: 717 passes and 171 explicit runtime
  errors. All their recorded projection policies are BFP4/LoFi. For K4096,
  input cores4/8/16/32/64 include every positive block divisor of the local K
  shard, including block1 from the extra sweep. For down K12288, cores12/24/
  48/96 are also tested, with non-power-of-two legal blocks such as3/6/12/24/
  48/96 where divisible. The chosen role configurations match the micro
  winners, including down48/block8/readers2 at71.468/71.479 microseconds and
  gate/up64/block2/readers3. Smaller/wider working shards were actually tried;
  the final small gate/up block is not justified merely by inherited defaults.
  Runtime profiler confirmation remains the separate required gate above.

- **No synthetic-only precision veto was found.** The initial device-cast
  BFP4 attention loss was followed by original-host packing and real recorded
  input controls. The final real-input outputs pass0.995. Synthetic weights
  are explicitly diagnostic, and raw state/cache PCC is not substituted for
  cache-consuming output acceptance. The real activation files both match
  the hashes in their manifest. Their source is actual checkpoint embeddings
  and HF layers0–2; the long/batch tests transparently reuse recorded rows.

- **The prefill outN48 rejection needs precise wording, but changed logical
  width alone does not invalidate its L1 evidence.**
  `combined_output_prefill_b8_m8_n48.log` fails on packed `gate_up`, requesting
  1602560 bytes against1572864. Final separate projections change
  `per_core_N` from96 to48; the final cap32 resolves their output block to24.
  However, the 2D matmul factory sizes interleaved output/intermediate CBs by
  `out_block_h * out_block_w`, and input CBs by the input/output block sizes,
  not by `per_core_N` (factory lines143–173). Thus an unchanged outM8/outN48/
  K8 candidate retains the same dominant CB ledger. A source-backed exact
  blocker is acceptable; this audit does not require an identical failing
  allocation merely because the logical N changed.

- **Separate full-attention projections have meaningful measured controls.**
  Their micro minima total approximately112.24 microseconds for Q/K/V/gate,
  versus69.90 for packed QKVG; the whole-layer separate candidate loses
  26.819 microseconds. Its concat compatibility boundary should remain
  disclosed, but this is materially stronger rejection evidence than a first
  API error. This audit does not require rebuilding every head helper from
  that result.

## Hard-Check Gaps

- The final cumulative contract table, operation rows, advice dispositions,
  context allocation accounting, complete checklist, final review, and commit
  records are still pending. No final pass is inferred from the in-progress
  README or from prepared plans.
- The initial inspected optimized runtime SHA256 was
  `3ea11870e9b85ab93f604f06ba07ce56afa3291f9cb777fb254020fd1f635fa9`, exactly
  matching `final_default_pair_v1`'s archive. During this audit the owner added
  a continuation override; a later observed hash was
  `d09e535064b7d0f8c236b8eb619ab6bc308252625e22b62c755e46818a2d8f3b`.
  The v1 timing/PCC therefore remains verified evidence for its archived
  source, and must not be presented as final verification of later edits.

## Anomaly Ledger

- Observed anomaly: Packed MLP loses only1–2 microseconds on an older layout.
  Evidence: Packed/control pairs and resolved configs cited above.
  Affected path: Gate/up slice/activation/down decode boundary.
  Control or comparison: Both candidates correct on residual64/down64.
  Likely subsystem: Sharding and layout movement.
  Investigation performed: Raw pair extraction and candidate/final source comparison.
  Resolution: more-work-needed; final combined control is being prepared.
- Observed anomaly: Real BFP4 device conversion initially fails linear prefill,
  while original-host BFP4 packing passes.
  Evidence: `bfp4_attention_recorded` and `host_bfp4_all_recorded` raw logs.
  Affected path: Weight materialization and attention projections.
  Control or comparison: Real checkpoint weights and recorded inputs.
  Likely subsystem: Precision conversion/packing.
  Investigation performed: Inspected packing code, both controls, final real PCC.
  Resolution: controlled; final setup packs original host weights directly.
- Observed anomaly: Continuation ELF/L1 failure and subsequent host wait.
  Evidence: `final_default_short_v1` and the continuation triage report.
  Affected path: Public unaligned prefill continuation.
  Control or comparison: Owner's active isolated resource controls.
  Likely subsystem: Tensor lifetime/concat allocation and binary-cache failure state.
  Investigation performed: Source/log inspection; no hardware rerun by reviewer.
  Resolution: more-work-needed; active repair must pass final gates.
- Observed anomaly: Reader pytest exits successfully while profiler processing
  and some recorded reader3 candidates fail.
  Evidence: Reader raw log/JSON and AutoDebug report.
  Affected path: Reader-selection evidence and profiler completeness.
  Control or comparison: Other reader counts and planned reduced/drained capture.
  Likely subsystem: Profiler capacity and output-storage assignment.
  Investigation performed: Inspected source and recorded errors.
  Resolution: more-work-needed; active owner repair.

## Scope Inspected

- Goal/skill paths: `state/multigoal/03-03-optimized-decoder.prompt.txt`,
  `.agents/skills/{optimize,tt-device-usage,stage-review}/SKILL.md`, and relevant
  `tech_reports/LLMs/llms.md` material.
- Artifact paths: D README/work log, all145 raw paired rows, associated98
  provenance/source archives, geometry JSON, topology/prefill/state/precision
  plans, final-default summary, activation manifest/files, historical perf
  report/provenance, batch/final logs, and continuation/reader AutoDebug reports.
- Code paths: R `tt/optimized_decoder.py`, relevant fused/functional decoder
  methods, optimization candidate classes, shared and optimized test harnesses,
  projection/profile tests, evidence recorders, and relevant existing matmul/
  chunk-GDN factory source.
- Commands run: `git status --short`; `rg`, `rg --files`, `cat`, `sed`, and
  `nl` over the above paths; standard-library Python JSON/gzip/hashlib/
  statistics analysis. All98 paired-run log hashes, compressed archive hashes,
  and archived per-source hashes checked successfully. No TTNN import, device
  operation, server, build, benchmark, reset, or hardware test was run by this
  reviewer. The only authored file is this report.

## Residual Risk

The worktree changed during the review. This initial report evaluates the
specified archived evidence and identifies work; it does not attest to later
repairs, forthcoming captures, or final hardware correctness. This decoder
stage cannot generate text, so full-model qualitative/sampling/serving gates
are not applicable here.
