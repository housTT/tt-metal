# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The selected teacher-forcing advantage over the refreshed baseline is small:
  22.611371 versus 22.567675 t/s/u (+0.194%). Two candidate repeats
  (22.614326 and 22.608416) and the separate default-path result (22.606825)
  are tightly grouped, so the measured ordering is supported, but cross-policy
  timing noise is less characterized than selected-policy repeatability.
- The selected token-out artifact records first/last generated token rather
  than the entire 128-token sequence. It is performance evidence; the separate
  six-prompt, exact-chat-template HF/TT suite provides the prompt-correct text
  quality evidence.
- The historical candidate cohorts were measured from uncommitted source whose
  exact patches were not preserved. The ledger now states that limitation
  honestly and gives row-specific behavior-equivalence arguments. The refreshed
  baseline is stronger: its measured source hash is retained separately from
  the final reviewed hash.

## Hard-Check Gaps

- `evidence/final/token_out_metrics.json` does not embed a precision summary.
  The benchmark source constructs the normal default `Generator` without an
  override, and the separate default-path teacher-forcing artifact embeds the
  selected config and exact selected artifact path. No code or artifact
  contradicts that construction path.
- `non_aligned.junit.xml` retains the passing test identity rather than its
  console transcript. The named test source exercises mixed logical lengths
  65/67, split trace capture/replay, changed-only page tables, and device
  sampling modes through the default selected policy.
- Runtime dtype/fidelity application is demonstrated by embedded resolved
  policy summaries, host propagation/strict-schema tests, and direct
  construction/operation code rather than retained profiler rows. These sources
  agree for every measured lowercase policy.

## Anomaly Ledger

- Observed anomaly: Case variants were previously validated but left raw for
  direct CCL and KV-cache consumers.
  Evidence: `PrecisionPolicy._dtype()` and `_fidelity()` now reject any spelling
  that differs from the canonical lowercase value. The retained host JUnit has
  19 passing cases, including `kv_cache.dtype=BFP8` and
  `ccl.linear_attention_mlp=BfP8` rejection cases; an independent in-memory
  probe also rejected both plus mixed-case fidelity. All selected and candidate
  artifacts use canonical lowercase values.
  Affected path: Precision override validation, CCL conversion, KV-cache
  allocation, and compute-fidelity dispatch.
  Control or comparison: Canonical selected policy resolves BFP4+LoFi for all
  six linear/full MLP role combinations and BFP8 KV/BF16-or-BFP8 role-specific
  CCL as recorded.
  Likely subsystem: Precision-policy schema boundary.
  Investigation performed: Inspected validator, direct runtime consumers,
  tests/JUnit, all candidate configs, and ran host-only rejection/resolution
  probes without opening a device.
  Resolution: fixed.

- Observed anomaly: The refreshed baseline predates the final canonical-case
  validation patch.
  Evidence: Baseline raw teacher-forcing and prefill metrics embed
  `baseline_optimized_mixed`, its absolute override path, and its completely
  lowercase resolved policy. `source_provenance.json` retains measured hash
  `6ab2b209...` separately from final source hash `72bf71adc0...`, marks the
  historical patch as not preserved, and states the exact validation-only
  difference. Recomputing the final seven-file hash yields `72bf71adc0...`.
  Affected path: Baseline source and command provenance.
  Control or comparison: The JSON/CSV row records the explicit baseline
  override command; raw teacher forcing is 0.97/1.00/1.00, 975.833597 ms TTFT,
  trace enabled, and 22.567675 t/s/u; raw prefill is 0.97/1.00/1.00.
  Likely subsystem: Measurement provenance.
  Investigation performed: Compared timestamps, raw resolved summaries,
  generated ledgers, builder logic, both source hashes, manifest cohort fields,
  and documentation.
  Resolution: controlled.

- Observed anomaly: The original canonical all-BFP8/HiFi2/BF16-cache policy
  failed traced warmup at an L1/static-CB collision.
  Evidence: The failure artifact records L1 floor 928000 and static-CB end
  1333760. AutoDebug reproduces the arithmetic; the one-field BFP4 linear-down
  repair reduces the endpoint to 811520. The repaired full-model candidate
  passes the original traced command at 0.98/1.00/1.00 and 18.257202 t/s/u.
  Affected path: Linear-attention MLP down during traced decode warmup.
  Control or comparison: `canonical_runnable_bfp8_hifi2_kv_bf16` preserves the
  remaining canonical fields and completes the workload.
  Likely subsystem: DRAM-sharded BFP8 weight/static-CB L1 footprint.
  Investigation performed: Inspected failure JSON, AutoDebug/AutoFix reports,
  repaired candidate resolution, and raw full-model metrics.
  Resolution: fixed.

- Observed anomaly: Every selected material MLP weight group uses BFP4, while
  the initial sweep lacked a clean same-weight HiFi2 control.
  Evidence: `all_mlp_bfp4_hifi2` matches the selected gate/up/down dtypes and
  every non-MLP field, changing only all six MLP fidelities. It passes at
  0.95/1.00/1.00 and 19.670285 t/s/u versus the selected LoFi
  0.95/1.00/1.00 and 22.611371 t/s/u.
  Affected path: Linear- and full-attention MLP compute fidelity.
  Control or comparison: Recursively resolved raw summaries match both configs;
  all results are traced full-model measurements on the same 100-token AIME24
  reference.
  Likely subsystem: Matmul compute fidelity.
  Investigation performed: Compared every selected weight/fidelity/non-MLP
  field, raw metrics, generated rows, ranking, and plots.
  Resolution: fixed.

- Observed anomaly: The selected accuracy trades two top-1 points for a small
  performance gain over baseline.
  Evidence: Selected is 0.95/1.00/1.00 and 22.611371 t/s/u; refreshed baseline
  is 0.97/1.00/1.00 and 22.567675 t/s/u. Both exceed the 0.90/0.98 gates, every
  ranked row is trace verified, and selected is the highest-throughput passing
  completed row. Both regenerated Pareto charts show all eight completed
  points, the non-dominated frontier, red selected point, and dotted threshold.
  Affected path: Final precision selection.
  Control or comparison: Selected repeats have 0.026% spread and the default
  selected rerun reports 22.606825 t/s/u.
  Likely subsystem: Full-attention down-weight precision versus normal timing
  variability.
  Investigation performed: Re-derived thresholds, means, ordering, CSV/JSON
  parity, trace status, and Pareto dominance; visually inspected both PNGs.
  Resolution: controlled.

- Observed anomaly: Qualitative completions stop inside visible reasoning at
  the 64-token cap.
  Evidence: All six HF and TT outputs use identical rendered chat prompts and
  exact checkpoint/tokenizer metadata. HF exhibits the same reasoning style and
  truncation; TT/HF matching prefixes range from 11 to 63 tokens, all automatic
  degeneracy checks pass, and manual review finds no wrong language, prompt
  echo, leakage, malformed controls, or mechanical loops.
  Affected path: Selected-policy prompt quality.
  Control or comparison: Exact-checkpoint HF outputs for every shared prompt.
  Likely subsystem: Base-checkpoint style plus short generation cap, not TT
  precision or token feedback.
  Investigation performed: Read prompt metadata, rendered prompts/token IDs,
  all HF/TT outputs, degeneracy report, and manual verdict.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: supplied datatype-sweep goal contract;
  `.agents/skills/datatype-sweep/SKILL.md`,
  `.agents/skills/stage-review/SKILL.md`, and
  `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: README/work log and all four prior reviews; selected and all
  candidate configs; JSON/CSV ledgers and artifact builder; source-provenance
  manifest; every baseline/candidate/final raw metric or failure artifact; all
  four final JUnits; qualitative metadata/HF/TT outputs/checks/verdict; both
  Pareto PNGs; AutoDebug/AutoFix reports; `doc/context_contract.json`; prior
  optimized-full-model token-out evidence; and the 100-token AIME24 reference.
- Code paths: `tt/precision.py`, `tt/model.py`, `tt/generator.py`,
  `tt/multichip_decoder.py`, `tt/optimized_decoder.py`,
  `tests/test_precision_policy.py`, relevant full-model non-aligned/token-out
  tests, and both readiness-runner diffs.
- Commands run: read-only `git status/diff/diff --check`, `find`, `rg`, `sed`,
  `jq`, `stat`, JSON/CSV/config/path/hash/ranking analysis scripts, host-only
  policy probes, AIME reference inspection, and visual inspection of both PNGs.
  No TT device, server, vLLM, reset, reservation, or hardware experiment was
  run.
- Scope isolation: No vLLM integration exists or was added. Stage-owned changes
  are isolated from the disclosed unrelated Tracy/UMD/cluster-descriptor dirty
  state and can be checkpointed separately.

## Residual Risk

- The performance margin over baseline is modest, but the repeat and default-
  path evidence is internally consistent and the user contract selects the
  fastest evaluated passing traced teacher-forcing policy.
- Historical dirty candidate patches cannot be reconstructed byte-for-byte;
  their limitation and behavior-equivalent final changes are explicit. The
  final checkpoint will make the selected implementation durable without
  relabeling those measurements.
- Future vLLM integration must preserve the normal `build_generator` /
  `QwenFullModel` default construction path, the 262,144-token shared cache
  contract, and the post-selection token-out measurement regime. That work is
  correctly outside this stage.
