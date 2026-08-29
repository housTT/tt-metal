# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The `full_model` block in `doc/context_contract.json` still uses
  `"status": "full_model_validation_in_progress"` (`context_contract.json:156`-`context_contract.json:158`).
  This is not a correctness blocker for this review because `work_log.md`
  explicitly says independent stage review and the local checkpoint commit are
  pending (`work_log.md:192`-`work_log.md:196`). The stage owner should update
  that status during normal post-review handoff.
- `artifacts/profiler/token_out_perf_report.csv` exists and is small
  (177210 bytes), but repo `.gitignore` ignores `*.csv`. The required raw
  profiler evidence was correctly losslessly recompressed to
  `raw_ops.csv.xz` and validates against the recorded uncompressed SHA
  (`profiler_analysis.json:35`-`profiler_analysis.json:39`), so the ignored
  derived CSV is not a pass blocker. If the checkpoint commit intentionally
  includes that derived table, it must be added explicitly with `git add -f`.
- `git status --short` includes unrelated dirty/untracked files outside the
  full-model stage. I found no overlap that prevents isolating the full-model
  stage-owned paths for the required local-only checkpoint.

## Hard-Check Gaps

- No blocking hard-check gaps found. I did not rerun hardware-gated tests by
  instruction; this review re-derived their claims from existing artifacts,
  source, docs, and JSON/log evidence.
- No vLLM work was found in the autoport path. The full-model stage remains
  repo-local model/generator/test/documentation work.

## Evidence Summary

- Full-model architecture and policy are preserved. `tt/model.py` pins
  `openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, rejects
  non-fitting resident targets instead of falling back, and constructs every
  resident layer through `MultichipDecoder.from_state_dict(..., policy=policy)`
  (`tt/model.py:490`-`tt/model.py:561`, `tt/model.py:464`-`tt/model.py:480`).
  The default policy remains the optimized multichip policy: BFP8 attention/KV,
  BFP4 experts, BF16/selected CCL policy, LoFi decode projections, Ring
  topology, replicated residual contract, selected sparse geometry, and paged
  cache (`tt/multichip_decoder.py:70`-`tt/multichip_decoder.py:111`).
- The batch-2 sparse-MLP repair stays device-side. The current B>1 path slices
  each logical row, materializes row-major, retiles, zeros padding, and invokes
  the unchanged batch-one sparse expert graph; it does not add a host,
  single-chip, replicated-weight, or reduced-policy fallback
  (`tt/multichip_decoder.py:1524`-`tt/multichip_decoder.py:1559`).
- Full context/capacity evidence now covers the resident full stack rather than
  decoder-only accounting. The contract retains 131072 tokens, rejects P150 and
  P150x2 on resident byte counts, accepts P150x4 at 21,543,073,152 bytes/device,
  and records batch-10 full-context, batch-11 130880-token, and batch-32
  44992-token limits (`context_contract.json:163`-`context_contract.json:212`).
- Low-level serving state and non-aligned prompt handling are covered. The
  context contract records explicit cache/page/position/prompt/batch state,
  fixed slots, inactive-row sentinel `-1`, public non-aligned prompts, and
  host-compatible sampling only as an explicit boundary
  (`context_contract.json:214`-`context_contract.json:228`). The tests cover
  mixed 7/5-token prompts and inactive fixed-slot decode
  (`tests/test_full_model.py:256`-`tests/test_full_model.py:283`,
  `tests/test_full_model.py:407`-`tests/test_full_model.py:459`).
- Readiness accuracy meets the contract. Prefill is 94/100/100 and teacher
  forcing is 95/100/100 top-1/top-5/top-100 over 100 reference tokens
  (`prefill_readiness.json:1`, `teacher_forcing_readiness.json:1`; summarized
  in `README.md:24`-`README.md:33`). The failed teacher-forcing path was fixed
  through the caller-authoritative token boundary without changing free-running
  device feedback (`AUTOFIX_teacher_forcing.md:18`-`AUTOFIX_teacher_forcing.md:34`).
- The AIME24 reference provenance is exact enough for replay. The reference
  artifact hash is `7e722ad241eee84148ed62b5accee20bc642a4a1de4cab98ae146a166ee9d2bc`;
  provenance records the exact model revision, CPU/NVMe offload layout,
  prompt source/hash, chat-template flag, injected date, 214 prompt tokens,
  100 generated tokens, top-100 extraction, and EOS/PAD settings
  (`aime24_chat_100_top100.provenance.json:1`-`aime24_chat_100_top100.provenance.json:73`).
  The command file contains the executed Python here-document
  (`aime24_chat_100_top100.command.sh:1`-`aime24_chat_100_top100.command.sh:57`).
- Autoregressive output was inspected directly. HF and TT completions both give
  coherent English analysis of the same AIME walking-time equation
  (`hf_completion.txt:1`-`hf_completion.txt:7`, `tt_completion.txt:1`-`tt_completion.txt:5`);
  the TT path records 100 tokens, 51.6229 t/s/u decode, 99 trace replays, one
  initial refresh, 98 page-table reuses, zero forced refreshes, zero full-logit
  reads, and zero host argmax calls (`work_log.md:79`-`work_log.md:91`).
- The canonical split-greedy sampler path is accepted. Source selects
  `SamplingGenerator`, rejects `Sampling1D`, and records temperature-zero as
  top-k=1/top-p=0 greedy (`tt/generator.py:42`-`tt/generator.py:57`). The
  reduced 16-step comparison has exact device/host predictions, while the
  device arm records zero host argmax and zero full-logit readbacks
  (`split_greedy_host_comparison.json:1`-`split_greedy_host_comparison.json:74`;
  `sampler_decision.md:17`-`sampler_decision.md:26`).
- Tracing/token-feedback/page-state evidence is current. High-level device
  generation rejects untraced optimized token-out (`tt/generator.py:391`-`tt/generator.py:394`);
  free-running device generation deliberately leaves Python token input stale
  because the sampling trace writes `tt_out_tok` into the persistent input
  (`tt/generator.py:421`-`tt/generator.py:450`). The refreshed full 36-layer
  smoke records one setup token/position/page upload, zero steady-state uploads,
  four caller-visible scalar-token syncs, and zero validation full-logit syncs
  (`full_model_token_out_smoke.json:37`-`full_model_token_out_smoke.json:66`).
- Full36 batch-2 reproducibility is fixed and evidenced. The current artifact
  records full 36 layers, full 131072 context, distinct physical page rows, and
  bitwise exact prefill/decode logits across rows and reset/reuse runs, with
  zero max differences, zero differing values, identical raw hashes, and HF
  reference greedy tokens `200005` and `35644`
  (`logit_reproducibility.json:1`-`logit_reproducibility.json:130`). The
  external stale-fabric recovery is classified: unchanged source passed twice
  in fresh processes after bounded reset, and the exact acceptance gate was not
  relaxed (`AUTOFIX_full36_reproducibility.md:1`-`AUTOFIX_full36_reproducibility.md:54`).
- Same-format HF controls now cover the six shared qualitative prompts.
  Provenance records the exact checkpoint, command, tokenizer/chat rendering,
  greedy 128-token settings, and CPU/NVMe split (`qualitative_hf_tt_comparison.json:1`-`qualitative_hf_tt_comparison.json:74`).
  Every prompt-token sequence matches the TT control, with common-prefix counts
  38/3/3/17/3/26 and no visible drift/collapse (`qualitative_hf_tt_comparison.json:75`-`qualitative_hf_tt_comparison.json:130`;
  `qualitative/verdict.md:13`-`qualitative/verdict.md:38`). The EOS-tail
  anomaly was fixed by honoring the full HF generation stop set, and the
  degeneracy checker reports no findings (`AUTOFIX_qualitative_eos.md:23`-`AUTOFIX_qualitative_eos.md:36`;
  `degenerate_check.json:1`-`degenerate_check.json:80`).
- Profiler/readme claims align. The reduced profiler uses a real terminal
  stack plus one sliding and one full-attention decoder layer, measures a
  708.553 us LM-head matmul and 27.500 us `SamplingDeviceOperation`, and
  classifies sampling as 3.8812% of LM-head device time and 0.1420% of the
  19.3712 ms/token wall result (`profiler_analysis.json:1`-`profiler_analysis.json:14`;
  `README.md:114`-`README.md:127`). The 18+18 optimized-layer lower bound is
  recorded as 14.2698105 ms/token, leaving 5.1013895 ms/token as full-model-only
  residual with LM head, sampling, final norm/orchestration/sync/scalar
  readback/timer gaps separated (`profiler_analysis.json:15`-`profiler_analysis.json:33`;
  `README.md:129`-`README.md:138`). The raw profiler artifact is losslessly
  recompressed to `.xz` and validates to the recorded uncompressed SHA.

## Anomaly Ledger

- Observed anomaly:
  Earlier stage review found missing same-format HF controls for the six-prompt
  qualitative suite.
  Evidence:
  Current artifacts now include `qualitative_hf_chat.json` and
  `qualitative_hf_tt_comparison.json` with exact model revision, command,
  rendering, generation settings, and prompt-token equality for all six rows
  (`qualitative_hf_tt_comparison.json:1`-`qualitative_hf_tt_comparison.json:130`).
  Affected path:
  Qualitative generation evidence.
  Control or comparison:
  Pinned HF controls generated from the same token sequences and chat template.
  Likely subsystem:
  Evidence/provenance gap, not current TT output failure.
  Investigation performed:
  Parsed both HF and TT qualitative artifacts and read the comparison/verdict.
  Resolution:
  Fixed.

- Observed anomaly:
  The first qualitative translation run produced a post-EOS repeated control-token tail.
  Evidence:
  AutoFix records that the harness forced 128 tokens with `stop_on_eos=False`,
  then the corrected run stopped the translation at 81 tokens and the
  degeneracy checker returned no findings (`AUTOFIX_qualitative_eos.md:12`-`AUTOFIX_qualitative_eos.md:36`;
  `degenerate_check.json:1`-`degenerate_check.json:80`).
  Affected path:
  Shared qualitative suite.
  Control or comparison:
  Complete checkpoint EOS set `{200002, 199999, 200012}` and same-format HF
  controls.
  Likely subsystem:
  Qualitative harness stop condition.
  Investigation performed:
  Read AutoDebug/AutoFix reports, generator EOS source, corrected artifacts,
  and degeneracy output.
  Resolution:
  Fixed.

- Observed anomaly:
  Batch-2 sparse-MLP row handling and a later source-unchanged full36
  reproducibility failure were both investigated.
  Evidence:
  The row bug was fixed with device-side row materialization
  (`tt/multichip_decoder.py:1524`-`tt/multichip_decoder.py:1559`); the later
  full36 failure had finite HF-matching greedy tokens and cleared only after
  bounded external device recovery, with two clean-process exact passes
  (`AUTOFIX_full36_reproducibility.md:8`-`AUTOFIX_full36_reproducibility.md:43`).
  Affected path:
  Batch-2 decode/logit reproducibility.
  Control or comparison:
  Two-layer exact artifact plus full 36-layer exact artifact.
  Likely subsystem:
  Fixed row materialization bug, then stale external fabric/collective state.
  Investigation performed:
  Inspected source diff, two-layer/full36 artifacts, and AutoFix reports.
  Resolution:
  Fixed/controlled; no fallback or relaxed equality accepted.

- Observed anomaly:
  Raw profiler evidence previously exceeded the repo large-file hook threshold
  as gzip.
  Evidence:
  Current `raw_ops.csv.xz` is 254724 bytes and decompresses to SHA
  `ae8f52f15a246988a2e75c5de4e677e4446bd1af3c8871d1d423fca6071c583d`,
  matching `profiler_analysis.json:35`-`profiler_analysis.json:39`.
  Affected path:
  Profiler artifact packaging and local checkpoint readiness.
  Control or comparison:
  Lossless decompressed CSV hash and retained derived JSON/TXT report.
  Likely subsystem:
  Artifact packaging.
  Investigation performed:
  Decompressed and hashed the `.xz` artifact; checked file sizes and ignore state.
  Resolution:
  Fixed. Keep the raw evidence as `.xz`; do not keep the large gzip in the
  full-model commit.

## Scope Inspected

- Goal/skill paths:
  Original full-model contract supplied in the review task; `.agents/skills/stage-review/SKILL.md`;
  `.agents/skills/full-model/SKILL.md`; `.agents/skills/qualitative-check/SKILL.md`;
  repository `AGENTS.md`.
- Artifact paths:
  `doc/context_contract.json`; `doc/full_model/README.md`; `doc/full_model/work_log.md`;
  `doc/full_model/runtime_fallback_audit.md`; `doc/full_model/sampler_decision.md`;
  `doc/full_model/artifacts/*.json`; `doc/full_model/artifacts/autoregressive/*`;
  `doc/full_model/artifacts/profiler/*`; `doc/full_model/qualitative/*`;
  `doc/full_model/references/*`; `doc/full_model/autofix/*`;
  `doc/optimized_multichip_decoder/README.md`; `doc/optimized_multichip_decoder/performance_summary.csv`.
- Code paths:
  `models/autoports/openai_gpt_oss_120b/tt/model.py`;
  `models/autoports/openai_gpt_oss_120b/tt/generator.py`;
  `models/autoports/openai_gpt_oss_120b/tt/multichip_decoder.py`;
  `models/autoports/openai_gpt_oss_120b/tests/test_full_model.py`.
- Commands run:
  `sed`/`nl`/`rg` source and docs inspection; `git status --short`;
  `git diff --stat`; `git check-ignore -v` for profiler CSV packaging;
  Python JSON/hash validation over AIME, qualitative, trace, reproducibility,
  and profiler artifacts; `python -m py_compile` over model/generator/test; and
  `python_env/bin/pytest -q models/autoports/openai_gpt_oss_120b/tests/test_full_model.py`,
  which passed 14 host-side tests and skipped 10 hardware-gated tests. No
  hardware, server, or vLLM command was run by this review.

## Residual Risk

- Hardware-gated full-model correctness, trace, qualitative, and profiler
  claims were not rerun in this review by explicit instruction. The clean-pass
  verdict relies on the saved current artifacts and source inspection.
- The profiler is a reduced terminal-plus-two-layer capture for sampler and
  full-model-only cost attribution. This is sufficient for the stage contract
  because the full 36-layer token-out wall metric and optimized 18+18 layer
  lower bound are both recorded, but it does not isolate every microsecond of
  final norm/orchestration/synchronization.
- The stage owner still needs the normal post-review local checkpoint commit
  and work-log/status handoff. Do not push, and do not include unrelated dirty
  files in that checkpoint.
