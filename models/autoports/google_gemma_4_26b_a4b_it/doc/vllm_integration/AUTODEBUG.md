# AutoDebug: Gemma 4 vLLM evidence closure

Verdict: **more-work-needed — evidence preservation and final independent review.**
No new Gemma serving implementation defect was established by this inspection.

Inspection date: 2026-09-08. This is the fresh, isolated AutoDebug entry step
for AutoFix. No device, server, reset, profiler, or implementation edit was run.
The only test executed was the host-only adapter contract suite below.

## State inspected

- Original contract: `bringup/artifacts/multigoal-runs/gemma4-26b-a4b-p150/09-09-vllm.prompt.txt`;
  AutoFix, AutoDebug, vLLM Integration, and Qualitative Check skill requirements.
- tt-metal initially clean at `c718255087b`, including implementation commit
  `40569292be814f72c62ef30fd06c8b8245957517`; vLLM initially clean at
  `2f81f493b969da7ce3cd64c0f6bc1895b8f229cb`.
- Inspected the changes since tt-metal `5e18deb453d` and vLLM `5ffebf412`,
  the stage README/work log, Sep 6/8 artifacts, adapter, shared runner,
  plugin sampling eligibility, and oracle/adapter tests.
- The coordinating agent corrected baseline hash references and qualitative
  comparison wording during this investigation. Those shared-workspace edits
  are distinguished from outstanding findings below.

## Required finding: server lifetimes do not match the main gate artifacts

For every profile, `readiness_vllm/<profile>/server.log.gz` is byte-for-byte
identical to `server_logit_determinism_final_20260908.log.gz`. Each contains
exactly 26 POST requests, matching the final determinism checker workload.

| Profile | Retained generic server lifetime, Sep 8 UTC | Earlier final gate evidence |
| --- | --- | --- |
| P150 | 17:05:44–17:08:05 | full sampling finished 12:36:54; benchmark logs 12:41–12:42 |
| P150x2 | 17:09:09–17:14:19 | full sampling finished 13:08:08; features/benchmarks 14:41–14:44 |
| P150x4 | 17:18:07–17:23:10 | full sampling finished 15:01:15; overlap/benchmarks 15:47–15:48 |

The special P150x2 `server_full_sampling_20260908.log.xz` begins at
12:43:29 and preserves that sampling lifetime. The corresponding later
feature/qualitative/benchmark lifetime is not the generic log. Sep 6 logs
cannot substantiate Sep 8 reruns. No matching main-gate server lifetime for
P150/P150x4 was found in the model evidence directory.

This does not refute the client results: each local `sampling_tests.log`
reports **72 passed, 1 skipped**, in 577.19/629.87/701.04 seconds, respectively.
It does prevent the retained generic logs from proving the claimed runtime
configuration, crash/fallback audit, and cleanup for those earlier checks.
`run_vllm_server.py:253` opens the fixed `server.log` in `wb` mode, so reusing
the same output directory for another `--stages serve` replaces its lifetime.

The three final sampling logs and six benchmark logs are also ignored by
`.gitignore:7` (`*.log`) and absent from the recorded local commits. They exist
in this workspace, but a checkout of those commits does not retain the log
triplets promised by the README.

Focused verify/fix sequence:

1. Recover exact original server files from existing archives/run records,
   if available. Verify their startup/shutdown timestamps and API process IDs
   against the corresponding check invocations; retain an explicit mapping
   from each result to its server lifetime. Do not rename a determinism log
   into purported main-gate evidence.
2. Preserve the existing sampling and benchmark logs losslessly in tracked
   `.gz`/`.xz` files, and update artifact links. No sampling rerun is required
   merely to repair their ignored-file status.
3. If a required lifetime cannot be recovered, rerun only its affected checks
   on the unchanged implementation, with the documented profile config, into
   a fresh output directory. P150x2's preserved full-sampling lifetime need not
   be repeated to repair the separate later lifetime. For P150/P150x4, the
   missing lifetime covers full sampling as well as qualitative/benchmark
   evidence. Keep `--sampling-profile full`, device `top_k=1/32`, concurrency
   32, advertised context, and both benchmark workloads unchanged.
4. Use a unique output directory for each new server lifetime, or archive the
   old log before the next launch. A small host-only runner test can verify
   that an existing lifetime is not silently truncated if the runner is fixed.
5. Update README/work-log mappings and obtain a fresh independent stage review;
   archive its verdict, then locally commit stage-owned repairs. No push.

Read-only reproducer for the proven mismatch, from the model directory:

```bash
for profile in P150 P150x2 P150x4; do
  sha256sum readiness_vllm/$profile/server.log.gz readiness_vllm/$profile/server_logit_determinism_final_20260908.log.gz
  gzip -cd readiness_vllm/$profile/server.log.gz | awk 'NR == 1 {print} /POST/ {n++} /09-08/ {last=$0} END {print last; print "POST count:", n}'
done
git check-ignore -v readiness_vllm/P150/sampling_tests.log readiness_vllm/P150/vllm_benchmark.log
```

## Prior findings now supported or corrected

- **Device qualitative coverage:** all 18 profile/prompt entries explicitly
  record greedy `top_k=1` and sampled `top_k=32`, matching runner requests and
  the plugin's supported unseeded device policy. Older `topk64_host_compat`
  artifacts are correctly labeled separately. The 36 retained completions
  were read: coherent English/task-appropriate French, no mechanical loops,
  doubled tokens, gibberish, or visible request contamination. This assessment
  still needs the corresponding main-gate server provenance above.
- **Prompt/control/degeneracy evidence:** all 18 rendered prompts and token-ID
  lists exactly match the same-suite HF/standalone-TT control. Prompt source,
  control-output, and control-assessment SHA256 values match. Model revision,
  tokenizer class, chat-template decision, endpoint, and generation settings
  are recorded in `qualitative_control_comparison.json`. All three degeneracy
  artifacts have zero findings and exit code 0. HF control is greedy/64 tokens;
  serving is greedy and sampled/256 tokens, so comparisons support coherent
  prefixes and appropriate continuations, not exact sampled-text equality.
- **Overstated qualitative completeness:** the initial control comparison
  claimed every output explained all three laws and both learning modes.
  Some outputs truncate during the second law or before the unsupervised
  explanation. The coordinating agent corrected those two descriptions during
  inspection. No generation or numerical fix was justified by that wording bug.
- **Full sampling:** the earlier missing/full-vs-smoke finding is refuted by
  the three current 72-pass logs. Some tests exercise explicit host compatibility
  for unsupported features; this is allowed and is not device-sampler proof.
- **Numeric vLLM determinism:** each profile now has an actual two-position
  comparison against a profile-local real-weight 30-layer standalone B1
  oracle. Prefill argmax is the decode input in both paths. Retained B1
  selected-logprob/common-top-10 deltas are zero with 10/10 overlap; repeated
  runs and duplicated prompts in different concurrent positions are also
  exact. This is the labeled optional host-logprobs diagnostic path, not the
  performance sampling path. The live determinism test submits eight requests
  to a physically padded B32 graph; 32-request capacity is demonstrated by the
  distinct full-sampling/CI burst evidence.
- **Stale standalone hashes:** initially every determinism artifact referenced
  an obsolete standalone file hash. Every consumed B1 numeric/metadata field
  and TP4 B32 control summary was compared to the current files with `jq`;
  there were zero content mismatches. The coordinator repaired these links.
  Current oracle hashes start `41538a4f` (P150), `702e821e` (P150x2), and
  `148ec593` (P150x4); all references now agree. A new hardware run was not
  necessary to repair this reference-only discrepancy.
- **Allocator-warning inference:** the old warning-count argument is removed.
  `tt_metal/impl/allocator/allocator.cpp:124` emits at most once per host
  thread for its lifetime, so absence of another warning never proved safety.
  Current documentation instead cites trace release/recapture, stable buffer
  addresses, page-table refresh counters, and stale-feedback checks.

## Residual scope and other concerns

- The TP4 standalone B1/B32 control records raw/centered cosine minima
  0.8968/0.9587 and probability total variation up to 0.2706. Selected tokens
  remain equal for the four tested synthetic prompts, and same-group B32
  positions/compositions are deterministic. This localizes the observed
  distribution movement to the standalone batch-shape path; it does not prove
  general B1/B32 numerical equivalence or correctness on arbitrary prompts.
  No new serving regression or kernel cause was established. Do not present
  these diagnostic deltas as zero-error batch invariance.
- The mutable-buffer device artifact identifies itself as standalone traced
  generator evidence. Adapter boundary behavior is additionally covered by the
  host mock tests and live overlap output checks; avoid relabeling the
  standalone artifact as direct hardware execution of the vLLM adapter.
- Ancillary shared-runner issue: `_run_qualitative_prompts` calls
  `tokenizer.apply_chat_template` at lines 417–418 before the API exception
  handler. A tokenizer with no template raises local `ValueError`, bypassing
  the intended raw-completion fallback. Gemma's template is present, so this
  does not explain its evidence. Focused follow-up: mock a no-template
  tokenizer and both API clients, then verify raw completions and correct
  metadata; move the template decision before rendering if reproduced.

## Verification performed

```bash
python_env/bin/pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_adapter_contract.py
```

Result: **20 passed, 3 warnings in 4.71 seconds**. Additional checks were
read-only git/status/diff inspection, archive decompression, SHA256 comparisons,
and `jq` equality checks described above. The report is not a substitute for
the goal's required independent `stage-review` clean pass.
