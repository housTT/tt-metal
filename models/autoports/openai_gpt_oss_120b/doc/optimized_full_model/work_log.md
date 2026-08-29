# Optimized-full-model work log

## Scope and starting point

This stage started from completed full-model commit
`5058406ce09eb40a1148b91d4f0f7cc485b1115e` and optimized the complete
`openai/gpt-oss-120b` model/generator path on the P150 family.  The pinned
checkpoint is
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.  The measured resident target is
P150x4 `(1,4)`; P150 `(1,1)` and P150x2 `(1,2)` remain in the mesh/capacity
contract but cannot host fixed resident state in 32 GiB/device.

Work covered embedding, final norm, all 36 optimized multichip decoder layers,
LM head, logits, canonical sampling, KV/cache/page state, CCL, trace replay,
token/position/RoPE feedback, request lifecycle, mixed prompts, fixed slots,
inactive rows, and generator orchestration.  vLLM and broad datatype Pareto
search were not started.

Hardware commands were serialized through `scripts/run_safe_pytest.sh`.
Profiler and watcher runs were always separate.  Before recovery-sensitive final
runs, ownership was checked, boards were reset with the Tenstorrent venv's
`tt-smi -r all`, and enumeration was confirmed with `tt-smi -ls`.

## Preserved policy

The selected optimized-multichip policy remained locked:

- logical BF16 replicated residual; decode L1, prefill DRAM;
- BFP8 attention weights and paged local-head KV cache;
- LoFi decode projections, HiFi2 prefill QKV, LoFi output projection;
- BFP8 decode-attention CCL, BF16 prefill-attention and expert CCL;
- BFP4/LoFi active-expert weights and BF16 router;
- 45-core gate/up, 15-core decode down, 45-core prefill down;
- TP4 physical 2944 attention reduction sliced to logical 2880.

No rejected dtype/fidelity/CCL/activation program, replicated stream, dense
all-expert path, or full-vocabulary logits stream was selected.

## Implemented path

- Added a fixed-length split token-out loop that sets sampling state once,
  queues model and sampler trace replays nonblocking, and reads only the first
  and final tokens.
- Preserved canonical `SamplingGenerator` logits shards, `tt_out_tok` feedback,
  device-side position/RoPE advance, changed-only page tables, greedy semantics,
  and top-k/top-p request state.
- Added counters for model/sampling replay, refreshes, page reuse, host argmax,
  logit/token reads, output collection, prefill variant compilation, and trace
  lifecycle.
- Made unseen padded-prefill compilation release live decode/sampling traces at
  a synchronized request boundary and recapture them after compilation.
- Rounded physical KV/page allocation to the 128-token SDPA decode K chunk while
  retaining logical non-aligned prompts and 64-token pages.
- Kept physical dirty-cache clear on request reset; the logical-only candidate
  did not repair reuse behavior.
- Added a behavior-preserving LM-head seam and profiled the selected head against
  an eight-way DRAM-sharded candidate.
- Guarded unused all-gather scatter-state initialization when the actual one-page
  path uses unicast, resolving the fully enabled watcher assert in multicast
  primary/alternate and standard-unicast writers.

## Candidate and rejection ledger

| Candidate | Evidence | Disposition |
| --- | --- | --- |
| Generic force-argmax/full-vocab gather | 139.1412 versus 154.2817 t/s/u, 32 exact tokens | rejected, 9.81% slower |
| Eight 8192-column DRAM-sharded LM-head matmuls | split 2342.560 versus 2335.916 us; prefill 29852.084 versus 29698.179 us; +191.25 MiB/device | rejected |
| 16384-column LM-head split | 2,229,248 B static CB request versus 1,572,864 B available | rejected physical limit |
| Logical-only cache invalidation | full-stack reuse remained incorrect | rejected |
| Extra model-to-sampler event ordering | divergence moved but remained upstream; candidate reverted | refuted |
| Relax batch-2 exactness to semantic top-k agreement | semantic diagnostic also failed; all relaxation code reverted | rejected; bitwise gate retained |

## Correctness incident and `$autofix`

An all-36-layer run became nondeterministic without a source change.  Repeated
synchronous sequences and a control before any unseen-prefill release diverged;
the real sampler-ready device logit buckets differed, while the sampler returned
the correct maximum for each capture.  The isolated sampler matrix remained
exact, localizing the symptom upstream of sampling.

The `$autofix` loop reproduced and tested hypotheses in isolation.  A bounded
physical board reset recovered the unchanged source, identifying stale external
fabric/CCL state as the observed recovery condition.  The strict batch-2 gate
then passed all rows and runs bitwise (`588e9848...`), with zero differing values.
Failed and semantic-diagnostic logs are retained as recovery provenance, not as
accepted gates.

The final resident acceptance command was:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_SLOW_DISPATCH_MODE \
  GPT_OSS_120B_FULL_MODEL_ACCEPTANCE=1 \
  GPT_OSS_120B_FULL_TRACE_LIFECYCLE_ISOLATION=1 \
  GPT_OSS_120B_FULL_ASYNC_ISOLATION_LENGTHS=7,8,122,128 \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k real_weight_36_layer_full_context_token_out_smoke -s
```

It passed in 391 seconds.  Repeated synchronous runs, the pre-release control,
and split endpoints 7/8/122/128 are exact; the unseen prompt bucket caused one
safe trace release.  The same run recorded warmed prompt-214/output-100 split
token-out at 3.635881 s TTFT and 62.766489 t/s/u, plus prompt-128/output-128 at
0.484781 s and 62.879214 t/s/u.  Log:
`artifacts/final_full_stack_acceptance_prompt214_gen100.log.gz`.

## Accuracy and qualitative commands

```bash
GPT_OSS_120B_FULL_MODEL_READINESS=1 \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k run_prefill_check_aime24_top100 -s

GPT_OSS_120B_FULL_MODEL_READINESS=1 \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k run_teacher_forcing_aime24_top100 -s

GPT_OSS_120B_FULL_MODEL_AUTOREGRESSIVE=1 \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k run_autoregressive_aime24_hf_and_tt -s

GPT_OSS_120B_FULL_MODEL_QUALITATIVE=1 \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k shared_qualitative_chat_suite -s

python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/openai_gpt_oss_120b \
  --missing-artifacts critical --scope autoregressive
```

Results:

- prefill top-1/top-5/top-100 = 94%/100%/100%;
- teacher forcing = 95%/100%/100%, TTFT 3747.043 ms, decode 52.306593 t/s/u;
- autoregressive = 100 tokens, TTFT 3.746314 s, decode 52.161699 t/s/u,
  no degeneration finding;
- six shared qualitative prompts, identical chat-template prompt IDs, no
  degeneration finding.  The pinned HF control is referenced by hash and the
  optimized HF/TT comparison is informational.

## Serving-contract and sampling commands

The reduced real-weight full-path probe was the main fast qualification:

```bash
GPT_OSS_120B_FULL_MODEL_PROBE=1 \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k real_weight_two_layer_full_model_split_sampling_probe -s
```

It covers one sliding and one full-attention layer, embedding, norm, LM head,
greedy, top-k/top-p, feedback, persistent state, token-out, async output, and
teardown.  Additional focused gates:

```bash
GPT_OSS_120B_FULL_SAMPLER_TRACE_ISOLATION=1 \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k tp4_sampling_trace_isolation -s

GPT_OSS_120B_FULL_MODEL_PROBE=1 GPT_OSS_120B_SAMPLER_BENCH=1 \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k real_weight_two_layer_full_model_split_sampling_probe -s

GPT_OSS_120B_FULL_MODEL_PROBE=1 GPT_OSS_120B_SAMPLER_BENCH=1 \
GPT_OSS_120B_FORCE_ARGMAX_CANDIDATE=1 \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k real_weight_two_layer_full_model_split_sampling_probe -s
```

The eight-case sampler matrix is exact.  The top-k/top-p artifact uses
temperature 0.8/top-k 20/top-p 0.9 and proves trace replay plus `tt_out_tok`
identity.  Mixed prompt/fixed-slot/inactive-row evidence is in
`artifacts/mixed_prompt_fixed_slots_inactive_row.json`.

## Profiler commands and conclusions

Each final-source phase used:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_SLOW_DISPATCH_MODE \
  TT_METAL_PROFILER_MID_RUN_DUMP=1 GPT_OSS_120B_PROFILE_DRAIN=1 \
  GPT_OSS_120B_FULL_MODEL_PROFILE=1 \
  GPT_OSS_120B_FULL_MODEL_PROFILE_PHASE=<prefill|teacher_forcing_decode|split_token_out_decode> \
  GPT_OSS_120B_LM_HEAD_POLICY=<interleaved|dram_sharded> \
  scripts/run_safe_pytest.sh --profile \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k real_weight_two_layer_optimized_full_model_profile -s
```

Raw reports were rendered twice—once with `--csv` and once as the advice table:

```bash
/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report \
  --start-signpost <start> --end-signpost <end> \
  --active-experts 4 --no-color --no-summary \
  --csv <phase>/tt_perf_report.csv <phase>/raw_ops.csv
```

Five final-source phases pass.  Selected windows are 29698.179 us prefill,
2339.117 us teacher forcing, and 2335.916 us split token-out.  The DRAM-sharded
candidate is 0.518% slower in prefill and 0.284% slower in split, so the
interleaved head is retained.  In selected split, the 707.269-us LM head is
30.278% of the window; the full 351.678-us sampler chain is 15.055% and does not
dominate.  Complete hashes and commands are in
`artifacts/profiler/final_source/provenance.json`.
The raw profiler CSV inputs are losslessly committed as `raw_ops.csv.xz`; all
other generated `.log` and `.csv` artifacts are stored as `.gz`.  Stored and
uncompressed hashes are recorded in provenance so hooks cannot rewrite raw
evidence and every file satisfies the repository's 500 KB gate.
Pre-commit also normalized only trailing padding/final newlines in the five
human-readable `tt_perf_report.txt` tables and EOFs in text/JSON artifacts;
provenance and the manifest record the normalized committed hashes.

The 36-layer optimized decoder lower bound is 14.2698105 ms.  Adding the
selected LM head and complete sampler gives 15.3287575 ms; measured split
token-out is 15.9035067 ms, a conservative 0.5747492 ms / 3.7495% residual.

## Watcher and CCL repair

The first fully enabled sampler watcher run tripped a line-279 NCRISC assert in
the sampler's UINT32 all-gather.  Source inspection showed that a one-page
packet selected unicast but still initialized unused scatter state below the
scatter API's minimum chunk count.  `if constexpr (use_scatter_write)` now
guards the multicast primary/alternate and standard-unicast scatter setup.

Focused UINT32 eager/trace all-gather passes with PCC 1.0.  The final model path:

```bash
env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_WATCHER=1 \
  GPT_OSS_120B_FULL_MODEL_PROBE=1 \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k real_weight_two_layer_full_model_split_sampling_probe -s
```

passed with watcher disabled features `None`, clean watcher stop, and clean
device teardown.  Host/device logs and the before/after root-cause ledger are
under `artifacts/watcher/`.

## Capacity and context

Batch-1 full-context resident bytes/device are 74,895,259,776 on P150,
39,272,108,928 on P150x2, and 21,543,073,152 on P150x4.  Fixed state already
exceeds 32 GiB for the first two, so their largest feasible resident context is
zero.  P150x4 retains 131072 tokens, batch 10 at full context, batch 11 at
130816, and batch 32 at 44928.  Non-aligned logical prompt support remains part
of `doc/context_contract.json`.

## Final verification ledger

- all-36-layer resident correctness/performance: pass;
- all-36-layer batch-2 bitwise reproducibility: pass after bounded reset;
- AIME24 prefill and teacher-forcing gates: pass;
- autoregressive and qualitative degeneration checks: pass;
- mixed prompts/fixed slots/inactive rows/non-aligned lengths: pass;
- greedy and top-k/top-p trace contracts: pass;
- profiler raw/table/CSV/provenance: pass;
- fully enabled watcher, separate from profiler: pass;
- runtime fallback audit: clean;
- pre-commit over every touched production source and the stage's core docs:
  pass on the final formatted tree; Black's two format-only hash changes are
  disclosed in profiler provenance;
- `python -m compileall -q` over the changed model/generator/test Python: pass;
- JSON parse over all 29 optimized-stage JSON files plus
  `doc/context_contract.json`: pass;
- 18 selected host-only capacity, state, trace-lifecycle, sampler, qualitative,
  and generator-contract pytest cases: pass, 12 deselected;
- `.github/scripts/copilot-build.sh --build-dir
  build_codex_optimized_full_model`: pass, all 1418 targets linked and
  installed.  The default build directory was not reused because its existing
  host-path CMake cache is incompatible with the wrapper's `/work` mount; the
  isolated build also avoided modifying that user-owned cache.  Garage
  credentials were absent, so this was a successful cold-cache build;
- fresh xhigh `$stage-review`: `clean-pass`, with the first-token shortcut
  wording corrected and focused rereview retaining `clean-pass`; report at
  `stage_review.md`;
- stage-owned local commit SHA: pending; never push.
