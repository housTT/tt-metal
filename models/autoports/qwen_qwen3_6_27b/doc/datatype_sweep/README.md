# Qwen3.6-27B datatype sweep

Status: **passed**. The selected policy is `full_down_bfp4_lofi`.

## Decision

The full-model gate is top-1 >= 0.90 and top-5 >= 0.98 on the main AIME24
chat-template readiness reference. Ranking uses only steady, post-capture traced
teacher-forcing decode on the 100-token reference; eager and untraced numbers
are excluded. All completed candidates pass the accuracy gate and retain
top-100 = 1.00. The selected candidate is the fastest evaluated passing point:
22.611 t/s/u averaged over two runs (22.614 and 22.608, 0.026% spread), with
top-1 = 0.95 and top-5 = 1.00.

| config | top-1 | top-5 | TTFT ms | traced decode t/s/u | decision |
|---|---:|---:|---:|---:|---|
| `full_down_bfp4_lofi` | 0.95 | 1.00 | 982.50 | **22.611** | selected |
| `baseline_optimized_mixed` | 0.97 | 1.00 | 975.83 | 22.568 | control |
| `ccl_all_bfp8` | 0.97 | 1.00 | 1007.29 | 22.558 | slower |
| `kv_bf16_control` | 0.96 | 1.00 | 979.63 | 22.550 | slower, larger cache |
| `full_down_bfp8_hifi2` | 0.97 | 1.00 | 990.98 | 22.301 | slower |
| `projection_hifi2` | 0.96 | 1.00 | 988.50 | 21.277 | slower |
| `all_mlp_bfp4_hifi2` | 0.95 | 1.00 | 983.20 | 19.670 | same BFP4 weights, HiFi2 slower |
| `canonical_runnable_bfp8_hifi2_kv_bf16` | 0.98 | 1.00 | 973.63 | 18.257 | slower |
| `canonical_bfp8_hifi2_kv_bf16` | - | - | - | - | traced warmup L1 failure; repaired candidate above |

The Pareto charts are [top1_perf_pareto.png](top1_perf_pareto.png) and
[top5_perf_pareto.png](top5_perf_pareto.png). They contain every completed
full-model point, the evaluated Pareto frontier, a red selected point, and the
dotted minimum-accuracy line. Top-5 is tied at 100% for every completed point,
so its frontier is determined by performance; Top-1 exposes the trade between
the selected 95%/22.611 point and higher-accuracy, slower policies.

The direct same-dtype fidelity comparison keeps every selected MLP weight and
all non-MLP policy fields fixed:

| material group | weight | LoFi result | HiFi2 result | decision |
|---|---|---|---|---|
| linear-attention gate/up/down | BFP4 | 0.95/1.00, 22.611 t/s/u | 0.95/1.00, 19.670 t/s/u | LoFi |
| full-attention gate/up/down | BFP4 | 0.95/1.00, 22.611 t/s/u | 0.95/1.00, 19.670 t/s/u | LoFi |

Thus every material selected BFP4 group has both legal LoFi and HiFi2
full-model evidence. Accuracy is identical and HiFi2 is 13.0% slower.

## Selected runtime policy

The authoritative artifact is
[selected_precision_config.json](selected_precision_config.json):

- BF16 embedding and final norm; BFP8 attention input/output projections and
  LM head; BFP4 gate/up/down weights for both linear- and full-attention MLPs.
- No layer exceptions. Projection, all MLP roles, and LM head use LoFi.
- Activations, residuals, and norm tensors remain BF16.
- CCL is BF16 for both attention branches and the full-attention MLP, and BFP8
  for the linear-attention MLP.
- KV cache is BFP8 tile-DRAM paged storage with block size 64.
- LM-head logits and sampler inputs are BF16; greedy sampling uses local BF16
  max/argmax, while stochastic probability math may promote where required.

`tt/precision.py` loads this file by default, or a candidate through
`QWEN36_PRECISION_CONFIG`/the `precision_config_path` constructor argument.
`QwenFullModel`, `MultichipDecoder`, `VocabParallelLMHead`, and KV-cache
allocation consume the resolved weight, fidelity, CCL, cache, and
logits/sampling policy. Readiness metrics embed the live model's resolved policy
and absolute source path. Unsupported activation/residual/logits alternatives
are rejected during validation instead of being accepted and ignored. Host
tests cover default loading, candidate inheritance, environment precedence,
layer exceptions, and rejection behavior. The later vLLM adapter must use the
same generator/full-model constructor; no vLLM integration was started here.

Every immediate, root, and per-kind policy key is exact-validated. In
particular, misspelled or extra root fields and nested `linear_attention` /
`full_attention` dtype/fidelity keys are rejected rather than appearing in a
runtime summary without affecting execution.

## Runtime source provenance

Every ledger row records the branch, base commit, environment, dirty-source
cohort, cohort identifier, final-runtime source SHA-256, and a row-specific
behavior-equivalence statement. [source_provenance.json](source_provenance.json)
maps each row to its cohort and hashes the final runtime source files.

The exact historical uncommitted patches were not preserved; the manifest says
so explicitly. Earlier rows predate host-only strict-schema and page-block
plumbing fixes. Their configs all contain valid known keys and page block 64,
so validated block 64 is algebraically identical to the former constant; no row
uses the later-added projection HiFi4 branch. The new BFP4+HiFi2 row was
measured after that plumbing and predates only exact rejection of invalid extra
keys. The refreshed baseline has its independently verified measured-source
hash and predates only canonical-lowercase rejection; its embedded policy is
entirely lowercase. These changes do not alter TT construction for any recorded
row. The final source hash identifies the reviewed implementation, while the
cohort mapping preserves the distinction between measured dirty state and final
code.

## Baseline and final validation

The final refreshed baseline explicitly used the frozen baseline config,
`readiness_aime24_chat.refpt`, its Qwen chat template, and all 100 generated
target tokens. Both raw artifacts embed the resolved baseline policy. Prefill
is 97/100 top-1 and 100/100 top-5/top-100. Traced teacher forcing is 97/100,
100/100, 100/100, 975.83 ms TTFT, and 22.568 t/s/u.

The normal default selected-config construction path was rerun after selection:
95/100 top-1, 100/100 top-5/top-100, 977.16 ms TTFT, and 22.607 t/s/u traced
teacher forcing. The separate warmed full-64-layer token-out benchmark reports
23.054 t/s/u for the device-only model-plus-distributed-argmax trace with no
readback. Its representative prompt-128/generate-128 caller-visible run reports
668.53 ms TTFT, 48.452 ms/token, and 20.639 t/s/u. Later reports and serving
comparisons should use this post-selection token-out result when available,
while the sweep ranking remains the teacher-forcing measurement above.

The selected policy preserves mixed non-aligned prompt lengths 65/67 under
split trace capture/replay. The six-prompt HF/TT chat-template qualitative suite
passes automated degeneracy checks and manual review; both controls show the
base checkpoint's prompt-relevant "thinking process" style, without wrong-
language drift, leakage, malformed controls, or mechanical loops.

## KV capacity

`doc/context_contract.json` was recomputed for both evaluated cache policies.
BFP8 cache storage is 2,281,701,376 bytes/device at the 262,144-token physical
pool. BF16 is 4,294,967,296 bytes/device, an increase of 2,013,265,920 bytes,
and still leaves 11,371,307,008 bytes/device outside the explicit plan. The
selected BFP4 full-attention down weights save 377,487,360 bytes/device versus
the optimized baseline, leaving 13,762,060,288 unreserved bytes/device. Both
KV candidates preserve the advertised 262,144-token shared pool and maximum
batch 32; there is no capability reduction.

## Limitations

- The selected performance edge over the final baseline is small (+0.194%), but it is
  repeatable across two selected runs; both raw values are retained.
- A same-dtype all-MLP BFP4+HiFi2 comparison was added after independent
  review. It matches selected accuracy but is 13.0% slower, closing the legal
  fidelity alternative without changing selection.
- Candidate TTFT is recorded but is not the selection axis. Compile, eager,
  and untraced decode are not Pareto inputs.
- The public graph contract currently requires BF16 activations/residuals/norm,
  so lower activation policies are explicit unsupported cases rather than fake
  sweep points.
- Datatype and compute-fidelity values use canonical lowercase spellings. The
  strict loader rejects case variants before they can reach direct CCL or
  KV-cache consumers.
- The original all-BFP8/HiFi2/BF16-cache canonical policy exceeded the observed
  MLP-down static-CB L1 boundary. AutoFix isolated the exact collision and
  verified a runnable one-field BFP4-linear-down repair; that repaired policy is
  valid but materially slower.
- Exact historical dirty patches were not retained. The source-provenance
  manifest records this limitation and the evidence-backed behavior-equivalence
  mapping; it does not mislabel the later checkpoint as the measured source.

## Artifacts

- [sweep_results.json](sweep_results.json) and
  [sweep_results.csv](sweep_results.csv): full candidate ledger, policies,
  metrics, regimes, commands, branch/base commit, dirty-source/environment
  provenance, hardware, mesh, and status.
- [selected_precision_config.json](selected_precision_config.json): default
  selected runtime policy.
- [source_provenance.json](source_provenance.json): per-row dirty cohorts,
  final runtime source hash/file list, and equivalence/limitation record.
- `evidence/baseline/`: refreshed baseline prefill and teacher forcing.
- `evidence/candidates/`: per-candidate full-model metrics and runtime failure.
- `evidence/final/`: default selected teacher forcing, post-selection token-out,
  non-aligned JUnit, and qualitative controls/verdict.
- `autofix/canonical_l1/`: source diagnosis, isolated hypothesis, repaired
  candidate, and hardware verification.
- [work_log.md](work_log.md): commands, chronology, and commit record.
