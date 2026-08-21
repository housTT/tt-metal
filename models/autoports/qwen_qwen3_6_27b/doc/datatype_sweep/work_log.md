# Datatype-sweep work log

Date: 2026-08-20. Hardware: four local Blackhole P300c devices, 1x4 tensor-
parallel physical Ring. Runtime checkout: `/home/ttuser/dev/tt-metal`.

## Entry checks and thresholds

The completed optimized full model and its 100-token AIME24 chat-template
reference were used as the starting point. `tt-smi -ls --local`, a 1x4 mesh
smoke, and post-run health checks passed. TT commands were serialized.

Accuracy gates are top-1 >= 0.90 and top-5 >= 0.98; top-100 is retained as a
diagnostic. Only warmed post-capture traced teacher-forcing decode is ranked.
The common command shape was:

```bash
TT_METAL_HOME=/home/ttuser/dev/tt-metal \
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal:/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal \
QWEN36_PRECISION_CONFIG=<candidate.json> \
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --reference models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING \
  --trace-region-size 1500000000 --warmup-repeats 1 \
  --output-json <candidate-evidence>/teacher_forcing_metrics.json
```

The final baseline refresh explicitly set
`QWEN36_PRECISION_CONFIG=.../candidates/baseline_optimized_mixed.json` for both
teacher forcing and prefill. Both raw metrics embed the resolved baseline policy
and source path. Exact expanded commands are stored in the work log above and
the teacher-forcing row in `sweep_results.json/.csv`.

## Candidate progression

The final refreshed baseline produced 0.97/1.00/1.00 top-1/top-5/top-100,
975.83 ms TTFT, and 22.568 t/s/u. Candidate outcomes are summarized in `README.md` and
fully recorded in the sweep ledgers.

The sweep covered the likely lower-precision wins and canonical controls:

- all four CCL payloads at BFP8;
- full-attention down weights at BFP4+LoFi (the selected extension of the
  baseline's existing BFP4+LoFi MLP groups);
- projection HiFi2;
- BF16 KV cache;
- full-attention down BFP8+HiFi2;
- an identical-weight all-MLP BFP4+HiFi2 comparison covering every selected
  BFP4 gate/up/down group;
- a canonical BFP8+HiFi2/BF16-CCL/BF16-KV policy.

The original canonical policy failed during traced warmup because the
linear-attention MLP-down BFP8 static-CB region ended at 1,333,760 while the
observed L1 allocation floor began at 928,000. The mandated AutoFix loop is in
`autofix/canonical_l1/`. It reproduced the arithmetic, retained BFP4 only for
linear-attention down, computed an 811,520 endpoint, and then passed the exact
full-model traced command at 0.98/1.00/1.00 and 18.257 t/s/u. This proves the
failure is repaired and also provides evidence for rejecting the canonical
policy family on performance.

`full_down_bfp4_lofi` was repeated because its gain over baseline is small. The
two results are 22.614326 and 22.608416 t/s/u, averaging 22.611371 with 0.026%
spread. It is the fastest evaluated candidate satisfying both accuracy gates.

Independent review requested the missing clean same-dtype fidelity comparison.
`all_mlp_bfp4_hifi2` therefore kept the selected BFP4 weights for all linear-
and full-attention gate/up/down groups, kept projection/CCL/KV/activation/head
policy identical, and changed only those six MLP fidelities to HiFi2. The exact
full-model traced run passed at 0.95/1.00/1.00, 983.20 ms TTFT, and 19.670 t/s/u.
It is 13.0% slower than the LoFi candidate at identical accuracy, so LoFi
remains selected with direct evidence for every material BFP4 group.

Every JSON/CSV row now records branch
`agentic-research/hous/qwen3.8-27b`, base commit
`b7b52f83305e1e7c350bde15c7b648d43652e4e2`, the dirty measured-source state,
environment notes, hardware, and mesh. Measurements used the live stage-owned
policy/model/readiness diff; the unrelated Tracy/UMD submodule dirtiness and
untracked cluster descriptors were excluded.

`source_provenance.json` separates `pre_strict_policy_v1` rows from the later
`strict_policy_v2` BFP4+HiFi2 row, hashes the final reviewed runtime source, and
records exact behavior-equivalence arguments for intervening host-only changes.
The refreshed baseline retains its independently verified measured-source hash
and predates only canonical-lowercase rejection; its embedded policy contains
only lowercase values, so construction is unchanged.
The exact historical uncommitted patches were not preserved; this is an
explicit limitation, not a claim that the later checkpoint was the measured
source. The post-clean-review checkpoint will make the final implementation
durable without retroactively relabeling either measurement cohort.

## Default integration and capacity

`selected_precision_config.json` is loaded by `tt/precision.py` unless a
candidate override is supplied. The policy is passed through `Generator` and
`QwenFullModel` to every decoder, LM head, and KV-cache allocation. The
readiness runners record the live resolved policy. Host tests prove the default,
override, inheritance, exception, and strict-validation contracts. No vLLM
adapter was created or modified.

`doc/context_contract.json` was recalculated for BFP8 and BF16 cache candidates.
Both retain the 262,144-token shared physical pool at batch 32; detailed byte
arithmetic and remaining DRAM are embedded in that contract.

## Post-selection commands and results

Default selected-path teacher forcing (no policy override):

```bash
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --reference models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING \
  --trace-region-size 1500000000 --warmup-repeats 1 \
  --output-json models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/evidence/final/teacher_forcing_metrics.json
```

Result: 0.95/1.00/1.00, 977.16 ms TTFT, 22.607 t/s/u, trace enabled. The
metrics contain `precision_summary.path` pointing to the selected artifact.

Warmed full-depth token-out through the same default construction path:

```bash
QWEN36_RUN_TOKEN_OUT_BENCHMARK=1 QWEN36_BENCH_LAYERS=64 \
QWEN36_TOKEN_OUT_METRICS_JSON=models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/evidence/final/token_out_metrics.json \
pytest -q -s models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py::test_reduced_token_out_latency_breakdown \
  --junitxml=models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/evidence/final/token_out.junit.xml
```

Result: 23.054 t/s/u device-only no-readback model+sampler trace; 668.53 ms
TTFT and 20.639 t/s/u in the separate representative caller-visible
prompt-128/generate-128 path.

Non-aligned selected-path gate:

```bash
QWEN36_RUN_FULL_MODEL_SMOKE=1 pytest -q -s \
  models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py::test_reduced_full_model_prefill_decode_and_split_trace \
  --junitxml=models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/evidence/final/non_aligned.junit.xml
```

Result: passed mixed prompt lengths 65/67, split trace capture/replay, changed-
only page tables, and deterministic/unseeded sampling modes.

Qualitative controls:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/run_qualitative_suite.py \
  --backend hf --output-dir models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/evidence/final/qualitative --max-new-tokens 64
python models/autoports/qwen_qwen3_6_27b/tests/run_qualitative_suite.py \
  --backend tt --output-dir models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/evidence/final/qualitative --max-new-tokens 64
python models/autoports/qwen_qwen3_6_27b/tests/run_qualitative_suite.py \
  --backend check --output-dir models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/evidence/final/qualitative
```

Automated and manual verdicts pass all six exact-chat-template prompts.

Host checks pass as 12 precision-policy cases plus one full-model static
contract. The readiness runner regression suite separately passes 7/7. Final
ledger generation and JSON parsing pass. Both pyplot charts were visually
inspected after regeneration. Retained JUnit paths are
`evidence/final/host_policy_readiness.junit.xml` and
`evidence/final/full_model_static.junit.xml`.

## Stage review and commits

The first independent review is `stage_review.md` and returned
`more-work-needed` for (1) the missing same-dtype BFP4 HiFi2 comparison and (2)
missing per-row branch/commit/dirty provenance. Both findings are remediated as
described above. The first fresh rereview, `stage_review_2.md`, requested exact
root/nested schema rejection and honest per-cohort source provenance; both were
added. The second, `stage_review_3.md`, found ambiguous baseline command
provenance; the baseline teacher-forcing and prefill runs were repeated through
an explicit baseline override, and their raw artifacts now embed the consumed
policy summary. The third, `stage_review_4.md`, found accepted case variants in
CCL/KV datatype fields that direct consumers would not interpret canonically.
The loader now rejects non-lowercase datatype and fidelity spellings, with
specific CCL and KV-cache regressions. The fourth fresh rereview,
`stage_review_5.md`,
returned `clean-pass` with no required work after independently re-deriving the
candidate ranking, default-path policy consumption, source provenance, context
capacity, non-aligned/token-out/qualitative gates, AutoFix evidence, and both
Pareto plots. The local checkpoint SHA is recorded below. No push will be
performed.

Primary stage checkpoint: `2d494f6d253` (`Add Qwen3.6 datatype sweep policy`).
