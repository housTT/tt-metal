# AutoFix: thermodynamics continuation control

## Starting evidence

[AutoDebug diagnosis](AUTODEBUG_thermodynamics.md): P150x4 qualitative output index 3 introduces a factual overgeneralization about all energy conversion wasting heat. Historical sampled text and unrelated numeric controls were insufficient to attribute or dismiss this particular claim.

## Hypothesis experiment

**Hypothesis:** the optimized serving adapter introduced the disputed continuation behavior.

**Experiment:** the coordinating agent ran [run_thermodynamics_control.py](run_thermodynamics_control.py), equivalent repository-root invocation:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/run_thermodynamics_control.py
```

The script compares adapter commit `6eb0427423392d7c6a7f87a511be892b8bf677ae` with selected adapter SHA-256 `649cd5ab17cc97af777881e6be63baac96ba27f964be59e753b61d8abb0f6325`. Model/runtime/hardware settings are shared: P150x4 profile (`P300x2`, four chips), `FABRIC_1D_RING`, 32 sequences, block size 64, context 262144, trace region 220000000, async scheduling, and the existing Gemma parsers. The exact launch arguments are retained in [manifest.json](thermodynamics_control/manifest.json).

Each fresh server receives three identical serial requests, cold then with warmed program/cache history. They use the actual rendered thermodynamics chat prompt plus the retained sampled prefix ending `processes. `, immediately before the disputed sentence. `/v1/completions` continues that rendered prefix at temperature 0.7, top-p 0.9, top-k 32, max tokens 256, with no explicit seed or requested logprobs. This preserves eligibility for device sampling. Every response is retained without selection.

**Result:** all corresponding original/optimized completion texts match exactly. All six responses report 247 prompt tokens, 256 completion tokens, and `finish_reason="length"`. Every continuation reproduces the substantive claim that converting energy always wastes some as heat. All also contain an absolute-zero molecular-motion simplification; these outputs are not declared factually correct.

| Request | Original artifact | Optimized artifact | Comparison |
| --- | --- | --- | --- |
| 0 | [original_0.json](thermodynamics_control/original_0.json) | [optimized_0.json](thermodynamics_control/optimized_0.json) | Exact completion-text equality |
| 1 | [original_1.json](thermodynamics_control/original_1.json) | [optimized_1.json](thermodynamics_control/optimized_1.json) | Exact completion-text equality |
| 2 | [original_2.json](thermodynamics_control/original_2.json) | [optimized_2.json](thermodynamics_control/optimized_2.json) | Exact completion-text equality |

The investigator independently compared response text, usage, finish reasons, and all six response hashes against the manifest. Both runner exit codes are zero; both launch logs report clean termination. The manifest records `adapter_restored=true`, and the restored file hash matches the selected adapter. Nanobind leak diagnostics remain present in both server tails; this experiment does not resolve those existing diagnostics.

**Verdict:** the hypothesis is refuted for these controlled continuations. The factual imprecision is reproducible in baseline TT serving under the retained prefix, with no observed adapter difference in the three matched continuations. **Fix:** none; no runtime or model edit is supported by this evidence.

## Final status and limits

Existing TT-serving qualitative limitation, retained explicitly. This is a conditional continuation control, not equivalence of the original full stochastic response, a statistical error-rate comparison, or proof that HF weights alone cause the error. Repeated positive-temperature requests have different seed keys, so this is not a claim that one stochastic trace was reused across requests. The original sampled suite remains preserved. No further serving change is justified by this finding; broader factual-quality improvement is separate work.
