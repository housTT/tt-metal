# Extended serving qualitative control

Date: 2026-09-08 UTC. The shared 256-token serving suite has a substantive
haiku count error beyond the earlier 128-token coverage. The longer serving
request reaches a final answer and retains that error. The fresh selected-policy
standalone run reproduces all 390 serving token IDs exactly, including EOS.
The error is therefore present in the selected full-model path and is not a
new vLLM regression on this prompt. It remains unresolved quality work against
the fresh pinned HF continuation, whose completed final answer has correct
5/7/5 syllables and whose first 256 IDs exactly match the prior HF control.

The qualitative-check skill was applied to all twelve saved texts in
`full_b32_qualitative_before_order.json` and the exact controls in
`qualitative_controls.json`. No production source or selected precision was
changed. This analysis uses saved artifacts, a CPU tokenizer comparison, and
a CPU-only HF generation; all TT execution belongs to the supervising lane.

## Comparable coverage

The snapshot is `../upstream`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. Its local Hugging Face download
metadata records that revision for configuration, tokenizer, template, index,
and all four weight shards. The shared prompt file is
`models/common/readiness_check/vllm_prompts.txt`, SHA256
`2ad452c15d8442d3fe641eb968bbd84e3bfe9acb01ddc540f451983c0da3cbbd`.
The original Qwen2Tokenizer template SHA256 is
`9dd2fbd270feaa1fbef2d4f634d7887c9c506e3bde140f8e7351c8944e8fd235`.
No prompt or sampling instruction was added to improve the haiku outcome.

All six greedy serving responses exactly start with the selected standalone
TT128 text. `qualitative_token_prefix_comparison.json` additionally checks all
128 saved standalone token IDs against the serving text retokenized with the
same tokenizer; all six agree. These reconstructed IDs are distinct from raw
API token evidence. Parent artifact `qualitative_prefix_comparison.json`
independently records the string comparison. The selected policy is
`head4_lofi_last8_c32_k4_r2`; the earlier standalone suite used batch one and
cache context 2048. This confirms stability within the previous coverage and
does not establish behavior after token 128.

## Twelve-text review

| Prompt | Greedy256 | Sampled256 |
| --- | --- | --- |
| 0, machine-learning haiku | Incorrectly counts the six-syllable draft “From data, patterns grow” as five; reasoning is unfinished at the cutoff. | Draft “Data shapes new minds / Patterns hidden in the noise / Learn to see what's true” has 5/7/5 syllables; reasoning continues at the cutoff. |
| 1, supervised versus unsupervised learning | Correct labeled/unlabeled distinction and relevant explanation; reasoning continues. | Same correct distinction with a relevant teaching analogy; reasoning continues. |
| 2, time-machine story | Relevant story planning, still reasoning at the cutoff. | Starts the requested story after the reasoning delimiter, then truncates. |
| 3, thermodynamics | Relevant first-, second-, and third-law discussion; reasoning continues. | Relevant discussion, truncated during the second law. |
| 4, English-to-French greeting | Reaches correct final “Bonjour, comment allez-vous aujourd'hui?” | Reaches the same translation but the last word is truncated. |
| 5, Python Fibonacci | Relevant approach and final code-fence opening; no complete function within budget. | Begins a relevant Python function, truncated at `if n <= 0:`. |

No obvious mechanical repetition, doubled subwords, wrong-language response,
or cross-prompt contamination was found in these twelve texts. Their common
long reasoning is consistent with the saved HF/selected-TT128 controls and
the template's assistant thinking prefix. A finite cutoff remains incomplete
coverage, particularly for the story and executable Fibonacci function. The
haiku's explicit false count is substantive independently of that cutoff.
The sampled results are descriptive checks; no same-seed HF sampled control
was generated, so they do not establish numerical equivalence.

## Haiku controls and finding

Prompt 0 remains exactly “Write a haiku about machine learning.” Its rendered
prompt is `<|im_start|>user\nWrite a haiku about machine learning.<|im_end|>\n<|im_start|>assistant\n<think>\n`.
The exact prompt IDs are
`[248045,846,198,7734,264,6185,36974,883,5484,6618,13,248046,198,248045,74455,198,248068,198]`.

An existing longer CPU HF control was found at
`../datatype_sweep/qualitative_first8_haiku256_v1/prompt_0/`.
Its prompt text, rendered template, and IDs match exactly, and its run metadata
records the pinned revision and a successful bounded run. Its HF256 draft is
“Data shapes the mind / Patterns emerge from the noise / Learning, line by
line,” with correct 5/7/5 counts. It explicitly counts “Data” as two syllables.
Its TT companion uses the previously rejected first-only BFP8 exception policy,
so that companion is not the current selected-policy TT256 control. No saved
selected-policy standalone output beyond 128 tokens was located.

`haiku_serving512.json` extends the same greedy API request without modifying
the prompt. It preserves the previous 256-token prefix, finishes with `stop`,
and returns “From data, patterns rise, / Models learn, they grow with time, /
Wisdom from the past.” The first line has six syllables:
From(1) + data(2) + patterns(2) + rise(1). The other lines have seven and five.
The reasoning falsely certifies the first line as five, and the final answer
does not correct it. This is a completed-task constraint error, not merely an
unfinished draft. The existing HF256 control does not reproduce that failure,
so checkpoint behavior is not an established explanation.

## Fresh controls

`hf_haiku_control.py` runs offline on CPU only, with the original `python_env`,
Torch 2.11.0+cpu, Transformers 5.12.1, eight CPU threads, and explicit BF16.
It validates the observed original class `Qwen3_5ForCausalLM`, all
8,953,803,264 parameters, CPU placement, parameter dtype, and empty load
diagnostics. It uses the same greedy kwargs as the common HF qualitative
helper and inherits `use_cache=False` and checkpoint EOS behavior. No dtype,
cache, model-class, or device fallback is introduced. The optional fast-path
library advisory is the original CPU Torch reference implementation behavior.
Preload memory inspection found about 178 GiB available and no cgroup hard
limit; the script requires more than 36 GiB of effective headroom.

The run requests 512 tokens, writes a separate 256-token prefix, and compares
that prefix to the earlier exact HF256 IDs. Artifacts are
`hf_haiku_512_v1/` and `hf_haiku_512_v1.log`. The run completed with exit 0,
376 generated tokens including EOS, 546.115 seconds of generation, and 0.325
seconds in the model-loading call. These are control-run wall times, not a
serving performance result. The exact bounded shell invocation was:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 USER=hous timeout 1200 \
  python_env/bin/python \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/hf_haiku_control.py \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/hf_haiku_512_v1 \
  > models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/hf_haiku_512_v1.log 2>&1
```

The run completed successfully;
the fresh first 256 IDs match the earlier HF256 control exactly, and the
decoded first 128 tokens equal the saved HF128 text. It stops normally at EOS
with a final answer:

> Data shapes the mind,
> Patterns emerge from the noise,
> Learning, line by line.

These lines have 5/7/5 syllables. The actual full-model CPU class and BF16
precision match the recorded earlier reference, and all load diagnostics are
empty (the JSON serializer represents the three empty key sets as `"set()"`,
and `error_msgs` as `[]`; in-process assertions checked the actual empty sets).
No TTNN module was imported. Thus the matched completed HF answer does
not exhibit the selected TT path's haiku constraint error.

`standalone_haiku_control.py` is source-only preparation for the supervising
hardware lane. It loads all 32 layers, batch one, selected precision unchanged,
and cache context 2048 matching the prior standalone suite. The generator
samples on device and requests 512 tokens with normal EOS trimming. It records
exact selected-TT128 token and serving256 text agreement, all new IDs/text,
precision/cache provenance, and independent teardown/mesh-close results.
The supervising lane completed this run successfully with clean teardown and
mesh close (`standalone_haiku_512_v1/metadata.json` and `.log`). It returned
390 tokens through EOS with the same incorrect final 6/7/5 haiku.
`haiku_standalone_serving_exact_comparison.json` compares actual raw API IDs
against actual generator IDs: all 390 match exactly, including EOS 248046.
The saved selected-TT128 token and serving256 text checks also pass. AST checks
and all applicable pre-commit hooks passed for both scripts
(`haiku_control_static_checks.log`); no C++ build was needed.

The matched standalone continuation localizes this output to the selected
full-model behavior. It does not identify a specific quantized weight group,
prove the original HF checkpoint shares the error, or demonstrate a general
serving pass. This newly observed quality finding against the longer HF
control still requires stage disposition. The earlier 128-token selection
coverage ended before the erroneous draft and therefore did not rule it out.
