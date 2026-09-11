# AutoDebug: original quality failure after expert decode correction

**The decode gate candidate failed the original quality requirement. The prompt
format is correct. One distinct prefill arithmetic hypothesis remains testable;
neither an implementation root cause for repetition nor an inherent checkpoint
limitation is established.** Do not retain a model precision policy, enlarge the
budget, enable thinking, change sampling, or waive this failure from this evidence.

This isolated AutoDebug pass follows AutoFix and qualitative-check. I read
`/home/hous/dev/gemma-4-26B-A4B-it/no_cpu_continuation.md` first. Work was confined
to saved outputs, source, native tokenizer files and host tokenization. No TT
imports, device calls, CPU model inference, new evaluation or production edits
were performed. Root owns all reported hardware measurements.

`W=/home/hous/dev/tti-release-gemma4`;
`M=/home/hous/dev/tt-metal/models/autoports/google_gemma_4_26b_a4b_it`.
Paths and freshly checked hashes are in [EVIDENCE.json](EVIDENCE.json).

## What the latest result establishes

Actual result `W/meta_gpqa_expert_precision_quality_control_run_v1/result.json`,
SHA256 `cff57f4cabc71877bfdc227f5a734408ae69cc15f2daaec91589bf241a7ffc2f`,
preserves the 326 prompt IDs, max2048, T0, seed42, stop[], context262144,
32 slots and execution width1. Both arms complete cleanly at 2048 tokens with
`finish_reason=length` and no final answer. I read both complete continuations
and verified their saved IDs decode exactly to their text. Neither emits native
stop IDs `{1,50,106}`. Baseline IDs and text exactly match the earlier direct B1
baseline. The candidate first differs at output ID index28, zero-based.

The candidate recognizes cyclohexane earlier but then repeats rejected chemical
alternatives. The complete line `If X is $C_6H_{10}$ and $C_6H_{10}$? No.` occurs
four times. Its incidental mention of D/18 is followed by rejection and more
reasoning; it never supplies the requested conclusion. This is substantive
failure, not merely a strict answer-regex mismatch. Root's retained
`ACTUAL_QUALITY_REVIEW.json` records no model policy applied and clean cleanup.

The preceding component result
`W/meta_gpqa_moe_gate_fp32_fidelity_run_v1/result.json`, SHA256
`5ff778940c500edc6a2cc2953b562c2229c2747e4d55764d639eba1c0852accb`,
measured packed gate relative L2 **2.4804% → 0.1675%**, and post-MoE norm
**1.3212% → 0.3097%**, against the validated TT FP32 SFPU product/add-tree
reference. All16 expert/chip rows and exact replay guards passed. That local
result is valid; it is insufficient to claim a repetition repair. The full-case
probe changes only packed B1 decode gate/up fidelity and FP32 destination at all
30 layers. It explicitly restores the native shared configuration outside the
decode method and proves prefill unchanged (`probe_tt.py:262-325`).

Earlier cache, mask, routing/index and execution controls, the failed decode
SDPA quality candidate, and the sparse FP32 contract repairs remain retained
evidence. They do not justify another precision matrix. The independently
verified sparse CB sizing, singleton multicast and FP32 reload repairs remain
separate from model-quality claims; they are retained in commit
`bc4752de65dce32b8e68eacd7ebcb40dac276cbd`.

## Prompt/template and TTI Meta audit

The checkpoint is `google/gemma-4-26B-A4B-it`, pinned revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, tokenizer `GemmaTokenizer`.
Its `chat_template.jinja:186` defaults `enable_thinking` to false;
lines381-385 append the native generation suffix:

```text
<|turn>model
<|channel>thought
<channel|>
```

That is an empty, closed native thought channel. Enabling thinking instead
injects `<|think|>` in the system turn (`chat_template.jinja:190-195`).

For this exact case, fresh inspection establishes all of the following:

- Original captured payload has exactly326 IDs and one BOS. Native tokenizer
  encoding/decoding reproduces them and the saved transport string exactly.
- Direct Jinja rendering of the fixture's original `chat_completion_input`
  equals that transport, under both the default and explicit false setting.
- The fixture explicitly requests `## Step 1: [Concise description]`, a brief
  explanation, subsequent steps, and `The best answer is [the_answer_letter].`
  It ends with `Let's think step by step.` Thus the generated heading is
  requested formatting, not evidence of a wrong template or missing instruct mode.
- `meta_fixture_wiring_autofix_v1/prepare.py:155-209` renders the native template
  once, tokenizes without adding special tokens again, and transports IDs to
  `/v1/completions`. Its manifest disables a second template/BOS application.
  The captured payload proves the effective request, independently of comments.
- `prepared/include/meta_gpqa_cot.yaml` retains T0/max2048 and explicitly declares
  `enable_thinking: false`. It scores unmodified completion text using
  `best answer is ([A-Z])`; no hidden reasoning parser or answer recovery is
  needed to explain either failed direct continuation. Historical
  `M/doc/tti_release/RUN_NOTES.md:1085` records this same native default recipe.

**No malformed, double-wrapped or untemplated prompt was found.** Natural-language
step-by-step instructions do not require the separate native thinking channel.
Changing that channel would change the preserved evaluation recipe.

## What the successful R1 case does and does not show

The retained timeout7200 R1 doc23 contains the exact same chemical question,
records exact-match1, and ends `Answer: (B)` followed by `\boxed{B}`. Its captured
stream completed with `stop`, and its complete text hash matches the sample.
I read the complete R1 continuation; it also contains extended mistaken reasoning
and repeated rejected alternatives before arriving at its final answer.

| Request property | Original Meta doc58 | Retained successful R1 doc23 |
| --- | --- | --- |
| Native mode | Closed thought, no `<|think|>` | Same closed thought, no `<|think|>` |
| Prompt tokens | 326 | 229 |
| Answer choices A/B/C/D | 22 / 16 / 12 / 18 | 22 / 18 / 12 / 16 |
| Required conclusion | `The best answer is D.` | Boxed B |
| Sampling | T0, seed42 | T1, top-k20, top-p.95, seed42 |
| Output allowance | 2048 | 32768 |
| Transport | Non-streaming completions | Streaming completions |

R1 additionally uses different user instructions and scheduling. Its text
re-encodes to4838 tokens, with the final `Answer: (B)` marker beginning at
re-encoded offset4827. These are **host re-tokenization counts, not original
emitted IDs**; the retained streaming summary has no usage count. This shows
that this different run's final answer occurs after2048 tokens. It does not
establish that the original greedy Meta trajectory would recover with more
budget, nor that2048 is inherently inadequate for a faithful implementation.
R1 is an outcome comparison with multiple changed variables, not a matched
canonical control or a prompt repair.

## One remaining discriminating test: untouched prefill gate

**Hypothesis:** native prefill gate/up arithmetic still materially perturbs the
prompt representation; improving decode alone cannot remove those pre-existing
states. This is supported as a testable hypothesis, not a proven quality cause:
`M/tt/multichip_decoder.py:729-739` gives prefill and decode the same resident
packed expert-weight object, and `_moe_prefill_chunk` in
`M/tt/optimized_decoder.py:3012-3135` consumes the unchanged shared
`expert_gate_compute_config`. Native prefill therefore still executes
LoFi/BF16-destination gate/up using the same BFP8_B weights. Its outputs feed
subsequent layers and their prompt KV states. The failed decode-only candidate
never tests this phase.

Before allocating a new capture, I checked the saved exact-history operation
artifacts. `prefill_1_capture_layer0.pt` contains the last row at position326 of
a327-token history (physical352); `prefill_68_capture_layer0.pt` contains row393
of a394-token history (physical416). Their metadata and producer source
`operation_capture.py:20-51` show **one-row** MoE input, routing and output
readbacks, no packed gate result, and no complete32-row routing group. Their
hashes are freshly verified in EVIDENCE.json. They are useful historical row
controls but cannot reproduce the original326-token prefill's exact routing
union. Filling the missing rows with zeros would silently change that control.

The smallest exact test is one original-prompt **layer0 final prefill group**:
retain rows320:352, including all26 padded rows, and assess logical row325
(group offset5). Source predicts326→352 physical rows, split into eleven32-row
calls, with `groups=1`, BF16 A `[1,1,32,2816]` in DRAM, resident BFP8_B B
`[1,128,2816,704]`, outputBF16/DRAM, K88 tiles/block44, per-core M/N1,
subblock1×1, grid11×2 and `fuse_batch=False`. TP2 sets prefill per-core N1
(`multichip_decoder.py:490`); `_optimized_sparse_prefill_config:760-800` derives
the grid. Routing is the actual32-row union, non-indexed, `nnz=None`.
These source predictions require runtime confirmation; the earlier decode
indexed-eight-expert geometry is not a substitute.

For this **single component control**, freeze the exact native group operands,
routing union, original packed gate output and row325 downstream MoE/norm
outputs. Reproduce native results exactly before evaluating the known
HiFi4+FP32-destination gate configuration with every other flag, weight,
downstream operation and program field unchanged. Compare row325's eight
routed expert outputs on both chips against the same validated FP32 TT SFPU
mul/pairwise-add reference, then compare its downstream weighted MoE/norm row.
Retain the existing all16-improve, mean-gate-error-halves and both-chip norm
improves20% admission thresholds. A reference small-term invariant, exact
quantized resident-weight/input readback guards and original native output
guards are mandatory. MVMUL remains an invalid high-precision oracle.

Capacity stays bounded by processing one selected row/expert at a time:
each FP32 `[1,1,4096,704]` reference scratch is11MiB per chip. Do not expand all128
experts to FP32 or duplicate the packed model weights. Keep scratch in DRAM,
free each expert's temporaries, and bind the actual peak/available capacity
before admitting execution. The native sparse program has one output tile per
core, so FP32 CB5 is 4096 bytes and its 1×1 subblock fits FP32 half-sync DST;
the repaired size/reload paths apply. This is a proposed diagnostic, not an
executed or capacity-certified probe.

The capture requires only the original 326-token prefill, with no generated
continuation; its untouched terminal logits must retain the saved native hash
`2f404628b93a6c59f01d8ae2d8f3f26dcd396f965e106ae9164265604a132c93`.
A failed native replay invalidates the control; absent material improvement
refutes the proposed prefill candidate. Neither is grounds for another sweep.
A component pass would justify root review of one candidate covering the
previously omitted **prefill plus the already tested decode gate path** under
the unchanged original case. Repeating decode-only generation would add no
discriminating evidence. This report does not prepare or authorize another
full generation; local agreement still does not prove a repetition fix.
If that bounded hypothesis is refuted, this investigation has no further
source-backed implementation repair to recommend. The faithful-model versus
implementation attribution remains unverified under the permitted evidence;
that limitation does not pass quality or Stage11.

## Verification

Exit0,31 artifact bindings, framework/model imports explicitly blocked:

```sh
/home/hous/dev/tt-metal/python_env/bin/python -I -B /home/hous/dev/tti-release-gemma4/meta_gpqa_after_expert_quality_diagnosis_v1/inspect_evidence.py
```

The script only inspects retained JSON/text, verifies hashes, renders the pinned
Jinja template and tokenizes existing text. It performs no model arithmetic or
evaluation. No build was needed for these report-only artifacts.
