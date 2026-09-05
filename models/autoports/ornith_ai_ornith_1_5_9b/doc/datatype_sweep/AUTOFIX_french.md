# AutoFix: datatype-sweep French branch

Status, 2026-09-05: selected policy `head4_lofi_last8_c32_k4_r2` passes its
final bounded default-path qualitative check, `qualitative_selected_v2`.
The invocation consumes 32 head cores/K4/two readers from the selected artifact
and reproduces all seven qualified explicit-config outputs exactly. French
is corrected with no new concrete quality defect in the recorded suite.
The corresponding raw policy without a layer exception fails French.
The earlier K1 default remains historical evidence; the changed geometry is
established by its own new final run.

The earlier precision controls remain historical evidence: raw reduced heads
failed French, and first-layer-only policies fixed French but introduced
concrete haiku count errors. The last-layer BFP8 exception qualifies with
the tested BFP4 heads. This investigation used `$autofix` and `$qualitative-check`,
read source and artifacts, and ran host-only experiments without importing
TTNN or accessing hardware. No implementation changes were made.

Source-audit scope, recorded **2026-09-05 23:01:55 UTC**: source equality
statements below describe historical checks during this investigation, before
the final selected-artifact loader change. They do not claim that the final
live worktree remains identical to earlier run snapshots. The final default
run below supplies separate new evidence. Immutable run provenance remains
authoritative for each earlier result.

## Starting evidence

The starting report is [the earlier French AutoFix](../full_model/AUTOFIX_french_head.md).
Its claims were checked against the actual capture, CPU oracle and generated
token artifacts. The current original command, exit status, source and binary
hashes are in `logs/qualitative_head8_lofi_v1.provenance.json`:

```bash
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.datatype_sweep.run_qualitative --precision-config models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/configs/head8_lofi.json --reuse-hf models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/qualitative_prefill_trace_release_v2 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/qualitative_head8_lofi_v1
```

Prompt 4 is the 24-token upstream chat rendering of
`Translate the following to French: "Hello, how are you today?"`.
Current TT emits `"Bonjour" (informal)`; HF says `good day/morning`, and the
prior qualified TT says `formal/greeting`. The current TT label starts at
generated index 34 (`inform`, token 39102). Greedy generation is capped at
128 tokens, with no completed-answer claim.

## Hypothesis experiments

**Prompt mismatch or stale HF control: refuted for the inspected runs.**
For all six shared prompts plus AIME, raw head8, edge8, first8, last8 and previous-stage
artifacts have identical HF snapshot, user text, chat decision, tokenizer
class, rendered prompt, input token IDs, token budget and HF output token IDs.
The pinned local `Qwen2Tokenizer` independently reproduces all seven prompt
renderings and token lists. Template SHA256 is
`9dd2fbd270feaa1fbef2d4f634d7887c9c506e3bde140f8e7351c8944e8fd235`;
the suite SHA256 is
`2ad452c15d8442d3fe641eb968bbd84e3bfe9acb01ddc540f451983c0da3cbbd`.
Revision is `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` throughout.

**Ignored precision argument or stale materialized weight policy: refuted
for the inspected construction path and recorded runtime tensors.**
During the historical preselection audits on 2026-09-05, all ten inspected
`tt/*.py` files matched the four initial qualitative runs' source hashes;
`run_qualitative.py` also matched the failing run's hash when inspected. A host AST experiment
executed the actual generator argument forwarding, model head selection and
layer construction statements with boundary stubs: baseline, head8, head8
edges and head4 edges forwarded their explicit policies; only layers 0 and 31
received BFP8 projection weights in the edge cases. These stubs check argument
propagation, not device kernels.

The six head8/head4, edge and first/last AIME runtime artifacts independently record the
expected head, all 32 layers' prefill/decode projection dtypes and fidelities,
and `propagation_assertions_passed: true`; a host audit verified those fields.
The earlier baseline artifact has only compute-object representations and no
propagation flag, so it cannot establish actual fidelity from serialized fields.
An initial generic artifact audit hit that older schema; the corrected audit
checks the richer candidate artifacts.

Source follows the selected head dtype into `OrnithModel.upload`; its
`LazyWeight(source=w, _value=w, dtype=...)` returns the materialized tensor.
`LMHead1DConfig.lm_head_dtype=logits_dtype` controls the linear **output** dtype
and does not overwrite weight dtype. Layer policies reach both optimized host
weight packing and multichip reuploads; packed GDN/MLP decode weights inherit
their selected projection dtype. Decode QKVG has its separate explicit policy.
There is no disk weight-cache lookup on this materialized head path.

**The old frozen-hidden oracle directly localizes this run: not established.**
The current French output is identical to
`../full_model/french_head_v1/french_bf8_lofi.json` for all 128 tokens, although
the old BF8 probe used head K2 and the current run uses K1. Thus the earlier
free-running BF8 failure is directly relevant. However, the old frozen-hidden
control prefix differs at generated index 14: old ` into` (1083), current
` to` (310). The old FP32 norm/head oracle selects `inform` at 23.004875 versus
`form` at 22.587357 on that old hidden tensor. It refutes a local terminal-only
explanation for that capture, not for every possible current hidden tensor.
The current run diverges from HF at index 29 and prior BF16 TT at index 14.

**Higher edge projection precision changes the failing branch: verified as
a bounded full-prompt control, without a localized kernel defect claim.**
`qualitative_head8_edges_v1/prompt_4/tt_completion.txt` says
`"Bonjour" (good day) or "Salut" (informal hello)` and correctly distinguishes
the formal/informal question forms. The same head8/LoFi head, BFP8 cache and
other policies are retained; only attention/MLP projection weights at layers
0 and 31 change from BFP4 to BFP8. The earliest edge/raw branch difference is
generated index 29 (` =`, 283, versus ` can`, 628), before the label at 34.
This establishes sensitivity to those decoder weight boundaries without
proving the earlier hidden-state diagnosis applies unchanged.

**With head8/LoFi, first-layer exception suffices for French and last-layer
exception does not: verified by the full-prompt controls.**
`qualitative_head8_lofi_first8_v1/prompt_4/tt_completion.txt` correctly says
`"Bonjour" (formal/greeting) or "Salut" (informal)` and distinguishes the
question forms. Its first difference from raw head8 is at index 29. All other
layers retain the raw head8 policy. In contrast,
`qualitative_head8_lofi_last8_v1/prompt_4/tt_completion.txt` still says
`"Bonjour" (informal greeting)`; its first difference is at index 36, after
the incorrect label. Last-only is rejected despite passing AIME93/100/100.
This establishes first-layer sufficiency among these tested alternatives;
attention versus MLP necessity within layer 0 remains untested.

**First8 also passes the haiku control: refuted by the 256-token continuation.**
The [first8 suite review](qualitative_head8_lofi_first8_v1/QUALITATIVE_REVIEW.md)
initially left prompt 0 uncertain because it stopped at
`"Patterns in the data flow" (5`. The new
`qualitative_first8_haiku256_v1` control completes that count as `(5)`, then
repeats it after saying it will count carefully:
`Pat-ters (2) - in (1) - the (1) - da (1) - flow (1) = 5 syllables ✓`.
The real line has seven syllables, the breakdown drops part of `data`, and its
listed numbers sum to six rather than five. HF's same-prompt control provides
correctly counted 5/7/5 draft lines. First8 is therefore rejected for a concrete
reasoning-quality error not explained by the matching HF output. Neither
256-token output reaches a final answer; no invalid final poem is claimed.

Host verification confirms the 256-token run uses the same original prompt,
tokenizer, revision and first8 precision policy, with both HF and TT preserving
their earlier first 128 tokens exactly. Each has 256 actual tokens, and all ten
runtime source hashes matched provenance during that historical audit. Exact command and exit0 are recorded
in `logs/qualitative_first8_haiku256_v1.provenance.json`:

```bash
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.datatype_sweep.run_qualitative_long --precision-config models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/configs/head8_lofi_first8.json --output models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/qualitative_first8_haiku256_v1
```

**Current BFP4 and HiFi2 controls: outcomes differ by complete policy.**
All seven outputs in `qualitative_head4_lofi_edges8_v1` were read against HF
and prior TT. French correctly describes Bonjour as `good morning/day`, the
haiku has coherent planning without a completed count mismatch, and the other
five outputs show no new concrete quality regression. This policy passes the
bounded smoke; see its [review](qualitative_head4_lofi_edges8_v1/QUALITATIVE_REVIEW.md).

`qualitative_head4_lofi_first8_v1` also corrects French, but explicitly labels
`Line 1 (5 syllables): "From data, patterns grow"`. That completed draft line
has six syllables (From 1 + data 2 + patterns 2 + grow 1); it is rejected for
the count error without claiming a wrong final poem. Other outputs remain
coherent within the budget. `qualitative_head8_hifi2_v1` is independently
rejected because prompt 4 still says `"Bonjour" (informal)`. Its prompt 0 also
contains the awkward one-off phrase `Patterns and patterns`; no extended
repetition is established. All three suites' prompt/HF metadata and runtime
source hashes were verified against their recorded provenance during the
historical preselection audits. HiFi2 is therefore not a proven
remedy for the observed head8 French branch.

`qualitative_head4_hifi2_v1` is likewise rejected: its current prompt 4 says
`"Bonjour" (informal) or "Salut" (casual)`. All seven outputs were reviewed;
the other prompts remain coherent at the recorded budgets, with no completed
haiku count in prompt 0. Input/HF metadata and runtime source hashes were
verified during the historical audit of that recorded command. This is a full-prompt rejection, not an
inference from the earlier terminal probe.

`qualitative_head4_lofi_v1` is rejected for the same current French label.
All seven TT output token lists are identical to the current head4/HiFi2
suite; its original prompt/HF metadata also matches the pinned control. This
output identity does not imply identical logits or ignored fidelity settings.

**Head4/LoFi last-layer-only qualifies: the required exception depends on
the complete head/decoder policy.** All seven actual outputs in
`qualitative_head4_lofi_last8_v1` were reviewed. Prompt 4 correctly describes
`"Bonjour" (good day) or "Salut" (informal hello)` and the question forms.
Prompt 0 remains coherent planning with no completed count mismatch; the
other prompts show no new concrete error relative to HF/prior TT. All seven
prompt/HF metadata records match the controls, and the runtime policy records
only layer 31's attention/MLP weights at BFP8. It passes AIME 92/100/100 and
the bounded smoke; see its [review](qualitative_head4_lofi_last8_v1/QUALITATIVE_REVIEW.md).
Unlike the head8/LoFi last-layer control, this complete policy qualifies.
The results do not identify a universal layer-0 defect or a required exception
independent of head precision.

**Head4/HiFi2 last-layer-only also qualifies.** All seven outputs in
`qualitative_head4_hifi2_last8_v1` were read against pinned HF and the qualified
head4/LoFi last-layer control. The TT token lists are identical to LoFi for
every prompt, including French and the haiku prefix; all nine input/HF-control
fields also match. No new concrete quality defect was found. This is a bounded
pass, not evidence of identical logits or unnecessary fidelity settings. See
its [review](qualitative_head4_hifi2_last8_v1/QUALITATIVE_REVIEW.md).

**Earlier selected K1 default passes: 2026-09-05 23:06:52..23:08:11 UTC.**
`logs/qualitative_selected_v1.provenance.json` records exit0 and a command with
no `--precision-config` or head override. The normal runner therefore uses
the selected-artifact default, and `qualitative_selected_v1` records
`head4_lofi_last8`. That complete recorded policy equals
`selected_precision_config.json` at review. All seven TT token lists and
saved completion texts are identical to `qualitative_head4_lofi_last8_v1`.
All nine input/HF-control metadata fields also match both the qualified
explicit-config run and pinned previous-stage controls. Each of the six shared
prompts has 128 TT tokens, and AIME has 100.

All seven actual outputs were read. French remains correct; the haiku has
coherent planning without a completed count mismatch, and no new concrete
quality defect appears in the other outputs. See the
[final default review](qualitative_selected_v1/QUALITATIVE_REVIEW.md).
This establishes the bounded qualitative default-path check for the recorded
32768-column/K1/two-reader head geometry. It does not claim validation of a
subsequent geometry change or completion of the whole stage.

**Geometry follow-up: the prior K1 verdict must be retested, not inferred.**
The new `head_geometry` is 32 cores, 32768 columns, K4 and two readers, versus
the earlier 64-core/K1 control. Both core count and accumulation block change,
so these runs test the complete geometry, not K4 alone. Full-model logits and
some greedy branches differ; the previous raw K1 rejection was not treated
as proof of the new result.

The new raw suite `qualitative_head4_c32_k4_v1` ran
23:31:59..23:33:16 UTC with exit0. Its current French says
`"Bonjour" (informal) or "Salut" (casual)`; all 128 French tokens match the
rejected raw K1 output. Prompt 1 changes at generated index 85, so geometry is
not generally output-identical. All seven outputs were read, and the current
French error disqualifies this candidate despite repeated AIME 93/100/100 at
median 87.391 t/s/u. See its [review](qualitative_head4_c32_k4_v1/QUALITATIVE_REVIEW.md).

The new last-layer suite `qualitative_head4_last8_c32_k4_v1` ran
23:33:16..23:34:34 UTC with exit0 and passes the bounded comparison. French
correctly describes Bonjour as `good day` and Salut as `informal hello`.
Six TT token lists exactly match the qualified K1 last-layer control; only
story prompt 2 changes, first at generated index 40, and remains coherent
planning. No new concrete quality defect was found in any of the seven outputs.
The policy retains only layer 31's attention/MLP weights at BFP8 and passes
every AIME repeat 92/100/100 at median 87.288 t/s/u. See its
[review](qualitative_head4_last8_c32_k4_v1/QUALITATIVE_REVIEW.md).

Both suites' nine input/HF-control fields match the pinned controls, their
recorded policies equal their explicit config files at review, and their
runtime head program records K4/per-core-N32/two readers. The exact commands,
geometry-bearing policy and exit statuses are recorded in their respective
`logs/qualitative_*.provenance.json` files. These are new explicit-config
qualitative controls; the final C32/K4 default result below supplies its own
separate invocation evidence.

**Final selected C32/K4 default passes: 2026-09-05 23:43:06..23:44:26 UTC.**
`logs/qualitative_selected_v2.provenance.json` records exit0 and the normal
runner command without a precision or head override. Its recorded policy
equals `selected_precision_config.json` at review and equals the qualified
`head4_lofi_last8_c32_k4_r2` explicit policy. `head_geometry` records 32 cores,
32768 columns, K4 and two readers; the runtime program independently records
K4/per-core-N32/two readers. Thus this check consumes the final selected
default, rather than reusing an explicit-config assertion.

All seven TT token lists and saved completion texts exactly match
`qualitative_head4_last8_c32_k4_v1`. All nine input/HF-control fields also
match that suite and the pinned previous-stage HF controls, with 128 TT tokens
per shared prompt and 100 for AIME. Every actual output was read: French
remains correct, the haiku has no completed count mismatch, the changed K4
story branch remains coherent, and no new concrete quality defect was found.
See the [final K4 default review](qualitative_selected_v2/QUALITATIVE_REVIEW.md).
This completes the bounded qualitative disposition of the selected policy;
overall stage closure is managed by the parent.

## Precision and quality ledger

These parent-generated AIME measurements are full 32-layer, batch-1, traced
teacher forcing with callback/logit readback on four Blackhole chips on two
physical P300c boards. They are not warmed token-out throughput rankings.

| Candidate/artifact stem | Head | BFP8 projection exceptions | AIME top1/top5/top100 | Decode t/s/u | Qualitative review |
| --- | --- | --- | --- | ---: | --- |
| `baseline_full_v1` | BF16/HiFi4 | None | 94/100/100 | 82.472 | Prior qualified TT passes |
| `head8_lofi_v1` | BFP8/LoFi | None | 92/100/100 | 84.994 | Fails |
| `head8_hifi2_v1` | BFP8/HiFi2 | None | 91/100/100 | 84.684 | Rejected: informal label remains |
| `head4_hifi2_v1` | BFP4/HiFi2 | None | 92/100/100 | 84.724 | Rejected: informal label remains |
| `head4_lofi_v1` | BFP4/LoFi | None | 92/100/100 | 84.574 | Rejected: informal label remains |
| `head8_lofi_edges8_v1` | BFP8/LoFi | Layers 0, 31 | 94/100/100 | 83.961 | Bounded shared suite passes |
| `head4_lofi_edges8_v1` | BFP4/LoFi | Layers 0, 31 | 94/100/100 | 84.137 | Bounded shared suite passes |
| `head4_lofi_first8_v1` | BFP4/LoFi | Layer 0 | 94/100/100 | 84.103 | Rejected: six-syllable line labeled five |
| `head4_lofi_last8_v1` | BFP4/LoFi | Layer 31 | 92/100/100 | 84.523 | Bounded shared suite passes |
| `head4_hifi2_last8_repeat5_v2` | BFP4/HiFi2 | Layer 31 | 92/100/100 | 84.262 (median/5) | Bounded shared suite passes |
| `head8_lofi_first8_v1` | BFP8/LoFi | Layer 0 | 93/100/100 | 84.607 | Rejected: French fixed, haiku counting fails at 256 |
| `head8_lofi_last8_v1` | BFP8/LoFi | Layer 31 | 93/100/100 | 84.576 | Rejected: informal label remains |
| `head4_lofi_c32_k4_r2_repeat5_v3` | BFP4/LoFi, C32/K4/R2 | None | 93/100/100 | 87.391 (median/5) | Rejected: current informal label remains |
| `head4_lofi_last8_c32_k4_r2_repeat5_v3` | BFP4/LoFi, C32/K4/R2 | Layer 31 | 92/100/100 | 87.288 (median/5) | New bounded shared suite passes |

The five-sample `*_repeat5_v2.json` controls give medians of 84.371 t/s/u for
head4/LoFi last-only, 84.262 for head4/HiFi2 last-only, 83.943 for head8/LoFi
edges, and 83.894 for head4/LoFi edges. Every last-only repeat retains
92/100/100; every edge repeat retains 94/100/100. LoFi/HiFi2 last-only timing
ranges overlap (84.237..84.515 versus 84.040..84.478), so the measured ordering
is not a precise universal speed difference. Parent selects head4/LoFi
last-only using its highest observed median and the bounded quality pass.

All rows retain BFP4/LoFi ordinary decoder projections outside exceptions,
BFP8/LoFi decode QKVG, BF16 activations/residuals/logits/sampling, FP32 recurrent
state and native CCL producer dtypes. Head FP32 accumulation remains enabled;
ordinary projection FP32 accumulation is disabled. Approximation is disabled
and packer L1 accumulation enabled for these projection/head matmuls. The
unchanged recurrent/SDPA compute policies are outside these projection groups.
The original rows use 64 head cores, 32768 local columns, K1 and two readers;
the geometry-follow-up rows use 32 cores and K4 at the same column count and
reader count. Final norm is sharded and prefill/decode tracing enabled.

The model context contract remains 262144. These qualitative/accuracy runs
allocate batch-1 context 2048: 32 pages of 64, contiguous per-request page
tables, tile-layout KV shape `[32, local_kv_heads, 64, head_dim]`, BFP8 cache,
`paged_fill_cache` prefill and `paged_update_cache`/paged SDPA decode.
The bad label predicts sequence index 58, using decode input/cache position
57 on page 0; no page boundary failure is demonstrated.

All seven actual edge outputs were read against HF, prior TT and raw head8.
No new concrete semantic error, wrong-language drift, mechanical repetition,
corrupt tokens or cross-prompt leakage was observed. Prompts 0/1/3/5 remain
coherent task reasoning; French is corrected; AIME restates the problem
coherently. Prompt 2 HF starts its story within the budget while all TT controls
remain in planning, an existing limitation. All edge TT outputs are unfinished
reasoning at the fixed budget. This review qualifies the bounded smoke and
does not establish complete responses, code execution or final AIME answers.

## Exact host evidence replay

Run from repository root; this rechecks the decisive artifact, tokenizer,
historical provenance agreement and exception facts without TTNN imports.
It deliberately compares recorded source hashes with each other, rather than
asserting that the final live source is unchanged:

```bash
USE_TORCH=0 python_env/bin/python - <<'PY'
import hashlib, json, pathlib, sys
from transformers import AutoTokenizer
r = pathlib.Path('models/autoports/ornith_ai_ornith_1_5_9b')
d = r / 'doc/datatype_sweep'
previous = r / 'doc/optimized_full_model/qualitative_prefill_trace_release_v2'
tokenizer = AutoTokenizer.from_pretrained('/home/hous/dev/ornith-1.5-9b/upstream', local_files_only=True)
fields = ['hf_model_id', 'prompt_text', 'chat_template', 'prompt_mode', 'tokenizer_class', 'rendered_prompt', 'prompt_token_ids', 'max_new_tokens', 'hf']
historical_runtime_hashes = None
for name in ['qualitative_head8_lofi_v1', 'qualitative_head8_edges_v1', 'qualitative_head8_lofi_first8_v1', 'qualitative_head8_lofi_last8_v1']:
    folder = d / name
    metadata = json.loads((folder / 'qualitative_prompt_format.json').read_text())
    assert metadata['revision'] == '489cb97981b8654bcfcf30ce1f94ed1b62e07b53'
    assert hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == metadata['template_sha256']
    assert hashlib.sha256(pathlib.Path(metadata['source']).read_bytes()).hexdigest() == metadata['source_sha256']
    for p in folder.glob('*/autoregressive_meta.json'):
        current, old = json.loads(p.read_text()), json.loads((previous / p.relative_to(folder)).read_text())
        assert all(current[k] == old[k] for k in fields)
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': current['prompt_text']}], tokenize=False, add_generation_prompt=True)
        assert rendered == current['rendered_prompt']
        assert tokenizer.encode(rendered, add_special_tokens=False) == current['prompt_token_ids']
    provenance = json.loads((d / 'logs' / (name + '.provenance.json')).read_text())
    recorded_runtime = {p: digest for p, digest in provenance['sources_sha256'].items() if p.startswith(str(r / 'tt') + '/')}
    if historical_runtime_hashes is None:
        historical_runtime_hashes = recorded_runtime
    assert len(recorded_runtime) == 10 and recorded_runtime == historical_runtime_hashes
raw = json.loads((d / 'qualitative_head8_lofi_v1/prompt_4/autoregressive_meta.json').read_text())['tt']['token_ids']
old_free = json.loads((r / 'doc/full_model/french_head_v1/french_bf8_lofi.json').read_text())['tokens']
old_prefix = json.loads((r / 'doc/full_model/qualitative_final_v1/french_branch_control.json').read_text())['tt_prefix_token_ids']
assert raw == old_free and len(raw) == 128 and raw[34] == 39102
assert [i for i, (a, b) in enumerate(zip(raw, old_prefix)) if a != b] == [14]
for name, expected in [('head8_lofi_edges8', [0, 31]), ('head4_lofi_edges8', [0, 31]), ('head8_lofi_first8', [0]), ('head8_lofi_last8', [31])]:
    policy = json.loads((d / 'configs' / (name + '.json')).read_text())
    assert sorted(int(i) for i, override in policy['layer_exceptions'].items() if override['weight_groups']['attention'] == 'bfloat8_b') == expected
haiku = json.loads((d / 'qualitative_head8_lofi_first8_v1/prompt_0/autoregressive_meta.json').read_text())
assert len(haiku['tt']['token_ids']) == haiku['max_new_tokens'] == 128
assert haiku['tt']['token_ids'][-1] == 20
assert tokenizer.decode(haiku['tt']['token_ids']).endswith('"Patterns in the data flow" (5')
extended = json.loads((d / 'qualitative_first8_haiku256_v1/prompt_0/autoregressive_meta.json').read_text())
assert all(extended[k] == haiku[k] for k in fields if k not in ('max_new_tokens', 'hf'))
assert extended['max_new_tokens'] == 256
for kind in ('hf', 'tt'):
    assert len(extended[kind]['token_ids']) == 256
    assert extended[kind]['token_ids'][:128] == haiku[kind]['token_ids']
assert 'da (1) - flow (1) = 5 syllables' in tokenizer.decode(extended['tt']['token_ids'])
assert not any(k == 'ttnn' or k.startswith('ttnn.') for k in sys.modules)
print('PASS: four suites, pinned tokenizer, historical source-hash agreement, prior BF8 identity, oracle prefix distinction, layer policies, exact haiku continuation; no TTNN import')
PY
```

## Final disposition and limits

The selected C32/K4 last-layer policy passes both its new explicit-config
suite and final default invocation; the raw C32/K4 candidate fails French.
The original covered quality failure is resolved under the selected policy.
The parent reports its separate final accuracy, token-out, native-context and
batch32 watcher checks passing and owns overall stage closure. This report's
completed verdict is the bounded qualitative check, with evidence from the
changed-geometry default run rather than the earlier K1 result.
The qualifying candidates' behavior beyond
the recorded 128-token budget is not established here. First8's AIME and French successes
do not override its confirmed haiku failure. Any narrower layer-0 weight-group
candidate needs full accuracy and qualitative gates. Higher precision is not
promoted from AIME alone.

Further localization is optional for any future reduced policy. The
smallest deeper localization is the exact common French prefix
through generated index 28, comparing the next-token logits at index 29 under
raw head8 and edge8. Capture layer-boundary hidden states, then substitute CPU
FP32 final norm/head on the same captured hidden tensors; use the current
prefix, not the old capture. A later probe at index 34 should force the raw
current prefix explicitly. If a broader decoder control becomes necessary,
keep head8/LoFi, BFP8 cache, pages and CCL unchanged while changing decoder
projection groups to BFP8/LoFi. That isolates projection sensitivity from cache
format changes. No cache/kernel defect is established by this report. The
selected C32/K4 default's bounded qualitative check is complete; this does
not independently certify the entire stage. No build was needed for this
report-only change.
