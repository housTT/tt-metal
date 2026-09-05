# Full-model work log

## Latest acceptance evidence (supersedes historical candidates below)

Final policy: unchanged optimized TP4 decoder, selected BF16/HiFi4 vocabulary
head with32768 columns/Kblock1/two readers. BFP4/BFP8 head candidates passed
aggregate top-k but failed the visible French register regression; the complete
policy/geometry controls and exact L1 blockers are in `AUTOFIX_french_head.md`.
The fix selects a correct earlier free-running branch; it does not claim that
head precision repairs the frozen local `form`/`inform` ranking.

| Final gate | Result | Exact command/source/library provenance |
|---|---|---|
| Prefill |96/100/100% top1/top5/top100 |[prefill_final_v2](logs/prefill_final_v2.provenance.json) |
| Traced teacher forcing |94/100/100%;80.85 t/s/u |[teacher_final_v2](logs/teacher_final_v2.provenance.json) |
| Six HF/TT chat prompts,128 tokens |Coherence/regression pass; no degeneration |[qualitative_final_v2](logs/qualitative_final_v2.provenance.json) |
| B1 prompt128/generate128 |Warm TTFT47.46 ms;81.56 token-out t/s/u |[perf_final_v2](logs/perf_final_v2.provenance.json) |
| B1 prompt2048/generate128 |TTFT113.11 ms;80.77 token-out t/s/u |[perf_context2048_final_v2](logs/perf_context2048_final_v2.provenance.json) |
| Native262143/262144 prefill and last-position decode |Pass;5,445,806,592 DRAM bytes/device |[native_context_final_v3](logs/native_context_final_v3.provenance.json) |
| All32 layers/32slots, mixed lengths, exact duplicate/permuted logits |Pass with worker watcher + allocation tracking |[full_batch32_final_v2](logs/full_batch32_final_v2.provenance.json) |
| Live scheduler/cache/sampling and request RNG |Pass with allocation tracking |[scheduler_sampling_final_v3](logs/scheduler_sampling_final_v3.provenance.json) |
| Host readiness/generator contracts |46 passed |[host_tests_final_v2](logs/host_tests_final_v2.provenance.json) |

The primary token-out loop has127 model and127 sampling replays, zero host
feedback/position/RoPE/page-table refreshes,127 pipelined output event waits,
and zero global device synchronizations. The separate logits-only model trace
measures85.61 t/s and has no sampling or autoregressive feedback. The cold first
generation TTFT463.11 ms includes program warmup and trace capture but excludes
weight loading. Detailed performance accounting is `performance_accounting.json`.

The six-prompt report is `qualitative_final_v2/QUALITATIVE_REVIEW.md` with hashed
prompt/output identities and passing degeneracy JSON. Both HF and TT mostly
remain in reasoning at128 tokens; completed answer/poem/code quality is not
claimed. Earlier failed `qualitative_final_v1` evidence remains intact and is
superseded by the selected-policy final suite.

Final source pre-commit hooks pass (`logs/precommit_sources_final_v4.log`).
Raw generated text and its metadata retain exact bytes; whitespace/EOF hooks
are not applied to those artifacts. An initial broad hook run added EOF newlines;
original text was restored from retained token IDs/JSON using the pinned
tokenizer, with the original French input metadata hash verified exactly.
No generated tokens or substantive output text were changed. The profiler CSV
reports also retain the tool's original CRLF bytes. Source/docs/JSON pass the
staged whitespace check; exact generated text and CSV whitespace are excluded.
The [artifact manifest](artifact_manifest.json) indexes immutable evidence by
path, size, SHA256 and repository-versus-retained-local storage.

The complete C++ CI-image build remains unverified: required wrapper commands
failed because Docker is unavailable. Both changed native kernels JIT-compiled
and passed focused hardware checks; the generic RMSNorm ROW/COL regression has
2 watcher passes. This is not a claim that the full host build ran successfully.

Reduced Tracy reports pass: `tracy/README.md`, per-rank CSV/text tables, and
compressed signposted operation windows. Decode sampling costs about0.58 ms,
under5% of measured full-model token-out latency. Runtime rows match decoder
BFP4/LoFi, QKVG BFP8/LoFi, recurrent FP32 and selected terminal BF16/HiFi4.
The original report failure was unreplayed trace definitions incorrectly required
to have device timing. `AUTOFIX_tracy_unreplayed.md` records the parser fix,10
passing CPU tests (one pre-existing missing-mock skip), and exact5772/5772 raw
execution/timing coverage. Fresh prefill profiling passes with3920/3920 rows.
No executed measurement was removed. The118% theoretical FLOPs display uses
eight instead of sixteen compute workers; the limitation and exact formulas are
in `tracy/head_roofline_classification.json`. No raw timing was altered.

All required runtime/accuracy/quality/performance/capacity gates now pass.
Independent [stage review](STAGE_REVIEW.md) returned **clean-pass** with no
required work. All773 manifest entries passed independent size/hash verification.
Local checkpoint SHAs are recorded below; no pushes or vLLM integration.

## Local checkpoint

Source hooks passed in full before committing. The combined evidence commit
skips only `trailing-whitespace,end-of-file-fixer` to preserve exact generated
text and profiler CSV bytes; all other hooks run at commit. The final source,
docs and JSON whitespace check passes.

The implementation/evidence SHA is recorded after the first local commit. A
second documentation commit records that SHA. The final receipt, including both
commit SHAs and clean-worktree verification, is retained at
`/home/hous/dev/ornith-1.5-9b/state/full_model_checkpoint.json` to avoid a
self-referential commit hash.

## Historical execution record

Stage started 2026-09-05 from branch `hous/ornith-1.5-9b`, commit
`c61ea5a4ca`. Initial working tree clean. Target checkpoint
`ornith-ai/Ornith-1.5-9B` revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`,
local snapshot `/home/hous/dev/ornith-1.5-9b/upstream`.

## Contract and environment

Use the completed optimized multichip decoder unchanged: TP4 on four Blackhole
chips on two P300c boards, native ring, BFP4/LoFi projections, decode QKVG BFP8/LoFi,
packed GDN and gate/up, BFP8 paged KV / BF16 update payload, FP32 recurrence,
BF16 residual and native row-projection reductions. Inter-layer outputs pass
directly to the next decoder. Prior precision/topology rejection ledger remains
`../optimized_multichip_decoder/optimization_evidence.md` and its AutoFix reports.
Native context remains 262144; temporary 2048-token probe allocation is not a
capability reduction. Full-stack capacity accounting remains pending.

`timeout 60 tt-smi -ls --local`: passed; four P300c Blackhole chips.
TP4 open/close with `fabric_router_config()` passed; exact source command is in
this session, output `logs/mesh_smoke.log`. State and logs are on persistent
`/home/hous/dev/ornith-1.5-9b` storage. All TT commands serialized.

## Initial implementation and reduced probe

Added `tt/model.py` and `tt/generator.py` (implementation in progress).
The probe uses real layers 0/3, real embedding/norm/LM-head weights, and native
TP4 partition sizes. Main commands:

```
OMP_NUM_THREADS=8 timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe
```

First command failed in Torch/Transformers import before opening devices;
`logs/probe_initial.log`. With the previous stage's documented persistent Torch
cache environment, imports and real model loading pass (10.31 seconds). The
terminal DRAM-sharded BF16 LM head requests 7,992,320 bytes of static L1 circular
buffers, exceeding 1,572,864 bytes/core. `logs/probe_cache_env.log`.
AutoFix source diagnosis requested in a fresh xhigh subagent. No decoder
precision changes. `timeout 180 tt-smi -r` output `logs/reset_terminal.log`.

## Readiness dependency AutoFix

This checkout has no `models/common/readiness_check/`. The pinned workflow commit
contains it. A fresh xhigh investigator is checking minimal restoration and
current Transformers/tokenizer compatibility. No references or accuracy/perf
claims from the old 35B checkpoint are reused.

## Status

In progress. No full-model accuracy, qualitative, performance, context execution,
trace-feedback or independent stage-pass claim yet. No commits or pushes.

## Completed initial gates (later than the initial status above)

- AutoFix readiness: `AUTOFIX_readiness.md`. Restored pinned non-serving runners,
  normalized BatchEncoding, and fixed exact chat-template token handling in the
  autoregressive runner. 30 host tests pass. Fresh reference
  `../../readiness_aime24_chat.refpt` and `.meta.json` has161 prompt tokens,
  exactly100 generated tokens and K100. The CPU BF16 HF loader reports no missing,
  unexpected or mismatched keys. Exact checkpoint revision and snapshot/template
  hashes are in the metadata. HF output read directly: coherent English math
  setup, truncated naturally at100. Cached HF generation differs from full-prefill
  rank1 once at reference index82 (rank2); control top5/top100 both100%.
- AutoFix terminal: `AUTOFIX_terminal.md`. Rank-preserving BF16/HiFi4 head chunks
  fix the L1 reader-buffer limit. Eight8192-column blocks/Kblock4 win the matched
  terminal comparison (1.85ms versus1.95ms). Real head/norm oracle top5/top100100%,
  repeated eager/trace exact; these are terminal-only measurements.
- `sampler_greedy_comparison.json`: common sampler physical top32 with semantic
  k1/p0/temp1 yields exact CPU greedy tokens on32 distinct rows, traced0.574ms;
  force-argmax2.748ms. Reject the latter as4.79x slower. Both were measured onTP4.
- `run_checks teacher --output .../teacher_v1.json`: all32 layers, AIMEchat100,
  traced token-out; top1=94/100, top5=100/100, top100=100/100. Readiness token-out
  decode77.83t/s/u; initial TTFT includes trace warmup/capture and is not the warm
  primary128/128 headline. `logs/teacher_v1.log`, `teacher_v1.json`.
- `run_checks prefill --output .../prefill_v1.json`: all32 layers, logical261
  tokens, scores100 reference predictions; top1=95/100, top5/top100=100/100.
  `logs/prefill_v1.log`, `prefill_v1.json`.
- Reduced generator smoke B1/B4/B32 passes repeat determinism and device position
  advance. `smoke_v1.json`, `smoke_b4_v1.json`, `smoke_b32_v1.json`.
  B1 seven-step steady loop has7 model and7 sampling replays,7 output reads, zero
  token/position/RoPE/page-table host refreshes. Reduced model tokens are not
  meaningful text and are never qualitative model evidence.
- Mixed prompt API now projects/samples combined last-hidden rows on device,
  preserving arbitrary fixed slots. `trace_b4_v1.json` passes mixed127/131/3-token
  prompts, permuted page tables, no copies for unchanged tables, exactly1 copy
  for a changed table, persistent token feedback across3 steps, inactive hybrid
  state freezing, repeated seeded sampling, and greedy/sampled/greedy alternation.
- `run_qualitative --output .../qualitative_v1`: running six shared prompts at128
  generated tokens via standard `run_autoregressive(chat_template=True)`, pinned
  local HF model and one reused full TT generator. Each prompt keeps HF/TT raw
  completions and tokens. No final qualitative verdict until all outputs read.

New validation runs use `record_run.py NAME <command>` to keep immutable command,
source archives/hashes, native-library hashes, timestamps and exit status. All
TT commands are still serialized. No full-stage closure or commit yet.

## 2026-09-05 full-model contract expansion

- `trace_b4_v4`: pass. Mixed prompts, fixed slots, changed/unchanged tables,
  persistent sampled-token feedback, device position advance, seeded replica
  equality, greedy/sampled/greedy transitions, model-card top-k20/top-p.95/
  presence-penalty1.5 repeats and resets. `trace_b4_v2` exposed a list-versus-
  tensor mismatch at common prompt-penalty initialization; public ragged prompts
  now pad with sentinel-1 solely for the common penalty host setup. `v3` was an
  indentation error before device open, corrected in `v4`.
- `cache_contract_v1`: pass with TT_METAL_TRACE_ALLOC_TRACKING=1 and allocation
  tracebacks. Fixed slots0/3 exact logits, repeated logits exact,127+3 continuation
  PCC.9959235 and same argmax1076,10 external cache buffers unchanged during trace
  warmup, explicit KV/state reset zero, host greedy and callback tokens equal
  optimized device tokens. Reduced layers0/3 output is a structural probe, not
  a language-quality result.
- `trace_b32_watcher_v1`: fail, worker0,0 BRISC pending NoC read at completion
  in DRAM-sharded matmul in1 sender. Saved watcher log/kernel mappings under
  `watcher_failure/`. Process aborted itself. AutoTriage is investigating;
  no watcher bypass or stage pass. Reset, four-device list and mesh smoke logs
  are `logs/reset_watcher*`. ETH watcher remains explicitly disabled because
  this environment's supported earlier watcher contract covers workers.
- Host compatibility callback and EOS helper:6 source-only tests passed.
  Public returned completions stop at EOS; fixed-window `stop_on_eos=False`
  remains the measured workload; device mechanics unchanged.

### 2026-09-05 — AutoFix B32 state handoff

Verified and fixed a model-side `where` condition dtype violation: FP32 predicates
corrupted BF16 convolution buffers during slot transfer and inactive restoration.
Exact `[32,1,2048]` control is in `batch32_where_control.json`; BF16 predicates
match the CPU oracle bitwise on all ranks. Kept decoder and state dtype policy.
`batch32_localize_fixed_v2.json` proves repeated/permuted same-slot logits and
hybrid states exact, all-rank sampling equal to CPU greedy; cross-slot logits
retain a measured maximum 0.0625 difference with equal greedy winners.
`trace_b32_masks_fixed.json` passes the strengthened original B32 trace contract
with worker watcher plus trace allocation tracking, including identical-prompt
31-active and 32-active token equality. Exact commands, intermediate failed
probe logs and limitations are in `AUTOFIX_batch32.md`. No reset was needed;
hardware returned to parent after normal close. No commits made by this agent.

## 2026-09-05 native context and scheduler closure

`native_context_v2` passes all32-layer262143/262144 logical prefills in43.904s/
43.832s, and traced decode at262143 advances to262144. DRAM after the second
window is5,059,783,168bytes/device; trace buffers13,369,344bytes/device in the
separately reserved100MB region. No context reduction. Native long repeated-token
execution is capacity/position evidence, not a language accuracy benchmark.

The original L1 conflict is fixed by a full-model-only GDN large-prefill output
block cap6, preserving11x10grid, K16, FP32 GDN input, BF16 output and BFP4/LoFi
weights. Exact native projection output match and actualfullstackL1reservation
are in `context_l1_probe_v2.json`; prior decoder defaults are unchanged.

`trace_b32_masks_fixed` passes workerwatcher plus trace-allocation checking. The
initial fixed-slot corruption was a FP32 WHERE predicate applied to BF16 conv
history; dtype-matched predicates preserve all previous slots exactly. Source,
exact-shape before/after test, A/A2/B cache/logit evidence and strengthened
all32active/31active free-run contracts are in `AUTOFIX_batch32.md`.

`scheduler_sampling_v2` passes exact32-bit lane merge, live sampling-mode changes
without cache/token/position mutation, continuing trace replay, partial-prefill
preservation of ongoing token/RNG/penalty state, and new-slot decode join. The
UINT32 predicate control reproduces truncation; signed INT32 predicate preserves
UINT32 values exactly. TTFT now includes cache reset, request parameter and seed
setup; prefill-only and setup components are separately retained.

### 2026-09-05 — RMSNorm row-order follow-up

Persistent traced boundary copies localized the remaining ≤0.0625 cross-slot
logit difference to Q/K RMSNorm. Frozen exact-width8/block_h32 norm controls
reproduce all selected full-trace rows bitwise on all four ranks. Native
all-to-all workers cyclically reorder partial sums across row groups; default
BF16 accumulation makes this visible. Diagnostic FP32 accumulation removes the
row differences, but runtime precision remains unchanged. Native validation
rejects direct height-sharded norm. See `AUTOFIX_norm_row_order.md`,
`batch32_boundaries_norm.{json,pt}`, `norm_rows_controls_v3.json` and
`batch32_norm_cpu_oracle.json`. Full-logit cross-slot PCC ≥0.9999645084;
original norm CPU-oracle PCC ≥0.9999918242. No decoder implementation change
from this follow-up; hardware returned to parent after normal close.

### 2026-09-05 — Canonical RMSNorm repair; full B32 gate passes

The earlier controlled-numerics qualification was superseded when all32 layers
actually diverged at slot6 token4. Fixed the native single-stage RMSNorm receiver:
cyclic NoC issue order is unchanged, but partials are stored into canonical peer
CB offsets so every worker accumulates in the same order. Precision/fidelity,
shards and communication count remain unchanged. Frozen norm outputs now match
across all32 rows and retain the original slot0/B1 anchor exactly on all4 ranks.
Norm timings are .024835/.024857ms Q/K versus .024813/.024835 before (~0.09%).
`full_batch32_canonical_norm.json` passes exact duplicate and permuted logits,
all32-layer greedy tokens, positions and device-only loop counters.
`trace_b32_canonical_norm.json` passes reduced B32 watcher+allocation tracking;
`norm_rows_canonical_b1_watcher.json` proves B1 anchors under watcher. Generic
ROW/COL duplicate-row RMSNorm tests both pass under watcher (2 passed), recorded
in `logs/norm_duplicate_rows_pytest.log`. Full commands and provenance are in
`AUTOFIX_norm_row_order.md`, which now leads with the retained fix and final gates.
The changed device kernel JIT-compiled; required CI build wrapper failed because
Docker is unavailable (`logs/norm_copilot_build.log`), so complete host build is
unverified. Black, native clang-format dry run, and diff whitespace checks pass.
No device reset or agent commit; hardware returned to parent after normal close.
