# Fused decoder work log

Stage 02, qualification complete, started 2026-10-10. Scope is `tt/fused_decoder.py`, tests and
stage documentation only. Functional runtime files remain unchanged. Initial
checkout is `53274a46` on `feat/qwen3-5-35b-a3b`; checkpoint revision is
`59d61f3ce65a6d9863b86d2e96597125219dc754`.

Startup: read the parent AGENTS.md, enabled plugin inventory and installed
model-bringup, graph-fusing, tt-device-usage, tracing, stage-review, AutoDebug,
AutoFix and telemetry contracts. Every model shell sources
`../scripts/environment.sh`, which validates both required plugin packages.
`timeout 60 tt-smi -ls --local` exited zero and listed all four Blackhole ASICs
on two P300c cards (`device_list.log`). No unrelated device job was running.
Tests use one ASIC, matching the functional stage. No reset or process kill.
Native 262144-token context, BF16 paged KV and independent GDN request state
remain required. Available storage is persistent and has 2.1 TB free.

The local evidence packet was initialized from the supplied template at
`/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/b499af22-a351-44ce-8553-1dfe93ca6aac.json`.
Performance is unknown until measured at the exact target. The runner already
owns telemetry delivery; no replacement runner or separate upload was started.

## Baseline and candidates

The accepted functional stage has whole-layer warmed device latency in us:
linear 1743870.348 prefill / 2190.255 traced decode; full attention
1720531.702 / 1608.712. Source and CSV references remain under
`../functional_decoder/`; these are baseline, never fused measurements.

The initial norm fusion folds the FP32 affine multiply into `rms_norm(weight=)`.
Command (all candidate logs are in this directory):

```bash
source ../scripts/environment.sh
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer \
  --decoder fused --layer 0 --length 129 --steps 8 --trace --audit \
  --output models/autoports/qwen_qwen3_5_35b_a3b/doc/fused_decoder/norm_linear_129.json
```

Exit zero: prefill PCC 0.999585688; minimum decode PCC 0.998990119,
identical repeated replay and device-only audit. This is an initial correctness
test, not candidate selection or headline performance.

`--fusions norm,router,expert_layout` tests normalized top-k logit softmax and
expert-major SwiGLU/down layout. Initial `three_linear_129.log` failed a
reshape: `fast_reduce_nc` exposed the physical 32-row tile to the shared-expert
add. The repair explicitly selects the single logical row before that add;
no padding enters the public output. Retry: `three_linear_129_fixed.{log,json}`.
First-failure evidence is retained. Final ranking and acceptance are pending.

## Extended fusion search (2026-10-10)

All hardware commands below use `source ../scripts/environment.sh`, one serialized
1x1 Blackhole mesh, pinned real weights, and the functional .995 per-token PCC
bar. Candidate command arrays and exit codes are in each directory's
`commands.json`; candidate JSON includes implementation hashes, HF PCC,
functional-output PCC, deterministic replay and explicitly **host** replay timing.
Host timings are screening evidence only, never device-time telemetry.

- `full_trials4096`: packed QKV, dedicated RoPE/head split, paired cache update,
  binary sigmoid gates and shared minimal-matmul epilogue passed 4096/128.
  Full-minimal minimum HF decode PCC .99576348; median host replay1277.008us
  versus1512.740us functional.
- `linear_trials4096`: FP32 grouped/ternary state, KDA convolution, RMS-based Q/K
  normalization, specialized router and shared epilogue passed 4096/128.
  Linear-fast minimum HF decode PCC .99784142; median host replay1642.3785us
  versus1924.249us functional. Native recurrence passes this particular workload
  but remains disqualified by the prior-stage B32 long-tail counterexample.
- `finalist_trials129`: all further GDN candidates passed: consolidated RM FIR
  plus FP32 tilize, in-place state update, flat chunk V, head-major chunk return,
  fused chunk program and their combined adapter. In-place update removes the
  persistent state copy; no state precision reduction was introduced.
- `positions_candidate`: both layer kinds pass the full32 independent requests,
  heterogeneous prefixes, permuted slots, traced per-row positions, deterministic
  repeated replay and full-attention sentinel guard pages.
- `reuse_candidate`: same-instance nonaligned tails, continuation and withheld
  length36 are tested with one live trace. Page32 and page128 are separate cases.

### Fused MoE AutoFix

The first fused FFN adapter's LoFi and HiFi4 failures were **not** proof of a
numerical kernel limitation. The isolated source investigation found that the
native reader bounds output region offsets against activation row capacity even
with `read_x_at_offset=False`; the old adapter silently skipped experts1..255.
`ffn_hypothesis.md` records exact source locations and the correction.
`ffn_repaired129` increases activation backing capacity. Both8x8 and8x2 then pass
real HF/functional equivalence; their median host replay2425.5235/1869.2775us is
slower than functional1479.0995us. Further compact-output adaptation is being
assessed before rejecting the dedicated FFN on performance.

`prefill_minimal129` tests tile-preserving horizontal expert packing and the
minimal-matmul SwiGLU epilogue. It passes real PCC. This candidate computes dense
expert gate/up work instead of the baseline's per-tile expert union, so a whole
prefill measurement is required before selection.

## Final candidate selection and correctness

`candidate_results.csv` indexes108 measured baseline/candidate records, with
exact commands in the corresponding `commands.json`. These are host latency
screens, not device rooflines. `graph_inventory.md` closes every skill pattern,
including dedicated native APIs beyond the skill's example list.

Compact eight-region FFN plus fused routing/reduction wins at4096/128. Grouped32
packed prefill wins over both unsplit and single-call32-split alternatives. The
selected GDN graph adds KDA tuple output, direct history tail, native fused chunk
and prefill norm/gate. Native decode and projection-residual are unselected after
adapted real-weight evidence. Native final B32 command used run_positions with
all final GDN flags except grouped/state_ternary/inplace, plus native_recurrence;
`native_positions_final_candidate.{json,log}` records failed_gate, minimum decode
PCC0.9884342. Selected FP32 recurrence passes0.9977745.

Final runtime SHA256 is
`a4c8f94f7d0275cdd7ea43d815474cc1763766965b5a1aa681fc82eb82f0a30a`.
Serial `run_fused_acceptance --group positions|reuse|pytest|watcher` commands and
exit codes are in `{positions,reuse,pytest,watcher}_final/commands.json`. All pass.
The source hash is unchanged during those runs. Thirteen CPU contract tests pass
(`cpu_contract.log`); an initial wrong expect_error-fixture invocation was repaired
and retained in `cpu_contract_initial_fixture_error.log`. Four actual device tests
pass, both target layer kinds with real and synthetic geometry. Watcher summaries
record2/3 completed dumps, clean device detach and zero error/assert/corruption
lines; full raw watcher logs are preserved.

Two hardware recoveries were required during experimentation (not at startup).
See AUTOFIX.md and the two triage directories for preserved evidence, owned PIDs,
bounded reset logs and successful mesh smokes. Neither triage result is used as
watcher-clean evidence.

## Independent review remediation

The fresh xhigh reviewer found an untested shared-down/gate/routed-add fusion
and a BF16 convolution-tap byte-estimate mismatch. The estimator now counts2
bytes/tap for fused KDA (functional retains4), reducing the named lower-bound
component by65536bytes. Tests-only `run_shared_epilogue_trial` adapts the native
epilogue without changing production code. Both4096/128 paths were measured
against the frozen default and functional tensors (`shared_epilogue4096`).
Linear is slower; full is slower and fails PCC. A separate coherent indexed-RoPE
adapter is being tested before final profile/closure. All four native-context
commands pass and context_contract.json now appends fused validation, with no
capacity reduction.

Indexed partial RoPE's adapted coherent4096 path passes but initially costs
928.654us decode. The natural-head caller removes its gratuitous layout round trip
and wins920.3985us versus924.030us frozen default, with minimumHF PCC0.99575436.
The exact native candidate is promoted to the default with construction-only
16-channel-group Q/K/gamma/cos packing, retaining original BFP8 coefficients.
New runtime SHA256: `eb835d592e90093bc7828ab0b978e7781c7436e071a0f1d0f20facb8a509d62d`.
The exact earlier a4c8 source is archived in qualified_a4c8_fused_decoder.py.gz;
indexed_production.diff proves the GDN branch is unchanged. The full-attention
positions/reuse/context/watcher gates are repeated under *_indexed_final; four
current-default device tests and both4096/128 headlines follow. No populated K
cache interchange with FunctionalDecoder is claimed: the fused first64 channels
use a consistent internal permutation; dtype/pages/context capacity are unchanged.

## Production reproduction anomaly and AutoFix control

The first indexed production headline had962.790us decode versus candidate920.399us.
This was investigated rather than attributed to noise. `placement_controls4096`
reproduces production962.3595, natural-layout trial920.675, retained-construction-
inputs920.4485 and old partial baseline924.276us, with matching indexed PCC.
The sole source difference retains the original norm/projection allocations until
all repacking finishes. Lifetime is controlled; physical placement is inferred,
not measured via address traces. Applied that bounded construction lifetime;
final default `placement_final/full.json` reproduces920.198us and minimumPCC
0.995754361. Source4792b57212a53a10f25313dfa56125dcd134090880476e8425a474d524448724.
Archived eb835 source and placement_production.diff preserve the exact change.
Full positions/reuse/watcher and full-only device profile are repeated. Native
context semantics/runtime memory are unchanged; no blanket repeat is needed for
GDN, whose AST is independently verified identical. The old full profile under
profile_final is retained as diagnostic; final full performance comes only from
profile_placement_final.

## Final evidence closure

The final candidate ledger contains119 measured control/candidate records.
On the exact4096-input/128-decode/batch1/single-request workload, whole-layer
device prefill changes1743870.348us to764671.617us for linear_attention and
1720531.702us to697717.625us for full_attention. Mean warmed traced device decode
changes2190.255us to1399.833us and1608.712us to978.654us respectively. Final
unprofiled host replay medians are1259.719us and920.198us; these reproduce or beat
the best correct candidate screens within run noise. See final_results.json,
profile_final/tracy/linear_attention and profile_placement_final/tracy/full_attention.
Both final roots preserve raw compressed CSV, whole-layer spans, provenance,
rooflines and prefill/decode tt-perf-report CSV/text. The denominator includes
all layer operations and gaps, not summed overlapping firmware intervals.

Final headline prefill/minimum-decode PCC is0.999676347/0.997841656 linear and
0.998591542/0.995754361 full. All exceed the original0.995 bar. Final full B32
positions, page32/page128 reuse and watcher runs pass under the placement_final
construction policy; paths are in acceptance_summary.json. Both kinds retain
native262144 and nonaligned262143+1 support. The indexed full K basis requires
conversion for populated-cache interchange with FunctionalDecoder; page
ownership, BF16 dtype and capacity are unchanged. context_contract.json records
the exact qualification scope and selected-HF-reference limitation.

The delivered source SHA256 is
e570144c58707bb0846be80075c0b26787e2b3cff8539e9a245bca739c1a168d.
Only explanatory comments differ from measured4792; normalized AST SHA256
9f54c7f13128ad6cca8e339b65ed3880bf7ce56d78efd7e13bf90965243ad79c
matches the archived measured source. source_provenance.json and the archived
a4c8/eb835/4792 sources preserve branch-specific qualification lineage. The
independent reviewer verified this lineage, raw timing calculations, warmup
program hashes and all complete measured timestamps.

cpu_contract_final.log records15 passing CPU contract checks (including indexed
guards); pytest_indexed_final records4 passing device tests. The CPU-only
nanobind teardown warning is reproduced in the unchanged functional control.
code_precommit_final.log records all14 delivered Python files passing the full
repository checks. No hardware jobs remain active.

The compact packet at the required external telemetry path now contains the
measured target workload, separate real layer kinds, whole-layer device times,
explicit useful-work/DRAM estimate assumptions and four accuracy records. No
full-model or serving performance is claimed. Review and local checkpoint
identifiers are appended after the independent verdict.

For portable evidence commits, the previously validated Python files are kept
unchanged and command-scoped SKIP=trailing-whitespace,end-of-file-fixer,black,
autoflake,isort,check-large-files preserves exact captured logs/CSV/source
archives. Every other repository hook remains enabled. This follows the
functional-stage artifact policy; raw measurements must retain their hashes.

The fresh xhigh independent reviewer returned **clean-pass** with no required
work remaining (`stage_review.md`). All review findings were fixed or rejected
with adapted measured controls and rereviewed. The final packet identities,
workload, roofline equations and source references were checked directly against
the supplied template. Local checkpoint follows this verdict; nothing is pushed.
