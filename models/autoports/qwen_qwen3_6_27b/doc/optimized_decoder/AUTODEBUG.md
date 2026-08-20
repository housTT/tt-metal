# AutoDebug: optimized-decoder stage-review blockers

## Scope and method

This is an inspection-only diagnosis of the `more-work-needed` verdict in
`stage_review_1.md`. No TT device was opened, and no implementation, test, or
other stage-document file was changed.

The required fresh-context command was run first:

```bash
.agents/scripts/autodebug.sh --agent codex models/autoports/qwen_qwen3_6_27b \
  "Performance-stage stage-review verdict: more-work-needed ..."
```

The nested Codex process and each of its four read-only forks failed before
workspace access with:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

It therefore could not create the repo-root `AUTODEBUG.md`. The findings below
were independently checked in the parent execution context against the current
source and retained artifacts. They are hypotheses until the hardware
experiments below verify or refute them.

## Direct observations

- The final path is still DRAM-interleaved at the decoder residual boundary.
  The only retained DRAM-sharded candidate is MLP-local, has 47 rather than 45
  decode operations, explicitly adds two layout conversions, and loses at
  1779.555 us versus its 1569.4-us interleaved control. This does not measure a
  residual/norm/MLP layout family that remains sharded across the two residual
  additions.
- `OptimizedDecoder.from_state_dict` assigns one dtype and one compute config to
  all three MLP weights. It cannot express mixed gate/up/down precision. The
  full-layer BFP4 report proves all three MLP matmuls used BFP4/LoFi, but its
  cited real-weight PCC 0.994177 has no retained correctness log.
- The fused-stage packed-MLP bundle is real and auditable, but it compares BF16
  packed versus BF16 separate. It does not answer whether packing gate/up at
  the selected BFP4 or BFP8 dtype beats a same-run low-precision separate
  control.
- The retained `gated_delta_attn_seq` candidate is correctness-valid and its
  dedicated kernel row is 141.937 us, but the adapter grows linear prefill from
  159 operations / about 5502.9 us to 344 operations / 6482.619 us. The result
  refutes that adapter, not the lower-movement use of the kernel. No repo-local
  recurrent GDN or dedicated Qwen gated-attention symbol was found in the
  checkout; the exercised sequence binding came from the installed runtime.
- `test_optimized_decoder_perf` now prints host E2E fields, but no retained
  artifact contains `E2E_US`. It also constructs only `OptimizedDecoder`; it is
  not a same-process fused-versus-optimized A/B. The current timing brackets do
  not explicitly synchronize the device at both measurement boundaries.
- Final raw profiler logs prove Blackhole compilation (`-DARCH_BLACKHOLE`), but
  the operation CSV `DEVICE ARCH` values are empty. Every final
  `tt-perf-report` therefore says it defaulted to Wormhole. The reported
  roofline percentages are not Blackhole-authoritative.
- `candidates/index.csv` currently parses with seven fields on every row, so
  the review's field-count defect has already been repaired. The static runtime
  audit also now includes inherited entry points, `_finish_layer`, and state
  paths; that review item has been addressed in source, subject to rerunning
  the test.
- Provenance remains inconsistent: `source_manifest.sha256` records test hash
  `5ad563...`, while the current test is `94ee4d...`; the exact-selected-default
  directory has JUnit XML but no `pytest.log`; the full-BFP4 candidate has
  profiler reports but no cited correctness log; `work_log.md` refers to a
  nonexistent `stage_review.md`; and README/work-log prose still presents the
  missing full-BFP4 PCC as conclusive. The candidate ledger now correctly calls
  that row inconclusive.

## Prioritized hypotheses and experiments

### H1 (highest expected value): mixed MLP precision can recover full-layer PCC while retaining most BFP4 bandwidth gain

**Prediction.** The full-layer all-BFP4 miss is localized to one or two of
gate/up/down. At least one of the six mixed BFP4/BFP8 policies passes PCC >=
0.995 and beats the all-BFP8/LoFi traced-decode control. If none passes or all
passing variants are slower, the selected all-BFP8 policy is justified.

**Smallest intervention boundary.** Only MLP weight loading and `_mlp` in
`tt/optimized_decoder.py`: permit per-role dtype/config, leaving residual,
attention, cache, and public semantics unchanged. Use real checkpoint weights;
do not use synthetic PCC as a veto.

**Experiment.** In one free-board session, run layer 3 for these policies:

```text
gate/up/down = 4/4/8, 4/8/4, 8/4/4, 4/8/8, 8/4/8, 8/8/4
```

For each policy run the existing non-aligned real-weight prefill/decode PCC
node first. Profile warmed prefill and ten-replay traced decode only for passing
policies, with an all-BFP8 control in the same session/cache state. Retain the
exact environment, source hash, stdout PCC log, raw op CSV, filtered report,
and per-role dtype/fidelity ledger. A first API/program-config error is a debug
result, not a rejection.

### H2: same-dtype packed low-precision gate/up can beat separate gate/up

**Prediction.** One BFP4 or BFP8 matmul producing concatenated gate/up removes a
same-input weight/activation traversal, but slicing and geometry may erase the
win. It must beat a well-tuned separate control under identical dtype,
fidelity, residual layout, checkpoint inputs, and warmed measurement.

**Smallest intervention boundary.** Add a candidate packed gate/up weight in
`from_state_dict` and a candidate `_mlp` branch in `optimized_decoder.py`.
Keep down projection separate. Do not retain duplicate packed and separate
weights outside the A/B candidate process because that can alter capacity.

**Experiment.** Compare, in the same run:

```text
linear layer: separate BFP4/LoFi vs packed BFP4/LoFi
full layer:   separate BFP8/LoFi vs packed BFP8/LoFi
```

Run real-weight PCC, warmed prefill, and traced decode for both layer kinds.
Record total ops plus the packed matmul and slice rows. The old fused-stage BF16
bundle is a useful implementation reference but is not a substitute for this
precision-locked A/B. If mixed precision wins H1, packed comparison applies
only to equal-dtype gate/up pairs.

### H3: a coherent sharded residual/norm/MLP family may remove the conversion tax that invalidated the MLP-local candidate

**Prediction.** The prior candidate lost because it sharded only at MLP entry
and immediately converted back. A candidate that holds the decode residual in
one width-sharded L1 geometry through RMSNorm, gate/up, activation, down,
residual add, post-attention norm, and the second residual add will remove those
boundaries and may beat interleaved. Conversely, if attention/GDN outputs or
RMSNorm/down-projection contracts force conversions at both residual joins,
the family is structurally invalid and the current interleaved path is
justified.

**Smallest intervention boundary.** `_finish_layer`/decode entry and the MLP
program/memory configs in `optimized_decoder.py`. Do not alter cache/state
ownership. A legal family must declare one residual shard spec and compatible
norm, gate/up, down, multiply, and add output memory configs; it is not enough
to use a DRAM-sharded matmul program with interleaved inputs around it.

**Experiment.** Start with batch-32 decode shape `[1,1,32,5120]`. Probe legal
width-sharded L1 grids (for example 32, 64, and the largest legal Blackhole
worker set) with per-core widths that divide the tiled hidden dimension. For
each grid:

1. statically print every input/output logical/physical shape, dtype, layout,
   memory config, shard spec, matmul program config, and compute config;
2. run one real-weight layer-3 PCC iteration;
3. inspect the profiler for conversions across the complete residual/MLP
   region;
4. only then run warmed traced decode against a same-run interleaved control;
5. repeat the best legal family on layer 0 because GDN output layout differs.

Reject a family only after resolving the first actionable API error or proving
an op contract cannot preserve the shard. This is an actual coverage gap, but
it is lower expected value than H1/H2 because the old valid sharded candidate
already lost by about 13%.

### H4: the issue-#50475 sequence kernel can win only if adapter movement is removed; recurrent decode remains an external capability boundary

**Prediction.** The 141.937-us kernel row leaves theoretical room, but the
current adapter performs enough padding, inverse preparation, transposes, and
tilize/untilize work to add 185 operations. Keeping Q/K/V/beta/A and recurrent
state in kernel-native tile/layout form across chunk boundaries, and consuming
the kernel output directly in the existing gate/output projection, should
remove most of that overhead. If the binding requires host-created inverses or
material layout round-trips per chunk, a wrapper-only fix cannot win.

**Smallest intervention boundary.** Linear prefill chunk preparation and result
consumption in `optimized_decoder.py`; use the installed composite only if its
runtime presence can be made an explicit supported dependency. A new TT-Metal
kernel or binding is outside this goal's allowed file scope. Recurrent decode
cannot be solved in Python if no recurrent composite exists.

**Experiment.** Build an operation ledger for the 344-op candidate and label
each row `required math`, `one-time preparation`, or `avoidable adapter`.
Implement only one movement removal at a time, preserving chunk-65 correctness,
then profile. Required gates are real-weight prefill PCC, first decode PCC,
state determinism, non-aligned logical length, and warmed prefill below the
best correct current 4681.596-us optimized result (not merely below the old
5502.9-us control). Separately run a static/runtime capability probe:

```python
hasattr(ttnn.transformer, "gated_delta_attn_seq")
# enumerate any recurrent GDN and fused gated-attention bindings and their help/signatures
```

Retain installed-runtime version/hash and binding provenance. If no recurrent
binding exists and repo search remains empty, record recurrent GDN and a
dedicated full-attention sigmoid-gate kernel as an upstream limitation. Asking
this stage to author a new C++/TT-Metal op would be out of scope; asking it to
remove avoidable Python-adapter movement is not.

### H5: the reported kernel speedups are credible, but the required same-run E2E and Blackhole roofline evidence is absent

**Prediction.** A synchronized same-process A/B will retain the direction of
the profiler win, though host E2E percentages may be smaller. A manual
Blackhole byte model will show decode's dominant weight matmuls remain
bandwidth-bound and will avoid the Wormhole default.

**Experiment.** Add a test-only A/B fixture that constructs `FusedDecoder` and
`OptimizedDecoder` from the same state dict, warms each path, synchronizes
before start and after completion, and alternates measurement order. Record at
least five samples for warmed prefill and traced decode for both layer kinds;
report median and range, not one wall-clock sample. Trace capture must occur
outside the timed replay interval, and the same replay count must be used on
both sides.

For the roofline, do not relabel the Wormhole report. Create a small documented
postprocessor over the retained raw CSV that:

1. derives bytes read/written from each op's physical tensor shapes, dtypes,
   memory locations, and BFP tile storage sizes;
2. records the authoritative p300c Blackhole per-device DRAM bandwidth source
   and assumption;
3. computes `bytes / measured_device_time / BH_peak_bandwidth` for each
   dominant op and the modeled phase;
4. separates unclassified head/cache composites rather than silently treating
   them as zero bytes;
5. cross-checks architecture from retained `-DARCH_BLACKHOLE` logs and a fresh
   `tt-smi` preflight artifact.

This is an evidence blocker because the user explicitly requested warmed
before/after timing and actionable profiler conclusions. It is not evidence
that the current kernel-duration measurements are wrong.

## Artifact and provenance repair experiment

Before the next stage review, run a deterministic ledger validator:

```bash
sha256sum \
  models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py \
  models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py \
  models/autoports/qwen_qwen3_6_27b/doc/context_contract.json
python - <<'PY'
import csv
from pathlib import Path
p = Path("models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/index.csv")
rows = list(csv.reader(p.open()))
assert len({len(row) for row in rows}) == 1
PY
```

Then verify every evidence cell resolves to a concrete file, every headline
PCC has stdout/JUnit provenance, every reported profiler total points to a raw
CSV plus signpost/replay rule, and every claimed review file exists. Regenerate
`source_manifest.sha256` last. Retain the exact selected-default stdout log and
rerun the full-BFP4 real-weight correctness case; until then label its PCC and
all mixed-policy conclusions inconclusive. Explicitly label full decode as the
mean of ten replays and linear decode as one replay. Repairing these records
does not require a runtime-code change.

## Actual blockers versus overreach

| Review request | Diagnosis |
|---|---|
| Coherent residual/norm/MLP sharding | Actual untested optimization family; H3 is the focused test. |
| Mixed MLP precision | Actual gap because all-BFP4 narrowly misses PCC and the implementation cannot express per-role policy. |
| Packed MLP | Actual comparison gap only at the selected low precision; the BF16 packed candidate itself is already well evidenced. |
| Lower-movement sequence GDN adapter | Actual gap; the retained result rejects only a high-movement adapter. |
| New recurrent GDN/gated-attention kernel | Upstream/out-of-scope if no installed or repo-local binding exists; this stage may prove the boundary but may not add TT-Metal files outside the authorized path. |
| Same-run synchronized E2E | Actual missing acceptance evidence. |
| Blackhole roofline | Actual analysis gap, but it does not invalidate raw device durations; use a manual BH byte model rather than fabricated CSV metadata. |
| Candidate CSV field counts | Already fixed; repeating the request is overreach unless the validator regresses. |
| Expanded inherited static audit | Already implemented; only a rerun/artifact is needed. |
| Provenance/ledger cleanup | Actual and independently confirmed. |

## Recommended AutoFix order

1. Repair artifact provenance and collect synchronized same-run controls so
   every later candidate has a trustworthy comparison basis.
2. Test mixed per-role MLP precision, then same-dtype packed gate/up.
3. Test one coherent residual/norm/MLP shard family across the full boundary.
4. Reduce sequence-GDN adapter movement one operation family at a time and
   prove the recurrent/full-gate capability boundary.
5. Rerun correctness, Watcher, profiler, manual Blackhole roofline, and stage
   review only for candidates that beat the same-run control.

Final diagnosis: **still failing stage evidence/coverage, not a demonstrated
correctness bug**. The existing selected path has credible PCC and raw
device-duration speedups, but clean-pass is blocked by H1-H5 and the confirmed
provenance defects. The fresh-context AutoDebug runner itself is also limited
by the nested bubblewrap failure described above.
