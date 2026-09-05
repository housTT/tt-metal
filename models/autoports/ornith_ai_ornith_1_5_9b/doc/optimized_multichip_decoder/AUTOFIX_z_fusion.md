# AutoFix: fused Z gating

Status: the mixed-format fused RHS failure is verified at the real TP4 decode
boundary. Two focused controls recover correct gating and projection. The
model-local swapped-LHS adaptation passes the batch-1 whole-layer/trace check
but fails the predicted batch-4 multiple-iteration case. Compact swapped-LHS
gating passes the batch-4 and batch-32 boundary controls. The production
adaptation passes four shorter batched HF checks and the affected user 31's
exact B32/T2048 HF check (whole-layer PCC 0.9989239221009265 on every rank).
Legacy-grid controls, four restored decodes, and two unchanged archived-v2
replays reproduce healthy boundaries. The original v2 constant observation
remains a historical non-reproducing anomaly with unknown cause; no semaphore
fix is credited for it. Retain the verified compact model adaptation. No
general binary kernel repair or overall stage pass is claimed.

## Starting evidence

Initial diagnosis: [AUTODEBUG_z_fusion.md](AUTODEBUG_z_fusion.md).
Original layer0, length2048 whole-layer experiment:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/record_run.py packed_gdn_fused_z_layer0 timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0 --length 2048 --variant packed_gdn_fused_z
```

The original fused RHS candidate fails decode PCC at -0.08527425494792604 on
all four ranks; its separate-BF16-SiLU control passes at 0.9999879890284807.
Both retain prefill PCC 0.9999630806578761 and exact eager/trace agreement.
Evidence: `logs/packed_gdn_fused_z_layer0.log.gz` and
`logs/packed_gdn_shared_mlp_layer0.log.gz`, with their provenance files.

## Focused experiment

Executed by the parent on the serialized device lane:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/record_run.py z_fusion_boundary timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --layer 0 --length 2048
```

Exit 0, 2026-09-05 08:17:27–08:17:34 UTC. Hardware: four Blackhole chips on
P300c boards, 1x4 ring; actual compute grid 11x10. Exit 0 means the diagnostic
completed and its separate-operation control passed; it does not mean every
tested variant passed.

Evidence: [log provenance](logs/z_fusion_boundary.provenance.json),
`logs/z_fusion_boundary.log.gz`, and `logs/z_fusion_boundary.sources.json.gz`.
Both compressed archive SHA256 values were independently checked against the
provenance. Executed diagnostic source SHA256:
`e89fcc21514fbfe4cb1b75d8b3de0d7f50557a259a832389bda4d9233051e575`.
Executed candidate source SHA256:
`1b592c559c794ffc7f67b119d9d569bb440fd9dfd9775a95898849a775856980`.
The source archives, rather than the evolving candidate file, define this run.
No raw tensor files were written.

The diagnostic runs the unchanged failing candidate through real-weight
prefill/decode, clones its `core` and raw Z at the output boundary, and applies
the alternative gates to the same reconstructed normalized/merged value. It
then passes each gate through the same `gdn_out` projection and native
all-reduce. All comparisons are rank-local before projection and cover all
four replicated outputs after reduction.

### Actual boundary contract

| Tensor | Logical shape per rank | Padded shape | Dtype | Memory |
| --- | --- | --- | --- | --- |
| Captured recurrent core | `[1,8,1,128]` | `[1,8,32,128]` | FP32 | Interleaved L1 |
| Raw Z | `[1,1,1024]` | `[1,32,1024]` | BF16 | Interleaved L1 |
| Normalized/merged core | `[1,1,1024]` | `[1,32,1024]` | FP32 | Interleaved L1 |
| Standalone SiLU(Z) | `[1,1,1024]` | `[1,32,1024]` | BF16 | Interleaved L1 |
| Every tested gate | `[1,1,1024]` | `[1,32,1024]` | FP32 | Interleaved L1 |
| Reduced `gdn_out` | `[1,1,4096]` | `[1,32,4096]` | BF16 | Interleaved L1 |

All are tiled. Captured core, raw Z, and merged values are finite on every rank.
The BF16-to-FP32 Z cast preserves every input value exactly on every rank.
No projection weight, recurrence, normalization, collective, or input data is
changed between gate variants.

### Results

PCC and errors below compare against the separate device SiLU/multiply path,
except the first control row, which compares against the CPU oracle. The CPU
oracle is captured FP32 merged value times BF16-rounded PyTorch SiLU of the
captured BF16 Z.

| Case | Gate PCC, minimum across finite comparisons | Gate maximum absolute error | Gate nonfinite counts, ranks0–3 | Reduced `gdn_out` PCC, every rank | Reduced `gdn_out` maximum absolute error |
| --- | --- | --- | --- | --- | --- |
| Separate BF16 SiLU then FP32 multiply vs CPU oracle | 0.9999938712338253 | 0.01434326171875 | `[0,0,0,0]` | Control | Control |
| Fused BF16 RHS | 0.00216854235201623 on rank0; undefined on ranks1–3 | 7.554183456880065e35 on rank0; undefined on ranks1–3 | `[0,2,3,1]` | -0.12429313058890092 | 9.503980169862148e37 |
| Fused FP32 RHS after exact Z cast | 0.9999986691455351 | 0.008253812789916992 | `[0,0,0,0]` | 0.9999996329993232 | 0.0009765625 |
| Swapped fused BF16 LHS, explicit FP32 output | 0.9999938712338253 | 0.01434326171875 | `[0,0,0,0]` | 0.9999996754432302 | 0.03125 |

Every fused variant's gate and reduced output reports `exact=false` against
the separate path on every rank. Standalone device SiLU itself differs from
the BF16-rounded PyTorch oracle by up to 0.015625. High PCC here establishes a
working local control, not bitwise equivalence.

## Hypothesis verdicts

- **Verified: the corruption begins in mixed-format fused RHS gating.** The
  exact captured inputs are finite; separate gating passes its CPU control;
  original fused RHS produces huge/nonfinite values before the output
  projection. The unchanged projection carries the bad result downstream.
- **Verified controls: same-format RHS and swapped BF16 LHS remove this local
  corruption.** The swapped control retains BF16 Z and a BF16 activation
  intermediate, so inherent inability to use BF16 Z is refuted. The FP32 cast
  is not sufficient evidence of a general precision limitation.
- **Source-backed causal explanation strongly confirmed: missing unpack
  format reconfiguration in the fused activation helper.** All predicted
  failing/passing configurations match the real boundary experiment. No
  register instrumentation or patched-kernel experiment was performed, so
  this is not a claim that an upstream kernel repair has been tested.
- **Refuted as sole explanations for this failure:** normal precision drift,
  a trace-only replay defect, or an output-projection-only defect. The
  diagnostic uses eager operations and directly observes the gate corruption.

## Upstream defect and why operand order matters

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`
starts unpack hardware on the post-LHS CB. For the failing invocation that is
FP32. It then invokes RHS activation preprocessing before its explicit binary
operand format switches.

`eltwise_utils_sfpu.hpp::preprocess_sfpu_impl` changes the packer format but
does not call `reconfig_data_format_srca` before copying raw BF16 RHS from its
input CB. `tt_metal/hw/inc/api/compute/tile_move_copy.h::copy_init` explicitly
does not change unpacker data types. The helper therefore reads a BF16 input
under the initial FP32 unpack format. Later binary input reconfiguration is
too late to recover the already-corrupted activated RHS.

The FP32 Z control removes the format mismatch. The swapped BF16 LHS control
starts hardware on the BF16 activated-LHS CB, so the first activation read
uses the correct format. It explicitly preserves FP32 gate output and the
original memory config:

```python
ttnn.multiply(
    z,
    merged,
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    dtype=ttnn.float32,
    memory_config=merged.memory_config(),
)
```

This is a validated **boundary adaptation for the tested shape**, not a
general repair of mixed-format activation preprocessing. At batch 1 there are
32 tiles and 110 worker cores: each active core processes one tile. A later
iteration on a core can inherit the other operand's unpack format. With
the same padded `[B,32,1024]` layout, batch 4 already has 128 tiles and some
cores process two. Batch coverage must therefore include a configuration
with multiple tiles per core before selecting the adaptation for all batches.
The parent owns that verification and any shape-dependent fallback.

The upstream repair boundary remains the preprocessing helper's own input
copy: explicitly establish the real `cb_pre` SrcA format before consuming it,
and audit the subsequent binary-copy state assumptions and shared callers.
Do not change `copy_init` globally or add a pre/post format comparison that
incorrectly assumes the live SrcA format equals the RHS post-CB format. No
binary kernel change is needed for this stage if a bounded model adaptation
passes every required contract and performance check.

## Precision and rounding ledger

Projection policy stays BFP4/LoFi for GDN and MLP; recurrent state is FP32;
residual and CCL payloads remain BF16. The local paged KV64 policy is untouched
and unused by this linear-attention layer. No weight, cache, or CCL precision
sweep is implied by these controls.

| Gate implementation | Z storage / activated CB | SiLU destination mode on Blackhole | Gate output |
| --- | --- | --- | --- |
| Separate unary then multiply | BF16 / BF16 | Unary FP32 destination disabled | FP32 |
| Original fused RHS | BF16 / BF16 | FP32 destination enabled; input-format defect | FP32 |
| Swapped fused LHS | BF16 / BF16 | FP32 destination enabled | Explicit FP32 |
| Cast Z then fused RHS | FP32 / FP32 | FP32 destination enabled | FP32 |

The unary dispatch in `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp`
disables FP32 destination accumulation for ordinary BF16-to-BF16 SiLU. The
binary factory enables it whenever an input or output is FP32. On Blackhole,
`ckernel_sfpu_sigmoid.h::_sfpu_sigmoid_` uses `exp_21f` and one reciprocal
iteration without FP32 destination mode, but accurate exponential and two
reciprocal iterations with that mode. `ckernel_sfpu_silu.h::calculate_silu`
also rounds internally to BF16 only when FP32 destination mode is disabled.
The fused LHS path eventually packs its activation into BF16, but its internal
SiLU calculation is different from standalone unary SiLU. This explains why
preserving storage dtypes does not establish identical rounding.

The FP32-RHS cast control additionally removes the BF16 activated-CB rounding
boundary; the cast itself changes no raw Z values. Neither passing control
may be described as bitwise equivalent to the separate operation policy.
The observed small differences must remain subject to the unchanged model
accuracy gates.

## Final status and remaining verification

Boundary failure localized and two controls verified. The parent subsequently
ran `packed_gdn_fused_z_lhs_layer0`: batch-1 decode PCC is
0.9999833000439305 on every rank, prefill PCC is 0.9999630806578761, and restored
trace equals eager exactly with max difference 0. The run reports decode latency
0.36200859176460654 ms. Evidence: `logs/packed_gdn_fused_z_lhs_layer0.log.gz`
and its provenance. This is one candidate measurement, not a claim that fusion
has won the final performance comparison. The original fused RHS path remains
a known failing diagnostic control.

Later batched boundary controls and shorter HF checks pass as recorded below.
They do not explain the changing upstream values at the exact 2048-token
batch-32 diagnostic boundary. That discrepancy remains open pending the
restored-state and semaphore-allocation controls.

This investigator reviewed archived logs and source and wrote this report;
no device commands or candidate/kernel edits were made during this follow-up.
The report is documentation-only and needs no build. Existing unrelated
working-tree edits are outside this finding. No performance improvement or
optimized-stage pass is claimed.

## Batched diagnostic and proposed compact adaptation

The parent requested extending only `tests/multichip_z_diagnostic.py` with
`--batch` (default 1, supported 1–32). The diagnostic uses distinct recorded
real rows: user `u` starts searching at source offset `137*u`, advances until
the final decode row differs from previously selected users, wraps within the
recorded source, and consumes `length` prompt rows followed by that user's next
row. The initial fixed-stride setup failure and its repair are recorded below.
User 0 with batch 1 therefore preserves the original inputs. All logical decode
positions equal `length`, and every user has a disjoint page-table row. State
is allocated by the current TP decoder's `allocate_state(batch)`; the resulting
recurrent/conv shapes, dtypes, memory choices, and L1 intermediate policy are
logged without overrides. Only the linear-attention layer uses these inputs;
KV allocation policy is left intact.

The existing three gate controls are retained. A fourth
`compact_fused_lhs_bf16` control reshapes both local gate inputs from
`[B,1,1024]` to `[1,B,1024]`, applies swapped-LHS SiLU/multiply with FP32
output, and restores `[B,1,1024]` before the unchanged projection. Device
readback verifies that compaction preserves each user's values and order;
restoration is checked exactly, with NaNs treated as equal only for this
movement check so a failed arithmetic hypothesis can still be diagnosed.
Accuracy/finiteness remain separately reported for every user and rank.
Actual padded tile counts are computed from tensor metadata.

Suggested serialized parent commands, not run by this investigator:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/record_run.py z_fusion_boundary_b4 timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --layer 0 --length 2048 --batch 4
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/record_run.py z_fusion_boundary_b32 timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --layer 0 --length 2048 --batch 32
```

**Source-backed compact adaptation:** the following replaces only the gate
multiply after constructing `merged`. The parent subsequently implemented this
reshape/restore approach in `PackedGDNAllModes` with `compact_z=True`; this
investigator did not edit the candidate file.

```python
public_shape = [batch, 1, self.cfg.linear_v_dim]
compact_shape = [1, batch, self.cfg.linear_v_dim]
z_work = ttnn.reshape(z, compact_shape)
merged_work = ttnn.reshape(merged, compact_shape)
gate_work = ttnn.multiply(
    z_work,
    merged_work,
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    dtype=ttnn.float32,
    memory_config=merged.memory_config(),
)
gated = ttnn.reshape(gate_work, public_shape)
for working, retained in ((z_work, z), (merged_work, merged), (gate_work, gated)):
    if working.buffer_address() != retained.buffer_address():
        ttnn.deallocate(working)
```

Both operands retain identical logical shape, so dispatch selects no broadcast.
For all supported batches 1–32, compact physical shape is `[1,32,1024]`, or
32 tiles. On the current default 110-core worker grid, each active core executes
one work iteration. SiLU is on the BF16 LHS, matching startup's format, so the
proposal removes the next-iteration hazard as well as the original first-RHS
hazard. This proof depends on that worker grid, width, and batch domain; a
different grid/domain requires reevaluating work distribution.

The reshape operations can involve real data movement because padded volume
changes. Their costs belong in the whole-layer timing, and their returned
buffers can alias at batch 1, hence the ownership checks. Public users, state,
positions, gate dtype, and the consumer layout are restored before projection.
The changed internal SiLU arithmetic described above still applies. Keep this
proposal only if the batched boundary probe and original whole-layer/HF/trace
gates pass and its measured total cost warrants selection.

Host checks on the extended diagnostic: Python AST parse and Black with
`--target-version py310`. No TTNN import or device command was run here.

## Batch-4 result and initial batch-32 harness failure

The parent ran the following underlying commands through the stage recorder
with the same cache/OMP environment as above:

```bash
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --layer 0 --length 2048 --batch 4
timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --layer 0 --length 2048 --batch 32
```

Evidence: `logs/z_fusion_boundary_batch4.{log.gz,provenance.json}` and
`logs/z_fusion_boundary_batch32.{log.gz,provenance.json}`.

Batch 4 completed at 08:27:55–08:28:07 UTC on 2026-09-05, exit 0. Actual work
counts are 128 public tiles versus 32 compact tiles on 110 cores. The swapped
BF16 LHS public-layout gate has nonfinite counts `[0,2,0,1]` by rank; finite
errors reach 2.6497751202594688e36. Users 0–1 are corrupted while users 2–3 retain
high PCC. This matches the work splitter: the first 18 cores each process two
adjacent tiles, covering the first 36 tiles; second iterations fall within
users 0–1. Later cores process one tile and do not encounter stale SrcA format.
Thus using LHS activation alone is refuted as a general batch 1–32 workaround.

The compact BF16 LHS gate stays finite for all users/ranks, with minimum per-user
PCC 0.9999938712338253 and maximum gate error 0.01434326171875 against separate
gating. Its reduced output has aggregate PCC 0.9999963681118038 on every rank,
minimum per-user PCC 0.9999996754432302, and maximum absolute error 0.03125.
Compaction and restoration preserve user values/order. The same-format FP32
RHS control also remains finite, with minimum gate per-user PCC 0.9999958805065036.
These are local boundary/projection results, not complete batched model gates.

The first batch-32 command exited 1 at 08:28:53–08:28:57 UTC before opening the
mesh. Its assertion was `duplicated real decode users`. This was a diagnostic
setup failure, not a device or model accuracy failure. The recorded layer0
source contains 2112 rows; distinct source indices at a fixed stride of 137
produced only 17 distinct actual final activation vectors across 32 users because
the recorded rows naturally repeat.

Repair: `distinct_user_offsets` starts each user's search at `137*u`, advances
one real source row at a time, and accepts an offset only when its final decode
row differs by `torch.equal` from every earlier user's final row. It wraps in
the same recorded source and raises a clear error if fewer than the required
distinct rows exist. No activation value is perturbed or synthesized. The
entire prompt-plus-token window moves together, positions remain `length`, and
both requested and selected offsets are logged. Batch 1 and batch 4 selections
remain unchanged.

A CPU-only check on the actual layer0 file extracted this helper from the
diagnostic AST without importing the module or TTNN. Batches 1, 4, 32 each produce
the requested number of distinct final rows; all offsets are distinct and
user 0 remains offset 0. Black and AST checks pass. The parent will rerun batch 32
as v2; no batch-32 device result is claimed from the failed initial attempt.

## Batch-32 v2: exact-constant comparison and unresolved upstream values

The parent ran:

```bash
timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --batch 32
```

Evidence: `logs/z_fusion_boundary_batch32_v2.log.gz` and its provenance.
The run entered the device path and exited 1 at 08:38:55–08:39:08 UTC on
2026-09-05, at the separate-gate per-user control assertion. It did not reach
the batched fused/compact comparison loop, so there is still no batch-32
compact-gate result from this run.

The log specifically shows user 31 with PCC 0, maximum difference 0, and no
nonfinite values on ranks 2–3. Ranks 0–1 have no failing user comparison at this
boundary. The original report of all ranks having zero PCC was therefore too
broad. The diagnostic's `H.pcc` helper centers both vectors and adds 1e-12 to
the denominator; exact constant vectors produce 0 rather than a defined Pearson
correlation. Treating that 0 as a mismatch rejected exact finite equality.
The current log does not show this user's per-boundary value range, so it does
not establish whether the equal constants are zeros or where they originate.

The input is not a padded zero row: CPU inspection of actual recorded source
row 2072 finds 4096 nonzero finite values, range `[-0.05712890625, 0.068359375]`,
and sample standard deviation 0.014307010918855667. None of the 2112 recorded
layer-0 source rows is all zero. The recorder uses real checkpoint embeddings
for layer 0 and HF hidden activations for layer 3;
there is no basis to explain the local constant result by replacing this user
with a fabricated zero input.

The runtime allocation and boundary shapes agree with the TP32 contract:
recurrent state `[32,8,128,128]` FP32 in DRAM; three convolution-state buffers
`[32,1,2048]` BF16 in DRAM; recurrent L1 intermediates enabled by the current
allocator; captured core `[32,8,1,128]` FP32 in L1; raw Z and merged core
`[32,1,1024]` BF16/FP32 respectively in L1. Captured values are finite in the
logged aggregate statistics. These shapes/finiteness checks do not establish
that every user's upstream recurrence/normalization values are correct.

Diagnostic correction, scoped to this helper:

- Keep the original numerical PCC value, including 0, in the log.
- Report `pcc_defined`, exact equality, constant flags, and the acceptance
  reason for each user. Accept finite exact equality directly. Otherwise
  require a defined PCC >=0.995. Unequal constants and nonfinite values fail.
- Emit per-user source-input/core/raw-Z/merged statistics: min, max, standard
  deviation, nonzero count, finiteness, and whether the vector is constant.
  This will show the earliest captured boundary with constant values on the
  rerun without changing those values or skipping the user.
- Include per-user pass flags and constant-reference user indices in the
  comparison summary and explicitly report
  `upstream_correctness_checked=false`. The CPU gate oracle consumes captured
  accelerator merged/Z values; matching that oracle cannot clear an upstream
  recurrence or normalization error shared by both gate paths.

The scoring change was checked on the CPU by AST-extracting the diagnostic
comparison helper and existing `H.pcc` implementation without importing TTNN:
equal zero/nonzero constants pass with raw PCC 0 and `accepted_by="exact"`;
unequal constants, constant-versus-variable inputs, low correlation, and equal
infinities fail. Python AST and Black checks pass. No device command or model
candidate edit was made by this investigator. The parent must rerun the
boundary diagnostic, inspect the new user 31 statistics, and retain the full
batched HF/model gates before claiming the candidate correct.

## Batch-32 v3: compact gating passes; changing upstream values remain open

The parent reran the same command as v2 with the richer diagnostic and the
current full-grid semaphore manager. Evidence:
`logs/z_fusion_boundary_batch32_v3.{log.gz,provenance.json,sources.json.gz}`.
The command completed at 08:46:50–08:46:58 UTC on 2026-09-05, exit 0.

Compact swapped-LHS gating is finite and passes every user on all four ranks:
minimum gate per-user PCC 0.9999931827148931, maximum gate error
0.01434326171875; reduced-output PCC 0.9999985313734151 on every rank,
minimum output per-user PCC 0.9999996512418013, maximum output error 0.03125.
Raw BF16 RHS and noncompact swapped BF16 LHS remain nonfinite/corrupt as
predicted. The same-format FP32 RHS control remains finite and highly
correlated. No user is accepted through an exact-constant exception in v3:
all `constant_gate_reference_users` lists are empty.

**The v2-to-v3 numerical change is unresolved, not explained by the scoring
fix.** In v3, user 31 has 1024 nonzero, nonconstant values in captured core,
raw Z, and merged core on every rank. The v2 gate comparisons were exact
constants on ranks 2–3, while v3 is variable there. Ranks 0–1 retain exactly
the same separate-gate PCC/maxdiff values between runs. The selected source
rows, positions, shapes, dtypes, weights, and native collective policy are
unchanged. V2 lacked per-user upstream statistics, so its earliest constant
boundary cannot be reconstructed from the archive alone.

Source-archive comparison identifies these relevant changes:

- `tt/multichip_decoder.py` creates `MeshCCLManager` with an actual 11x10
  semaphore grid instead of the inherited 8x8 grid. The manager creates 12
  global semaphores during setup. The default replicated/native path calls
  native `ttnn.all_reduce` and does not consume the manager's asynchronous
  semaphore handles. The old `_init_subdevice` merely constructs an unused
  `SubDevice` object; it does not load a subdevice manager onto the mesh.
  Therefore this change has no demonstrated direct arithmetic effect in this
  diagnostic, but it changes semaphore allocation/initialization coverage.
- The diagnostic computes additional host statistics and handles exact
  constants explicitly. Those changes do not alter TT tensor arithmetic, but
  additional host work can change enqueue timing. No device-state comparison
  across identical configurations was present in either run.
- The candidate source gained other candidate classes/imports; the executed
  `PackedGDNFusedZ` implementation is unchanged between these archives.

Allocation/timing sensitivity is a live hypothesis, not a proven diagnosis.
The disappearance of constants cannot be dismissed as expected BF16 behavior,
and a passing gate using captured tensors cannot clear upstream correctness.

### New focused controls, not run by this investigator

`tests/multichip_z_diagnostic.py` now accepts:

- `--legacy-semaphore-grid`: temporarily substitutes the original 8x8 manager
  only while `from_state_dict` constructs the decoder, restoring the module
  class in `finally`. The created manager retains its chosen setup, and the
  actual class/grid is logged. There is no runtime fallback or kernel edit.
- `--restored-repeats N` (default 1, range 0–3): snapshot the exact current
  recurrent and convolution state after prefill, verify snapshot equality,
  then restore the same state before each repeated eager decode. Compare
  core, raw Z, and reconstructed merged core exactly per rank/user before
  proceeding to gate variants. Check immutable state snapshots and the
  retained first core/Z/merged buffers after each repeat. Any mismatch is
  logged and raises before the gate comparison. State hashes cover logical
  tensor bytes returned by TTNN; unused physical padding is not hashed.

Setting repeats to 0 omits additional state snapshots and preserves the prior
diagnostic's device-allocation schedule. This matters when testing whether
snapshot allocation itself suppresses or exposes the symptom. Suggested
serialized commands, each recorded under a distinct stage label:

```bash
timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --batch 32 --legacy-semaphore-grid --restored-repeats 0
timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --batch 32 --restored-repeats 2
timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --batch 32 --legacy-semaphore-grid --restored-repeats 2
```

If the old grid reproduces constants and the full grid does not, the failure
is sensitive to this setup allocation boundary; that still does not make
asynchronous collective arithmetic its cause in a native run. If restored
core/Z changes with byte-equal logical state and input, localize the changed
producer before retaining the candidate. If core/Z is stable but merged
changes, focus on RMSNorm/reshape/permute or buffer ownership. If all restored
boundaries are stable, compare independently reset prefills and repeat the
setup allocation contrast: stable decode from a fixed state does not establish
that the state was computed correctly or identically across runs.

The parent also ran `ORNITH_MULTICHIP_CANDIDATE=packed_gdn_all_modes` with:

```bash
timeout 300 python -m pytest -x -q -s models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_decoder.py -k batched_prefill_decode_pcc
```

`logs/packed_allmodes_batch_contracts.{log.gz,provenance.json}` records four
passes: batches 4 and 32, both layer kinds. The test uses 96 prefill tokens and
its existing recorded-input seeds, then checks each user against HF. This
supports the candidate's shorter batched contract; it does not replace the
unresolved exact-input 2048-token case above.

The new controls pass Python AST and Black checks. No hardware command,
production candidate change, or binary kernel change was made by this
investigator. The remaining upstream discrepancy must be adjudicated before
the optimized-stage evidence is declared complete.

## Batch-32 v4: repeat and grid controls do not reproduce the anomaly

Independently inspected the following parent-run logs and provenance, and
verified every compressed log/source archive SHA256 against its provenance.
All use layer 0, length 2048, batch 32, the same distinct recorded windows,
position 2048, and the native collective path.

| Artifact prefix in `logs/` | Actual semaphore grid | Restored decodes | UTC on 2026-09-05 | Exit |
| --- | --- | --- | --- | --- |
| `z_legacy_batch32_v4` | 8x8, `CCLManager` | 0 | 09:05:28–09:05:36 | 0 |
| `z_restored_batch32_v4` | 11x10, `MeshCCLManager` | 2 | 09:05:36–09:05:44 | 0 |
| `z_legacy_restored_batch32_v4` | 8x8, `CCLManager` | 2 | 09:05:44–09:05:52 | 0 |

Each prefix has `.log.gz`, `.provenance.json`, and `.sources.json.gz` artifacts.
Commands are the three proposed commands above, with the absolute
`python_env/bin/python` executable. Executed diagnostic SHA256 is
`e6b014f4efc384671ef90c1781d9b93ae77155471fa6217670e0771ac98173fa`.
The recorded runtime binary hashes and tracked native-source hashes are
unchanged from v2 through v3 and all three v4 controls.

All three v4 runs have no constant or all-zero user at captured core, raw Z,
or merged core on any rank. User 31 has 1024 nonzero values at each boundary
on every rank. More strongly, the entire logged boundary records—including
every user's min, max, standard deviation and nonzero count—equal v3's records.
The separate-gate, separate-projection, compact-gate and compact-projection
records also equal v3 exactly. These are equal recorded statistics/comparisons,
not a cross-process bytewise comparison of the original tensors.

For each of the four restored decodes across the two repeated runs:

- Every recurrent/conv state restore matched the original prefill state's
  logical-byte hashes; all state snapshots remained unchanged.
- Every user on every rank had exact core, raw Z, and merged values relative
  to the first decode, with maximum difference 0.
- Retained first core/Z/merged tensors remained exact after the repeat.

The compact gate again passes every user and is finite on every rank, with
minimum per-user gate PCC 0.9999931827148931, reduced-output PCC
0.9999985313734151 on all ranks, and minimum per-user output PCC
0.9999996512418013. No exact-constant acceptance is used. Raw BF16 RHS and
noncompact swapped BF16 LHS still fail, independently confirming that the
healthy compact result is not caused by the bad fused kernels becoming healthy.

**Verdict:** the simple prediction that the old 8x8 setup necessarily produces
the constants is refuted by these controls. The full-grid change cannot be
credited as a verified fix for this native-path anomaly. Instability or snapshot
aliasing during the four tested restored decodes is also refuted. A rare
allocation/timing defect, a differing original prefill, or a readback/lifetime
defect is not excluded. The original v2 anomaly remains unreproduced and
unexplained; these passes do not establish that it was harmless.

### Tighter localization from the original v2 log

The v2 and v3 `standalone_silu` records have exactly the same aggregate and
per-user PCC/maxdiff on every rank, including user 31. Its user-31 PCCs are
0.9999999113686256, 0.9999997667235239, 0.9999998223149572, and
0.9999997206167037. Thus v2 already contains evidence of variable raw-Z/SiLU
values for that user; a constant raw-Z explanation is contradicted.

In `separate_gate_vs_cpu`, every per-user PCC/maxdiff also matches between v2
and v3 except user 31 on ranks 2–3. Those two v2 comparisons are PCC 0,
maxdiff 0; v3 has PCCs 0.9999999948398732 and 0.9999938231739598, with errors
0.0001277923583984375 and 0.0140380859375. The zero/error pair establishes
equal constant gate vectors under the helper's semantics; v2 did not log
their value, so "all-zero core" is still an inference, not an observed fact.
The remaining localization interval is the core producer, its normalization
and permutation/reshape, or observation of the resulting merged value.

### Strongest minimal remaining control

First rerun the exact v2 Python source archive in an isolated temporary package
overlay, preserving the original device operations, source versions, and host
read/report schedule. Run its original command twice before adding pre-failure
statistics or snapshots:

```bash
timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --batch 32
```

The overlay must use `z_fusion_boundary_batch32_v2.sources.json.gz`, including
its archived `tt/multichip_decoder.py`, rather than replacing files in the
shared working tree. Archive the replay source/hash and result separately.
V4's legacy manager restores the grid choice but does not restore v2's full
host object lifetime and reporting schedule.

If the original separate-gate assertion reproduces, add a failure-only
postmortem at that assertion, preserving the original failing result. Compare
the first saved merged host values with a synchronized second read copied into
owned host storage; reread core and Z similarly. Compute CPU RMSNorm of the
captured core with the uploaded `gdn_norm` weight and configured epsilon, then
apply the exact head reshape/permutation. Report per-user/rank finiteness,
constants, PCC and maxdiff against merged. Allow the known norm precision
difference; do not demand bitwise CPU/TTNN norm equality. This separates an
incorrect/constant core from normalization/data movement and from changing
readback. Capture logical-byte hashes before changing any device allocation.

Source support: the diagnostic's reconstruction is exactly
`rms_norm → reshape([B,H,1,D]) → permute(0,2,1,3) → reshape([B,1,H*D])`.
`ttnn/ttnn/operations/core.py::to_torch` calls `from_device`, whose default in
`ttnn/cpp/ttnn/operations/core/core.hpp` is `blocking=true`; `core.cpp` passes
that value to `Tensor::cpu`. Therefore a synchronization-sensitive readback
would require diagnosis as a defect; it cannot be dismissed as expected
asynchronous API behavior.

Independently establish present-production correctness with the exact B32,
2048-token real workload and HF for user 31's window (offset 24, final source
row 2072), retaining all 32 accelerator users and the actual TP4 state shapes.
`reference/hf_reference.py::reference_prefill/reference_decode` already provide
the HF path. Running HF for only this user's 2048-token window preserves the
relevant reference semantics while bounding CPU work. Compare that user's
whole-layer decode and, if it fails, local core/Z boundaries before the gate.
This is the smallest independent reference control of the affected user that
does not substitute a shorter prompt or a smaller accelerator batch. It does
not retroactively assign a cause to the v2 failure.

### Production adaptation evidence

`production_candidate_batch_contracts` passes all four batch/layer cases with
`pack_gdn=True` and the compact swapped-LHS implementation in
`tt/multichip_decoder.py`. `production_qkv4_c8_batch_contracts` also passes all
four with the lower QKVG dtype and its DRAM/MLP role configuration. Both have
four per-user HF-checked cases, covering batches 4/32 and layers 0/3. Their
layer-0 aggregate decode PCCs are 0.999267 and 0.999009; the first candidate's
layer-3 values are 0.999284/0.999148, and the QKVG4 c8 candidate's are
0.998832/0.998638. These logged numbers are rounded to six decimals.

These tests use 96-token prefills and their existing input seeds. They support
the production adaptation's batched contract but do not close the exact-input
v2 anomaly.

## Prepared archived replay and exact-input HF controls

`replay_z_v2.py` reconstructs only archived Python source files under a
temporary package overlay. The original canonical module name and repository
working directory are preserved; external repository dependencies remain
available. Relative data paths use symlinks to the existing HF configuration
and recorded activations, with those input files hashed in the manifest.
The overlay is removed after the runs. No duplicate source tree or raw tensor
files are retained in the repository.

The runner verifies the archive SHA256 against the original provenance and
every extracted source SHA256 against its recorded hash. It writes a small
`logs/z_v2_replay_<mode>_<id>.manifest.json` recording original/executed source
hashes, runner hash, input hashes, commands, times and return codes. An outer
`record_run.py` invocation should capture its streamed subprocess logs.

```bash
# Two separate processes with every archived Python source unchanged.
python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/replay_z_v2.py

# One archived process; additional work executes only after the original gate assertion fails.
python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/replay_z_v2.py --mode postmortem
```

Each child has a 300-second timeout. The two-run command requires an outer
timeout greater than 600 seconds. A timeout or signal stops further children;
the parent owns any required device recovery. Ordinary original assertion
failure remains a failure and is recorded separately for each process.

The postmortem mode changes only the archived diagnostic at its original
separate-gate assertion. It retains that assertion and, on failure only,
synchronizes and rereads core/Z/merged into owned host copies, checks retained
first host values, and compares merged against CPU RMSNorm/head assembly.
It logs per-user statistics, comparisons and logical FP32 host-byte hashes;
it creates no new device tensors. The first two unchanged runs do not contain
this block.

The current diagnostic now provides the independent production check:

```bash
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --batch 32 --hf-user31 --restored-repeats 0
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --batch 32 --hf-user31 --hf-variant production_qkv4_c8 --restored-repeats 0
```

`--hf-user31` requires layer 0, batch 32, and length 2048. Its default target
is `production_candidate`; `--hf-variant` selects another explicit production
trial policy. HF runs the unchanged user-31 window and captures the decode
inputs to its gated RMSNorm, providing independent core and raw-Z references.
The accelerator runs all 32 users with the current TP4 state allocation.
The diagnostic logs whole-layer output comparisons on all ranks plus the
corresponding rank-local core/Z/merged comparisons, then requires whole-layer
PCC at least 0.995 or finite exact equality. It exits after this check and
any requested restored repeats, without executing the known-bad gate controls.
Only user 31 is claimed as independently HF-checked by this option.

Host validation completed without TTNN imports or hardware: both reconstruction
modes pass `--check-only`; the unchanged extracted diagnostic retains SHA256
`1e1cbf849f6d1a42e1606ce35e0d14132c34b6ff6a23b2fff821891b59f0fee6`.
A fresh stdlib-only subprocess verified that canonical diagnostic, functional
decoder and HF-reference module resolution selects the archived overlay, and
that TTNN was not imported. Both new/updated Python files pass AST and Black
checks. Production and candidate implementations were not edited by this
investigator. Executed follow-up results are recorded below.

## Exact archived-v2 replay: both unchanged runs pass

Evidence: `logs/z_v2_archived_replay.{log.gz,provenance.json,sources.json.gz}`
and `logs/z_v2_replay_unchanged_4c328fd952.manifest.json`.
The parent executed the unchanged two-run replay at 09:19:31–09:19:46 UTC on
2026-09-05. Both separate child processes returned 0. Every executed Python
source hash in the manifest equals its original archived hash, including the
diagnostic's `1e1cbf849f6d1a42e1606ce35e0d14132c34b6ff6a23b2fff821891b59f0fee6`.
The replay log/source archive hashes match the outer provenance.

In both runs, every aggregate and per-user PCC/maxdiff for standalone SiLU,
the separate gate, compact gate, and compact projection matches v3/v4 exactly.
User 31's separate-gate comparisons on ranks 2–3 are now variable and healthy:
PCC 0.9999999948398732 / 0.9999938231739598, maximum error
0.0001277923583984375 / 0.0140380859375. Compact reduced-output PCC is
0.9999985313734151 on every rank in both processes. The original strict
per-user PCC assertion passes without the later constant-aware scoring change.

**Verdict:** the anomaly is a historical non-reproducing observation with
unknown cause. The original source, original 8x8 manager, and original
pre-assertion reporting schedule all reproduce the healthy later metrics.
Neither the new semaphore grid nor richer scoring is necessary for those
healthy results. This does not prove the historical observation harmless or
identify an upstream fix. There is no assertion failure on which to execute
the failure-only postmortem, so a postmortem replay is not indicated unless
the failure recurs. The exact-input production HF check remains the independent
current-correctness control; final long/batched watcher coverage belongs to
the parent's final stage validation.

The first exact-input HF attempt, `logs/z_user31_hf_current`, ran at
09:20:58–09:21:07 UTC and exited 1 after HF reference and device forward
completed. The diagnostic used `decoder.cfg.hidden_size` for the output slice,
but `OrnithDecoderConfig` names that field `dim`. No HF PCC was read, so this
attempt supplies no numerical verdict. The diagnostic now uses `cfg.dim`;
a host AST audit confirms that all of its `decoder.cfg` field accesses exist
in the actual config class, and Black passes. The parent owns the corrected
rerun.

A CPU-only postmortem control also passes: controlled constant first-read
merged values on user 31/ranks 2–3 are distinguished from a healthy reread
and exact CPU norm reconstruction, while saved first host values remain
unchanged. This validates the postmortem's reporting path only; it is not
device evidence about the historical anomaly.

## Exact-input production HF retry passes

Evidence: `logs/z_user31_hf_current_v2.{log.gz,provenance.json,sources.json.gz}`.
The parent ran the corrected diagnostic at 09:24:47–09:24:56 UTC on
2026-09-05, exit 0:

```bash
timeout 400 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic --batch 32 --hf-user31 --restored-repeats 0 --hf-variant production_qkv4_c8
```

The log and source archive SHA256 values match the provenance. The contract
confirms all 32 distinct real windows, length 2048, user-31 offset 24/final
row 2072, `pack_gdn=True`, native replicated TP4, QKVG BFP4 with eight DRAM
cores/block width 16/two readers, and two-reader DRAM MLP roles. The accelerator
batch and local state shapes were not reduced for this targeted HF check.

| User-31 comparison against independent HF | Minimum PCC across ranks | Maximum absolute difference across ranks |
| --- | --- | --- |
| Whole-layer decode | 0.9989239221009265 (all ranks) | 0.36062145233154297 |
| Recurrent core before normalization | 0.9999146574158444 | 0.09815406799316406 |
| Raw Z | 0.997165041047299 | 0.46020984649658203 |
| Normalized/merged core | 0.9991031592919498 | 1.0353031158447266 |

Every reported comparison passes the 0.995 PCC threshold. Core, raw Z, and
merged each contain 1024 nonzero, finite, nonconstant values on every rank,
including the historically affected ranks 2–3. Whole-layer output is finite
and nonconstant, with 4088 nonzero values out of 4096 on each replicated rank;
the reference has 4096 nonzero values. No exact-constant exception is involved,
and no bitwise or same-precision equivalence is claimed. HF was run only for
user 31; the other users are present on the accelerator to preserve the exact
workload and allocation contract.

## Final AutoFix conclusion

The mixed-dtype fused activation defect and its multiple-iteration extension
are verified. The compact swapped-LHS model adaptation is supported by the
focused TP4 B1/B4/B32 controls, the original whole-layer/trace gate, shorter
batched HF contracts, and the affected user's exact B32/T2048 production HF
check. Keep this measured model-local adaptation; no speculative semaphore or
general kernel change is justified by the historical constant observation.

The v2 observation remains unexplained and non-reproducing even under two
unchanged archived source replays. It is recorded as such, not classified as
harmless, normal precision loss, or causally fixed by the full-grid manager.
The targeted independent HF check now establishes current correctness for
the affected user at the original shape/input boundary. If the observation
recurs, the prepared failure-only postmortem provides the next localization
step. Final long-prefill/batch-32 watcher coverage and the overall optimized
stage decision remain with the parent; this report does not claim those
checks have run or passed.
