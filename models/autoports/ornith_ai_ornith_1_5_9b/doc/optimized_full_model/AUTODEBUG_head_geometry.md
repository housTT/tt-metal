# AutoDebug: precision-locked LM-head input geometry

Source-only diagnosis, 2026-09-05. This report precedes the probe script and any
implementation edits. No TTNN import, device command, build, or new performance
measurement was run during this investigation. The parent owns hardware and
the selected model; this agent changes stage evidence only.

## Findings

Changing the 32-core input to 16 or 64 cores does **not** change the dominant
static weight circular buffer. It does change legal K blocks, multicast senders,
reader placement, and dynamic input/output allocations. Therefore the earlier
32768/K2/two-reader collision is not sufficient to reject a coherent 64-core
candidate: its smaller output shards may raise the live allocation frontier.
The static end remains 1299456 bytes, so fit is tight and needs a device probe.

Three readers have a stronger source-backed route. Pad each logical 32768-column
chunk to **33024 physical columns**, retaining the same BF16 weights and
BF16/HiFi4, FP32 destination accumulation, packer L1 accumulation policy. Eight
DRAM banks then contain 129 tiles each, divided exactly into three 43-tile
readers. K2's predicted static end is **918528 bytes**, leaving substantial
room for full-stack residency and dynamic tensors. Trim each chunk separately
back to 32768 after the common head, preserving local padded vocabulary 65536
and the original TP4 token order. Padding is 256 columns/chunk (0.78125%).

Neither arithmetic proves faster execution or full-model correctness. K changes
can alter accumulation and logits, and the earlier French policy rejection
remains binding. A real-hidden local top-1 match is not a French quality gate.

## Evidence and source contract

Repository-relative native sources:

- `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1303`
  requires width-sharded L1 input, row-major shard orientation, one tile row,
  `K_tiles % in0_block_w == 0`, and
  `input_shard_width_tiles % in0_block_w == 0`.
- The output planner in the same file at line 2513 derives output core count
  from `ceil(N_tiles/per_core_N)`. Program `per_core_N` controls **storage**
  width, not the reader compute width. It need not equal input-core count.
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:124`
  assigns one to three readers per DRAM bank. Lines 144–168 require per-bank
  weight width divisible by readers and, for multiple readers, equal to
  `readers * ceil(N_tiles/worker_count)`.
- Factory lines 160–202 derive reader width independently from input-core count
  and may round compute subblocks upward. With FP32 destination accumulation,
  43 reader tiles round to 44 compute tiles with a three-valid-tile tail.
- Factory lines 204–249 allocate double-buffered BF16 input, triple-buffered
  BF16 weights, BF16 output, and separate FP32 intermediate. Lines 567–637 show
  CB2 and CB6 alias allocated input/output tensors; they are dynamic allocations,
  not extra static payload. All static CBs span the sender/reader bounding box.
- `ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:341`
  allows readers 1–3; lines 404 onward exclude input storage cores from secondary
  reader selection. The 11x10 device grid leaves enough non-input cores even
  for the 8x8 input: 46 available for at most 16 secondary readers.
- `models/common/modules/lm_head/lm_head_1d.py:113` calls common `ttnn.linear`
  with sharded L1 output, converts each output to DRAM, then concatenates chunks.
  `_load_input_device_tensor` does not reshard an already-materialized tensor:
  the probe must explicitly match the candidate input shard.

Model evidence is under `models/autoports/ornith_ai_ornith_1_5_9b/`:

- `tt/model.py:175` currently uses input 8x4, shard [32,128], two logical
  32768-column weights, K1, readers2, and `per_core_N=32`. Terminal norm and pad
  use the same 8x4 input layout. Preserve this normalization in the experiment.
- `doc/full_model/probe_terminal.py` fixed the input layout at 32 cores; its
  optional 114688-byte reserve is obsolete for all 24 recurrent states.
- `doc/full_model/probe_french_head.py` uses the accurate persistent target
  **221952 bytes/bank**, including 25344 constants and **196608** recurrent
  state bytes. It reserves the missing amount as a full-grid row-major sharded
  BF16 tensor. The new probe follows that allocation, before terminal temporaries.
- `doc/full_model/AUTODEBUG_context_l1.md` derives recurrent residency as
  24 allocations × ceil(128 FP32 tiles / 110 banks) × 4096 = 196608 bytes/bank.
- `doc/full_model/logs/french_geom_bf16_hifi4_n32k_k2_r2.log.gz` records
  static end 1299456 and live frontier 1244416, collision 55040. K4's matching
  log records static end 2094080, beyond physical 1572864. Both are host
  allocation exceptions; neither is a hang or a reason to reset hardware.
- `doc/optimized_full_model/terminal_contract_probe.json` qualifies the current
  common head and fixed sharded norm at approximately 1.3678 ms. That probe
  did not reserve full-stack L1; new paired timings must state the reservation.

## Exact legal geometries and resource arithmetic

All candidates use hidden width4096 =128 tiles, input BF16 tiles32x32, M=1,
two logical32768 chunks/rank, output BF16, HiFi4, FP32destacc, packer_l1_acc.

| Input cores / grid | Input shard | Legal K blocks | Input bytes/core | R2 output storage tiles/core | R3 output storage tiles/core |
| --- | --- | --- | ---: | ---: | ---: |
| 16 / 8x2 | 32x256 | 1,2,4,8 | 16384 | 64 | 65 |
| 32 / 8x4 | 32x128 | 1,2,4 | 8192 | 32 | 33 |
| 64 / 8x8 | 32x64 | 1,2 | 4096 | 16 | 17 |

For R3, `per_core_N=ceil(1032/input_cores)`. This yields 16,32,61 output
storage cores respectively and no output-core-capacity issue. The input still
uses exactly 16,32,64 coherent shards covering K. The extra storage padding is
handled by the native output shape, followed by logical chunk trimming.

With K block `k`, reader width `rN`, compute width `cN`, no bias, all K loops >1:

```text
CB0 = 2 * k * 2048
CB1 = 3 * rN * k * 2048
CB4 = cN * 2048
CB5 = cN * 4096
static_end = 111616 + CB0 + CB1 + CB4 + CB5
```

111616 is the fixed base inferred independently from both previous R2/K2 and
R2/K4 errors. Firmware/dispatch changes may change it; the new errors and memory
reports must verify it. BF16 tiles already satisfy 64-byte DRAM alignment.

| Readers | Physical N/chunk | Reader / compute tiles | K1 static end | K2 static end | K4 static end |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 32768 | 128 /128 | 1688576 | 2479104 | 4060160 |
| 2 | 32768 | 64 /64 | 902144 | 1299456 | 2094080 |
| 3 | 33024 | 43 /44 | 650240 | 918528 | 1455104 |

One reader is physically impossible at this fixed precision and chunk width,
even K1. R2/K4 is physically impossible. R3/K4 fits physical L1 but exceeds
the optimistic resident-only frontier `1572864-221952=1350912`, before any
terminal tensors, so it is rejected for the full stack. C16/K8 is still larger.
Do not run these source-refuted points.

Changing C32 to C64 saves 4096 bytes per input shard and 32768 per R2 output
shard. Common LMHead can transiently retain the previous sharded output during
the next `linear` assignment, so exact allocator lifetime matters. Keeping norm
at 8x4 adds an explicit reshard for C16/C64; do not assume the saved bytes alone
prove the 55040-byte collision resolved. Probe head-only and terminal paths at
the same resident target, keeping both chunk executions in the common module.

## Minimal verify/refute sequence

`probe_head_geometry.py` is a stage-only experiment, with host-only `--plan`
that imports neither torch nor TTNN. A device invocation evaluates one candidate
alongside C32/K1/R2, on `full_model/french_head_v1/hidden.pt` dictionary
`hidden[0]`, with the same fixed8x4 final norm. It saves full logical logits,
hashes, top-1/top-5 comparison, boundary errors, memory reports, and exact
eager/repeated/trace checks. It measures normalized-input head (including the
candidate reshard/trim) and full terminal separately with alternating paired
trace timings. The parent must serialize and record provenance.

Run these primary candidate invocations, each paired with the baseline:

1. C64/K2/R2: direct test whether larger input/output geometry rescues K2.
2. C16/K1/R2 and C64/K1/R2: smaller/larger legal input controls at selected K.
3. C32/K2/R3: smallest padded-reader repair to the dominant CB footprint.
4. C32/K1/R3: separate reader/padding effect from K2 accumulation/performance.

If the R3 path wins materially, compare C16/K2/R3 and C64/K2/R3 next. These
are required before claiming selected input geometry is fastest, but are not
needed to refute the original R2 fit limitation. An updated C32/K2/R2 host-error
control is useful if C64 passes or the measured CB base/frontier changed.

Sample pending command (wrap with the parent's stage provenance recorder):

```bash
OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 timeout 240 python_env/bin/python -m \
models.autoports.ornith_ai_ornith_1_5_9b.doc.optimized_full_model.probe_head_geometry \
--cores 64 --in0-block-w 2 --readers 2 \
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/head_geometry_c64_k2_r2.json
```

For source-legal candidates an allocation exception is a valid refutation;
preserve the failure JSON/log and close the mesh normally. A live stall instead
requires the existing tt-device-usage/tt-triage procedure before any kill/reset.
Watcher and profiler remain separate jobs. Only after a material paired win:
watcher/trace checks, all-layer top-k, full French generation with reviewed text,
then original full-model performance gates. No default or decoder policy change
is authorized by this source report.

## AutoFix status

Hypothesis A: larger coherent input/output sharding can rescue R2/K2 via the
dynamic allocation frontier. Verdict: **uncertain**, focused C64 experiment ready.

Hypothesis B: three-reader padded weights reduce the weight CB enough for K2.
Verdict: **source-supported, unverified on device**; predicted CB end918528.

No implementation fix has been applied. The hardware handoff and focused
measurements remain pending; there is no new speedup claim.


## Final stage closure

The earlier investigation status above is preserved as historical evidence.
The selected implementation and completed final gates are recorded in the
[stage report](README.md), [runtime audit](runtime_audit.md),
[final full32 watcher control](prefill_integration_full32_v2/summary.json),
[long exact replay controls](prefill_integration_long_v2.json), and
[final profiling report](tracy/README.md). Earlier pending experiments are not
claims that these final gates remain unrun; rejected hypotheses and failed
receipts remain preserved. Independent stage review owns the final verdict.
