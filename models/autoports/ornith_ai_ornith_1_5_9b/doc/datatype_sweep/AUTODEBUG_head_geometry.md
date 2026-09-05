# AutoDebug: selected BFP4/LoFi LM-head geometry

Source-only diagnosis, 2026-09-05, before adapting the verification harness.
The parent owns all hardware and production integration. This investigation
uses no TTNN import, device access, or new performance measurement. The older
`optimized_full_model/AUTOFIX_head_geometry.md` conclusion used BF16/HiFi4;
its measured C64/K2/R2 allocation failure does not reject the selected precision.

## Finding and precision contract

The selected `head4_lofi_last8` policy uses BFP4 weights, BF16 input/output,
LoFi, FP32 destination accumulation, packer L1 accumulation, and no math
approximation. BFP4 reduces the weight tile from 2048 to **576 bytes**. The
same C64/K2/R2 static end falls from 1299456 to **734208 bytes**. This is a
verified source-level explanation for why the earlier L1 blocker is stale,
not evidence of a performance gain. One reader at K1 and larger K blocks on
smaller coherent input grids also become plausible.

The experiment fixes two logical 32768-column head chunks per rank, TP4,
hidden width 4096, one 32-row tile, the production common `LMHead1D`, and the
8x4 final norm with shard `[32,128]`. Baseline is explicitly C64/K1/R2,
independent of future model defaults. Both sides load the exact selected policy
and checkpoint weights; the same frozen real hidden comes from
`doc/full_model/french_head_v1/hidden.pt`, `hidden[0]`, with equal replicas
asserted. This hidden was captured in the prior full-model stage; it is a
component control, not a replacement for current-policy full-model accuracy
or French generation evidence. Full-stack persistent L1 is reserved at
**221952 bytes/bank**, as established in the preceding real-hidden probe.

## Native arithmetic and legality

Repository-relative source anchors:

- `tt_metal/impl/data_format/tile.cpp:67`: a 32x32 BFP4 tile has 512 payload
  bytes and 64 exponent bytes. Factory alignment to Blackhole's 64-byte DRAM
  boundary leaves 576 bytes, not 512 or the old BF16 2048 bytes.
- `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1305`:
  width-sharded L1 input, row-major orientation, M=1, and K plus input shard
  tile width divisible by `in0_block_w`.
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:146`:
  each DRAM-bank shard width must divide exactly across the readers. FP32
  destination subblock selection pads 43 reader tiles to 44 compute tiles.
  Lines 225-263 derive static CB sizes; lines 568-636 bind CB2/CB6 to dynamic
  input/output tensors, so those are not extra static allocations.
- `ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:341`:
  the native reader family is 1, 2, or 3; multiple readers require Blackhole
  NOC0. Secondary readers exclude input storage cores (lines 404-460), so
  changing input grid can change reader placement despite identical static CBs.
- `models/common/modules/lm_head/lm_head_1d.py:128`: both weight chunks run
  through the same linear/sharded-to-interleaved/concat path. Its tensor
  lifetimes are included in the probe, rather than timing one isolated matmul.

| Input cores / grid | Input shard | Legal K blocks | Input bytes/core | R1/R2 output tiles/core | R3 output tiles/core |
| --- | --- | --- | ---: | ---: | ---: |
| 16 / 8x2 | 32x256 | 1, 2, 4, 8 | 16384 | 64 | 65 |
| 32 / 8x4 | 32x128 | 1, 2, 4 | 8192 | 32 | 33 |
| 64 / 8x8 | 32x64 | 1, 2 | 4096 | 16 | 17 |

R1/R2 use physical N=32768. R3 requires physical **33024**, or 129 weight
tiles per bank = 3 readers x 43 tiles. Keep real columns at their original
indices, pad 256 zeros at the end of each rank's chunk, and trim each chunk
independently to 32768 after common-head projection. Storage `per_core_N` is
`ceil(physical_N / (32 * input_cores))`, not the reader compute width. R3
uses 16/32/61 output storage cores for C16/C32/C64 respectively.

For block width k, reader width rN, compute width cN, and this no-bias policy:

```text
CB0 = 2 * k * 2048
CB1 = 3 * rN * k * 576
CB4 = cN * 2048
CB5 = cN * 4096
predicted_static_end = 111616 + CB0 + CB1 + CB4 + CB5
```

All legal blocks here have more than one K loop; FP32 intermediate and BF16
output use separate CBs. The 111616 fixed base is inferred from preserved
allocation exceptions and must be checked if runtime/firmware changes. Static
size does not depend on input core count. Input/output allocations and reader
placement do.

| Readers | Physical N | Reader / compute tiles | K1 static end | K2 static end | K4 static end | K8 static end |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 32768 | 128 / 128 | 1123328 | 1348608 | 1799168 | 2700288 |
| 2 | 32768 | 64 / 64 | 619520 | 734208 | 963584 | 1422336 |
| 3 | 33024 | 43 / 44 | 460352 | 538752 | 695552 | 1009152 |

Physical L1 is 1572864 bytes. Persistent residency alone leaves an optimistic
frontier of 1350912; the frozen 8x4 normalized BF16 tensor consumes another
8192 bytes/bank, leaving at most **1342720** before input/output temporaries.
Therefore R1/K2 (1348608) already fails this necessary bound, and R1/K4/K8
are physically impossible. R2/K8 fails even the resident-only bound. Do not
run these points or illegal C64/K4/K8 and C32/K8. R1/K1, R2/K4, and C16/R3/K8
remain source-feasible; their dynamic allocation fit needs device validation.
This bound is specific to the fixed norm/residency experiment, not a universal
claim that R1/K2 cannot run under another lifecycle.

## Bounded verify/refute controls

Every candidate is paired with C64/K1/R2 at selected precision. The initial
high-value set is:

1. C64/K2/R2: revisit the stale blocker without changing input grid/readers.
2. C64/K1/R1, C64/K1/R3, C64/K2/R3: test all legal reader families on the
   selected grid, separating K and the adapted three-reader path.
3. C32/K1/R2, C32/K2/R2, C32/K4/R2: isolate input core count and exercise
   the larger K block that cannot be expressed on C64.
4. C16/K1/R2, C16/K2/R2, C16/K4/R2, C16/K8/R3: complete the two-reader
   coherent-core controls and exercise the largest source-feasible K block.

This is **11 candidates** plus the same selected baseline. If none wins, that
establishes only this bounded comparison. To close the complete existing
C16/C32/C64 x legal-K x R1/R2/R3 family, also run C16/K1/R1, C32/K1/R1,
C16/K1/R3, C16/K2/R3, C16/K4/R3, C32/K1/R3, C32/K2/R3, C32/K4/R3.
These **8 additional candidates** make 19 source-feasible alternatives plus
the baseline. Nonmonotonic core/reader placement means the old BF16 timings
cannot prune those interactions at BFP4. No broader tile, chunk-width, norm,
precision, or kernel rewrite is proposed.

The adapted stage-only `probe_head_geometry.py` offers a pure-Python `--plan`
without torch/TTNN imports. The device run records actual tensor dtypes, input
and weight layouts, fixed policy, source hashes, memory views, all-logit error
metrics, top-1/top-5, boundary error, eager/trace determinism and score hashes.
It does not save giant score tensors. Alternating paired trace medians measure
head plus reshard/trim, and final norm plus that head, separately. For watcher,
use `--serial-traces` to release the baseline trace before candidate allocation;
the old paired watcher allocation-tracker failure is preserved in the preceding
stage and must not be reproduced as a supposed kernel bug.

Example (the parent wraps device commands with its existing provenance runner):

```bash
OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 timeout 300 python_env/bin/python -m \
models.autoports.ornith_ai_ornith_1_5_9b.doc.datatype_sweep.probe_head_geometry \
--cores 64 --in0-block-w 2 --readers 2 \
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/head_geometry_c64_k2_r2_v1.json
```

## AutoFix handoff

Hypothesis: selected precision reopens material geometry choices excluded at
BF16. Verdict: **verified analytically; performance and correctness pending**.
The harness is a verification artifact only. No production edit or integration
is justified by source arithmetic. A measured winner requires current-policy
full-model numerical checks, full bounded qualitative controls if logits
change, watcher/trace checks for any retained geometry, and original end-to-end
performance gates. A local top-1 match on the frozen hidden does not waive those
checks. Hardware execution and any proven fix belong to the parent.
