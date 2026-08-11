# AutoFix Report - sdpa_decode cross-core tree reduction

**Verdict: FIXED.** `max_cores_per_head_batch > 1` on
`ttnn.transformer.paged_scaled_dot_product_attention_decode` is now correct at every decode
position. The one-line cause is a DEST-register bounds violation that only bites under
`fp32_dest_acc_en`, which is why the defect looked like a runtime divisibility condition.

## Starting Evidence

No AUTODEBUG.md/AUTOTRIAGE.md existed for this defect; the starting report was the recorded
defect in the model port:

- `models/autoports/qwen_qwen3_6_27b/tt/functional_decoder.py` lines 151-178
  (`SDPA_DECODE_K_CHUNK` / `SDPA_DECODE_CORES_PER_HEAD` docstring): "the cross-core tree
  reduction is wrong for most positions ... correct only when `num_k_chunks` is 1 or a multiple
  of `2 * cores_per_head`".
- `doc/context_contract.json` -> `upstream_defects_affecting_long_context.decode_sdpa`.
- `doc/functional_decoder/logs/sdpa_decode_cfg_sweep_v2.log` (stock build): alpha 3705.6 at
  position 1023, NaN at 261887, 37.7 at 262143 with the default program config.
- Model-free reproducer: `doc/functional_decoder/probes/probe_sdpa_decode_synthetic.py`.

Original failing check (stock build):

```
POSITIONS=1023,4095,12287,16383,65535,131071,261887,262143 KCHUNK=128 MAXCORES=16 QCHUNK=32 \
  python doc/functional_decoder/probes/probe_sdpa_decode_synthetic.py
  -> pos=1023  alpha=3705.57772
  -> pos=261887 alpha=nan
```

## Hypothesis Experiments

### H1 - the recorded rule is really `num_k_chunks % (2 * cores_per_head) == 0`

**Experiment.** `probes/probe_sdpa_decode_cores.py` (derived from the functional-stage probe;
adds a `max_cores_per_head_batch` sweep in one process and prints the derived work split).
Smallest possible reduction tree: `cores_per_head = 2` (root core 0 + one child core 1),
`k_chunk = 512`, positions chosen so `num_k_chunks` walks 1,2,3,4,5,6,8.

```
CACHE=8192 POSITIONS=511,1023,1535,2047,2559,3071,4095 KCHUNK=512 MAXCORES=2 \
  python probes/probe_sdpa_decode_cores.py     # logs/exp1_parity_2cores.log
```

| num_k_chunks | chunks per core `[root, child]` | alpha | ok? |
|---|---|---|---|
| 1 | [1, 0] | 0.99755 | yes |
| 2 | [1, 1] | 22.548 | **no** |
| 3 | [1, 2] | 17.721 | **no** |
| 4 | [2, 2] | 0.99791 | yes |
| 5 | [2, 3] | 0.99688 | yes |
| 6 | [3, 3] | 14.933 | **no** |
| 8 | [4, 4] | 0.99629 | yes |

**Verdict: refuted as stated, refined.** `num_k_chunks = 5` violates
`num_k_chunks % (2*cores_per_head) == 0` yet is exact, and the child's own chunk count parity
is irrelevant (3 chunks on the child at `num_k_chunks=5` is fine). The real predicate is **the
parity of the *reducer* core's own k-chunk count**, and only when that core actually has a child
to combine (`num_k_chunks=1` leaves the root childless, so it is exempt).

### H2 - candidate wrong combination formulas

**Experiment.** `probes/probe_sdpa_decode_fingerprint.py`: computes both cores' exact local flash
states in float64 using the kernel's own convention (running max is the *unscaled* `q.k`, `scale`
folded into every exp), walks the real binary tree from `get_tree_reduction_params`, and compares
the device output against 7 candidate formulas (correct, missing `scale` in `exp_max_diff`,
dropped local/child `l`, uncorrected `l`, swapped correction factors).

```
CACHE=8192 POSITIONS=1023,2047,3071 KCHUNK=512 MAXCORES=2 \
  python probes/probe_sdpa_decode_fingerprint.py   # logs/exp2_fingerprint_2cores.log
```

**Verdict: all refuted.** Every candidate is 14x-38x off at the broken positions. The kernel is
not computing a plausible-but-wrong reduction.

### H3 - solve for what the kernel actually produced instead of guessing

**Experiment.** `probes/probe_sdpa_decode_solve.py`: per head, cosine-align the device output
against candidate numerator *directions* and least-squares-solve the scalar divisor the device
applied.

```
CACHE=8192 POSITIONS=1023,3071,2047 KCHUNK=512 MAXCORES=2 HEADS=0,1,6 \
  python probes/probe_sdpa_decode_solve.py       # logs/exp3_solve_2cores.log
```

| case | cos(dev, O_root) | cos(dev, O_root+O_child) | solved divisor / correct L |
|---|---|---|---|
| pos 1023 (root count 1, broken) | **0.99967** | 0.7147 | 0.023 |
| pos 3071 (root count 3, broken) | **0.99977** | 0.7417 | 0.313 |
| pos 2047 (root count 2, ok) | 0.7687 | **0.99981** | 1.003 |

**Verdict: localised.** In the broken cases the child's output vector is *entirely absent* from
the numerator and the denominator is per-head garbage - i.e. the correction step's inputs were
destroyed, not mis-combined. That points at register state, not at CB plumbing (a full audit of
push/pop balance across `reduce_c` / `sub_exp_block` / `mul_block_inplace` /
`mul_block_bcast_cols` / `add_block_inplace` / `move_block` / `correction_block` found every CB
balanced per iteration, and every SDPA-decode CB is allocated exactly one block deep, so there
is no slot alternation to blame).

### H4 - `correction_block` overruns DEST under fp32 destination accumulation (VERIFIED)

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`
`correction_block` (line ~706) loads **five** tiles into DEST - `dst_reg_0..dst_reg_4` for
prev_max, worker_max, cur_max, prev_sum, worker_sum - and the SFPU kernel it calls,
`tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h`
`calculate_fused_max_sub_exp_add_tile` (line 232), addresses them as
`dst_reg[0 / 32 / 64 / 96 / 128]`.

But this very op's host side states the fp32 DEST budget itself -
`sdpa_decode_program_factory.cpp:387`:

```cpp
const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
```

Under `fp32_dest_acc_en` only **four** DEST tiles per half are addressable, so `dst_reg[128]`
(tile 4) lands outside the live half. Which live value that write destroys depends on which DEST
half is currently active, and the active half alternates with the number of
`tile_regs_acquire`/`tile_regs_release` pairs the core has already executed - i.e. with the
parity of the reducer's own k-chunk loop count. That is exactly the H1 predicate.

**Experiment (no code change, no rebuild).** Re-run the H1 sweep with `fp32_dest_acc_en = False`,
which makes five tiles fit in a half:

```
CACHE=8192 POSITIONS=511,1023,1535,2047,2559,3071,4095 KCHUNK=512 MAXCORES=2 FP32DEST=0 \
  python probes/probe_sdpa_decode_cores.py     # logs/exp4_fp32dest_off_2cores.log
```

alpha = 1.00643, 1.01010, 1.00837, 1.01161, 1.01000, 1.00762, 1.00745 - **every** position
correct, including all three that fail with fp32 dest acc on.

**Verdict: VERIFIED.** The cross-core reduction is not architecturally broken; it is broken only
in fp32-dest-accumulation mode, by a 5-tile DEST use in a 4-tile budget.

## Fix

One file, one hunk: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`
(the tree-reduction combine, ~line 550).

Under `if constexpr (DST_ACCUM_MODE)` the fused `correction_block` call is replaced by the same
arithmetic unfused, built from helpers that already exist in `compute_common.hpp` and that the
flash loop in this same kernel already uses, so no step ever needs more than 2 DEST tiles:

```
cur_max        = max_block(prev_max, worker_max)
exp_max_diff   = sub_exp_block(prev_max,   cur_max)
exp_max_diff_2 = sub_exp_block(worker_max, cur_max)
prev_sum      *= exp_max_diff            (mul_block_inplace)
worker_sum    *= exp_max_diff_2          (mul_block_inplace)
prev_sum      += worker_sum              (add_block_inplace<true>  -> pops worker_sum)
cur_sum        = prev_sum                (move_block<true>         -> pops prev_sum)
```

This reproduces `correction_block`'s exact pre/post CB contract (both sum inputs consumed;
`cur_max`, `cur_sum`, `exp_max_diff`, `exp_max_diff_2` produced; `prev_max` and `worker_max` left
produced for the caller to pop). Explicit `reconfig_data_format` / `pack_reconfig_data_format`
calls are added at each step, matching how the flash loop drives the same helpers.

Notes on the shape of the fix:

- **Scope.** The `else` branch keeps the fused SFPU path verbatim, so nothing changes for
  `fp32_dest_acc_en == false`. The fp32 branch is reached only with fp32 dest acc **and**
  `num_cores_per_head > 1`, a combination that returned garbage before this change, so there is
  no path that was correct and is now slower.
- **Deliberately not fixed here:** the underlying 5-tile `correction_block` /
  `calculate_fused_max_sub_exp_add_tile` pair. `correction_block` has exactly one caller in the
  repo (this kernel), and the register layout could be squeezed into 4 tiles by writing `cur_max`
  over the `worker_sum` slot (4 inputs, 4 outputs), but that means editing
  `transformer/sdpa/device/kernels/compute/compute_common.hpp` and `tt_metal/hw/ckernels/...`,
  both outside this stage's writable area. Recommended upstream follow-up.
- **No C++ rebuild needed.** Only a device kernel source changed; those are JIT-compiled from
  `$TT_METAL_HOME`. `build_Release` was not rebuilt (the host-side rebuild command, for reference,
  is in `doc/functional_decoder/logs/rebuild_sdpa_decode.log`: `ninja -C build_Release`).

## Verification - correctness

Device/float32-golden scale `alpha` (PCC is blind to this defect - it is a pure scale error).
Full 262144-token paged cache, shuffled page table, 24 q / 4 kv heads, head_dim 256, page 64,
`q_chunk = 32`, `fp32_dest_acc_en = True`, `exp_approx_mode = False`. Bar: alpha in [0.98, 1.02].

**k_chunk = 512** (`logs/exp6_matrix_k512.log`):

| position | 1 core | 2 cores | 4 cores | 8 cores | 16 cores |
|---|---|---|---|---|---|
| 1023 | 0.99506 | 0.99460 | 0.99460 | 0.99460 | L1 |
| 4095 | 0.99677 | 0.99549 | 0.99334 | 0.99291 | L1 |
| 12287 | 0.99730 | 0.99882 | 0.99722 | 0.99521 | L1 |
| 16383 | 0.99837 | 0.99711 | 0.99703 | 0.99427 | L1 |
| 65535 | 0.99931 | 0.99848 | 0.99995 | 0.99744 | L1 |
| 131071 | 1.00036 | 1.00638 | 0.99986 | 0.99723 | L1 |
| 261887 | 1.01684 | 0.97715 | 0.99716 | 1.00107 | L1 |
| 262143 | 1.00635 | 0.98026 | 0.99926 | 0.99869 | L1 |

**k_chunk = 256** (`logs/exp7_default_and_k256.log`):

| position | 2 cores | 4 cores | 8 cores | 16 cores |
|---|---|---|---|---|
| 1023 | 0.99394 | 0.99350 | 0.99350 | 0.99350 |
| 4095 | 0.99535 | 0.99322 | 0.99179 | 0.99177 |
| 12287 | 0.99873 | 0.99619 | 0.99532 | 0.99462 |
| 16383 | 0.99722 | 0.99759 | 0.99586 | 0.99308 |
| 65535 | 0.99857 | 0.99785 | 0.99652 | 0.99740 |
| 131071 | 0.98957 | 1.00785 | 0.99823 | 0.99618 |
| 261887 | 1.26410 | 0.98144 | 1.00093 | 0.99677 |
| 262143 | 1.28096 | 0.98038 | 1.00331 | 0.99675 |

Every catastrophic value is gone: no NaN, no 3705x, no 333x, and no dependence on
`num_k_chunks mod anything`. Note `num_k_chunks = 1023` at position 261887 - the position that
returned NaN on the stock build for every core count tested - is now 0.981 / 1.001 / 0.997 at
4 / 8 / 16 cores.

`16 cores` at k_chunk 512 raises `TT_THROW ... circular buffers ... grow to 1692672 B which is
beyond max L1 size of 1572864 B`. That is a **pre-existing L1 capacity limit**, present
identically on the stock build (`doc/functional_decoder/logs/sdpa_decode_cfg_sweep_v2.log`,
`kchunk=512 maxcores=16` row), not a regression: the receive buffer c_19 is sized
`(out_tiles + 2*PNHt) * (num_cores_per_head - 1)` and 16 cores at k_chunk 512 does not fit
alongside the double-buffered K/V.

**The two cells still outside the bar are not the tree reduction.** They are the *other*,
already-recorded defect - bf16 flash statistics - and they land exactly where that defect
predicts. `fp32_local_accumulators` in `sdpa_decode_program_factory.cpp:470` requires
`max_cores_per_head_batch == 1`, so every multi-core run keeps its running max / denominator /
output accumulator in bf16, and the error is a function of the **per-core** merge count:

| per-core merges | config | alpha at 262143 |
|---|---|---|
| 512 | 2 cores, k_chunk 256 | 1.281 |
| 512 | 1 core, k_chunk 512, bf16 stats (stock, quoted in the program factory) | 1.290 |
| 256 | 2 cores, k_chunk 512 | 0.980 |
| 128 | 4 cores, k_chunk 512 / 8 cores, k_chunk 256 | 0.999 / 1.003 |
| 64 | 8 cores, k_chunk 512 / 16 cores, k_chunk 256 | 0.999 / 0.997 |

The 2-core / k_chunk-256 number (1.281) reproduces the documented 512-merge bf16 figure (1.290)
to 1%, which is what identifies the residual. Spreading work over *more* cores reduces per-core
merges and therefore *improves* accuracy - the opposite of the pre-fix behaviour.

**Default (auto) program config** - no `SDPAProgramConfig` at all (`logs/exp7_default_and_k256.log`
vs the stock `doc/functional_decoder/logs/sdpa_decode_default_v2.log`):

| position | before (stock) | after (fix) |
|---|---|---|
| 1023 | 0.99586 | 0.99414 |
| 4095 | 0.99539 | 0.99674 |
| 16383 | 1.21088 | 0.99981 |
| 65535 | 4.76497 | 1.24702 |
| 131071 | 12.65878 | 1.61426 |
| 262143 | 37.65942 | 1.66865 |

A 23x improvement at 262143, but the default config is still **not** inside the bar. It should
not be used: it picks `max_dynamic_chunk_size = dst_size = 4` tiles (k_chunk 128) so 262144 keys
become 2048 chunks -> 128 bf16 merges per core, it turns `exp_approx_mode` on by default, and it
is also the slowest configuration measured (37.2 ms/op vs 2.8 ms for k_chunk 512 / 8 cores).
Callers must pass an explicit program config.

## Verification - device time

Trace-captured (`probes/probe_sdpa_decode_perf.py`, 32 ops per capture, best of 8 replays),
position 262143, full 262144-token cache, `logs/exp8_perf_matrix.log`. us per op:

| k_chunk | 1 core | 2 cores | 4 cores | 8 cores | 16 cores |
|---|---|---|---|---|---|
| 64 | 22586.6 | 11187.2 | 5606.0 | 2826.9 | 2746.5 |
| 128 | 15417.8 | 7583.2 | 3817.1 | 2783.7 | 2744.8 |
| 256 | 11625.6 | 5745.8 | 2896.1 | 2767.3 | 2747.0 |
| 512 | **9774.2** | 4832.1 | **2794.9** | **2772.8** | L1 |
| default (auto) | 37152.3 | | | | |

**The win: 9774 us -> 2795 us, a 3.50x speedup** at k_chunk 512 going from the pinned
`max_cores_per_head_batch = 1` to 4, at equal-or-better accuracy (alpha 0.993-1.000 vs
0.995-1.017). 8 cores adds nothing over 4 (2773 vs 2795 us): the op saturates at ~2.75 ms, which
is 1.07 GB of K+V at ~390 GB/s, i.e. DRAM-bandwidth bound, so ~3.5x is the whole available win.

## Verification - k_chunk sweep at cores_per_head = 1

Requested as the fallback alternative. Device time from the table above; accuracy from
`doc/functional_decoder/logs/sdpa_decode_fp32acc_v2.log` (k_chunk 512/256/128, same build state
for the 1-core path, which this change does not touch) plus `logs/exp9_kc64_and_expapprox.log`
for k_chunk 64. Worst alpha over the 8 positions:

| k_chunk | merges @262143 | device time @262143 | worst alpha over the 8 positions | passes [0.98,1.02]? |
|---|---|---|---|---|
| 64 | 4096 | 22586.6 us | 1.676 (pos 261887) | no |
| 128 | 2048 | 15417.8 us | 1.336 (pos 262143) | no |
| 256 | 1024 | 11625.6 us | 0.989 (pos 262143) | yes |
| 512 | 512 | **9774.2 us** | 1.017 (pos 261887) | yes |

(k_chunk 64 row: `logs/exp9_kc64_and_expapprox.log`; 128/256/512 rows:
`doc/functional_decoder/logs/sdpa_decode_fp32acc_v2.log` - the 1-core path is not touched by this
change, and the k_chunk 512 numbers reproduce exactly in `logs/exp6_matrix_k512.log`.)

**A smaller k_chunk does not beat k_chunk 512 at 1 core on either axis** - it is monotonically
slower (more sequential merges, less DRAM-friendly reads) and monotonically less accurate (more
merges to lose the denominator in; 64 and 128 fail the bar outright). k_chunk 256 has a
marginally tighter worst-case alpha (0.989 vs 1.017) but costs 19% more time and both pass, so
the functional stage's choice of k_chunk 512 with 1 core was the right single-core point. The fix
simply makes 4-8 cores available on top of it, for 3.5x.

## Verification - cleanest before/after A/B (identical program config)

`k_chunk = 128`, `max_cores_per_head_batch = 16`, `exp_approx_mode = True` - the exact
configuration in the stock sweep's `kchunk=128 maxcores=16 expapprox=1` row
(`doc/functional_decoder/logs/sdpa_decode_cfg_sweep_v2.log`) vs
`logs/exp9_kc64_and_expapprox.log`:

| position | num_k_chunks | before (stock) | after (fix) |
|---|---|---|---|
| 1023 | 8 | **3705.57772** | 0.99036 |
| 4095 | 32 | 0.99671 | 0.99100 |
| 12287 | 96 | 0.99893 | 0.99354 |
| 16383 | 128 | 0.99975 | 0.99320 |
| 65535 | 512 | 1.00195 | 0.99674 |
| 131071 | 1024 | 1.00178 | 0.99621 |
| 261887 | 2046 | **NaN** | 1.00104 |
| 262143 | 2048 | 1.00613 | 1.00132 |

Same op, same shapes, same program config, same compute config; only the kernel hunk differs.
Both catastrophic cells - the 3705x at a position where `num_k_chunks < cores_per_head`, and the
NaN at `num_k_chunks % (2*cores_per_head) = 14` - are gone, and no position regressed.

## Verification - blast radius

Same selector the functional stage used (`doc/functional_decoder/logs/ttnn_sdpa_decode_op_tests.log`
header): every non-nightly unit-test file under `tests/ttnn/unit_tests/operations/sdpa/` that
reaches `SdpaDecodeDeviceOperation::create_descriptor`.

```
python -m pytest tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py \
                 tests/ttnn/unit_tests/operations/sdpa/test_paged_sdpa_decode_flexible_geometry.py \
                 tests/ttnn/unit_tests/operations/sdpa/test_bounded_sliding_kv_cache.py \
                 tests/ttnn/unit_tests/operations/sdpa/test_mla_decode.py -v -m "not nightly"
```

**`30 passed, 1 skipped` - byte-identical to the before state** (the same 30/1 recorded in
`doc/functional_decoder/logs/ttnn_sdpa_decode_op_tests.log`; the one skip is the (10,11)-grid test
that this 11x10 board cannot run). Full output in
`logs/ttnn_sdpa_decode_op_tests_after_fix.log`. **No regression.**

Caveat worth stating: these tests exercise `SdpaDecodeDeviceOperation` broadly but mostly with
the default compute config, so most of them take the `else` (fused) branch. They therefore prove
the fused fast path is untouched rather than exercising the new branch; the new branch is covered
by the `exp5`/`exp6`/`exp7` probe sweeps above, which are the ones that were catastrophically
wrong before.

## Final Status

- **Fixed.** `max_cores_per_head_batch > 1` is correct at every decode position; the recorded
  "no core count above 1 is correct at every position" restriction is lifted.
- **Commands that prove the final state:** the four probe invocations quoted above
  (`exp1`/`exp4`/`exp6`/`exp7` logs for correctness, `exp8` for device time) plus the unit-test
  selector.
- **What the caller must do to use this.** Pass an explicit `SDPAProgramConfig`. Recommended:
  `k_chunk_size = 512`, `max_cores_per_head_batch = 4` (alpha 0.993-1.000 at every position,
  2795 us vs 9774 us). `max_cores_per_head_batch = 8` is equally accurate and equally fast.
  Avoid `max_cores_per_head_batch = 2` (only 256->512 bf16 merges per core; 0.977-0.980 at the
  longest positions, and only a 2x win) and avoid the default/auto config entirely.
  For the model port this means `SDPA_DECODE_CORES_PER_HEAD` can go from 1 to 4 (or 8) with
  `SDPA_DECODE_K_CHUNK` left at 512. `functional_decoder.py` is owned by another agent right now,
  so it is deliberately left unedited here.
- **`exp_approx_mode` is not implicated.** `exp_approx_mode = True` at k_chunk 128 / 16 cores is
  0.990-1.001 at every position (`logs/exp9_kc64_and_expapprox.log`), the same as
  `exp_approx_mode = False`, so it can stay at whatever the caller prefers.
- **Remaining risks / follow-ups (all pre-existing, none introduced here):**
  1. `correction_block` + `calculate_fused_max_sub_exp_add_tile` still declare 5 DEST tiles.
     Fixing them at the source (4 inputs, 4 outputs, so a 4-tile layout exists) would restore
     the fused fast path and protect any future caller; it needs edits in
     `transformer/sdpa/device/kernels/compute/compute_common.hpp` and
     `tt_metal/hw/ckernels/*/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h`.
  2. `fp32_local_accumulators` is still gated on `max_cores_per_head_batch == 1`, so multi-core
     runs keep bf16 flash statistics. That is now the only thing standing between multi-core and
     alpha ~1.00 at *every* (k_chunk, core count) pair rather than only at per-core merge counts
     <= 128. Extending it is plausible - the local/wire boundary already goes through
     `move_block`, which converts formats - but it costs L1 and interacts with the c_19 sizing
     that already caps 16 cores at k_chunk 512.
  3. 16 cores at k_chunk 512 exceeds L1 (pre-existing). Use k_chunk 256 for 16 cores, or 4-8
     cores at k_chunk 512.

## Artifacts

Probes (`models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/sdpa/probes/`):

- `probe_sdpa_decode_cores.py` - core-count x position alpha sweep with the derived work split.
- `probe_sdpa_decode_fingerprint.py` - candidate-formula fingerprinting against exact per-core
  float64 flash states walked through the real reduction tree.
- `probe_sdpa_decode_solve.py` - solves for the numerator direction and effective divisor the
  device applied.
- `probe_sdpa_decode_perf.py` - trace-captured device time.

Logs (`.../sdpa/logs/`): `exp1_parity_2cores.log`, `exp2_fingerprint_2cores.log`,
`exp3_solve_2cores.log`, `exp4_fp32dest_off_2cores.log`, `exp5_fix_2cores.log`,
`exp6_matrix_k512.log`, `exp7_default_and_k256.log`, `exp8_perf_matrix.log`,
`exp9_kc64_and_expapprox.log`, `ttnn_sdpa_decode_op_tests_after_fix.log`.
