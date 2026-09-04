# AutoDebug: native-context exact-cache decode accuracy

Date: 2026-09-04. Isolated AutoFix investigation. No device access, hardware
tests, implementation edits, nested agents, or dependency changes. Read current
code, logs, local AGENTS contract, HF source, relevant pinned 35B reference, and
the paged SDPA wrapper, factory, reader, compute and cache-update paths. A small
CPU-only real-weight query probe was also run below.

## Verdict

**The root cause is not yet proven.** There is a concrete decode/prefill
precision-policy discrepancy and a length-dependent BF16 accumulation boundary
worth testing. Neither establishes that a precision change fixes PCC .94535.
The examined fixture permutation, HF cache length, fixed-chunk read coverage,
and integer page addressing are consistent. Do not reduce context, waive the
PCC gate, or change several numerical settings together on this evidence.

## Starting evidence and input provenance

Original command:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 ORNITH_WEIGHTS=real pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_contract_extensions.py -k native_context_decode_oracle -x -v -s
```

`logs/native_decode_oracle.log` reports finite output but
`PCC=0.9453528402993279 < 0.995` at position 262143. The test uses real layer-3
weights, deterministic synthetic BF16 historical KV, shuffled physical pages,
and a one-token FP32 HF decoder oracle. It does not claim the history came from
HF prefill.

**That run used input standard deviation .5.** Current `make_activations` uses
the newly measured embedding standard deviation .014285416342318058. A rerun
of the same command now is a different input experiment. Preserve the original
seed-971 input generated with .5 for causal comparisons, and separately record
the calibrated run. Existing short/8192 results and calibrated real-weight
results are useful controls, but do not substitute for this exact fixture.

`logs/native_prefill.log`, inspected while the coordinator's run was in progress,
also records full-attention prefill length 262143 followed by final-position
eager/replay PCC 1.0. This supports trace determinism on a naturally produced
native-length cache. It is neither HF accuracy evidence nor a direct
eager/replay comparison for the failing synthetic history.

## Effective precision and geometry ledger

| Boundary | Failing path |
| --- | --- |
| Input and ordinary uploaded weights | BF16; original input scale .5 |
| HF layer computation | FP32, checkpoint weights converted to FP32 |
| HF historical cache | Exactly the BF16 fixture values expanded to FP32 |
| TT historical cache | BF16, `[4096,4,64,256]` per K/V tensor |
| New KV row | HF FP32 projection/norm/RoPE; TT BF16 intermediates and BF16 cache write |
| Other TT projections | HiFi4, approximate math off, FP32 destination on |
| Prefill SDPA config | HiFi2, approximate math off, FP32 destination on |
| Decode SDPA effective config | HiFi2, approximate math on, FP32 destination off, packer accumulation off |
| Decode program | 8×8 grid; fixed K chunk 64; exponential approximation explicitly off |
| Decode intermediate and statistics CBs | BF16, even if FP32 destination is requested |
| Placement | One device; replicated tensor mapper; DRAM interleaved Q/K/V; no CCL |
| Pages and positions | INT32 row-major table/position; UINT32 RoPE index; 4096 permuted 64-token pages |
| Cache operations | Host fixture copy, then `paged_update_cache`, then paged SDPA decode |

The new FP32 HF cache entry is an expected reference/TT numerical boundary.
The history is exact between implementations; the entire post-update cache is
not guaranteed bitwise identical until its final row is read back and checked.

## H1: decode inherits a different compute policy

**Verified source discrepancy; causal hypothesis only.**
`tt/functional_decoder.py:564` does not pass `compute_kernel_config` to paged
decode, unlike `_attention_prefill`. The wrapper at
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:154` fills its
missing configuration with HiFi2, `math_approx_mode=true`,
`fp32_dest_acc_en=false`, `packer_l1_acc=false`. The program's
`exp_approx_mode=False` is a separate setting and does not override all four.
The pinned 35B functional decoder has the same omission; inherited code alone
is not correctness evidence.

Prediction: if destination precision or approximate math owns the failure,
changing that setting alone on fixed Q/K/V improves the direct SDPA comparison;
then the same change should improve the original full decoder case. Keep
HiFi2, cache dtype, page permutation, query, position, and K chunk unchanged.
Test explicit defaults first, then only `fp32_dest_acc_en=True`, then only
`math_approx_mode=False`. Passing `decoder.sdpa_compute_kernel_config` changes
both settings and is useful as a sensitivity screen, but needs those isolated
controls before attributing cause.

## H2: repeated BF16 online and tree reductions lose accuracy at length

**Verified mechanism and work-count difference; causal hypothesis only.**
The factory hard-codes `im_df` and `stats_df` to `Float16_b`
(`sdpa_decode_program_factory.cpp:439`). Circular buffers 24–31, 21–23 and
tree outputs retain BF16 storage. The compute loop
(`device/kernels/compute/sdpa_flash_decode.cpp:458–526`) repeatedly rescales
the previous sum/output, adds a new chunk, and packs the results back into
those buffers. FP32 destination accumulation does not make these persistent
partial sums or maxima FP32.

The explicit program keeps `SDPAProgramConfig.max_cores_per_head_batch` at its
struct default 16. With batch 1, four KV heads and 64 available cores, the
factory assigns 16 cores per KV head, one head per core, and four tree-reduction
rounds. The per-core work instantiated from `rt_args_common.hpp` is:

| Read length | Total 64-token chunks | Local chunks/core | Core 0 chunk range |
| --- | --- | --- | --- |
| 8192 | 128 | 8 | `[120,128)` |
| 262144 | 4096 | 256 | `[3840,4096)` |

Core 15 handles the beginning of the sequence; core 0 handles the final chunk
and applies its causal mask. This is a concrete 32× increase in local reduction
steps, not a dynamic chunk-size transition: K chunk 64 is explicitly selected.

Prediction: on the identical post-update cache and query, changing only K chunk
64→256 reduces local steps from 256 to 64. A substantial improvement localized
to direct attention would implicate chunk-dependent numerical behavior; a
clean final decoder result must still be reproduced. It does not, by itself,
prove BF16 cache storage is faulty or justify a C++ accumulator rewrite. A
CPU BF16 recurrence simulation can illustrate sensitivity but is not a
bit-accurate model of the device kernel.

## Exact cache and final-row checks

The fixture's `physical[table_host[0]] = logical` is the correct inverse for a
reader that uses `physical[table_host[0,virtual_page]]`. Both K and V use the
same table. Flattening `[1,4,262144,256]` to `[4,4096,64,256]` and permuting
to `[4096,4,64,256]` preserves head/token coordinates.

The HF cache receives 262143 historical entries at layer 3, then appends one
current entry. Its unmasked decode row has length 262144, exactly matching that
cache. The test's warmup, capture, and replay repeatedly overwrite the same TT
row with the same current input; for full attention this is semantically
idempotent. There is no recurrent-state rewind requirement for this layer.

Read coverage is exact: `ceil((262143+1)/64)*64 = 262144`; allocation and table
both cover 4096 pages. Final virtual page is 4095, row 63, tile-row 8191. SDPA's
tile address is `table[4095]*64 + head*16 + 8 + column_tile`; the cache updater
uses the same page and block-row calculation with row offset 31 inside its
last tile. Page IDs/positions are handled as 32-bit integers on this interleaved
path. No rounding beyond allocation, 16-bit truncation, or dynamic read-window
cliff was found in this exact case.

This source consistency does not prove the hardware write/read is correct.
Read back physical page `table[4095]`, especially rows 62 and 63 across all four
heads. Compare row 63 with the actual TT projected/rotated K and projected V,
and confirm row 62 and sampled earlier pages retain the fixture values. For
this control import the model allocator, page size, KV-head count, mapper and
update helper; a nearby generic SDPA shape is weaker evidence.

### Why the final row matters: measured CPU query probe

A CPU-only command loaded real layer 3 with `hf_reference`, generated
`(randn([1,1,4096], seed=971)*scale).to(bfloat16).float()`, ran its input norm,
Q/K/V projections and Q/K norms, then HF RoPE at 262143. It ran under
`TORCHINDUCTOR_CACHE_DIR=.../state/torch-cache OMP_NUM_THREADS=8 python_env/bin/python`.
No TTNN operation or device was invoked.

| Quantity | Original scale .5 | Calibrated scale .014285416342318058 |
| --- | --- | --- |
| Actual input std | .50677508 | .01447985 |
| Query RMS across heads | 1.30313–1.37833 | 1.30315–1.37841 |
| New V std | 1.2747313 | 1.2717737 |
| Self logit, zero-based head 13 | 8.9242411 | 8.9263086 |
| Self logit, zero-based head 15 | 17.8488140 | 17.8532887 |

The synthetic historical K standard deviation .125 implies historical-logit
standard deviation about .163–.172 for these queries (an analytical estimate,
not measurement of all historical logits). Consequently the newest token can
dominate head 15 while other heads largely average historical entries. The
newest V is also roughly ten times the fixture's standard deviation. This
makes a final-row write/read and current-token masking probe particularly
informative. RMSNorm leaves Q/K nearly unchanged by the input calibration;
the calibration is not evidence that the cache/kernel failure is repaired.
The synthetic history remains a valid mathematical oracle despite differing
from the real model's cache distribution.

## Recommended experiment order

1. **Freeze inputs and localize trace.** Keep both original-scale and calibrated
   runs labelled. On the failing fixture retain the first eager output instead
   of discarding it, then compare eager, captured output and replay to each
   other and HF. If eager already fails similarly, investigate shared math
   first. A replay-only discrepancy requires a trace/buffer-lifetime diagnosis.

2. **Locate the first numerical divergence.** Compare input norm, Q/K/V before
   and after RoPE, actual final KV row, direct SDPA result per head, gated output
   projection, residual, and MLP output. Report relative error and norms in
   addition to PCC. Inspect head 15 explicitly. CPU/reference substitution of
   the attention result through the TT decoder tail separates attention drift
   from downstream amplification.

3. **Run the same-cache high-precision control.** Read TT Q and the actual
   post-update BF16 cache into torch, undo the physical permutation, and run
   FP32 eager attention with fourfold GQA repetition, scale 1/16, and exactly
   262144 valid entries. Compare with TT's direct SDPA output. Separately feed
   HF Q with the same readback cache: this separates query production from
   cache/write and attention-reduction boundaries. Avoid rounding away the
   distinction between HF's new FP32 entry and TT's new BF16 entry.

4. **Test H1, then H2 independently.** Use the isolated configuration controls
   above, preserving the exact uploaded cache and RNG stream. Only keep a
   change after direct attention and the original full decoder gate support
   it. An unchanged/worse FP32 result refutes that particular intervention;
   it does not refute H2 because the CBs still hold BF16.

5. **If needed, separate length from position.** A component probe can hold Q
   (already rotated at 262143) fixed while varying the number of historical
   entries and moving its current K/V to the last valid slot. This isolates
   reduction length from RoPE. Preserve cache shapes and original fixture
   values; regenerating a smaller fixture changes the RNG stream. An identity
   table control must repack the same logical cache, not merely replace the
   table. If a boundary cliff or readback discrepancy appears, extend allocation
   and table by 32 valid pages without changing their original prefix. Do not
   reinterpret over-allocation as required when the fixed read-window math
   already fits.

Full-context capacity and source-level trace consistency do not satisfy the
missing native HF accuracy gate. The original failure remains open pending
these focused device-owner experiments. No proposed fix has been integrated.
