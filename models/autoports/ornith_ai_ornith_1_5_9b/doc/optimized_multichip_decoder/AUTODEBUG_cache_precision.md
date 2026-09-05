# AutoDebug: TP4 BFP4 cache precision gate

Date: 2026-09-05. Fresh, source-only AutoFix investigation. This report was
written before diagnostic implementation. The investigator did not import
TTNN, access hardware, run tests, reset devices, or build native code. The
parent agent owns the serialized hardware lane and all runtime adjudication.

## Finding

The existing failure is a real **failed acceptance gate**, but it is not yet
evidence of incorrect head partitioning or cache addressing. In
`tests/multichip_probe.py`, the reference is an `OptimizedDecoder` with its
default BFP8 K/V cache. `ProductionCache4` instead allocates BFP4 K/V. The
probe concatenates the four local one-head caches along the head dimension
and compares those dequantized values directly with the single-device,
four-head BFP8 cache at PCC >= 0.99. That comparison includes cache
quantization error and projection-geometry differences as well as any
partition error. The failure string, `head-local state/cache partition
mismatch`, cannot distinguish these causes.

Do not change or skip the 0.99 check and do not promote BFP4 while it fails.
Output PCC >= 0.995, even across the recorded batch tests, does not erase this
state gate. Equally, the earlier single-chip BFP4 rejection is not a proof of
the present TP4 failure: that run used four local KV heads and a different
decode QKV geometry. The appropriate AutoFix next step is a matched TP4
producer comparison and an exact local cache update/addressing control.

## Evidence available before this investigation

`logs/production_cache4_layer3.log` records TP4 output PCC
0.998538876/0.999103862 for prefill/decode against the single-chip control,
exact restored eager/trace agreement, and traced decode 0.272232312 ms.
It then fails the unchanged cache/state gate. The archived harness did not
print the individual failing state PCC; no number should be inferred.

The corresponding BFP8-cache production family measured 0.272172 ms, so the
available BFP4 run does not establish a decode latency improvement. Reduced
cache memory remains a material hypothesis requiring the same correctness
adjudication. The parent reports that the current B4/B32 real-HF output
contracts pass, while final BFP4 QKV geometry is still being crossed with the
cache candidate.

Historical `../optimized_decoder/AUTOFIX_cache4_batch.md` independently
localized the earlier BFP4 failure to accumulated key/cache precision:
the same failed users also failed extracted B1 runs; matched BFP8 caches
passed; independently packed device update controls and CPU SDPA on the
dequantized cache proved correct addressing and cache consumption for that
single-chip shape. Those results motivate the TP4 probe; they do not stand
in for it.

## Source findings

- `tt/optimized_decoder.py::_attention_prefill` converts the actual K/V
  producer to each cache dtype before `paged_fill_cache`. Quantization is
  part of the stored model state, not merely a reporting conversion.
- `_attention_decode` feeds the same BF16 K/V update tensors to
  `paged_fused_update_cache`; K and V use disjoint per-user core sets.
- `FunctionalDecoder.allocate_kv_cache` uses
  `[num_blocks, local_kv_heads, page_block_size, head_dim]`; TP4 has one
  local head, page64 and head256. Concatenating rank caches on axis1 is the
  correct logical head reconstruction for this partition.
- The candidate changes only cache dtype through the allocator. No new
  page table, head mapper, update op, or SDPA layout is selected by it.
- Host `from_torch` quantization is not an exact oracle for the device
  update packer. The previous investigation proved that discrepancy.
  Use independently cloned cache buffers and unfused device updates.

## Smallest decisive diagnostic

Run the same production candidate on the actual 1x4 ring, first with K4/V4
and then K8/V8, using the same pinned real weights, QKV configuration,
recorded inputs, page table and position. Explicit allocation dtype is the
only difference. Both are measured diagnostic executions, not a runtime
fallback. The diagnostic should:

1. Hash all four ranks of the pre-cast prefill K/V and BF16 decode K/V
   producers. Require exact equality between the cache-policy runs, so
   differences cannot be assigned to different projection geometry.
2. For every paged fill, reconstruct expected physical rows on CPU from
   the **already device-quantized** fill input and the actual page table.
   Require bitwise cache equality on all four ranks after prefill.
3. For fused decode updates, clone each original cache and independently
   call `paged_update_cache`. Require bitwise equality with the fused
   result, and independently require that only the mapped row changes.
   Exercise identity and permuted disjoint tables at B1/T2048 and B32/T96.
4. Compare paged SDPA with CPU attention over the captured query and
   actual dequantized per-rank cache, at the existing diagnostic 0.999 PCC
   threshold. Keep the model's 0.995 per-user HF output gate.
5. Report K4-versus-K8 and V4-versus-V8 PCC both over the full cache and
   over used logical rows, per rank/user. Keep the existing 0.99 state
   gate. If it fails despite identical producers, exact mapped updates,
   and correct cache consumption, that demonstrates true cache precision
   loss at the unchanged bar; it is not a partition fix to loosen it.

Use the BFP4-QKV candidate as well as the original BFP8-QKV family because
the parent's final selected projection geometry changes the cumulative
policy. A first diagnostic API error is implementation work and requires
an adapted retry. No production changes are proposed from source alone.

## Initial runtime status (superseded by completed adjudication below)

Pending parent execution. The source-only implementation is now
`tests/multichip_cache_diagnostic.py`. AST parsing and the repository
`pre-commit run --files` check pass. No TTNN import or hardware execution
was performed by this investigator. The tool refuses to overwrite JSON
evidence, records all controls before its final assertions, and returns
nonzero if any unchanged precision gate fails. Such an exit with completed
exact-row/producer/SDPA results is useful failed-candidate evidence, not a
claim of a passing optimization.

Parent commands, each serialized through the existing provenance recorder:

```bash
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8
D=models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder
M=models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_cache_diagnostic
python "$D/record_run.py" cache_precision_qkv8_b1 timeout 240 python -m "$M" --variant production_cache4 --batch 1 --length 2048 --output "$D/cache_precision_qkv8_b1.json"
python "$D/record_run.py" cache_precision_qkv4_b1 timeout 240 python -m "$M" --variant production_qkv4_cache4 --batch 1 --length 2048 --output "$D/cache_precision_qkv4_b1.json"
python "$D/record_run.py" cache_precision_qkv4_b1_permuted timeout 240 python -m "$M" --variant production_qkv4_cache4 --batch 1 --length 2048 --permuted --output "$D/cache_precision_qkv4_b1_permuted.json"
python "$D/record_run.py" cache_precision_qkv4_b32 timeout 300 python -m "$M" --variant production_qkv4_cache4 --batch 32 --length 96 --legacy-batch-inputs --output "$D/cache_precision_qkv4_b32.json"
python "$D/record_run.py" cache_precision_qkv4_b32_permuted timeout 300 python -m "$M" --variant production_qkv4_cache4 --batch 32 --length 96 --legacy-batch-inputs --permuted --output "$D/cache_precision_qkv4_b32_permuted.json"
```

The original BFP8-QKV family control can establish the cause of the first
failure; the following four controls concern the parent's final BFP4-QKV
geometry. They are diagnostic forward runs with deliberate host readback,
not latency or trace measurements. The ordinary multichip probe remains
responsible for warmed performance and deterministic replay.

If only one of K4/V4 fails the matched 0.99 gate, the independent key/value
scores identify the remaining mixed-cache option. It would require separate
device update calls and a new whole-layer correctness/performance trial;
never feed mixed dtypes to the fused update API. If both fail, the same-bar
rejection applies to both payload reductions. No such result is assumed
before parent execution.

## Completed adjudication

All five parent-run controls have now completed. The matched TP4 producers,
exact fill/update mappings and CPU SDPA checks pass for both cache dtypes.
Both BFP4 key and value state correlations fail the unchanged 0.99 bar;
the selected BFP4-QKV B32 path also fails real-user HF decode accuracy at
the unchanged 0.995 bar. Identity and permuted mappings agree.
Keep K8/V8. See [the final AutoFix report](AUTOFIX_cache_precision.md),
[JSON summary](cache_precision_summary.json), and
[CSV table](cache_precision_summary.csv) for all measured outcomes and
immutable provenance. The original source-only hypotheses above are
preserved as the investigation record, not outstanding work.
