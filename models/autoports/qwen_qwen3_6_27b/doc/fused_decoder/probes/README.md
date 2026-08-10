# Fused-decoder probes

Eighteen model-free probes and five pieces of tooling. Every probe is self-checking: it computes the
same quantity two ways (device against torch, or fused against unfused) and prints the PCC next
to the timing, so a number in this stage's documents can be reproduced without the model, the
checkpoint or the test harness.

Run them one at a time — they open a device.

```bash
cd /home/ttuser/dev/qwen/tt-metal
source models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh
python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/<probe>.py
```

| probe | question it answers | figures it produces | log |
|---|---|---|---|
| `probe_chunk_gdr.py` | is `ttnn.transformer.chunk_gated_delta_rule` correct and fast at this model's GatedDeltaNet shapes, and which of its two call shapes should be used? | the flat rank-3 path at chunk 32 holds PCC 0.999994 at 2048 tokens and is the faster of the two; the rank-4 path at chunk 64 collapses to PCC 0.903635 (work_log.md §3.1 quotes the run's table) | `../logs/probe_chunk_gdr.log` |
| `probe_small_ops.py` | the grid size; does `rotary_embedding_hf` reproduce the partial RoPE; how many cores should the decode RMS norm shard over; does folding SiLU into the multiply pay? | RoPE PCC 0.999997 against the spelled-out form; the sharded norm is several times faster than the interleaved one and flat between 16 and 20 cores; folding the SiLU into the multiply is worth ~40 % of that op pair (work_log.md §3.3, §3.8) | `../logs/probe_small_ops.log` |
| `probe_causal_conv.py` | which formulation of the 4-tap depthwise causal conv is fastest, and why | the float32 height-broadcast multiply runs several times below the same-shape one; every bfloat16 formulation is about 3x its float32 twin, and building *and* shifting the window in ROW_MAJOR is the fastest of them (work_log.md §3.7 quotes the run's tables) | `../logs/probe_causal_conv.log` |
| `probe_decode_heads.py` | does `nlp_create_qkv_heads_decode` already emit the memory config `paged_update_cache` wants, and does `paged_fused_update_cache` accept it? | configs compare equal at batch 1 and 4; the fused cache write fails its own disjoint-core validation | `../logs/probe_decode_heads.log` |
| `probe_decode_recurrence.py` | how wide can the single-token recurrence matmuls run? | an explicit core grid is ~3x the default batched program factory for both shapes, exactly (PCC 1.000000); `group_attn_matmul` is rejected by its own contract (work_log.md §3.6) | `../logs/probe_decode_recurrence.log` |
| `probe_output_paths.py` | should `chunk_gated_delta_rule` return head-major, and would a head-channel permutation make the partial RoPE one op? | the shipped token-major epilogue is ~2x faster than `output_head_major` (PCC 0.999994 between them); the permuted 256-wide RoPE saves only tens of microseconds (work_log.md §3.13, §3.2) | `../logs/probe_output_paths.log` |
| `probe_gated_norm_batch.py` | which form of the decode z-gated norm is cheaper at each batch size? | the group reduction is nearly batch-independent, the reshape form's relayouts grow with the row count, and they cross between batch 16 and 32 - which is where `_GATED_NORM_GROUP_BATCH` comes from | `../logs/probe_gated_norm_batch.log` |
| `probe_qkv_gate_pack.py` | should `full_attention`'s `wqkv` and `wgate` be one matmul, as HF ships them? | no: the merged output has to be cut back apart, and those two slices cost more at prefill than the activation re-read they save; a wash at decode (work_log.md §6.2) | `../logs/probe_qkv_gate_pack.log` |
| `probe_matmul_bound.py` | why does the report call some matmul rows `SLOW`, and does any graph lever move them? | the rows whose N is 2 or 4 tiles are 2-3x faster on a small explicit `core_grid`, which is now shipped per phase; the dtype and DEST levers are worth a few percent and are precision policy; `in_proj_qkv` is unmoved by any of them (work_log.md §3.18, §6.1) | `../logs/probe_matmul_bound.log` |
| `probe_gdn_input_folds.py` | can `exp(g)` and `sigmoid(b)` ride on the multiplies that consume them, and should the rank-3 change move ahead of the `a`/`b` slices? | both folds are bit-exact and cheaper, and both are shipped; the rank-3 reorder is a tie and was not taken (work_log.md §3.19) | `../logs/probe_gdn_input_folds.log` |
| `probe_decode_conv_dtype.py` | should the *decode* causal-conv FIR run in bfloat16, as the prefill one does? | no: its taps read the carried float32 state, so it pays `K - 1` typecasts per step that prefill does not, and they cancel the bandwidth won (work_log.md §6) | `../logs/probe_decode_conv_dtype.log` |
| `probe_gdn_decode_heads.py` | should the decode Q/K L2 norm run before the GQA expansion instead of after? | no: normalising 16 heads instead of 48 is real work saved, but expanding on the flattened head axis costs more than it saves, at both batch sizes (work_log.md §6) | `../logs/probe_gdn_decode_heads.log` |
| `probe_decode_qk_pair.py` | Q and K are the same shape and take the same path at decode - should they be one tensor through the norm, scale and rank change? | no: one `rms_norm` over twice the heads is real work saved, but concatenating Q and K and cutting the result apart costs more at batch 32 and ties at batch 1 (work_log.md §6) | `../logs/probe_decode_qk_pair.log` |
| `probe_prefill_qkv_split.py` | should `in_proj_qkv` be three projections and three FIRs instead of one and three slices? | no: the packed chain is faster end to end, the same result §6.2 found for `wqkv`/`wgate` in the other direction | `../logs/probe_prefill_qkv_split.log` |
| `probe_addcmul_state.py` | can the recurrent-state update be one `ttnn.addcmul` instead of a multiply and an add? | yes, and it is shipped: one pass over the 100 MB carried state instead of two, bit-exact in place (work_log.md §3.21) | `../logs/probe_addcmul_state.log` |
| `probe_decode_rope_half.py` | is `ttnn.experimental.rotate_half` faster than the four ops it replaces at decode? | on wall clock yes, on traced device time no - the op is single-core by construction, so the spelled-out form ships on the decode path (work_log.md §3.22) | `../logs/probe_decode_rope_half.log` |
| `probe_dense_recurrence.py` | should the recurrence's transient vectors be dense instead of one padded row per head? | yes, and it is shipped: bit-identical, and at the advertised batch the chain measures a little over half what it did (work_log.md §3.24) | `../logs/probe_dense_recurrence.log` |
| `probe_mlp_variants.py` | fused gate/up matmul or split, and where should the SiLU live? | the fused gate/up matmul with the SiLU folded into the multiply is the fastest of the three at prefill; splitting the matmul is a wash at decode (work_log.md §3.8) | `../logs/probe_mlp_variants.log` |

Tooling:

| script | what it does |
|---|---|
| `run_perf.sh <kind> <phase> <impl>` | one Tracy device-profiler run of `tests/test_fused_decoder_perf.py`, then `tt-perf-report` over the signposted window; writes `../tracy/<impl>/<kind>/<phase>_*` and `../logs/tracy_<impl>_<kind>_<phase>.log`. `<impl>` is `functional` or `fused`, so the before/after pair comes from the same script on the same machine |
| `make_watcher_audit.py` | writes `../watcher/WATCHER_AUDIT.md` out of the committed watcher log and run log — line count, dump count, category histogram, selected tests, pass/deselect counts and the offender grep — so the audit cannot describe a different run than the one committed |
| `make_doc_tables.py` | fills every `<!-- GENERATED:... -->` block in these documents from `../perf_summary.json` and the probe logs, so no volatile figure is hand-maintained; `tests/test_fused_decoder_docs.py::test_generated_blocks_are_current` re-runs it and requires the committed text to match |
| `make_perf_summary.py` | re-derives `../perf_summary.json` from those CSVs — device time by summing the `Device Time` column, and an exact op-code periodicity check on each decode window. Reads committed artifacts only |
