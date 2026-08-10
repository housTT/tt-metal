# Watcher audit — Qwen3.6-27B fused decoder

*Generated from the committed artifacts by `../probes/make_watcher_audit.py`; every quantity
below is read out of them, not typed.*

Hardware: one Blackhole chip (device 2) of a p300c board — see `../README.md`. Run against the
final fused code on branch `agentic-research/hous/qwen3.6-27b-v2`, i.e. with
`ttnn.transformer.chunk_gated_delta_rule` on the `linear_attention` prefill path, the bfloat16
causal-conv FIR with its SiLU folded into the last tap, the batch-major decode conv tap buffers,
the width-sharded decode RMS norms, the group-reduction gated norm and the explicit recurrence
core grids.

Command, from the repo root with `ttenv.sh` sourced:

```bash
export ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache or test_traced_decode_batched or test_alternate_page_block_size or test_repeated_runs_stable or (test_batched_users and 32) or (test_merged_unaries_are_not_dispatched and 32) or test_conv_state_after_decode_matches_reference" \
    -v -s
```

Result: **21 passed, 66 deselected** in 384.82 s. Selected tests:

```
  test_alternate_page_block_size[128]
  test_alternate_page_block_size[32]
  test_batched_users[32-full_attention]
  test_batched_users[32-linear_attention]
  test_conv_state_after_decode_matches_reference[1-1]
  test_conv_state_after_decode_matches_reference[1-5]
  test_conv_state_after_decode_matches_reference[32-1]
  test_conv_state_after_decode_matches_reference[32-5]
  test_decode_pcc[2049-full_attention]
  test_decode_pcc[2049-linear_attention]
  test_merged_unaries_are_not_dispatched[32-full_attention]
  test_merged_unaries_are_not_dispatched[32-linear_attention]
  test_repeated_runs_stable[full_attention]
  test_repeated_runs_stable[linear_attention]
  test_traced_decode_batched[32-full_attention]
  test_traced_decode_batched[32-linear_attention]
  test_traced_decode_batched[4-full_attention]
  test_traced_decode_batched[4-linear_attention]
  test_traced_decode_pcc[full_attention]
  test_traced_decode_pcc[linear_attention]
  test_bfloat8_kv_cache
```

That is both layer kinds through paged prefill at 2049, paged decode, trace capture and replay
at batch 1 *and* batch 4, six repeated prefill+decode cycles, the BFP8 KV-cache path, the two
alternate page block sizes through prefill *and* decode, and the advertised-`max_batch` branch
(batched users, the merged-unary graph check and the post-decode conv-state check at batch 32).
The selected tests are listed below, read from the run log rather than described. Run log:
`../logs/watcher_run.log`.

Watcher log: `generated/watcher/watcher.log` (11101 lines, 78 `Dump` header/footer
lines). `watcher.log` and `kernel_names.txt` are committed **gzipped** because each exceeds this
repo's 500 KB per-file commit limit, and so is `kernel_elf_paths.txt`. The
`generated/inspector/` tree the run also emits is not stage evidence and is not committed.

## Clean-run audit

```
$ grep -niE 'fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected' \
      generated/watcher/watcher.log | grep -v 'highest stack usage'
(no matches — grep -c returns 0)
```

Line categories present, all normal watcher bookkeeping:

```
$ awk '{print $1}' generated/watcher/watcher.log | sort | uniq -c | sort -rn | head -6
   5148 Device
   1615 k_ids:
    880 k_ids:215|214|216|216|216
    312 k_ids:3779|3778|3780|3780|3780
    288 k_ids:4021|4020|4022|4022|4022
    283 k_id[
```

`Dump` lines delimit the periodic watcher dumps; `Device` / `k_id` / `k_ids` lines are the
per-core waypoint and active-kernel-id bookkeeping every dump emits, and the `BRISC` / `-----`
lines are the stack-usage summary block. A watcher stack overflow, NOC sanitisation failure, CB
out-of-bounds transaction or L1 overflow would each be an explicit error line; there are none.

The check is also a test: `tests/test_fused_decoder_docs.py::test_watcher_log_is_clean` re-runs
the grep over the committed log, and `::test_watcher_audit_matches_its_artifacts` re-derives the
line count, the dump count, the histogram and the pass/deselect counts above, so this audit
cannot silently describe a different run than the one committed next to it.

Watcher and the device profiler were kept in separate runs, as `$tt-device-usage` requires: the
12 Tracy runs under `../tracy/` were launched separately with no `TT_METAL_WATCHER` set.
