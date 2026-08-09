# Watcher audit — Qwen3.6-27B **fused** decoder

Hardware: one Blackhole chip (device 2) of a p300c board — see `../README.md`.
Run against the final fused code (all of F1–F16 landed).

Command, from /home/ttuser/dev/qwen/rundir with `ttenv.sh` sourced.  The device profiler is
**off** for this run; watcher and Tracy are never combined (tt-device-usage skill).

```bash
export ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache \
        or test_fused_ops_are_used" -v -s
```

Result: `7 passed, 57 deselected, 3 warnings in 59.33s` — both layer kinds' paged prefill at
2049, paged decode, trace capture + replay, the BFP8 KV-cache path, and the fused-op dispatch
assertions (so the ops the watcher saw are provably the fused ones).

Console log: `../logs/watcher_run.log`.  Raw watcher log:
`generated/watcher/watcher.log` (1705 lines, 12 dumps).

## Clean-run audit

```
$ grep -niE 'fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected' \
      watcher.log | grep -v 'highest stack usage'
(no matches)
```

Line categories present, all normal watcher bookkeeping:

```
    792 Device
    345 k_ids:
    220 k_ids:145|144|146|146|146
     82 k_ids:570|569|571|571|571
     58 k_ids:1128|1127|1130|1130|1130
     52 k_ids:1128|1127|1129|1129|1129
     36 k_id[
     28 k_ids:570|569|572|572|572
```

The fused path adds width-sharded L1 buffers (the decode RMSNorm) and L1-resident intermediates
(the triangular inverse and the per-chunk recurrence loop) that the functional path did not
have, which is exactly the class of change watcher is there to catch — no L1 overflow, no
out-of-bounds NOC transaction and no unexpected kernel state was reported.
