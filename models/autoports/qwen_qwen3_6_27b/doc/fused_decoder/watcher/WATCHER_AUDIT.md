# Watcher audit — Qwen3.6-27B **fused** decoder

Hardware: one Blackhole chip (device 2) of a p300c board — see `../README.md`.
Run against the final fused code (F1–F20 as landed, i.e. after the stage review's corrections).

Command, from `/home/ttuser/dev/qwen/rundir` with `ttenv.sh` sourced.  The device profiler is
**off** for this run; watcher and Tracy are never combined (`$tt-device-usage`).

```bash
export ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache \
        or test_fused_graph_is_smaller" -v -s
```

Result: `7 passed, 57 deselected, 3 warnings in 51.28s` — both layer kinds' paged prefill at
2049, paged decode, trace capture + replay, the BFP8 KV-cache path, and the device-op-count
comparison against the unfused layer (so watcher covered both implementations in one run).

Console log: `../logs/watcher_run.log`.  Raw watcher log: `generated/watcher/watcher.log`
(1720 lines, 12 dumps).

## Clean-run audit

```
$ grep -niE 'fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected' \
      watcher.log | grep -v 'highest stack usage'
(no matches)
```

Line categories present, all normal watcher bookkeeping:

```
    792 Device
    382 k_ids:
    224 k_ids:142|141|143|143|143
     58 k_ids:1073|1072|1075|1075|1075
     52 k_ids:1073|1072|1074|1074|1074
     40 k_id[
     20 k_ids:157|158|161|161|161
     18 k_ids:159|160|161|161|161
```

The fused path adds L1 residency the functional path did not have — the width-sharded decode
RMSNorm, the triangular inverse, the per-chunk recurrence loop, and the three per-tap conv
state buffers — which is exactly the class of change watcher exists to catch.  No L1 overflow,
no out-of-bounds NOC transaction and no unexpected kernel state was reported.
