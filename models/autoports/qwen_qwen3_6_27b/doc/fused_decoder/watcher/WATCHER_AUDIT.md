# Watcher audit — Qwen3.6-27B **fused** decoder

Hardware: one Blackhole chip (device 2) of a p300c board — see `../README.md`.
Run against the final fused code (F1–F22 as landed).

Command, from `/home/ttuser/dev/qwen/rundir` with `ttenv.sh` sourced.  The device profiler is
**off** for this run; watcher and Tracy are never combined (`$tt-device-usage`).

```bash
export ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k "test_traced_decode_pcc or (test_prefill_pcc and 2048) or (test_decode_pcc and 2049) \
        or test_bfloat8_kv_cache or test_fused_graph_is_the_fused_graph" -v -s
```

Result: `9 passed, 55 deselected, 3 warnings in 62.73s`.  The selection deliberately includes the
**2048-token prefill** for both layer kinds, which is the pass that carries the new L1 residency
(F6/F21) and the chunk-grouped triangular inverse, as well as paged decode, trace capture and
replay, the BFP8 KV-cache path, and the device-op-count comparison against the unfused layer (so
watcher covered both implementations in one run).

Console log: `../logs/watcher_run.log`.  Raw watcher log: `generated/watcher/watcher.log`
(1988 lines, 14 dumps).

## Clean-run audit

```
$ grep -niE 'fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected' \
      watcher.log | grep -v 'highest stack usage'
(no matches)
```

Line categories present, all normal watcher bookkeeping:

```
    924 Device
    389 k_ids:
    330 k_ids:142|141|143|143|143
     96 k_ids:355|354|356|356|356
     58 k_ids:1068|1067|1070|1070|1070
     52 k_ids:1068|1067|1069|1069|1069
     47 k_id[
     14 Dump
```

The fused path adds L1 residency the functional path did not have — the width-sharded decode
RMSNorm, the chunk-grouped triangular inverse and matmuls, the per-chunk recurrence loop, and the
three per-tap conv state buffers — which is exactly the class of change watcher exists to catch.
No L1 overflow, no out-of-bounds NOC transaction and no unexpected kernel state was reported.
