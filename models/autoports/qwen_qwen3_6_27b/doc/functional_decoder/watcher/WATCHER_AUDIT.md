# Watcher audit — Qwen3.6-27B functional decoder

Hardware: one Blackhole chip (device 2) of a p300c board — see `../README.md`.
Run against the final code (TRI_INV_BASE 16, fp32 SDPA destination accumulation,
SDPA_MAX_K_CHUNKS k-chunk cap).

Command, from /home/ttuser/dev/qwen/rundir with ttenv.sh sourced:

```bash
export ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py \
    -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache" -v -s
```

Result: `5 passed, 52 deselected, 3 warnings in 41.47s` — both layer kinds, paged prefill
at 2049, paged decode, trace capture + replay, and the BFP8 KV-cache path.

Log: `watcher/generated/watcher/watcher.log` (1421 lines, 10 dumps).

## Clean-run audit

```
$ grep -niE 'fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected' watcher.log | grep -v 'highest stack usage'
(no matches)
```

Line categories present, all normal watcher bookkeeping:

```
    660 k_ids
    660 Device
     42 k_id
     10 Dump
      4 Stack
      3 At
      1 Legend
```

Smallest reported free stack across all 20 stack-usage summaries:
**1312 bytes free**. The summaries report headroom; a watcher stack-overflow
report would be an explicit error line, and there are none.

Watcher and the device profiler were kept in separate runs, as the skill requires.
