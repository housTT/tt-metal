# Watcher audit — Qwen3.6-27B functional decoder

Hardware: one Blackhole chip (device 2) of a p300c board — see `../README.md`.
Run against the final code on branch `agentic-research/hous/qwen3.6-27b-v2`, i.e. with the
`q 128 / k 256` prefill SDPA config, the `k 512 / 1 core per head` decode SDPA config and the
fp32 local flash accumulators in `sdpa_decode_program_factory.cpp`.

Command, from the repo root with `ttenv.sh` sourced:

```bash
export ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py \
    -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache or test_traced_decode_batched or test_alternate_page_block_size" \
    -v -s
```

Result: **9 passed, 48 deselected** in 56.76 s — both layer kinds, paged prefill at 2049,
paged decode, trace capture + replay at batch 1 *and* batch 4, the BFP8 KV-cache path, and
the two alternate page block sizes (32 and 128) through prefill *and* decode.
Run log: `../logs/watcher_run.log`.

Watcher log: `generated/watcher/watcher.log` (1720 lines, 6 dumps over 12 `Dump` header/footer
lines).

## Clean-run audit

```
$ grep -niE 'fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected' \
      generated/watcher/watcher.log | grep -v 'highest stack usage'
(no matches — grep -c returns 0)
```

Line categories present, all normal watcher bookkeeping:

```
$ awk '{print $1}' generated/watcher/watcher.log | sort | uniq -c | sort -rn | head -6
    792 Device
    285 k_ids:
    273 k_ids:607|606|608|608|608
     58 k_ids:1183|1182|1185|1185|1185
     52 k_ids:1183|1182|1184|1184|1184
     42 k_ids:2133|2132|2134|2134|2134
```

`Dump` lines (12 of them: a header and a footer for each of the 6 dumps, the last being
`Dump #6 completed`) delimit the periodic watcher dumps; `Device` / `k_id` / `k_ids` lines are
the per-core waypoint and active-kernel-id bookkeeping every dump emits, and the `BRISC` /
`-----` lines are the stack-usage summary block. A watcher stack overflow, NOC sanitisation
failure, CB out-of-bounds transaction or L1 overflow would each be an explicit error line;
there are none.

Watcher and the device profiler were kept in separate runs, as the skill requires: the four
Tracy runs under `../tracy/` were launched separately with no `TT_METAL_WATCHER` set.
