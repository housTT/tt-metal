# Watcher run classification

Command (watcher only — never combined with the profiler, and with its own log path):

```bash
rm -f generated/watcher/watcher.log
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_functional_decoder.py -v -p no:randomly \
  -k "(decode_pcc and 130) or traced_decode or determinism or permuted or poisoned or ragged or unaligned_max_context"
cp generated/watcher/watcher.log      <this dir>/watcher_log.txt
cp generated/watcher/kernel_names.txt <this dir>/kernel_names.txt
```

`TT_METAL_WATCHER_APPEND=1` is required: watcher truncates its log on every device open, so without it
the log holds only the last test's session.

Result: **17 passed** in 66.57 s (console log: `../logs/watcher_pytest.txt`).
Watcher reported `disabled features: None` — Ethernet checks were left enabled.

Selected subset: paged decode across a page boundary, traced decode capture/replay, the
determinism loop, the permuted page table, the poisoned-free-pool aliasing regression, the
ragged per-user-position batched decode (including batch 13, the non-rectangular shard grid) and
the unaligned-`max_context` prefill — i.e. every path that writes a cache/state buffer in place,
replays a trace, or exercises a newly added shape.

Artifacts in this directory: `watcher_log.txt` (the watcher log itself, copied out of
`generated/watcher/watcher.log` because the repo `.gitignore` excludes both `*.log` and `generated/`)
and `kernel_names.txt` (the id→kernel map the log's `k_ids` lines refer to).

## Line-by-line classification of `watcher_log.txt`

18 564 lines, every one routine. Every bucket below is a disjoint prefix/substring rule, and they
sum exactly to the file. `census.py`, committed next to this file, reproduces the table, the stack
headroom check and the fatal-class grep, and writes its output to `census_summary.txt` so the counts
below are a generated artifact rather than a transcription:

| Count | Line kind (matching rule) | Classification |
| --- | --- | --- |
| 8 976 | starts `Device <n>` — per-core status dump header (waypoints, rmsg/smsg) | normal polling output |
| 8 976 | starts `k_ids:` — the kernel-id continuation of a status dump | normal per-core kernel id reporting |
| 68 | starts `k_id[` — the id → kernel-source map printed at the end of a dump section | normal |
| 187 | starts `At <t>s` or `Dump #` — dump banners | normal 10 s-interval dumps |
| 238 | legend block rows (one block per device open) | normal header |
| 102 | blank | — |
| 17 | starts `-----` | separators |

This log contains **no** `Stack usage summary` block. Watcher emits one only for dumps where firmware
had recorded a stack watermark, so its absence means "not measured", not "no overflow" — `census.py`
prints that distinction rather than reporting a minimum over an empty list. An earlier run of the same
subset did report one, with no overflow anywhere; it was superseded by this run, which
postdates the last source edit, and the summary did not reappear under
`TT_METAL_WATCHER_APPEND`, `TT_METAL_WATCHER_DUMP_ALL` or a disabled kernel cache. A real stack
overflow would still be caught by the fatal-class grep below, which matches `overflow`.

Fatal-class grep over the whole log returns **zero** matches:

```bash
grep -inE "watcher.*(error|fatal|assert)|out of bounds|overflow|sanitiz|corrupt|unexpected|\
invalid (noc|address|coord)|hang|deadlock|NOC_ERR|CB_ERR" watcher_log.txt   # -> no output
```

No asserts, no invalid NOC coordinates or addresses, no CB out-of-bounds transactions, no L1
overflow, no stack overflow, no hardware faults, no suspicious waypoint states. No false positives
to explain.
