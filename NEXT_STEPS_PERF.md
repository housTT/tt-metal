# DeepSeek-V4 on QuietBox — Perf Next Steps (handoff)

Ground rule: **no claim without a measurement.** Every number below is measured on the
4-chip (1,4) Blackhole mesh (cool, 34–45 °C). Unproven items are labeled UNVERIFIED.

## Current best (measured, STABLE) — 2.32 tok/s
- `ATTN_TP=1 / LRU=32 / WARM_ALL=1` (+ pagecache preload): **2.32 tok/s, CV=0.05**
  (16 tokens 390–492 ms, disk=0/cold=0). +37% over the old 1.69 baseline. This is the
  number to beat now.
- Old baseline `ATTN_TP=0 / LRU=0`: stable **1.69 tok/s** (uploads all experts/token).

## How the 2.32 was reached (the full variance investigation, now closed)
Root cause chain, each step measured: async-backlog ✗ / thermal ✗ / DRAM ✗ → **cold expert
builds** (~300 ms each) driven by **non-deterministic routing** (25–27/43 layers flip on
identical input) → after warming the mem cache, a **page-cache-fault** residual (mmap payloads
not resident) → fixed by reading all cache files into page cache at startup. Controlled A/B
proved page-cache: identical config, only pretouch differs → CV 1.15→0.05, 1.98→2.34 tok/s.
Wired into the `WARM_ALL` startup hook (`warm_all` + `preload_pagecache`).

## ROOT CAUSE of the TP+LRU variance — SOLVED (2026-07-16)
The async per-token "variance" is **cold expert builds**, caused by **non-deterministic routing**.
Two measured experiments (`demo/sync_cadence_sweep.py`, `demo/determinism_probe.py`):
- A fully-warm token (disk=0, cold=0) = **449 ms = 2.2 tok/s**. Each COLD expert build adds
  ~300 ms (dequant+tilize); latency correlates ~1:1 with the cold-build count.
- Routing differs on **25–27 of 43 layers on bit-identical input** (low-bit compute
  non-determinism flips near-tied top-6 scores; compounds up the stack from ~layer 4).
- A partial cache (LRU K=32) therefore keeps missing on a fluctuating ~90-expert/token tail
  → recurring cold builds → the 449 ms→9500 ms swings. Invisible at LRU=0 (uniform uploads).
- DISPROVEN by this data: async-dispatch backlog (sync cadence had no effect), thermal, DRAM.

## THE WIN and the fix (in progress / to verify)
Warm floor **2.2 tok/s** vs 1.69 baseline is a real ~1.3×. The only blocker to hitting it
stably is eliminating cold/disk builds in steady state. Since routing is non-deterministic,
the robust fix is to **fully warm the host bf4 cache (all 11008 experts)** → every routing
choice becomes a cheap mem hit → flat per-token latency regardless of routing.
- Implemented: `Fp4ExpertCacheMesh.warm_all(num_layers, num_experts)` (CPU+disk-bound,
  resumable via disk cache, ~145 GB host RAM — fits in ~227 G available). Env
  `DEEPSEEK_V4_WARM_ALL=1` triggers it at `FastDecoder.__init__`.
- Verify with `demo/warm_stable_bench.py` (warm_all → measure steady-state per-token).
  EXPECT flat ~450–600 ms, disk=0/cold=0 → stable ~2.0–2.2 tok/s. **[status: measuring]**
- Bonus: warm_all fully populates the disk cache → future cold boots ~330 s (load) not 286 s.

### Server test (once warm_stable_bench confirms the flat number)
```
cd /home/ttuser/code/tt-inference-server && python run.py --model DeepSeek-V4-Flash \
  --workflow server --local-server --tt-device p300x2 --no-auth \
  --disable-trace-capture --disable-metal-timeout
```
with env `DEEPSEEK_V4_MODEL=nvidia/DeepSeek-V4-Flash-NVFP4 DEEPSEEK_V4_ATTN_TP=1
DEEPSEEK_V4_EXPERT_LRU=32 DEEPSEEK_V4_WARM_ALL=1`. Startup warm-all is a one-time ~40 min
(or ~5 min once the disk cache is fully populated), then steady decode should hold ~2 tok/s.

## Further levers (ordered, each ends in a measurement)
1. **Reduce routing non-determinism** (cleaner than brute-force warming, if achievable): the
   origin is on-device low-bit compute, not app-level, so likely hard — but worth a bounded
   probe (e.g. does forcing fp32 router-input reductions stabilize top-6?). UNVERIFIED.
2. **MoE is the dominant section** (~215–262 ms/token). Overlap the layer-end all_gather with
   compute; try a better program config for the coalesced `[E,...]` batched matmul.
3. **Attention all_gather → reduce_scatter** fusion (~106–121 ms section).
4. **Larger K** now that ATTN_TP frees DRAM (15–29% used) — raise resident hit rate above ~65%.
   Re-measure hit rate + tok/s; guard against DRAM growth (instrument per-token).

## Guardrails (relearned this session)
- Every device run gets a no-progress watchdog — but the script MUST emit per-token heartbeats,
  else the watchdog kills a slow-but-working process (a 286 s cold prefill got killed → wedged a
  chip → `tt-smi -r`). See memory `device-run-watchdog`.
- kill -9 mid-device-op wedges a chip (`NOC0 hung`); recover with `tt-smi -r` + ~8 s settle.
- fp4 (bfloat4_b) cannot be host-sliced/concat'd (TT_FATAL); device concat only.
- Disk-cache stamp is model-snapshot-sensitive: base (60d8d707) vs NVFP4 (e3cd60e7) get separate
  dirs. The stale 47 G base-model cache dir can be deleted if disk is needed.
- Defaults remain `ATTN_TP=0 / LRU=0 / WARM_ALL=0` (stable 1.69) until warm_stable_bench confirms
  the flat ~2.2, then flip to TP=1/LRU=32/WARM_ALL=1.

## Key files
- `tt/mla_v4_device.py` — TP sharding, Fp4ExpertCacheMesh (+warm_all), coalesced MoE, `_bf4_cache_stamp`.
- `tt/fast_decode.py` — FastDecoder (+WARM_ALL hook, +SYNC_EVERY debug env — leave 0).
- `demo/warm_stable_bench.py`, `demo/sync_cadence_sweep.py`, `demo/determinism_probe.py`.
