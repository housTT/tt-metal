# Stage 0 (device): TP=2 path on a (1,2) submesh with a 4-layer Clef config

Scope: prove or characterize the `qwen36` tensor-parallel (TP) path at TP=2 on this box, before stage 1 builds the engine on it. Plan: `/home/hous/.claude/plans/expressive-orbiting-tome.md` (Stage 0). Work log: `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/doc/stage0/work_log.md`. Times are ET (UTC-4).

## Mesh facts (measured)

Source: `/home/hous/dev/clef/reports/stage0_mesh_probe.json` (first probe, log `/home/hous/dev/clef/logs/stage0_mesh_probe.log`) and `/home/hous/dev/clef/reports/stage0_mesh_probe_ccl.json` (fixed collective probe, log `/home/hous/dev/clef/logs/stage0_mesh_probe_ccl.log`).

| Fact | Value | Evidence |
|---|---|---|
| Cluster type | `ClusterType.P300_X2` (2 x p300c, 4 chips) | both probe JSONs |
| Direct `open_mesh_device(MeshShape(1,2))` under `FABRIC_1D` | fails: `Fabric Router Sync: Timeout after 10000 ms on Device 0` (19.7 s) | `stage0_mesh_probe.json` `direct_1x2_fabric1d` |
| Path A: `FABRIC_1D` + `(1,4)` parent + `create_submesh((1,2), offset (0,0))` | opens; parent chips `[1, 0, 3, 2]`, submesh chips `[1, 0]` | `direct_1x4_with_1x2_submesh` |
| Path B: `FABRIC_2D` + `(2,2)` parent + the same submesh | opens; parent chips `[1, 0, 2, 3]`, submesh chips `[1, 0]` | `parent_2x2_submesh_1x2` |
| Generic collectives on the submesh (`all_gather_async`, `reduce_scatter_minimal_async` through `TT_CCL`) | pass with 1 or 2 links and Linear or Ring, on both parents; gather error 0.0, reduce-scatter error 0.03125 (one bf16 ulp) | `stage0_mesh_probe_ccl.json` |
| `get_num_links(submesh)` | 2 (`get_device_name` maps 2 Blackhole chips to `P300`, link table `(2, 2)`) | `/home/hous/dev/clef/tt-metal/models/common/modules/tt_ccl.py` lines 151 to 184, `/home/hous/dev/clef/tt-metal/models/common/device_utils.py` |
| `ModelArgs.ccl_topology()` on the submesh | `Topology.Ring` (cluster type `P300_X2`) | `/home/hous/dev/clef/tt-metal/models/tt_transformers/tt/model_config.py` lines 2732 to 2755; test log `model:` line |
| `ModelArgs.device_name` on the submesh | `P300`; tensor cache root `/home/hous/dev/clef/tt_cache/P300` | `/home/hous/dev/clef/logs/stage0_tp2_sanity_l4.log` |

### Probe fix

The first probe's `collective_smoke` compared a bf16 `ttnn.to_torch` result against a float32 tensor in `torch.allclose`, which raises `BFloat16 did not match Float` before any device result is read. The fixed script (`/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/scripts/mesh_probe.py`) casts the device result to float32, builds the expected values from the bf16-rounded input, and calls the same helpers the model calls (`tt_all_gather`, `tt_all_reduce` in `/home/hous/dev/clef/tt-metal/models/tt_transformers/tt/ccl.py`) instead of `ttnn.all_gather`, whose `num_links` argument is deprecated and ignored. It runs the four link and topology variants and writes the JSON after each one.

Re-run command:

```
cd /home/hous/dev/clef && nohup /home/hous/dev/clef/bin/devrun timeout 600 env PROBES=parent_ccl,1x4 python models/autoports/cloudflare_clef/scripts/mesh_probe.py /home/hous/dev/clef/reports/stage0_mesh_probe_ccl.json > /home/hous/dev/clef/logs/stage0_mesh_probe_ccl.log 2>&1 &
```

## TP=2 sanity: 4-layer Clef on the (1,2) submesh

Test: `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/tests/test_tp2_sanity.py`. Final run (run 4): log `/home/hous/dev/clef/logs/stage0_tp2_sanity_l4_run4.log`, JSON `/home/hous/dev/clef/reports/stage0_tp2_sanity_l4_1x4.json`. Result: `6 passed, 1 xfailed in 141.90 s`.

Command:

```
cd /home/hous/dev/clef/tt-metal && nohup /home/hous/dev/clef/bin/devrun timeout 5400 env OMP_NUM_THREADS=8 CLEF_MODEL=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c HF_MODEL=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c CLEF_PARENT=1x4 CLEF_N_LAYERS=4 pytest models/autoports/cloudflare_clef/tests/test_tp2_sanity.py --timeout=3000 > /home/hous/dev/clef/logs/stage0_tp2_sanity_l4_run4.log 2>&1 &
```

### What the test does

- Mesh fixture (module scope): `CLEF_PARENT=1x4` (default, path A: `FABRIC_1D`, `(1,4)` parent) or `CLEF_PARENT=2x2` (path B: `FABRIC_2D`, `(2,2)` parent), `l1_small_size=24576` (`GDN_CONV1D_L1_SMALL_SIZE`), `num_command_queues=2`, `trace_region_size` from `CLEF_TRACE_REGION` (default 0, eager only); yields `create_submesh(MeshShape(1,2), MeshCoordinate(0,0))`; closes the submeshes, then the parent, then sets `FabricConfig.DISABLED`.
- Model: `ClefModelArgs(mesh_device=submesh, max_batch_size=1, max_seq_len=4096)` from `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/tt/loader.py`, truncated with `n_layers` and `attention_type_list[:n]` exactly as `Qwen36Model.from_pretrained` does; raw `model.language_model.*` weights of the active layers plus `lm_head.weight` read from the safetensors and passed through `remap_qwen36_state_dict`; `Qwen36Model(submesh, args, state_dict, tensor_cache_path=<fresh scratch dir>)`; `allocate_kv_caches((64, n_local_kv_heads=2, 64, 256), bf16)`.
- Reference: HF `Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, config=cfg, dtype=bfloat16)` with `cfg.text_config.num_hidden_layers = n` and `layer_types[:n]`, CPU, `OMP_NUM_THREADS=8`; logits at the last position.
- Requests: real Clef requests from `/home/hous/dev/clef/reports/reference/records_text.jsonl` through the release `encode_record` (via `tt/encode.py`). The smallest real request is 260 tokens, so buckets 128 and 256 take the first 128 and 256 tokens of `readme_invoice`; buckets 512, 1024 and 2048 are the `readme_invoice` schema with a long state encoded with `encode_record(max_length=T)`, which truncates the state to exactly `T` tokens.
- Bar: PCC >= 0.97 and matching argmax at the last position. Every case also reports the gap between the two device replicas of the logits (0.0 everywhere, the LM-head all-gather is consistent).

### Results (4 layers: G, G, G, F; path A; stock qwen36)

| Case | Path | PCC vs HF | argmax TT / HF | top-5 TT vs HF | TT first call (s) | TT second call (s) | HF CPU (s) | Verdict |
|---|---|---|---|---|---|---|---|---|
| T=128 | `prefill_masked_bucket`, bucket 128 | 0.999667 | 220 / 220 | first 3 equal | 0.374 | 0.054 | 0.6 | PASS |
| T=256 | masked, bucket 256 | 0.999592 | 220 / 220 | first 4 equal | 17.208 | 0.036 | 1.1 | PASS |
| T=512 | masked, bucket 512 | 0.999640 | 220 / 220 | all 5 equal | 20.913 | 0.034 | 2.1 | PASS |
| T=1024 | masked, bucket 1024 | 0.999605 | 220 / 220 | same set, order of 3rd/4th swapped | 18.547 | 0.049 | 4.0 | PASS |
| T=2048 | `prefill_traced_chunked` with `_chunked_chunk_size=1024` (two 1024 chunks, eager TP chunk-outer, GDN and KV state carried) | 0.999559 | 220 / 220 | all 5 equal | 0.072 | 0.077 | 9.2 | PASS |
| T=2048 | masked, bucket 2048, GDN conv forced to DRAM by a test-side patch (experiment for fallback path 2) | 0.999559 | 220 / 220 | all 5 equal | 19.819 | 0.060 | 9.1 | PASS |
| T=2048 | masked, bucket 2048, stock | n/a | n/a | n/a | fails at first prefill | n/a | n/a | XFAIL: L1 OOM (below) |

Notes on the timings: first calls include program compilation (the 128 bucket compiled in 0.37 s because the on-disk kernel cache from run 1 was warm); second calls are warmed eager runs with host dispatch included, 4 layers only, so they are latency floors and not a 64-layer estimate. The 2048-as-two-1024-chunks first call is fast because the 1024 bucket programs were already compiled by the T=1024 case. Model build (fresh cache write): host state dict 4.8 s, `Qwen36Model` build 31.7 s including the write of about 5.3 GB of per-device tensors; run 1 measured 2.3 s and 21.1 s for the same steps.

Precision used by the sanity run (qwen36 defaults, not a stage 0 choice): `Qwen36ModelArgs.weight_dtype = bfloat8_b` and the `QWEN36_*` knobs at their defaults (`MLP gate/up bfp4`, `MLP down bfp8`, `proj bfp8`; cache tag `gu-bfp4_dn-bfp8_pj-bfp8`, LM head `output.weight.vshard` bfp8, embedding bf16). PCC 0.9996 against bf16 HF at this precision. Stage 4 owns the precision decision.

### Finding 1: no link-count or topology hook is needed

The plan expected the three fused prefill collectives (`num_links = 2` hard-coded in `/home/hous/dev/clef/tt-metal/models/demos/blackhole/qwen36/tt/tp_common.py` lines 388, 433, 566) and the `P300_X2` Ring topology from `ModelArgs.ccl_topology()` to fail on the 1x2. They do not: the fixed probe passes all four link and topology variants for the generic `all_gather_async` and `reduce_scatter_minimal_async`, and the model sanity ran all of its collectives (fused all-gather matmuls, fused matmul reduce-scatter, `tt_all_reduce`, DistributedNorm gathers, LM-head gather) with `num_links=2` and `Topology.Ring` on the submesh (JSON `model.ccl_topology`, `model.ccl_num_links`), with PCC 0.9996 and a zero replica gap. `tp_common.py` and `model_config.py` are unchanged; `git -C /home/hous/dev/clef/tt-metal diff --stat` is empty (the autoport directory is untracked).

### Finding 2: the 2048 bucket overflows L1 at TP=2 in the GDN prefill conv (characterized, not fixed)

Signature (run 1 and run 4): `TT_FATAL: Out of Memory: Not enough space to allocate 20971520 B L1 buffer across 110 banks, where each bank needs to store 192512 B, but bank size is 1436672 B (allocated: 1263872 B, free: 172800 B, largest free block: 161792 B)`, raised from `ttnn.slice` -> `to_layout` -> `tilize` in `_causal_conv1d_fir` (`/home/hous/dev/clef/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` line 222, `x_slice = x_padded[:, k : k + T]`), called from `/home/hous/dev/clef/tt-metal/models/demos/blackhole/qwen36/tt/gdn/tp.py` line 534 with `memory_config=ttnn.L1_MEMORY_CONFIG` (line 541), inside `_forward_prefill_chunk_masked_tp` for a GDN layer. Full traceback: `/home/hous/dev/clef/logs/stage0_tp2_sanity_l4.log` lines 179 to 340.

Derivation: 20,971,520 B = 2048 rows x 5120 columns x 2 B, a `[T=2048, gdn_qkv_dim_tp]` bf16 tensor with `gdn_qkv_dim_tp = (2048 + 2048 + 6144) / 2 = 5120` at TP=2 (JSON `model.gdn_qkv_dim_tp`), against 2560 at TP=4. The TP GDN prefill arm keeps the fused qkvzab projection output and its qkv slice in L1 (`gdn/tp.py` line 514, `out_mc=ttnn.L1_MEMORY_CONFIG`) and runs the conv FIR in L1 (line 541), which allocates the padded conv input, a re-tilized slice per tap and the FIR accumulator, each of that size. Per device L1 is 110 banks x 1,436,672 B = 158.0 MB; the allocator already held 1,263,872 B per bank (139.0 MB, about 6.6 such tensors plus the resident weights and CBs) when the slice asked for another 192,512 B per bank. At bucket 1024 the same tensors are 10 MiB each and fit (T=1024 passes). This is the plan's expected failure class (2) (per-device width doubled at the 2048 chunk) but it is an L1 tensor allocation in `gdn/tp.py`, not a circular-buffer clash in a `_PREFILL_TUNING` matmul, so a `2:` row in `_PREFILL_TUNING` would not change it, and `gdn/tp.py` is outside the plan's allowed change set.

Orchestrator decision (2026 Oct 04, 16:45 ET) and outcome:

- Path 1 (primary, taken): cap the masked bucket and chunk size at 1024 for TP=2 at the engine level. Evidence: buckets 128, 256, 512, 1024 pass (table above), and a 2048-token request runs as two 1024 chunks through `prefill_traced_chunked` with `_chunked_chunk_size = 1024` (the eager TP chunk-outer path `_prefill_chunked_eager_tp`, which calls `_forward_prefill_chunk_masked_tp` per chunk with `chunk_start = 0, 1024`, carries the GDN recurrent and conv state and the paged KV across the boundary, and reads the logits from the last chunk) with PCC 0.999559 and the same top-5 as the single 2048 bucket. `prefill_masked_bucket` also accepts `chunk_start > 0` for a tail (used by that path when the length is not a multiple of the chunk).
- Cost check for path 2: the warmed 2048-token run takes 0.077 s as two 1024 chunks against 0.060 s as one 2048 bucket (the DRAM-conv experiment), a ratio of 1.28x at 4 layers with host dispatch included, under the 1.5x threshold. Path 2 (a `gdn/tp.py` change behind a `ModelArgs` L1-budget flag) is therefore not taken; the experiment shows it would work (PCC 0.999559, identical top-5) if stage 3 needs the 2048 bucket for latency. The experiment is a test-side monkeypatch (`gdn_conv_in_dram` in the test: `_causal_conv1d_fir(memory_config=DRAM)` and `_project_qkvzab(out_mc=None)`), applied only inside that one case; no qwen36 file is modified.

### Finding 3: cached mesh-tensor reload spins on the submesh (open item, worked around)

Runs 2 and 3 (`/home/hous/dev/clef/logs/stage0_tp2_sanity_l4_run2.log`, `..._run3.log`) stalled at 100% host CPU with zero I/O while `ttnn.as_tensor(..., cache_file_name=...)` reloaded a tensor written by run 1: first the row-major bf16 embedding (`tok_embeddings.weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin`, 2.54 GB), then, with that file removed, the first bfp8 TILE per-device layer weight (`layers.0/tp/qkvzab.il_dtype_BFLOAT8_B_layout_TILE.tensorbin`). Host-only `ttnn.load_tensor` of the embedding file returns in 0.0 s with shape `[1, 1, 248320, 2560]` (`/home/hous/dev/clef/logs/stage0_embedding_cache_load_timing.log`), so the spin is in the placement of the loaded multi-device tensor onto the `(1,2)` submesh, after the file read. Run 1 and run 4 wrote fresh caches and never reloaded, and were fast. The device was idle in both stalls (not a device hang; no tt-triage), and both processes were stopped with SIGTERM with the lock free afterwards.

Workaround: unless `CLEF_WEIGHT_CACHE=1`, the test writes its weight cache to `/home/hous/dev/clef/tt_cache/_scratch_tp2_sanity_<pid>` and removes it at teardown. Run 1's cache directory was moved to `/home/hous/dev/clef/tt_cache/_quarantine/` so a stage 1 run cannot reload it by accident. Bounded repro, not yet run (no new device runs after the orchestrator's stop): `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/scripts/cache_reload_probe.py` (`as_tensor` write then reload for sharded bfp8 TILE, replicated bf16 TILE and sharded bf16 ROW_MAJOR, on the `(1,4)` parent and on the `(1,2)` submesh; `CACHE_PROBE_ORDER=sub,parent` to flip the order). Command: `cd /home/hous/dev/clef/tt-metal && /home/hous/dev/clef/bin/devrun timeout 600 python models/autoports/cloudflare_clef/scripts/cache_reload_probe.py`.

### Settings for the stage 1 engine (from this evidence)

- Mesh: path A. `ttnn.set_fabric_config(FABRIC_1D)`, `open_mesh_device(MeshShape(1,4), l1_small_size=24576, num_command_queues=2, trace_region_size=...)`, `create_submesh(MeshShape(1,2), MeshCoordinate(0,0))` (chips `[1, 0]`). Path B also opens and passes the collective probe but was not used for the model run.
- Collectives: stock `qwen36` (2 links, `ccl_topology()` Ring). No hook.
- Prefill: masked buckets up to 1024; requests above 1024 tokens as 1024-token chunks (`_chunked_chunk_size = 1024`, or `capture_prefill_trace_chunked(chunk_size=1024)` when traces arrive in stage 3). Do not use the 2048 bucket at TP=2 unless path 2 is adopted.
- Weight cache: do not reload a mesh-tensor cache on the submesh until the spin is understood; convert from the host tensors each start (about 8 s per layer measured at 4 layers including cache writes; see the estimate in the work log).

## Open items

1. Cached mesh-tensor reload spins on the `(1,2)` submesh (Finding 3). Repro script written, not run. Needs an `autodebug` pass or a run of the repro on the parent and the submesh to localize (submesh-specific or any mesh).
2. The 2048 bucket at TP=2 needs the GDN prefill conv in DRAM (Finding 2, path 2). Only if stage 3 shows the 1024 chunking costs more than it should at 64 layers; the test-side experiment is the template.
3. The 8-layer run (two full-attention layers, layers 3 and 7) did not run: the orchestrator stopped new device runs after run 4. Command when allowed: the run 4 command with `CLEF_N_LAYERS=8` and the log name `stage0_tp2_sanity_l8_run1.log`.
4. Path B (`2x2` parent) was not exercised by the model test, only by the collective probe. Command: the run 4 command with `CLEF_PARENT=2x2`.
5. The HF reference loads the full composite model class with the vision tower; text-only parity is all stage 0 checked.
