# Stage 0 (device): work log

Append-only. One entry per run. Times are ET (UTC-4); the host clock is UTC. Every device command ran through `/home/hous/dev/clef/bin/devrun` with a `timeout`.

## 2026 Oct 04 16:11 ET: first mesh probe (orchestrator run, recorded here for completeness)

Command: `/home/hous/dev/clef/bin/devrun timeout 600 python models/autoports/cloudflare_clef/scripts/mesh_probe.py /home/hous/dev/clef/reports/stage0_mesh_probe.json`. Log `/home/hous/dev/clef/logs/stage0_mesh_probe.log`.

Outcome: direct `(1,2)` open under `FABRIC_1D` fails (fabric router sync timeout on device 0, 19.7 s). `FABRIC_2D` + `(2,2)` parent + `(1,2)` submesh at offset `(0,0)` opens (chips `[1, 0]`). `FABRIC_1D` + `(1,4)` parent + `(1,2)` submesh opens (same chips). Cluster type `ClusterType.P300_X2`. Both collective smokes failed with `RuntimeError: BFloat16 did not match Float`: the probe compared a bf16 `ttnn.to_torch` result with a float32 tensor in `torch.allclose`. That is a probe bug, not a device result.

## 2026 Oct 04 16:19 ET: collective probe, fixed

Fix in `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/scripts/mesh_probe.py`: `collective_smoke` now casts the device result to float32 before the compare, and it mirrors the model's generic collectives instead of `ttnn.all_gather` (whose `num_links` is deprecated and ignored): `TT_CCL(mesh)` + `tt_all_gather(..., cluster_axis=None, dim=3)` (`all_gather_async`) and `tt_all_reduce(..., cluster_axis=0, dim=3)` (`reduce_scatter_minimal_async`), from `/home/hous/dev/clef/tt-metal/models/tt_transformers/tt/ccl.py`. `collective_variants` runs four variants in order (1 link Linear, 2 links Linear, 1 link Ring, 2 links Ring) and dumps the JSON after each one so a hang leaves evidence.

Command:

```
cd /home/hous/dev/clef && nohup /home/hous/dev/clef/bin/devrun timeout 600 env PROBES=parent_ccl,1x4 python models/autoports/cloudflare_clef/scripts/mesh_probe.py /home/hous/dev/clef/reports/stage0_mesh_probe_ccl.json > /home/hous/dev/clef/logs/stage0_mesh_probe_ccl.log 2>&1 &
```

Outcome: all eight variants pass on both parents. All-gather max abs error 0.0, reduce-scatter max abs error 0.03125 (one bf16 ulp at magnitude 4 to 8). `parent_2x2_submesh_1x2_collectives` 2.13 s, `direct_1x4_with_1x2_submesh` 1.32 s. JSON `/home/hous/dev/clef/reports/stage0_mesh_probe_ccl.json`, log `/home/hous/dev/clef/logs/stage0_mesh_probe_ccl.log`. Consequence: the plan's expected failure (1) (collectives reject 2 links or Ring on the 1x2) does not reproduce for the generic collectives, so the model test ran first with no change to `qwen36`.

## 2026 Oct 04 16:21 ET: HF reference check (host)

Command: `OMP_NUM_THREADS=8 CLEF_MODEL=<snapshot> HF_MODEL=<snapshot> /home/hous/dev/clef/bin/hostrun python <scratchpad>/hf_ref_check.py 4`, log `/home/hous/dev/clef/logs/stage0_hf_ref_check_l4.log`.

Outcome: `Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, config=cfg, dtype=bfloat16)` with `cfg.text_config.num_hidden_layers = 4` and `layer_types[:4]` loads 4.53 G parameters in 0.1 s (lazy safetensors), reports the layers 4 to 63 as UNEXPECTED, and runs text-only: 128 tokens in 0.3 s, 2048 tokens in 4.6 s, finite logits of shape `[1, T, 248320]`, argmax 220 at both lengths. The smallest real Clef request is 260 tokens (`readme_invoice`), and a schema alone is 290 tokens for `readme_checkout`, so the 128 bucket uses the first 128 tokens of `readme_invoice`; the 2048 bucket is a full request (`readme_invoice` questions with a long state built from the 16 reference records) encoded with `max_length=2048`, which `encode_record` truncates to exactly 2048 tokens.

## 2026 Oct 04 16:22 ET: TP=2 sanity, 4 layers, run 1 (stock qwen36)

Test: `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/tests/test_tp2_sanity.py` (own module-scoped fixture: path A by default, `CLEF_PARENT=2x2` for path B; `ClefModelArgs` on the submesh, `n_layers` truncation as `Qwen36Model.from_pretrained` does it, the raw language weights plus `lm_head.weight` read from the safetensors and remapped with `remap_qwen36_state_dict`, `Qwen36Model(...)` built directly; HF reference `Qwen3_5ForConditionalGeneration` with `text_config.num_hidden_layers` truncated, bf16, CPU).

Command:

```
cd /home/hous/dev/clef && nohup /home/hous/dev/clef/bin/devrun timeout 5400 env OMP_NUM_THREADS=8 CLEF_MODEL=<snapshot> HF_MODEL=<snapshot> CLEF_PARENT=1x4 CLEF_N_LAYERS=4 pytest models/autoports/cloudflare_clef/tests/test_tp2_sanity.py --timeout=3000 > /home/hous/dev/clef/logs/stage0_tp2_sanity_l4.log 2>&1 &
```

Outcome (`/home/hous/dev/clef/logs/stage0_tp2_sanity_l4.log`, JSON `/home/hous/dev/clef/reports/stage0_tp2_sanity_l4_1x4.json`, later overwritten by run 2 with a superset of the fields):

- Model args on the submesh: device name `P300`, `ccl_topology() = Topology.Ring`, `TT_CCL.get_num_links() = 2`, `prefill_tuning` = the TP=4 row, `n_local_heads 12`, `n_local_kv_heads 2`, `kv_replication False`. Host state dict 2.3 s, model build 21.1 s.
- `T=128`, masked bucket 128: PCC 0.999667 against HF, argmax 220 on both (the space token), TT top-5 `[220, 13, 318, 30, 12]`, HF top-5 `[220, 13, 318, 11, 0]`, first call 23.9 s (compile), second call 0.021 s, HF CPU 0.3 s, replica gap 0.0. PASS. No qwen36 change: the fused prefill collectives ran with their hard-coded 2 links and the `P300_X2` Ring topology.
- `T=2048`, masked bucket 2048: FAIL at the first prefill with `TT_FATAL: Out of Memory: Not enough space to allocate 20971520 B L1 buffer across 110 banks, where each bank needs to store 192512 B, but bank size is 1436672 B (allocated: 1263872 B, free: 172800 B, largest free block: 161792 B)`. Python frames: `model.py:2161 prefill_masked_bucket` -> `:1978 _forward_prefill_chunk_masked` -> `:2095 _forward_prefill_chunk_masked_tp` (a GDN layer) -> `layer.py:240` -> `gdn/tp.py:534 forward_prefill` -> `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py:222 _causal_conv1d_fir`, `x_slice = x_padded[:, k : k + T]` -> `ttnn.slice` -> `to_layout` -> `tilize` output allocation in L1.
- Derivation: 20971520 B = 2048 x 5120 x 2 B, a `[T=2048, gdn_qkv_dim_tp=10240/2=5120]` bf16 tensor. `gdn/tp.py:514` places the fused qkvzab projection and its qkv slice in L1 (`out_mc=ttnn.L1_MEMORY_CONFIG`) and `gdn/tp.py:541` runs the conv FIR with `memory_config=ttnn.L1_MEMORY_CONFIG`, so the padded conv input, each tap slice (re-tilized), and the FIR accumulator are all L1 tensors of that size. At TP=4 the same tensors are 10 MiB each (width 2560) and the validated 2048 chunk fits; at TP=2 they double and the allocator has 139 MB of the 158 MB L1 (110 banks x 1436672 B) in use when the slice asks for another 20 MiB. This is the plan's expected failure (2) class (per-device width doubled at the 2048 chunk) but the overflow is an L1 tensor allocation in the GDN prefill arm, not a circular-buffer clash in a `_PREFILL_TUNING` matmul, so a `2:` tuning row cannot fix it. The file that owns the placement (`gdn/tp.py`) is outside the plan's allowed change set.

## 2026 Oct 04 16:27 ET: TP=2 sanity, 4 layers, run 2 (stalled on the embedding cache; stopped)

Test extended to four cases per the orchestrator's decision: `T128-masked`, `T2048-chunked1024` (`prefill_traced_chunked` with `_chunked_chunk_size = 1024`, the eager TP chunk-outer path `_prefill_chunked_eager_tp`: two 1024-token chunks, GDN and KV state carried, logits from the last chunk), `T2048-masked-conv-dram` (the stock 2048 masked bucket with a test-side patch that forces `_causal_conv1d_fir(memory_config=DRAM)` and `_project_qkvzab(out_mc=None)` in `gdn/tp.py`, as evidence for the fallback path 2), and `T2048-masked` (stock, expected L1 OOM, recorded and xfailed).

Command: as run 1 with the log `/home/hous/dev/clef/logs/stage0_tp2_sanity_l4_run2.log`.

Outcome: no log line after `20:27:33 UTC` (the end of `ModelArgs.__init__`). After 22 min: main thread at 100% CPU in user space, 59 threads, state R, `/proc/<pid>/fd` held `/home/hous/dev/clef/tt_cache/P300/tensor_cache_bfp8_mesh1x2_clef_2f3de3dd_gu-bfp4_dn-bfp8_pj-bfp8/tok_embeddings.weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin` (2,542,797,248 B, written by run 1) open for reading, `rchar` 315 MB and not moving over a 10 s sample. `py-spy dump` (with and without sudo) printed no frames for this Python 3.12.14 build. Diagnosis: a host-side spin in the cached reload of the row-major bf16 embedding (`Embedding` in `/home/hous/dev/clef/tt-metal/models/tt_transformers/tt/embedding.py` calls `ttnn.as_tensor(..., layout=ROW_MAJOR, mesh_mapper=ShardTensor2dMesh(dims=(None, 3)), cache_file_name=...)`); run 1 took this path as a cache write (0.6 s between the `Loading 4 transformer layers` log and the embedding cache timestamp) and never reloaded. The device was idle (no device op in flight, process not blocked on the driver), so tt-triage did not apply; the process was stopped with SIGTERM at 16:50 ET, the device lock was free afterwards, and the embedding cache file was moved to `/home/hous/dev/clef/tt_cache/_quarantine/` for a host-only load timing (`/home/hous/dev/clef/logs/stage0_embedding_cache_load_timing.log`).

Workaround in the test: the `tt_model` fixture deletes `tok_embeddings.weight*` from the weight cache directory before `Qwen36Model(...)`, so the embedding is converted from the host tensor on every run (recorded in the JSON as `dropped_embedding_cache_files`). Open item for stage 1: the engine must not reload the row-major embedding cache (skip caching for `tok_embeddings`, or cache it in TILE layout) until the slow path is understood.

## 2026 Oct 04 16:51 ET: TP=2 sanity, 4 layers, run 3 (stalled on the first cached layer tensor; stopped)

Command: as run 1 with the log `/home/hous/dev/clef/logs/stage0_tp2_sanity_l4_run3.log`; the embedding cache file had been moved aside so the embedding converted from the host tensor (0.6 s), and `Loading 4 transformer layers` printed at `20:51:17 UTC`.

Outcome: the next log line never came. After 6 min the main thread was at 100% CPU, `rchar` flat, with `/home/hous/dev/clef/tt_cache/P300/tensor_cache_bfp8_mesh1x2_clef_2f3de3dd_gu-bfp4_dn-bfp8_pj-bfp8/layers.0/tp/qkvzab.il_dtype_BFLOAT8_B_layout_TILE.tensorbin` open: the first cached per-device weight of layer 0 (bfp8, TILE, a `tpc.shard_w` tensor). So the spin is not specific to the row-major embedding: every cached mesh-sharded tensor reload onto the `(1,2)` submesh spins. Host-only `ttnn.load_tensor` of the quarantined embedding cache returns in 0.0 s with shape `[1, 1, 248320, 2560]` (the two per-device shards), so the file read is fine and the spin is in the device placement of the loaded multi-device tensor (`/home/hous/dev/clef/logs/stage0_embedding_cache_load_timing.log`). Run 1 (fresh cache) built the whole 4-layer model in 21 s because it only wrote caches. Process stopped with SIGTERM at 16:58 ET (device idle, lock free afterwards).

Workaround in the test: unless `CLEF_WEIGHT_CACHE=1`, the `tt_model` fixture points `tensor_cache_path` at a fresh `/home/hous/dev/clef/tt_cache/_scratch_tp2_sanity_<pid>` directory and removes it at teardown, so every weight converts from the host tensor and nothing is reloaded. Bounded repro for the open item: `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/scripts/cache_reload_probe.py` (as_tensor write then reload, sharded bfp8 TILE, replicated bf16 TILE, sharded bf16 ROW_MAJOR, on the parent and on the submesh).

## 2026 Oct 04 16:59 ET: TP=2 sanity, 4 layers, run 4 (final)

Command: as run 1 with `CLEF_N_LAYERS=4 CLEF_PARENT=1x4` and the log `/home/hous/dev/clef/logs/stage0_tp2_sanity_l4_run4.log`; seven cases (masked 128, 256, 512, 1024; 2048 as two 1024 chunks; 2048 masked with the GDN conv forced to DRAM; 2048 masked stock). Fresh scratch weight cache.

Outcome: `6 passed, 1 xfailed in 141.90 s`. JSON `/home/hous/dev/clef/reports/stage0_tp2_sanity_l4_1x4.json`. Host state dict 4.8 s, model build 31.7 s (writes about 5.3 GB of per-device tensors), HF 4-layer load 0.2 s. Per case (PCC, argmax TT/HF, TT first and second call s, HF CPU s): T128 masked 0.999667, 220/220, 0.374 / 0.054, 0.6; T256 masked 0.999592, 220/220, 17.208 / 0.036, 1.1; T512 masked 0.999640, 220/220, 20.913 / 0.034, 2.1; T1024 masked 0.999605, 220/220, 18.547 / 0.049, 4.0; T2048 as two 1024 chunks 0.999559, 220/220, 0.072 / 0.077, 9.2; T2048 masked with DRAM conv 0.999559, 220/220, 19.819 / 0.060, 9.1; T2048 masked stock: the run 1 L1 OOM again (xfail). Replica gap 0.0 in every passing case. The stale run 1 cache directory was moved to `/home/hous/dev/clef/tt_cache/_quarantine/` after the run; the scratch cache was removed by the fixture.

Estimate for 64 layers from these numbers (not measured): the 4-layer build spent about 31.7 s, of which the embedding (2.5 GB bf16) and the LM head (1.35 GB bfp8 shard) are one-off; run 1 measured 21.1 s for the same build, so the per-layer host conversion plus device write is roughly 3 to 6 s per layer with cache writes on. 64 layers: about 4 to 7 min for the layer weights plus the one-off tensors, plus a host state dict read of about 55 GB from the safetensors (4.8 s for 4 layers and the embedding; expect 1 to 2 min for 64 layers from page cache, longer from disk). Without cache writes this should be somewhat lower; stage 1 should measure it.

No further device runs after this one (orchestrator instruction). The 8-layer run and the path B model run are open items.
