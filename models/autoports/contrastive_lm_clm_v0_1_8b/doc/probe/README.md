# Probe: first full-encoder run on p150 and the tensor-cache pinning hang

Date: 2026 Oct 1. Host `qb2-120-p11t01` (2 x p300c, 4 Blackhole chips, TT-KMD 2.10.0, firmware 19.15.0,
IOMMU enabled, transparent hugepages `madvise`). tt-metal `b725040266` (`v0.79.0-dev20260903-104`).

## What the probe does

`tests/probe_full_encoder.py` builds Qwen3-8B through `models.tt_transformers` (`create_tt_model`, bfp8 weights,
`accuracy` precision policy, paged KV cache, 1x1 mesh), runs `Generator.prefill_forward_text(...,
return_hidden_states=True)` on the README example texts, and compares the last-token hidden states (after the
final RMSNorm) with an HF `AutoModel` bf16 CPU reference. It also pushes both embeddings through the CLM heads.

## Result (run 3, `probe_full_encoder.json`)

| item | value |
|---|---|
| device | P150 (one chip), 36 layers, bfp8 weights, bf16 activations |
| model load | 36.1 s (layer 0 from cache, layers 1 to 35 converted) |
| first traced call (128-token bucket) | 12.6 s |
| warm single text | 57.8 ms |
| warm batch of 4 texts (one prefill pass) | 92 ms |
| run-to-run determinism (cosine) | 1.0000 |
| batched vs single (cosine) | 0.9997 to 0.9998 |
| TT vs HF bf16 (cosine, 7 texts) | 0.9987 to 0.9998 |
| tides ranking, TT vs HF | Moon 0.9921 vs 0.9931; round 0.0079 vs 0.0069; photosynthesis 3.3e-5 vs 2.5e-5 |
| customer routing, TT vs HF | billing 0.9904 vs 0.9896 |

## The hang, root cause and workaround

Runs 1 and 2 and a stock control (`models/tt_transformers/tests/test_decoder_prefill.py` forced to a 1x1 mesh)
never finished model construction. The host main thread spun at 100 % CPU in user space, device watcher dumps
showed every core idle, and `dmesg` reported `tenstorrent 0000:02:00.0: could only pin 512 of 16385 pages`.
`perf record` on the stuck process put the time in the kernel driver:
`tt_cdev_ioctl -> ioctl_pin_pages -> pin_user_pages_fast -> gup_fast_fallback -> __gup_longterm_locked ->
handle_mm_fault -> __split_huge_pmd -> native_flush_tlb_local`, called from UMD `tt_pin_pages`.

Isolation (`/home/hous/dev/clm-v0.1-8B/bin/cache-load-test.py`, 64 MB bf16 tile tensor, 1x1 mesh):

| step | result |
|---|---|
| `ttnn.as_tensor(..., cache_file_name=...)` create + write | 0.043 s |
| fresh process, same call, loads the `.tensorbin` from disk to device | never returns (killed at 120 s), new `could only pin` line |
| same with `TT_METAL_DISABLE_DMA_OPS=1` | still hangs |
| same with `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` | 0.063 s, round trip exact |

Mechanism: loading a cached tensor memory-maps the file. The host-to-device write path tries to long-term pin
that host buffer (`PinnedMemoryCache::try_pin`, `tt_metal/distributed/pinned_memory_cache.cpp`). The kernel cannot
long-term pin file-backed pages beyond the first 2 MB PMD; the ioctl fails partially and the user-mode driver
retries without end. Fresh conversions (anonymous memory) are unaffected, which is why the first run got as far as
layer 0 and why a stock test with a fresh cache directory passed on a 1x4 mesh.

Workaround used everywhere in this port: `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`. With it, `try_pin` returns
null and writes take the unpinned path. Measured cost: none visible (1.28 GB tile write 0.19 s vs 0.53 s pinned).
Set in `/home/hous/dev/clm-v0.1-8B/bin/ttenv.sh`, `tt/encoder.py` (`open_mesh` default) and `tt-model.yaml`
(`serve.env`). This should be reported upstream against tt-metal: either copy mmapped cache tensors into
anonymous memory before pinning, or skip pinning for file-backed ranges.

## Shutdown message

After `PROBE_RESULT`, three `TT_THROW: SubDeviceManagerTracker is not initialized on MeshDevice 0` lines appeared
during interpreter exit because the `Generator` still held traces when `ttnn.close_mesh_device` ran. The encoder
now releases traces and model objects (`TtQwen3Encoder.release()`) before the mesh closes.

## Evidence

- `probe_full_encoder.json`, `tt_single.npy`, `tt_batch4.npy`, `hf_bf16.npy` (this directory)
- `/home/hous/dev/clm-v0.1-8B/logs/probe_full_encoder.log` (run 1), `probe_full_encoder_run2.log`,
  `probe_full_encoder_run3.log`, `control_test_decoder_prefill*.log`, `cache_load_*.log`, `dma_pin0.log`
- `work_log.md` (timeline)
