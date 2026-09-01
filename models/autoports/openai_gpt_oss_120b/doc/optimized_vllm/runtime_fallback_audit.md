# Optimized vLLM runtime fallback audit

Verdict: **clean for the measured P150x4 serving path**.

## Measured-path evidence

The final benchmark-only capability snapshot reports:

- 226 decode calls, all 226 device sampled;
- 0 host-sampled decodes;
- 226 nonblocking model trace submissions and 226 nonblocking sampling trace
  submissions;
- 0 unclassified execute submissions;
- 226 async output collections;
- 226 minimal token readbacks, 226 device shards transferred, and 678
  replicated TP shards skipped;
- 0 host argmax calls, 0 full-logits readbacks, 0 forced-token refreshes, and
  0 validation full-logit synchronizations;
- 224 fixed sampling-state replays after two phase initializations;
- 221 unchanged page-table reuses.

The exact snapshot is
`artifacts/after_final_remediation/vllm_serving_capability.json`.

## Source-path audit

| Boundary | Result |
| --- | --- |
| vLLM adapter | `decode_forward(..., read_from_device=False)` returns device output. |
| Model replay | Shared generator submits the retained B1/B32 model trace nonblocking. |
| Sampling replay | `models/common/sampling/generator.py` calls `ttnn.execute_trace(..., blocking=False)`. |
| Token read | GPT-OSS selects one distinct shard, then calls `.cpu(blocking=False)`. |
| Synchronization | Plugin finalization synchronizes the returned event after the async submission boundary. |
| Host conversion | Only the selected host-resident token shard is converted with `ttnn.to_torch`; there is no full-logit conversion. |
| Token feedback | The sampler writes `tt_out_tok`, the persistent next-token trace input. |
| Position/RoPE | Persistent trace inputs progress on device for fixed replay. |
| Page tables | Persistent per-layer tensors update only on scheduler-table changes and reuse otherwise. |
| Cache | vLLM-owned per-layer hybrid KV buffers are bound directly; there is no standalone cache fallback. |
| Sampling | Canonical full-model vocab-sharded split sampling; no adapter sampler or force-argmax path. |

Host `torch.argmax` and full-logit compatibility code remains in the standalone
validation/generation API. The final measured snapshot proves those branches
were not entered. Likewise, explicit host sampling remains available for
unsupported plugin sampling controls and was exercised by the 73-pass
correctness suite, not by either benchmark.

No extra `torch`/`from_torch`/`to_torch`, tilize/untilize, reshard, blocking
readback, private cache allocation, or plugin/vLLM checkout fallback was added
to steady measured decode. The retained change removes device-to-host work.

The final full-36 allocation-tracker run left program-cache accounting enabled
and completed 226 B1/B32 model plus sampler replays with zero unsafe live
buffers. Startup now inspects KV-cache block size without indexing a device
tensor and precompiles the exact explicit B1 paged-fill signature before trace
capture; neither change adds steady decode work.

## Cleanup

The TT worker calls `release_persistent_capture()` before closing the mesh.
The adapter writes the final counters first, tears down generator-owned model
and sampler traces, clears released trace stores, and drops its generator
reference. The final server terminated cleanly. The timestamped
`artifacts/after_final_remediation/post_run_process_audit.txt` contains no
vLLM API server or EngineCore process, and its paired `post_run_tt_smi.json`
reports all four Blackhole boards DRAM-healthy with no device holder. No reset
was needed.

Known nanobind reference-leak diagnostics still print during interpreter exit.
They occur after explicit runner shutdown and do not leave a process or device
holder; they are not a serving fallback or request failure.

## Profiling boundary

No Tracy, `tt-perf-report`, live-server device profiler, adapter profiler, or
`ReadDeviceProfiler` collection was attempted. This is intentional under the
vLLM-serving optimization contract. `perf_summary.json` leaves device and
roofline fields null with reason
`vllm_serving_profiler_disabled_to_protect_hardware`.
