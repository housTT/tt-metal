# Qwen3.5-35B-A3B functional decoder

Stage 01. Complete. All functional-decoder gates pass; the independent [final review](stage_review_final.md) returned `clean-pass`.
The implementation preserves the advertised 262144-token text context on one
Blackhole ASIC. Target checkpoint revision:
`59d61f3ce65a6d9863b86d2e96597125219dc754`; checkout base:
`98dc4ae62ccfc32836ddcfad50f9fd97863612d4`; branch `feat/qwen3-5-35b-a3b`.
Four ASICs are available; this stage uses mesh 1×1 and 110 user cores. It implements
one decoder layer, including its real 256-expert top-8 MoE and shared expert.

## Contract

[contract.md](contract.md) specifies ownership, shapes, precision boundaries,
padding, positions and trace preparation. The exact target config is stored in
`../../tests/target_config.json`; [context_contract.json](../context_contract.json)
records advertised and measured capacity.

```python
FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx,
    mesh_device, tensor_cache_path=None, max_batch_size=32,
    expert_precision="inherited")
model.bind_paged_kv_cache(key_cache, value_cache)
model.prefill_forward(hidden_states, *, valid_len=None, page_table=None,
    start_pos=0, request_slots=None)
model.decode_forward(hidden_states, *, position_tensor=None, page_table=None,
    request_slots=None)
model.reset_request_state(request_slots=None)
```

Inputs/outputs are BF16 TTNN `[B,T,2048]` for prefill and `[B,1,2048]` for decode.
Explicit `valid_len` selects logical rows inside a prepared physical input bucket;
the output preserves physical padding but exposes the logical shape. Full attention
uses caller-owned BF16 paged KV, int32 row-major page tables, and absolute device
positions. Linear attention keeps independent convolution and FP32 recurrent state
per selected request slot. Prefix continuation retains state; fresh requests reset
GDN slots before prefill. Batch rows execute separately on the single ASIC.

Layers 0,1,2,4,… are `linear_attention` (K16/V32, head dimension 128, convolution width 4). Layers 3,7,…,39
are `full_attention` (Q16/KV2, head dimension 256, 64 partial-RoPE coordinates). Both use hidden dimension 2048,
MoE intermediate dimension 512 and a gated shared expert of dimension 512. Tests load real layers 0 and 3.
Default routed/shared gate-up weights are BFP4 and down weights BFP8. Full-attention
projection weights are BFP8/LoFi; GDN projection weights are BF16/HiFi4 with FP32
accumulation/state. The two outer zero-centered norms (`input_layernorm` and
`post_attention_layernorm`) and router probability arithmetic retain FP32
boundaries. Inherited full-attention Q/K norm offsets are prepared in BF16.
The optional BF16-expert setting is a recorded diagnostic control.

## Correctness and capability

All acceptance thresholds remain PCC ≥ 0.995. Headline tests compare every prefill
output and gate every individual decode token. Real pinned weights are paired with
seeded BF16 hidden activations at the actual target shapes.

| Layer kind | 4096 prefill PCC | Minimum of 128 traced decode PCCs | 262144 prefill PCC | Decode at position 262143 PCC |
|---|---:|---:|---:|---:|
| linear_attention | 0.999701798 | 0.997864842 | 0.999611497 | 0.999875247 |
| full_attention | 0.998596847 | 0.995969236 | 0.998373568 | 0.998352349 |

Headline artifacts: `linear_accept_4096_128.json`, `full_accept_4096_128.json`.
Native-context artifacts: `{linear,full}_accept_context26214{3,4}.json`, with matching
console logs. No physical context reduction was needed. Native HF comparison keeps
every recurrent/KV contribution and compares 16 selected prefill outputs; TT executes
the entire layer for every token. `hf_sparse_control.json` validates this reference
against dense HF at 1153 tokens. Native decode consumes the complete filled prefix.

`positions_linear_sfpu.json` and `positions_full_accept.json` cover 32 independent
requests with heterogeneous prefixes 1..1006, reversed/nonzero slots, changing
per-row positions, 8 traced steps, exact repeated replay, and eager-vs-replay equality.
The full-attention run permutes ownership and checks nonzero guard pages.
`{linear,full}_reuse_accept.json` and `full_reuse_page128.json` cover 17 B2 requests:
1,31,32,33,127,128,129,511,512,513,1153, return to 33, padded 35/36, and prefix continuation.
Logical 36 and continuation start 36 are withheld from warmup within prepared physical
buckets. One trace survives all requests without recapture or forbidden program
cache misses; page sizes 32 and 128 are tested.

`pytest_device_accept.log` records four target-shape 129/8 tests: real and statistics-
matched synthetic weights for each layer kind. Synthetic tests use saved per-tensor
mean/std/shape/dtype and an offline config fixture. `cpu_contract_postreview.log` records
nine host contract tests. Run artifacts, commands and exit codes are indexed in
`acceptance_commands.json`; failed/superseded controls remain separately identifiable.

Small-pool regressions use exactly three or nine 32-token pages per row, with
no guard/slack allocation: `small_pool3_b{1,2}.json` and `small_pool9_b2.json`.
They validate unpadded update tables and the 32-token attention chunk fallback
for odd page capacities. `full_reuse_postreview.json` repeats the 17-case trace
reuse matrix after this repair. `completion_commands.json` records the commands.

| Capability claim | Evidence | Remaining scope/risk |
|---|---|---|
| Both real HF layer kinds and active MoE | Headline JSONs; saved weight statistics | Layer inputs are synthetic hidden states; no full-model claim |
| Native 262144-token context and near-limit tail | Four native-context JSONs; sparse-reference control | HF checks 16 prefill outputs while retaining all contributions |
| Page ownership and absolute positions | B32 position tests, B2 reuse, exact small pools | Batch 32 and native context tested separately |
| Arbitrary logical tails and continuation | 17-case reuse matrix, lengths 35/36 withheld from warmup | New physical buckets require preparation before tracing |
| Stable device-only traced decode | Runtime audit, allocation tracking, replay equality | Input refresh and result checking are explicit host boundaries |

## Device-only execution, determinism and diagnostics

`runtime_audit.py` blocks Torch dispatch and TTNN host conversions inside both
forward methods. Audited runs include headline, native context, batch32, and mixed
request reuse. All decode acceptance runs capture actual TTNN execution, refresh
stable inputs/positions, forbid program-cache misses, and use allocation tracking
including program-cache allocations. Only the final output tensor allocation is
acknowledged as intentionally overwritten by replay. Persistent state and caches
predate capture. Repeated identical inputs produce bit-identical outputs.

`AUTODEBUG.md`, `AUTOFIX.md`, and `stage_review_initial.md` preserve the diagnosis,
failed controls, and required review repairs. The GDN token recurrence uses explicit
FP32 SFPU operations after the native FPU path's repeated TF32 state rounding caused
MoE routing differences at long unaligned tails. This change fixes tails and decode;
aligned prefill retains the exact-unit-diagonal native chunk algorithm.

Watcher and profiler runs use separate processes. Startup warnings about an unknown
motherboard grouping (PCI fallback), opening a subset of MMIO devices, and another
user's memory-statistics shared-memory object do not stop device execution. No reset,
firmware/topology change, or unrelated process termination has occurred in this stage.
`watcher_summary.json` indexes five clean logs: normal attach/dump/detach records,
with no assertion, invalid-address, corruption, stack-overflow, or NoC error. The
four passing device pytest cases emit nanobind `CoreRangeSet`/`MemoryConfig`
reference-count warnings at interpreter shutdown, after device/driver close.
These are binding shutdown metadata; no live tensor or trace leak is reported.

## Performance

Warmed 4096/128, batch 1, one request; complete single-layer device spans:

| Layer kind | Prefill device ms | Mean traced decode device ms |
|---|---:|---:|
| linear_attention | 1743.870 | 2.190 |
| full_attention | 1720.532 | 1.609 |

[functional_decoder.md](functional_decoder.md) defines the complete-layer timing
window, roofline assumptions, precision, nominal ASIC peaks, and exact CSV/table
artifacts. All operations and internal device gaps remain in the denominator.
Each layer has separate human-readable tt-perf-report tables, filtered CSVs,
lossless source CSV archives, and provenance. Profiling failures were repaired
and preserved; no partial capture supplies a measurement.

## Reproduction

From the checkout, source `../scripts/environment.sh`. Hardware commands are serial.

```bash
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_acceptance
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer \
  --layer 0 --length 4096 --steps 128 --trace --audit --output <result.json>
```

Use layer 3 for full attention. `run_acceptance.py` records its exact subprocess
commands and environments; the native tests add `--sparse-reference`, and watcher
runs set `TT_METAL_WATCHER=10` with unique log directories. The tests exercise layer
mathematics with supplied hidden states. Full-model generation, optimized decoder,
multi-chip and serving work belong to subsequent goals.

Commit-time formatting preserves all production ASTs.
`commit_format_provenance.json` maps measured source hashes to final bytes;
`precommit_source.log` records passing repository hooks on current code.
Historical evidence retains its original bytes.
