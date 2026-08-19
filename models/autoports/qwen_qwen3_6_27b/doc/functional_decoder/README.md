# Qwen3.6-27B functional decoder

This stage implements the single-device functional decoder for `Qwen/Qwen3.6-27B`. It covers both meaningful `qwen3_5_text` layer kinds with real checkpoint weights: layer 0 represents Gated DeltaNet (`linear_attention`) and layer 3 represents gated GQA (`full_attention`). It does not contain optimized-decoder, multichip, full-model, or serving work.

## Runtime contract

`tt/functional_decoder.py` exposes these keyword-explicit entry points:

```python
prefill_forward(
    self, hidden_states, *, logical_seq_len: int, cos=None, sin=None,
    page_table=None, kv_cache=None, linear_state=None,
)
decode_forward(
    self, hidden_states, *, current_positions, cos=None, sin=None,
    page_table=None, kv_cache=None, linear_state=None,
)
```

Their tensor contract is:

- Prefill hidden states are `[batch, 1, physical_sequence, 5120]`; `logical_seq_len` excludes tile padding.
- Linear attention owns caller-supplied convolution state `[batch, 1, 4, 10240]` and recurrent state `[batch, 48, 128, 128]`.
- Full attention owns caller-supplied paged K/V caches `[num_blocks, 4, block_size, 256]`, a device `int32` page table, and RoPE tensors. Short prefill uses causal SDPA; sequences over 32,768 use 4,096-token paged chunks and chunked SDPA.
- Decode hidden states are `[1, 1, padded_batch, 5120]`. Full attention requires a device `int32` current-position vector and page table. Both decode paths are trace-capture/replay safe.
- Up to 32 active users are correctness-gated. Functional prefill builds independent per-user TTNN graphs when outer-batch kernels lose accuracy; full decode uses an explicit 8x8 exact-exp SDPA program configuration.
- Setup converts real HF weights once. A measured forward closure contains no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback.

The HF-native context is 262,144 tokens. Real-weight prefill passed at 262,144 for both layer kinds, and traced decode passed at current position 262,143. There is no capability reduction; see `../context_contract.json`.

## Correctness evidence

All acceptance comparisons use PCC `>= 0.995`.

| Gate | PCC / result |
|---|---:|
| Linear prefill, 64 tokens | 0.9983487530 |
| Linear first decode after 64 | 0.9987730167 |
| Linear traced decode after 64 | 0.9991342265 |
| Linear prefill, 65 tokens | 0.9983532128 |
| Linear traced decode after 65 | 0.9996729401 |
| Full paged prefill, 65 logical / 96 physical | 0.9986537700 |
| Full paged decode after that prefill | 0.9979848879 |
| Full traced decode | 0.9984844516 |
| Full forced-chunk prefill, 257 logical / 384 physical | 0.9984344296 |
| Full traced decode at position 262,143 | 0.9997240988 |
| Two-user tests, minimum aggregate/per-user PCC | 0.9970185861 |
| 32-active-user prefill/decode, minimum per-user PCC | 0.9970447455 |
| Repeated-input / repeated-replay determinism | 1.0 |

Real execution covers lengths 1, 31, 32, 33, 63, 64, 65, the non-aligned long-full path at 32,769, and native limits 262,143/262,144. The 257-token forced-chunk test compares the actual long-path algorithm to the HF layer while forcing 128-token chunks, including a one-token logical tail. Page-table tests use reversed and disjoint physical mappings. The two-user full-attention test decodes active users at distinct positions 32 and 64; padded lanes receive distinct pages to avoid update races.

`real_weight_stats.json` records every representative tensor's shape, shard, population statistics, checkpoint snapshot `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, and provenance.

## Performance evidence

Measurements use a Blackhole 1x1 mesh, real weights, warmed 32-token prefill, and one warmed TTNN trace replay for decode. `perf/summary.csv` is the concise index.

| Layer kind | Warmed prefill device time | Traced decode device time |
|---|---:|---:|
| Linear attention | 6.046 ms | 2.901 ms/token |
| Full attention | 2.594 ms | 2.639 ms/token |

Human-readable per-op tables are in `perf/reports/*.txt`, normalized CSVs are in `perf/reports/*.csv`, and Tracy host-op provenance plus enriched device-op reports are in the four `perf/raw_{linear,full}_{prefill,decode}` directories. The installed profiler is an older revision and required legacy post-processing with pandas string inference disabled; `triage/AUTODEBUG_TRACY.md` and `work_log.md` preserve that provenance. Large duplicated binary/raw-device captures were intentionally not retained after report generation.

## Reproduction

Run hardware tests outside the checkout so installed-runtime kernel sources are not shadowed:

```bash
cd /tmp
env \
  PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
  TT_METAL_CACHE=/tmp/qwen36_functional_tt_cache \
  pytest -q /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py -s
```

The complete-suite watcher artifacts are under `watcher/final/`; post-review long-context and batch-32 regressions are under `watcher/review_rerun/` and `watcher/batch32/`. Device-open diagnosis is in `triage/AUTODEBUG.md`; profiler diagnosis is in `triage/AUTODEBUG_TRACY.md`.
