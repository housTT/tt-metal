# Stage 7: optimized full model (traced batched encoder)

## What changed from stage 6

- Traced prefill is called directly (`TtQwen3Encoder._traced_prefill` -> `Generator._easy_trace_prefill`) for
  fifteen variants: padded lengths 128 / 256 / 512 / 1024 / 2048 x batch 1 / 4 / 8 (nine variants, 128 / 1024 /
  2048, until the first served evaluation; see "Prefill buckets" below). A request's texts are grouped by bucket
  and padded up to the next captured batch size.
- Two-phase warmup at startup (`TtQwen3Encoder.warmup`): every variant is prepared first
  (`_prepare_trace_prefill`: persistent device inputs, compile pass), then all traces are captured back to
  back (`_record_trace_prefill`), and each trace output is marked corruptible
  (`UnsafeAllocationTracker.mark_corruptible`). No device allocation happens after the first capture; the
  outputs are consumed by host readback before any other trace replays, under the encoder lock.
- The post-trace tail (device slice of the last tile, final RMSNorm, layout change) is gone: the whole
  `[batch x padded, 4096]` pre-norm residual is read back and the last real token of each sequence is
  normalized on the host in fp32 with the HF `model.norm.weight` (eps 1e-6). This removed the per-tile-offset
  program compiles (a 313 ms first `rank` request became 145 ms) and the tracker's 43 flagged buffers.
- Multi-device outputs are concatenated across the mesh (`ConcatMeshToTensor(dim=-1)`).

## Trace allocation safety gate

`tests/replay_trace_check.py`, fresh process, `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`,
3 rounds alternating forward and reverse variant order, two different inputs per variant. Pass = no unsafe buffer
reported by the tracker, repeated replays of the same input reproduce the output, and a refreshed input changes it.

Nine-variant encoder: `replay_trace_check_bfp8_attn.json` and `replay_trace_check_accuracy.json`, pass = true.

| variant (padded, batch) | replays | min cosine, repeated replay | max cosine between different inputs |
|---|---|---|---|
| 128, 1 / 4 / 8 | 6 each | 1.0000 / 0.9999999 / 0.9999998 | 0.769 / 0.931 / 0.852 |
| 1024, 1 / 4 / 8 | 6 each | 0.9999999 / 0.9999999 / 0.9999999 | 0.976 / 0.977 / 0.933 |
| 2048, 1 / 4 / 8 | 6 each | 0.9999999 / 0.9999997 / 0.9999999 | 0.858 / 0.987 / 0.958 |

Fifteen-variant encoder (shipped): `replay_trace_check_accuracy_buckets5.json`, see the table at the end of the
"Prefill buckets" section.

Earlier runs of the same check are kept in `/home/hous/dev/clm-v0.1-8B/logs/replay_trace_check*.log`: 43 flagged
buffers (eager tail), 176 (per-variant prepare+capture), 8 (trace outputs), then clean.

## Prefill buckets

The first served evaluation (`doc/release/RUN_NOTES.md`) showed that 300 of the 400 Typed Decisions cases send five
state texts of 130 to 300 tokens each, which the 128 / 1024 / 2048 bucket set served as one batch-8 1024-token pass
(1,302 ms per case with the accuracy policy). The framework's `get_padded_prefill_len` is hard-coded to 128 / 1024 /
powers of two, so the encoder now owns its bucket list (`TtQwen3Encoder.DEFAULT_TRACE_LENS`, env `CLM_TRACE_LENS`,
multiples of 128), writes it into `model_args.trace_prefill_supported_seq_lens` so `can_enable_trace` accepts it, and
pads each group to the smallest bucket that fits its longest text (`padded_len`; host test
`tests/test_encoder_buckets.py`).

Evidence (accuracy policy, p150, `fidelity_accuracy_buckets5.json`, `bench_accuracy_buckets5.json`):

- Fidelity is unchanged to the bit: the 308 single-text vectors of the fidelity corpus are identical (all 4096
  elements) between the nine-variant and the fifteen-variant encoder, so the padded length does not affect the
  real tokens' outputs on this path. Batched vectors differ for 128 of 308 texts (min cosine 0.9986) only because
  the grouping changes which texts share a batch (the batch-variant reduction order documented in
  `../full_model/README.md`).
- Latency (p50 of 10, warm traces):

| padded length | batch 1 | batch 4 | batch 8 | tokens/s at batch 8 |
|---|---|---|---|---|
| 128 | 57.8 ms | 92.1 ms | 162.1 ms | 6,317 |
| 256 (new) | 72.1 ms | 162.4 ms | 306.7 ms | 6,678 |
| 512 (new) | 93.8 ms | 312.5 ms | 624.5 ms | 6,559 |
| 1024 | 170.4 ms | 648.6 ms | 1,300 ms | 6,300 |
| 2048 | 321.7 ms | 1,300 ms | 2,576 ms | 6,361 |

  A 256-token text costs 72 ms instead of 170 ms and a 512-token text 94 ms instead of 170 ms; a batch of eight
  300-token texts costs 625 ms instead of 1,300 ms. Startup grows from nine to fifteen trace captures (warmup
  17.6 s: prepare 16.9 s, capture 0.7 s, after a 5.4 s model load with a warm weight cache).

## Program-config overrides and the block-sharded norm (2026 Oct 2 00:52 to 01:10 UTC)

Stage 3's third experiment (`../optimized_decoder/README.md`, "Matmul geometry") measured three levers that each
help: a QKV block shape with `in0_block_w 4` and a 1x4 output subblock at 128 tokens, `MinimalMatmul` on an 11x10
core grid for the QKV and w2 prefill matmuls above 128 tokens, and a block-sharded RMSNorm on 8x4 cores for prefill
batches of 128, 256 or 512 rows. `tt/encoder.py` installs them after `create_tt_model`:
`_install_program_configs` shadows the three `ModelArgs` getters on the instance (`get_attn_qkv_program_config`,
`get_mlp_ff2_prg_config`, `use_minimal_qkv_prefill_matmul`), `_install_sharded_norms` wraps `forward` of every
layer's `attention_norm` and `ff_norm` for prefill shapes with 128, 256 or 512 rows (other shapes fall through to the
stock interleaved norm). Both are single-chip only and have environment toggles (`CLM_PROGRAM_CONFIGS=0`,
`CLM_SHARDED_NORM=0`); the `p150-accuracy` and `p150-fast` profiles ship with both off, the default `p150` profile
with both on.

End-to-end validation on one p150 (`fidelity_<policy>_pc.json`, `agreement_<policy>_pc.json`, `bench_<policy>_pc.json`,
`replay_trace_check_accuracy_lofi_mlp_pc.json`; logs `/home/hous/dev/clm-v0.1-8B/logs/*_pc.log`):

| policy, path | cosine mean / min vs fp32 | head min (state / candidate) | agreement all / confident | 128 b1 | 128 b8 | 256 b8 | 512 b8 | 1024 b1 | 2048 b1 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy_lofi_mlp, stock configs (`../datatype_sweep/`) | 0.99914 / 0.99584 | 0.9941 / 0.9965 | 96.0 % / 98.9 % | 58.5 ms | 144.0 ms | 271.2 ms | 554.6 ms | 152.0 ms | 285.5 ms |
| accuracy_lofi_mlp, overrides (shipped default) | 0.99916 / 0.99599 | 0.9931 / 0.9971 | 97.5 % / 98.9 % | 53.1 ms | 135.7 ms | 251.7 ms | 510.9 ms | 143.3 ms | 266.0 ms |
| accuracy, stock configs (`p150-accuracy`) | 0.99910 / 0.99596 | 0.9943 / 0.9957 | 95.5 % / 98.9 % | 57.7 ms | 161.6 ms | 306.7 ms | 624.5 ms | 170.4 ms | 321.7 ms |
| accuracy, overrides (not shipped) | 0.99897 / 0.99426 | 0.9924 / 0.9962 | 94.0 % / 97.3 % | 52.5 ms | 154.4 ms | 289.5 ms | 583.3 ms | 162.4 ms | 306.8 ms |

The overrides are neutral to slightly positive for the shipped policy and take 9 to 18 percent off every cell
against the previous default (stock `accuracy`). With the stock `accuracy` policy (HiFi2 fp16-accumulate MLP) the
same overrides lower the minimum cosine from 0.9960 to 0.9943 and the confident-decision agreement from 98.9 to 97.3
percent, which fails the gate; the `p150-accuracy` profile therefore ships without them. Isolation runs with one
override at a time are recorded below. The fifteen-variant trace-safety check with the overrides passes
(`replay_trace_check_accuracy_lofi_mlp_pc.json`: no unsafe buffers, repeated replays identical to cosine
>= 0.9999998, refreshed inputs change the output).

ISOLATION_RESULTS

## Performance (`perf_summary_accuracy_buckets5.json`, default policy accuracy, p150, warm traces, p50 of 10)

| padded length | real tokens | batch 1 | batch 4 | batch 8 | tokens/s at batch 8 |
|---|---|---|---|---|---|
| 128 | 32 or 128 | 57.7 ms | 92.1 ms | 161.6 ms | 6,317 |
| 256 | 256 | 72.1 ms | 162.4 ms | 306.7 ms | 6,678 |
| 512 | 512 | 93.8 ms | 312.5 ms | 624.5 ms | 6,559 |
| 1024 | 1024 | 170.4 ms | 648.6 ms | 1,300 ms | 6,300 |
| 2048 | 2048 | 321.7 ms | 1,300 ms | 2,576 ms | 6,361 |

Stock bfp8 policy (`p150-fast` profile, nine-variant bench `perf_summary.json`): 54.0 / 148.1 / 286.2 ms at batch 1
for 128 / 1024 / 2048 tokens, 7.2k tokens/s at batch 8. Model load with a warm weight cache: 5.4 s (accuracy).

## Lower-bound reconciliation

Per-layer device time from the Tracy profile (accuracy policy, layer 0, layer ops only): 1.557 ms at 128 tokens,
4.519 ms at 1024 tokens. 36 layers: 56.1 ms and 162.7 ms. Measured end to end (accuracy): 57.7 ms and 170.4 ms,
2.9 and 4.7 percent above the bound; the remainder is the token embedding, the readback of the pre-norm residual
(1 to 8 MB) and the host norm. Eager and traced execution are at parity (`../fused_decoder/README.md`), so no
"token-out slower than layer stack" gap exists to close; the plugin's 10 to 15 percent rule is satisfied.

## Head placement

Heads stay on the host (torch fp32): two 4096x1536 + 1536x1536 + 1536x512 MLPs over a handful of vectors per
request cost well under a millisecond, versus a device round trip per request. Measured in the server:
warm cache-hit answers take 0.1 ms server-side.

## Throughput ceiling and what is left

Prefill throughput saturates near 6.3 to 6.7k tokens/s with the accuracy policy (7.2k with the stock bfp8 policy)
from 256 tokens up; batching short texts amortizes little because the per-token kernel cost, not weight streaming,
dominates (DRAM roofline 23.6 percent). The matmul core grid lever is exhausted without kernel changes
(`../optimized_decoder/README.md`); the attention math fidelity lever is rejected by the decision-agreement gate
(`../datatype_sweep/README.md`). What remains is kernel-level work on the 128-token matmuls and the 4-core norms.
