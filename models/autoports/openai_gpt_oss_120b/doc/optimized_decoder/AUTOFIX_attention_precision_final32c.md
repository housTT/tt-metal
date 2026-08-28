# AutoFix: cumulative attention precision on the final 32-core topology

Date: 2026-08-28

## Starting evidence and hypotheses

Stage review found that the earlier BFP8/LoFi, BFP8 output-activation, and
BFP4 policies omitted the promoted `32c_ibw4_pcn3_sb3` output projection.
Earlier BFP4 evidence also used random activations and the cited BFP4/HiFi2 PCC
run had no retained transcript. Those results could not reject a cumulative
candidate under the production topology.

The isolated predictions were:

1. Attaching the final 32-core output projection could change whole-trace
   latency enough to overturn old precision decisions.
2. Random or synthetic PCC failures would not predict PCC on exact checkpoint
   prompt-derived activations.
3. Output activation dtype is independent of weight dtype and decode projection
   fidelity, so the legal non-DRAM matrix is the full
   `BFP8/BFP4 x HiFi2/LoFi x BF16/BFP8-output` cross-product.
4. A candidate could be selected only if both sliding and full layers passed
   unchanged PCC bars, remained bitwise deterministic after 1000 trace replays,
   and beat the correct baseline in order-reversed measurements.

## Harness and phase contract

`test_optimized_real_activation_batch_two_precision_and_latency_ab` now accepts
one named candidate against omission/automatic selection. It uses exact rows
from checkpoint revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`:

- layer 0 consumes exact embeddings for two fixed 33-token text sequences;
- layer 1 consumes the exact BF16 output of the checkpoint HF layer 0;
- decode positions are `[8, 32]` for sliding and `[26, 16]` for full attention;
- requested and effective policies, actual QKV/output weight dtypes, actual
  decode projection math fidelity, output activation dtype, and the final
  output geometry are asserted and printed;
- first/second trace replay and the output after 1000 timed replays must be
  bitwise equal;
- prefill PCC remains at least 0.95 and batch decode PCC remains at least 0.99.

Fidelity language is phase-qualified. `attention_weight_dtype` controls the
attention projection weights used by prefill and decode. The policy's
`decode_math_fidelity` configures decode QKV and output projections only;
prefill retains `GPTOSSAttentionProgramConfig` and its own compute settings.
`output_activation_dtype` controls only the optional decode concat-heads to
output-projection typecast. SDPA and FullLocal MoE keep their independent
compute configs, and the paged KV cache remains BFP8.

## Final-32-core matrix

Each row below ran both orders against BFP8/HiFi2, both representative layers,
and 1000 trace replays. Decode deltas are candidate versus BFP8; negative is
faster. All candidates passed the unchanged PCC bars and determinism checks.

| Candidate | Sliding prefill/decode PCC | Full prefill/decode PCC | Sliding decode delta, two orders | Full decode delta, two orders | Verdict |
| --- | --- | --- | ---: | ---: | --- |
| BFP8/LoFi, BF16 output | 0.999290613 / 0.997431061 | 0.999312823 / 0.998478216 | +0.381%, +0.451% | +0.371%, +0.455% | Rejected: slower. |
| BFP8/HiFi2, BFP8 output | 0.999290613 / 0.996680031 | 0.999312823 / 0.998440448 | +0.409%, +0.448% | +0.806%, +0.855% | Rejected: slower and lower sliding PCC. |
| BFP8/LoFi, BFP8 output | 0.999290613 / 0.997156000 | 0.999312823 / 0.998542851 | +0.726%, +0.749% | +1.107%, +1.163% | Rejected: slower. |
| BFP4/HiFi2, BF16 output | 0.993718903 / 0.993574877 | 0.994043785 / 0.993987661 | -1.291%, -1.239% | -7.407%, -7.352% | Correct, superseded by LoFi. |
| BFP4/HiFi2, BFP8 output | 0.993718903 / 0.993354090 | 0.994043785 / 0.993926748 | -0.799%, -0.738% | -6.760%, -6.700% | Rejected: slower/lower PCC than BF16 output. |
| BFP4/LoFi, BF16 output | 0.993718903 / 0.993574877 | 0.994043785 / 0.993987661 | -1.561%, -1.480% | -7.695%, -7.606% | Best correct multibatch candidate; promoted. |
| BFP4/LoFi, BFP8 output | 0.993718903 / 0.993354090 | 0.994043785 / 0.993926748 | -1.016%, -0.968% | -7.015%, -6.954% | Rejected: slower/lower PCC than BF16 output. |

Warmed prefill timing was also printed for every row. It was order-sensitive
at this short sequence and did not show a consistent regression: BFP4/LoFi
was `-0.094%/+1.944%` sliding and `-0.462%/-0.265%` full versus BFP8. No
prefill policy uses the decode-only LoFi config.

## Capacity selection

Promoting BFP4/LoFi at every capacity was tested rather than assumed. At
configured batch 1 and sequence 128 it produced:

| Layer | All-capacity BFP4/LoFi | Reproduced BFP8/HiFi2 | Decision |
| --- | ---: | ---: | --- |
| sliding | 0.554860079 ms | 0.535201296 ms | BFP4 regresses 3.67%; retain BFP8. |
| full | 0.542234370 ms | 0.547659984 ms | BFP4 improves 0.99%, but cannot compensate for the sliding regression. |

The final automatic contract therefore resolves omission by configured
capacity: BFP8/HiFi2 for `max_batch_size == 1`, BFP4/LoFi with BF16 output
activation for `max_batch_size > 1`. Explicit BFP8 and BF16 remain exact.
Selection never changes from a smaller logical runtime batch.

After this repair, the direct two-order batch-2 A/B reproduced the selection:

| Layer | Automatic BFP4/LoFi | Explicit BFP8 | BFP8 cost |
| --- | ---: | ---: | ---: |
| sliding | 0.788169--0.788481 ms | 0.800528--0.800549 ms | +1.528--1.571% |
| full | 0.744771--0.745105 ms | 0.806037--0.806344 ms | +8.178--8.267% |

Both policies passed bars and 1000-replay determinism in both orders. The
automatic batch-32 gate also passed both layer kinds with the full
131072-token-per-user context allocation and zero replay differences.

## Watcher and recovery evidence

The heavy `scripts/run_safe_pytest.sh --dev` preset could not compile paged
SDPA: lightweight/LLK assert instrumentation enlarged the program to
80896--82032 bytes, beyond the Blackhole TENSIX kernel-config limit of 70656
bytes. This is a build/instrumentation limit before model execution, not a
correctness failure. A separate watcher-only configuration kept polling
watcher, NoC/CB checks, `TT_METAL_WATCHER_NOINLINE=1`, and disabled watcher
assert/dispatch handling without enabling the extra lightweight/LLK asserts.
That run passed both exact multibatch layer kinds and 1000 replay determinism
with no watcher error. Watcher results were not used as latency evidence.

One post-promotion batch-1 retry encountered a transient SIGBUS before a model
result. The immediate health probe could not map device 0, so one whole-system
`tt-smi -r` recovery was performed. All four devices recovered, the identical
retry passed, and final health listed all four p300c boards as reset-capable.

## Commands and artifacts

All hardware commands used the canonical environment in `work_log.md`, exact
checkpoint revision and cache, `TT_VISIBLE_DEVICES=0,1,2,3`, and the serialized
safe wrapper. The decisive selectors were:

```bash
GPT_OSS_120B_BATCH2_POLICY_AB_REPEATS=1000 \
GPT_OSS_120B_BATCH2_POLICY_AB_ORDER=automatic,bfp8 \
scripts/run_safe_pytest.sh --run-all \
  models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py::test_optimized_real_activation_batch_two_precision_and_latency_ab -q -s

GPT_OSS_120B_BATCH2_POLICY_AB_ORDER=bfp8,automatic # same command

GPT_OSS_120B_OPTIMIZED_PERF=1 \
GPT_OSS_120B_OPTIMIZED_PERF_REPEATS=1000 \
scripts/run_safe_pytest.sh --run-all \
  '...performance[blackhole-1x1-automatic-sliding]' \
  '...performance[blackhole-1x1-automatic-full]' -q -s

GPT_OSS_120B_RUN_BATCH32=1 scripts/run_safe_pytest.sh --run-all \
  models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py::test_optimized_batch_32_paged_prefill_and_traced_decode_capacity -q -s
```

The seven automatic-first and seven candidate-first matrix transcripts are
`evidence/logs/optimized_autofix_attention_precision_final32c_*.log.gz`.
Final capacity-selected artifacts are:

- `optimized_autofix_attention_precision_capacity_auto_batch1_bfp8_retry_final.log.gz`;
- `optimized_autofix_attention_precision_capacity_auto_batch2_automatic_then_bfp8_final.log.gz`;
- `optimized_autofix_attention_precision_capacity_auto_batch2_bfp8_then_automatic_final.log.gz`;
- `optimized_autofix_attention_precision_capacity_auto_batch32_bfp4_lofi_final.log.gz`;
- `optimized_autofix_attention_precision_capacity_auto_integrated_final.log.gz`;
- `optimized_autofix_attention_precision_capacity_auto_batch2_watcher_only_final.log.gz`;
- `optimized_autofix_attention_precision_capacity_auto_batch2_watcher_ringbuffer_final.log.gz`;
- `optimized_autofix_attention_precision_capacity_auto_final_tt_smi.log.gz`.

The rejected all-capacity BFP4 run, heavy watcher diagnostics, SIGBUS/recovery,
and pre/post health transcripts are retained alongside them. No profiler was
run for this group because topology did not change at batch 1 and the matrix
decision used uninstrumented whole-trace wall latency.
