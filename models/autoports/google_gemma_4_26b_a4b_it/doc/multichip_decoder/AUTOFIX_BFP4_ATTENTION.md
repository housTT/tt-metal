# AutoFix: real-weight BFP4 attention projections

## Starting evidence

Stage review identified that the selected TP4 decoder had no real-weight BFP4
trial for either material attention projection. The selected sliding layer uses
BF16 QKV/O weights, while the selected full-attention layer uses BFP8 QKV/O
weights. The source already accepts independent decode-only QKV and O dtype
environment roles, but full attention intentionally bypasses its decode QKV
DRAM copy.

All fork slots were occupied, so the AutoFix hypothesis loop was run serially.
No stale AutoDebug report described this precision question; the focused source
audit above supplied the starting hypothesis.

## Experiment contract

`test_tp4_real_weight_attention_decode_precision_probe` is an explicit opt-in
probe with four process-isolated modes: selected, QKV DRAM control, QKV BFP4,
and O BFP4. Every case uses:

- one P300C QB2 as the four-chip P150x4 proxy, TP4 1D ring, and the selected
  graph/CCL/MoE policy;
- real Gemma layer-0 sliding-attention or layer-5 full-attention weights;
- S=32 real-weight prefill into BF16 paged local KV cache, followed by
  cache-consuming decode at position 32;
- the separately captured optimized single-chip TTNN output as the PCC oracle,
  with threshold 0.995;
- the selected attention fidelity (HiFi4 sliding, LoFi full) and explicit SDPA
  program (8x4 grid, Q chunk 32, K chunk 64, exact exponential);
- five trace warmups and 30 blocking replays under throwing runtime fallback;
- exact checks for eager/trace agreement, repeated trace output, replicated TP
  output, populated cache shards, preserved prefill weights, preserved other
  attention role, and the dtype of the weight observed at each actual matmul.

QKV control/BFP4 modes add QKV to the same decode DRAM-sharded family used by
the selected O path. For full attention only, the test instance overrides the
selected `_use_decode_dram_weight` bypass. This is deliberately test-local: it
makes the requested QKV tensor reach the actual full-attention decode matmul
without keeping an unproven production escape hatch. The temporary equivalent
source change was reverted before the final artifact reruns.

## Hypothesis experiments

### QKV BFP4

Hypothesis: a decode-only DRAM-width-sharded BFP4 QKV copy can replace the
selected QKV policy without changing prefill, O, cache, SDPA, or fidelity.

| Layer kind | Selected PCC / ms | dtype-matched QKV-DRAM PCC / ms | QKV BFP4 PCC / ms | Verdict |
| --- | ---: | ---: | ---: | --- |
| sliding | 0.999319 / 0.651875 | 0.994189 / 0.642380 | 0.984498 / 0.640203 | rejected: control topology already misses; BFP4 loses another 0.009691 PCC |
| full | 0.999875 / 0.952709 | 0.999862 / 0.920498 | 0.978292 / 0.909841 | rejected: BFP4 loses 0.021571 PCC versus its topology control |

The QKV BFP4 weight is recorded as `BFLOAT4_B`, DRAM width-sharded, and
`device_typecast_retained` on both layers. Prefill stays BF16 sliding/BFP8 full,
and O stays BF16 sliding/BFP8 full. Both candidates remain finite, cache-using,
bit-exact across eager/trace/replay/replicas, and faster nominally, but their
model-visible decode PCC failures are decisive. The lower precision is not
selected. The QKV production bypass remains unchanged.

### O BFP4

Hypothesis: a decode-only DRAM-width-sharded BFP4 O copy can replace the
selected O policy without changing prefill, QKV, cache, SDPA, or fidelity.

| Layer kind | Selected PCC / ms | O BFP4 PCC / ms | nominal speedup | Verdict |
| --- | ---: | ---: | ---: | --- |
| sliding | 0.999319 / 0.651875 | 0.991448 / 0.650775 | 1.00169x | rejected: PCC below 0.995 |
| full | 0.999875 / 0.952709 | 0.998744 / 0.948993 | 1.00392x | passing screen, not selected |

The full-only result clears PCC, but its 0.39% host-timed difference over only
30 replays is below a robust policy-selection signal and requires an additional
retained decode weight (`device_typecast_retained`). Because the same O policy
fails the other meaningful layer kind, no global policy is possible; the tiny
full-only nominal change does not justify a new layer exception. Selected O
precision remains BF16 sliding and BFP8 full.

## Commands and artifacts

The common command tail was:

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
timeout 1800 python_env/bin/python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k test_tp4_real_weight_attention_decode_precision_probe \
  --junitxml=<artifact>
```

It was run separately with
`GEMMA4_MULTICHIP_ATTENTION_PROBE=selected`, `qkv_control`, `qkv_bfp4`, and
`o_proj_bfp4`. Exact expanded commands are in
`artifacts/bfp4_attention_summary.json`.

- Selected: `artifacts/bfp4_attention_selected.xml`, 2/2 pass.
- Expected rejected-control evidence:
  `artifacts/bfp4_attention_qkv_control.xml`, 1/2 fail.
- Expected rejected-candidate evidence:
  `artifacts/bfp4_attention_qkv_bfp4.xml`, 2/2 fail, and
  `artifacts/bfp4_attention_o_proj_bfp4.xml`, 1/2 fail.
- Each JUnit has two adjacent per-layer JSON files. They contain runtime weight
  call records, PCC, cache geometry, trace time/determinism, SDPA/fidelity,
  fallback policy, and exact source/test/reference hashes.

Final tested hashes are source
`7279e13a379bd9260006f2ab220a1fa7b35d01ccec1ccc79dddd17bab058f3bf` and
test `72adec376c8865825f7ce120ce8299e42865dbeb1b82cfa5afb1eca9141bcdf9`.

## Device health

Hardware commands were serialized and devices were closed between processes.
Bounded pre-runs found all four P300C devices. Final `timeout 60 tt-smi -s` at
2026-09-05T20:13:34Z reported DRAM healthy on every device, zero corrected and
uncorrected GDDR errors, maximum GDDR temperature 44 C, and maximum ASIC
temperature 39.2 C. No reset, kill, triage, or recovery was required.

## Final status and uncertainty

The missing trial is resolved with real-weight, role-reached, cache-consuming,
traced evidence. Both BFP4 policies are rejected and the selected attention
policy is unchanged. This is one representative real-weight layer and input
seed per attention kind at batch 1 and position 32. It is strong enough to
reject QKV and sliding O at the decoder PCC gate, but it is not a dataset or
full-model accuracy study. The full O result could be re-examined only if a
future measurement shows a material device-time win; this run did not collect a
new profiler window because its whole-layer host delta was under 0.4%.
