# AutoFix Report: canonical candidate L1 repair

## Starting evidence

- Source diagnosis: `AUTODEBUG.md` in this directory.
- Original failure: traced-decode warmup raised `TT_THROW` in the linear-attention
  MLP down `ttnn.linear`; the L1 allocation floor was `928000` and the static-CB
  region ended at `1333760`.
- Original candidate and command are recorded in
  `evidence/candidates/canonical_bfp8_hifi2_kv_bf16/failure.json`.

## Hypothesis experiment

**Hypothesis.** Keep canonical BFP8 gate/up weights, HiFi2 projection/MLP
fidelities, BF16 CCL, and BF16 KV cache, but retain the frozen baseline's BFP4
linear-attention down weight while leaving full-attention down BFP8.

**Experiment.** Resolve that one-field repair through the recursive candidate
config merge, trace the resulting per-kind/per-role weight and fidelity policy
into `MultichipDecoder.from_state_dict`, and recompute every DRAM-sharded MLP
static-CB endpoint from the selected geometry and factory formulas. No TT
hardware command was run.

| decode MLP row | resolved weight | static-CB end | margin below `928000` |
|---|---|---:|---:|
| linear gate | BFP8 | 744064 | 183936 |
| linear up | BFP8 | 744064 | 183936 |
| linear down | BFP4 | 811520 | 116480 |
| full gate | BFP8 | 776704 | 151296 |
| full up | BFP8 | 776704 | 151296 |
| full down | BFP8 | 778752 | 149248 |

The repaired linear-down arithmetic is:

```text
111360 + 2*1*17*2048 + 3*20*17*576 + 21*2048 = 811520
```

The original BFP8 linear-down arithmetic reproduces the failure exactly:

```text
111360 + 2*1*17*2048 + 3*20*17*1088 + 21*2048 = 1333760
```

**Result.** The resolved policy is gate/up BFP8 for both layer kinds, down
linear=BFP4/full=BFP8, projection and all MLP fidelities HiFi2, all four CCL
roles BF16, KV BF16 with inherited tile-DRAM paged layout/block 64, BF16
activations/residual/norm, and inherited BFP8 LM head with BF16 logits/sampling.
Every MLP CB endpoint is below the observed L1 floor.
The original warmup reached the down call only after executing gate and up, so
the failure location also proves that canonical BFP8+HiFi2 linear gate/up were
admitted with the same BF16 CCL pool live; they are not merely arithmetic fits.

**Verdict: verified source-only.** The candidate removes the diagnosed static-CB
collision. This does not prove end-to-end execution, accuracy, or performance.

**Fix.** Added
`candidates/canonical_runnable_bfp8_hifi2_kv_bf16.json`, inheriting the frozen
baseline and overriding only the canonical policy fields plus the explicit
linear-down BFP4 repair. No implementation code changed.

## Hardware verification

The original full teacher-forcing command was rerun with the repaired candidate:

```bash
TT_METAL_HOME=/home/ttuser/dev/tt-metal \
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal:/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal \
QWEN36_PRECISION_CONFIG=models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/candidates/canonical_runnable_bfp8_hifi2_kv_bf16.json \
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --reference models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt \
  --mesh-device P300 \
  --fabric-config FABRIC_1D_RING \
  --trace-region-size 1500000000 \
  --warmup-repeats 1 \
  --output-json models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/evidence/candidates/canonical_runnable_bfp8_hifi2_kv_bf16/teacher_forcing_metrics.json
```

Result: passed full traced warmup and 100-token AIME24 teacher forcing with
98/100 top-1, 100/100 top-5, 100/100 top-100, 973.63 ms TTFT, and
18.26 t/s/u steady post-capture traced decode. The runtime metrics resolve the
repaired candidate policy and record `decode_trace_enabled=true`.

Evidence:
`evidence/candidates/canonical_runnable_bfp8_hifi2_kv_bf16/teacher_forcing_metrics.json`.

## Final status

- Reported L1/CB failure: fixed by source arithmetic, config resolution, and the
  original full-model traced hardware verification.
- Full runtime and accuracy: passed.
- Selection: rejected because 18.26 t/s/u is materially slower than the selected
  BFP4+LoFi policy (22.61 t/s/u).
