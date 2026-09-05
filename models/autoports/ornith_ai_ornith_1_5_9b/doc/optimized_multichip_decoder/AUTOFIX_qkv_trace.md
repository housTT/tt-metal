# AutoFix: final B32 traced full-attention PCC

## Starting evidence

Fresh source-only report: `AUTODEBUG_qkv_trace.md`.
Original failing artifact: `logs/final_default_v1_watcher_contracts.log`,
unchanged B32 full-attention trace selector, first replay PCC
`0.9942707596653978 < 0.995` on an unreported user.

## Hypothesis experiments

The root agent owns all device work. This investigator must not import TTNN,
open hardware, test on hardware, build, reset devices, or launch other agents.
The diagnostic is instrumented evidence, not a proposed model fix.

| Experiment | First-step minimum HF PCC | Result |
| --- | ---: | --- |
| QKV4, B32, 8/block16/readers2, LoFi | 0.9942707597 (user 31) | fails original value exactly |
| Same original user 31 extracted into B1 | 0.9943646417 | fails; batch-dependent fallback is insufficient |
| QKV4, B32, QKVG-only HiFi2 after prefill | 0.9942707597 | unchanged; HiFi2 is not a fix |
| QKV8, B32, same 8/block16/readers2 | 0.9964473790 (user 31) | all 96 user/step checks pass |
| QKV4, B32, QKVG-only HiFi4 + FP32 destination | 0.9948409006 | improves but still fails |
| QKV4, B32, 32/block4/reader1, LoFi | 0.9945455234 | geometry adaptation still fails |
| QKV4 with only gate output substituted from BFP8 | 0.9966528050 | all 96 user/step checks pass |

Artifacts are respectively `qkv_trace_b4_b32.json`,
`qkv_trace_b4_user31.json`, `qkv_trace_b4_hifi2_b32.json`,
and `qkv_trace_b8_b32.json`, plus matching `.pt` raw outputs and archived
logs/source/provenance. The original pytest selector independently passes with
both QKV8 32/block4/readers2 and matched QKV8 8/block16/readers2
(`qkv8_trace32_control`, `qkv8_c8_trace32_control`).

- **Trace/cache-state corruption: refuted for the exact failing inputs.**
  All four diagnostics have exact restored eager/trace/replay-after-eager
  outputs, exact final K/V cache hashes between execution modes, and unchanged
  saved-state hashes. The QKV4 B32 failure is solely user 31 at position 63;
  every user passes positions 64 and 65.
- **Decode projection sensitivity: verified at this boundary.** QKV4/QKV8
  use identical real inputs and identical prefill K/V hashes on all ranks.
  Cache dtype BFP8, layout, pages, allocation, local heads and other weights
  are fixed. This is a same-cache higher-precision decode control.
- **QKVG HiFi2 repair: refuted.** Its three step minima match LoFi exactly.
- **Geometry/HiFi4/FP32 repairs: refuted for the measured configurations.**
  The programs execute cleanly but still fail the unchanged accuracy gate.
- **Gate precision intervention: sufficient.** BFP8 gate-only substitution
  retains Q/K/V BFP4 and passes all users. Both its initial and final K/V
  payloads are bit-identical to the failing BFP4 run, including all three
  decoded rows. This rescue does not change the cache. It does not prove no
  other component correction could also rescue the combined output.

The final three artifacts are `qkv_trace_b4_hifi4_fp32_b32.json`,
`qkv_trace_b4_c32_r1_b32.json`, and `qkv_trace_gate8_b32.json`.

## Exact diagnostic commands

Run these only under the stage's serialized hardware owner. The output paths
must be new: the diagnostic refuses to overwrite evidence. `D` denotes
`models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder`.

```bash
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_trace_diagnostic --boundaries --output "$D/qkv_trace_b4_b32.json"
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_trace_diagnostic --users 31 --boundaries --output "$D/qkv_trace_b4_user31.json"
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_trace_diagnostic --qkv-dtype bfloat8_b --boundaries --output "$D/qkv_trace_b8_b32.json"
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_trace_diagnostic --qkv-fidelity HiFi2 --boundaries --output "$D/qkv_trace_b4_hifi2_b32.json"
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_trace_diagnostic --qkv-fidelity HiFi4 --qkv-fp32 --output "$D/qkv_trace_b4_hifi4_fp32_b32.json"
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_trace_diagnostic --qkv-cores 32 --qkv-block 4 --qkv-readers 1 --output "$D/qkv_trace_b4_c32_r1_b32.json"
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_qkv_trace_diagnostic --restore-fields gate --output "$D/qkv_trace_gate8_b32.json"
```

The failure cases finish collecting every rank/user/step and persist JSON/raw
outputs before the final assertion exits nonzero. Root provenance additionally
records the watchdog's `timeout 300` wrapper and environment for every run.

## Actual mixed-projection implementation

The diagnostic substitution executes two full packed projections and is not
suitable for timing the mixed precision family. The new test-only
`multichip_qkv_mixed_candidate.MixedQKVG` mixin implements actual QKV4
`[4096,1536]` and gate8 `[4096,1024]` matmuls with one shared 8-core
activation conversion. Default K blocks are 16, with QKV readers2 and gate
reader1. It removes the unused full packed decode copy, keeps packed BFP4
prefill, and includes the necessary output conversion/concat.

Integration without modifying the production model:

```python
from .multichip_qkv_mixed_candidate import MixedQKVG

CANDIDATES["production_qkv4_gate8_split"] = type(
    "ProductionQkv4Gate8Split", (MixedQKVG, ProductionCandidate), {}
)
```

After root-owned registration, measure the complete layer:

```bash
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 3 --length 2048 --prefill-iterations 16 --variant production_qkv4_gate8_split
ORNITH_MULTICHIP_CANDIDATE=production_qkv4_gate8_split python -m pytest -x -q -s 'models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_decoder.py::test_traced_decode_pcc[blackhole-32-full_attention-mesh_device0-device_params0]'
```

Both commands pass (`qkv4_gate8_split_layer3`,
`qkv4_gate8_split_trace32`). The original trace selector checks all 96
user/step outputs and reports aggregate PCC 0.998939 / 0.999059 / 0.999025.
The whole-layer probe passes unchanged output/state gates and exact replay,
including restored replay after intervening eager execution.

## Mixed-family tuning and retained policy

The initial split plus all 13 additional cases in
`mixed_qkv_geometry_matrix.json` finish with return code 0 and pass probe
checks. All execute actual smaller projections sharing the input conversion.
The following values are TP4 complete-layer traced decode milliseconds at
context 2048, B1, on the four physical Blackhole P300c chips.

For common input8 cores and both K blocks16:

| QKV readers | Gate reader1 | Gate readers2 | Gate readers3 |
| --- | ---: | ---: | ---: |
| 1 | 0.276003346 | 0.277514690 | 0.287074221 |
| 2 | **0.275439845** | 0.275918497 | 0.285937342 |
| 3 | 0.276902563 | 0.278658095 | 0.288323157 |

Gate readers3 uses `gate_per_core_n=6`: 24 readers × 2 tiles require 48
output tiles, covered by 8 cores × 6. These are successfully adapted timings.

Additional common-input geometry and gate-block controls:

| Input cores | QKV block/readers | Gate block/readers | Decode ms |
| --- | --- | --- | ---: |
| 4 | 32 / 2 | 32 / 1 | 0.278513155 |
| 16 | 8 / 2 | 8 / 1 | 0.283549030 |
| 32 | 4 / 2 | 4 / 1 | 0.302230968 |
| 8 | 16 / 2 | 8 / 1 | 0.278485877 |
| 8 | 16 / 2 | 8 / 2 | 0.279038813 |

The passing packed QKV8 controls are 0.271286342 ms at
32 cores/block4/reader1 and 0.271270063 ms at 8/block16/reader1. Their
0.000016279 ms difference is treated as a tie; retain the established
32-core interface. The best split is 0.004153502 ms slower than that retained
packed configuration. No split candidate is retained as the production path.

Exact retained-family probe command:

```bash
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --length 2048 --layer 3 --decode-qkvg-dtype bfloat8_b --role-configs '{"qkvg": {"cores": 32, "block_w": 4, "readers": 1}}'
```

The runner archives each matrix command and full source in the matching
`logs/<case>.provenance.json` and `.sources.json.gz`; latency values above
come directly from the `tp4` result records in those logs.

## Final status

**The isolated accuracy failure is resolved by packed decode QKV8.** Reject
the original all-BFP4 packed decode projection at the unchanged 0.995 gate,
after same-input/cache controls, two arithmetic controls, geometry adaptation,
single-user extraction and gate localization. The actual QKV4/gate8 split is
accurate but slower across the measured family, so retain packed QKV8 for
every batch with K8/V8 caches. No trace, cache, or TP partition implementation
fix is justified by the evidence.

The original failing pytest selector passes QKV8 at both tested two-reader
geometries and passes the actual mixed split. The selected one-reader packed
family passes the complete-layer probe. Its promoted-default full watcher,
topology and final performance reruns remain the coordinating agent's stage
gates; this report does not claim those broader reruns have finished.

Investigator checks completed: Python `py_compile`, `git diff --check`, and
all applicable `pre-commit run --files` hooks on both new Python files.
The diagnostic now explicitly checks the original strict aggregate PCC gate
as well as the inclusive per-user gate; every archived aggregate already
passed, so this does not change any reported verdict. Python/docs-only
investigator changes do not need a C++ build.
