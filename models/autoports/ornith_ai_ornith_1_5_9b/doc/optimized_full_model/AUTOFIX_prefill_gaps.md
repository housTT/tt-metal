# AutoFix: prefill gaps and reusable exact-shape capture

The eager gaps are partly avoidable host submission work. A contained prepared
prefill trace is legal and numerically exact for B1/start0 logical lengths128
and131. The same public128-token prefill/sample window takes **4.872266ms**
uninstrumented versus the recorded profiled **5.263950ms**. A prepared traced
request takes **4.224427ms**, including state reset, token/page refresh, seed
preparation, full embedding/layers/head/sample, and final synchronization.
This is a promising reduced-model candidate, not a full-model TTFT result.

## Starting evidence and gap classification

`AUTODEBUG_prefill_gaps.md` preceded the probe. The144 rank1 device rows contain
3.043997ms kernels and2.071031ms interior gaps, yielding5.115028ms. Narrow native
host function durations sum **0.176745ms**, but their first-to-last timestamp
envelope spans **4.933206ms**. Thus4.756461ms of that host envelope lies outside
the measured function bodies. It includes between-call Python/runtime work and
instrumentation; it cannot be classified as required device waiting.

| Next operation | Device gap us | Between narrow host functions us |
| --- | ---: | ---: |
| Embedding | 139.271 | 148.905 |
| First head matmul | 76.784 | 67.058 |
| ChunkGdnPrep | 70.846 | 62.249 |
| Embedding all-gather | 60.226 | 65.145 |

These correlations support a host-submission component without assigning every
gap to one cause. Host and device clocks are distinct; no absolute-clock
alignment is inferred. Device gaps also include dispatch/fabric scheduling.
`prefill_gap_classification.json` preserves the largest15 rows and raw fields.

The profiled and uninstrumented runs have identical runtime `tt/` source hashes.
Five unchanged public-window measurements are5.090072,4.891221,4.872266,
4.869099,4.830746ms. The profiled window is0.391684ms (8.04%) above that median.
This is an instrumentation-associated/process difference, not an isolated causal
profiler-overhead measurement. It explains part of the original gap but does not
make the remaining gaps mandatory. No uninstrumented device-kernel sum is
fabricated by subtracting numbers from different measurement regimes.

## Focused capture experiment

`probe_prefill_gaps.py` constructs the same real layers0/3, full embeddings,
selected BF16/HiFi4 C64/K1/R2 terminal, common sampling and native262144 cache.
After measuring the original public window, it releases generator decode traces
and tests one prefill trace at a time. Stable IDs/page-table tensors are refreshed
outside capture; DeltaNet state reset and request sampling preparation also stay
outside. The traced body executes the existing embedding, both original decoder
methods, last-logical-row slice/clone, terminal and common device sample.

The existing decoder retains internal padding/trimming for length131. Nothing is
rounded in the public prompt contract. The test alternates original tokens/page
table, changed token IDs with reversed physical page mapping, then original
inputs again. Against the original model prefill path it requires:

- all248320 logical prefill logits bit-identical;
- all TP-rank recurrent/conv state hashes equal;
- sampled token equal;
- complete next-decode logits equal, proving populated KV and hybrid state agree.

Every check passes for both lengths, and changed token IDs produce different
logits. `prefill_gaps_v1.json` retains exact per-case checks and all trial times.

| Logical length | Prepared eager request median ms | Prepared trace request median ms | Eager/trace host body submission ms | Capture-only ms | Trace bytes/bank |
| --- | ---: | ---: | --- | ---: | ---: |
| 128 | 6.799340 | 4.224427 | 5.347934 /0.009548 | 9.230087 | 122880 |
| 131 | 7.241485 | 4.628720 | 5.879220 /0.010811 | 11.080515 | 147456 |

Both prepared modes include about1.03–1.08ms host preparation per request and
use the same body. Their large submission difference verifies removable eager
dispatch work. However, the prepared eager path is slower than the unchanged
public4.872266ms path. Moving preparation earlier and running after diagnostic
reads/capture changes the orchestration being timed. Therefore **do not claim
the2.575ms prepared-body difference as a production TTFT improvement**. The
practical reduced128 comparison is about0.648ms (13.3%) versus the original
public median, and still needs integrated same-harness validation.

All samples remain visible: one128 trace request is7.201118ms; the other four are
4.053791–4.278900ms. No jitter/percentile claim is made from five trials.
Capture-only times exclude persistent-input allocation, warm compilation and
coordinated recapture of generator traces. They are not cold-request costs or
complete amortization estimates. TRACE memory fields are bytes per DRAM bank;
they do not prove all-layer trace memory fit.

## Watcher and lifecycle scope

`prefill_gaps_watcher_v1` passes, exit0, using worker watcher10,
`TT_METAL_WATCHER_DISABLE_ETH=1`, and `TT_METAL_TRACE_ALLOC_TRACKING=1`, with
no device profiler, suppressions or corruptible scopes. Both128 and131 pass the
same original/changed/original prefill-state-next-decode checks. Watcher/tracker
timings are heavily instrumented and excluded from performance conclusions.
No capture failure, hang, reset or recovery occurred; devices closed normally.

This standalone probe intentionally releases decode traces before its prefill
capture. Integrated qualification must preserve all four relevant traces and
their lifetime: model decode, sampler, sampler/history, and prefill. Use stable
prepared inputs and a canonical shared logits destination; prefill should be
captured after earlier traces so a newly retained output cannot overlap them.
The current generator deallocates an owned prefill result, so a borrowed internal
canonical tensor must use a distinct private path; public device-logit callers
still receive owned results. A changed exact shape or new program must release
and rebuild the coordinated trace set before execution.

Keep existing validation, cache identity checks, logical length/start/slot keys,
and eager handling of unsupported/new shapes, mixed batches, continuation and
all-logit requests. A bounded recently used trace can accelerate repeated exact
shapes without weakening any public capability or allocating an unbounded cache.
The parent owns integration and all32-layer, mixed-batch, native-context,
qualitative, repeated-request and final TTFT gates.

## Artifacts and status

`prefill_gaps_summary.json` contains the classification, all candidate trials,
normal and watcher commands, provenance links and limitations. Immutable
`logs/prefill_gaps_v1.*` and `logs/prefill_gaps_watcher_v1.*` preserve exact
environment, source snapshots, binaries and exit status. Host compile and Black
passed for the probe. No C++ build was required; this agent edited stage files
only and did not change model/generator implementation.

Status: profiler-only explanation insufficient; prepared exact-shape capture
hypothesis verified at component scope. Hardware was explicitly returned to the
parent after watcher success. The parent is integrating the contained candidate
and owns the remaining full public-lifecycle and performance proof.
