# Multichip decoder and host-cache stage review

Date: 2026-08-27

Reviewer: fresh xhigh `$stage-review` subagent, read-only and hardware-free

Repository: `tt-metal`, branch `hous/qwen3.8-flash-next`, live worktree based
on `dc7afb94eb72917301b33648e60483dd19a481d1`

Verdict: **clean-pass**

## Required work

None. The reviewer found no P1/P2 blocker against the multichip, exact bounded
host-weight-cache/PLE, correctness, context, trace, stress, watcher, or
performance contract.

## Independently re-derived evidence

- The nine accepted contract/correctness/performance/watcher XMLs contain 91
  tests, zero failures, zero errors, and zero skips. The three profiler pytest
  XMLs also pass.
- Real-checkpoint prefill/first-decode PCC is
  `0.99942303/0.99996978` for layer 0,
  `0.99949104/0.99984211` for layer 1, and
  `0.99972457/0.99988294` for layer 3.
- Seven-sample host EP2 median prefill/decode latency is
  `3583.877886/3.766486 ms` for layer 0,
  `1541.191421/4.184993 ms` for layer 1, and
  `2064.964676/3.050174 ms` for layer 3.
- The context remains 262,144 tokens. The full-stack plan is
  10,103,303,168 bytes/device against 34,225,520,640 bytes/device, leaving
  24,122,217,472 bytes/device of planned headroom.
- `evidence_manifest.sha256` verified from the repository root.

## Other concerns

- Older diagnostic XMLs and profiler provenance coexist in the evidence
  directory. They are not acceptance evidence. The final checkpoint is
  intentionally limited to the manifest-delimited EP2 evidence plus diagnostic
  artifacts directly referenced by the retained AutoDebug/AutoFix reports.
- Performance is correctly reported but warm-state-sensitive. Layer-0 samples
  vary materially, so the medians must not be interpreted as cold-start or
  independent-process service latency.
- Host-backed decode is batch one. The resident optimized multichip path keeps
  its batch-32 contract; later full-model/vLLM work must account for this
  serving limitation.

## Hard-check gaps

None material. A literal 262,144-token public prefill was not rerun as 128
separate 2,048-token chunks. Instead, the stage proves capacity and chunk plans
through 262,143/262,144 and executes the full-width QSA cache/update/trace path
at position 262,143. This is disclosed in `context_contract.json` and is
consistent with the decoder-stage scope.

## Anomaly ledger

### Older profiler provenance

- Observed anomaly: `profiler_provenance.txt` names pre-EP2 artifacts.
- Evidence: the file references older `tracy_host_layer*` captures, while the
  accepted README and manifest use `tracy_host_ep2_layer{0,1,3}.xml` and the
  `17_*` EP2 reports.
- Affected path: historical profiler documentation only.
- Control: every current manifest entry verifies.
- Resolution: controlled; stale provenance is excluded from the checkpoint.

### Host Tracy sampling permissions

- Observed anomaly: raw Tracy logs contain `perf_event_paranoid: 4`, host
  sampling setup failure, and SysTraceWorker-priority messages.
- Affected path: host Linux sampling metadata, not TT op capture.
- Control: all three profiler tests and `tt-perf-report` postprocessors pass,
  reports merge both devices, and no accepted run reports profiler DRAM-buffer
  overflow.
- Resolution: controlled.

### ETH watcher instrumentation disabled

- Observed anomaly: the final watcher command sets
  `TT_METAL_WATCHER_DISABLE_ETH=1`.
- Affected path: ETH watcher instrumentation during fabric teardown.
- Control: the separate watcher suite has four passes; its 556-line archived
  log hashes to
  `50e398a486e7b8290e31e8837284c1e2bcf7b2d4127e855c6eeb76be5b101830`
  and has no error/assert/panic/hang/timeout signature.
- Resolution: controlled; Tensix, dispatch, NoC/CB, stack, and waypoint checks
  remain enabled, and profiler and watcher were kept separate.

## Scope inspected

- Complete `$stage-review`, `$multichip`, `$host-weight-cache`,
  `$tt-device-usage`, and `$autofix` skills, plus `tech_reports/LLMs/llms.md`
  section 3.3.
- README, mesh plan, work log, context/host contracts, AutoDebug/AutoFix and
  capacity-audit reports, accepted XMLs, watcher log, EP2 Tracy output, and
  `tt-perf-report` tables.
- `tt/host_weight_cache.py`, `tt/multichip_decoder.py`, and the three stage test
  files.
- Read-only `sed`, `rg`, `find`, `jq`, Git inspection, XML/perf parsers, and
  `sha256sum -c`. The reviewer modified no file and opened no TT device.

## Residual risk

- Full 48-layer end-to-end generation and vLLM remain intentionally outside
  this stage.
- PLE trailing-padding/multi-request history beyond the delivered batch-one
  host-backed path is not separately proven.
- Cold mmap and BFP4 packing latency can exceed the warmed medians; the README
  already discloses the high demand-loaded prefill cost.
