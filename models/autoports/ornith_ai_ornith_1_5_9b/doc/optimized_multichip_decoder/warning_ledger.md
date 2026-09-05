# Runtime warning and anomaly ledger

## Allocations while a trace is active

Observed: allocator.cpp warns that buffers allocated while a trace exists can
be corrupted when that trace replays. This is a conditional runtime warning,
not an observed corruption report. The probe deliberately exercises that risk:
it captures T, saves its result, restores state, executes an intervening eager
forward E, restores again, and executes T2. `trace_after_eager_exact` compares
all output ranks exactly. State snapshots and restored recurrent/KV state are
also checked outside timing. Examples are the `packed_promoted_probe_layer0`
and `packed_promoted_probe_layer3` log/provenance pairs. The final watcher suite
additionally covers poisoned free pools, changed inputs/page tables, repeated
restored traces, and refreshed prefill traces with immutable input checks.

Affected path: test-owned trace/eager coexistence and state restoration.
Resolution: controlled by explicit output/state and input-immutability tests;
no warning/assert is suppressed. Final watcher evidence is recorded in README.
The decoder does not promise arbitrary caller-owned allocations can overlap
captured transient storage. Full-model trace ownership must preserve the tested
persistent input/state/output lifetimes.

## BFP8 collective packet sizing

Observed: CCL candidates using BFP8 pages warn that8192-byte fabric packets are
suboptimal for1088-byte pages and recommend4352 bytes. This is throughput
advice. `packedfinal_ccl8_packet4352_*` explicitly compares that packet size for
both kinds and replicated/retained-sharded residuals with the matched8192-byte
`packedfinal_ccl_bfp8_*` runs. All four4352-byte runs pass, with decode0.364267/0.472058ms linear replicated/sharded and0.279579/0.395405ms full replicated/sharded. They remain slower than the selected BF16 native replicated defaults (approximately0.3557/0.2687ms). The packet adaptation does not make the BFP8 family the winner; exact before/after values are in the candidate CSV.
The final BF16-payload path uses8192-byte packets; no BFP8 collective is silently
substituted into it.

## Topology discovery metadata

Observed: the B850M-C motherboard is not in the UMD tray-name lookup, so UMD
uses PCI bus IDs as tray IDs. The actual PCI devices and four-chip ring are
recorded by discovery, mesh-open logs and native TP4 weight partitioning.
`resumed_mesh_smoke` opens/closes that ring. This metadata fallback is not a
replicated model or single-chip execution fallback.

The separate one-chip numerical control can warn that opening a subset of MMIO
devices slows remote read/write. It contributes no TP4 timing claims. Each
measured `tp4` segment opens all four chips after the control mesh is closed.

## Interrupted runner

The runner container was recreated before
`packedfinal_persistent_replicated_layer0` emitted output. No workload process
survived and its exec session disappeared. `interrupted_runs.json` preserves
this classification and replacement run name. Device listing and
`resumed_mesh_smoke` passed before resuming. This run has no performance or
correctness result; completed earlier archives remain immutable.

## Diagnosed numerical and implementation anomalies

The QKV4 real-input failure, cache4 precision failures, mixed-dtype Z-fusion
failure and historical non-reproducing GDN observation, mesh-reader descriptor
failure, CCL semaphore coverage failure, and SDPA L1 allocation failure each
have dedicated `AUTODEBUG_*`/`AUTOFIX_*` reports linked from README. Historical
failed candidates are evidence, never passing final gates. No tolerance was
lowered and no failure was marked as an expected passing test.

## Profiler core counts and modeled utilization

The final raw profiles report a `CORE COUNT` of110 for multi-reader
MLP matmuls and80 for QKVG. These are not the32/8 input-storage core counts
in the model configuration. The human-readable tool table can instead display
eight cores for DRAM-sharded matmul and a modeled compute percentage above100%
for packed gate/up. Its DRAM-matmul core inference does not represent the
explicit three-reader-per-bank compute topology; it is not a measurement of
physical compute saturation. This is a candidate `tt-perf-report` improvement:
account for the configured reader workers and separate participant/storage/
compute core roles. Actual kernel durations and raw metadata are retained;
the report's nominal dtype-byte throughput also differs from BFP storage bytes
including exponent headers. `reader_packed_device_accounting` therefore labels
its independently calculated weight-storage rate and512GB/s model fraction.
No headline speedup or correctness result depends on those modeled percentages.
