# AutoFix: oversized split-bank NoC reads

Starting evidence: [AUTOTRIAGE_watcher.md](AUTOTRIAGE_watcher.md), and the original
`trace_b32_watcher_v1` worker-watcher assertion. The isolated experiment used
real terminal weights, 32768 columns/chunk, two readers/bank, K block 4,
BFP4_B/LoFi and 114688 resident L1 bytes/core on the TP4 P300c mesh.

The split-bank reader passed 36864 bytes to a one-packet specialization whose
Blackhole limit is 16384 bytes. Prediction: the same terminal alone trips the
firmware NoC read-accounting assertion; letting the existing any-length API
packetize each row repairs accounting without changing numerical semantics.

## Experiment and fix

`logs/watcher_terminal_original.log` reproduces the exact assertion in the
split-bank reader and aborts (child signal 6; recording wrapper exit 250).
The source archive includes the original kernel. This verifies the narrow
hypothesis before editing implementation code. Watcher stopped the device and
aborted its host, so no live process remained for tt-triage attachment; saved
watcher and kernel-name logs accompany the immutable command log.

The only implementation change omits `NOC_MAX_BURST_SIZE` from the split-bank
`noc.async_read<NocOptions::TXN_ID>` call, allowing the existing any-length
primitive to packetize and count every request. Transaction IDs, CB pipeline,
reader geometry, precision, and non-split bounded-page path are preserved.
Two explanatory comment lines accompany this one-line change.

`watcher_terminal_fixed_v2.json` and its immutable log/source/provenance archive
pass eager repetition and traced equivalence for rows 1 and 32 under worker
watcher. Both output SHA256 values are **identical** to the original unmodified
non-watcher `terminal_32768_block4_readers2_bfloat4_b_LoFi_l1114688.json`.
Rows 32 top-5/top-100 are 100%; this is a terminal oracle check, not the
full-model accuracy gate. Watcher-run terminal trace latency is 1.0331 ms
(rows 1) and 0.9118 ms (rows 32); use separate uninstrumented full-model timing
for performance claims. The first repaired run passed the same device checks
but hit a probe-only JSON Path serialization error; its log is preserved as
`watcher_terminal_fixed`, and the probe was corrected before the clean rerun.

The exact original terminal and repaired commands are recorded as structured
argv in `logs/watcher_terminal_original.provenance.json` and
`logs/watcher_terminal_fixed_v2.provenance.json`. Both use environment
`TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache`,
`OMP_NUM_THREADS=8`, `HF_HUB_OFFLINE=1`, `TT_METAL_WATCHER=10`, and
`TT_METAL_WATCHER_DISABLE_ETH=1`. ETH watcher remains explicitly outside this
worker-watcher check, consistent with the original failing run.

## Recovery and verification

After the abort, bounded `tt-smi -ls --local`, `tt-smi -r`, another list, and
TP4 ring mesh open/close all returned exit 0. All four chips were visible.
No stale process was killed, no locks were deleted, and no second reset was
needed. Logs have prefix `logs/watcher_terminal_reset`.

Device JIT compiled the changed kernel and successfully executed it under
watcher. `clang-format --dry-run --Werror` and `git diff --check` pass.
The required `.github/scripts/copilot-build.sh --build-ttnn-tests` was attempted
with a 60-second cap and immediately returned exit 1: Docker is unavailable.
`logs/watcher_kernel_copilot_build.log` preserves that environment failure.
The CI-wrapper build is therefore **unverified**, independently of successful
runtime kernel compilation and hardware verification.

The original B32 worker-watcher command was rerun as `trace_b32_watcher_fixed`.
It passed firmware watcher checks across all four devices, then exited 1 at a
**different Python correctness assertion**: changed physical page-table mapping
preserved slots 0–29 but altered two decode tokens in final active slot 30.
That slot produced `[12,13,220,220,220,220,220,220]` initially and
`[12,13,220,12,13,220,220,220]` with permuted pages. Slot 31 was inactive.
Devices closed normally; no reset was needed. Its log/source/provenance and
watcher dump are preserved. This is an unresolved generator/cache/slot gate,
not evidence that the NoC-accounting fix failed; the parent must diagnose it
and rerun the complete B32 contract. No requirement has been waived.

Final status: the isolated native reader bug is **fixed with before/after
hardware evidence**. The full stage remains incomplete until the separate B32
page-table/token disagreement and all remaining gates are resolved.
