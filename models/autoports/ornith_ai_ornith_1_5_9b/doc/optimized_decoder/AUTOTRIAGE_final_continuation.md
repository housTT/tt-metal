# AUTOTRIAGE

## Diagnosis

The final-default short suite stops on a **host binary-cache wait after an earlier 67-input concat ELF-load failure**, with no active compute operation on the selected Blackhole device. The inherited public prefill implementation retains one optimized decode output per unaligned leading token and submits all those outputs to one concat. Two concrete resource contracts are violated: the 67-input row-major concat reader exceeds NCRISC local storage, and a longer unaligned prefix retains enough optimized L1 output shards to collide with static kernel circular buffers. The source contains a separate exception-safety defect in the binary cache that converts a second load of the rejected ELF into a permanent host wait.

This diagnosis is based on the live capture and current source. No implementation was changed during this pass. The exact host call stack could not be captured; the binary-cache stop-site is a strongly supported source inference, separated below from directly observed facts.

## Triage Evidence

- Workload command: `python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder/record_run.py final_default_short_v1 python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_decoder.py -m 'not long' -v -s`.
- Parent recorder PID 107741, pytest PID 107747. The test stopped after device setup at 2026-09-05 02:18:40 UTC in `test_prefill_continuation[blackhole-63-full_attention-mesh_device0-device_params0]`.
- Capture: `timeout 180 tools/tt-triage.py --llm-output --llm-output-path models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder/triage/final_continuation_v1/tt-triage.txt --triage-summary-path models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder/triage/final_continuation_v1/triage-summary.txt`; exit 0. Console saved in the same directory as `console.log`.
- [Live report](triage/final_continuation_v1/tt-triage.txt): `dump_op_mesh` shows device 1 idle; `dump_running_operations` lists no operation. The only active stacks are dispatch/service kernels. Prefetch is at `fetch_q_get_cmds`, dispatch is acquiring pages, and subordinate dispatch is waiting for its page stream. There is no stalled model producer/consumer or active SDPA kernel in this capture.
- [Host thread evidence](triage/final_continuation_v1/host_threads.txt): main Python thread and many workers sleep in `futex_do_wait`. No `gdb`, `gdb-multiarch`, `lldb`, or `py-spy` executable was available in the inspected tool directories. `/proc/107747/stack` returned permission denied. No signal injection was performed.
- Preserved [program log](triage/final_continuation_v1/programs_log.yaml), [kernel log](triage/final_continuation_v1/kernels.yaml), and [mesh workload log](triage/final_continuation_v1/mesh_workloads_log.yaml.gz). The final workload 3340/program 6736 is created and compilation begins at 02:18:43.787 UTC. Its `writer_unary_stick_layout_interleaved_start_id` finishes compilation; no reader completion or program completion follows. Program 6734, an untilize operation, completed immediately before it. The missing concat-reader completion is consistent with a wait inside reader binary loading.
- Earlier at 02:17:38.347, `test_unaligned_continuation_to_capacity` failed loading `reader_concat_stick_layout_interleaved_start_id/16813403863354097576/ncrisc/ncrisc.elf`: `segment[1] [0xffb00c20,+0x1470) overflows region:1 limit of 0x12e0 bytes, reduce the size of thread_local variables`. The rejected segment is 5232 bytes versus a 4832-byte limit.
- Earlier at 02:18:34.788, full-attention `test_no_host_fallback_in_forward` failed: L1 allocation at 636928 overlaps a static CB region ending at 641664, program 5282, core range `[0-0 - 7-7]`. This test explicitly runs prefill 256, decode position 256, then prefill continuation starting at 257.
- Binary-integrity mismatches refer to previously used slice/unpad kernels on idle workers. The capture does **not** establish executable corruption during a running model kernel. Ethernet counter mismatches and temporary halt warnings likewise do not identify the first stuck operation. Do not use the summary's script `pass` statuses as a workload-health verdict; individual reports contain findings.

## Source Evidence

### Public continuation ownership and concat fan-in

[FunctionalDecoder.prefill_forward](../../tt/functional_decoder.py) computes `leading = min(seq_len, (-start_pos) % 128)` for full attention, executes each leading token through `self._block(..., mode="decode")`, retains every returned tensor in `outputs`, then retains each aligned prefill block and finally calls `ttnn.concat(outputs, dim=1)` (lines 926–956 in the inspected snapshot).

[OptimizedDecoder._block](../../tt/optimized_decoder.py) chooses width-sharded L1 residual memory for single-user decode (32 cores by default) and returns its final residual sum in that memory; there is no spill before public prefill retains it. Aligned prefill uses DRAM residual memory. The proposed output spill and bounded concat are absent from this snapshot.

| Case | Leading decode outputs | Aligned prefill outputs | Final concat inputs | Retained decode L1 payload per residual core |
| --- | ---: | ---: | ---: | ---: |
| Capacity: total 4096, split 63, chunk 2048 | 65 | 2 (2048 + 1920 tokens) | 67 | 520 KiB |
| Short continuation: total 384, split 63, chunk 128 | 65 | 2 (128 + 128 tokens) | 67 | 520 KiB |
| No-host guard: 256-token continuation at 257, chunk 2048 | 127 | 1 (129 logical tokens) | 128 | 1016 KiB |

Each retained single-user BF16 output physically stores a 32-row tile by a 128-element shard per core: `32 * 128 * 2 = 8192` bytes. This accumulation is additional to transient activation buffers and static CBs. The full no-host guard's collision is therefore explained by output lifetime/resource ownership, not by an actual forbidden host call. The two intentional host-guard exceptions in that test are positive controls.

### Row-major concat reader ledger

[Concat wrapper](../../../../../ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp) selects untilize → row-major concat → tilize when the concat dimension contains tile padding; a one-token output has such padding. [Concat factory](../../../../../ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp) puts input count, each page size, and tensor accessor descriptors into reader compile-time arguments (lines 200–204). Input page counts and addresses are runtime arguments (lines 271–278). Consequently both 67-input continuations have the same reader structure/page widths and can reuse the same reader binary even though their token counts differ.

[Reader kernel](../../../../../ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp) instantiates `num_tensors`-sized page-count/page-id arrays and an accessor/wrapper tuple. For each assigned page, it reserves CB0, reads one 8192-byte BF16 row from the current source tensor, waits for the read, and pushes one CB0 page. The writer consumes and writes one page per assigned page. These page counts are balanced in source; **the first failure occurs before any of that protocol runs**, because local storage grows with all 67 inputs and the loader rejects the binary. There is no evidence of a CB-count deadlock.

### Binary-loader exception state

[DataMovementKernel.read_binaries](../../../../../tt_metal/impl/kernels/kernel.cpp) calls `llrt::get_risc_binary` during program compilation before the inspector records a kernel compile completion. [get_risc_binary](../../../../../tt_metal/llrt/llrt.cpp), lines 49–70:

1. Inserts a null entry keyed by ELF path while holding a mutex.
2. Unlocks and constructs `ll_api::memory(path, loading)`.
3. Only on successful construction fills the cache slot and notifies waiters.
4. A later caller finding the null slot waits until it becomes non-null.

There is no catch/cleanup or failed state for an exception at step 2. [ELF region validation](../../../../../tt_metal/llrt/tt_elffile.cpp), lines 391–405, throws the exact observed segment-overflow exception. Once the first capacity test catches that exception at pytest level, the process-static binary-cache entry remains null. The second 67-input concat then reaches a condition-variable wait with no producer capable of filling that slot. Its sibling writer compile finishes; the parent compile waits for the reader future; dispatch drains and becomes idle. This explains the observed ordered failure-to-hang transition, host futex waits, missing reader-completion record, and idle device together.

The nearby JIT build cache already documents cleanup when build callbacks throw. That does not repair this distinct lower-level ELF-loading cache; the missing transition is in `get_risc_binary` itself.

## Downstream Effects

- The current dispatch/service waits are downstream of a host compile/binary-load wait, not the initiating fault.
- The concat local-storage error is the first observed rejected operation for this continuation family.
- The later L1 collision is a separate resource-lifetime violation exposed by a 127-token unaligned prefix and optimized L1 outputs.
- The live binary mismatch/ETH observations do not justify blaming XIP, fabric, hardware corruption, cache addressing, or SDPA numerics.
- Resetting silicon alone cannot repair the poisoned process-static binary cache. Terminating the owning pytest process is required before retrying.

## Proposed Fix

Within the authorized optimized-runtime/tests/docs scope:

1. Keep inherited cache/position semantics, but move each unaligned leading decode output to DRAM immediately, releasing its L1 allocation before producing the next token. Preserve ordinary decode's optimized sharded return path.
2. Bound public prefill concat fan-in (for example, merge small groups/tree levels with explicit DRAM output) so one row-major kernel never instantiates dozens or hundreds of input accessors. Free consumed intermediate tensors as each merge completes.
3. Verify these independently through the autofix loop: first isolate short split-63 continuation in a fresh process to confirm the ELF error replaces the poisoned-cache hang; prove bounded concat fixes that failure. Separately isolate the no-host guard's continuation and prove immediate output spill removes the L1 collision. Then rerun capacity continuation and the original short suite. Keep all PCC and no-host guards intact.

Outside this stage's authorized edit scope, `get_risc_binary` needs an explicit success/failure state (or exception-safe entry cleanup with correct waiter ownership/notification). A failure must wake waiters and allow deterministic retry/rethrow rather than leaving a permanent null slot. Do not change tt-metal C++ as part of this optimized-model repair without expanding scope and satisfying its build requirement.

Bounded recovery handed to the main agent after capture: stop task-owned pytest PID 107747 and let recorder PID 107741 record its exit, killing the recorder only if necessary; confirm no task-owned hardware process remains; run serialized `timeout 60 tt-smi -ls --local`, `timeout 180 tt-smi -r`, `timeout 60 tt-smi -ls --local`, then a one-device mesh open/close smoke. Repeat reset only if enumeration is incomplete, following the device-usage skill. This diagnosis agent performed no kill, reset, lock clearing, or new device workload. Main owns all subsequent recovery and measurements.

## Uncertainty

- No native host backtrace was available. The precise `get_risc_binary` frame is inferred from the exception path, cache state machine, matching 67-input geometry, futex waits, and incomplete reader compilation. A fresh-process reproduction should produce the explicit ELF rejection and strengthens this attribution.
- The final incomplete reader hash is not recorded by the inspector because registration occurs after successful binary loading. The observed prior hash is explicit; its reuse is inferred from the factory's compile-time/runtime split.
- Exact allocation inventory at the no-host collision was not captured. The unbounded L1-output lifetime is directly present and predicts the pressure; verify a focused DRAM-spill A/B before keeping a fix.
- No implementation fix or passing retest is claimed. The main agent is running the autofix verification/repair loop from this report.
