# AUTOTRIAGE — native capacity read-completion stall

## Diagnosis

The observed stall is a host buffer-read completion wait after native prefill,
with all four device prefetch queues waiting for additional host commands. The
busy host thread is directly resolved to
`SystemMemoryManager::completion_queue_wait_front` through
`buffer_dispatch::copy_completion_queue_data_into_user_space`. An active
SDPA/CCL kernel deadlock is not supported by either device capture. The exact
cause of the missing completion bytes or queue-pointer disagreement remains
unproven; the evidence does not justify a C++ runtime or fabric fix.

This report uses captured evidence and source only. The reviewing agent ran no
hardware commands and made no implementation changes. The hardware-owning
parent captured device and host state. Scope is the TP4 decoder capacity test,
not full-model execution or serving.

## Triage Evidence

- Reproduction: `timeout 1800 python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_capacity.py -x -q`.
  [Run provenance](logs/final_capacity.provenance.json) records start
  `2026-09-05T06:02:37.088721+00:00`, commit
  `483920f536f64462ba6c3dc785c59b8ff002cdc1`, and the archived source hashes.
  Source files cited below matched those run hashes during this review.
- [Run log](logs/final_capacity.log) reports successful mesh creation and
  reservation of **6,385,827,840 DRAM bytes per device**, plus 24 recurrent L1
  tensors, before the native-context test. Hardware is four Blackhole chips on
  two physical P300c boards; the application mesh is 1x4 with `FABRIC_1D_RING`.
- [First raw capture](triage/capacity.txt) shows idle operation mesh and no
  running model operations. [Second raw capture](triage/capacity_stable.txt)
  reproduces the same dispatch stop sites. Each capture contains 64 callstack
  rows: 32 fabric router RISCs, 12 subordinate compute RISCs, and four each of
  prefetch, dispatch, subordinate dispatch, realtime profiler and profiler push.
  There are no executing model worker kernels in either callstack table.
- On **every device**, `cq_prefetch` is in `fetch_q_get_cmds` at
  `cq_prefetch.cpp:848`; `cq_dispatch` is in `CBReader::acquire_pages` at
  `cq_common.hpp:505–507`; subordinate dispatch waits for pages at
  `cq_dispatch_subordinate.cpp:324–325`. These are upstream-input waits, not
  waits for completion of a particular attention or collective program.
- The parent observed the main thread in `futex_do_wait`, while one helper
  thread, TID 450521, consumed 200 CPU ticks in two seconds. RSS was
  3,564,940 KiB, swap zero, with no compiler child processes. Early cumulative
  child CPU was about 376 seconds, so initial compilation explains some early
  elapsed time but not the later isolated helper spin.
- [Host PC evidence](triage/capacity_host_pc.json) directly identifies that
  helper's stack as `completion_queue_wait_front` →
  `copy_completion_queue_data_into_user_space` →
  `FDMeshCommandQueue::copy_buffer_data_to_user_space` worker →
  `NumaAwareExecutor`. The ELF-address correction is recorded in the artifact;
  symbolization gives functions but no source line numbers. Main-thread evidence
  identifies a futex wait, not its complete Python callstack.
- Read-only Inspector files had matching startup time `06:02:40Z`. Their last
  programs at `06:04:42.966–06:04:42.973Z` were row-major concat, untilize, slice,
  and tilize-with-padding; program 1150 was committed at
  `1788588282973698465 ns`. Recent SDPA programs finished compiling about once
  per second before that. Inspector compilation/commit records are not proof of
  execution completion, but together with idle device stacks and the host stack
  they place the blockage at output readback rather than model construction.
- Raw `check_binary_integrity.py` and `check_noc_status.py` report failures even
  though [the summary](triage/capacity_summary.txt) labels them `pass`. The raw
  evidence takes precedence. The NoC examples show equal nonposted write-sent
  and write-ack counters (for example 484/484), with disagreement against
  software counters read as zero. They do not establish an outstanding ack.
- Raw integrity checks compare prior kernel metadata against L1 contents on
  idle workers. The captures provide no executing worker or invalid source
  write tying these mismatches to the stall. `check_broken_components.py` says
  cores halted by triage no longer appeared halted; fast-dispatch symbol reads
  also failed to halt some cores. These are capture-integrity limitations and
  possible secondary effects, not evidence that those cores caused the earlier
  stall. Ethernet reports link-up, zero retrains and live heartbeat. DRAM
  reports successful training/BIST and zero corrected/uncorrected errors.
- Watcher and LLK asserts were disabled; operation timeout was `0s` in the
  captured runtime configuration. A passing assert section therefore does not
  prove an assertion-enabled execution. A zero operation timeout also explains
  why the native completion reader can spin indefinitely until external action.

## Source Evidence

### Producer/consumer ledger

| Boundary | Producer | Consumer and count contract | Observed state |
| --- | --- | --- | --- |
| Host → prefetch | Host writes a nonzero fetch descriptor | `fetch_q_get_cmds` polls `*prefetch_q_rd_ptr` | All four wait in the source branch explicitly described as nothing to fetch, pending, or available |
| Prefetch → dispatch | Prefetch publishes command pages and upstream semaphore credits | `CBReader::acquire_pages` waits while `upstream_count == local_count` | All four wait for new credits |
| Dispatch → subordinate | Dispatch publishes subordinate pages | `cb_acquire_pages_dispatch_s(1)` requires `num_pages_acquired + 1 <= *sem_addr` | All four wait for a new page |
| Device read → host completion | Read command emits the selected buffer pages into the completion queue | Host descriptor requires `num_pages_read * padded_page_size + sizeof(CQDispatchCmd)` bytes | Host worker is still waiting for bytes |
| Completion worker → main thread | Worker consumes the descriptor and reduces outstanding reads | `finish_nolock` waits until outstanding reads reach zero or an exception arrives | Main thread's futex wait is consistent with this boundary, but its full callstack was not captured |

The first three contracts are in
`tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:838–850`,
`cq_common.hpp:494–507`, and `cq_dispatch_subordinate.cpp:317–325`.
`SystemMemoryManager::completion_queue_wait_front`, in
`tt_metal/impl/dispatch/system_memory_manager.cpp:756`, repeats while the host
completion read pointer **and toggle** equal the device-published write pointer
and toggle. `tt_metal/impl/buffers/dispatch.cpp:1662` loops until the descriptor's
remaining byte count is zero. A busy CPU here is polling, not useful tensor
computation. The missing values are the actual descriptor byte count, remaining
count, physical device/CQ, and both pointer/toggle pairs.

`FDMeshCommandQueue::read_shard_from_device` creates the device transfer and host
descriptor from the same dispatch parameters (`fd_mesh_command_queue.cpp:816`).
The source already pairs those operations; proposing to add a missing read
descriptor without inspecting runtime values would be unsupported.
`copy_buffer_data_to_user_space` submits per-device reader tasks and waits for
them (`:1076`). `finish_nolock` waits on outstanding reads (`:715`).

### Test and tensor geometry

`tests/test_multichip_capacity.py` allocates 23 blocks of 256 MiB and a final
202 MiB DRAM block. Every `[1,8,128,128]` FP32 recurrent reservation is 512 KiB,
so 24 reserve 12 MiB of aggregate L1 payload per device. This is a reservation
test; these tensors do not execute 24 additional layers.

`tests/test_functional_decoder.py:876` constructs a full-attention decoder at
262144 capacity, runs a 262143-token prefill, and first reads the final 64
tokens. Trace/decode occurs only after this read and its sanity checks. The
main Python frame was not captured, but the source order and Inspector's final
untilize/slice/tilize sequence strongly support this first read as the boundary.

Two large data-movement paths matter for focused controls:

1. `MultichipDecoder.prefill_forward` (`tt/multichip_decoder.py:344`) creates
   128 chunks at size 2048, with final logical length 2047. It trims that final
   chunk before `ttnn.concat(outputs, dim=1)`. In
   `ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp:101`, any padding on
   the concat dimension selects untilize → row-major concat → retilize for all
   inputs. Thus the documented 6 GiB input/chunk-output/merged-output estimate
   is not a complete inventory of this fallback's possible intermediates.
2. The tail read starts at `262143 - 64 = 262079`, which is **31 modulo 32**.
   `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:226–237` selects the
   row-major path for a non-tile-aligned start; `:345–349` converts the entire
   input to row-major before slicing. Reading a 64-token tail can therefore
   convert a roughly 2 GiB input. An aligned envelope starting at 262048 and
   ending at 262143 contains 95 logical tokens; selecting its final 64 on the
   host preserves the same tested values while avoiding this full-input
   conversion. This is a concrete unnecessary conversion, not a proven cause
   of the completion mismatch.

The multichip fixture (`tests/test_multichip_decoder.py:102`) reads four local
device tensors separately and checks exact equality for replicated residuals.
Any readback control must retain all four ranks and that equality check.

## Downstream Effects

Dispatch/subordinate waits follow from no new host commands; fabric router poll
sites alone do not establish a blocked fabric route. No payload route, missing
credit, worker CB imbalance, LLK assert, or outstanding NoC write is identified
as the first failure. The earlier "CPU and I/O increased" observation does not
prove forward progress because the resolved completion poll consumes CPU.
Conversely, the initial compilation cost must not be called a hang.

The supplied approximately 17.6-second comparison is not a matched control for
this source, cache state, reservation geometry, and native test. It cannot
establish a performance regression or prove memory exhaustion. Neither the
reservation print nor an idle mesh proves that the capacity gate passed.

## Proposed Fix

No root-cause fix is established. Use the following narrow controls under the
parent's exclusive hardware ownership; preserve native capacity and all gates:

1. Instrument the test with flushed phase markers around construction, upload,
   prefill return, tail slice, each rank read, eager decode and trace replay.
   Add a host traceback timer before entering native work. Use the existing
   supported operation timeout/capture workflow so a native spin leaves bounded
   evidence; do not merely increase pytest's timeout.
2. Keep the reservations and original native prefill. Change only tail extraction
   to the aligned 95-token envelope described above, then compare/check the
   same final 64 tokens on all ranks. This tests the smallest source-visible
   extra conversion before changing decoder execution.
3. If necessary, compare small aligned readback with the same reservations, then
   the original 262143 native case without reservations, and 262144 versus
   262143 under reservations. Keep weight policy, mesh, page table, and warmed
   program cache comparable; record which phase blocks.
4. If all devices again drain while the host reader spins, isolate the local-rank
   read adapter against a supported mesh-composer read of the small tail,
   retaining all four residual replicas. Capture descriptor device/CQ, expected
   and consumed byte counts, and completion read/write pointers and toggles.
   Those values discriminate a missing transfer, wrong queue selection,
   over-read, or pointer publication/visibility error.
5. If the large concat fallback is independently implicated, test keeping the
   final chunk physically padded through concat and trimming only the merged
   result with a tile-aligned start. The current source does not already do
   this. Verify logical length, neutral padding, all-rank output equality,
   continuation and native traced decode before retaining the change. This is
   a candidate within `multichip_decoder.py`, not an excuse to modify baseline
   C++ or weaken the capacity target.

## Uncertainty

- Host stack proves the read-completion function, but not which rank, descriptor,
  remaining byte count or Python read call. Inspector operation records are
  supporting temporal evidence, not a substitute for those runtime values.
- No proof yet connects reservations, unaligned concat, or unaligned tail slicing
  to the missing completion. DRAM capacity failure is not established by this
  capture, and no memory-limit reduction is justified.
- Triage's halt inconsistencies limit low-level integrity conclusions. The
  supervisor should preserve both raw captures and recover devices using the
  repository workflow before controlled reruns.
- This report is diagnosis and a control plan. It does not claim a passing
  capacity test, a validated fix, or an on-device performance improvement.
