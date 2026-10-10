# AUTOTRIAGE

## Diagnosis

The 256-way `minimal_matmul_split(..., fuse_swiglu=True)` exceeds the Blackhole NCRISC local stack budget while materializing its output-accessor tuple. The compiled `kernel_main` frame alone is 5,168 bytes, larger than the maximum 5,056 bytes between the kernel's globals and local-memory top. Caller frames reduce that budget further. Stack stores overlap the kernel's NoC transaction counters before the first activation block is published. This explains the first-read-barrier hang and the complete downstream activation starvation. It is a native split-count resource-contract bug, not a gate/up orientation or numerical-precision failure.

Investigation used the AutoTriage skill, captured text, source, and offline inspection of existing ELF files only. The parent agent owns device capture/recovery. No implementation or device state was changed. No `AUTOTRIAGE_INPUT.md` was supplied; the parent problem statement and this directory's capture were the inputs.

## Triage Evidence

- [tt-triage.txt](tt-triage.txt), SHA256 `35a0374c38b1b29ea36a5bbde85378c1e3e6d1e888a9210e4e5a18061f51103f`, identifies op **100, MinimalMatmulDeviceOperation**, on device 3, 110 worker cores. Previous op 99 is router-output tilization. Ops 101–107 are queued concatenations; op 109 is the inherited sparse down projection.
- Actual stuck input is BF16 `[1,1,128,2048]` and BFP4 weight `[2048,262144]`. Thus the 129-token request hangs in its first 128-token planner chunk, before the one-token tail.
- Ten activation injectors, logical `(0,y)` for `y=0..9`, run `dm_in0_sender_metal2` kernel ID 217. Their stacks stop in `read_in0_block_sync<8,8>` at `matmul_dataflow_common_metal2.hpp:118`, through `noc_async_read_barrier`. Example physical `(1,2)` PC is `0x17050`.
- The other 100 NCRISCs, kernel ID 218, wait on `in0_receiver_semaphore.wait(VALID)` at `dm_in0_sender_metal2.cpp:361`.
- All 110 TRISC0s wait for activation tiles at `compute_metal2.cpp:625`. All 110 BRISCs are reserving more weight-buffer space at `dm_in1_sender_out_metal2.cpp:305`. TRISC1/2 stop inside the corresponding math/pack work, waiting for the unpack progression.
- Every core has identical CB counts: CB0 activation `pushed=0,popped=0`; CB1 weights `pushed=128,popped=0`; CB2 output and CB3 intermediate both `0,0`. No output block exists yet.
- Binary integrity, core magic, DRAM health, and lightweight-assert checks report pass. Watcher, LLK asserts, and lightweight kernel asserts were disabled. Ethernet NoC mismatch text is not evidence of an earlier Ethernet cause: this operation is single-device, and its worker activation producer already explains the wait chain. `summary.txt` records successful execution of each diagnostic, not proof that every nested diagnostic finding is healthy.

## Source Evidence

Source root is the repository at HEAD `53274a46856a80e3872967fd0e79d2d9758f48a7`. The live adapter was rechecked during investigation; its split branch supplies `chunks=256` and then concatenates expert outputs. The main source continued changing outside this investigation.

### Producer/consumer and route ledger

| Resource | Producer and count | Consumer and count | Captured state |
|---|---|---|---|
| CB0, 128 BF16 pages | NCRISC reserves/publishes 64 pages per K block | Compute waits/pops 64 pages; final K block may be reused at the next N block | No first publish on any core |
| CB1, 128 BFP4 pages | BRISC reserves/publishes 64 pages per K block | Compute waits/pops 64 pages after activation arrives | Two blocks published, third reserve blocked |
| CB2, 64 BF16 pages | Compute publishes 32 pages per SwiGLU output block | NCRISC output writer drains them | No output produced |
| CB3, 64 FP32 pages | Compute accumulation | Compute final-K SwiGLU | No completed block |
| Activation receiver semaphore 1 | Predecessor relays VALID after first DRAM read/publish | Each noninjector clears to 0, requests predecessor, waits for 1 | 0 on all 110 cores |
| Activation request semaphore 0 | Each successor increments predecessor once per activation block | Predecessor waits for 1 before forwarding | 100 values are 1, 10 sink values are 0 |
| Activation valid semaphore 2 | Each NCRISC initializes persistent source value 1 | Relay copies it to successor's semaphore 1 | 1 on all 110 cores |

The factory supplies a uniform 11-by-10 grid, `M_block=K_block=N_block=8`, eight K blocks, one M block per core and 94 N blocks per core. Gate/up pairs remain intact: `N_tiles=8192`, pair-padded N=8206, 746 weight tiles per core, `N_tiles_per_chunk=32`, 16 output tiles per expert. These geometry/count contracts do not explain a failure before the very first activation publish.

In `minimal_matmul_program_descriptor.cpp:492–495,605–622`, every output becomes a tensor binding. `dm_in0_sender_metal2.cpp:76` eagerly calls `make_tensor_accessors(tensor::outputs)`. `tensor_accessor.h:732–733` constructs a tuple containing all accessors. The writer's fold dispatch in `matmul_dataflow_common_metal2.hpp:358–377` receives already-materialized accessors, so its selection of one chunk does not avoid the large tuple.

### Compiled stack proof

The matching cached injector specialization is:

```text
/home/hous/.cache/tt-metal-cache/10931811594613591564/kernels/
  dm_in0_sender_metal2/2399048435991163597/
```

Its generated headers contain `N_chunks=256`, `M_tiles=4`, `K_tiles=64`, `N_tiles=8192`, `N_blocks_per_core=94`, and output bindings `out0..out255`. Offline `riscv-tt-elf-objdump` and `nm` show:

```text
kernel_main prologue: sp -= 1072; sp -= 2048; sp -= 2048
kernel_main frame:   5168 bytes
kernel __stack_base: 0xffb00c40
noc_reads_num_issued:0xffb00c20  (two uint32 counters)
firmware stack top:  0xffb02000
maximum free range: 0x13c0 = 5056 bytes
```

The cached firmware startup subtracts 16 bytes, firmware `main` subtracts 48, and kernel `_start` subtracts 16. That leaves `kernel_main` SP at `0xffb00b80`; its emitted stores to `160(sp)` and `164(sp)` overlap exactly the two `noc_reads_num_issued` counters. Other nearby NoC globals and firmware-local data also lie inside the frame. This is an actual generated stack overlap, not just a theoretical sum of C++ object sizes.

`ncrisck.cc:61–64` initializes NoC state before calling `kernel_main` at line 73. The tuple construction then spills over those initialized globals. `read_in0_block_sync` issues the activation reads and calls the barrier before `dfb_in0.push_back` (`dm_in0_sender_metal2.cpp:330–366`). Even padded M rows with no actual activation reads still execute that barrier. That explains why all ten injectors stall, including rows beyond the four real M tiles.

The validator (`minimal_matmul_device_operation.cpp:132–172`) checks chunk divisibility and tile/pair alignment but has no split-count/stack-budget check. Those present checks all pass for E=256, H=2048, I=512. Changing fidelity, output precision, or grid shape does not remove the 256-accessor tuple.

## Downstream Effects

The broken activation injector comes first. Receiver semaphore waits, full weight buffers, compute waits, lack of output, queued concatenations, sparse down, and host synchronization all follow from it. Concat has not started and is not the cause. No SwiGLU result exists, so the hang gives no negative PCC evidence about packed real weights. The parent's passing unsplit run is consistent: one output accessor avoids the oversized frame while using the same weight packing.

## Proposed Fix

Within the Python-only scope, keep the already-passing unsplit `minimal_matmul` plus tiled reshape/permute as the qualified adapter. A smaller native split remains a valid experiment rather than rejecting the op after one failure:

1. Smallest configuration/layout change: call the same packed matrix with `chunks=32`. Each result is `[1,1,T,8*512]`; reshape each to `[1,T,8,512]`, permute to `[1,8,T,512]`, then concatenate along expert axis 1. This retains one matmul invocation, preserves expert order, and reduces the accessor count eightfold. Layout overhead may make it slower than unsplit.
2. To retain per-expert outputs, prepare eight BFP4 packed weight groups of 32 contiguous experts during construction, each `[2048,32768]`. Call `minimal_matmul_split(..., chunks=32, fuse_swiglu=True, dtype=BF16)` eight times, flatten returned lists in group/expert order, and concatenate along axis 1. Packing/repacking must retain whole 32-column tiles and the existing BFP4 values. This has extra matmul launches and therefore needs performance measurement.

The owning agent implemented option 2: construction-time tiled weight slices, eight calls of 32 outputs, per-group expert-axis concat followed by group concat. The investigator did not execute it. The owner's [repaired real-weight result](../../prefill_split_repaired129/prefill_split.json) now measures the same layer-3 B1, prefix129, eight-decode-token workload:

| Adapter | HF prefill PCC | Functional prefill PCC | HF minimum decode PCC | Warmed blocking host prefill, microseconds |
|---|---:|---:|---:|---:|
| Functional control | .998847961 | reference | .998132229 | 56824.796 |
| Unsplit packed | .998835623 | .999960482 | .998121917 | 29727.359 |
| Eight groups, 32 outputs each | .998835087 | .999959826 | .998121917 | 26272.256 |

Grouped split's functional minimum decode PCC is .999948978. Runtime audit and traced decode pass; the parent also reports trace allocation tracking enabled. All three JSONs record trial source SHA256 `f6879658217f36fbb17fdf06c2e479fc82ad90cc98cc16b67c31956e1f1f08ef`. The final formatted implementation was rechecked at SHA256 `bc78fb8d141f5e67e1cb26c1198f5a088da49db648c2e09da6cf7bbde3efe32a`; its grouped implementation is present. These are blocking host timings, not isolated kernel/device measurements. Full 4096-token and B32 qualification are separate from this repaired129 result. No claim of a universal safe maximum chunk count is made.

Subsequent owner runs also pass prefix4096/128-decode at source `bc78fb8d...`: [full grouped split](../../split_full4096/full_split.json) measures HF prefill .998573661, minimum decode .995726228, functional prefill .999981582 and minimum decode .999637008; warmed host prefill 698045.450 microseconds versus unsplit 830929.520. [Linear grouped split](../../split_linear4096/linear_split.json) measures HF prefill .999678731, minimum decode .997841656, functional prefill 1.0 and minimum decode .999664128; warmed host prefill 777233.958 versus unsplit 895166.741. B32 qualification and any further single-call/chunks32 variant remain separate evidence.

An upstream native fix would pass compile-time binding tokens into the output dispatch and construct only the selected `TensorAccessor` inside the selected branch, or use another bounded-memory descriptor strategy. Merely avoiding output bindings on nonwriter BRISC kernels does not fix the NCRISC writer. Add a resource validation/test for large split counts. This C++ change is outside the authorized stage scope.

## Uncertainty

The capture does not contain the numeric worker NoC software counters or SP register values, so the exact corrupted runtime counter value is inferred from the emitted stores and firmware stack layout. The cached specialization matches every reported shape/block/chunk parameter; raw cached ELF addresses need runtime load relocation when comparing against captured PCs. The oversized frame and overlap are directly demonstrated offline. The owner's smaller-split real-weight rerun validates the workaround and its measured129 performance; it does not independently read back the corrupted counter from the failed execution.
