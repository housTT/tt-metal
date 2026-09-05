# AUTOTRIAGE — full-model worker watcher failure

## Diagnosis

The new two-reader LM-head geometry exposes a concrete NoC packet-size contract
violation in the shared DRAM-sharded matmul reader. Its `SPLIT_DRAM_BANK` branch
passes a **36,864-byte row** to a template specialization that promises **at most
one 16,384-byte Blackhole NoC packet**. The firmware assertion checks global
read-response accounting after the kernel returns; it does not prove that a
final transaction-ID barrier was omitted. The oversized single-packet request is
the leading explanation for that exact assertion, pending the isolated
before/after experiment below.

Investigation was source-only on 2026-09-05. No device access, reset, runtime
experiment, implementation edit, or build was performed by this investigator.
The parent owns recovery and subsequent hardware execution.

## Triage Evidence

- `logs/trace_b32_watcher_v1.log` records the reduced real-weight layer-0/layer-3,
  batch-32 trace-contract run. Watcher was enabled for workers; ETH checks were
  explicitly disabled. At 16:11:48.439 device 0 logical worker `(0,0)`, virtual
  `(1,2)`, BRISC reports missing NoC reads flushed barrier, then aborts the host.
- The current BRISC kernel is
  `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`; NCRISC is its
  `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` partner, and all TRISCs
  name `bmm_large_block_zm_fused_bias_activation.cpp`.
- Last waypoints are `NKFW, W, W, W, W`. `tt_metal/hw/firmware/src/tt-1xx/brisck.cc:91`
  performs the failed assertion **after `kernel_main()` returns**. This is an
  intentional firmware stop, not a live NoC-wait call stack or a demonstrated
  fabric deadlock.
- `watcher_failure/watcher.log` ends at completed dump #2, about 12.622 seconds
  after initialization; it does not contain the failure dump or failure kernel
  IDs. The abort therefore left no numeric issued/received counter values,
  failure runtime args, or complete device/core fanout. Do not invent them.
- `watcher_failure/kernel_names.txt` and `kernel_elf_paths.txt` preserve compiled
  kernels. Head-adjacent reader IDs 535/538/541/544 share ELF cache hash
  `14531292800595639909`; their compute partners share
  `15141621231386301773`. The cached head compute named args include
  `bias_ntiles=64`, and the reader descriptor gives operand-1 tile size 576 bytes.
  These support the source-derived head geometry, but the saved log cannot
  bind a particular head chunk/kernel ID to the failing core. A terminal-only
  reproducer is required for that attribution.

## Source Evidence

Paths below are relative to the repository root.

1. `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:127`
   selects the split-bank row loop. `read_size = reader_width_tiles *
   in1_tile_size_bytes`, while line 133 calls
   `noc.async_read<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(...)`.
2. `tt_metal/hw/inc/api/dataflow/noc.h:189` forwards that maximum-size template
   argument unchanged to `noc_async_read`. Its TXN_ID setup tags each request
   and limits outstanding transactions, but does not split an oversized row.
3. `tt_metal/hw/inc/api/dataflow/dataflow_api.h:566` selects
   `noc_async_read_one_packet` whenever that template argument is at most
   `NOC_MAX_BURST_SIZE`. The one-packet primitive invokes
   `ncrisc_noc_fast_read` once.
4. `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:503`
   increments `noc_reads_num_issued[noc]` by **one** per invocation.
   Its line 515 checks hardware `NIU_MST_RD_RESP_RECEIVED` for equality to that
   software count. In contrast, its any-length reader at line 860 emits
   bounded packets in a loop and accounts for every emitted packet.
5. `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:286`
   defines 512-bit payload words, 256 words per burst: the Blackhole limit is
   **16,384 bytes**, not the 8,192-byte limit of other configurations.
6. The factory
   `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:345`
   supplies block count, row width, and reader count; lines 255–257 bound
   `in1_buffer_page_size` for the non-split loop. The split loop bypasses those
   bounded pages by reading whole rows.

The selected head in `tt/model.py` has local N=32768, eight DRAM banks, two
readers per bank, K=4096, and K block width four tiles. BFP4_B tiles occupy
576 bytes, already aligned to Blackhole DRAM requirements.

| Resource / count | Producer | Consumer / completion | Exact geometry |
| --- | --- | --- | --- |
| Weight columns | Eight banks, two readers each | Each reader computes one column interval | 32768 / 32 / 8 / 2 = 64 tiles/reader |
| Weight row read | Split-bank BRISC loop | NoC response and CB input-1 | 64 × 576 = 36864 bytes |
| Weight blocks | BRISC reader | Compute pops input-1 blocks | 4096 / 32 / 4 = 32 blocks |
| Rows per block | BRISC reader | Same block's transaction-ID barrier | 4 rows, 256 tiles, 147456 bytes |
| Input-1 CB | Reader reserves/pushes | Compute waits/pops | Triple buffer: 442368 bytes; 32 pushes, 32 consumed blocks |
| Transaction IDs | Blocks use 1,2,3 cyclically | Loop waits preceding block, final wait drains last block | Blocks 0–30 waited in iterations 1–31; block 31 waited after loop |
| Actual packet requirement | Any-length equivalent | Global read-response accounting | 3 packets per row; 384 bounded requests/reader for all 128 rows |
| Current issued accounting | One-packet path | Firmware global equality check | Only 128 software increments/reader |
| Output | Compute pushes CB output | BRISC waits, reshards, write-barriers, pops | Existing output write barrier is present |

The 128-versus-384 packet accounting discrepancy is a **prediction**, not a
captured hardware-counter reading. The source proves the invalid one-packet
API usage; the captured symptom strongly supports the resulting accounting
failure. A global barrier alone could spin forever if responses already exceed
the incorrectly incremented count.

The optimized decoder's packed gate/up has local width 6144 and three readers
per bank: 8 BFP4 tiles/reader = 4608 bytes per row. Its down projection with two
readers similarly uses 8 BFP4 tiles per row. These fit the packet contract,
explaining why earlier decoder watcher evidence can pass while the wider head
fails. The previous 16384-column/one-reader head takes the non-split bounded-page
branch instead. This is a geometry-dependent contrast, not evidence that all
multi-reader matmuls fail.

The existing end-of-kernel custom-VC restore was inspected and is present.
`ncrisc_noc_read_set_state` issues no transaction and changes no issued counter.
Adding that already-present restore cannot fix the observed assertion. The
packet-tag firmware check concerns write/atomic command buffers, not the read
command buffer, so a speculative read-tag clear is not the first intervention.

## Downstream Effects

The host abort and any device recovery needed afterward follow the worker
firmware assertion. No captured evidence identifies Ethernet/fabric, sampler,
page-table contents, cache position, or token feedback as the first fault.
They should remain unchanged during the isolated repair experiment.

## Proposed Fix

First reproduce the exact terminal geometry under worker watcher, then test a
single source change in the split-bank read call: use
`noc.async_read<NocOptions::TXN_ID>(...)`, omitting its false one-packet maximum.
The default `NOC_MAX_BURST_SIZE + 1` selects the existing any-length primitive,
which sends 16384 + 16384 + 4096 bytes for each 36864-byte row and increments the
issued counter three times. Keep the same transaction ID, column interval,
weight layout/dtype, K blocking, CB pipeline, compute, CCL, and output layout.
The non-split bounded-page call should remain unchanged.

Minimal command from repository root (parent must serialize hardware and record
provenance/logs before running):

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
timeout 180 python_env/bin/python -m \
models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_terminal \
--columns 32768 --in0-block-w 4 --readers 2 \
--dtype bfloat4_b --fidelity LoFi --resident-l1-bytes 114688
```

Preserve the unmodified failure artifact, recover according to device skill,
then rerun this exact command with the one-line repair. Require watcher-clean
eager and trace executions, unchanged numerical/packing checks, and measured
terminal latency. Follow with the original batch-32 trace-contract watcher
case and full-model readiness checks. A 16384-column/one-reader watcher control
can further test the predicted branch contrast; it is a diagnostic control,
not the final optimized implementation.

This is a C++ kernel edit. The author must perform the repository-required
`.github/scripts/copilot-build.sh --build-ttnn-tests` (or document its actual
environment failure) as well as successful device kernel JIT/repro checks.
Neither build nor device validation has been performed in this investigation.

## Uncertainty

The exact failing Python operation and head chunk are not directly named by
the buffered watcher artifacts. Numeric NoC counters and the aborted failure
dump are unavailable. If the isolated wide-head case does not reproduce,
capture the next live failure before recovery and obtain running-op/runtime-arg
and NoC-counter evidence; do not declare the head repaired from source alone.
If removing the incorrect one-packet specialization does not resolve the
failure, return the new evidence to AutoFix/AutoTriage instead of disabling
watcher, adding a blanket barrier, or reducing the optimized strategy.
