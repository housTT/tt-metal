# AUTOTRIAGE

## Diagnosis

- The advertised-context test is stuck before decoder construction or any paged-cache/decode compute kernel runs. The first plausible stuck point is a host-to-device cache initialization issued by one of the two `ttnn.full` calls at `test_functional_decoder.py:915-932`.
- The test's stated allocation contract is not satisfied by the selected API. Despite the docstring saying that the multi-gigabyte tensors are allocated and initialized directly on device, BF16 `ttnn.full(..., device=...)` constructs and fills a host `std::vector`, wraps it in a host tensor, and transfers it to the device. For the first, sliding-attention parameter this creates two 1 GiB host-to-device writes (2 GiB total).
- One such pinned-memory write is hard-stalled in fast dispatch: `cq_prefetch` is waiting forever for outstanding NoC reads from the relay source, so it cannot release more downstream circular-buffer pages; `cq_dispatch` is consequently waiting for those pages while executing the DRAM paged write. This is not evidence of a `paged_update_cache`, paged SDPA, trace-replay, or decoder cache-geometry hang because execution has not reached those operations.
- The source-level stage fix is to use an actual on-device initializer, most directly `ttnn.moreh_full`, for these caches. The lower-level reason that the large pinned transfer lost five NoC read responses remains unresolved and should not be presented as a decoder bug.

## Triage Evidence

- Failing command:

  ```bash
  source python_env/bin/activate && GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 timeout 1800 python -m pytest -q -s models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py -k advertised_context_traced_decode
  ```

- The live processes were preserved for capture: timeout PID 152559 and pytest PID 152565. No process was killed and no device reset was performed during this investigation.
- Primary LLM-readable capture: `models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/triage/tt-triage.txt`. Focused repeat capture: `models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/triage/tt-triage-focused-repeat.txt`.
- Both captures show the same stationary stop-sites and the same NoC counters:
  - Device 1 functional worker `16-2 (11,0)`, BRISC `cq_prefetch`, PC `0xbf90`: `noc_async_read_barrier_with_trid()` at `dataflow_api.h:2430`, called by `process_relay_linear_cmd()` at `cq_prefetch.cpp:1756`.
  - On that core, NOC0 reports 1,905,962 reads issued and 1,905,957 responses received: five reads have not completed.
  - Device 1 functional worker `16-3 (11,1)`, BRISC `cq_dispatch`, PC `0xa770`: `CBReader::acquire_pages()` at `cq_common.hpp:507`, reached from `process_write_paged<true>()` at `cq_dispatch.cpp:645`.
  - `dump_running_operations` is empty and `dump_op_mesh` reports Device 1 idle. Thus no TTNN model operation is resident on compute workers; the active work is the fast-dispatch data transfer itself.
- The pytest process had about 3.9 GiB RSS and one worker thread consuming about one CPU core while its main thread waited on a futex. This is consistent with retained multi-gigabyte host backing and a host-side completion wait; it is not proof by itself of which of the two cache writes is active.
- DRAM training/BIST and error counters, ARC heartbeat, Ethernet links/heartbeats, L1 checks, binary integrity, and core-magic checks passed. There is no direct evidence of a board-wide ARC, Ethernet, DRAM, or firmware-health failure.
- The detailed report prints `check_noc_status.py: fail`, but `triage-summary.txt` incorrectly labels every script as `pass`, including `check_noc_status`. The detailed report and repeated raw counter/call-stack evidence are authoritative here; the summary is internally inconsistent.
- The `check_broken_components` messages saying that cores were halted and then no longer halted arose while triage was actively halting/resuming cores. They are capture-side observations and do not identify the original stuck point.

## Source Evidence

- The local Gemma text configuration advertises 262,144 positions. The test selects layer 0 (sliding attention) before layer 5 (full attention), computes `num_blocks` from that advertised length, uploads a small page table, and then calls `ttnn.full` twice. It does not construct `FunctionalDecoder` until `test_functional_decoder.py:942`, after both cache initializers.
- Sliding-attention cache geometry is `(4096, 8, 64, 256)` BF16:

  ```text
  4096 blocks * 8 KV heads * 64 tokens/block * 256 elements/head * 2 bytes
      = 1,073,741,824 bytes per cache
      = 2,147,483,648 bytes for K + V
  ```

  The later full-attention parameter would allocate `(2048, 2, 128, 512)`, or 512 MiB per cache and 1 GiB total. The live failure is in the first/sliding parameter.
- `ttnn/cpp/ttnn/operations/creation/creation.cpp:51-73` implements the relevant concrete `full_impl` by allocating a `std::vector` for the physical tensor, filling every host element, constructing a `HostBuffer`, and calling `host_tensor.to_device`. The BF16 dispatch at `creation.cpp:164-202` selects that path even when a device is supplied. Therefore the test comment at `test_functional_decoder.py:884-886` is false for `ttnn.full` in this checkout.
- `ttnn.moreh_full` is materially different: `ttnn/cpp/ttnn/operations/full/full.cpp:13-20` launches `ttnn::prim::full`; its interleaved program allocates the output device tensor and has device data-movement kernels build one filled page and replicate it over their assigned output pages (`full_program_factory_interleaved.cpp:31-198` and `kernels/writer_full.cpp:13-69`).
- Producer/consumer ledger for the observed stuck transfer:

  | Resource/state | Producer | Consumer | Proven state |
  | --- | --- | --- | --- |
  | Pinned host cache bytes | `ttnn.full` host `std::vector` mapped for transfer | `cq_prefetch::process_relay_linear_cmd` | A relay-linear command is active; five NoC read responses are absent. |
  | Prefetch scratch half, TRID 6/7 | `cq_prefetch` issues NoC reads into alternating scratch halves | The same prefetch kernel barriers the prior TRID before forwarding/reusing that half | BRISC is stationary in the TRID barrier at line 1756. |
  | Prefetch-to-dispatch CB pages/credit | `cq_prefetch` forwards a completed scratch half and releases page credits | `cq_dispatch::process_write_paged<true>` | Prefetch cannot forward the stalled half; dispatch is waiting in `acquire_pages` for a producer-count change. |
  | Destination DRAM pages | `cq_dispatch` writes each available CB segment to interleaved DRAM pages | The later cache readers | The paged write is incomplete, so no later cache read/update/SDPA operation can safely begin. |

- `process_relay_linear_cmd` explicitly double-buffers the transfer, issues reads with TRIDs 6 and 7, and at line 1756 waits for the prior half's TRID before releasing its pages downstream. `CBReader::acquire_pages` at line 507 waits while the upstream producer count equals the local count. These contracts explain both observed PCs without treating the downstream dispatch waiter as the root cause.

## Downstream Effects

- `cq_dispatch` waiting for CB pages is downstream of the prefetch read barrier, not an independent circular-buffer protocol bug.
- The host's busy worker/completion wait, the pytest silence, and eventual outer `timeout 1800` are consequences of the incomplete cache upload.
- Sentinel writes/readbacks, decoder weight upload, eager decode, trace capture, replay, output PCC, and preservation checks have not run. This attempt provides no correctness or performance result for advertised-context decode.
- The healthy ARC/DRAM/Ethernet checks make a board-wide failure unlikely, but they do not prove why the five pinned-source NoC reads failed.

## Proposed Fix

- Change only the test's two advertised-context cache initializers from `ttnn.full` to the actual on-device primitive, for example:

  ```python
  ttnn.moreh_full(
      cache_shape,
      fill_value,
      mesh_device,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      memory_config=ttnn.DRAM_MEMORY_CONFIG,
  )
  ```

  This matches the test's documented intent, avoids constructing or pinning a 1 GiB host vector per cache, and bypasses the failed relay-linear upload path. Verify first that the replacement is absent from the prepared source; it is absent in the inspected test.
- Focused experiment after the current hung process is terminated/times out and device health is restored:
  1. Run a small-cache initializer/readback comparison using `ttnn.moreh_full` for the same two nonzero BF16 values. Confirm exact baseline values and confirm pytest RSS does not grow by the logical cache byte size.
  2. Run only the sliding advertised-context parameter. If it reaches sentinel readback and decoder construction, the diagnosed initialization bug is confirmed independently of later decoder behavior.
  3. Run only the full-attention parameter, then the original two-parameter command.
  4. If `moreh_full` itself stalls, capture a fresh triage report. A compute-worker `full_interleaved` stop-site would be a different bug from the pinned relay shown here.
- Because the current fast-dispatch state has five reads permanently outstanding, do not launch another device workload on it. After preserving this report and the triage files, follow the bounded list/reset/list and mesh-open/close smoke from the device-usage instructions before the next experiment.
- No implementation code was changed in this investigation.

## Uncertainty

- Triage proves that five NoC reads in the pinned host-to-device relay did not complete across repeated captures. It does not expose the individual read addresses or TRID counters, so it cannot distinguish an invalid/out-of-range pinned mapping, an IOMMU/PCIe response loss, or another low-level large-transfer defect.
- The capture cannot distinguish whether the active transfer belongs to the key-cache or value-cache `ttnn.full` call. That distinction does not affect the test-level diagnosis because both calls use the same unsupported-for-intent host-backed path and the same 1 GiB geometry.
- Replacing the initializer removes this pre-decoder hang but does not by itself establish that 262,143-position paged decode, full-context SDPA, or trace replay is correct. Those remain the intended checks once initialization completes.
- No hardware-dependent fix or performance claim has been validated in this diagnostic pass.
