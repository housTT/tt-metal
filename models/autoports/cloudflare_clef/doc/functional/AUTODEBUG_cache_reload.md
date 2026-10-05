# AutoDebug report: host spin when placing a cached multi-device tensor of 45 MB onto a Blackhole mesh

Inspection only. No device runs. Paths are absolute. Line numbers are for commit `76ec1de49ed` in `/home/hous/dev/clef/tt-metal`. The focus paths named in the request (`ttnn/cpp/ttnn/tensor`) no longer exist; the code lives in `/home/hous/dev/clef/tt-metal/ttnn/core/tensor`, `/home/hous/dev/clef/tt-metal/tt_metal/impl/tensor`, and `/home/hous/dev/clef/tt-metal/tt_metal/distributed`.

## Headline

- The spin is not in the flatbuffer load. `load_tensor_flatbuffer` only `mmap`s the file and builds `HostBuffer` views into the mapping (`/home/hous/dev/clef/tt-metal/ttnn/core/tensor/serialization.cpp:79-130`, `/home/hous/dev/clef/tt-metal/ttnn/core/tensor/flatbuffer/tensor_flatbuffer.cpp:264-307`). That is why `ttnn.load_tensor(path)` returns in 0.0 s for a 2.5 GB file.
- Both failing call patterns (`load_tensor_flatbuffer(path, device=mesh)` and `load_tensor(path)` then `to_device`) converge on the same host-to-mesh write: `MeshCommandQueue::enqueue_write_tensor` in `/home/hous/dev/clef/tt-metal/tt_metal/impl/tensor/tensor_apis.cpp:155-215`.
- That write has a size gate. When the sum of local shard bytes exceeds 32 MiB (`k_pin_write_threshold_bytes`, `tensor_apis.cpp:39-47`), it pins each shard's host memory for direct device DMA (`PinnedMemoryCache::try_pin`, `/home/hous/dev/clef/tt-metal/tt_metal/distributed/pinned_memory_cache.cpp:163-304`) and issues a `CQ_PREFETCH_CMD_RELAY_LINEAR` that makes the device prefetcher read the host pages over PCIe with a 64-bit address (`/home/hous/dev/clef/tt-metal/tt_metal/impl/buffers/dispatch.cpp:1014-1047`, `/home/hous/dev/clef/tt-metal/tt_metal/impl/dispatch/kernels/cq_prefetch.cpp:1719-1790`). Below the gate it uses the ordinary copy-through-command-queue path.
- On this host the pinned path is live. The IOMMU is on with translated domains (`DMA-FQ` for the four Tenstorrent functions in iommu groups 14 to 17), the kernel module is 2.10.0 (device-read-only pinning needs 2.9.0 or newer), and Blackhole advertises unlimited pins and 64-bit PCIe addressing (`/home/hous/dev/clef/tt-metal/tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp:520-542`). All four chips are MMIO devices, so the local `RELAY_LINEAR` variant is used, not the `_H` tunnel variant.
- Every observation lines up with this gate. The 2.6 MB probe tensors are far below 32 MiB and never pin. The 45 MB tensors and the 2.5 GB embedding are above it. The one 45 MB write that works, the fresh `as_tensor` write from torch, passes the same gate with the same pinning parameters, so by the code it also takes the pinned path (the logs do not record pinning directly). If so, the pinned path itself is not broken in general, and what differs is only the host memory handed to it.

| Case | Local shard bytes | Path | Host memory | Shard start mod 64 | Result |
| --- | --- | --- | --- | --- | --- |
| 2.6 MB probe, all variants | 1.4 MB to 2.6 MB | copy through CQ | heap | any | 0.0 s |
| 45 MB `as_tensor` write (torch to mesh) | 45,260,800 (1,4) or 44,912,640 (1,2) | pinned DMA (inferred from the gate; not logged) | heap `std::vector`, glibc mmap chunk, pointer at page + 16 (inferred from glibc, section 3) | 16 | 0.2 s |
| 45 MB reload, either pattern | same | pinned DMA | `mmap(PROT_READ, MAP_PRIVATE)` of the file, shards 64-byte aligned | 0 | never returns |
| 2.5 GB embedding reload (stage 0) | 2 x 1,271,398,400 | pinned DMA | same file mapping | 0 | never returns |

- The exact instruction where the process spins cannot be proven from source. Two concrete differences between the passing heap write and the hanging file write survive inspection, and both are inside the pinned path:
  1. The file shards are read-only, file-backed page-cache pages pinned with `TENSTORRENT_PIN_PAGES_READ_ONLY`, while the heap shards are anonymous memory. The pin itself should succeed for both on this kernel module (`/usr/src/tenstorrent-2.10.0/memory.c:636-642` uses `FOLL_LONGTERM` without `FOLL_WRITE` for read-only pins), so this difference would have to act at the kernel or IOMMU level during the device's PCIe reads. Unproven.
  2. A heap shard sits 16 bytes past a page boundary, so the write sends a 48-byte inline prefix and starts the device read at a 64-byte boundary. A file shard is already 64-byte aligned, so `alignment_prefix_bytes` is 0 and the write takes the no-prefix branch (`dispatch.cpp:111-123`, `dispatch.cpp:556-575`). The file format was aligned to 64 bytes precisely to enable this branch (`/home/hous/dev/clef/tt-metal/ttnn/core/tensor/flatbuffer/tensor_file_layout.hpp:31-38`, commit `627f631e801`). The no-prefix branch is therefore the branch a file load always takes and a `std::vector` source of this size does not take. A line-by-line walk of its device-side page accounting found no defect (section 5). Unproven as the cause.
- The existing regression test for this scenario does not cover this hardware. `test_large_read_only_file_backed_tensor_upload` in `/home/hous/dev/clef/tt-metal/tests/ttnn/unit_tests/tensor/test_tensor_serialization.py:62-80` states that every current CI runner lacks device-read-only pinning and therefore takes the fallback copy path. The pinned read-only file path is unexercised by CI, and it is unexercised for multi-shard mesh tensors anywhere.

## Workaround that needs no tt-metal change

Set `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` in the environment of the process that loads the cache.

- The option is parsed in `/home/hous/dev/clef/tt-metal/tt_metal/llrt/rtoptions.cpp:656-685` and accepts 0.
- With a global limit of 0, `evict_oldest_until_within_limit` returns false for every request (`pinned_memory_cache.cpp:117-124`, `incoming_size_bytes > global_limit_bytes`), so `try_pin` returns `nullptr` (`pinned_memory_cache.cpp:265-267`).
- `enqueue_write_tensor` then falls back to `enqueue_write(mesh_buffer, distributed_host_buffer, false)` (`tensor_apis.cpp:198-204`), the same copy path that the 2.6 MB tensors take and that the serialization test describes as the path every CI runner takes.
- Cost: every host-to-device write above 32 MiB, including fresh conversions, copies through the command queue instead of DMA. Device-to-host reads also lose the pinned destination fast path (`tensor_apis.cpp:139-145`). The H2D and D2H socket services call `PinnedMemory::Create` directly (`/home/hous/dev/clef/tt-metal/tt_metal/distributed/h2d_socket.cpp:99,137`, `/home/hous/dev/clef/tt-metal/tt_metal/distributed/d2h_socket.cpp:130`, `/home/hous/dev/clef/tt-metal/ttnn/core/services/h2d_socket_service.cpp:664`), bypass the cache, and are not affected.
- The tt-metal unit tests already treat a limit of 0 as the supported disable switch (`/home/hous/dev/clef/tt-metal/tests/tt_metal/tt_metal/api/tensor/test_mesh_tensor.cpp:558-560`).

If the environment variable is not acceptable, the second option is to keep the loaded tensor off the file mapping: `ttnn.load_tensor(path)` on host, `ttnn.to_torch` with a `ConcatMeshToTensor` composer, then `ttnn.from_torch` with the mesh mapper and `device=mesh`. This reproduces the heap write that works, at the cost of a decode and re-encode per load. A host-side same-spec `to_dtype` or `to_layout` does not copy the buffer (`/home/hous/dev/clef/tt-metal/tt_metal/impl/tensor/host_to_tensor_spec_apis.cpp:120-122` returns the input unchanged), so it cannot be used to materialize the mapping into heap memory.

Answers to the configuration questions in the request:

- `DumpTensorMode`: no effect. `LOCAL` only skips the multi-host all-gather and barrier (`serialization.cpp:38-49`, `67-70`). The file layout and the loader are identical.
- Loading per-device shards separately: not available through the public API in a way that helps. `ttnn.copy_host_to_device_tensor` and `ttnn.to_device` both route through the same gated write. A single-shard host tensor on a multi-device mesh is treated as a replicate request (`/home/hous/dev/clef/tt-metal/tt_metal/impl/tensor/distributed_tensor_apis.cpp:291-299`).
- Avoiding distributed host storage: the gate is on total local bytes, not on the storage type, so a unit mesh load above 32 MiB takes the same pinned path.

## 1. Direct observations versus interpretations

Observations, from the logs in `/home/hous/dev/clef/logs/`:

- `stage1_cache_reload_probe.log`: six `as_tensor` write and reload pairs of a `[1,1,256,5120]` source on the (1,4) parent and the (1,2) submesh all complete in 0.0 s.
- `stage1_cache_reload_probe_parent.log`: the (1,4) write of `[1,1,5120,8240]` bfp8 TILE sharded on dim 3 takes 0.23 s and writes 45,261,440 bytes; the reload through `load_tensor_flatbuffer(path, device=parent)` prints `reload: start` and nothing more.
- `stage1_cache_reload_probe_alts.log`: the (1,2) write takes 0.22 s and writes 44,913,088 bytes; `load_tensor` returns in 0.00 s with host storage and shape `[1,1,5120,4120]`; `to_device` prints `start` and nothing more.
- `stage0_tp2_sanity_l4_run3.log`: the 2.5 GB embedding is regenerated from torch (works), small cached norm weights load, then the first 45 MB bfp8 layer weight reload stalls.
- UMD logs `IOMMU: enabled` and `KMD version: 2.10.0` in every run.
- No `Pinned source memory start address` or `Pinned memory region must contain` info message appears in any log, so the pinned path never rejected a source for alignment or range reasons.
- `stage1_cache_reload_probe_alts_gdb.txt` is empty. There is no stack capture of a stalled process.

Interpretations carried by the request, not verified here: that the spinning thread is the main thread, and that the CPU time is user time rather than system time. No dispatch, reader, or worker thread in tt-metal sets a thread name, so every thread of the process shows the Python process name in `top`; only a thread id equal to the process id identifies the main thread. Section 6 gives the checks that settle both.

## 2. Causal chain with line numbers

Python entry, `/home/hous/dev/clef/tt-metal/ttnn/ttnn/operations/core.py`:

- `as_tensor` with an existing cache file calls `ttnn._ttnn.tensor.load_tensor_flatbuffer(cache_file_name, device=device)` (line 904).
- `load_tensor` calls the same binding with the device (line 773).
- `to_device` is the C++ `ttnn::to_device` binding.
- The write path of `as_tensor` (`from_torch_and_dump`, lines 871-897) converts on host with the mesh mapper, dumps, then calls `tensor.to(device, memory_config)`. This is the passing 45 MB write.

C++ load, `/home/hous/dev/clef/tt-metal/ttnn/core/tensor/serialization.cpp:79-130`:

- Opens the file, `mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0)` (line 90), wraps the mapping in a `MemoryPin` (line 94), checks the data region is 8-byte aligned (lines 120-123), builds the tensor from the mapping (line 125), and if a device was given calls `tensor.to_device(device, spec.memory_config())` (line 127).
- `from_flatbuffer` (`tensor_flatbuffer.cpp:264-307`) creates one `HostBuffer` per shard as a `Span` into the mapping sharing the single `MemoryPin` (lines 289-295) inside a `DistributedHostBuffer` whose shape is the file's mesh shape (lines 275-279).

C++ write, both patterns:

- `ttnn::to_device` (`/home/hous/dev/clef/tt-metal/ttnn/core/tensor/tensor_ops.cpp:133-161`): host mesh shape equals device shape and every coordinate is present, so `is_uniform_write` (`distributed_tensor_apis.cpp:74-85`) is true and `cq.enqueue_write_tensor(host_tensor, mem_config)` runs.
- `MeshCommandQueue::enqueue_write_tensor(host, memory_config)` (`tensor_apis.cpp:90-108`) allocates the mesh buffer and calls the two-argument overload.
- `MeshCommandQueue::enqueue_write_tensor(host, device_tensor)` (`tensor_apis.cpp:155-215`): `select_local_host_shards` sums the shard bytes (`/home/hous/dev/clef/tt-metal/tt_metal/impl/tensor/tensor_impl.cpp:19-34`); `should_use_pinned_write_path` (lines 41-47) is true above 32 MiB when the IOMMU is enabled and pins are available; for each shard `try_pin(..., map_to_noc=true, ReadOnly)` (lines 179-196); with any pin, `enqueue_write_shards(mesh_buffer, transfers, blocking=true)` (line 199).
- `enqueue_write_shards_nolock` (`mesh_command_queue_base.cpp:230-285`) runs `write_shard_to_device` per device on the device-bound thread pool, waits, resets the prefetcher cache manager, and calls `finish_nolock()`.
- `FDMeshCommandQueue::write_shard_to_device` (`fd_mesh_command_queue.cpp:810-859`) calls `buffer_dispatch::write_to_device_buffer(src, shard, ..., pinned_memory)`.
- `write_to_device_buffer` (`dispatch.cpp:1157-1385`): the buffer is interleaved and unpadded (page size 1088 equals the aligned page size on Blackhole, DRAM alignment 64), the source start passes the 16-byte L1 read alignment check (line 1200), and the source lies inside the pinned region (line 1211), so `use_pinned_transfer` is set with `pinned_src_addr` equal to the shard's IOVA (lines 1222-1229). `initialize_interleaved_buf_dispatch_params` builds `InterleavedBufferWriteDispatchParams`, whose constructor sets `alignment_prefix_bytes = 64 - (src_addr % 64)` when the IOVA is not 64-byte aligned (lines 111-123).
- `write_interleaved_buffer_to_device` (lines 1054-1100), pinned branch: one `issue_buffer_dispatch_command_sequence` (lines 945-1048) that emits `RELAY_INLINE_NOFLUSH` plus `CQ_DISPATCH_CMD_WRITE_PAGED` for all pages, then in a separate fetch-queue entry `CQ_PREFETCH_CMD_RELAY_LINEAR` with the PCIe core coordinate, the full byte length and the 64-bit IOVA (lines 1014-1047).
- Device side: `process_relay_inline_noflush_cmd` (`cq_prefetch.cpp:915-941`) copies the dispatch command into the dispatcher's buffer page without releasing it; `process_relay_linear_cmd` (`cq_prefetch.cpp:1719-1790`) reads the host memory through `noc_read_64bit_any_len` (lines 1690-1717) into its scratch double buffer and relays it to the dispatcher; `process_write_paged` (`/home/hous/dev/clef/tt-metal/tt_metal/impl/dispatch/kernels/cq_dispatch.cpp:601-698`) writes the pages into DRAM.
- Host completion: `finish_nolock` (`fd_mesh_command_queue.cpp:756-794`) records an event through the worker threads and then sleeps on a condition variable (`wait_for_outstanding_reads`, line 776) until the completion reader thread, started at line 232, has consumed the event; that reader polls the completion queue without yielding (`system_memory_manager.cpp:763-811`). The dispatch timeout `TT_METAL_OPERATION_TIMEOUT_SECONDS` defaults to 0 (`/home/hous/dev/clef/tt-metal/tt_metal/llrt/rtoptions.hpp:389`), so a device that never completes is waited for forever. For the `load_tensor_flatbuffer(path, device)` pattern the mapping is unpinned and unmapped only when the function returns (the local `MemoryPin` at `serialization.cpp:94` holds the last reference until line 130), so the unpin cannot be what stalls before the write completes.

Pinning, both patterns:

- `PinnedMemory::Create` (`/home/hous/dev/clef/tt-metal/tt_metal/distributed/pinned_memory.cpp:389-428`) and `PinnedMemoryImpl::initialize_from_devices` (lines 90-168): the host pointer is rounded down to a page, `host_offset_` is recorded, and on Blackhole `map_to_noc` is forced off in favour of 64-bit addressing (lines 149-153). `cluster.map_sysmem_buffer(mmio_device_id, aligned_ptr, size + host_offset_, false, READ_ONLY)`.
- UMD `pin_and_wrap` (`/home/hous/dev/clef/tt-metal/tt_metal/third_party/umd/device/chip_helpers/silicon_sysmem_manager.cpp:473-530`) page-rounds the range and calls `PCIDevice::map_for_dma` (`/home/hous/dev/clef/tt-metal/tt_metal/third_party/umd/device/pcie/pci_device.cpp:699-738`) with `TT_DMA_FLAG_READ_ONLY`.
- Kernel module `ioctl_pin_pages` (`/usr/src/tenstorrent-2.10.0/memory.c:593-790`): read-only requires IOMMU translation (line 638), pins with `FOLL_LONGTERM` and no `FOLL_WRITE` (lines 641, 671), maps with `DMA_TO_DEVICE` (lines 642, 697), requires one contiguous IOVA range (lines 701-719), and rejects only an identical virtual address and page count (lines 646-656). The device DMA mask is 58 bits (`/usr/src/tenstorrent-2.10.0/enumerate.c:338`, `/usr/src/tenstorrent-2.10.0/blackhole.c:814`), and sysfs reports 58 for both masks of `0000:01:00.0`.
- The device-side address programming handles the full 64-bit IOVA: the explicit-coordinate `noc_read_with_state` overload writes `NOC_TARG_ADDR_LO` and `NOC_TARG_ADDR_MID` from the 64-bit source address and the PCIe coordinate separately (`/home/hous/dev/clef/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:1892-1915`). So IOVAs above 4 GB are not a differentiator.

## 3. Instantiated geometry

Parent (1,4), source `[1,1,5120,8240]` bfp8 TILE sharded on dim 3:

- Per-device logical shard `[5120, 2060]`, padded to `[5120, 2080]`, 160 x 65 = 10,400 tiles of 1,088 bytes = 11,315,200 bytes per device; file 45,261,440 bytes = 4 x 11,315,200 + 640 bytes of header. Local shard bytes 45,260,800 > 33,554,432.
- 11,315,200 = 64 x 176,800, so consecutive shards in the file are adjacent with no padding, and 11,315,200 = 2762 x 4096 + 2048, so each shard boundary falls in the middle of a 4 KiB page. Adjacent shards share one page; each is pinned as its own page-rounded range.

Submesh (1,2): per-device `[5120, 4120]` padded to 4128, 129 x 160 = 20,640 tiles, 22,456,320 bytes per device, total 44,912,640 > 33,554,432. 22,456,320 = 5482 x 4096 + 2048, same page sharing.

Embedding (stage 0): per-device `[248320, 2560]` bf16 ROW_MAJOR, page size 5,120 bytes (a multiple of 64, so unpadded), 1,271,398,400 bytes per device.

The 2.6 MB probe: `[256, 1280]` bfp8 is 348,160 bytes per device, four shards 1,392,640 bytes; the replicated bf16 `[256, 5120]` is 2,621,440 bytes; all below the gate.

Heap sources: a `std::vector` of 11 MB or more is normally served by glibc through its own mmap chunk, whose user pointer is 16 bytes past a page boundary, so the IOVA of the shard start is 16 mod 64 and `alignment_prefix_bytes` is 48. glibc raises its mmap threshold dynamically up to 32 MiB as such chunks are freed, so a later, smaller shard can come from the main arena with any 16-byte residue; within each probe run the shards were the same size as the first allocation and stay on the mmap path. The relay then starts at IOVA + 64 with length 11,315,152 and the first 48 bytes travel inline. File sources: the data region and every shard start on a 64-byte boundary (`tensor_file_layout.cpp:48-78`), so the relay starts at the shard IOVA with the full length and nothing travels inline.

## 4. Why the shard width not being a multiple of 32 is not the cause

The failing shards have width 2060 and 4120, which are not multiples of 32, while the passing probe widths 1280 and 2560 are. This difference is a coincidence of the chosen sizes. The host tensor stored in the file already holds the padded physical layout (`[5120, 2080]` of tiles), the write transfers `compute_packed_buffer_size_bytes` whole tiles, and the embedding that also hangs is `[248320, 2560]`, fully tile aligned. Padding does not enter the write path.

## 5. Candidate mechanisms, ranked

### 5.1 Pinned DMA from a read-only, file-backed private mapping (unproven)

What is different: the kernel pins page-cache pages of `/home/hous/dev/clef/tt_cache/...tensorbin` (ext4 on `/dev/nvme0n1p2`) instead of anonymous pages. Everything after the pin is identical to the heap case: `DMA_TO_DEVICE` mapping, one contiguous IOVA, device reads.

What the code says: the pin should succeed. The KMD uses `FOLL_LONGTERM` without `FOLL_WRITE` for read-only pins (`memory.c:641`), the mapping is `PROT_READ`, the filesystem is not DAX, this host has no CMA area and an empty movable zone (`/proc/meminfo` `CmaTotal: 0 kB`, `/proc/zoneinfo` `Movable` `managed 0`), so the long-term pin needs no page migration. A pin failure would not spin either: `try_pin` catches the UMD exception and returns `nullptr` (`pinned_memory_cache.cpp:289-303`), and the write falls back to the copy path.

What would prove or refute it: run the same reload with the cache file copied to `/dev/shm` (tmpfs, shmem pages), and run a 45 MB write from a 64-byte-aligned anonymous source (section 6). The kernel log is restricted on this host (`dmesg_restrict=1`) and `journalctl -k` returned nothing, so pin failures or `IO_PAGE_FAULT` events could not be checked from this session.

### 5.2 The no-prefix branch of the pinned interleaved write (unproven)

What is different: `RELAY_INLINE_NOFLUSH` carries 16 bytes (the dispatch command only) instead of 64, `WRITE_PAGED` is emitted by `add_dispatch_write_paged(flush_prefetch=false, ...)` (`/home/hous/dev/clef/tt-metal/tt_metal/impl/dispatch/device_command.cpp:632-670`) instead of `add_dispatch_write_paged_with_custom_inline_size` (lines 672-707), and the relay length is a multiple of 64.

What the code says: the device-side accounting is symmetric. An independent line-by-line walk with Blackhole constants (dispatch page 4096 B, scratch half 65,536 B, burst 16,384 B) gives identical ledgers for both variants: for 10,400 pages the prefetcher acquires 1 page in `process_relay_inline_noflush_cmd` (`cq_prefetch.cpp:915-941`), then 16 pages per 64 KiB chunk for 172 chunks and 10 for the 43,008-byte tail in `write_pages_to_dispatcher` (`cq_prefetch.cpp:947-1000`), 2,763 pages in total, and releases 2,763; the dispatcher consumes 2,763 pages because `process_write_paged` reads 11,315,200 bytes from `cmd_ptr + 16` and rounds its end pointer to the next page. The prefix variant lands on the same totals with a 64-byte header. No chunk boundary coincides with a page boundary in either variant, no transfer has zero length, and the tail splits into bursts of 16,384, 16,384 and 10,240 (no prefix) or 10,192 (prefix). The dispatcher reads its data at `cmd_ptr + 16` in both variants and in the ordinary copy path (`process_relay_inline_cmd`, `cq_prefetch.cpp:872-909`, places the command at the page start and data at +16). The host side is consistent too: the populated sequence is padded to 64 bytes by `align_write_offset()` (`dispatch.cpp:619`, `device_command.cpp:962`), the calculator says 64 (`device_command_calculator.hpp:161-169`, `add_alignment` at line 61), and the prefetch stride is `align(16 + 16, 64) = 64` (`device_command.cpp:1251`).

Why it stays on the list: it is the one dispatch-level branch that a file load always selects and a `std::vector` source of this size never selects, and it has no CI coverage on hardware with read-only pinning (section headline). Nothing in the branch itself has been shown to fail.

### 5.3 Ruled out by inspection

- 64-bit IOVA width: the prefetcher programs `NOC_TARG_ADDR_MID` from the 64-bit source on every read (section 2), the KMD advertises 58 address bits, and the passing heap writes already use the same addressing.
- Overlapping pins of the shared boundary page: the KMD rejects only an identical (address, page count) pair, consecutive shards start at different addresses, and the shards target different chips and therefore different IOMMU domains.
- A host-side loop in `try_pin`: the create-and-evict loop is bounded by the number of cache entries, and a held entry is skipped (`pinned_memory_cache.cpp:91-104`, `272-303`). With the default 4 GiB cache limit and Blackhole's unlimited per-device budget, the eviction loops run zero times for 45 MB.
- Any other user-space spin on the main thread: the only bare spin reachable from this flow is `EventSynchronize` (`/home/hous/dev/clef/tt-metal/tt_metal/distributed/distributed.cpp:138-143`), called from `drain_barrier_events` when a pinned buffer is released. Barrier events are recorded only for non-blocking pinned transfers (`mesh_command_queue_base.cpp:273-284`), and this write is blocking (`tensor_apis.cpp:199`), so that spin is unreachable. The thread pool's `wait` sleeps on a futex (`thread_pool.cpp:270-284`). If the device is stuck, the busy thread is a dispatch worker in `fetch_queue_reserve_back` (copy path only) or the completion reader thread in `completion_queue_wait_front`, never the main thread. A main thread that is genuinely at 100% would therefore be inside the pin ioctl in kernel time.
- The submesh, `num_command_queues=2`, and `FABRIC_1D`: the passing 45 MB heap write ran on the same mesh objects and queues.
- The shard width padding: section 4.

## 6. Diagnostics for the next device session

Run these on the stalled process before killing it. Each one answers a question the source cannot.

1. Which thread spins and where. `py-spy dump --native --pid <pid>`, or `gdb -p <pid> -batch -ex 'thread apply all bt'`. By the code, a stuck device shows the completion reader thread in `completion_queue_wait_front` and the main thread asleep in `wait_for_outstanding_reads` (`fd_mesh_command_queue.cpp:776`). A main thread inside `ioctl_pin_pages` or `__gup_longterm_locked` points at the kernel pin. A main or worker thread in `fetch_queue_reserve_back` or `wrap_issue_queue_wr_ptr` points at a full issue queue, which only happens on the copy path.
2. User versus system time. Sample `/proc/<pid>/task/<tid>/stat` fields 14 and 15 twice. System time growing on the main thread means the pin ioctl; user time growing on another thread means polling the device. `/proc/<pid>/task/<pid>/syscall` and `/proc/<pid>/task/<pid>/stack` (root) show whether the main thread sits in `ioctl` and where in the kernel.
3. Device state. `tt-triage` on the mesh while stalled, following the `tt-debug-tools:tt-triage` skill, and read the prefetcher and dispatcher waypoints. Prefetcher at `NBTW` (`/home/hous/dev/clef/tt-metal/tt_metal/hw/inc/api/dataflow/dataflow_api.h:2445`, the NoC read barrier inside `process_relay_linear_cmd`) means the PCIe read of the pinned pages never completed. Prefetcher at `DAPW` (`/home/hous/dev/clef/tt-metal/tt_metal/impl/dispatch/kernels/cq_common.hpp:391`) means it is waiting for dispatcher page credits. Prefetcher at `HQW` (`cq_prefetch.cpp:847`) means it is waiting for the host fetch queue, which only fits the copy path. Dispatcher parked in `wait_for_available_data_and_release_old_pages` inside `process_write_paged` with the prefetcher idle means the relay delivered fewer bytes than the paged write expects.
4. Kernel log. `sudo dmesg | grep -i 'tenstorrent\|AMD-Vi\|IO_PAGE_FAULT\|pin_user_pages'`. A `pin_user_pages_longterm failed` line means the pin failed and the fallback copy path was taken; an `IO_PAGE_FAULT` means the device read an unmapped IOVA.

Controlled contrasts that split the two candidates, each a single short script:

- `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` with the unchanged probe. Pass confirms the pinned path as the carrier and gives the workaround.
- A 64-byte-aligned anonymous source above 32 MiB. Open a unit mesh (`ttnn.open_mesh_device(ttnn.MeshShape(1, 1))`) and call `ttnn.from_torch(torch_bf16_row_major, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=unit_mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)` with a tensor larger than 32 MiB. With a device given, a single device, and no dtype or layout change, `from_torch` borrows the torch storage instead of copying it (`/home/hous/dev/clef/tt-metal/ttnn/core/tensor/py_to_tt_tensor.cpp:302-317`), and PyTorch's CPU allocator aligns that storage to 64 bytes, so the upload takes the no-prefix pinned branch from anonymous memory. The host-only `from_torch` (no device) always copies into a vector, so it cannot produce this contrast. A hang implicates 5.2; a pass implicates 5.1.
- The same cache file copied to `/dev/shm` and reloaded. A pass implicates the ext4 page cache specifically.

## 7. Other potential issues

- `MeshDevice cq ID 0 is in use by parent mesh ID 0 during close of mesh ID 1` (`/home/hous/dev/clef/tt-metal/tt_metal/distributed/mesh_device.cpp:1036-1047`) in `stage1_cache_reload_probe.log` is unrelated to the spin. `FDMeshCommandQueue::in_use_` is set by every write, read, workload, and event record (`fd_mesh_command_queue.cpp:410, 641, 680, 728, 847, 876, 974, 1040, 1226`) and a successful `finish_nolock` does not clear it; only `finish_and_reset_in_use` (lines 1824-1842) and the exception path at line 790 do. Closing a submesh after both it and its parent have been used therefore throws unless the flags were reset first, regardless of close order. A script that writes through the parent and a submesh must expect this at teardown.
- `/proc/meminfo` on this host reports `CmaTotal: 0 kB` with `CmaFree: 18952192 kB` and `/proc/vmstat` `nr_free_cma 4738048`. These disagree; on kernel 7.0.0-34 the CMA counters may not mean what older kernels meant. If a CMA area does exist, long-term pins of file pages that landed in it require page migration, which widens the kernel-side explanation in 5.1.

## 8. Reference values

| Item | Value | Source |
| --- | --- | --- |
| Pinned write threshold | 32 MiB of local shard bytes | `tensor_apis.cpp:39` |
| Pinned cache limit default | 4 GiB | `rtoptions.hpp:280` |
| Blackhole L1 read alignment | 16 B | `noc_parameters.h:375` |
| Blackhole DRAM and PCIe alignment | 64 B | `noc_parameters.h:377-379` |
| Serialized shard alignment | 64 B | `tensor_file_layout.hpp:38` |
| Dispatch buffer page | 4 KiB | `dispatch_settings.hpp:94` |
| KMD read-only pinning minimum | 2.9.0; installed 2.10.0 | `kmd_versions.hpp:33`, `/sys/module/tenstorrent/version` |
| Device DMA mask | 58 bits | `enumerate.c:338`, sysfs |
| IOMMU groups 14 to 17 | `DMA-FQ` | `/sys/kernel/iommu_groups/*/type` |
| Operation timeout default | 0 (none) | `rtoptions.hpp:389` |
