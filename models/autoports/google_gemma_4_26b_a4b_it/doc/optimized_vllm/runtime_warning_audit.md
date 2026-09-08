# Runtime warning audit: TP2 dense projection shard geometry

Source-only audit, 2026-09-08. No TTNN imports, pytest, device access, profiling, implementation edits, or log modifications. [The JSON evidence](runtime_warning_audit.json) records a bounded scan ending at each log's initial file size; the candidate server was still active.

## Verdict

The repeated matmul warning exposes an inaccurate requested output shard grid in the TP2 dense MLP path. The runtime computes and allocates the correct rank-local output geometry directly. The warning does **not** invoke a CPU fallback, device-to-host transfer, or corrective reshard. It is not evidence of a new adapter/cache/trace correctness regression.

This is still a real configuration inconsistency with substantial repeated host logging, not a reason to suppress runtime warnings globally. Downstream multiplication separately requests the larger intermediate grid, so changing the shared intermediate configuration would change more than the warning. Correcting only the projection's requested output contract is the smallest follow-up experiment.

## Matched log control

| P150x2 run | Matmul warnings | Legacy tilize warnings | Scope |
| --- | ---: | ---: | --- |
| [Cold before](../../readiness_vllm/P150x2/optimized_vllm/before/server.log) | 240 | 3 | Entire saved log |
| [Warmed before](../../readiness_vllm/P150x2/optimized_vllm/before_warmed/server.log.xz) | 480 | 3 | Entire saved log |
| [Warmed candidate](../../readiness_vllm/P150x2/optimized_vllm/after_warmed/server.log.xz) | 480 | 3 | Prefix before first `POST /v1/chat/completions`, line 1344 |
| Same candidate | 97,860 | 3 | Snapshot of 92,998,717 bytes / 102,266 lines at 21:09:54 UTC, including later gates |

Every matmul warning in each scanned scope has the same normalized message: computed WIDTH_SHARDED/L1, grid `(0,0)..(10,0)`, versus provided WIDTH_SHARDED/L1, grid `(0,0)..(10,1)`; both use `[32,96]` shards and row-major orientation. The candidate's first example is line 786. Thus the **matched benchmark scope** has no increase in these warnings. Comparing the candidate's full correctness/sampling log with a benchmark-only baseline would compare different work.

The growth after that prefix is compatible with ordinary eager decode in the host compatibility path. [Generator `decode_forward`](../../tt/generator.py) lines 612–618 deliberately forces `enable_trace=False` for host sampling. Each eager model invocation dispatches the dense projections again; trace replay itself does not rerun their Python/C++ validation. This is a source explanation of repetition, not a timing measurement or a proof that every later warning belongs to one particular test.

## Why these exact dimensions occur

1. [Functional decoder constants](../../tt/functional_decoder.py) define hidden size 2816, dense MLP intermediate size 2112, and tile size 32.
2. [Multichip profile setup](../../tt/multichip_decoder.py) lines 223–238 pads the intermediate to `TP * 32` and divides by TP. At TP2 it remains 2112 globally and becomes **1056 per rank**, or 33 tiles. Lines 651–658 upload gate/up weights sharded on output width; down weights are sharded on input width.
3. [Inherited residual setup](../../tt/optimized_decoder.py) lines 1017–1040 builds the intermediate config from global `MLP_INTERMEDIATE_SIZE` on the 22-core residual grid. The requested shard width is `2112 / 22 = 96`, and `_width_sharded_linear_program_config` lines 736–756 sets `per_core_M=1`, `per_core_N=3` for gate/up.
4. [Multichip `_dense_mlp`](../../tt/multichip_decoder.py) lines 1622–1628 uses the coherent sharded path when enabled; TP2 enables it by default at lines 960–963. [The inherited gate and up calls](../../tt/optimized_decoder.py) lines 2763–2778 pass those global-width configs with the rank-local weights. The single-row candidate helper uses the same fallback projection contract at lines 2805–2822 when those roles are not DRAM candidates.
5. [Matmul output-spec construction](/home/hous/dev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2450) uses the actual weight width. With one tiled M block and N=33 tiles, it computes `ceil(33 / 3) = 11` output cores and `[1*32, 3*32] = [32,96]` shards. Eleven row-major cores on this worker grid are exactly `(0,0)..(10,0)`.

This derivation matches all fields of the warning. No runtime call stack was collected, so individual warning-to-layer attribution remains source-derived; the eligible operations are the dense gate/up projections across the 30-layer stack.

The printed computed config also has an auto-populated `nd_shard_spec`, while the provided config prints `nullopt`. That is not the cause: both report `created_with_nd_shard_spec=0`, and [MemoryConfig equality](/home/hous/dev/tt-metal/tt_metal/impl/tensor/spec/memory_config/memory_config.cpp:112) compares their authoritative legacy shard specs in that case. The differing grid is a semantic difference, not just a formatting difference.

## What the runtime does, and does not do

[`validate_matmul_optional_tensors`](/home/hous/dev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:163) first obtains the computed spec. Lines 218–239 require compatible memory layout and buffer type, then log a differing explicit shard spec. The branch only logs. [Output allocation](/home/hous/dev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2704) calls `compute_output_specs` and `create_device_tensor` with the computed spec. It does not allocate the requested 22-core matmul output and then copy it into 11 cores. The selected operation remains the configured device matmul.

The warning is not a count of program-cache misses. [The generic cache-hit handler](/home/hous/dev/tt-metal/ttnn/api/ttnn/device_operation.hpp:261) calls `validate_on_program_cache_miss` when an operation lacks a separate cache-hit validator. [Matmul's declaration](/home/hous/dev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.hpp:18) lacks that separate method, so eager cache hits can emit the same warning repeatedly. Formatting and writing almost 98,000 messages is real host work; its latency contribution was not measured here.

The consumer deserves separate attention: [the GELU/multiply and down projection](../../tt/optimized_decoder.py) lines 2779–2792 explicitly request the shared 22-core intermediate config, then produce the 22-core residual output. [Binary output construction](/home/hous/dev/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:518) respects an explicit shard spec and inherits input shards only when no spec was supplied. Thus the whole dense chain cannot be described as an unchanged 11-core chain merely because matmul overrides its own output. There is no extra `to_memory_config` call in this three-operation path, but a smaller hidden-output allocation would be a separate geometry change requiring validation.

## The three tilize messages

[Tilize output-spec construction](/home/hous/dev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_device_operation.cpp:242) emits this warning whenever the legacy optimized sharded factory is applicable. It does not test whether the requested and inherited shard specs actually differ. It keeps the input shard grid, the requested buffer type, and logical shape while producing TILE layout.

The eligibility helper at lines 41–145 requires compatible sharded layouts and L1 buffers; the factory selector at lines 317–322 then chooses `TilizeMultiCoreShardedProgramFactory`. This warning announces selection of the optimized sharded tilizer, not its fallback path. Tilization itself is real layout-conversion work. All three messages occur during startup before the matmul warnings and have identical counts in the controls. [Model setup](../../tt/model.py) lines 305–331 constructs three persistent sharded collective buffers, which is a plausible source of that startup triplet, but exact call-site attribution is not established by these logs. Nothing in the warning alone demonstrates an incorrect conversion or a serving regression.

## Actionable follow-up and stage implications

The audit does not reveal a broken serving input/output contract that requires interrupting the current server. It does establish a model output-spec cleanup opportunity. Keep the completed benchmark-prefix control and the ongoing functional gates as separate evidence; do not claim the warnings are cost-free or infer a performance change from their totals.

For a subsequent isolated candidate, give only the affected gate/up matmuls an output request consistent with the runtime's existing computed grid. Derive it from rank-local N and the actual program `per_core_M/N`, or use a WIDTH_SHARDED/L1 request without an explicit shard spec where the non-gather program contract allows the runtime to choose that spec. Keep the existing multiplication/down configuration and program geometry unchanged for the first experiment. This corrects the inaccurate model request without disabling the validator or global logging. Do not change all TP profiles to a hard-coded 11-core grid.

Validate returned gate/up logical shapes and actual configs against the current control, exact logits/tokens for TP2 B1 and padded B32, repeated eager host-compatible decode, and retained trace replay. The focused prediction is identical materialized matmul output specs and numerics with this warning absent. Any proposal to shrink the multiplication/down intermediate grid is a second experiment because it changes actual allocation/consumer geometry. No implementation or device experiment was performed by this audit.

Large server/watcher logs are stored losslessly as `.log.xz`; use `xz -dc` to
read the original text and line numbers. [Archive index](artifact_compression.json)
records original paths, byte counts and uncompressed SHA-256. Embedded runner
paths and historical line references name those original decompressed logs.
