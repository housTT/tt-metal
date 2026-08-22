// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_local_dispatch_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe {

namespace {

constexpr uint32_t CB_PLAN_MAPPING = tt::CBIndex::c_0;
constexpr uint32_t CB_PLAN_INDICES = tt::CBIndex::c_1;
constexpr uint32_t CB_PLAN_COUNTS = tt::CBIndex::c_2;
constexpr uint32_t CB_PLAN_OFFSETS = tt::CBIndex::c_3;
constexpr uint32_t CB_PLAN_LOCAL_TO_GLOBAL = tt::CBIndex::c_4;
constexpr uint32_t CB_PLAN_ASSIGNMENTS = tt::CBIndex::c_5;
constexpr uint32_t CB_PLAN_INVERSE = tt::CBIndex::c_6;
constexpr uint32_t CB_PLAN_VALID = tt::CBIndex::c_7;
constexpr uint32_t CB_DISPATCH_ASSIGNMENTS = tt::CBIndex::c_8;
constexpr uint32_t CB_DISPATCH_ROW = tt::CBIndex::c_9;
constexpr uint32_t CB_DISPATCH_ZERO = tt::CBIndex::c_10;

uint32_t aligned_page_size(const ttnn::Tensor& tensor) {
    return static_cast<uint32_t>(tensor.buffer()->aligned_page_size());
}

void create_cb(
    tt::tt_metal::Program& program,
    const CoreRangeSet& cores,
    uint32_t cb_id,
    uint32_t size,
    uint32_t page_size,
    tt::DataFormat data_format) {
    auto config = tt::tt_metal::CircularBufferConfig(size, {{cb_id, data_format}}).set_page_size(cb_id, page_size);
    tt::tt_metal::CreateCircularBuffer(program, cores, config);
}

std::vector<uint32_t> planner_runtime_args(
    const TopkLocalDispatchParams& op,
    const TopkLocalDispatchInputs& tensors,
    const TopkLocalDispatchTensors& outputs,
    uint32_t ready_semaphore,
    const std::vector<CoreCoord>& dispatch_cores) {
    std::vector<uint32_t> args = {
        tensors.topk_indices.buffer()->address(),
        tensors.global_to_local_expert.buffer()->address(),
        outputs[1].buffer()->address(),
        outputs[2].buffer()->address(),
        outputs[3].buffer()->address(),
        outputs[4].buffer()->address(),
        outputs[5].buffer()->address(),
        outputs[6].buffer()->address(),
        op.valid_tokens,
        ready_semaphore,
        static_cast<uint32_t>(dispatch_cores.size()),
    };
    const auto* device = tensors.x.device();
    args.reserve(args.size() + 2 * dispatch_cores.size());
    for (const auto& logical_core : dispatch_cores) {
        const auto noc_core = device->worker_core_from_logical_core(logical_core);
        args.push_back(noc_core.x);
        args.push_back(noc_core.y);
    }
    return args;
}

std::vector<uint32_t> dispatch_runtime_args(
    const TopkLocalDispatchParams& op,
    const TopkLocalDispatchInputs& tensors,
    const TopkLocalDispatchTensors& outputs,
    uint32_t row_start,
    uint32_t row_count,
    uint32_t ready_semaphore) {
    return {
        tensors.x.buffer()->address(),
        outputs[4].buffer()->address(),
        outputs[0].buffer()->address(),
        op.valid_tokens,
        row_start,
        row_count,
        ready_semaphore,
    };
}

}  // namespace

TopkLocalDispatchProgramFactory::cached_program_t TopkLocalDispatchProgramFactory::create(
    const TopkLocalDispatchParams& op, const TopkLocalDispatchInputs& tensors, TopkLocalDispatchTensors& outputs) {
    tt::tt_metal::Program program;

    const uint32_t tokens = tensors.x.logical_shape()[-2];
    const uint32_t topk = tensors.topk_indices.logical_shape()[-1];
    const uint32_t num_global_experts = tensors.global_to_local_expert.logical_shape()[-1];
    // packed_assignment_ids always retains the exact expert-region capacity.
    // Planner-only production mode deliberately gives output[0] a one-row
    // sentinel so deriving capacity from compact_x would truncate the plan.
    const uint32_t capacity = outputs[4].logical_shape()[-1];
    const uint32_t slots = tokens * topk;

    auto* device = tensors.x.device();
    const auto grid = device->compute_with_storage_grid_size();
    const CoreRangeSet all_cores(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
    const uint32_t num_cores = all_cores.num_cores();
    auto worker_cores = tt::tt_metal::corerange_to_cores(all_cores, num_cores, true);
    TT_FATAL(!worker_cores.empty(), "topk_local_dispatch requires at least one worker core");
    const CoreCoord planner_core = worker_cores.front();
    const CoreRangeSet planner_core_set{CoreRange(planner_core)};

    uint64_t planner_cb_bytes =
        aligned_page_size(tensors.global_to_local_expert) + aligned_page_size(tensors.topk_indices);
    for (uint32_t output_index = 1; output_index < TOPK_LOCAL_DISPATCH_OUTPUTS; ++output_index) {
        planner_cb_bytes += aligned_page_size(outputs[output_index]);
    }
    const uint32_t l1_reserved = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    constexpr uint32_t l1_margin = 32 * 1024;
    TT_FATAL(
        device->l1_size_per_core() > l1_reserved + l1_margin &&
            planner_cb_bytes <= device->l1_size_per_core() - l1_reserved - l1_margin,
        "top-k planner CB footprint ({} bytes) exceeds per-core L1 budget (size {}, reserved {}, margin {})",
        planner_cb_bytes,
        device->l1_size_per_core(),
        l1_reserved,
        l1_margin);

    const auto uint32_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);

    // Planner-only L1 arena. The largest buffers are compact integer maps, not
    // activations. Device-op validation and the budget guard above admit only
    // token spans whose complete planner footprint fits on the planner core.
    create_cb(
        program,
        planner_core_set,
        CB_PLAN_MAPPING,
        aligned_page_size(tensors.global_to_local_expert),
        aligned_page_size(tensors.global_to_local_expert),
        uint32_format);
    create_cb(
        program,
        planner_core_set,
        CB_PLAN_INDICES,
        aligned_page_size(tensors.topk_indices),
        aligned_page_size(tensors.topk_indices),
        uint32_format);
    for (const auto& [cb_id, output_index] : std::initializer_list<std::pair<uint32_t, uint32_t>>{
             {CB_PLAN_COUNTS, 1},
             {CB_PLAN_OFFSETS, 2},
             {CB_PLAN_LOCAL_TO_GLOBAL, 3},
             {CB_PLAN_ASSIGNMENTS, 4},
             {CB_PLAN_INVERSE, 5},
             {CB_PLAN_VALID, 6}}) {
        create_cb(
            program,
            planner_core_set,
            cb_id,
            aligned_page_size(outputs[output_index]),
            aligned_page_size(outputs[output_index]),
            uint32_format);
    }

    // No activation-sized output or dispatch-worker L1 is created in the
    // production planner-only mode. The planner's semaphore fanout is empty,
    // so it completes after publishing the integer plan.
    const uint32_t ready_semaphore = op.materialize_x ? tt::tt_metal::CreateSemaphore(program, all_cores, 0) : 0;
    const std::vector<CoreCoord> no_dispatch_cores;
    const auto& planner_signal_cores = op.materialize_x ? worker_cores : no_dispatch_cores;

    std::vector<uint32_t> planner_compile_args = {
        CB_PLAN_MAPPING,
        CB_PLAN_INDICES,
        CB_PLAN_COUNTS,
        CB_PLAN_OFFSETS,
        CB_PLAN_LOCAL_TO_GLOBAL,
        CB_PLAN_ASSIGNMENTS,
        CB_PLAN_INVERSE,
        CB_PLAN_VALID,
        aligned_page_size(tensors.global_to_local_expert),
        aligned_page_size(tensors.topk_indices),
        aligned_page_size(outputs[1]),
        aligned_page_size(outputs[2]),
        aligned_page_size(outputs[3]),
        aligned_page_size(outputs[4]),
        aligned_page_size(outputs[5]),
        aligned_page_size(outputs[6]),
        tokens,
        topk,
        num_global_experts,
        op.num_local_experts,
        capacity,
        slots,
    };
    tt::tt_metal::TensorAccessorArgs(tensors.topk_indices.buffer()).append_to(planner_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensors.global_to_local_expert.buffer()).append_to(planner_compile_args);
    for (uint32_t output_index = 1; output_index < TOPK_LOCAL_DISPATCH_OUTPUTS; ++output_index) {
        tt::tt_metal::TensorAccessorArgs(outputs[output_index].buffer()).append_to(planner_compile_args);
    }

    const auto planner_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/topk_routed_expert_moe/device/kernels/dataflow/"
        "topk_local_plan.cpp",
        planner_core_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = planner_compile_args});

    tt::tt_metal::SetRuntimeArgs(
        program,
        planner_kernel,
        planner_core,
        planner_runtime_args(op, tensors, outputs, ready_semaphore, planner_signal_cores));

    tt::tt_metal::KernelHandle dispatch_kernel = 0;
    if (op.materialize_x) {
        // A DRAM NOC read requires the source and L1 destination to have the
        // same low alignment bits. Both tensor pages and CB bases are DRAM
        // aligned, so keep every active assignment slice start aligned too.
        // An even row split (for example, six UINT32 rows per core) would make
        // later workers read from 24-byte-offset DRAM addresses into an
        // aligned CB and silently alias an earlier assignment range on BH.
        const uint32_t dram_alignment_bytes = tt::tt_metal::hal::get_dram_alignment();
        TT_FATAL(
            dram_alignment_bytes % sizeof(uint32_t) == 0,
            "DRAM alignment ({}) must be divisible by the assignment element size ({})",
            dram_alignment_bytes,
            sizeof(uint32_t));
        const uint32_t rows_per_alignment = dram_alignment_bytes / sizeof(uint32_t);
        TT_FATAL(rows_per_alignment > 0, "DRAM alignment must cover at least one UINT32 assignment");
        const uint32_t target_rows_per_core = (capacity + num_cores - 1) / num_cores;
        const uint32_t rows_per_core =
            ((target_rows_per_core + rows_per_alignment - 1) / rows_per_alignment) * rows_per_alignment;
        const uint32_t assignment_chunk_bytes = rows_per_core * sizeof(uint32_t);
        const uint32_t x_page_size = aligned_page_size(tensors.x);
        const uint32_t output_page_size = aligned_page_size(outputs[0]);
        TT_FATAL(
            x_page_size == output_page_size,
            "x/output ROW_MAJOR page sizes must match ({} vs {})",
            x_page_size,
            output_page_size);
        const auto bf16_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::BFLOAT16);

        // Bring-up workers hold only their contiguous assignment slice plus
        // one activation row and one reusable zero row.
        create_cb(
            program, all_cores, CB_DISPATCH_ASSIGNMENTS, assignment_chunk_bytes, assignment_chunk_bytes, uint32_format);
        create_cb(program, all_cores, CB_DISPATCH_ROW, x_page_size, x_page_size, bf16_format);
        create_cb(program, all_cores, CB_DISPATCH_ZERO, output_page_size, output_page_size, bf16_format);

        std::vector<uint32_t> dispatch_compile_args = {
            CB_DISPATCH_ASSIGNMENTS,
            CB_DISPATCH_ROW,
            CB_DISPATCH_ZERO,
            assignment_chunk_bytes,
            x_page_size,
            output_page_size,
            topk,
        };
        tt::tt_metal::TensorAccessorArgs(tensors.x.buffer()).append_to(dispatch_compile_args);
        tt::tt_metal::TensorAccessorArgs(outputs[4].buffer()).append_to(dispatch_compile_args);
        tt::tt_metal::TensorAccessorArgs(outputs[0].buffer()).append_to(dispatch_compile_args);

        dispatch_kernel = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/topk_routed_expert_moe/device/kernels/"
            "dataflow/topk_local_dispatch.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = dispatch_compile_args});

        uint32_t row_start = 0;
        for (uint32_t index = 0; index < num_cores; ++index) {
            const uint32_t rows_remaining = capacity - row_start;
            const uint32_t row_count = std::min(rows_per_core, rows_remaining);
            tt::tt_metal::SetRuntimeArgs(
                program,
                dispatch_kernel,
                worker_cores[index],
                dispatch_runtime_args(op, tensors, outputs, row_start, row_count, ready_semaphore));
            row_start += row_count;
        }
        TT_FATAL(row_start == capacity, "internal dispatch split covered {} rows, expected {}", row_start, capacity);
    }

    return cached_program_t{
        std::move(program),
        TopkLocalDispatchSharedVariables{
            .planner_kernel = planner_kernel,
            .dispatch_kernel = dispatch_kernel,
            .dispatch_cores = op.materialize_x ? std::move(worker_cores) : std::vector<CoreCoord>{planner_core},
            .materialize_x = op.materialize_x,
        }};
}

void TopkLocalDispatchProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TopkLocalDispatchParams& op,
    const TopkLocalDispatchInputs& tensors,
    TopkLocalDispatchTensors& outputs) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    const CoreCoord planner_core = shared.dispatch_cores.front();

    auto& planner_args = GetRuntimeArgs(program, shared.planner_kernel, planner_core);
    planner_args[0] = tensors.topk_indices.buffer()->address();
    planner_args[1] = tensors.global_to_local_expert.buffer()->address();
    for (uint32_t output_index = 1; output_index < TOPK_LOCAL_DISPATCH_OUTPUTS; ++output_index) {
        planner_args[1 + output_index] = outputs[output_index].buffer()->address();
    }
    planner_args[8] = op.valid_tokens;

    if (shared.materialize_x) {
        for (const auto& core : shared.dispatch_cores) {
            auto& dispatch_args = GetRuntimeArgs(program, shared.dispatch_kernel, core);
            dispatch_args[0] = tensors.x.buffer()->address();
            dispatch_args[1] = outputs[4].buffer()->address();
            dispatch_args[2] = outputs[0].buffer()->address();
            dispatch_args[3] = op.valid_tokens;
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::topk_routed_expert_moe
