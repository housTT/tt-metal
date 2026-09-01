// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<1>();

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile = get_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_output);
    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    cb_wait_front(cb_output, 1);
    noc_async_write_tile(tile, output_accessor, get_read_ptr(cb_output));
    noc_async_write_barrier();
    cb_pop_front(cb_output, 1);
}
