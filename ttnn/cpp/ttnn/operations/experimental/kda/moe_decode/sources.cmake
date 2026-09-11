set(TTNN_OP_EXPERIMENTAL_KDA_MOE_DECODE_API_HEADERS moe_decode.hpp)
set(TTNN_OP_EXPERIMENTAL_KDA_MOE_DECODE_SRCS
    moe_decode.cpp
    device/moe_route_topk_device_operation.cpp
    device/moe_route_topk_program_factory.cpp
    device/moe_weighted_sum_device_operation.cpp
    device/moe_weighted_sum_program_factory.cpp
    device/moe_swiglu_device_operation.cpp
    device/moe_swiglu_program_factory.cpp
    device/moe_sort_slabs_device_operation.cpp
    device/moe_sort_slabs_program_factory.cpp
)
set(TTNN_OP_EXPERIMENTAL_KDA_MOE_DECODE_NANOBIND_SRCS moe_decode_nanobind.cpp)
