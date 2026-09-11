set(TTNN_OP_EXPERIMENTAL_KDA_RECURRENT_GATED_DELTA_RULE_API_HEADERS recurrent_gated_delta_rule.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_RECURRENT_GATED_DELTA_RULE_SRCS
    recurrent_gated_delta_rule.cpp
    device/recurrent_gated_delta_rule_device_operation.cpp
    device/recurrent_gated_delta_rule_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_RECURRENT_GATED_DELTA_RULE_NANOBIND_SRCS
    recurrent_gated_delta_rule_nanobind.cpp
    ../kda_nanobind.cpp
)
