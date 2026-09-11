set(TTNN_OP_EXPERIMENTAL_KDA_HC_MIXER_API_HEADERS hc_mixer.hpp)
set(TTNN_OP_EXPERIMENTAL_KDA_HC_MIXER_SRCS
    hc_mixer.cpp
    device/hc_mix_post_device_operation.cpp
    device/hc_mix_post_program_factory.cpp
    device/hc_inject_device_operation.cpp
    device/hc_inject_program_factory.cpp
)
set(TTNN_OP_EXPERIMENTAL_KDA_HC_MIXER_NANOBIND_SRCS hc_mixer_nanobind.cpp)
