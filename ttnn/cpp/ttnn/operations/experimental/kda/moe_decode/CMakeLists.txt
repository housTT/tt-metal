include(sources.cmake)

add_library(ttnn_op_experimental_kda_moe_decode ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::KDA::MoeDecode ALIAS ttnn_op_experimental_kda_moe_decode)

tt_reuse_precompile_headers(ttnn_op_experimental_kda_moe_decode TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_kda_moe_decode)

file(GLOB_RECURSE kernels device/kernels/*.cpp)
target_sources(
    ttnn_op_experimental_kda_moe_decode
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_KDA_MOE_DECODE_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_KDA_MOE_DECODE_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_kda_moe_decode
    PRIVATE
        TT::Metalium
        TTNN::Ops::Experimental::KDA::Common
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_kda_moe_decode
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/kda/moe_decode
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_experimental_kda_moe_decode FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_KDA_MOE_DECODE_NANOBIND_SRCS})
endif()
