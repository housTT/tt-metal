#!/usr/bin/env bash
# Source this file before every Qwen3.8-Flash-Next functional-decoder run.
# It deliberately pins Python, native TTNN, and device JIT sources to one checkout.

if [[ -n "${ZSH_VERSION:-}" ]]; then
    _qwen38_self="${(%):-%N}"
else
    _qwen38_self="${BASH_SOURCE[0]}"
fi
_qwen38_repo="$(cd "$(dirname "${_qwen38_self}")/../../../../.." && pwd)"

export TT_METAL_HOME="${_qwen38_repo}"
export TT_METAL_RUNTIME_ROOT="${_qwen38_repo}"
export PYTHONPATH="${_qwen38_repo}"
export TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-0}"
export TT_MESH_GRAPH_DESC_PATH="${_qwen38_repo}/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
unset TT_METAL_KERNEL_PATH

if [[ "${VIRTUAL_ENV:-}" != "${_qwen38_repo}/python_env" ]]; then
    # shellcheck disable=SC1091
    source "${_qwen38_repo}/python_env/bin/activate"
fi

unset _qwen38_repo _qwen38_self
