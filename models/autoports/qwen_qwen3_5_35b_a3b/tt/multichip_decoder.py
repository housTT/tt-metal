# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy

from .functional_decoder import FunctionalDecoderConfig, _text_config
from .optimized_decoder import (
    DEFAULT_POLICIES,
    OptimizedDecoder,
    _Projection,
    _StackedDispatch,
    _bucket_sizes,
    _memory_bytes,
    _PREFIX_MATRICES,
)

MESH_POLICIES = {
    1: {"dispatch_weights": "stacked"},
    2: {
        "dispatch_weights": "expert_parallel",
        "prefill_moe_pieces": 2,
        "sparse_gate_cores": 16,
        "sparse_gate_k": 64,
        "sparse_down_cores": 64,
        "sparse_down_k": 8,
        "shared_decode_minimal_config": {"grid": [8, 2], "M_block_size": 1, "K_block_size": 16, "N_block_size": 4,
                                         "subblock_h": 1, "subblock_w": 2},
        "projections": {"shared_down": {"dram": False, "decode_grid": [8, 2], "k_block": 8, "per_n": 4}},
        "projections_by_kind": {
            "full_attention": {"attention_in": {"decode_grid": [8, 2], "k_block": 8, "per_n": 9}},
        },
    },
    4: {
        "dispatch_weights": "expert_parallel",
        "prefill_moe_pieces": 2,
        "sparse_gate_cores": 8,
        "sparse_gate_k": 64,
        "sparse_down_cores": 64,
        "sparse_down_k": 4,
        "shared_decode_minimal_config": {"grid": [4, 2], "M_block_size": 1, "K_block_size": 16, "N_block_size": 4,
                                         "subblock_h": 1, "subblock_w": 2},
        "projections": {"shared_down": {"dram": False, "decode_grid": [8, 1], "k_block": 4, "per_n": 8}},
        "projections_by_kind": {
            "full_attention": {"attention_in": {"decode_grid": [8, 5], "k_block": 8, "per_n": 2}},
            "linear_attention": {"attention_in": {"decode_grid": [10, 2], "k_block": 2, "per_n": 5}},
        },
    },
}

CCL_LINKS = 2
_ROW_INDICES = {}


def _construction_sample(mesh, label):
    host = {}
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith(("VmRSS:", "VmHWM:")):
                host[line.split(":")[0]] = int(line.split()[1]) * 1024
    return dict(step=label, device_dram_allocated_bytes=_memory_bytes(mesh)["dram"]["allocated_bytes"],
                host_rss_bytes=host.get("VmRSS"), host_peak_rss_bytes=host.get("VmHWM"))


def mesh_policy(policy, tp, kind=None):
    result = copy.deepcopy(dict(policy))
    override = copy.deepcopy(MESH_POLICIES[tp])
    projections = override.pop("projections", {})
    by_kind = override.pop("projections_by_kind", {})
    for role, options in by_kind.get(kind, {}).items():
        projections[role] = {**projections.get(role, {}), **options}
    result.update(override)
    merged = copy.deepcopy(result.get("projections", {}))
    for role, options in projections.items():
        merged[role] = {**merged.get(role, {}), **options}
    result["projections"] = merged
    return result


def local_hf_config(hf_config, tp):
    local = copy.deepcopy(hf_config)
    text = _text_config(local)
    heads, kv_heads = text.num_attention_heads, text.num_key_value_heads
    if heads % tp or (kv_heads % tp and tp % kv_heads):
        raise ValueError(f"{heads} query and {kv_heads} KV heads do not split over {tp} devices")
    for name in ("linear_num_key_heads", "linear_num_value_heads", "moe_intermediate_size",
                 "shared_expert_intermediate_size", "num_experts"):
        if getattr(text, name) % tp:
            raise ValueError(f"{name} does not split over {tp} devices")
    text.num_attention_heads = heads // tp
    text.num_key_value_heads = max(1, kv_heads // tp)
    text.linear_num_key_heads //= tp
    text.linear_num_value_heads //= tp
    text.moe_intermediate_size //= tp
    text.shared_expert_intermediate_size //= tp
    return local


def remapped_layer_state(state_dict, layer_idx):
    if f"layers.{layer_idx}.input_layernorm.weight" not in state_dict:
        from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

        state_dict = remap_qwen36_state_dict(dict(state_dict))
    prefix = f"layers.{layer_idx}."
    return {key: value for key, value in state_dict.items() if key.startswith(prefix)}


def tensor_parallel_state(state, layer_idx, tp, rank, config):
    import torch

    def part(value, dim, size):
        return value.narrow(dim, rank * size, size)

    heads, head_dim, hidden = config.n_heads // tp, config.head_dim, config.dim
    kv_heads = max(1, config.n_kv_heads // tp)
    kv_start = rank * config.n_kv_heads // tp
    qk, v = config.linear_q_dim // tp, config.linear_v_dim // tp
    value_heads = config.linear_num_value_heads // tp
    width = config.moe_intermediate_size // tp
    full_width = config.moe_intermediate_size
    shared = config.moe_shared_intermediate_size // tp
    replicated = {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
                  "self_attn.k_norm.weight", "linear_attn.norm.weight", "mlp.gate.weight",
                  "mlp.shared_expert_gate.weight"}
    rows = {"linear_attn.in_proj_a.weight": value_heads, "linear_attn.in_proj_b.weight": value_heads,
            "linear_attn.in_proj_z.weight": v, "linear_attn.dt_bias": value_heads, "linear_attn.A_log": value_heads,
            "linear_attn.q_conv.weight": qk, "linear_attn.k_conv.weight": qk, "linear_attn.v_conv.weight": v,
            "mlp.shared_expert.gate_proj.weight": shared, "mlp.shared_expert.up_proj.weight": shared}
    columns = {"linear_attn.out_proj.weight": v, "self_attn.o_proj.weight": heads * head_dim,
               "mlp.shared_expert.down_proj.weight": shared}
    prefix = f"layers.{layer_idx}."
    result = {}
    for key, value in state.items():
        name = key.removeprefix(prefix)
        if name in replicated:
            local = value
        elif name in rows:
            local = part(value, 0, rows[name])
        elif name in columns:
            local = part(value, 1, columns[name])
        elif name == "self_attn.q_proj.weight":
            local = part(value.reshape(config.n_heads, 2 * head_dim, hidden), 0, heads).reshape(-1, hidden)
        elif name in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            local = value.narrow(0, kv_start * head_dim, kv_heads * head_dim)
        elif name == "linear_attn.qkv_proj.weight":
            q_rows, k_rows, v_rows = value.split((config.linear_q_dim, config.linear_k_dim, config.linear_v_dim))
            local = torch.cat((part(q_rows, 0, qk), part(k_rows, 0, qk), part(v_rows, 0, v)))
        elif name == "mlp.experts.gate_up_proj":
            gate, up = value.split((full_width, full_width), dim=1)
            local = torch.cat((part(gate, 1, width), part(up, 1, width)), dim=1)
        elif name == "mlp.experts.down_proj":
            local = part(value, 2, width)
        else:
            raise KeyError(f"No tensor-parallel rule for {key}")
        result[key] = local.contiguous()
    return result


def expert_parallel_state(state, layer_idx, tp, rank, config):
    experts = config.moe_num_experts // tp
    prefix = f"layers.{layer_idx}.mlp.experts."
    gate_up = state[prefix + "gate_up_proj"].narrow(0, rank * experts, experts)
    down = state[prefix + "down_proj"].narrow(0, rank * experts, experts)
    return gate_up, down, rank * experts


class _ExpertParallelDispatch(_StackedDispatch):

    def __init__(self, gate_up, down, mesh, policy, offset):
        import torch
        import ttnn

        if (policy.dispatch_rank != "matmul" or policy.dispatch_ffn != "fused" or policy.dispatch_weight_nd
                or policy.dispatch_weights != "expert_parallel"):
            raise ValueError("Expert-parallel dispatch implements the stacked matmul-rank fused-FFN policy")
        self.policy = policy
        self.local_experts, width = down.shape[0], down.shape[-1]

        def put(x, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(x.contiguous(), device=mesh, dtype=dtype, layout=layout,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        gate, up = gate_up[:, :width].transpose(1, 2), gate_up[:, width:].transpose(1, 2)
        self.bind_weights(put(torch.cat((up, gate), dim=-1).unsqueeze(0), ttnn.bfloat4_b),
                          put(down.transpose(1, 2).unsqueeze(0), ttnn.bfloat8_b))
        self.expert_ids = put(torch.arange(self.local_experts, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.offset = put(torch.full((1, 1, 1, 1), float(offset)), ttnn.float32)
        self.zero_row = put(torch.zeros(1, 1, 1, 2048), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
                                                              fp32_dest_acc_en=False, packer_l1_acc=False,
                                                              math_approx_mode=False)
        self.valid_rows = None
        self.prepare_maps(mesh)

    def prepare_maps(self, mesh):
        import torch
        import ttnn

        policy = self.policy
        local = self.local_experts
        width = local + 32
        self.prefix_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False,
            math_approx_mode=False)
        self.region_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False,
            math_approx_mode=False)

        def put(x, dtype, layout):
            return ttnn.from_torch(x, device=mesh, dtype=dtype, layout=layout,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        self.region_matrix = put(torch.ones((width, width)).triu(1).reshape(1, 1, width, width), ttnn.bfloat16,
                                 ttnn.TILE_LAYOUT)
        self.buffers, self.rank_buffers = {}, {}
        for length in sorted(set(_bucket_sizes(policy)) | {128, 512}):
            routes = length * 8
            capacity = ((routes + local * 31 + 32 + 31) // 32) * 32
            if capacity >= 65536:
                raise ValueError("Internal local-dispatch chunk must fit UINT16 route maps")
            self.buffers[length] = (None, put(torch.zeros((1, 1, 1, capacity), dtype=torch.int32), ttnn.uint32,
                                              ttnn.ROW_MAJOR_LAYOUT), None, capacity)
            key = (mesh, length)
            self.mesh_key = mesh
            if key not in _PREFIX_MATRICES:
                _PREFIX_MATRICES[key] = put(torch.ones((length, length)).tril(-1).reshape(1, 1, length, length),
                                            ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
            tokens = (torch.arange(routes, dtype=torch.int32) % length if self.tiles(length)
                      else torch.arange(routes, dtype=torch.int32) // 8)
            if key not in _ROW_INDICES:
                rows = torch.arange(length, dtype=torch.float32).reshape(1, 1, length, 1).expand(1, 1, length, 8)
                _ROW_INDICES[key] = put(rows.contiguous(), ttnn.float32, ttnn.TILE_LAYOUT)
            self.rank_buffers[length] = (
                put(torch.zeros((1, 1, length, width), dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                put(torch.ones((1, 1, length, 8), dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                put(tokens.reshape(1, 1, 1, routes), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                _PREFIX_MATRICES[key],
                put(torch.ones((1, 1, 1, length)), ttnn.bfloat16, ttnn.TILE_LAYOUT))

    def route_maps(self, routing):
        import ttnn

        values, ids = routing
        length = ids.shape[2]
        local = self.local_experts
        _, map_zeros, _, capacity = self.buffers[length]
        zeros, ones, tokens, prefix, count_row = self.rank_buffers[length]

        def rm_uint32(x):
            return ttnn.typecast(ttnn.to_layout(ttnn.typecast(x, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32)

        index = ttnn.typecast(ttnn.to_layout(ttnn.reshape(ids, (1, 1, length, 8)), ttnn.TILE_LAYOUT), ttnn.uint32)
        relative = ttnn.subtract(ttnn.typecast(index, ttnn.float32), self.offset)
        inside = ttnn.logical_and(ttnn.ge(relative, 0.0), ttnn.lt(relative, float(local)))
        if self.valid_rows is not None and self.valid_rows < length:
            inside = ttnn.logical_and(inside, ttnn.lt(_ROW_INDICES[(self.mesh_key, length)], float(self.valid_rows)))
        index = ttnn.typecast(ttnn.where(inside, relative, float(local)), ttnn.uint32)
        chosen = ttnn.scatter(zeros, dim=-1, index=index, src=ones)
        count = ttnn.matmul(count_row, chosen, dtype=ttnn.float32, compute_kernel_config=self.prefix_compute)
        counts = ttnn.reshape(rm_uint32(count[:, :, :, :local]), (local,))
        aligned = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(count, 31.0), 0.03125)), 32.0)
        tiles = ttnn.typecast(ttnn.multiply(aligned, 0.03125), ttnn.bfloat16)
        region_start = ttnn.multiply(ttnn.matmul(tiles, self.region_matrix, dtype=ttnn.float32,
                                                 compute_kernel_config=self.region_compute), 32.0)
        earlier = ttnn.matmul(prefix, chosen, dtype=ttnn.float32, compute_kernel_config=self.prefix_compute)
        slots = ttnn.gather(ttnn.add(earlier, region_start), dim=-1, index=index)
        slots = ttnn.where(inside, slots, float(capacity - 1))
        if self.tiles(length):
            slots = ttnn.transpose(slots, -2, -1)
        target = ttnn.reshape(rm_uint32(slots), (1, 1, 1, length * 8))
        dispatch_map = ttnn.scatter(map_zeros, dim=-1, index=target, src=tokens)
        offsets = ttnn.reshape(rm_uint32(region_start[:, :, :, :local]), (local,))
        return counts, offsets, dispatch_map, target, capacity

    def restore_table(self, result, capacity):
        import ttnn

        table = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(self.zero_row, table, (0, 0, capacity - 1, 0), (1, 1, capacity, 2048),
                                      (1, 1, 1, 1))
        return ttnn.reshape(table, (capacity, 2048))


_CCL_SEMAPHORES = {}


def _prefill_semaphores(mesh):
    import ttnn

    if mesh not in _CCL_SEMAPHORES:
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        make = lambda n: [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(n)]
        _CCL_SEMAPHORES[mesh] = [dict(barrier=make(2), rs=make(3), ag=make(2)) for _ in range(2)]
    return _CCL_SEMAPHORES[mesh]


class _AllReduceOutput:

    def __init__(self, module, topology, mesh):
        object.__setattr__(self, "reduced_module", module)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "semaphores", _prefill_semaphores(mesh))
        object.__setattr__(self, "semaphore_index", 0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "reduced_module"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "reduced_module"), name, value)

    def forward(self, *args, **kwargs):
        import ttnn

        partial = object.__getattribute__(self, "reduced_module").forward(*args, **kwargs)
        topology = object.__getattribute__(self, "topology")
        if partial.shape[-2] == 1:
            return ttnn.all_reduce(partial, cluster_axis=1, topology=topology, num_links=CCL_LINKS,
                                   memory_config=partial.memory_config())
        index = object.__getattribute__(self, "semaphore_index")
        object.__setattr__(self, "semaphore_index", index ^ 1)
        semaphores = object.__getattribute__(self, "semaphores")[index]
        return ttnn.experimental.all_reduce_async(
            partial, cluster_axis=1, mesh_device=object.__getattribute__(self, "mesh"),
            barrier_semaphores=semaphores["barrier"], rs_global_semaphores=semaphores["rs"],
            ag_global_semaphores=semaphores["ag"], math_op=ttnn.ReduceType.Sum,
            memory_config=partial.memory_config(), topology=topology, num_links=CCL_LINKS)


def _pad_gdn_columns(attention, mesh, options):
    import torch
    import ttnn

    heads = attention.num_v_heads
    if heads == 32:
        return
    qkv = attention.cfg.q_dim + attention.cfg.k_dim + attention.cfg.v_dim
    weight = ttnn.to_torch(ttnn.get_device_tensors(attention.mega_weight)[0])
    weight = weight.reshape(weight.shape[-2], weight.shape[-1])
    pad = torch.zeros(weight.shape[0], 32 - heads, dtype=weight.dtype)
    a, b, z = weight[:, qkv:qkv + heads], weight[:, qkv + heads:qkv + 2 * heads], weight[:, qkv + 2 * heads:]
    padded = torch.cat((weight[:, :qkv], a, pad, b, pad, z), dim=1).contiguous()
    attention.mega_weight = ttnn.from_torch(padded, device=mesh, dtype=attention.mega_weight.dtype,
                                            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    attention.optimized_projections["attention_in"] = _Projection(attention.mega_weight, mesh, options)
    attention.ab_stride = 32


def merge_replicas(mesh, roots):
    import torch
    import ttnn

    memo, seen = {}, set()
    stats = {"sharded": 0, "replicated": 0, "sharded_bytes": 0}

    def tensor(items):
        key = id(items[0])
        if key in memo:
            return memo[key]
        hosts = [ttnn.from_device(ttnn.get_device_tensors(t)[rank]) for rank, t in enumerate(items)]
        first = ttnn.to_torch(hosts[0])
        same = all(torch.equal(first, ttnn.to_torch(h)) for h in hosts[1:])
        if same:
            memo[key] = items[0]
            stats["replicated"] += 1
        else:
            memo[key] = ttnn.to_device(ttnn.from_host_shards(hosts, mesh.shape), mesh,
                                       memory_config=items[0].memory_config())
            stats["sharded"] += 1
            stats["sharded_bytes"] += items[0].buffer_size() if hasattr(items[0], "buffer_size") else 0
        return memo[key]

    def walk(items):
        head = items[0]
        if isinstance(head, ttnn.Tensor):
            return tensor(items)
        if isinstance(head, list):
            for i in range(len(head)):
                head[i] = walk([x[i] for x in items])
            return head
        if isinstance(head, tuple):
            values = [walk([x[i] for x in items]) for i in range(len(head))]
            if all(a is b for a, b in zip(values, head)):
                return head
            return type(head)(*values) if hasattr(head, "_fields") else tuple(values)
        if isinstance(head, dict):
            for key in list(head):
                head[key] = walk([x[key] for x in items])
            return head
        if type(head).__module__.startswith("models.") and hasattr(head, "__dict__"):
            if id(head) in seen:
                return head
            seen.add(id(head))
            state = vars(head)
            for key in list(state):
                state[key] = walk([vars(x)[key] for x in items])
            return head
        return head

    walk(list(roots))
    return stats


class MultichipDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, optimization=None,
                        tensor_cache_path=None, **kwargs):
        import gc

        import ttnn

        tp = mesh_device.get_num_devices()
        config = FunctionalDecoderConfig.from_hf_config(hf_config)
        kind = config.layer_kind(layer_idx)
        policy = mesh_policy(DEFAULT_POLICIES[kind] if optimization is None else optimization, tp, kind)
        if tp == 1:
            result = super().from_state_dict(state_dict, hf_config=hf_config, layer_idx=layer_idx,
                                             mesh_device=mesh_device, optimization=policy,
                                             tensor_cache_path=tensor_cache_path, **kwargs)
            result.tensor_parallel = 1
            result.global_config = config
            return result
        if tuple(mesh_device.shape) != (1, tp):
            raise ValueError("The tensor-parallel decoder uses a 1xN mesh")
        state = remapped_layer_state(state_dict, layer_idx)
        local_config = local_hf_config(hf_config, tp)
        samples = [_construction_sample(mesh_device, "start")]
        replicas = []
        for rank in range(tp):
            replicas.append(cls._replica(state, config, local_config, layer_idx, tp, rank, mesh_device, policy,
                                         tensor_cache_path, kwargs))
            samples.append(_construction_sample(mesh_device, f"replica_{rank}"))
        result = replicas[0]
        result.merge_stats = merge_replicas(mesh_device, replicas)
        samples.append(_construction_sample(mesh_device, "merged_before_release"))
        del replicas
        gc.collect()
        samples.append(_construction_sample(mesh_device, "released"))
        result.construction_memory = samples
        result.tensor_parallel = tp
        result.global_config = config
        result.weight_memory = {"after_merge": _memory_bytes(mesh_device)}
        result.expert_dispatch = result.layer.feed_forward.module.experts.dispatch
        topology = ttnn.Topology.Ring if tp > 2 else ttnn.Topology.Linear
        result.layer.attention = _AllReduceOutput(result.layer.attention, topology, mesh_device)
        result.layer.feed_forward = _AllReduceOutput(result.layer.feed_forward, topology, mesh_device)
        return result

    @classmethod
    def _replica(cls, state, config, local_config, layer_idx, tp, rank, mesh_device, policy, tensor_cache_path, kwargs):
        import gc

        cache = None if tensor_cache_path is None else tensor_cache_path / f"tp{tp}_rank{rank}"
        replica = super().from_state_dict(tensor_parallel_state(state, layer_idx, tp, rank, config),
                                          hf_config=local_config, layer_idx=layer_idx, mesh_device=mesh_device,
                                          optimization=policy, tensor_cache_path=cache, replicated_mesh=True, **kwargs)
        if replica.optimization.prefill_backend != "dispatch":
            raise ValueError("The tensor-parallel decoder requires the dispatch prefill backend")
        experts = replica.layer.feed_forward.module.experts
        experts.original.reduce_cluster_axis = 1
        gate_up, down, offset = expert_parallel_state(state, layer_idx, tp, rank, config)
        experts.dispatch = _ExpertParallelDispatch(gate_up, down, mesh_device, replica.optimization, offset)
        if config.is_deltanet_layer(layer_idx):
            _pad_gdn_columns(replica.layer.attention, mesh_device, replica.optimization.projections["attention_in"])
        gc.collect()
        return replica
