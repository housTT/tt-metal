from pathlib import Path

p = Path("/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py")
s = p.read_text()
old = """            result.tensor_parallel = 1
            result.global_config = config
            return result"""
new = """            result.tensor_parallel = 1
            result.global_config = config
            _release_gdn_input_weight(result)
            return result"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """        if sharded:
            result.layer = _ShardedResidualLayer(result.layer, _decode_allreduce(mesh_device, tp, sharded)["memory"])
        return result"""
new = """        if sharded:
            result.layer = _ShardedResidualLayer(result.layer, _decode_allreduce(mesh_device, tp, sharded)["memory"])
        _release_gdn_input_weight(result)
        return result"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """class _ShardedResidualLayer:
"""
new = """def _release_gdn_input_weight(decoder):
    attention = decoder.layer.attention
    if type(attention).__name__ == "_AllReduceOutput":
        attention = object.__getattribute__(attention, "reduced_module")
    projection = getattr(attention, "optimized_projections", {}).get("attention_in")
    if (getattr(attention, "mega_weight", None) is not None and projection is not None and not projection.parts
            and projection.weight is not attention.mega_weight and not decoder.optimization.gdn_ab_split):
        attention.mega_weight = None


class _ShardedResidualLayer:
"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print("patched")
