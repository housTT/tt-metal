import json
import sys
from pathlib import Path

import torch
import ttnn

dump = torch.load(sys.argv[1], weights_only=False)
out = Path(sys.argv[2])
replays = int(sys.argv[3]) if len(sys.argv) > 3 else 200
device_index = dump["device"]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[3])
DT = {"DataType.BFLOAT16": ttnn.bfloat16, "DataType.BFLOAT8_B": ttnn.bfloat8_b, "DataType.BFLOAT4_B": ttnn.bfloat4_b,
      "DataType.UINT32": ttnn.uint32, "DataType.UINT16": ttnn.uint16, "DataType.INT32": ttnn.int32}
LAYOUT = {"Layout.TILE": ttnn.TILE_LAYOUT, "Layout.ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT}


def put(value, dtype, layout):
    return ttnn.from_torch(value, device=mesh, dtype=DT[dtype], layout=LAYOUT[layout],
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


try:
    args = []
    for value, dtype, layout in zip(dump["args"], dump["arg_dtypes"], dump["arg_layouts"]):
        if dtype is None:
            args.append(value)
        elif isinstance(dtype, list):
            args.append([put(v[device_index], d, l) for v, d, l in zip(value, dtype, layout)])
        else:
            args.append(put(value[device_index], dtype, layout))
    kwargs = {}
    for key, value in dump["kwargs"].items():
        if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], list):
            kwargs[key] = put(value[0][device_index], value[1], value[2])
        elif key == "core_grid":
            kwargs[key] = ttnn.CoreCoord(*value)
        else:
            kwargs[key] = value
    compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
                                                     fp32_dest_acc_en=False, packer_l1_acc=False,
                                                     math_approx_mode=False)
    outs = []
    for _ in range(replays):
        output = ttnn.zeros(dump["output_shape"], device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(*args, output=output, dtype=ttnn.bfloat16,
                                                                     compute_kernel_config=compute, **kwargs)
        outs.append(ttnn.to_torch(result))
        ttnn.deallocate(result)
    distinct = []
    for o in outs:
        if not any(torch.equal(o, x) for x in distinct):
            distinct.append(o)
    reference = dump["reference"][device_index]
    result = dict(experiment="E2a single-chip replay of captured mesh call", device_of_capture=device_index,
                  capture_index=dump["capture_index"], replays=replays, distinct_outputs=len(distinct),
                  equal_to_model_call=sum(torch.equal(o, reference) for o in outs),
                  max_abs_vs_model_call=max(float((o.float() - reference.float()).abs().max()) for o in outs))
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
finally:
    ttnn.close_mesh_device(mesh)
