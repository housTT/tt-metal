# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Live device-buffer ownership census for one constructed optimized layer."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, choices=[0, 3], required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("../weights/Qwen3.5-35B-A3B"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decoder", choices=("optimized", "multichip"), default="optimized")
    p.add_argument("--mesh", type=int, choices=(1, 2, 4), default=1)
    args = p.parse_args()

    import ttnn
    from transformers import AutoConfig

    from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
    from models.autoports.qwen_qwen3_5_35b_a3b.tt import optimized_decoder as od

    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    state = load_layer_state_dict(args.checkpoint, args.layer)
    from .mesh_util import decoder_class, open_mesh

    dev = int(os.environ.get("QDEV", "0"))
    if args.mesh == 1:
        os.environ.setdefault("QDEV", str(dev))
    mesh = open_mesh(args.mesh)

    def allocated():
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        return view.total_bytes_allocated_per_bank * view.num_banks, view.total_bytes_per_bank * view.num_banks

    try:
        baseline, total = allocated()
        model = decoder_class(args.decoder).from_state_dict(state, hf_config=cfg, layer_idx=args.layer,
                                                             mesh_device=mesh, expert_precision="inherited")
        if cfg.layer_types[args.layer] == "full_attention":
            kc = ttnn.zeros((35, model.config.n_kv_heads, 32, cfg.head_dim), dtype=ttnn.bfloat8_b,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            model.bind_paged_kv_cache(kc, ttnn.zeros_like(kc))
        after, _ = allocated()

        seen_objects, roots = set(), defaultdict(list)

        def visit(obj, path, depth=0):
            if depth > 14 or id(obj) in seen_objects:
                return
            seen_objects.add(id(obj))
            if isinstance(obj, ttnn.Tensor):
                if obj.storage_type() != ttnn.StorageType.DEVICE or not obj.is_allocated():
                    return
                mem = obj.memory_config()
                roots[obj.buffer_unique_id()].append(dict(
                    path=path, dtype=str(obj.dtype), layout=str(obj.layout), shape=list(obj.shape),
                    buffer=str(mem.buffer_type), bytes=obj.buffer_num_pages() * obj.buffer_page_size()))
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    visit(v, f"{path}[{k!r}]", depth + 1)
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    visit(v, f"{path}[{i}]", depth + 1)
            elif hasattr(obj, "__dict__") and not isinstance(obj, type):
                for k, v in vars(obj).items():
                    visit(v, f"{path}.{k}", depth + 1)

        visit(model, "decoder")
        for name in ("_PREFIX_MATRICES", "_ZERO_ROWS"):
            visit(getattr(od, name, {}), f"module.{name}")

        def consumer(path):
            rules = [("_states", "gdn_recurrent_and_conv_state_slots"), ("paged_kv", "kv_cache"),
                     ("dispatch", "routed_experts_prefill_dispatch"), ("experts", "routed_experts_decode"),
                     ("shared", "shared_expert"), ("router", "router"), ("gate_pack", "router_shared_gate_pack"),
                     ("packed", "router_shared_gate_pack"), ("attention", "attention"), ("norm", "norms"),
                     ("module.", "module_cache"), ("rotary", "rope"), ("rope", "rope"), ("_index", "index_tables")]
            low = path.lower()
            return next((c for key, c in rules if key.lower() in low), "other")

        rows = []
        for uid, refs in roots.items():
            primary = min(refs, key=lambda r: len(r["path"]))
            rows.append(dict(uid=uid, consumer=consumer(primary["path"]), bytes=max(r["bytes"] for r in refs),
                             aliases=len(refs), **{k: primary[k] for k in ("path", "dtype", "layout", "shape", "buffer")},
                             alias_paths=[r["path"] for r in refs[1:6]]))
        rows.sort(key=lambda r: -r["bytes"])
        by_consumer = defaultdict(lambda: dict(bytes=0, roots=0))
        for r in rows:
            if r["buffer"].endswith("DRAM"):
                by_consumer[r["consumer"]]["bytes"] += r["bytes"]
                by_consumer[r["consumer"]]["roots"] += 1
        named = sum(r["bytes"] for r in rows if r["buffer"].endswith("DRAM"))
        result = dict(
            layer_idx=args.layer, layer_type=cfg.layer_types[args.layer], device=dev,
            decoder=args.decoder, mesh_shape=list(mesh.shape), device_ids=list(mesh.get_device_ids()),
            optimization=__import__("dataclasses").asdict(model.optimization),
            weight_memory=getattr(model, "weight_memory", None),
            construction_memory=getattr(model, "construction_memory", None),
            allocator=dict(baseline_bytes=baseline, after_construct_bytes=after, delta_bytes=after - baseline,
                           total_dram_bytes=total),
            named_dram_payload_bytes=named,
            named_note="Payload lower bound grouped by root buffer_unique_id; bank-spread reservation can exceed it.",
            by_consumer=dict(sorted(by_consumer.items(), key=lambda kv: -kv[1]["bytes"])),
            roots=rows,
            source_sha256=__import__("hashlib").sha256(Path(od.__file__).read_bytes()).hexdigest(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("allocator", "named_dram_payload_bytes", "by_consumer")}, indent=1))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
