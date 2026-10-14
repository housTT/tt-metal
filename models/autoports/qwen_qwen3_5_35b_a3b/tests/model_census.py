# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import defaultdict

LAYER = re.compile(r"^gen\.model\.layers\[(\d+)\]")
STATE = re.compile(r"\._states\[(\d+)\]")


def _category(paths):
    layers = {m.group(1) for m in (LAYER.match(p) for p in paths) if m}
    if any("kv_pool" in p for p in paths):
        return "kv_cache"
    if any(STATE.search(p) for p in paths):
        return "gdn_request_state"
    if len(layers) > 1:
        return "shared_across_layers"
    if layers:
        return "layer_private"
    first = paths[0]
    for prefix, name in (("gen.model.embedding", "embedding"), ("gen.model.lm_head", "lm_head"),
                         ("gen.model.norm", "final_norm"), ("gen.sampler", "sampler"), ("gen.tt_", "generator_io")):
        if first.startswith(prefix):
            return name
    return "other"


def census(gen):
    import ttnn

    roots, seen = {}, set()

    def visit(obj, path, depth=0):
        if depth > 20:
            return
        if isinstance(obj, ttnn.Tensor):
            if not ttnn.is_tensor_storage_on_device(obj) or not obj.is_allocated():
                return
            uid = obj.buffer_unique_id()
            entry = roots.setdefault(uid, dict(paths=[], bytes=obj.buffer_num_pages() * obj.buffer_page_size(),
                                               buffer=str(obj.memory_config().buffer_type)))
            entry["paths"].append(path)
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            items = [(f"{path}[{key!r}]", value) for key, value in obj.items()]
        elif isinstance(obj, (list, tuple)):
            items = [(f"{path}[{i}]", value) for i, value in enumerate(obj)]
        else:
            try:
                items = [(f"{path}.{key}", value) for key, value in object.__getattribute__(obj, "__dict__").items()]
            except (AttributeError, TypeError):
                return
        for child_path, child in items:
            if not isinstance(child, (str, bytes, int, float, bool, type(None))):
                visit(child, child_path, depth + 1)

    visit(gen, "gen")
    totals = defaultdict(lambda: dict(dram_bytes=0, l1_bytes=0, buffers=0))
    for entry in roots.values():
        row = totals[_category(entry["paths"])]
        row["dram_bytes" if entry["buffer"].endswith("DRAM") else "l1_bytes"] += entry["bytes"]
        row["buffers"] += 1
    view = ttnn.get_memory_view(gen.mesh_device, ttnn.BufferType.DRAM)
    allocated = view.total_bytes_allocated_per_bank * view.num_banks
    named = sum(row["dram_bytes"] for row in totals.values())
    largest = sorted(roots.values(), key=lambda e: -e["bytes"])[:12]
    return dict(per_device=True, by_category=dict(totals), named_dram_bytes=named, allocator_dram_bytes=allocated,
                unnamed_dram_bytes=allocated - named,
                largest=[dict(bytes=e["bytes"], buffer=e["buffer"], path=e["paths"][0], aliases=len(e["paths"]))
                         for e in largest])
