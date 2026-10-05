"""Stage 0 device probe for the Cloudflare/clef bringup.

Records which mesh-open paths work on this box and whether one-link Linear
collectives run on a (1,2) submesh. Writes a JSON report. Run through devrun
with a timeout; each probe opens and closes its own mesh.
"""

import json
import os
import sys
import time
import traceback

import torch

import ttnn

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/hous/dev/clef/reports/stage0_mesh_probe.json"
DEVICE_KW = dict(l1_small_size=24576, num_command_queues=2, trace_region_size=0)
report = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "probes": {}}


def record(name, fn):
    t0 = time.time()
    entry = {"ok": False}
    try:
        entry.update(fn() or {})
        entry["ok"] = True
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        entry["traceback"] = traceback.format_exc()[-4000:]
    entry["seconds"] = round(time.time() - t0, 2)
    report["probes"][name] = entry
    print(f"[probe] {name}: {'OK' if entry['ok'] else 'FAIL'} ({entry['seconds']} s)", flush=True)
    if not entry["ok"]:
        print(entry["error"], flush=True)
    json.dump(report, open(OUT, "w"), indent=2)


def mesh_info(mesh):
    info = {"shape": list(mesh.shape), "num_devices": mesh.get_num_devices()}
    try:
        info["device_ids"] = list(mesh.get_device_ids())
    except Exception:
        pass
    try:
        info["cluster_type"] = str(ttnn.cluster.get_cluster_type())
    except Exception as exc:
        info["cluster_type_error"] = str(exc)[:200]
    return info


def collective_smoke(mesh, num_links, topology):
    from models.tt_transformers.tt.ccl import TT_CCL, tt_all_gather, tt_all_reduce

    rows, cols = 32, 64
    nd = mesh.get_num_devices()
    x = torch.arange(rows * cols * nd, dtype=torch.float32).reshape(1, 1, rows, cols * nd) / 1000.0
    x_bf = x.to(torch.bfloat16).to(torch.float32)
    tt_ccl = TT_CCL(mesh)
    out = {"num_links": num_links, "topology": str(topology)}

    def shard():
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    gathered = tt_all_gather(shard(), mesh, tt_ccl, cluster_axis=None, dim=3, num_links=num_links, topology=topology)
    g = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).to(torch.float32)
    out["all_gather_shape"] = list(g.shape)
    out["all_gather_max_abs_err"] = float((g[0] - x_bf[0]).abs().max())
    out["all_gather_match"] = bool(torch.allclose(g[0], x_bf[0], rtol=1e-2, atol=1e-3))

    reduced = tt_all_reduce(
        shard(),
        mesh,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=num_links,
        num_all_gather_links=num_links,
        topology=topology,
    )
    r = ttnn.to_torch(reduced, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).to(torch.float32)
    total = sum(x_bf[..., i * cols : (i + 1) * cols] for i in range(nd))
    width = r.shape[-1]
    expect = torch.cat([total[..., i * width : (i + 1) * width] for i in range(nd)], dim=0)
    out["reduce_scatter_shape"] = list(r.shape)
    out["reduce_scatter_max_abs_err"] = float((r - expect).abs().max())
    out["reduce_scatter_match"] = bool(torch.allclose(r, expect, rtol=1e-2, atol=1e-2))
    ttnn.synchronize_device(mesh)
    return out


CCL_VARIANTS = [
    (1, "Linear"),
    (2, "Linear"),
    (1, "Ring"),
    (2, "Ring"),
]


def collective_variants(mesh, into):
    results = {}
    into["collectives"] = results
    for num_links, topo_name in CCL_VARIANTS:
        key = f"links{num_links}_{topo_name.lower()}"
        t0 = time.time()
        try:
            results[key] = collective_smoke(mesh, num_links, getattr(ttnn.Topology, topo_name))
            results[key]["ok"] = results[key]["all_gather_match"] and results[key]["reduce_scatter_match"]
        except Exception as exc:
            results[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:1500]}
        results[key]["seconds"] = round(time.time() - t0, 2)
        print(f"[ccl] {key}: {'OK' if results[key]['ok'] else 'FAIL'} {results[key]}", flush=True)
        json.dump(report, open(OUT, "w"), indent=2)
    return results


def probe_direct_1x2():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), **DEVICE_KW)
    try:
        return mesh_info(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def probe_parent_submesh(with_collectives):
    def run():
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
        parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2), **DEVICE_KW)
        try:
            sub = parent.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(0, 0))
            info = {"parent": mesh_info(parent), "submesh": mesh_info(sub)}
            if with_collectives:
                report["probes"]["parent_2x2_submesh_1x2_collectives"] = {"ok": False, **info}
                collective_variants(sub, info)
            return info
        finally:
            ttnn.close_mesh_device(parent)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    return run


def probe_1x4():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), **DEVICE_KW)
    try:
        info = mesh_info(mesh)
        sub = mesh.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(0, 0))
        info["submesh_1x2_of_1x4"] = mesh_info(sub)
        report["probes"]["direct_1x4_with_1x2_submesh"] = {"ok": False, **info}
        collective_variants(sub, info)
        return info
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


which = os.environ.get("PROBES", "direct,parent,parent_ccl,1x4").split(",")
if "direct" in which:
    record("direct_1x2_fabric1d", probe_direct_1x2)
if "parent" in which:
    record("parent_2x2_submesh_1x2", probe_parent_submesh(False))
if "parent_ccl" in which:
    record("parent_2x2_submesh_1x2_collectives", probe_parent_submesh(True))
if "1x4" in which:
    record("direct_1x4_with_1x2_submesh", probe_1x4)
print("MESH_PROBE_DONE", OUT, flush=True)
