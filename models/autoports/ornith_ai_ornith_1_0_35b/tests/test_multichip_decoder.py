# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness, capability and performance tests for the Ornith-1.0-35B **multichip** decoder.

Target mesh: ``ttnn.MeshShape(1, 4)`` on the 4-chip Blackhole ``p300c`` ring, under
``FabricConfig.FABRIC_1D_RING``. Every test in this file opens that mesh.

The file is organised around what a multichip stage has to prove that a single-chip one does not:

* **agreement with the single-chip TTNN baseline.** The primary bar here is not HF: it is
  :class:`~...tt.optimized_decoder.OptimizedDecoder` built **in the same process, on the same mesh,
  from the same weights**, which isolates sharding and collective bugs from the HF-vs-TTNN numerical
  difference the previous stages already characterised. The single-chip decoder replicates every
  weight across the mesh, so on a 1x4 mesh it computes four identical copies of the single-chip
  result and device 0's copy *is* the single-chip answer.
* **the stacked-layer layout contract.** A decoder layer in this stage takes a replicated activation
  and returns a tensor that is bit-identical on all four devices, so a stack of them needs no
  boundary conversion. ``test_output_is_identical_on_every_device`` is that contract.
* **the sharding is real.** ``test_weights_are_sharded_not_replicated`` and
  ``test_expert_partition_is_disjoint_and_complete`` fail if a weight silently ends up replicated,
  which would still produce correct-looking output for the TP tensors and *wrong* output for the
  expert-parallel ones — and would make every speedup claim meaningless.
* **the collectives are the ones the design says.** ``test_collectives_per_forward`` pins exactly two
  per layer and where they sit.
* **everything the single-chip stage asserted still holds** — paged KV behaviour, per-user page
  tables, ragged decode positions, chunked-prefill continuation, non-aligned lengths and
  ``max_context``, determinism, trace replay, the advertised 262144-token context, and the absence of
  any host fallback in a measured forward.

Weight source is selected by ``ORNITH_WEIGHTS`` exactly as in the optimized suite.

    pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_multichip_decoder.py -v -p no:randomly

The ``long`` cases (full advertised context) run in the same invocation; the marker exists to narrow
a run, not to deselect.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as OD
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig
from models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder import (
    DEFAULT_CCL_TOPOLOGY,
    DEFAULT_FABRIC_CONFIG,
    DEFAULT_MESH_SHAPE,
    MULTICHIP_DECODE_MATMUL_GEOMETRY,
    MultichipDecoder,
    local_decoder_config,
    num_blocks_for_context,
)
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import TILE, OptimizedDecoder, _physical_rows

DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"

LINEAR_LAYER = 0
FULL_LAYER = 3
LAYERS = (LINEAR_LAYER, FULL_LAYER)
LAYER_IDS = {LINEAR_LAYER: "linear_attention", FULL_LAYER: "full_attention"}

#: Acceptance bar against the float32 HF golden, inherited from the functional-decoder stage and not
#: lowered by this one.
PCC_BAR = 0.995

#: Acceptance bar against the **single-chip TTNN baseline**. Deliberately much tighter than
#: :data:`PCC_BAR`: the two implementations compute the same math from the same weights in the same
#: dtypes, and differ only by where the sums are reassociated (a 4-way split of every contraction
#: plus a bfloat16 collective). Anything looser would let a real sharding or collective bug through
#: on the strength of the HF bar's slack.
BASELINE_BAR = 0.999

TEST_CONTEXT = 8192
ADVERTISED_CONTEXT = 262144

#: Realised sparse-matmul grid the shipped policy must produce at each advertised decode batch.
#:
#: Derived from the shipped constants rather than guessed, and asserted rather than assumed:
#: ``OptimizedMoE._active_expert_bound`` is ``min(num_experts_local, rows * top_k)``, so the bound is
#: 8 / 16 / 32 / 64 at batch 1 / 2 / 4 / >=8; ``MultichipMoE._sparse_cfg`` scales it by
#: ``SPARSE_CORES_PER_ACTIVE[role]``; the parent clamps to ``[8, 32]`` and reduces to the largest
#: divisor of ``Nt`` (32 for ``gate_up``, 64 for ``down``). The net effect is
#: ``cores = clamp(bound, 8, 32)`` for both roles.
#:
#: Every row is measured as a win by the ``sparse`` arm of
#: ``doc/multichip_decoder/logs/probe_decode_batch.txt`` except batch 1, where the geometry is
#: identical to the inherited one and the arms tie.
SPARSE_DECODE_CORES = {
    1: {"gate_up": 8, "down": 8},
    2: {"gate_up": 16, "down": 16},
    4: {"gate_up": 32, "down": 32},
    8: {"gate_up": 32, "down": 32},
    32: {"gate_up": 32, "down": 32},
}

#: The same, for **prefill**. Every prefill expert group is a full 32 rows, so the bound saturates at
#: ``min(64, 32*8) = 64`` at every sequence length and batch and the geometry is one pair of numbers.
#: Review round 3 found the prefill row of README section 5.7's table was derivation-only — the spy
#: below used to be installed after ``prefill_forward`` returned — so it is pinned here as well.
SPARSE_PREFILL_CORES = {"gate_up": 32, "down": 32}

#: The active-expert count a full 32-token prefill group produces on one device, and the band the
#: suite holds it to. ``tracy/run_profiling.sh`` passes :data:`PREFILL_ACTIVE_MODEL` to
#: ``tt-perf-report --active-experts`` and ``probe_sparse_matmul_local.py`` sweeps at it, so it is a
#: modelling input for every prefill DRAM and FLOPs figure rather than a description. Expectation:
#: ``64 * (1 - (1 - 1/64)^(32*8/4))`` = 40.6, measured 39-44 across layer kinds and sweeps.
PREFILL_ACTIVE_MODEL = 41
PREFILL_ACTIVE_BAND = (30, 55)

#: Bar for a BFP4 expert weight block against the float checkpoint. See
#: `test_expert_partition_is_disjoint_and_complete`; the quantisation itself costs ~7e-3 here.
EXPERT_WEIGHT_BAR = 0.99

#: The traced-decode speedup the suite gates on. See `test_multichip_beats_single_chip_traced_decode`.
DECODE_SPEEDUP_BAR = 1.4

#: Warmed 2048-token prefill wall-clock bar, milliseconds. See `test_perf_prefill`.
PREFILL_MS_BAR = 48.0

#: The mesh every test opens, and the fabric it needs. ``l1_small_size`` matches the optimized
#: suite; the CCL ops allocate their semaphores out of it.
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "trace_region_size": 0,
        "fabric_config": DEFAULT_FABRIC_CONFIG,
        # The fabric packet size the runtime asks for on this layer's 2048-element pages. Measured,
        # not assumed: `probe_ccl.txt`'s `CCLPKT` rows. Review round 5 found 864 warnings a suite
        # recommending it, unclassified, on the stage's own critical path.
        "fabric_router_config": MC.fabric_router_config(),
    },
]

pytestmark = [
    pytest.mark.parametrize("mesh_device", [DEFAULT_MESH_SHAPE], indirect=True),
    pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True),
]


class _OpRecorder:
    """Count selected ``ttnn`` entry points during one measured forward pass."""

    def __init__(self, monkeypatch, names):
        self.calls: dict[str, int] = {}
        for name in names:
            module, _, attr = name.rpartition(".")
            target = ttnn
            for part in module.split(".") if module else []:
                target = getattr(target, part)
            original = getattr(target, attr)

            def wrapper(*args, _name=name, _original=original, **kwargs):
                self.calls[_name] = self.calls.get(_name, 0) + 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(target, attr, wrapper)

    def count(self, name):
        return self.calls.get(name, 0)

    def reset(self):
        self.calls.clear()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def decode_grid_for(grid, cores: int) -> tuple:
    """The worker rectangle a decode matmul target realises as, mirroring `_decode_1d_matmul_config`.

    The widest legal rectangle at or below ``cores``: full rows of the device grid's x extent, capped
    by its y extent. Mirrored rather than imported so that changing the rule takes a deliberate edit
    in both places -- the point of the assertion is that the shipped layer passes the swept target
    through to this rule, which a `realised <= target` check could not see (review round 4).
    """
    cols = min(grid.x, cores)
    return cols, min(math.ceil(cores / cols), grid.y)


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    """Pearson correlation, accumulated in float64."""
    a = golden.double().flatten()
    b = actual.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _snapshot_available() -> bool:
    try:
        return (R.resolve_model_path() / "model.safetensors.index.json").is_file()
    except Exception:  # noqa: BLE001 - offline / not downloaded
        return False


def default_weight_source() -> str:
    src = os.environ.get("ORNITH_WEIGHTS")
    if src:
        return src
    return "real" if _snapshot_available() else "synthetic"


_CACHE: dict = {}


def hf_config():
    if "cfg" not in _CACHE:
        _CACHE["cfg"] = R.load_text_config()
    return _CACHE["cfg"]


def layer_state_dict(layer_idx: int, source: str) -> dict:
    key = ("sd", layer_idx, source)
    if key in _CACHE:
        return _CACHE[key]
    if source == "real":
        if not _snapshot_available():
            pytest.skip("Ornith-1.0-35B checkpoint snapshot not available")
        sd = R.load_layer_state_dict(layer_idx)
    elif source == "synthetic":
        path = DOC_DIR / f"weight_stats_layer{layer_idx}.json"
        if not path.is_file():
            pytest.skip(f"missing recorded weight statistics: {path}")
        with open(path) as f:
            sd = R.synthetic_state_dict(json.load(f)["tensors"], seed=1234 + layer_idx)
    else:
        raise ValueError(f"unknown weight source {source!r}")
    _CACHE[key] = sd
    return sd


def reference_layer(layer_idx: int, source: str):
    key = ("ref", layer_idx, source)
    if key not in _CACHE:
        _CACHE[key] = R.build_reference_layer(hf_config(), layer_idx, layer_state_dict(layer_idx, source))
    return _CACHE[key]


def make_activations(batch: int, seq_len: int, *, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, seq_len, hf_config().hidden_size, generator=gen) * 0.5).to(torch.bfloat16)


def to_device(mesh_device, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Replicate a host tensor across the mesh — the decoder's input distribution."""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def shards(mesh_device, tensor):
    """Every device's copy of a mesh tensor, as a list of host tensors."""
    lead = int(tensor.shape[0])
    whole = ttnn.to_torch(tensor, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0))
    return [whole[i * lead : (i + 1) * lead] for i in range(mesh_device.get_num_devices())]


def to_host(mesh_device, tensor):
    """Device 0's copy of a mesh tensor. Every decoder output is identical across the mesh."""
    lead = int(tensor.shape[0])
    whole = ttnn.to_torch(tensor, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0))
    return whole[:lead]


def build_decoder(
    mesh_device,
    layer_idx: int,
    source: str,
    *,
    batch: int = 1,
    max_context: int = TEST_CONTEXT,
    prefill_chunk: int = 2048,
    num_blocks: int | None = None,
    cls=MultichipDecoder,
    kv_cache_dtype=None,
    **kwargs,
):
    """Construct the layer, allocate its paged cache / recurrent state, and build a page table."""
    sd = layer_state_dict(layer_idx, source)
    decoder = cls.from_state_dict(
        sd,
        hf_config=hf_config(),
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_context=max_context,
        prefill_chunk=prefill_chunk,
        **kwargs,
    )
    blocks_per_user = num_blocks if num_blocks is not None else num_blocks_for_context(max_context)
    total_blocks = blocks_per_user * batch
    decoder.allocate_kv_cache(total_blocks, dtype=kv_cache_dtype)
    decoder.allocate_state(batch)
    page_table = None
    if decoder.is_full_attention:
        table = torch.arange(total_blocks, dtype=torch.int32).reshape(batch, blocks_per_user)
        page_table = to_device(mesh_device, table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return decoder, page_table, blocks_per_user


def decode_inputs(mesh_device, positions: torch.Tensor):
    current_pos = to_device(mesh_device, positions.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    rot_idxs = to_device(
        mesh_device, positions.to(torch.int32).reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    return current_pos, rot_idxs


def run_reference(layer_idx, source, x, *, decode_x=None, decode_steps=0, start_pos=0):
    ref = reference_layer(layer_idx, source)
    cfg = hf_config()
    with torch.no_grad():
        prefill_out, cache = R.reference_prefill(ref, cfg, x.float(), start_pos=start_pos)
        decode_outs = []
        pos = start_pos + x.shape[1]
        for step in range(decode_steps):
            positions = torch.full((x.shape[0],), pos + step, dtype=torch.long)
            decode_outs.append(R.reference_decode(ref, cfg, decode_x[step].float(), positions, cache))
    return prefill_out, decode_outs


def _snapshot_state(mesh_device, decoder):
    """Host copies of the mutable state, for rewinding around trace capture."""
    snapshot = {}
    if decoder.is_full_attention:
        snapshot["k"] = shards(mesh_device, decoder.k_cache)
        snapshot["v"] = shards(mesh_device, decoder.v_cache)
    else:
        snapshot["rec"] = shards(mesh_device, decoder.recurrent_state)
        snapshot["conv"] = [shards(mesh_device, buf) for buf in decoder.conv_state]
    return snapshot


def _restore(mesh_device, saved_shards, target, dtype):
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.cat(saved_shards, dim=0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=0),
        ),
        target,
    )


def _restore_state(mesh_device, decoder, snapshot):
    if decoder.is_full_attention:
        _restore(mesh_device, snapshot["k"], decoder.k_cache, decoder.k_cache.dtype)
        _restore(mesh_device, snapshot["v"], decoder.v_cache, decoder.v_cache.dtype)
    else:
        _restore(mesh_device, snapshot["rec"], decoder.recurrent_state, ttnn.float32)
        for saved, buf in zip(snapshot["conv"], decoder.conv_state):
            _restore(mesh_device, saved, buf, ttnn.bfloat16)


# --------------------------------------------------------------------------------------
# the mesh plan itself
# --------------------------------------------------------------------------------------
def test_local_config_is_the_per_device_view(mesh_device):
    """``local_decoder_config`` divides exactly the dims the mesh plan says it divides."""
    g = OrnithDecoderConfig.from_hf_config(hf_config())
    tp = 4
    local = local_decoder_config(g, tp)
    # sharded
    assert local.n_heads == g.n_heads // tp == 4
    assert local.linear_num_key_heads == g.linear_num_key_heads // tp == 4
    assert local.linear_num_value_heads == g.linear_num_value_heads // tp == 8
    assert local.num_experts == g.num_experts // tp == 64
    assert local.shared_expert_intermediate_size == g.shared_expert_intermediate_size // tp == 128
    # 2 kv heads over 4 devices: one each, duplicated across the pair that shares it
    assert local.n_kv_heads == 1
    assert local.n_heads % local.n_kv_heads == 0, "the GQA grouping must survive the split"
    # deliberately NOT sharded
    assert local.dim == g.dim == 2048
    assert local.head_dim == g.head_dim
    assert local.moe_intermediate_size == g.moe_intermediate_size, "expert parallelism keeps experts whole"
    assert local.num_experts_per_tok == g.num_experts_per_tok, "top-k is a global decision"
    # derived widths the inherited code slices on
    assert local.conv_dim == g.conv_dim // tp == 2048
    assert local.linear_v_dim == g.linear_v_dim // tp == 1024
    logger.info(
        f"multichip local config tp={tp}: q_heads {g.n_heads}->{local.n_heads}, kv {g.n_kv_heads}->"
        f"{local.n_kv_heads}, gdn k/v {g.linear_num_key_heads}/{g.linear_num_value_heads}->"
        f"{local.linear_num_key_heads}/{local.linear_num_value_heads}, experts {g.num_experts}->"
        f"{local.num_experts}, shared inter {g.shared_expert_intermediate_size}->"
        f"{local.shared_expert_intermediate_size}"
    )


def test_local_config_rejects_an_indivisible_mesh(mesh_device, expect_error):
    """A ``tp`` the model cannot be split by is refused at build time, not silently rounded."""
    g = OrnithDecoderConfig.from_hf_config(hf_config())
    with expect_error(ValueError, "not divisible"):
        local_decoder_config(g, 3)


def test_mesh_is_the_target_shape(mesh_device):
    """The suite runs on the mesh the module was designed for, with the fabric it needs."""
    assert mesh_device.get_num_devices() == 4
    assert tuple(mesh_device.shape) == DEFAULT_MESH_SHAPE
    logger.info(
        f"multichip target mesh {tuple(mesh_device.shape)} devices={mesh_device.get_num_devices()} "
        f"fabric={DEFAULT_FABRIC_CONFIG} topology={DEFAULT_CCL_TOPOLOGY}"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_weights_are_sharded_not_replicated(mesh_device, layer_idx):
    """Every tensor the plan shards has the per-device shape, and differs across devices.

    A silently-replicated weight is the failure this catches: the TP tensors would still produce a
    plausible output (four times the work, then a 4x-too-large all-reduce), and the expert-parallel
    ones would produce a *wrong* one. Both are invisible without checking the shards themselves.
    """
    source = default_weight_source()
    decoder, _, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    g = OrnithDecoderConfig.from_hf_config(hf_config())
    tp = decoder.tp
    expected = {
        "expert_gate_up": [1, g.num_experts // tp, g.dim, 2 * g.moe_intermediate_size],
        "expert_down": [1, g.num_experts // tp, g.moe_intermediate_size, g.dim],
        "shared_in": [1, 1, g.dim, 2 * (g.shared_expert_intermediate_size // tp) + TILE],
        "shared_down": [1, 1, g.shared_expert_intermediate_size // tp, g.dim],
    }
    replicated = {"router": [1, 1, g.dim, g.num_experts]}
    if decoder.is_full_attention:
        hpd = g.n_heads // tp
        expected["attn_in"] = [g.dim, (2 * hpd + 2) * g.head_dim]
        expected["o_proj"] = [hpd * g.head_dim, g.dim]
    else:
        expected["gdn_in"] = [g.dim, g.conv_dim // tp + g.linear_v_dim // tp + 2 * TILE]
        expected["gdn_out"] = [g.linear_v_dim // tp, g.dim]

    store = {**decoder.w, **decoder.moe.w}
    for name, shape in expected.items():
        tensor = store[name]
        assert list(tensor.shape) == shape, f"{name} per-device shape {list(tensor.shape)} != {shape}"
        parts = shards(mesh_device, tensor)
        # Every pair, not just (0, 1): review round 4 pointed out that a weight replicated across
        # devices 2 and 3 -- exactly what a wrong kv-head or expert-block index would produce -- passed
        # the two-device version of this check.
        for a in range(len(parts)):
            for b in range(a + 1, len(parts)):
                assert not torch.equal(parts[a], parts[b]), f"{name} is identical on devices {a} and {b}"
        logger.info(
            f"multichip weight layer={layer_idx} {name}: per-device {shape}, all "
            f"{len(parts) * (len(parts) - 1) // 2} shard pairs differ"
        )
    for name, shape in replicated.items():
        tensor = store[name]
        assert list(tensor.shape) == shape, f"{name} should stay whole: {list(tensor.shape)} != {shape}"
        parts = shards(mesh_device, tensor)
        for d in range(1, len(parts)):
            assert torch.equal(parts[0], parts[d]), f"{name} must be replicated (global top-k) but shard {d} differs"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_expert_partition_is_disjoint_and_complete(mesh_device, layer_idx):
    """The four devices' expert blocks reconstruct exactly the checkpoint's 256 experts, in order."""
    source = default_weight_source()
    decoder, _, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    g = OrnithDecoderConfig.from_hf_config(hf_config())
    sd = layer_state_dict(layer_idx, source)
    host = sd["mlp.experts.gate_up_proj"].float().transpose(-2, -1)  # [E, dim, 2I]

    parts = shards(mesh_device, decoder.moe.w["expert_gate_up"])
    rebuilt = torch.cat([p.reshape(-1, g.dim, 2 * g.moe_intermediate_size) for p in parts], dim=0)
    assert rebuilt.shape[0] == g.num_experts
    # Every expert, not the 1-in-17 sample review round 4 called a coverage gap. The bar stays at
    # 0.99: these are BFP4 weights compared against the float checkpoint, which costs ~7e-3 of PCC
    # (the suite log has the per-layer-kind worst values), so a tighter bar would be
    # measuring the dtype rather than the partition. What makes the check sharp is not the bar but
    # the mismatch bound below -- an in-order expert scores 0.99 against its own weights and 0.004
    # against any other, so the two are three orders of magnitude apart.
    per_expert = [pcc(host[e].float(), rebuilt[e].float()) for e in range(g.num_experts)]
    worst = min(per_expert)
    # Disjointness, asserted rather than inferred: expert `e` must match checkpoint expert `e` better
    # than it matches any other expert. Checked against the neighbours a block-boundary error would
    # actually produce -- the same index in another device's block, and the adjacent index.
    confusions = []
    for e in range(0, g.num_experts, 8):
        for other in {(e + g.num_experts // decoder.tp) % g.num_experts, (e + 1) % g.num_experts}:
            confusions.append((e, other, pcc(host[other].float(), rebuilt[e].float())))
    worst_confusion = max(v for _, _, v in confusions)
    logger.info(
        f"multichip expert partition layer={layer_idx}: 4 x {parts[0].shape[1]} experts rebuild the "
        f"{g.num_experts}-expert checkpoint tensor in order, worst per-expert PCC={worst:.6f}, "
        f"best mismatched-expert PCC={worst_confusion:.6f}"
    )
    assert worst > EXPERT_WEIGHT_BAR, f"expert blocks are not the checkpoint's experts in order (worst PCC {worst})"
    assert worst_confusion < 0.5, f"a device's expert block matches the wrong checkpoint expert (PCC {worst_confusion})"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_kv_cache_is_local_heads(mesh_device, layer_idx):
    """The paged cache holds this device's kv head only, and the pair sharing one agrees on it.

    ``n_kv_heads`` is 2 and the mesh is 4 devices, so devices 0,1 own kv head 0 and devices 2,3 own
    kv head 1. After a prefill the two members of a pair must hold *identical* cache contents (same
    kv head, same tokens) and the two pairs must differ — which is what proves the split is by kv
    head rather than by anything that happens to look right.
    """
    source = default_weight_source()
    if layer_idx != FULL_LAYER:
        decoder, _, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
        assert decoder.k_cache is None and decoder.v_cache is None
        pytest.skip("linear_attention has no KV cache")
    decoder, page_table, blocks = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    g = OrnithDecoderConfig.from_hf_config(hf_config())
    assert list(decoder.k_cache.shape) == [blocks, 1, decoder.page_block_size, g.head_dim]
    assert list(decoder.v_cache.shape) == [blocks, 1, decoder.page_block_size, g.head_dim]
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 130, seed=55)), page_table=page_table)
    )
    parts = shards(mesh_device, decoder.k_cache)
    written = [p[: 130 // decoder.page_block_size + 1].float() for p in parts]
    assert torch.equal(written[0], written[1]), "devices 0,1 must hold the same kv head"
    assert torch.equal(written[2], written[3]), "devices 2,3 must hold the same kv head"
    assert not torch.equal(written[0], written[2]), "the two device pairs must hold different kv heads"
    logger.info(
        f"multichip KV cache: per-device {list(decoder.k_cache.shape)} (1 of {g.n_kv_heads} kv heads); "
        f"pairs (0,1) and (2,3) agree, pairs differ"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_collectives_per_forward(mesh_device, layer_idx, monkeypatch):
    """Exactly two collectives per layer forward, in both phases, and no others.

    The design's whole claim rests on this count: one after the row-parallel token mixer and one
    after the MoE. A third would mean an accidental gather (a sharded tensor being restored somewhere
    it should not be), and a first would mean the mixer's partial sums never got reduced.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    calls: list[str] = []

    # `all_gather_async` is watched alongside the `ttnn` collectives so that every spelling the knobs
    # can select is counted. The shipped one is `ttnn.all_reduce` at every shape
    # (doc/optimized_multichip_decoder/work_log.md section 11: the stack-sum crossover's fast
    # spelling is the one that diverges across devices, and of the two correct spellings all_reduce
    # is the faster). The test's claim is about the COUNT and placement of collectives, and the
    # assertion below additionally pins which spelling ships.
    for namespace, name in (
        (ttnn, "all_reduce"),
        (ttnn, "all_gather"),
        (ttnn, "reduce_scatter"),
        (ttnn.experimental, "all_gather_async"),
    ):
        original = getattr(namespace, name)

        def spy(*args, _name=name, _orig=original, **kwargs):
            calls.append(_name)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(namespace, name, spy)

    x = to_device(mesh_device, make_activations(1, 256, seed=63))
    ttnn.deallocate(decoder.prefill_forward(x, page_table=page_table))
    prefill_calls = list(calls)
    calls.clear()
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([256]))
    ttnn.deallocate(
        decoder.decode_forward(
            to_device(mesh_device, make_activations(1, 1, seed=64)),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
    )
    decode_calls = list(calls)
    logger.info(f"multichip collectives layer={layer_idx}: prefill {prefill_calls}, decode {decode_calls}")
    # Prefill is above the stack-sum crossover, so both collectives are `ttnn.all_reduce`.
    assert prefill_calls == ["all_reduce", "all_reduce"], prefill_calls
    # Decode now takes `ttnn.all_reduce` too. The multichip stage's stack-sum crossover was measured
    # against the deprecated `ttnn.all_gather`, and that is the spelling that diverges across devices
    # under sustained traced replay; with it removed, the stable op is both correct and the faster of
    # the two remaining candidates at the decode tile (work_log section 11).
    want = {"all_reduce": "all_reduce", "rs_ag": "reduce_scatter", "stack_sum": "all_gather"}.get(
        MC.CCL_MODE, "all_gather_async"
    )
    # Exact equality, not a prefix-and-length check: two collectives, both the shipped spelling.
    # `rs_ag` is the one mode that lowers to two ops per collective, so it has its own expectation.
    expected = [want, "all_gather", want, "all_gather"] if MC.CCL_MODE == "rs_ag" else [want, want]
    assert decode_calls == expected, decode_calls
    # And the deprecated semaphore-free gather is NOT what the shipped default reaches.
    assert MC.CCL_MODE == "all_reduce", f"the shipped decode collective changed: CCL_MODE is {MC.CCL_MODE!r}"


# --------------------------------------------------------------------------------------
# agreement with the single-chip TTNN baseline
# --------------------------------------------------------------------------------------
def _baseline_pair(mesh_device, layer_idx, source, **kwargs):
    """``(multichip, single-chip)`` decoders built from the same weights on the same mesh.

    The single-chip decoder replicates every weight across the mesh, so device 0 runs exactly the
    single-chip graph and every other device runs an identical copy of it.
    """
    multi = build_decoder(mesh_device, layer_idx, source, **kwargs)
    single = build_decoder(mesh_device, layer_idx, source, cls=OptimizedDecoder, **kwargs)
    return multi, single


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [1, 7, 32, 128, 129, 250, 2048, 2049, 3000])
def test_prefill_matches_single_chip(mesh_device, layer_idx, seq_len):
    """Multi-chip prefill against the single-chip TTNN baseline, aligned and non-aligned lengths."""
    source = default_weight_source()
    (multi, page_table, _), (single, single_pt, _) = _baseline_pair(mesh_device, layer_idx, source)
    x = to_device(mesh_device, make_activations(1, seq_len, seed=100 + seq_len))
    got = to_host(mesh_device, multi.prefill_forward(x, page_table=page_table))
    want = to_host(mesh_device, single.prefill_forward(x, page_table=single_pt))
    value = pcc(want, got)
    logger.info(f"multichip-vs-single prefill layer={layer_idx} seq_len={seq_len} PCC={value:.6f}")
    assert torch.isfinite(got.float()).all()
    assert value > BASELINE_BAR, f"prefill seq_len={seq_len} PCC {value} <= {BASELINE_BAR}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("prefill_len", [130, 2048])
def test_decode_matches_single_chip(mesh_device, layer_idx, prefill_len):
    """Four decode steps after a prefill, each compared to the single-chip TTNN baseline.

    ``prefill_len=130`` puts the decode writes across a 64-token page boundary; 2048 puts them past
    an internal prefill chunk boundary.
    """
    source = default_weight_source()
    (multi, page_table, _), (single, single_pt, _) = _baseline_pair(mesh_device, layer_idx, source)
    steps = 4
    x = to_device(mesh_device, make_activations(1, prefill_len, seed=200 + prefill_len))
    ttnn.deallocate(multi.prefill_forward(x, page_table=page_table))
    ttnn.deallocate(single.prefill_forward(x, page_table=single_pt))
    for step in range(steps):
        d = to_device(mesh_device, make_activations(1, 1, seed=300 + step))
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([prefill_len + step]))
        got = to_host(
            mesh_device,
            multi.decode_forward(d, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table),
        )
        want = to_host(
            mesh_device,
            single.decode_forward(d, current_pos=current_pos, rot_idxs=rot_idxs, page_table=single_pt),
        )
        value = pcc(want, got)
        logger.info(
            f"multichip-vs-single decode layer={layer_idx} prefill_len={prefill_len} step={step} PCC={value:.6f}"
        )
        assert torch.isfinite(got.float()).all()
        assert value > BASELINE_BAR, f"decode step {step} PCC {value} <= {BASELINE_BAR}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_output_is_identical_on_every_device(mesh_device, layer_idx):
    """The stacked-decoder layout contract: input replicated in, identical output on every device.

    This is what lets a stack of these layers pass activations straight through with no boundary
    conversion. It is a **bitwise** check, not a PCC one: the collective leaves every device holding
    the same bytes, and anything weaker would hide a per-device divergence that compounds over 40
    layers.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    out = decoder.prefill_forward(to_device(mesh_device, make_activations(1, 250, seed=71)), page_table=page_table)
    parts = shards(mesh_device, out)
    ttnn.deallocate(out)
    for d in range(1, len(parts)):
        assert torch.equal(parts[0], parts[d]), f"prefill output differs on device {d}"
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([250]))
    dec = decoder.decode_forward(
        to_device(mesh_device, make_activations(1, 1, seed=72)),
        current_pos=current_pos,
        rot_idxs=rot_idxs,
        page_table=page_table,
    )
    dparts = shards(mesh_device, dec)
    ttnn.deallocate(dec)
    for d in range(1, len(dparts)):
        assert torch.equal(dparts[0], dparts[d]), f"decode output differs on device {d}"
    logger.info(
        f"multichip stacked-layer contract layer={layer_idx}: prefill and decode outputs bit-identical "
        f"on all {len(parts)} devices"
    )


# --------------------------------------------------------------------------------------
# agreement with the HF float32 golden
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [1, 7, 32, 64, 128, 129, 250, 2048, 2049, 3000])
def test_prefill_pcc(mesh_device, layer_idx, seq_len):
    """Prefill against the float32 HF golden, at the bar the functional stage set."""
    source = default_weight_source()
    x = make_activations(1, seq_len, seed=13 + seq_len)
    ref_prefill, _ = run_reference(layer_idx, source, x)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    got = to_host(mesh_device, out)
    ttnn.deallocate(out)
    value = pcc(ref_prefill, got)
    logger.info(f"multichip prefill layer={layer_idx} seq_len={seq_len} PCC={value:.6f}")
    assert value > PCC_BAR, f"prefill PCC {value} <= {PCC_BAR}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("prefill_len", [130, 2048])
def test_decode_pcc(mesh_device, layer_idx, prefill_len):
    """Four decode steps after a prefill, against the float32 HF golden."""
    source = default_weight_source()
    steps = 4
    x = make_activations(1, prefill_len, seed=23 + prefill_len)
    decode_x = [make_activations(1, 1, seed=2300 + i) for i in range(steps)]
    _, ref_decode = run_reference(layer_idx, source, x, decode_x=decode_x, decode_steps=steps)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    ttnn.deallocate(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))
    for step in range(steps):
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([prefill_len + step]))
        out = decoder.decode_forward(
            to_device(mesh_device, decode_x[step]),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
        got = to_host(mesh_device, out)
        ttnn.deallocate(out)
        value = pcc(ref_decode[step], got)
        logger.info(f"multichip decode layer={layer_idx} prefill_len={prefill_len} step={step} PCC={value:.6f}")
        assert value > PCC_BAR, f"decode step {step} PCC {value} <= {PCC_BAR}"


# --------------------------------------------------------------------------------------
# paged KV cache, page tables, positions, batching
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_permuted_page_table(mesh_device, layer_idx):
    """A non-identity page table must give the same answer as the identity one."""
    source = default_weight_source()
    if layer_idx != FULL_LAYER:
        pytest.skip("linear_attention has no page table")
    seq_len = 300
    x = make_activations(1, seq_len, seed=45)
    decode_x = make_activations(1, 1, seed=46)
    ref_prefill, ref_decode = run_reference(layer_idx, source, x, decode_x=[decode_x], decode_steps=1)

    decoder, _, blocks = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    permuted = torch.randperm(blocks, generator=torch.Generator().manual_seed(9)).to(torch.int32).reshape(1, blocks)
    page_table = to_device(mesh_device, permuted, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    value = pcc(ref_prefill, to_host(mesh_device, out))
    ttnn.deallocate(out)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
    dec = decoder.decode_forward(
        to_device(mesh_device, decode_x), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
    )
    dvalue = pcc(ref_decode[0], to_host(mesh_device, dec))
    ttnn.deallocate(dec)
    logger.info(f"multichip permuted page table layer={layer_idx}: prefill PCC={value:.6f} decode PCC={dvalue:.6f}")
    assert value > PCC_BAR and dvalue > PCC_BAR


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("batch", [1, 4, 13, 32])
@pytest.mark.parametrize("seq_len", [192, 130])
def test_batched_prefill_decode_pcc(mesh_device, layer_idx, batch, seq_len):
    """Batched prefill and decode, per-user page tables, against the HF golden.

    The batch list is the one ``doc/context_contract.json`` advertises, and it is exercised **on the
    mesh** rather than inherited from the single-chip stage: TP=4 changes the per-device head counts
    that every batch-sensitive decode op is bounded by — ``nlp_create_qkv_heads_decode`` (a 32-user
    op limit), ``paged_fused_update_cache`` (2 x batch cores) and ``sdpa_decode`` (one core per
    page-table row) all see ``n_heads`` 16 -> 4 and ``n_kv_heads`` 2 -> 1 here. Review round 1 of this
    stage found this test pinned at batch 4 while three documents claimed batch 32, which is exactly
    the class of defect the contract's own notes record being caught twice before.

    ``seq_len`` 130 is not a multiple of the 32-token tile. Round 2 of the review added it believing
    it would exercise the prefill side of the ``CCL_COMPACT_ROWS`` fold; round 8's correctness audit
    showed it cannot — ``prefill_forward`` pads every chunk to ``PREFILL_ALIGN`` = 128 rows before the
    layer runs, so a prefill operand's physical row count always equals ``align_up(b * t, 32)`` and
    the fold's strict-greater guard is false at every prefill shape and batch. The fold is
    decode-only, and §5.9's decode batches are where it is covered. The parameter stays because it is
    the suite's only batched **non-aligned prefill** coverage, which is worth having on its own.
    """
    source = default_weight_source()
    x = make_activations(batch, seq_len, seed=51 + seq_len)
    decode_x = make_activations(batch, 1, seed=52 + seq_len)
    ref_prefill, ref_decode = run_reference(layer_idx, source, x, decode_x=[decode_x], decode_steps=1)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, batch=batch, max_context=1024)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    value = pcc(ref_prefill, to_host(mesh_device, out))
    ttnn.deallocate(out)
    positions = torch.full((batch,), seq_len, dtype=torch.int32)
    current_pos, rot_idxs = decode_inputs(mesh_device, positions)
    dec = decoder.decode_forward(
        to_device(mesh_device, decode_x), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
    )
    dvalue = pcc(ref_decode[0], to_host(mesh_device, dec))
    ttnn.deallocate(dec)
    logger.info(
        f"multichip batched layer={layer_idx} batch={batch} seq_len={seq_len}: "
        f"prefill PCC={value:.6f} decode PCC={dvalue:.6f}"
    )
    assert value > PCC_BAR and dvalue > PCC_BAR


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_batched_decode_ragged_positions(mesh_device, layer_idx):
    """Users at different positions decode independently and correctly."""
    source = default_weight_source()
    if layer_idx != FULL_LAYER:
        pytest.skip("ragged per-user positions only apply to the paged KV cache")
    batch = 4
    lengths = [37, 130, 200, 64]
    decoder, page_table, blocks = build_decoder(mesh_device, layer_idx, source, batch=batch, max_context=1024)
    outs = []
    for user, length in enumerate(lengths):
        x = make_activations(1, length, seed=600 + user)
        table = torch.arange(blocks * batch, dtype=torch.int32).reshape(batch, blocks)[user : user + 1]
        pt = to_device(mesh_device, table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(decoder.prefill_forward(to_device(mesh_device, x), page_table=pt, start_pos=0))
        ref_prefill, ref_decode = run_reference(
            layer_idx, source, x, decode_x=[make_activations(1, 1, seed=700 + user)], decode_steps=1
        )
        outs.append(ref_decode[0])
    decode_x = torch.cat([make_activations(1, 1, seed=700 + u) for u in range(batch)], dim=0)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor(lengths, dtype=torch.int32))
    got = to_host(
        mesh_device,
        decoder.decode_forward(
            to_device(mesh_device, decode_x), current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
        ),
    )
    for user in range(batch):
        value = pcc(outs[user], got[user : user + 1])
        logger.info(f"multichip ragged decode layer={layer_idx} user={user} pos={lengths[user]} PCC={value:.6f}")
        assert value > PCC_BAR, f"user {user} PCC {value}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_prefill_continuation(mesh_device, layer_idx):
    """Two prefill calls over one sequence equal one call over the concatenation.

    Against **both** references, because they answer different questions: the HF golden says the
    continuation is right, and a single 258-token TTNN prefill in the same process says the split
    changed nothing about *this* implementation. Review round 4 found the second comparison claimed
    in the docstring and not made.

    The continued segment is deliberately **130 tokens** — not a multiple of the 32-token tile, the
    64-token page or the 128-token chunk — because that is the freedom the public contract has to
    keep. ``start_pos`` itself is a different matter: ``OptimizedDecoder.prefill_forward`` requires
    ``start_pos % chunk_size == 0`` and this stage inherits that unchanged, so a caller resumes on
    chunk boundaries and may end anywhere. :func:`test_prefill_continuation_rejects_unaligned_start`
    pins that boundary as a clean error rather than a silent wrong answer, and
    ``doc/context_contract.json`` records it.
    """
    source = default_weight_source()
    chunk = 128
    tail = 130
    total = chunk + tail
    x = make_activations(1, total, seed=21)
    ref_prefill, _ = run_reference(layer_idx, source, x)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, prefill_chunk=chunk)
    x_tt = to_device(mesh_device, x)
    first = decoder.prefill_forward(
        ttnn.slice(x_tt, [0, 0, 0], [1, chunk, hf_config().hidden_size]),
        start_pos=0,
        page_table=page_table,
        chunk_size=chunk,
    )
    second = decoder.prefill_forward(
        ttnn.slice(x_tt, [0, chunk, 0], [1, total, hf_config().hidden_size]),
        start_pos=chunk,
        page_table=page_table,
        chunk_size=chunk,
    )
    got = torch.cat([to_host(mesh_device, first), to_host(mesh_device, second)], dim=1)
    decoder.reset_state()
    whole = to_host(mesh_device, decoder.prefill_forward(x_tt, page_table=page_table, chunk_size=chunk))
    golden = pcc(ref_prefill, got)
    split = pcc(whole, got)
    logger.info(
        f"multichip prefill continuation layer={layer_idx} tail={tail}: "
        f"PCC vs HF golden={golden:.6f} PCC vs one call={split:.6f}"
    )
    assert golden > PCC_BAR
    assert split > BASELINE_BAR, f"splitting the prefill changed the result (PCC {split})"


@pytest.mark.parametrize("layer_idx", LAYERS[:1], ids=lambda i: LAYER_IDS[i])
def test_prefill_continuation_rejects_unaligned_start(mesh_device, layer_idx, expect_error):
    """A resume at a non-chunk-aligned ``start_pos`` is refused, not silently mis-cached.

    The inherited public contract: ``chunk_size`` is a multiple of ``PREFILL_ALIGN`` and ``start_pos``
    a multiple of ``chunk_size``. This stage adds no restriction of its own — sharding does not
    narrow it — but review round 4 pointed out that the restriction existed, that no document
    recorded it, and that README section 4.4 described this stage's continuation test as covering the
    case the API rejects.
    """
    source = default_weight_source()
    chunk = 128
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, prefill_chunk=chunk)
    x_tt = to_device(mesh_device, make_activations(1, chunk, seed=22))
    with expect_error(ValueError, "must be a multiple of chunk_size"):
        decoder.prefill_forward(x_tt, start_pos=chunk // 2, page_table=page_table, chunk_size=chunk)


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_unaligned_max_context(mesh_device, layer_idx):
    """A ``max_context`` that is not a multiple of the internal prefill alignment must work.

    Against the HF golden, not just against finiteness: 5000 is well inside the range the reference
    can run (``test_long_context_pcc`` takes it to 8000), and review round 3 pointed out that one of
    the three levels at which this stage claims non-aligned support was asserting shape and variance
    only. The cache write is what the unaligned ``max_context`` actually stresses, so the decode step
    at slot ``max_context - 1`` is checked against the reference's continuation as well.
    """
    source = default_weight_source()
    max_context = 5000
    x = make_activations(1, max_context, seed=77)
    decode_x = make_activations(1, 1, seed=78)
    ref_prefill, ref_decode = run_reference(layer_idx, source, x, decode_x=[decode_x], decode_steps=1)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=max_context)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    assert list(out.shape) == [1, max_context, hf_config().hidden_size]
    host_out = to_host(mesh_device, out)
    ttnn.deallocate(out)
    value = pcc(ref_prefill, host_out)
    tail = host_out[:, max_context - 64 :, :].float()
    assert torch.isfinite(tail).all()
    assert tail.std() > 0.01
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([max_context - 1]))
    dec = decoder.decode_forward(
        to_device(mesh_device, decode_x),
        current_pos=current_pos,
        rot_idxs=rot_idxs,
        page_table=page_table,
    )
    host_dec = to_host(mesh_device, dec)
    ttnn.deallocate(dec)
    dvalue = pcc(ref_decode[0], host_dec)
    assert torch.isfinite(host_dec.float()).all()
    logger.info(
        f"multichip unaligned max_context={max_context} layer={layer_idx}: "
        f"prefill PCC={value:.6f} decode PCC={dvalue:.6f}"
    )
    assert value > PCC_BAR and dvalue > PCC_BAR


# --------------------------------------------------------------------------------------
# MoE / routing under expert parallelism
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_local_routing_selects_this_devices_experts(mesh_device, layer_idx):
    """The router's device-local score vector is the right 64-wide slice of the global 256.

    Concatenating the four devices' local routing must reproduce the dense 256-wide vector the
    single-chip router produces from the same activation, exactly — the selection is a one-hot
    matmul over the replicated scores, so "exactly" is the right bar, not a PCC one.
    """
    source = default_weight_source()
    decoder, _, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    single, _, _ = build_decoder(mesh_device, layer_idx, source, cls=OptimizedDecoder, max_context=1024)
    x = to_device(mesh_device, make_activations(1, 32, seed=81).reshape(1, 1, 32, hf_config().hidden_size))
    decoder.moe._decode_phase = False
    single.moe._decode_phase = False
    local = decoder.moe.routing_weights(x)
    want = to_host(mesh_device, single.moe.routing_weights(x))
    got = torch.cat(shards(mesh_device, local), dim=-1)
    ttnn.deallocate(local)
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    assert torch.equal(got, want), "the concatenated device-local routing is not the global routing"
    nonzero = int((want != 0).sum())
    logger.info(
        f"multichip routing layer={layer_idx}: 4 x {decoder.cfg.num_experts} local scores rebuild the "
        f"{decoder.global_cfg.num_experts}-wide global vector exactly ({nonzero} non-zero entries over 32 tokens)"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_zero_local_active_experts(mesh_device, layer_idx):
    """A device whose expert block wins none of the global top-8 must still produce the right answer.

    With 8 experts drawn from 256 over four blocks this happens for about one device in ten per
    token, so it is a *routine* case, not an edge one. The sparsity mask is floored at local expert
    0 (:data:`MOE_MASK_FLOOR`) so the sparse matmul is never handed an all-zero sparsity; the floored
    expert's score is zero, so its contribution is exactly zero. This test forces the case by hand
    rather than waiting for it, and checks both halves: the mask has exactly one entry, and the
    routed output for that device is exactly zero.
    """
    source = default_weight_source()
    decoder, _, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    moe = decoder.moe
    e_local = decoder.cfg.num_experts
    # Routing that puts every selected expert on device 0's block, so devices 1..3 get none.
    host = torch.zeros(1, 1, TILE, e_local)
    per_device = [host.clone() for _ in range(mesh_device.get_num_devices())]
    per_device[0][..., :8] = 1.0 / 8
    dense = ttnn.from_torch(
        torch.cat(per_device, dim=0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=0),
    )
    mask = moe._active_expert_mask(dense, 1, None)
    mask_shards = shards(mesh_device, mask)
    assert int((mask_shards[0] != 0).sum()) == 8, mask_shards[0]
    for d in range(1, len(mask_shards)):
        assert int((mask_shards[d] != 0).sum()) == 1, f"device {d} mask should be floored to exactly one expert"
        assert mask_shards[d].flatten()[0] != 0, f"device {d}'s floored expert should be local expert 0"

    x = to_device(mesh_device, make_activations(1, TILE, seed=83).reshape(1, 1, TILE, hf_config().hidden_size))
    moe._decode_phase = True
    moe._call_tokens = TILE
    routed = moe._routed_experts(x, dense, TILE, valid_tokens=None)
    parts = shards(mesh_device, routed)
    ttnn.deallocate(routed)
    ttnn.deallocate(mask)
    ttnn.deallocate(dense)
    for d in range(1, len(parts)):
        assert torch.count_nonzero(parts[d]) == 0, f"device {d} contributed {torch.count_nonzero(parts[d])} non-zeros"
    assert torch.count_nonzero(parts[0]) > 0, "device 0 should have produced the whole routed output"
    logger.info(
        f"multichip zero-local-expert case layer={layer_idx}: devices 1-3 floored to 1 expert each and "
        f"contributed exactly zero; device 0 carried all 8"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_gate_selected_experts_not_dense(mesh_device, layer_idx, monkeypatch):
    """The routed path stays gate-selected: the sparse matmul is dispatched with a real sparsity.

    A multichip MoE that quietly ran all 64 local experts densely would still be numerically right
    and much slower; this asserts the shipped path is the sparse one and that the sparsity it is
    given holds far fewer than the local expert count.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    seen: list[int] = []
    original = ttnn.sparse_matmul

    shapes: list[tuple[int, ...]] = []

    def spy(*args, sparsity=None, **kwargs):
        parts = shards(mesh_device, sparsity)
        # Per **group**, not per call: the sparsity tensor is [1, groups, 1, E] and the op loops once
        # per active expert within each group, so the per-group count is what `--active-experts`
        # models. Review round 2 found the profiling scripts calibrated against a per-call total.
        groups = max(1, int(parts[0].shape[1]))
        seen.append(max(int(torch.count_nonzero(p)) for p in parts) / groups)
        shapes.append(tuple(int(d) for d in parts[0].shape))
        return original(*args, sparsity=sparsity, **kwargs)

    monkeypatch.setattr(ttnn, "sparse_matmul", spy)
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 128, seed=91)), page_table=page_table)
    )
    prefill_seen = list(seen)
    seen.clear()
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([128]))
    ttnn.deallocate(
        decoder.decode_forward(
            to_device(mesh_device, make_activations(1, 1, seed=92)),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
    )
    logger.info(
        f"multichip active-expert path layer={layer_idx}: decode sparsity max non-zeros per device "
        f"per group {seen}, prefill {prefill_seen[:4]}... over {len(prefill_seen)} calls, sparsity "
        f"shapes {sorted(set(shapes))}, local expert count {decoder.cfg.num_experts}"
    )
    assert seen, "no sparse_matmul was dispatched in decode — the routed path is not the sparse one"
    top_k = decoder.global_cfg.num_experts_per_tok
    # `top_k + 1`, not `top_k`: `MOE_MASK_FLOOR` floors the mask at local expert 0, so a device whose
    # block holds all `top_k` selected experts *and* does not include local expert 0 among them runs
    # one more. That is ~6e-5 per call rather than impossible, and round 8's correctness audit found
    # the tighter bound would have been a rare flake rather than a real assertion.
    assert max(seen) <= top_k + 1, (
        f"decode activated {max(seen)} local experts, more than the global top-{top_k} plus the " "floored expert"
    )
    assert max(seen) < decoder.cfg.num_experts, "decode is running every local expert, i.e. densely"

    # The prefill count is asserted too, because it is a *modelling input* elsewhere and not just a
    # sanity check: `tracy/run_profiling.sh` passes it to `tt-perf-report --active-experts`, and
    # `probe_sparse_matmul_local.py` sweeps at it. Review round 2 found the profiling scripts using
    # 63 — the uniform-draw expectation `E*(1-(1-1/E)^(32*top_k))` — against a measured 39-44, which
    # is a ~50% overestimate feeding every prefill DRAM/FLOPs figure. The bound below is the
    # structural one (a 32-token group cannot activate more than 64 local experts, and must activate
    # at least one per token's worth of routing); the exact figure lives in the log line above, which
    # is what the scripts are calibrated against.
    assert prefill_seen, "no sparse_matmul was dispatched in prefill"
    assert (
        max(prefill_seen) < decoder.cfg.num_experts
    ), f"prefill activated {max(prefill_seen)} of the {decoder.cfg.num_experts} local experts, i.e. densely"
    # And the *band* the modelling input sits in, not just "less than dense". Review round 4 pointed
    # out that the structural bound above would still pass if the count drifted back to round 2's 63,
    # which is what `--active-experts` was wrongly set to and which feeds every prefill DRAM and FLOPs
    # figure. The band is wide enough for run-to-run routing variation on random activations and far
    # too narrow to admit the uniform-draw over-count.
    assert PREFILL_ACTIVE_BAND[0] <= max(prefill_seen) <= PREFILL_ACTIVE_BAND[1], (
        f"prefill activated {max(prefill_seen)} local experts per group, outside the band "
        f"{PREFILL_ACTIVE_BAND} that `tracy/run_profiling.sh --active-experts "
        f"{PREFILL_ACTIVE_MODEL}` and `probe_sparse_matmul_local.py` are calibrated against"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_routing_select_modes_agree(mesh_device, layer_idx, monkeypatch):
    """``gather`` and the shipped ``select_matmul`` narrowing agree, on the vector and on the layer.

    Two comparisons, because the routing vector is not the deliverable: the local scores feed a
    sparsity mask and a score multiply, so a narrowing that produced the right vector in the wrong
    dtype or layout could still move the layer. Review round 4 found this test comparing only the
    vector while README section 4.3 claimed the layer output, and running a ``select_matmul`` arm
    against itself — a parametrization where half the cases were tautological. The reference arm is
    now fixed at ``select_matmul`` and the single compared arm is ``gather``.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    x = to_device(mesh_device, make_activations(1, 32, seed=84).reshape(1, 1, 32, hf_config().hidden_size))
    layer_x = to_device(mesh_device, make_activations(1, 128, seed=86))
    decoder.moe._decode_phase = False
    monkeypatch.setattr(MC, "ROUTING_SELECT_MODE", "select_matmul")
    want = torch.cat(shards(mesh_device, decoder.moe.routing_weights(x)), dim=-1)
    decoder.reset_state()
    want_layer = to_host(mesh_device, decoder.prefill_forward(layer_x, page_table=page_table))
    monkeypatch.setattr(MC, "ROUTING_SELECT_MODE", "gather")
    got = torch.cat(shards(mesh_device, decoder.moe.routing_weights(x)), dim=-1)
    decoder.reset_state()
    got_layer = to_host(mesh_device, decoder.prefill_forward(layer_x, page_table=page_table))
    layer_value = pcc(want_layer, got_layer)
    logger.info(
        f"multichip routing select gather vs select_matmul layer={layer_idx}: "
        f"routing vectors bit-equal = {torch.equal(got, want)}, layer PCC = {layer_value:.6f}"
    )
    assert torch.equal(got, want), "ROUTING_SELECT_MODE=gather disagrees with select_matmul"
    assert layer_value > BASELINE_BAR, f"the narrowing spelling moved the layer (PCC {layer_value})"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("candidate", ["fused_gate", "fused_gate_local"])
def test_router_modes_agree(mesh_device, layer_idx, candidate, monkeypatch):
    """The fused router gate and the inherited ``topk`` chain agree on the layer and on the route.

    ``ROUTER_MODE="fused_gate"`` is a **precision change**, not only a fusion: the
    ``generalized_moe_gate`` kernel reads bfloat16 logits where the ``topk`` chain reads float32, so
    the two can in principle select different experts near the top-8/top-9 boundary. That is exactly
    the thing a router change has to be measured on, because a swapped expert is a different
    computation and not a rounded value.

    Both quantities are checked on real checkpoint weights and on the traced-shape decode path: the
    per-token selected-expert SET taken from the dense routing vector, and the layer output against
    the same float32 HF golden the rest of the suite uses. The optimized stage rejected this op
    without timing it and without this measurement; this test is what makes the shipped default
    answerable.
    """
    source = default_weight_source()
    prefill_len, steps = 128, 8
    x = make_activations(1, prefill_len, seed=61)
    decode_x = [make_activations(1, 1, seed=6100 + i) for i in range(steps)]
    _, ref_decode = run_reference(layer_idx, source, x, decode_x=decode_x, decode_steps=steps)

    def run(mode):
        monkeypatch.setattr(MC, "ROUTER_MODE", mode)
        decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
        # `allocate_state` is where the gate buffers are built, and `build_decoder` has already run
        # it under whatever mode was active then; re-run it so the mode under test is the one the
        # buffers (and therefore the decode path) reflect.
        decoder.allocate_state(1)
        ttnn.deallocate(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))
        outs, routes = [], []
        for step in range(steps):
            current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([prefill_len + step]))
            decoder.moe._captured_route = None
            out = decoder.decode_forward(
                to_device(mesh_device, decode_x[step]),
                current_pos=current_pos,
                rot_idxs=rot_idxs,
                page_table=page_table,
            )
            outs.append(to_host(mesh_device, out))
            routes.append(decoder.moe._captured_route)
            ttnn.deallocate(out)
        del decoder, page_table
        return outs, routes

    # Capture the device-local routing vector each step without changing the forward: the wrapper
    # stashes what `routing_weights` returned, which is the narrowed 64-wide block this device runs.
    # It is read back to HOST here rather than kept as a handle - `OptimizedMoE.forward` frees the
    # routing tensor before the caller ever sees it, so holding the handle is a use-after-free (the
    # first version of this test segfaulted on exactly that).
    original = MC.MultichipMoE.routing_weights

    def capture(self, x_in):
        out = original(self, x_in)
        self._captured_route = torch.cat(shards(mesh_device, out), dim=-1)
        return out

    monkeypatch.setattr(MC.MultichipMoE, "routing_weights", capture)

    want_outs, want_routes = run("topk")
    got_outs, got_routes = run(candidate)

    agree, total = 0, 0
    for step in range(steps):
        a = (want_routes[step][0, 0, 0] != 0).nonzero().flatten().tolist()
        b = (got_routes[step][0, 0, 0] != 0).nonzero().flatten().tolist()
        agree += int(set(a) == set(b))
        total += 1
        want_value = pcc(ref_decode[step], want_outs[step])
        got_value = pcc(ref_decode[step], got_outs[step])
        cross = pcc(want_outs[step], got_outs[step])
        logger.info(
            f"multichip router modes layer={layer_idx} candidate={candidate} step={step}: "
            f"topk-vs-golden PCC {want_value:.6f}, "
            f"{candidate}-vs-golden PCC {got_value:.6f}, {candidate}-vs-topk PCC {cross:.6f}, "
            f"expert sets {'equal' if set(a) == set(b) else f'{a} vs {b}'}"
        )
        assert got_value > PCC_BAR, f"{candidate} decode step {step} PCC {got_value} <= {PCC_BAR}"
        assert cross > BASELINE_BAR, f"the two router modes disagree at the layer (PCC {cross})"
    logger.info(
        f"multichip router modes layer={layer_idx} candidate={candidate}: "
        f"identical expert set on {agree}/{total} decode steps"
    )
    # ASSERTED, not only logged. The whole defence of a float32 -> bfloat16 router-logit change is
    # that expert selection is discrete and does not move; review round 1 of this stage pointed out
    # that the documents claimed the change was "gated" on this quantity while the test merely
    # printed it, so a future change that swapped one expert on one step would still have passed.
    assert agree == total, (
        f"{candidate} selected a different expert set on {total - agree}/{total} decode steps; "
        "the router-mode change is only admissible while the discrete selection is unchanged"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_decode_runs_the_fused_router_gate(mesh_device, layer_idx, monkeypatch):
    """The shipped default actually reaches the kernel, and prefill actually does not.

    A policy that only exists in a module constant is not implemented. This pins both halves of
    :data:`~...multichip_decoder.ROUTER_MODE`'s contract in the measured runtime path: exactly one
    ``generalized_moe_gate`` call and **no** ``ttnn.topk`` in a decode forward, and the reverse in a
    prefill forward, where the gate op's one-token-per-core shape would need 19 sequential calls for
    a 2048-token chunk and ``TopK`` is 0.17% of the window.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    counts = {"gate": 0, "topk": 0}
    gate_original = ttnn.experimental.deepseek.moe.generalized_moe_gate
    topk_original = ttnn.topk

    def gate_spy(*args, **kwargs):
        counts["gate"] += 1
        return gate_original(*args, **kwargs)

    def topk_spy(*args, **kwargs):
        counts["topk"] += 1
        return topk_original(*args, **kwargs)

    monkeypatch.setattr(ttnn.experimental.deepseek.moe, "generalized_moe_gate", gate_spy)
    monkeypatch.setattr(ttnn, "topk", topk_spy)

    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 128, seed=63)), page_table=page_table)
    )
    prefill_counts = dict(counts)
    counts["gate"] = counts["topk"] = 0
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([128]))
    ttnn.deallocate(
        decoder.decode_forward(
            to_device(mesh_device, make_activations(1, 1, seed=64)),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
    )
    logger.info(f"multichip router op census layer={layer_idx}: prefill {prefill_counts}, decode {counts}")
    assert counts["gate"] == 1, f"decode ran {counts['gate']} generalized_moe_gate calls, expected exactly 1"
    assert counts["topk"] == 0, f"decode still ran {counts['topk']} ttnn.topk calls"
    assert prefill_counts["gate"] == 0, "prefill ran the one-token-per-core gate op"
    assert prefill_counts["topk"] == 1, f"prefill ran {prefill_counts['topk']} ttnn.topk calls, expected 1"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("batch", [1, 4, 32, 40, 56])
def test_fused_router_gate_covers_every_supported_batch(mesh_device, layer_idx, batch, monkeypatch):
    """The fused gate is reached at every supported batch, not only at batch 1.

    ``routing_weights`` falls back to the ``topk`` chain **silently** when the gate buffers for a
    decode row count were not prepared, which is correct and slower. A silent fallback that nothing
    checks is a performance cliff nobody would notice, so the row counts the advertised contract can
    produce are pinned: ``align_up(batch, 32)`` is 32 up to batch 32 and 64 at the 40 and 56 the
    single-chip suite exercises above the dedicated decode op's limit. All are far below the 110-core
    ceiling. Review round 1 of this stage found only batch 1 covered.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024, batch=batch)
    counts = {"gate": 0, "topk": 0}
    gate_original = ttnn.experimental.deepseek.moe.generalized_moe_gate
    topk_original = ttnn.topk

    def gate_spy(*args, **kwargs):
        counts["gate"] += 1
        return gate_original(*args, **kwargs)

    def topk_spy(*args, **kwargs):
        counts["topk"] += 1
        return topk_original(*args, **kwargs)

    ttnn.deallocate(
        decoder.prefill_forward(
            to_device(mesh_device, make_activations(batch, 128, seed=65 + batch)), page_table=page_table
        )
    )
    monkeypatch.setattr(ttnn.experimental.deepseek.moe, "generalized_moe_gate", gate_spy)
    monkeypatch.setattr(ttnn, "topk", topk_spy)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.full((batch,), 128, dtype=torch.int32))
    ttnn.deallocate(
        decoder.decode_forward(
            to_device(mesh_device, make_activations(batch, 1, seed=66 + batch)),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
    )
    rows = ((batch + TILE - 1) // TILE) * TILE
    logger.info(f"multichip router op census layer={layer_idx} batch={batch} decode rows={rows}: {counts}")
    assert counts["gate"] == 1, (
        f"decode at batch {batch} ({rows} router rows) ran {counts['gate']} gate calls and "
        f"{counts['topk']} ttnn.topk calls - the fused gate silently fell back"
    )
    assert counts["topk"] == 0


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("phase", ["decode", "prefill"])
@pytest.mark.parametrize("mode", ["rs_ag", "stack_sum", "stack_sum_async"])
def test_ccl_modes_agree(mesh_device, layer_idx, monkeypatch, mode, phase):
    """Every collective spelling produces the same layer output, so the knob is a latency knob.

    Run on **both sides of the ``auto`` crossover**, which means running both *phases* rather than two
    prefill lengths. Round 4 found the original single-shape version comparing the shipped path with
    itself; round 8's correctness audit found the two-length replacement doing the same thing, for a
    subtler reason: ``prefill_forward`` pads every chunk to ``PREFILL_ALIGN`` = 128 rows **before**
    the layer runs, so a 32-token prefill still hands the collective 128 physical rows and ``auto``
    still resolves to ``all_reduce``. No prefill shape can reach the ``stack_sum`` regime at all.

    Decode can: a batch-1 step is one 32-row tile, so ``auto`` resolves to a stack-sum spelling there
    while the shipped ``CCL_MODE`` is ``all_reduce`` at every shape. The **reference arm is the
    shipped default**, and the parametrized arms are the alternatives, so no case compares the
    shipped path with itself — the tautology round 4 and round 8 of the previous stage each caught a
    version of. ``all_reduce`` is therefore absent from the arm list: it *is* the reference now.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    seen = []
    original = MC.MultichipDecoder._all_reduce

    def spy(self, tensor):
        seen.append(_physical_rows(tensor.shape))
        return original(self, tensor)

    monkeypatch.setattr(MC.MultichipDecoder, "_all_reduce", spy)

    def run():
        decoder.reset_state()
        prefilled = decoder.prefill_forward(
            to_device(mesh_device, make_activations(1, 128, seed=85)), page_table=page_table
        )
        if phase == "prefill":
            return to_host(mesh_device, prefilled)
        ttnn.deallocate(prefilled)
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([128]))
        return to_host(
            mesh_device,
            decoder.decode_forward(
                to_device(mesh_device, make_activations(1, 1, seed=86)),
                current_pos=current_pos,
                rot_idxs=rot_idxs,
                page_table=page_table,
            ),
        )

    want = run()  # the shipped default, whatever CCL_MODE is set to at module level
    rows = seen[-1] if phase == "decode" else max(seen)
    resolved = MC.AUTO_STACK_SUM_MODE if rows <= MC.CCL_STACK_SUM_MAX_ROWS else "all_reduce"
    seen.clear()
    monkeypatch.setattr(MC, "CCL_MODE", mode)
    got = run()
    value = pcc(want, got)
    logger.info(
        f"multichip CCL_MODE={mode} layer={layer_idx} phase={phase}: last collective saw {rows} "
        f"physical rows so `auto` would resolve to {resolved}; PCC vs the shipped default "
        f"({MC.CCL_MODE}) = {value:.6f}"
    )
    assert value > 0.9999, f"CCL_MODE={mode} changed the result (PCC {value})"
    if phase == "decode":
        assert rows <= MC.CCL_STACK_SUM_MAX_ROWS, (
            f"decode's collective saw {rows} physical rows, so the `auto`/`stack_sum` arms are not "
            "covering the crossover regime this test exists for"
        )


# --------------------------------------------------------------------------------------
# trace, determinism, stress, runtime audit
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_traced_decode_pcc(mesh_device, layer_idx):
    """Decode replays from a captured trace on the target mesh, and the replay matches HF."""
    source = default_weight_source()
    prefill_len = 128
    steps = 3
    x = make_activations(1, prefill_len, seed=31)
    decode_x = [make_activations(1, 1, seed=3100 + i) for i in range(steps)]
    _, ref_decode = run_reference(layer_idx, source, x, decode_x=decode_x, decode_steps=steps)

    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    ttnn.deallocate(decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table))

    x_buf = to_device(mesh_device, decode_x[0])
    pos_buf, rot_buf = decode_inputs(mesh_device, torch.tensor([prefill_len]))

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

    saved = _snapshot_state(mesh_device, decoder)
    ttnn.deallocate(forward())
    ttnn.synchronize_device(mesh_device)
    _restore_state(mesh_device, decoder, saved)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = forward()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    _restore_state(mesh_device, decoder, saved)

    for step in range(steps):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                decode_x[step],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            x_buf,
        )
        position = torch.tensor([prefill_len + step], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                position,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            pos_buf,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                position.reshape(1, -1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            rot_buf,
        )
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        parts = shards(mesh_device, trace_out)
        for d in range(1, len(parts)):
            assert torch.equal(parts[0], parts[d]), f"traced decode output differs on device {d}"
        value = pcc(ref_decode[step], parts[0])
        logger.info(f"multichip traced decode layer={layer_idx} step={step} replay PCC={value:.6f}")
        assert torch.isfinite(parts[0].float()).all()
        assert value > PCC_BAR, f"traced decode replay step {step} PCC {value} <= {PCC_BAR}"

    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_determinism_repeated_inputs(mesh_device, layer_idx):
    """Identical inputs give bit-identical outputs, in prefill and in decode, on every device.

    Determinism is a sharper question here than on one chip: a collective that reduced in a
    data-dependent order, or a routing decision that differed per device, would show up as a
    non-repeating result even though every individual op is deterministic.
    """
    source = default_weight_source()
    prefill_len = 256
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    x_tt = to_device(mesh_device, make_activations(1, prefill_len, seed=11))
    d_tt = to_device(mesh_device, make_activations(1, 1, seed=12))
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([prefill_len]))

    results = []
    for _ in range(3):
        decoder.reset_state()
        prefill = shards(mesh_device, decoder.prefill_forward(x_tt, page_table=page_table))
        decode = shards(
            mesh_device, decoder.decode_forward(d_tt, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)
        )
        results.append((prefill, decode))

    for idx in (1, 2):
        for d in range(mesh_device.get_num_devices()):
            assert torch.equal(results[0][0][d], results[idx][0][d]), f"prefill run {idx} device {d} differs"
            assert torch.equal(results[0][1][d], results[idx][1][d]), f"decode run {idx} device {d} differs"
    logger.info(f"multichip determinism layer={layer_idx}: 3/3 runs bit-identical on all 4 devices")


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_traced_replay_does_not_leak(mesh_device, layer_idx):
    """A long run of trace replays allocates nothing after the first.

    ``test_repeated_run_stress`` bounds DRAM growth over the **untraced** path. The traced path is the
    one the next stage actually replays, and it is where a CCL semaphore or a persistent all-gather
    buffer allocated per replay would hide: a captured trace re-runs the same program, so a leak here
    is invisible to PCC and to the untraced stress test. Review round 1 of this stage named this gap.

    128 replays, and the allocation is sampled after the first 8 so trace-region warm-up is excluded
    from the baseline rather than from the assertion.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    prefill_len = 128
    ttnn.deallocate(
        decoder.prefill_forward(
            to_device(mesh_device, make_activations(1, prefill_len, seed=44)), page_table=page_table
        )
    )
    x_buf = to_device(mesh_device, make_activations(1, 1, seed=45))
    pos_buf, rot_buf = decode_inputs(mesh_device, torch.tensor([prefill_len]))

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

    ttnn.deallocate(forward())
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = forward()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    replays = 128
    warmup = 8
    allocations = []
    for replay in range(replays):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        allocations.append(ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)
    parts = shards(mesh_device, trace_out)
    ttnn.release_trace(mesh_device, trace_id)

    growth = allocations[-1] - allocations[warmup]
    logger.info(
        f"multichip traced replay leak layer={layer_idx}: {replays} replays, DRAM allocated "
        f"{allocations[warmup]} -> {allocations[-1]} bytes (growth {growth})"
    )
    assert torch.isfinite(parts[0].float()).all()
    for d in range(1, len(parts)):
        assert torch.equal(parts[0], parts[d]), f"replayed output differs on device {d}"
    assert growth == 0, f"DRAM allocation grew by {growth} bytes over {replays} trace replays"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_chunk_size_invariance(mesh_device, layer_idx):
    """The same prefill run with two different internal chunk sizes must agree on the mesh.

    The single-chip stage's control for its long-prefill path
    (``test_full_context_chunk_size_invariance``) has no counterpart here, which review round 1
    recorded as the reason the mesh's full-context path had no correctness cross-check at all: beyond
    8000 tokens the HF golden is intractable on host, so the only way to check a long prefill is
    against *itself* under a different internal decomposition.

    This runs at 6000 tokens rather than the full 262144 — enough to cross both the 2048- and the
    1024-token chunk boundary several times and to leave a non-aligned tail under both — because the
    control is about the chunking, not about the length, and a 262144-token pair costs an hour of
    device time to say the same thing. The length is deliberately not a multiple of either chunk.
    """
    source = default_weight_source()
    seq_len = 6000
    x = make_activations(1, seq_len, seed=63)
    outs = []
    for chunk in (2048, 1024):
        decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=8192, prefill_chunk=chunk)
        out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
        outs.append(to_host(mesh_device, out).float())
        ttnn.deallocate(out)
        del decoder, page_table
    value = pcc(outs[0], outs[1])
    logger.info(
        f"multichip chunk-size invariance layer={layer_idx} seq_len={seq_len}: " f"chunk 2048 vs 1024 PCC={value:.6f}"
    )
    assert torch.isfinite(outs[1]).all()
    assert value > BASELINE_BAR, f"chunk size changed the prefill result (PCC {value})"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_repeated_run_stress(mesh_device, layer_idx):
    """Many back-to-back prefill/decode cycles stay correct and bounded on the mesh."""
    source = default_weight_source()
    lengths = [96, 130, 257, 96]
    cycles = 12
    steps = 4
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    decode_x = [to_device(mesh_device, make_activations(1, 1, seed=8000 + i)) for i in range(steps)]
    inputs = {n: to_device(mesh_device, make_activations(1, n, seed=9000 + n)) for n in set(lengths)}
    positions = {(n, s): decode_inputs(mesh_device, torch.tensor([n + s])) for n in set(lengths) for s in range(steps)}

    baseline = {}
    allocations = []
    for cycle in range(cycles):
        n = lengths[cycle % len(lengths)]
        decoder.reset_state()
        out = decoder.prefill_forward(inputs[n], page_table=page_table)
        prefill = to_host(mesh_device, out)
        ttnn.deallocate(out)
        decodes = []
        for step in range(steps):
            current_pos, rot_idxs = positions[(n, step)]
            out = decoder.decode_forward(
                decode_x[step], current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table
            )
            decodes.append(to_host(mesh_device, out))
            ttnn.deallocate(out)
        assert torch.isfinite(prefill.float()).all(), f"cycle {cycle}: non-finite prefill"
        for step, d in enumerate(decodes):
            assert torch.isfinite(d.float()).all(), f"cycle {cycle} step {step}: non-finite decode"
        if n in baseline:
            assert torch.equal(baseline[n][0], prefill), f"cycle {cycle}: prefill drifted at n={n}"
            for step, d in enumerate(decodes):
                assert torch.equal(baseline[n][1][step], d), f"cycle {cycle} step {step}: decode drifted at n={n}"
        else:
            baseline[n] = (prefill, decodes)
        allocations.append(ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)

    growth = allocations[-1] - allocations[len(lengths)]
    logger.info(
        f"multichip stress layer={layer_idx}: {cycles} prefill+{steps}-step-decode cycles over lengths "
        f"{sorted(set(lengths))}, repeats bit-identical, DRAM allocated {allocations[len(lengths)]} -> "
        f"{allocations[-1]} bytes (growth {growth})"
    )
    assert growth == 0, f"DRAM allocation grew by {growth} bytes across stress cycles: {allocations}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_no_host_fallback_in_forward(mesh_device, layer_idx, monkeypatch, expect_error):
    """A single prefill and a single decode pass must not touch the host — collectives included."""
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    x_tt = to_device(mesh_device, make_activations(1, 256, seed=13))
    d_tt = to_device(mesh_device, make_activations(1, 1, seed=14))
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([256]))

    banned = ["from_torch", "to_torch", "as_tensor", "from_device", "to_device"]

    def _raise(name):
        def stub(*args, **kwargs):
            raise AssertionError(f"host fallback: ttnn.{name} called inside a measured forward pass")

        return stub

    for name in banned:
        monkeypatch.setattr(ttnn, name, _raise(name))

    class NoTorchOps(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            raise AssertionError(f"host fallback: torch op {getattr(func, '__name__', func)} in a measured pass")

    probe = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
    with expect_error(AssertionError, "host fallback"):
        ttnn.from_torch(probe)
    with NoTorchOps():
        with expect_error(AssertionError, "host fallback"):
            torch.add(probe, probe)

    with NoTorchOps():
        ttnn.deallocate(decoder.prefill_forward(x_tt, page_table=page_table))
        ttnn.deallocate(decoder.decode_forward(d_tt, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table))
    logger.info(
        f"multichip fallback audit layer={layer_idx}: both guards verified to fire; prefill+decode clean for "
        f"ttnn {banned} and all torch ops"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_decode_runs_the_multichip_program_configs(mesh_device, layer_idx, monkeypatch):
    """The re-swept decode geometry reaches the dense projections it was swept for.

    Only the roles the local sweep moved are checked, and they are checked against
    :data:`MULTICHIP_DECODE_MATMUL_GEOMETRY` rather than against a literal, so the assertion follows
    the table. The ``in0_block_w`` bound also has to stay compatible with the residual norm's shard
    carry, which is the property ``mcast_in0`` validates and which a larger cap would break.
    """
    source = default_weight_source()
    role = "attn_in" if layer_idx == FULL_LAYER else "gdn_in"
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 128, seed=41)), page_table=page_table)
    )
    seen = []
    original = ttnn.linear

    def spy(x, w, *args, program_config=None, **kwargs):
        seen.append((int(x.shape[-1]), int(w.shape[-1]), program_config))
        return original(x, w, *args, program_config=program_config, **kwargs)

    monkeypatch.setattr(ttnn, "linear", spy)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([128]))
    ttnn.deallocate(
        decoder.decode_forward(
            to_device(mesh_device, make_activations(1, 1, seed=42)),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
    )
    width = int(decoder.w[role].shape[-1])
    matches = [cfg for k, n, cfg in seen if n == width and k == decoder.cfg.dim]
    assert (
        matches
    ), f"no decode call matched {role} (K={decoder.cfg.dim}, N={width}); saw {[(k, n) for k, n, _ in seen]}"
    cfg = matches[0]
    assert cfg is not None, f"{role} ran on ttnn's heuristic, not the swept config"
    target_cores, cap = MULTICHIP_DECODE_MATMUL_GEOMETRY[role]
    grid = cfg.compute_with_storage_grid_size
    realised = grid.x * grid.y
    logger.info(
        f"multichip decode geometry layer={layer_idx} {role}: grid {grid.x}x{grid.y} ({realised} cores) "
        f"in0_block_w={cfg.in0_block_w} per_core_N={cfg.per_core_N} target={target_cores} cap={cap}"
    )
    assert cfg.in0_block_w == cap, f"{role} in0_block_w {cfg.in0_block_w} != the swept cap {cap}"
    # Equality against the grid the swept target realises as, not `<=`: review round 4 pointed out
    # that `<=` passes a silent fallback to 8 cores on a role whose swept target is 110. The rule is
    # `_decode_1d_matmul_config`'s -- the widest legal rectangle at or below the target on this
    # device's worker grid -- mirrored here rather than imported so that a change to it has to be
    # made deliberately in two places.
    want_grid = decode_grid_for(mesh_device.compute_with_storage_grid_size(), target_cores)
    assert (grid.x, grid.y) == want_grid, (
        f"{role} ran on a {grid.x}x{grid.y} grid; the swept target {target_cores} realises as "
        f"{want_grid[0]}x{want_grid[1]} on this device"
    )
    assert cfg.per_core_N == math.ceil(
        (width // TILE) / realised
    ), f"{role} per_core_N {cfg.per_core_N} does not cover {width // TILE} output tiles on {realised} cores"
    # The residual norm hands this projection a width shard over NORM_SHARD_CORES cores, so the
    # per-core shard is dim/NORM_SHARD_CORES/32 tiles and mcast_in0 requires in0_block_w to divide it.
    shard_tiles = decoder.cfg.dim // decoder.NORM_SHARD_CORES // TILE
    assert shard_tiles % cfg.in0_block_w == 0, (
        f"in0_block_w {cfg.in0_block_w} does not divide the norm's {shard_tiles}-tile per-core shard, "
        "so the shard carry cannot reach this projection"
    )

    # `shared_down` is the third retuned role and the only one the local sweep moved *down*: TP=4
    # cuts its K from 512 to 128 (4 tiles), so the inherited 48-core target is launch overhead. It
    # is checked here rather than in its own test because it runs on both layer kinds and needs the
    # same decode trace. Its `in0` is the shared-expert SwiGLU product, which is interleaved rather
    # than width-sharded, so the norm's shard-carry bound above does not apply to it.
    sd_target, sd_cap = MULTICHIP_DECODE_MATMUL_GEOMETRY["shared_down"]
    sd_k = decoder.moe.cfg.shared_expert_intermediate_size
    sd = [cfg for k, n, cfg in seen if k == sd_k and n == decoder.cfg.dim]
    assert sd, f"no decode call matched shared_down (K={sd_k}, N={decoder.cfg.dim})"
    sd_cfg = sd[0]
    assert sd_cfg is not None, "shared_down ran on ttnn's heuristic, not the swept config"
    sd_grid = sd_cfg.compute_with_storage_grid_size
    logger.info(
        f"multichip decode geometry layer={layer_idx} shared_down: grid {sd_grid.x}x{sd_grid.y} "
        f"({sd_grid.x * sd_grid.y} cores) in0_block_w={sd_cfg.in0_block_w} per_core_N={sd_cfg.per_core_N} "
        f"target={sd_target} cap={sd_cap}"
    )
    assert (sd_grid.x, sd_grid.y) == decode_grid_for(mesh_device.compute_with_storage_grid_size(), sd_target)
    assert sd_cfg.in0_block_w == min(sd_cap, sd_k // TILE)


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 32])
def test_sparse_cores_match_the_local_sweep(mesh_device, layer_idx, batch, monkeypatch):
    """The routed sparse matmuls run on the core counts ``probe_sparse_matmul_local.txt`` selected.

    The dense projections have had this guard since the stage opened
    (``test_decode_runs_the_multichip_program_configs``); the routed matmuls did not, and review
    round 2 found the consequence. ``SPARSE_SCALE_CORES_BY_TP`` rescales the bound handed to the
    inherited core rule, and that bound is ``OptimizedMoE._active_expert_bound`` =
    ``min(num_experts_local, rows * top_k)`` — so it is 8 at batch 1, 16 at batch 2, 32 at batch 4
    and saturates at 64 from batch 8. The rescale therefore changes decode geometry at every batch
    above 1, which four documents had asserted it did not, on the strength of a batch-1-only A/B.

    This pins the realised grid at every advertised decode batch against
    :data:`SPARSE_DECODE_CORES`, and the prefill grid against :data:`SPARSE_PREFILL_CORES`, both
    transcribed from the shipped policy and cross-checked against the sweep's winner. A silently
    different geometry is indistinguishable from the intended one in every other measurement.
    """
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(
        mesh_device, layer_idx, source, batch=batch, max_context=1024, num_blocks=64 * batch
    )
    seen: dict[str, dict[tuple[int, int], int]] = {"prefill": {}, "decode": {}}
    phase = ["prefill"]
    original = ttnn.sparse_matmul

    def spy(a, b, *args, program_config=None, **kwargs):
        grid = program_config.compute_with_storage_grid_size
        seen[phase[0]][(int(b.shape[-2]), int(b.shape[-1]))] = grid.x * grid.y
        return original(a, b, *args, program_config=program_config, **kwargs)

    monkeypatch.setattr(ttnn, "sparse_matmul", spy)
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(batch, 128, seed=77)), page_table=page_table)
    )
    phase[0] = "decode"
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.full((batch,), 128, dtype=torch.int32))
    ttnn.deallocate(
        decoder.decode_forward(
            to_device(mesh_device, make_activations(batch, 1, seed=78)),
            current_pos=current_pos,
            rot_idxs=rot_idxs,
            page_table=page_table,
        )
    )
    cfg = decoder.moe.cfg
    roles = {
        "gate_up": (cfg.dim, 2 * cfg.moe_intermediate_size),
        "down": (cfg.moe_intermediate_size, cfg.dim),
    }
    for phase_name, expected in (("prefill", SPARSE_PREFILL_CORES), ("decode", SPARSE_DECODE_CORES[batch])):
        calls = seen[phase_name]
        got = {role: calls[shape] for role, shape in roles.items() if shape in calls}
        logger.info(f"multichip sparse cores layer={layer_idx} batch={batch} {phase_name}: {got}")
        assert set(got) == set(roles), f"missing a routed {phase_name} matmul at batch {batch}: saw {sorted(calls)}"
        for role, cores in got.items():
            want = expected[role]
            assert cores == want, f"{role} at {phase_name} batch {batch} ran on {cores} cores, expected {want}"


# --------------------------------------------------------------------------------------
# advertised context
# --------------------------------------------------------------------------------------
@pytest.mark.long
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [ADVERTISED_CONTEXT - 3, ADVERTISED_CONTEXT])
def test_full_context_prefill_and_decode(mesh_device, layer_idx, seq_len):
    """Prefill at the full advertised context on the target mesh, then decode at the last slot."""
    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=ADVERTISED_CONTEXT)
    x = make_activations(1, seq_len, seed=97)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    assert list(out.shape) == [1, seq_len, hf_config().hidden_size]
    tail = to_host(mesh_device, ttnn.slice(out, [0, seq_len - 64, 0], [1, seq_len, hf_config().hidden_size])).float()
    ttnn.deallocate(out)
    assert torch.isfinite(tail).all(), "long prefill produced non-finite outputs"
    assert tail.std() > 0.01, f"long prefill output is near-constant (std={tail.std():.4g})"
    decoded = seq_len < ADVERTISED_CONTEXT
    if decoded:
        current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
        got = to_host(
            mesh_device,
            decoder.decode_forward(
                to_device(mesh_device, make_activations(1, 1, seed=98)),
                current_pos=current_pos,
                rot_idxs=rot_idxs,
                page_table=page_table,
            ),
        ).float()
        assert torch.isfinite(got).all()
        assert got.std() > 0.01
    logger.info(
        f"multichip full-context layer={layer_idx} seq_len={seq_len} "
        f"{'prefill+decode' if decoded else 'prefill'} completed on the 4-chip mesh, tail std={tail.std():.4f}"
    )


@pytest.mark.long
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_long_context_pcc(mesh_device, layer_idx):
    """Largest context where the HF golden is tractable, at a non-aligned length."""
    source = default_weight_source()
    seq_len = 8000
    x = make_activations(1, seq_len, seed=61)
    ref_prefill, ref_decode = run_reference(
        layer_idx, source, x, decode_x=[make_activations(1, 1, seed=62)], decode_steps=1
    )
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=16384)
    out = decoder.prefill_forward(to_device(mesh_device, x), page_table=page_table)
    value = pcc(ref_prefill, to_host(mesh_device, out))
    ttnn.deallocate(out)
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([seq_len]))
    dec = decoder.decode_forward(
        to_device(mesh_device, make_activations(1, 1, seed=62)),
        current_pos=current_pos,
        rot_idxs=rot_idxs,
        page_table=page_table,
    )
    dvalue = pcc(ref_decode[0], to_host(mesh_device, dec))
    ttnn.deallocate(dec)
    logger.info(f"multichip long context layer={layer_idx} seq_len={seq_len}: prefill {value:.6f} decode {dvalue:.6f}")
    assert value > PCC_BAR and dvalue > PCC_BAR


# --------------------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("seq_len", [2048])
def test_perf_prefill(mesh_device, layer_idx, seq_len):
    """Warmed prefill timing between ``PERF_PREFILL`` signposts, on the target mesh."""
    from tracy import signpost

    source = default_weight_source()
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    x_tt = to_device(mesh_device, make_activations(1, seq_len, seed=71))
    for _ in range(2):
        ttnn.deallocate(decoder.prefill_forward(x_tt, page_table=page_table))
    ttnn.synchronize_device(mesh_device)

    signpost("PERF_PREFILL")
    start = time.time()
    out = decoder.prefill_forward(x_tt, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.time() - start
    signpost("PERF_PREFILL_END")
    ttnn.deallocate(out)
    logger.info(
        f"MULTICHIP PERF prefill layer={layer_idx} ({LAYER_IDS[layer_idx]}) seq_len={seq_len} "
        f"wall={elapsed * 1e3:.2f} ms  tok/s={seq_len / elapsed:.1f}"
    )
    # This test exists to be profiled between signposts, but review round 4 pointed out that it
    # asserted nothing at all, so nothing in the suite gated the prefill speedup the README quotes.
    # The bar is the single-chip baseline's warmed prefill (~95-102 ms) divided by a conservative 2x,
    # against a measured 3.3-3.5x: it fails a lost parallelisation and tolerates a slow machine.
    assert elapsed * 1e3 < PREFILL_MS_BAR, (
        f"warmed multichip prefill {elapsed * 1e3:.2f} ms is above the {PREFILL_MS_BAR} ms bar; the "
        "single-chip baseline is ~95-102 ms, so this is at most a 2x speedup"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_perf_decode_traced(mesh_device, layer_idx):
    """Warmed traced decode timing between ``PERF_DECODE`` signposts, on the target mesh."""
    from tracy import signpost

    source = default_weight_source()
    iters = 32
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source)
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 128, seed=81)), page_table=page_table)
    )
    x_buf = to_device(mesh_device, make_activations(1, 1, seed=82))
    pos_buf, rot_buf = decode_inputs(mesh_device, torch.tensor([128]))

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

    ttnn.deallocate(forward())
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = forward()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    for _ in range(4):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    signpost("PERF_DECODE")
    start = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.time() - start
    signpost("PERF_DECODE_END")
    logger.info(
        f"MULTICHIP PERF decode(traced) layer={layer_idx} ({LAYER_IDS[layer_idx]}) iters={iters} "
        f"wall/iter={elapsed / iters * 1e3:.3f} ms  steps/s={iters / elapsed:.1f}"
    )
    assert torch.isfinite(to_host(mesh_device, trace_out).float()).all()
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_multichip_beats_single_chip_traced_decode(mesh_device, layer_idx):
    """The gate on the whole stage: same process, same weights, multichip must be faster.

    The single-chip arm here runs on the **same 4-chip mesh** with every weight replicated, which is
    four copies of the single-chip graph running in lockstep — i.e. the same per-device work and the
    same wall clock as one chip, plus whatever the mesh costs in dispatch. That makes this a
    conservative comparison for the multichip arm rather than a flattering one, and it is why the
    README's speedup table quotes a separate 1x1 run as well.
    """
    source = default_weight_source()
    iters = 32
    timings = {}
    for name, cls in (("multichip", MultichipDecoder), ("single-chip-replicated", OptimizedDecoder)):
        decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, cls=cls)
        ttnn.deallocate(
            decoder.prefill_forward(to_device(mesh_device, make_activations(1, 128, seed=81)), page_table=page_table)
        )
        x_buf = to_device(mesh_device, make_activations(1, 1, seed=82))
        pos_buf, rot_buf = decode_inputs(mesh_device, torch.tensor([128]))

        def forward(decoder=decoder, page_table=page_table, x_buf=x_buf, pos_buf=pos_buf, rot_buf=rot_buf):
            return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

        ttnn.deallocate(forward())
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_out = forward()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        start = time.time()
        for _ in range(iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        timings[name] = (time.time() - start) / iters
        assert torch.isfinite(to_host(mesh_device, trace_out).float()).all()
        ttnn.release_trace(mesh_device, trace_id)
        del decoder

    speedup = timings["single-chip-replicated"] / timings["multichip"]
    logger.info(
        f"MULTICHIP same-process traced decode layer={layer_idx} ({LAYER_IDS[layer_idx]}): "
        f"single-chip-replicated {timings['single-chip-replicated'] * 1e3:.3f} ms -> multichip "
        f"{timings['multichip'] * 1e3:.3f} ms ({speedup:.2f}x)"
    )
    # A real bar, not `> 1.0`: the measured speedup is 1.65-1.69x on both layer kinds across every
    # sweep this stage ran, so 1.4x leaves ample room for machine noise while still failing if the
    # parallelisation regresses. Review round 4 pointed out that `> 1.0` gated nothing the README's
    # figures depend on.
    assert (
        speedup > DECODE_SPEEDUP_BAR
    ), f"multichip traced decode speedup {speedup:.2f}x is below the {DECODE_SPEEDUP_BAR}x bar: {timings}"


# ---------------------------------------------------------------------------------------------
# gathered routed experts on the mesh (optimized_decoder.MOE_GATHER_EXPERTS)
# ---------------------------------------------------------------------------------------------
#
# The gathered path is entirely device-local: it replaces this device's two routed sparse matmuls with
# one fused FFN program over its 64 local experts' gathered rows. Nothing about the expert-parallel
# decomposition moves — the result is still this device's partial sum over its own experts, and
# `MultichipDecoder._block` still closes it with the same single all-reduce. The tests below pin exactly
# that: the same answer, no extra collective, and the zero-local-expert case that the sparse path needs
# a floored mask for and this path does not.


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("tokens", [1024, 2048], ids=lambda t: f"tok{t}")
def test_gathered_experts_match_sparse_multichip(mesh_device, layer_idx, tokens, monkeypatch):
    """On the 4-chip mesh the gathered routed experts must reproduce the sparse ones.

    Same layer, same weights, same activation; the reference arm is the shipped sparse path and runs
    first. The bar is :data:`BASELINE_BAR` rather than an exact one because the fused kernel keeps
    gate/up/down inside a single program while the sparse chain round-trips through DRAM at
    ``expert_act_dtype``, and because it evaluates each expert's real row count rather than a whole
    32-row group per activated expert.
    """
    source = default_weight_source()
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)  # before the build: uploads both layouts
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=TEST_CONTEXT)
    assert decoder.moe.gather_experts, "the per-expert weights were not built; the flag was read too late"
    x = to_device(mesh_device, make_activations(1, tokens, seed=1300 + tokens))

    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", False)
    assert decoder.moe._gather_reason(tokens, False, None) is not None, "the reference arm must be the sparse path"
    want = to_host(mesh_device, decoder.prefill_forward(x, page_table=page_table))
    decoder.reset_state()

    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    engaged = decoder.moe._gather_reason(tokens, False, None)
    assert engaged is None, f"the gathered arm was refused at {tokens} tokens, so both arms are sparse: {engaged}"
    got = to_host(mesh_device, decoder.prefill_forward(x, page_table=page_table))

    value = pcc(want, got)
    logger.info(
        f"multichip gathered vs sparse routed experts layer={layer_idx} ({LAYER_IDS[layer_idx]}) "
        f"tokens={tokens} PCC={value:.6f}"
    )
    assert torch.isfinite(got.float()).all(), "the gathered path produced a non-finite layer output"
    assert value > BASELINE_BAR, f"multichip gathered vs sparse PCC {value} <= {BASELINE_BAR}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_gathered_experts_zero_local_active_experts(mesh_device, layer_idx, monkeypatch):
    """A device whose expert block wins none of the global top-8 must contribute exactly zero.

    The gathered counterpart of ``test_zero_local_active_experts``, and it needs no floored expert to
    get there. Under the sparse path :data:`MC.MOE_MASK_FLOOR` keeps local expert 0 in the sparsity so
    ``ttnn.sparse_matmul`` is never handed an all-zero mask, and the argument that it contributes
    nothing rests on its routing score being zero. The gathered path has no sparsity mask at all: an
    all-zero routing vector gives every local expert a count of zero — the fused FFN launches and
    evaluates nothing — and marks every reverse slot invalid, so every score is multiplied by exactly
    zero. This checks both halves, that the counts are zero and that the output is.

    With 8 experts drawn from 256 over four blocks this happens for about one device in ten per token,
    so it is routine; the routing is forced by hand rather than waited for, exactly as the sparse test
    does, by synthesising the per-device dense routing vector instead of running the router.
    """
    source = default_weight_source()
    rows = OD.MOE_GATHER_SUB_CHUNK  # the one admitted, measured compact-dispatch shape
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    decoder, _, _ = build_decoder(mesh_device, layer_idx, source, max_context=1024)
    moe = decoder.moe
    assert moe.gather_experts, "the per-expert weights were not built; the flag was read too late"
    moe._gather_consts(rows)  # setup, exactly as prepare_gather_experts does
    e_local = decoder.cfg.num_experts

    # Routing that puts every selected expert on device 0's block, so devices 1..3 get none.
    host = torch.zeros(1, 1, rows, e_local)
    per_device = [host.clone() for _ in range(mesh_device.get_num_devices())]
    per_device[0][..., :8] = 1.0 / 8
    dense = ttnn.from_torch(
        torch.cat(per_device, dim=0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=0),
    )

    # Record the per-expert counts the fused kernel is handed, per device.
    counts_seen: list = []
    real_ffn = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe

    def spy(dispatched_buffer, expert_region_offsets, expert_token_counts, *args, **kwargs):
        counts_seen.append([t.flatten() for t in shards(mesh_device, expert_token_counts)])
        return real_ffn(dispatched_buffer, expert_region_offsets, expert_token_counts, *args, **kwargs)

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "unified_routed_expert_moe", spy)

    x = to_device(mesh_device, make_activations(1, rows, seed=83).reshape(1, 1, rows, hf_config().hidden_size))
    moe._decode_phase = False
    moe._call_tokens = rows
    routed = moe._gather_routed_experts(x, dense, rows)
    parts = shards(mesh_device, routed)
    ttnn.deallocate(routed)
    ttnn.deallocate(dense)

    assert len(counts_seen) == 1, f"expected one fused-FFN call, saw {len(counts_seen)}"
    counts = counts_seen[0]
    assert int(counts[0].sum()) == 8 * rows, f"device 0 should hold every assignment, saw {int(counts[0].sum())}"
    for d in range(1, len(counts)):
        assert int(counts[d].sum()) == 0, f"device {d} was handed {int(counts[d].sum())} token assignments"
    for d in range(1, len(parts)):
        assert torch.count_nonzero(parts[d]) == 0, f"device {d} contributed {torch.count_nonzero(parts[d])} non-zeros"
    assert torch.count_nonzero(parts[0]) > 0, "device 0 should have produced the whole routed output"
    logger.info(
        f"gathered zero-local-expert case layer={layer_idx}: devices 1-3 saw zero counts and contributed "
        f"exactly zero with no mask floor; device 0 carried all {8 * rows} assignments"
    )


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_gathered_experts_add_no_collective(mesh_device, layer_idx, monkeypatch):
    """The gathered path adds no cross-device traffic and never touches the sparsity mask.

    Two invariants in one forward, because they are the same invariant seen from two sides:

    * the layer still performs exactly the two all-reduces it always did — one on the token mixer's
      row-parallel output, one on the MoE's partial — so a token's contribution from a non-local
      expert still arrives only through that collective and nowhere else;
    * ``_active_expert_mask`` is never called, so :data:`MC.MOE_MASK_FLOOR` and its floored local
      expert 0 play no part. There is no sparsity to keep non-empty, which is why the zero-local case
      above needs no floor.
    """
    source = default_weight_source()
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=TEST_CONTEXT)
    reduces: list = []
    masks: list = []
    real_reduce = MC.MultichipDecoder._all_reduce
    real_mask = MC.MultichipMoE._active_expert_mask

    def spy_reduce(self, tensor, _real=real_reduce):
        reduces.append(_physical_rows(tensor.shape))
        return _real(self, tensor)

    def spy_mask(self, dense_routing, groups, valid_tokens, _real=real_mask):
        masks.append(groups)
        return _real(self, dense_routing, groups, valid_tokens)

    monkeypatch.setattr(MC.MultichipDecoder, "_all_reduce", spy_reduce)
    monkeypatch.setattr(MC.MultichipMoE, "_active_expert_mask", spy_mask)
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 2048, seed=84)), page_table=page_table)
    )
    logger.info(f"gathered prefill collectives={len(reduces)} rows={reduces} sparsity_mask_calls={len(masks)}")
    assert len(reduces) == 2, f"the gathered prefill performed {len(reduces)} collectives, expected 2"
    assert len(masks) == 0, f"the gathered path built the sparsity mask {len(masks)} times"


# ---------------------------------------------------------------------------------------------
# gathered routed experts: the tests that need real expert parallelism
# ---------------------------------------------------------------------------------------------
#
# These live here rather than in tests/test_optimized_decoder.py because the gathered path is validated
# only under expert parallelism. Compact dispatch removes the old `num_experts_local * sub_chunk`
# buffer, but its surrounding permutation and duplicate weight layout are only validated at 64 local experts.
# `optimized_decoder.MOE_GATHER_MAX_LOCAL_EXPERTS` therefore keeps the admitted shape at EP=4's 64
# experts/device — the deployment geometry and the only one with device correctness evidence.


def _assert_gather_engaged(moe, tokens):
    """The gathered path must be *taken* at this shape, with the reason surfaced when it is not."""
    reason = moe._gather_reason(tokens, False, None)
    assert reason is None, f"the gathered path was refused at {tokens} tokens: {reason}"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
@pytest.mark.parametrize("rows", [1024], ids=lambda r: f"rows{r}")
def test_gathered_experts_dispatch_combine_is_a_permutation(mesh_device, layer_idx, rows, monkeypatch):
    """``combine(dispatch(x))`` is a permutation: with an identity FFN the round trip returns ``x``.

    The gathered path is a scatter of token rows into per-expert regions, a per-expert FFN, and a gather
    of those rows back weighted by the router score. Replace the FFN with the identity and the whole
    thing collapses to ``out[t] = (Σ_e score[t, e]) · x[t]`` over this device's experts — checkable in
    closed form on the host from the routing vector alone.

    That one comparison pins three contract items at once, which is why it is worth doing in isolation
    rather than only through the layer:

    * **The permutation.** Any mis-derived rank, region offset or inverse index sends some token's row
      to the wrong slot, and the value that comes back stops being a multiple of that token's own row.
    * **Score placement.** The scale each row returns with is exactly its router score, applied once.
    * **Padding exclusion.** Each tight-packed expert region is rounded to one tile. Its padding is
      exact zero from the scatter base and is never read back; if the reverse slot were one row off,
      the per-token ratio would stop being constant.

    Checked per device, not just on device 0: the permutation is built from each device's own local
    routing, so a bug that only mis-indexed a non-zero expert block would be invisible in the
    all-reduced layer output.
    """
    source = default_weight_source()
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    decoder, _, _ = build_decoder(mesh_device, layer_idx, source, max_context=TEST_CONTEXT)
    moe = decoder.moe
    assert moe.gather_experts, "the per-expert weights were not built; the flag was read too late"
    moe._gather_consts(rows)  # setup, exactly as prepare_gather_experts does

    # The identity FFN. `unified_routed_expert_moe` returns a TILE tensor and its input here is the
    # ROW_MAJOR gathered buffer, so the identity is the layout conversion and nothing else.
    def identity_ffn(dispatched_buffer, *args, **kwargs):
        return ttnn.to_layout(dispatched_buffer, ttnn.TILE_LAYOUT)

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "unified_routed_expert_moe", identity_ffn)

    x_host = make_activations(1, rows, seed=85).reshape(1, 1, rows, hf_config().hidden_size)
    x = to_device(mesh_device, x_host)
    moe._decode_phase = False
    moe._call_tokens = rows
    dense = moe.routing_weights(x)
    dense_parts = shards(mesh_device, dense)
    routed = moe._gather_routed_experts(x, dense, rows)
    got_parts = shards(mesh_device, routed)
    ttnn.deallocate(routed)
    ttnn.deallocate(dense)

    worst = 1.0
    for d, (dense_d, got_d) in enumerate(zip(dense_parts, got_parts)):
        # out[t] = (sum over device d's experts of score[t, e]) * x[t]
        scale = dense_d.float().reshape(rows, -1).sum(dim=-1).reshape(rows, 1)
        want = x_host.float().reshape(rows, -1) * scale
        value = pcc(want, got_d.float().reshape(rows, -1))
        worst = min(worst, value)
        logger.info(
            f"gathered permutation device {d} rows={rows}: local score mass in "
            f"[{float(scale.min()):.4f}, {float(scale.max()):.4f}] PCC={value:.8f}"
        )
    # Liveness: with 8 experts drawn from 256 over four blocks, some device must hold a partial share —
    # if every device saw the same uniform mass the comparison would be nearly vacuous.
    masses = [float(p.float().reshape(rows, -1).sum(dim=-1).max()) for p in dense_parts]
    assert max(masses) - min(masses) > 1e-3, f"local score mass is uniform across devices ({masses}); test is inert"
    assert worst > 0.9999, f"combine(dispatch(x)) is not a permutation of x (worst per-device PCC {worst})"


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_gathered_experts_prefill_uses_the_fused_kernel(mesh_device, layer_idx, monkeypatch):
    """A prefill above the floor dispatches the fused kernel once per sub-chunk and no sparse matmul.

    The inverse of the decode test below: without it, a silent fallback to the sparse path would make
    every equivalence assertion in this file pass trivially.
    """
    source = default_weight_source()
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=TEST_CONTEXT)
    _assert_gather_engaged(decoder.moe, 2048)
    reduce_input_dtypes = []
    real_all_reduce = MC.MultichipDecoder._all_reduce

    def record_all_reduce_dtype(self, tensor):
        reduce_input_dtypes.append(tensor.dtype)
        return real_all_reduce(self, tensor)

    monkeypatch.setattr(MC.MultichipDecoder, "_all_reduce", record_all_reduce_dtype)
    recorder = _OpRecorder(
        monkeypatch,
        [
            "sparse_matmul",
            "experimental.deepseek_prefill.unified_routed_expert_moe",
            "scatter",
            "sort",
            "embedding",
            "where",
        ],
    )
    recorder.reset()
    out = decoder.prefill_forward(
        to_device(mesh_device, make_activations(1, 2048, seed=86)),
        page_table=page_table,
    )
    ttnn.deallocate(out)
    sub_chunks = 2048 // min(OD.MOE_GATHER_SUB_CHUNK, 2048)
    fused = recorder.count("experimental.deepseek_prefill.unified_routed_expert_moe")
    logger.info(
        f"gathered prefill layer={layer_idx}: sparse_matmul={recorder.count('sparse_matmul')} fused_ffn={fused} "
        f"scatter={recorder.count('scatter')} sort={recorder.count('sort')} "
        f"embedding={recorder.count('embedding')} where={recorder.count('where')} sub_chunks={sub_chunks}"
    )
    assert recorder.count("sparse_matmul") == 0, "the gathered prefill still dispatched a sparse matmul"
    assert fused == sub_chunks, f"expected one fused-FFN call per sub-chunk ({sub_chunks}), saw {fused}"
    # Compact dispatch contributes one scatter per sub-chunk. The complete decoder layer may issue
    # another scatter outside the MoE (currently one at both layer kinds), so this is a lower bound;
    # the exact fused/sort/embedding/where counts below pin the gathered branch itself.
    assert recorder.count("scatter") >= sub_chunks, "compact dispatch did not scatter once per sub-chunk"
    assert recorder.count("sort") == sub_chunks, "only the reverse expert sort should remain per sub-chunk"
    assert recorder.count("embedding") == 2 * sub_chunks
    assert recorder.count("where") == sub_chunks, "invalid reverse rows were not selected away before scoring"
    assert len(reduce_input_dtypes) == 2, "one attention and one MoE all-reduce must run"
    assert decoder.moe.policy.expert_act_dtype == ttnn.bfloat8_b, "this regression pins C25's routed-output policy"
    assert (
        reduce_input_dtypes[-1] == decoder.moe.policy.expert_act_dtype == ttnn.bfloat8_b
    ), "gathered MoE must restore the BF8 routed-output contract before the expert-parallel all-reduce"


@pytest.mark.parametrize("layer_idx", [LINEAR_LAYER], ids=lambda i: LAYER_IDS[i])
def test_gathered_experts_batch8_times_128_is_one_safe_subchunk(mesh_device, layer_idx, monkeypatch):
    """Batch 8 by 128 tokens flattens to the one admitted 1024-row gathered shape.

    This is the short-prompt geometry in the requested batch-8 latency sweep. The MoE is tokenwise, so
    folding the batch and sequence axes must give the same layer result as sparse routing, while the op
    census proves the comparison did not pass because both arms silently used the sparse fallback.
    """
    batch, seq_len = 8, 128
    flattened = batch * seq_len
    assert flattened == OD.MOE_GATHER_SUB_CHUNK == OD.MOE_GATHER_MIN_TOKENS

    source = default_weight_source()
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    decoder, page_table, _ = build_decoder(
        mesh_device,
        layer_idx,
        source,
        batch=batch,
        max_context=1024,
    )
    _assert_gather_engaged(decoder.moe, flattened)
    x = to_device(mesh_device, make_activations(batch, seq_len, seed=91))

    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", False)
    sparse_out = decoder.prefill_forward(x, page_table=page_table)
    want = to_host(mesh_device, sparse_out)
    ttnn.deallocate(sparse_out)
    decoder.reset_state()

    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    recorder = _OpRecorder(
        monkeypatch,
        ["sparse_matmul", "experimental.deepseek_prefill.unified_routed_expert_moe"],
    )
    recorder.reset()
    gathered_out = decoder.prefill_forward(x, page_table=page_table)
    got = to_host(mesh_device, gathered_out)
    ttnn.deallocate(gathered_out)
    ttnn.deallocate(x)

    value = pcc(want, got)
    logger.info(f"gathered batch={batch} seq={seq_len} flattened={flattened} vs sparse PCC={value:.9f}")
    assert torch.isfinite(got.float()).all()
    assert value > BASELINE_BAR, f"batch-8 flattened gathered vs sparse PCC {value} <= {BASELINE_BAR}"
    assert recorder.count("experimental.deepseek_prefill.unified_routed_expert_moe") == 1
    assert recorder.count("sparse_matmul") == 0


@pytest.mark.parametrize("layer_idx", [LINEAR_LAYER], ids=lambda i: LAYER_IDS[i])
def test_gathered_experts_nonaligned_logical_padding_stays_sparse_and_matches_reference(
    mesh_device, layer_idx, monkeypatch
):
    """A non-aligned logical prompt remains on the safe sparse path and matches its reference.

    The final 127 rows are prefill padding, not real tokens. They can change device-side expert counts
    and adaptive chunk geometry, but must not change any of the 897 returned logical rows. Gathered MoE
    deliberately refuses ``valid_tokens`` today, so this A/B pins that refusal instead of accidentally
    claiming the fused kernel handled logical padding.
    """
    logical_tokens = OD.MOE_GATHER_MIN_TOKENS - PREFILL_ALIGN + 1
    physical_tokens = ((logical_tokens + PREFILL_ALIGN - 1) // PREFILL_ALIGN) * PREFILL_ALIGN
    assert (logical_tokens, physical_tokens) == (897, OD.MOE_GATHER_SUB_CHUNK)

    source = default_weight_source()
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=2048)
    reason = decoder.moe._gather_reason(physical_tokens, False, logical_tokens)
    assert reason is not None and "tile-padded call" in reason
    x = to_device(mesh_device, make_activations(1, logical_tokens, seed=92))

    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", False)
    sparse_out = decoder.prefill_forward(x, page_table=page_table)
    want = to_host(mesh_device, sparse_out)
    ttnn.deallocate(sparse_out)
    decoder.reset_state()

    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    recorder = _OpRecorder(
        monkeypatch,
        ["sparse_matmul", "experimental.deepseek_prefill.unified_routed_expert_moe"],
    )
    recorder.reset()
    gathered_out = decoder.prefill_forward(x, page_table=page_table)
    got = to_host(mesh_device, gathered_out)
    ttnn.deallocate(gathered_out)
    ttnn.deallocate(x)

    value = pcc(want, got)
    logger.info(
        f"gathered logical={logical_tokens} physical={physical_tokens} padding="
        f"{physical_tokens - logical_tokens} vs sparse PCC={value:.9f}"
    )
    assert tuple(got.shape[:2]) == (1, logical_tokens)
    assert torch.isfinite(got.float()).all()
    assert value > BASELINE_BAR, f"non-aligned gathered vs sparse PCC {value} <= {BASELINE_BAR}"
    assert recorder.count("experimental.deepseek_prefill.unified_routed_expert_moe") == 0
    assert recorder.count("sparse_matmul") == 2 * (physical_tokens // OD.DEFAULT_MOE_GROUP_TOKENS)


@pytest.mark.parametrize("layer_idx", [LINEAR_LAYER], ids=lambda i: LAYER_IDS[i])
def test_gathered_experts_small_tail_uses_sparse_fallback(mesh_device, layer_idx, monkeypatch):
    """A profitable gathered prefix may coexist with a sub-minimum sparse tail.

    At 1152 rows the 1024-row prefix should use one fused gathered program. The remaining 128 rows
    are below ``MOE_GATHER_MIN_TOKENS`` and therefore use the two sparse projections without building
    gathered constants for that losing shape.
    """
    source = default_weight_source()
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=TEST_CONTEXT)
    _assert_gather_engaged(decoder.moe, 1152)
    assert 128 not in decoder.moe._gather_const
    recorder = _OpRecorder(
        monkeypatch,
        ["sparse_matmul", "experimental.deepseek_prefill.unified_routed_expert_moe"],
    )

    x = to_device(mesh_device, make_activations(1, 1152, seed=90))
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", False)
    sparse_out = decoder.prefill_forward(x, page_table=page_table)
    want = to_host(mesh_device, sparse_out)
    ttnn.deallocate(sparse_out)
    decoder.reset_state()

    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    recorder.reset()
    out = decoder.prefill_forward(
        x,
        page_table=page_table,
    )
    got = to_host(mesh_device, out)
    ttnn.deallocate(out)
    ttnn.deallocate(x)
    assert torch.isfinite(got.float()).all()
    value = pcc(want, got)
    logger.info(f"mixed gathered/sparse tail: rows=1152 gathered=1024 sparse=128 PCC={value:.9f}")
    assert value > BASELINE_BAR, f"mixed gathered/sparse tail PCC {value} <= {BASELINE_BAR}"
    assert recorder.count("experimental.deepseek_prefill.unified_routed_expert_moe") == 1
    assert recorder.count("sparse_matmul") == 2


@pytest.mark.parametrize("layer_idx", LAYERS, ids=lambda i: LAYER_IDS[i])
def test_gathered_experts_decode_keeps_sparse_matmul(mesh_device, layer_idx, monkeypatch):
    """Decode keeps ``sparse_matmul`` even with the switch on, and the op counts prove it.

    A decode call is one 32-row tile, so a gathered expert region is one tile whatever its real count:
    the gathered path would evaluate ``num_experts_local * 32`` rows for the ``32 * top_k / tp`` the
    routing asks and add the dispatch/reverse-permutation graph. The FFN itself is fused, but the
    arithmetic and glue are still worse, so the switch is scoped to prefill; this pins the scoping
    rather than trusting it, and checks that the reason says so.
    """
    source = default_weight_source()
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=TEST_CONTEXT)
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 1024, seed=87)), page_table=page_table)
    )
    reason = decoder.moe._gather_reason(TILE, True, None)
    assert reason is not None and "decode keeps sparse_matmul" in reason, f"decode was admitted (reason={reason!r})"
    recorder = _OpRecorder(monkeypatch, ["sparse_matmul", "experimental.deepseek_prefill.unified_routed_expert_moe"])
    recorder.reset()
    x = to_device(mesh_device, make_activations(1, 1, seed=88))
    current_pos, rot_idxs = decode_inputs(mesh_device, torch.tensor([1024]))
    ttnn.deallocate(decoder.decode_forward(x, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table))
    fused = recorder.count("experimental.deepseek_prefill.unified_routed_expert_moe")
    sparse = recorder.count("sparse_matmul")
    logger.info(f"decode with the switch set: sparse_matmul={sparse} fused_ffn={fused}")
    assert fused == 0, f"decode dispatched the gathered path {fused} times"
    assert sparse == 2, f"decode dispatched {sparse} sparse matmuls, expected the shipped 2"


@pytest.mark.parametrize("layer_idx", [LAYERS[0]], ids=lambda i: LAYER_IDS[i])
def test_gathered_experts_refuse_untested_sub_chunk(mesh_device, layer_idx, monkeypatch):
    """A sub-chunk above the validated cap refuses at setup rather than being launched untested.

    Every L1 circular-buffer footprint in the permutation glue grows with the sub-chunk, and 1024 is
    the only value that has been launched. This is the guard that bounds the reachable shape space on
    this mesh to the one that was measured, so it is enforced rather than documented — and it refuses
    loudly at setup, leaving the correct-and-slower sparse path running.
    """
    source = default_weight_source()
    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    monkeypatch.setattr(OD, "MOE_GATHER_SUB_CHUNK", OD.MOE_GATHER_MAX_SUB_CHUNK * 2)
    decoder, page_table, _ = build_decoder(mesh_device, layer_idx, source, max_context=TEST_CONTEXT)
    moe = decoder.moe
    assert moe.gather_experts, "the per-expert layout should still be uploaded; only the sub-chunk is out of range"
    assert (
        moe.gather_refusal is not None and "MOE_GATHER_MAX_SUB_CHUNK" in moe.gather_refusal
    ), f"an oversized sub-chunk was accepted (refusal={moe.gather_refusal!r})"
    recorder = _OpRecorder(monkeypatch, ["sparse_matmul", "experimental.deepseek_prefill.unified_routed_expert_moe"])
    recorder.reset()
    ttnn.deallocate(
        decoder.prefill_forward(to_device(mesh_device, make_activations(1, 2048, seed=89)), page_table=page_table)
    )
    assert recorder.count("experimental.deepseek_prefill.unified_routed_expert_moe") == 0
    assert recorder.count("sparse_matmul") > 0, "the fallback did not run the sparse path"
    logger.info(f"oversized sub-chunk refused at setup: {moe.gather_refusal}")
