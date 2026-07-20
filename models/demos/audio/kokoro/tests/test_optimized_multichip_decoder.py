# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-chip decoder tests for hexgrad/Kokoro-82M (plbert / ALBERT encoder).

Target: 4x Blackhole p300c exposed as a (1, 4) mesh, TP=4 (head-parallel
attention + sequence-parallel FFN, sequence-sharded residual). See
``tt/optimized_multichip_decoder.py`` and ``doc/optimized_multichip_decoder/README.md``.

Correctness is validated primarily against the **single-chip TTNN optimized
baseline** (``OptimizedDecoder``) with identical real weights/inputs/masks - this
isolates sharding/collective bugs from HF-vs-TTNN numerics (multichip skill
"Validation Heuristics"). We also carry the HF >= 0.995 bar end-to-end and a
per-layer-kind (embedding / AlbertLayer) component comparison.

Kokoro is non-autoregressive (one weight-tied AlbertLayer kind, no KV cache, no
causal mask); "decode" is the full-sequence encode captured into a TTNN trace and
replayed. Paged-cache / current-position are N/A, same as stages 01/02.
"""

import json
import random

import pytest
import torch

import ttnn
from models.demos.audio.kokoro.tt.optimized_decoder import OptimizedDecoder
from models.demos.audio.kokoro.tt.optimized_multichip_decoder import OptimizedMultichipDecoder as MultichipDecoder

MODEL_ID = "hexgrad/Kokoro-82M"
PCC_BAR = 0.995  # HF end-to-end bar (same as stages 01/02)
# vs single-chip TTNN baseline (sharding/collective isolation). A genuine sharding
# or collective bug drops PCC far below 0.99; 0.997 is a strong isolation bar with
# honest margin for the observed short-sequence floor (T=16 = 0.99803, where the
# 4-device reduce_scatter bf16 accumulation has slightly more relative error).
PCC_BAR_SC = 0.997
MESH_SHAPE = (1, 4)

_REP_SYMBOLS = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "
_IPA_SENTENCES = [
    "hɛlˈO wˈɜːld",
    "ðə kwˈɪk brˈaʊn fˈɑks ʤˈʌmps ˈOvɚ ðə lˈeɪzi dˈɔɡ",
    "tˈɛnstɔɹɛnt bˈɪldz ˈAI ˈaksɛlɚˌeɪɾɚz",
]


# --------------------------------------------------------------------- helpers
def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rep_pool(vocab):
    return [vocab[c] for c in _REP_SYMBOLS if c in vocab]


def _rep_ids(vocab, batch, seq_len, seed):
    pool = _rep_pool(vocab)
    rng = random.Random(seed)
    rows = []
    for _ in range(batch):
        body = [rng.choice(pool) for _ in range(max(seq_len - 2, 0))]
        ids = ([0] + body + [0])[:seq_len]
        while len(ids) < seq_len:
            ids.append(0)
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def _ipa_ids(vocab, text):
    return torch.tensor([[0] + [vocab[c] for c in text if c in vocab] + [0]], dtype=torch.long)


def _kokoro_config_dict():
    from huggingface_hub import hf_hub_download

    return json.load(open(hf_hub_download(MODEL_ID, "config.json")))


def _albert_config():
    from transformers import AlbertConfig

    cfg = _kokoro_config_dict()
    return AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])


def _kokoro_vocab():
    return _kokoro_config_dict()["vocab"]


def _real_state_dict():
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(MODEL_ID, "kokoro-v1_0.pth")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["bert"]
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}


def _hf_model(config, state_dict):
    from transformers import AlbertModel

    m = AlbertModel(config).eval()
    m.load_state_dict(state_dict, strict=False)
    return m


def _hf_forward(hf, ids, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        return hf(ids, attention_mask=attention_mask).last_hidden_state


# ---- gather helpers: multichip output is sequence-sharded [b,1,S/TP,H] --------
def _gather_mc(mesh, out, batch, padded, seq_len, hidden):
    t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
    return t.reshape(batch, padded, hidden)[:, :seq_len, :].float()


def _gather_sc(mesh, out, batch, seq_len):
    # single-chip baseline runs replicated on the mesh; take device-0's copy
    t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return t[:batch, :seq_len, :].float()


def _mc_prefill(mc, mesh, ids, attention_mask=None):
    p = mc.prepare_inputs(ids, attention_mask=attention_mask)
    o = mc.prefill_forward(
        p["input_ids"],
        p["position_ids"],
        p["token_type_ids"],
        p["attention_mask"],
        batch=p["batch"],
        seq_len=p["padded_seq_len"],
    )
    return _gather_mc(mesh, o, p["batch"], p["padded_seq_len"], p["seq_len"], mc.hidden_size)


def _mc_decode(mc, mesh, ids, attention_mask=None):
    p = mc.prepare_inputs(ids, attention_mask=attention_mask)
    o = mc.decode_forward(
        p["input_ids"],
        p["position_ids"],
        p["token_type_ids"],
        p["attention_mask"],
        batch=p["batch"],
        seq_len=p["padded_seq_len"],
    )
    return _gather_mc(mesh, o, p["batch"], p["padded_seq_len"], p["seq_len"], mc.hidden_size)


def _sc_prefill(sc, mesh, ids, attention_mask=None):
    p = OptimizedDecoder.prepare_inputs(ids, mesh, attention_mask=attention_mask)
    o = sc.prefill_forward(
        p["input_ids"],
        p["position_ids"],
        p["token_type_ids"],
        p["attention_mask"],
        batch=p["batch"],
        seq_len=p["padded_seq_len"],
    )
    return _gather_sc(mesh, o, ids.shape[0], p["seq_len"])


# -------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def mesh():
    try:
        sys_desc = ttnn._ttnn.multi_device.SystemMeshDescriptor()
        sys_shape = tuple(sys_desc.shape())
    except Exception as e:
        pytest.skip(f"cannot query system mesh: {e}")
    if sys_shape[0] * sys_shape[1] < MESH_SHAPE[0] * MESH_SHAPE[1]:
        pytest.skip(f"system mesh {sys_shape} too small for {MESH_SHAPE}")
    # Physical 4-ring -> ring fabric + Topology.Ring is the fastest config.
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), trace_region_size=90000000)
    yield dev
    ttnn.close_mesh_device(dev)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def real_ctx(mesh):
    config = _albert_config()
    vocab = _kokoro_vocab()
    sd = _real_state_dict()
    hf = _hf_model(config, sd)
    mc = MultichipDecoder.from_state_dict(sd, hf_config=config, mesh_device=mesh)
    sc = OptimizedDecoder.from_state_dict(sd, hf_config=config, mesh_device=mesh)
    yield config, hf, mc, sc, vocab
    mc.release_traces()
    sc.release_traces()


# --------------------------------------------------- multichip-path assertions
def test_uses_multichip_tp_path(real_ctx, mesh):
    """The delivered path is real tensor parallelism: attention weights fractured
    across the mesh, FFN weights replicated, correct per-device shapes."""
    config, hf, mc, sc, vocab = real_ctx
    assert mc.tp == MESH_SHAPE[0] * MESH_SHAPE[1] == 4
    assert mc.local_heads == config.num_attention_heads // mc.tp == 3
    H = config.hidden_size
    gw = mc.local_hidden  # 192
    # QKV fractured: per-device out width == 3 * local_hidden (Q|K|V local heads)
    assert tuple(mc.w["qkv_w"].shape) == (H, 3 * gw), mc.w["qkv_w"].shape
    assert tuple(mc.w["qkv_b"].shape)[-1] == 3 * gw
    # WO fractured on the concat-heads input dim
    assert tuple(mc.w["dense_w"].shape) == (gw, H), mc.w["dense_w"].shape
    # FFN replicated (full intermediate on every device)
    assert tuple(mc.w["ffn_w"].shape) == (H, config.intermediate_size)
    assert tuple(mc.w["ffn_out_w"].shape) == (config.intermediate_size, H)
    # precision policy preserved from the optimized baseline
    assert mc.w["qkv_w"].get_dtype() == ttnn.bfloat8_b
    assert mc.w["ffn_w"].get_dtype() == ttnn.bfloat8_b
    assert mc.activation_dtype == ttnn.bfloat16
    assert mc.policy.fp32_dest_acc is True


def test_only_two_collectives_per_layer(real_ctx, mesh):
    """Guard the collective budget: exactly one all_gather + one reduce_scatter per
    layer (the whole design's performance contract)."""
    import inspect

    src = inspect.getsource(MultichipDecoder._albert_layer)
    assert src.count("_all_gather_seq(") == 1, "expected exactly 1 all_gather per layer"
    assert src.count("_reduce_scatter_seq(") == 1, "expected exactly 1 reduce_scatter per layer"


def test_selected_optimizations_active(real_ctx, mesh):
    """The optimized default path must actually use the two selected wins: a
    block-sharded L1 LayerNorm (off the ~4-core interleaved kernel) and persistent
    CCL output buffers. Guards against a silent revert to the stage-03 defaults."""
    config, hf, mc, sc, vocab = real_ctx
    assert mc.opt.norm_sharded is True, "sharded LayerNorm must be enabled in the optimized default"
    assert mc.opt.persistent_ccl is True, "persistent CCL buffers must be enabled in the optimized default"
    assert mc.opt.fuse_residual is False, "residual-fusion was rejected (slower interleaved kernel)"
    # sharded-norm config is legal for every tile-padded local-seq (m_tiles 1..4)
    for m_tiles in (1, 2, 3, 4):
        mem, pc = mc._norm_sharded_config(m_tiles)
        assert mem.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    # after a decode, the persistent AG + RS buffers were actually materialised
    _mc_decode(mc, mesh, _rep_ids(vocab, 1, 128, seed=1))
    assert len(mc._ag_bufs) >= 1 and len(mc._rs_bufs) >= 1, "persistent CCL buffers were not allocated"


# ----------------------------- PCC vs single-chip TTNN baseline (isolation) ----
@pytest.mark.parametrize("seq_len", [8, 16, 31, 32, 33, 64, 96, 128, 256, 500, 511, 512])
def test_prefill_pcc_vs_single_chip(real_ctx, mesh, seq_len):
    config, hf, mc, sc, vocab = real_ctx
    ids = _rep_ids(vocab, 1, seq_len, seed=seq_len)
    got = _mc_prefill(mc, mesh, ids)
    ref_sc = _sc_prefill(sc, mesh, ids)
    ref_hf = _hf_forward(hf, ids)
    pcc_sc = _pcc(got, ref_sc)
    pcc_hf = _pcc(got, ref_hf)
    assert pcc_sc >= PCC_BAR_SC, f"prefill T={seq_len} vs single-chip PCC={pcc_sc:.6f} < {PCC_BAR_SC}"
    assert pcc_hf >= PCC_BAR, f"prefill T={seq_len} vs HF PCC={pcc_hf:.5f} < {PCC_BAR}"


@pytest.mark.parametrize("batch", [2, 4, 8, 32])
def test_prefill_batch_pcc(real_ctx, mesh, batch):
    config, hf, mc, sc, vocab = real_ctx
    ids = _rep_ids(vocab, batch, 128, seed=1000 + batch)
    got = _mc_prefill(mc, mesh, ids)
    assert _pcc(got, _sc_prefill(sc, mesh, ids)) >= PCC_BAR_SC
    assert _pcc(got, _hf_forward(hf, ids)) >= PCC_BAR


def test_prefill_real_ipa_sentences(real_ctx, mesh):
    config, hf, mc, sc, vocab = real_ctx
    for text in _IPA_SENTENCES:
        ids = _ipa_ids(vocab, text)
        got = _mc_prefill(mc, mesh, ids)
        assert _pcc(got, _hf_forward(hf, ids)) >= PCC_BAR, f"IPA '{text[:24]}'"
        assert _pcc(got, _sc_prefill(sc, mesh, ids)) >= PCC_BAR_SC


# ------------------------------------------------ decode (traced) PCC ----------
@pytest.mark.parametrize("seq_len", [16, 32, 33, 64, 128, 500, 511, 512])
def test_decode_traced_pcc(real_ctx, mesh, seq_len):
    config, hf, mc, sc, vocab = real_ctx
    ids = _rep_ids(vocab, 1, seq_len, seed=5000 + seq_len)
    got = _mc_decode(mc, mesh, ids)
    assert _pcc(got, _sc_prefill(sc, mesh, ids)) >= PCC_BAR_SC, f"decode T={seq_len} vs single-chip"
    assert _pcc(got, _hf_forward(hf, ids)) >= PCC_BAR, f"decode T={seq_len} vs HF"


def test_decode_traced_masked_batch(real_ctx, mesh):
    """Non-aligned, masked variable-length batch on the target mesh."""
    config, hf, mc, sc, vocab = real_ctx
    lengths = [113, 47]  # non-aligned; max 113 -> padded 128
    max_len = max(lengths)
    ids = torch.zeros((len(lengths), max_len), dtype=torch.long)
    mask = torch.zeros((len(lengths), max_len), dtype=torch.long)
    for i, L in enumerate(lengths):
        ids[i, :L] = _rep_ids(vocab, 1, L, seed=6000 + i)[0]
        mask[i, :L] = 1
    got = _mc_decode(mc, mesh, ids, attention_mask=mask)
    ref = _hf_forward(hf, ids, attention_mask=mask)
    for i, L in enumerate(lengths):
        assert _pcc(got[i, :L], ref[i, :L]) >= PCC_BAR, f"masked row {i} (len {L})"


# ----------------------------------------- per-layer-kind component comparison -
def test_component_pcc_vs_single_chip(real_ctx, mesh):
    """Validate each meaningful layer kind (embedding + one AlbertLayer covering
    attention and FFN sublayers + both collectives) against the single-chip TTNN
    baseline, not just the stacked-12-layer output."""
    config, hf, mc, sc, vocab = real_ctx
    S = 256
    ids = _rep_ids(vocab, 1, S, seed=4242)
    H = config.hidden_size

    # --- embedding layer kind ---
    p_mc = mc.prepare_inputs(ids)
    emb_mc = mc._embed(
        p_mc["input_ids"], p_mc["position_ids"], p_mc["token_type_ids"], p_mc["batch"], p_mc["padded_seq_len"] // mc.tp
    )
    emb_mc_t = ttnn.to_torch(emb_mc, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)).reshape(1, S, H).float()

    p_sc = OptimizedDecoder.prepare_inputs(ids, mesh)
    emb_sc = sc._embed(p_sc["input_ids"], p_sc["position_ids"], p_sc["token_type_ids"])
    emb_sc_t = ttnn.to_torch(emb_sc, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1, :S, :].float()
    pcc_emb = _pcc(emb_mc_t, emb_sc_t)
    assert pcc_emb >= PCC_BAR_SC, f"embedding component PCC={pcc_emb:.6f}"

    # --- one AlbertLayer kind (attention sublayer + FFN sublayer + AG + RS) ---
    # feed identical hidden into both: sc gets full replicated, mc gets seq-shard
    torch.manual_seed(7)
    hidden = torch.randn(1, S, H) * 0.5
    hid_sc = ttnn.from_torch(
        hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    out_sc = sc._albert_layer(hid_sc, None, 1, S)
    out_sc_t = ttnn.to_torch(out_sc, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1, :S, :].float()

    hid_mc = ttnn.from_torch(
        hidden.reshape(1, 1, S, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 2), mesh_shape=MESH_SHAPE),
    )
    out_mc = mc._albert_layer(hid_mc, None, 1, S, S // mc.tp)
    out_mc_t = ttnn.to_torch(out_mc, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)).reshape(1, S, H).float()
    pcc_layer = _pcc(out_mc_t, out_sc_t)
    assert pcc_layer >= PCC_BAR_SC, f"AlbertLayer component PCC={pcc_layer:.6f}"


# ----------------------------------------------------------------- determinism
def test_decode_determinism(real_ctx, mesh):
    config, hf, mc, sc, vocab = real_ctx
    ids = _rep_ids(vocab, 1, 128, seed=88)
    a = _mc_decode(mc, mesh, ids)
    b = _mc_decode(mc, mesh, ids)
    assert torch.equal(a, b), "traced multichip decode not deterministic"


def test_decode_is_stateless(real_ctx, mesh):
    """Non-autoregressive: multichip decode == multichip prefill for same input."""
    config, hf, mc, sc, vocab = real_ctx
    ids = _rep_ids(vocab, 1, 64, seed=303)
    assert torch.equal(_mc_prefill(mc, mesh, ids), _mc_decode(mc, mesh, ids))


# ------------------------------------------------------ stress / repeated runs
def test_decode_stress_repeated_replay(real_ctx, mesh):
    """Revisited shapes, repeated traced replays: correct + bit-stable (trace
    buffer reuse + persistent CCL semaphore/buffer stress on the mesh)."""
    config, hf, mc, sc, vocab = real_ctx
    shapes = [512, 128, 511, 64, 128, 512]
    for rep, L in enumerate(shapes):
        ids = _rep_ids(vocab, 1, L, seed=9000 + L)
        ref = _hf_forward(hf, ids)
        first = None
        for _ in range(3):
            got = _mc_decode(mc, mesh, ids)
            if first is None:
                first = got
            else:
                assert torch.equal(first, got), f"replay nondeterministic T={L} rep={rep}"
        assert _pcc(first, ref) >= PCC_BAR, f"stress decode T={L}"
