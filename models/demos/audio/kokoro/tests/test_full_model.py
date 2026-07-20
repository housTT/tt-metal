# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Full-model tests for hexgrad/Kokoro-82M (tt/model.py + tt/generator.py).

Covers the full HF forward path (last_hidden_state PCC vs HF), the readiness
Generator contract, on-device greedy argmax (no host argmax), the split-sampling
trace-feedback/determinism contract, non-aligned prompt lengths, and batch>1.
The model is a non-autoregressive bidirectional encoder: KV cache / paged cache /
current-position advance / token-feedback loop are N/A (see tt/model.py header).
"""
import inspect
import json

import pytest
import torch

import ttnn
from models.common.readiness_check.contract import Generator
from models.demos.audio.kokoro.tt.generator import KokoroGenerator, build_generator

MODEL_ID = "hexgrad/Kokoro-82M"
MESH_SHAPE = (1, 4)
PCC_BAR = 0.995


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.fixture(scope="module")
def ctx():
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig, AlbertModel

    try:
        sys_desc = ttnn._ttnn.multi_device.SystemMeshDescriptor()
        sys_shape = tuple(sys_desc.shape())
    except Exception as e:
        pytest.skip(f"cannot query system mesh: {e}")
    if sys_shape[0] * sys_shape[1] < MESH_SHAPE[0] * MESH_SHAPE[1]:
        pytest.skip(f"system mesh {sys_shape} too small for {MESH_SHAPE}")

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    vocab = cfg["vocab"]
    config = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    hf = AlbertModel(config).eval()
    hf.load_state_dict(_state_dict(), strict=False)

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), trace_region_size=90000000)
    gen = build_generator(model_dir="models/demos/audio/kokoro", mesh_device=mesh)
    yield config, hf, vocab, gen, mesh
    gen.teardown()
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _state_dict():
    from huggingface_hub import hf_hub_download

    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}


def _ids(vocab, text):
    return torch.tensor([[0] + [vocab[c] for c in text if c in vocab] + [0]], dtype=torch.long)


def _hidden_ref(hf, ids):
    with torch.no_grad():
        return hf(ids, attention_mask=torch.ones_like(ids)).last_hidden_state


# ---------------------------------------------------------------- contract
def test_generator_contract():
    assert issubclass(KokoroGenerator, Generator)
    # enable_trace must be an explicit keyword (teacher-forcing runner requires it)
    params = inspect.signature(KokoroGenerator.generate).parameters
    assert params["enable_trace"].kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


# ------------------------------------------------------ full-model accuracy
@pytest.mark.parametrize("seq_len", [8, 31, 33, 64, 127, 128, 200, 511, 512])
def test_full_model_pcc_vs_hf(ctx, seq_len):
    config, hf, vocab, gen, mesh = ctx
    body = [vocab[c] for c in "ðəkwɪkbraʊnfɑksʤʌmpsOvɚleɪzidɔɡ" if c in vocab]
    ids = torch.zeros((1, seq_len), dtype=torch.long)
    for i in range(1, seq_len - 1):
        ids[0, i] = body[i % len(body)]
    hidden = gen.model.forward(ids, traced=False)
    assert _pcc(hidden, _hidden_ref(hf, ids)) >= PCC_BAR, f"T={seq_len}"


def test_full_model_pcc_batch(ctx):
    config, hf, vocab, gen, mesh = ctx
    body = [vocab[c] for c in "ðəkwɪkbraʊnfɑks" if c in vocab]
    ids = torch.zeros((4, 64), dtype=torch.long)
    for r in range(4):
        for i in range(1, 63):
            ids[r, i] = body[(i + r) % len(body)]
    assert _pcc(gen.model.forward(ids, traced=False), _hidden_ref(hf, ids)) >= PCC_BAR


def test_traced_matches_eager(ctx):
    config, hf, vocab, gen, mesh = ctx
    ids = _ids(vocab, "ðə kwˈɪk brˈaʊn fˈɑks")
    eager = gen.model.forward(ids, traced=False)
    traced = gen.model.forward(ids, traced=True)
    assert _pcc(eager, traced) >= 0.999


# ------------------------------------------------------- generator low level
def test_prefill_forward_shapes(ctx):
    config, hf, vocab, gen, mesh = ctx
    ids = _ids(vocab, "ðə kwˈɪk brˈaʊn fˈɑks")
    alll = gen.prefill_forward(ids, prompt_lens=[ids.shape[1]], return_all_logits=True)
    assert tuple(alll.shape) == (1, ids.shape[1], gen.vocab_size)
    last = gen.prefill_forward(ids, prompt_lens=[ids.shape[1]], return_all_logits=False)
    assert tuple(last.shape) == (1, 1, gen.vocab_size)


def test_decode_on_device_argmax_matches_host(ctx):
    config, hf, vocab, gen, mesh = ctx
    ids = _ids(vocab, "tˈɛnstɔɹɛnt bˈɪldz ˈAI")
    tok = int(gen.decode_forward(ids, sample_on_device=True, enable_trace=True)[0])
    _, logits = gen.decode_forward(ids, sample_on_device=False, enable_trace=True, want_logits=True)
    assert tok == int(torch.argmax(logits[0]))
    # no host argmax was used on the on-device greedy path
    assert gen.counters["host_argmax"] == 0


# ----------------------------------------------------------- generate paths
def test_free_running_non_degenerate(ctx):
    config, hf, vocab, gen, mesh = ctx
    prompt = _ids(vocab, "ðə sˈʌn wˈʌz ʃˈaɪnɪŋ ˈOvɚ ðə mˈaʊntənz ænd ðə vˈæli")[0].tolist()
    out = gen.generate(prompt_token_ids=prompt, max_new_tokens=len(prompt), next_input=None, enable_trace=True)
    assert len(out) >= 20
    adj = sum(1 for a, b in zip(out, out[1:]) if a == b) / (len(out) - 1)
    assert adj < 0.10, f"adjacent-dup {adj} indicates degenerate output"


def test_teacher_forcing_callback_contract(ctx):
    config, hf, vocab, gen, mesh = ctx
    prompt = [0]
    gt = [vocab[c] for c in "ðə kwˈɪk brˈaʊn fˈɑks" if c in vocab]
    calls = []

    def ni(i, pred):
        calls.append((i, pred))
        return gt[i] if i < len(gt) else 0

    preds = gen.generate(prompt_token_ids=prompt, max_new_tokens=len(gt), next_input=ni, enable_trace=True)
    assert len(preds) == len(gt)
    assert len(calls) == len(gt)  # exactly one callback per requested token


def test_host_sampling_compat_mode(ctx):
    config, hf, vocab, gen, mesh = ctx
    prompt = _ids(vocab, "ðə kwˈɪk brˈaʊn")[0].tolist()
    on_dev = gen.generate(
        prompt_token_ids=prompt, max_new_tokens=len(prompt), next_input=None, enable_trace=True, host_sampling=False
    )
    host = gen.generate(
        prompt_token_ids=prompt, max_new_tokens=len(prompt), next_input=None, enable_trace=True, host_sampling=True
    )
    assert on_dev == host  # on-device argmax and host argmax agree (greedy)


# ------------------------------------------------ split-sampling trace test
def test_split_sampling_trace_feedback(ctx):
    """Two decode steps with different context -> outputs differ; repeated replay
    is deterministic; the greedy token is produced on device."""
    config, hf, vocab, gen, mesh = ctx
    a = _ids(vocab, "ðə kwˈɪk brˈaʊn")
    b = _ids(vocab, "tˈɛnstɔɹɛnt bˈɪldz ˈAI")
    ta = int(gen.decode_forward(a, sample_on_device=True, enable_trace=True)[0])
    tb = int(gen.decode_forward(b, sample_on_device=True, enable_trace=True)[0])
    ta2 = int(gen.decode_forward(a, sample_on_device=True, enable_trace=True)[0])
    assert ta != tb, "different contexts must produce different tokens (trace inputs refreshed)"
    assert ta == ta2, "repeated traced replay must be deterministic"


def test_reset_is_stateless_noop(ctx):
    config, hf, vocab, gen, mesh = ctx
    ids = _ids(vocab, "ðə kwˈɪk brˈaʊn")
    before = int(gen.decode_forward(ids, sample_on_device=True, enable_trace=True)[0])
    gen.reset()
    after = int(gen.decode_forward(ids, sample_on_device=True, enable_trace=True)[0])
    assert before == after
