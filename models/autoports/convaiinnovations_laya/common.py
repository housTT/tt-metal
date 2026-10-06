# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HF_MODEL = "convaiinnovations/laya"
LAYA_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
MODEL_ID = HF_MODEL
MODEL_REVISION = LAYA_REVISION

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"

ROOT_ALLOW_PATTERNS = [
    "config.json",
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
]

DEFAULT_MAX_LEN = 512
DEFAULT_HEAD_MAX_LEN = 192
DEFAULT_THREADS = 6
DEFAULT_GATE_SEED = 13
DEFAULT_GATE_PER_WORKFLOW = 10
DEFAULT_TYPED_DECISIONS_PARQUET = "/home/hous/dev/laya/evals/typed_decisions/data/test-00000-of-00001.parquet"

AUTOPORT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = AUTOPORT_DIR / "vendor"


def cpu_threads() -> int:
    return int(os.environ.get("LAYA_CPU_THREADS", DEFAULT_THREADS))


def configure_torch_threads(threads: int | None = None) -> int:
    n = int(threads) if threads else cpu_threads()
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    torch.set_num_threads(n)
    return n


def model_dir() -> str:
    explicit = os.environ.get("LAYA_MODEL_DIR")
    if explicit:
        d = explicit
    else:
        from huggingface_hub import snapshot_download

        repo = os.environ.get("HF_MODEL", HF_MODEL)
        rev = os.environ.get("LAYA_REVISION", LAYA_REVISION)
        try:
            d = snapshot_download(repo, revision=rev, allow_patterns=ROOT_ALLOW_PATTERNS, local_files_only=True)
        except Exception:
            d = snapshot_download(repo, revision=rev, allow_patterns=ROOT_ALLOW_PATTERNS)
    sub = os.environ.get("LAYA_SUBFOLDER")
    if sub:
        d = os.path.join(d, sub)
    if not os.path.isfile(os.path.join(d, "model.safetensors")):
        raise FileNotFoundError(f"model.safetensors not found under {d}")
    return d


def checkpoint_pins(d: str | None = None) -> dict:
    """Hub repo and revision of a checkpoint directory, read from its huggingface_hub snapshot path (models--<org>--<name>/snapshots/<revision>); the English pins when the path is not a snapshot."""
    real = os.path.realpath(d or model_dir())
    parts = real.split(os.sep)
    repo, revision = HF_MODEL, LAYA_REVISION
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i >= 1 and parts[i - 1].startswith("models--") and i + 1 < len(parts):
            repo = parts[i - 1][len("models--") :].replace("--", "/")
            revision = parts[i + 1]
    return {"hf_model": repo, "revision": revision, "model_dir": real}


def load_rl_config(d: str | None = None) -> dict:
    with open(os.path.join(d or model_dir(), "rl_agent_config.json")) as f:
        return json.load(f)


def load_config(d: str | None = None):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(os.path.join(d or model_dir(), "encoder"))


def rope_theta(config, layer_type: str) -> float:
    return float(config.rope_parameters[layer_type]["rope_theta"])


def rope_thetas(config) -> dict:
    return {lt: rope_theta(config, lt) for lt in (FULL_ATTENTION, SLIDING_ATTENTION)}


def layer_types(config) -> list:
    return list(config.layer_types)


def sliding_window_half(config) -> int:
    half = int(config.local_attention) // 2
    if int(getattr(config, "sliding_window", half)) != half:
        raise ValueError(f"config.sliding_window {config.sliding_window} != local_attention // 2 {half}")
    return half


def load_tokenizer(d: str | None = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.path.join(d or model_dir(), "tokenizer"))


def load_state_dict(d: str | None = None, dtype=torch.float32) -> dict:
    from safetensors.torch import load_file

    sd = load_file(os.path.join(d or model_dir(), "model.safetensors"))
    if dtype is None:
        return sd
    return {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}


def encoder_state_dict(sd: dict) -> dict:
    return {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")}


def load_torch_model(dtype=torch.float32, attn_implementation=None, d: str | None = None):
    from transformers import AutoModel

    d = d or model_dir()
    config = load_config(d)
    model = AutoModel.from_config(config, attn_implementation=attn_implementation or "eager")
    sd = encoder_state_dict(load_state_dict(d, dtype=torch.float32))
    result = model.load_state_dict(sd, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"encoder load: missing={result.missing_keys} unexpected={result.unexpected_keys}")
    model = model.to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def vendor_module(name: str):
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    return importlib.import_module(name)


def rl_common():
    from models.autoports.convaiinnovations_laya.vendor import rl_common as mod

    return mod


def to_internal(qdef: dict) -> dict:
    return vendor_module("rl_agent_api").RLAgent._to_internal(qdef)


def typed_decisions_path() -> str:
    return os.environ.get("LAYA_TYPED_DECISIONS_PARQUET", DEFAULT_TYPED_DECISIONS_PARQUET)


def load_typed_decisions(path: str | None = None) -> list:
    import pyarrow.parquet as pq

    rows = pq.read_table(path or typed_decisions_path()).to_pylist()
    cases = []
    for i, r in enumerate(rows):
        state = r["state"]
        try:
            state = json.loads(state)
        except Exception:
            pass
        cases.append(
            {
                "index": i,
                "id": r["id"],
                "workflow": r["workflow"],
                "state": state,
                "questions": json.loads(r["questions"]),
                "gold": json.loads(r["gold"]),
            }
        )
    return cases


def gate_subset(cases: list, seed: int = DEFAULT_GATE_SEED, per_workflow: int = DEFAULT_GATE_PER_WORKFLOW) -> list:
    by_wf = {}
    for c in cases:
        by_wf.setdefault(c["workflow"], []).append(c["index"])
    rng = np.random.RandomState(seed)
    picked = []
    for wf in by_wf:
        idxs = by_wf[wf]
        n = min(per_workflow, len(idxs))
        picked.extend(sorted(int(i) for i in rng.choice(idxs, n, replace=False)))
    picked = sorted(picked)
    by_index = {c["index"]: c for c in cases}
    return [by_index[i] for i in picked]


def case_items(tok, case: dict, max_len: int = DEFAULT_MAX_LEN, head_max_len: int = DEFAULT_HEAD_MAX_LEN) -> list:
    rc = rl_common()
    items = []
    for qid, qdef in case["questions"].items():
        q = to_internal(qdef)
        ids, markers = rc.build_sequence(tok, case["state"], q, max_len, head_max_len)
        k = len(rc.render_options(q))
        if len(markers) != k:
            raise ValueError(f"case {case.get('id')} question {qid}: {len(markers)} markers for {k} options")
        items.append(
            {
                "ids": ids,
                "markers": markers,
                "qtype": rc.QTYPES[q["t"]],
                "target": [0.0] * k,
                "label": -1,
                "episode": 0,
                "ep_step": 0,
                "ep_len": 1,
                "src": "typed_decisions",
                "case_id": case.get("id"),
                "case_index": case.get("index"),
                "workflow": case.get("workflow"),
                "qid": qid,
                "q": q,
                "k": k,
            }
        )
    return items


def collate(items: list, pad_id: int, seq_len: int | None = None) -> dict:
    b = rl_common().collate_items([items], pad_id)
    if seq_len is not None:
        n, L = b["input_ids"].shape
        if L > seq_len:
            raise ValueError(f"items are {L} tokens long, bucket is {seq_len}")
        ids = torch.full((n, seq_len), pad_id, dtype=torch.long)
        att = torch.zeros((n, seq_len), dtype=torch.long)
        ids[:, :L] = b["input_ids"]
        att[:, :L] = b["attention_mask"]
        b["input_ids"], b["attention_mask"] = ids, att
    return b


_GATE_ITEMS_CACHE = {}


def gate_items(tok, seq_len: int = DEFAULT_MAX_LEN, head_max_len: int = DEFAULT_HEAD_MAX_LEN) -> list:
    key = (seq_len, head_max_len)
    if key not in _GATE_ITEMS_CACHE:
        cases = gate_subset(load_typed_decisions())
        items = []
        for c in cases:
            items.extend(case_items(tok, c, max_len=seq_len, head_max_len=head_max_len))
        _GATE_ITEMS_CACHE[key] = items
    return _GATE_ITEMS_CACHE[key]


def build_batch(
    seq_len: int = DEFAULT_MAX_LEN, batch_size: int = 1, seed: int = 0, head_max_len: int = DEFAULT_HEAD_MAX_LEN
) -> dict:
    tok = load_tokenizer()
    items = gate_items(tok, seq_len=seq_len, head_max_len=head_max_len)
    order = np.random.RandomState(seed).permutation(len(items))
    chosen = [items[int(order[i % len(items)])] for i in range(batch_size)]
    return collate(chosen, tok.pad_token_id, seq_len=seq_len)


def build_inputs(seq_len: int = DEFAULT_MAX_LEN, batch_size: int = 1, seed: int = 0):
    b = build_batch(seq_len=seq_len, batch_size=batch_size, seed=seed)
    return b["input_ids"], b["attention_mask"]


SAMPLE_TEXT = """The development of specialized hardware for machine learning has followed a
winding path. Early neural networks ran on general purpose processors, where the dominant
cost was moving data rather than computing on it. Graphics processors changed that calculus
by offering wide parallel arithmetic, though they retained a memory hierarchy designed for
rendering triangles rather than multiplying matrices. More recent accelerators abandon that
inheritance entirely. They place large scratchpad memories adjacent to compute units and
expose the movement of tensors as an explicit part of the programming model. The result is a
machine that rewards careful placement of data and punishes casual copying. Rivers carve
their valleys slowly, and the shape of a watershed records centuries of small decisions made
by water. A meander begins as a trivial asymmetry, perhaps a fallen tree or a patch of harder
rock. Flow accelerates on the outer bank and slows on the inner one, so sediment erodes from
one side and accumulates on the other. Over time the curve deepens until the river doubles
back on itself and cuts a new channel across the neck, abandoning the loop as an oxbow lake.
Farmers who work such floodplains learn to read these scars in the soil, because the
abandoned channels hold water differently and grow different crops. Fermentation is
controlled decay. A baker who leaves flour and water in a warm room is cultivating a
community of wild yeasts and lactic acid bacteria, each consuming sugars and excreting
compounds that the others tolerate or exploit. The sour flavour of a mature starter comes
from acids that also suppress competing organisms, which is why the culture becomes more
stable as it ages. Temperature shifts the balance. Warmer conditions favour bacteria that
produce sharper acids, while cooler ones let yeasts dominate and yield a milder loaf with
more gas. Questions about legal personhood rarely arrive in tidy form. A corporation is not a
person in any biological sense, yet it can own property, enter contracts, sue and be sued.
Courts extended these capacities gradually and for practical reasons, not because anyone
believed a firm possessed an inner life. Difficulty appears when doctrines built for one
purpose are borrowed for another. Rights of conscience presuppose a bearer capable of holding
convictions, and applying them to an entity whose decisions emerge from committees and
fiduciary duties produces conclusions that satisfy neither the letter nor the spirit of the
original rule. Cartographers of the eighteenth century faced a stubborn problem of longitude.
Latitude could be read from the sun or the pole star, but establishing how far east or west a
ship had travelled required knowing the time at a reference meridian, and no pendulum clock
survived the pitching of a deck. The eventual solution was mechanical rather than
astronomical, a sequence of marine chronometers whose escapements tolerated motion and
temperature swings. Astronomers had meanwhile proposed lunar distance tables, which worked but
demanded hours of computation from an exhausted navigator. Both methods persisted side by side
for decades, because redundancy at sea is worth more than elegance. The vocabulary of colour
varies enormously between languages, and the variation is not arbitrary. Communities that
distinguish few basic terms almost always separate dark from light first, then add red, then
green or yellow, then blue. Researchers once read this ordering as evidence that perception
itself differed, but later work suggested the constraint lies in what distinctions prove
useful to name. Dyes and pigments matter here: a culture with access to a stable blue colorant
tends to lexicalise blue earlier than one without. Glassblowers work within a narrow window of
temperature where the material is neither liquid nor solid but something between, stiff enough
to hold a shape yet mobile enough to yield. Skill consists largely in anticipating how quickly
that window closes, and in reheating before it does. An apprentice learns to read the colour of
the glowing mass rather than trust a clock, because the same nominal temperature behaves
differently depending on the thickness of the piece and the draught in the room."""
