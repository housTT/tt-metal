"""Write every sweep candidate as a precision config under doc/datatype_sweep/configs/."""
import copy
import json
from pathlib import Path

from models.autoports.qwen_qwen3_8b.tt.precision import baseline_config

OUT = Path("models/autoports/qwen_qwen3_8b/doc/datatype_sweep/configs")
OUT.mkdir(parents=True, exist_ok=True)

BFP4_READERS = {"qkv": 3, "o_proj": 2, "gate_proj": 3, "up_proj": 3, "down_proj": 2}
INNER = [str(i) for i in range(1, 35)]
EDGE = ["0", "35"]


def bfp4(group: str, fidelity: str = "lofi") -> dict:
    return {"dtype": "bfp4", "fidelity": fidelity, "readers": BFP4_READERS[group]}


def bfp8(group: str, fidelity: str = "lofi") -> dict:
    return {"dtype": "bfp8", "fidelity": fidelity, "readers": 1}


def candidate(config_id: str, description: str, groups: dict | None = None, exceptions: dict | None = None, **top):
    cfg = copy.deepcopy(baseline_config())
    cfg["config_id"] = config_id
    cfg["description"] = description
    for name, spec in (groups or {}).items():
        cfg["weight_groups"][name].update(spec)
    cfg["layer_exceptions"] = exceptions or {}
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    (OUT / f"{config_id}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(config_id)


def edge_baseline(groups: list[str]) -> dict:
    base = baseline_config()["weight_groups"]
    return {layer: {g: dict(base[g]) for g in groups} for layer in EDGE}


candidate(
    "c0_stage7_baseline", "stage-7 shipped policy (BFP8 LoFi QKV/O/gate/down, BFP4 LoFi up, BF16 KV, LoFi BFP8 LM head)"
)
candidate(
    "c1_all_bfp8_lofi",
    "canonical conservative policy: every projection BFP8 LoFi (stage-3 ALL_BFP8_POLICY)",
    {"up_proj": bfp8("up_proj")},
)
candidate(
    "c2_bfp8_hifi2",
    "baseline dtypes with HiFi2 on the BFP8 decode projections (QKV/O/gate/down); up stays BFP4 LoFi",
    {g: {"fidelity": "hifi2"} for g in ("qkv", "o_proj", "gate_proj", "down_proj")},
)
candidate("c3_kv_bfp8", "baseline weights with a BFP8 paged KV cache", kv_cache_dtype="bfp8")
candidate(
    "c4_up_bfp4_hifi2",
    "baseline with the BFP4 up projection at HiFi2 (BFP4+HiFi2 comparison for the selected BFP4 group)",
    {"up_proj": {"fidelity": "hifi2"}},
)
candidate(
    "c5_mlp_bfp4_inner",
    "gate/up/down BFP4 LoFi on layers 1 to 34; layers 0 and 35 keep the baseline MLP (BFP8 gate/down, BFP4 up)",
    {g: bfp4(g) for g in ("gate_proj", "up_proj", "down_proj")},
    edge_baseline(["gate_proj", "down_proj"]),
)
candidate(
    "c6_mlp_bfp4_all",
    "gate/up/down BFP4 LoFi on every layer",
    {g: bfp4(g) for g in ("gate_proj", "up_proj", "down_proj")},
)
candidate(
    "c7_gateup_bfp4_inner",
    "gate and up BFP4 LoFi on layers 1 to 34 (down BFP8, the first restore step after c5); layers 0 and 35 baseline",
    {"gate_proj": bfp4("gate_proj")},
    edge_baseline(["gate_proj"]),
)
candidate(
    "c8_attn_bfp4_inner",
    "QKV and O BFP4 LoFi on layers 1 to 34 with the baseline MLP; layers 0 and 35 baseline",
    {"qkv": bfp4("qkv"), "o_proj": bfp4("o_proj")},
    edge_baseline(["qkv", "o_proj"]),
)
candidate(
    "c9_lm_head_bfp4",
    "baseline with a BFP4 LoFi LM head",
    lm_head={"dtype": "bfp4", "decode_fidelity": "lofi", "prefill_fidelity": "lofi"},
)
candidate(
    "c10_mlp_bfp4_inner_hifi2",
    "c5 dtypes with HiFi2 on the BFP4 MLP groups (BFP4+HiFi2 comparison for the material BFP4 groups)",
    {g: bfp4(g, "hifi2") for g in ("gate_proj", "up_proj", "down_proj")},
    edge_baseline(["gate_proj", "down_proj"]),
)
candidate(
    "c11_down_bfp4_inner",
    "down BFP4 LoFi on layers 1 to 34 next to the baseline BFP4 up (gate BFP8); layers 0 and 35 baseline",
    {"down_proj": bfp4("down_proj")},
    edge_baseline(["down_proj"]),
)

ALL = ("qkv", "o_proj", "gate_proj", "up_proj", "down_proj")
LM4 = {"dtype": "bfp4", "decode_fidelity": "lofi", "prefill_fidelity": "lofi"}
candidate(
    "c12_all_bfp4_inner",
    "every projection BFP4 LoFi on layers 1 to 34 (c5 plus c8); layers 0 and 35 baseline",
    {g: bfp4(g) for g in ALL},
    edge_baseline(["qkv", "o_proj", "gate_proj", "down_proj"]),
)
candidate(
    "c13_all_bfp4_inner_lm_bfp4",
    "c12 plus a BFP4 LoFi LM head",
    {g: bfp4(g) for g in ALL},
    edge_baseline(["qkv", "o_proj", "gate_proj", "down_proj"]),
    lm_head=LM4,
)
candidate(
    "c14_mlp_bfp4_all_lm_bfp4",
    "c6 (gate/up/down BFP4 LoFi on every layer) plus a BFP4 LoFi LM head",
    {g: bfp4(g) for g in ("gate_proj", "up_proj", "down_proj")},
    lm_head=LM4,
)
candidate(
    "c15_mlp_bfp4_inner_lm_bfp4",
    "c5 (gate/up/down BFP4 LoFi on layers 1 to 34) plus a BFP4 LoFi LM head",
    {g: bfp4(g) for g in ("gate_proj", "up_proj", "down_proj")},
    edge_baseline(["gate_proj", "down_proj"]),
    lm_head=LM4,
)
candidate(
    "c16_all_bfp4_all_layers",
    "every projection BFP4 LoFi on every layer plus a BFP4 LoFi LM head (the most aggressive legal policy)",
    {g: bfp4(g) for g in ALL},
    lm_head=LM4,
)
candidate(
    "c17_mlp_attn_bfp4_all_layers",
    "every projection BFP4 LoFi on every layer, BFP8 LoFi LM head (c16 without the LM-head change)",
    {g: bfp4(g) for g in ALL},
)

candidate(
    "c18_lm_head_bfp4_readers2",
    "c9 (BFP4 LoFi LM head) with 2 DRAM readers per bank on the LM-head splits",
    lm_head={"dtype": "bfp4", "decode_fidelity": "lofi", "prefill_fidelity": "lofi", "readers": 2},
)
candidate(
    "c19_lm_head_bfp4_readers4",
    "c9 (BFP4 LoFi LM head) with 4 DRAM readers per bank on the LM-head splits",
    lm_head={"dtype": "bfp4", "decode_fidelity": "lofi", "prefill_fidelity": "lofi", "readers": 4},
)
