import os
import time

import pytest
import torch
from loguru import logger

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
os.environ.setdefault("HF_MODEL", SNAPSHOT)

from models.autoports.cloudflare_clef.tests import test_engine as te
from models.autoports.cloudflare_clef.tt import encode as clef_encode
from models.autoports.cloudflare_clef.tt.engine import ClefEngine, tp2_mesh
from models.autoports.cloudflare_clef.tt.loader import ClefModelArgs

LAYERS = [0, 30]
T = 8192
BLOCK = 1024
RESULTS = {}


def row_pcc(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def subset_args(gate_fp32):
    class SubsetArgs(ClefModelArgs):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.layer_indices = list(LAYERS)
            self.n_layers = len(LAYERS)
            self.gdn_gate_fp32 = gate_fp32

    return SubsetArgs


@pytest.fixture(scope="module")
def submesh():
    with tp2_mesh(os.environ.get("CLEF_PARENT", "1x4")) as sub:
        yield sub


@pytest.fixture(scope="module")
def hidden_states():
    tokenizer = clef_encode.load_tokenizer(SNAPSHOT)
    records = te.read_jsonl(te.RECORDS)
    ids = te.request_ids(tokenizer, records, T)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    hf = te.hf_model(64)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = hf.model.language_model(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs = {i: out.hidden_states[i][0].float() for L in LAYERS for i in (L, L + 1)}
    logger.info(f"HF T={T}: {time.perf_counter() - t0:.1f} s")
    return hs


def layer_delta_pcc(engine, submesh, layer_idx, h_in, h_out):
    import ttnn

    model = engine.model
    margs = engine.args
    layer = dict(zip(model.layer_indices, model.layers))[layer_idx]
    comp3 = ttnn.ConcatMeshToTensor(submesh, dim=3)
    model._reset_gdn_state_for_new_sequence()
    got = torch.empty(T, margs.dim)
    for cs in range(0, T, BLOCK):
        ce = cs + BLOCK
        full = torch.zeros(1, 1, BLOCK, margs.dim, dtype=torch.bfloat16)
        full[0, 0] = h_in[cs:ce].to(torch.bfloat16)
        x_in = ttnn.from_torch(
            full,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=3),
        )
        x_out = layer.forward(x_in, mode="prefill", chunk_size=margs.gdn_chunk_size, valid_len=BLOCK)
        ttnn.deallocate(x_in)
        got[cs:ce] = ttnn.to_torch(x_out, mesh_composer=comp3)[0, 0, :BLOCK].float()
        ttnn.deallocate(x_out)
    p = row_pcc(got - h_in.to(torch.bfloat16).float(), h_out - h_in)
    return dict(
        mean=round(float(p.mean()), 6),
        min=round(float(p.min()), 6),
        last_block_mean=round(float(p[T - BLOCK :].mean()), 6),
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("gate_fp32", [False, True], ids=["gate_bf16", "gate_fp32"])
def test_gdn_gate_fp32_isolated_layers(submesh, hidden_states, gate_fp32):
    engine = ClefEngine(submesh, args_cls=subset_args(gate_fp32), max_state_len=te.MAX_STATE_LEN, snapshot_slots=1)
    assert engine.device_dtypes["gdn_gate_fp32"] is gate_fp32
    assert engine.device_dtypes["gdn_dt_bias"] == ("DataType.FLOAT32" if gate_fp32 else "DataType.BFLOAT16")
    rows = {}
    for L in LAYERS:
        rows[L] = layer_delta_pcc(engine, submesh, L, hidden_states[L], hidden_states[L + 1])
        logger.info(f"gate_fp32={gate_fp32} layer {L} T={T}: {rows[L]}")
    RESULTS[gate_fp32] = rows
    if gate_fp32:
        assert rows[0]["mean"] >= 0.9999 and rows[0]["min"] >= 0.9995, rows[0]
        assert rows[30]["mean"] >= 0.999 and rows[30]["min"] >= 0.99, rows[30]
        if False in RESULTS:
            before = RESULTS[False]
            assert rows[30]["mean"] > before[30]["mean"] + 0.03, (before[30], rows[30])
            assert abs(rows[0]["mean"] - before[0]["mean"]) < 1e-4, (before[0], rows[0])
            logger.info(
                f"layer 30 delta PCC mean {before[30]['mean']} -> {rows[30]['mean']}, layer 0 {before[0]['mean']} -> {rows[0]['mean']}"
            )
    else:
        assert rows[0]["mean"] >= 0.9999, rows[0]
