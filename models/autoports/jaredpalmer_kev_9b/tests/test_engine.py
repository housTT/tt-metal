import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.autoports.jaredpalmer_kev_9b.tt.engine import KevEngine
from models.autoports.jaredpalmer_kev_9b.tt.loader import KevModelArgs
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

TRACE_REGION_SIZE = 1 << 30
DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": TRACE_REGION_SIZE}]
LAYER_CASES = [pytest.param(4, id="l4"), pytest.param(32, id="l32", marks=pytest.mark.slow)]
MODE_CASES = [
    pytest.param("eager", id="eager"),
    pytest.param("traced", id="traced"),
    pytest.param("policy", id="policy"),
]


def mode_kwargs(mode):
    return {"traced": mode != "eager", "matmul_policy": mode == "policy"}


MAX_STATE_LEN = 8192
FLIP_MARGIN = 0.05
MAX_ABS_DP = 0.10
VOCAB_SAMPLE = 100000
REFERENCE_DIR = Path("/home/hous/dev/kev/reports/reference")

_hf_cache = {}
_results = {}


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-8)).item()


def fixed_ids(seed, length):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SAMPLE, (1, length), generator=g, dtype=torch.long)


def sample_positions(length, n, first=1):
    pos = torch.linspace(first, length - 1, n).round().long().tolist()
    return sorted(set(pos + [length - 1]))


def hf_model(n_layers):
    if n_layers in _hf_cache:
        return _hf_cache[n_layers]
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    ckpt = os.environ["HF_MODEL"]
    cfg = Qwen3_5TextConfig.from_pretrained(ckpt)
    cfg.num_hidden_layers = n_layers
    cfg.layer_types = cfg.layer_types[:n_layers]
    model = Qwen3_5ForCausalLM.from_pretrained(ckpt, config=cfg, dtype=torch.bfloat16).eval()
    _hf_cache.clear()
    _hf_cache[n_layers] = model
    return model


def hf_hidden(ids, positions, n_layers):
    model = hf_model(n_layers)
    with torch.no_grad():
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
    return h[positions].float()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("n_layers", LAYER_CASES)
@pytest.mark.parametrize("T", [300, 1500, 2300])
def test_hidden_vs_hf(device, n_layers, T):
    device.enable_program_cache()
    engine = KevEngine(device, args_cls=Qwen36ModelArgs, max_state_len=MAX_STATE_LEN, n_layers=n_layers)
    assert type(engine.args) is Qwen36ModelArgs
    ids = fixed_ids(T, T)
    positions = sample_positions(T, 8)
    t0 = time.perf_counter()
    tt = engine.prefill_hidden(ids, positions)
    t_tt = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = hf_hidden(ids, positions, n_layers)
    t_hf = time.perf_counter() - t0
    pccs = [pcc(ref[i], tt[i]) for i in range(len(positions))]
    logger.info(f"hidden_vs_hf layers={n_layers} T={T} tt={t_tt:.2f}s hf={t_hf:.2f}s")
    for p, v in zip(positions, pccs):
        logger.info(f"  pos={p} pcc={v:.6f}")
    logger.info(f"  min_pcc={min(pccs):.6f}")
    assert torch.isfinite(tt).all()
    bar = 0.99 if n_layers >= 32 else 0.97
    assert min(pccs) > bar, f"min pcc {min(pccs):.6f} < {bar}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("n_layers", LAYER_CASES)
@pytest.mark.parametrize("T", [300, 1500])
def test_readout_matches_upstream_logits(device, n_layers, T):
    import ttnn

    device.enable_program_cache()
    engine = KevEngine(device, args_cls=Qwen36ModelArgs, max_state_len=MAX_STATE_LEN, n_layers=n_layers, traced=False)
    assert type(engine.args) is Qwen36ModelArgs
    ids = fixed_ids(T, T)
    upstream = ttnn.to_torch(engine.model.prefill_masked_bucket(ids, engine.page_table, actual_len=T)).squeeze().float()
    row = engine.prefill_hidden(ids, [T - 1])[0]
    w = ttnn.to_torch(engine.model.lm_head_weight).float()
    mine = row @ w
    p = pcc(upstream, mine)
    logger.info(
        f"readout layers={n_layers} T={T} logit_pcc={p:.6f} argmax upstream={int(upstream.argmax())} mine={int(mine.argmax())}"
    )
    assert p > 0.999, f"logit pcc {p:.6f}"
    assert int(upstream.argmax()) in mine.topk(2).indices.tolist()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("n_layers", LAYER_CASES)
@pytest.mark.parametrize("S", [2048, 2175, 2200])
@pytest.mark.parametrize("Q", [50, 400])
def test_tail_matches_full_row(device, n_layers, S, Q):
    device.enable_program_cache()
    engine = KevEngine(device, args_cls=Qwen36ModelArgs, max_state_len=MAX_STATE_LEN, n_layers=n_layers)
    state = fixed_ids(S, S)
    q1 = fixed_ids(Q, Q)
    q2 = fixed_ids(Q + 1, Q)
    pos = sample_positions(Q, 4, first=0)
    t0 = time.perf_counter()
    handle = engine.prefill_state(state)
    t_state = time.perf_counter() - t0
    assert handle.S0 == (S // 128) * 128
    times = []
    outs = []
    for q in (q1, q2, q1):
        t0 = time.perf_counter()
        outs.append(engine.question_hidden(handle, q, pos))
        times.append(time.perf_counter() - t0)
    assert torch.equal(outs[0], outs[2]), "question 3 differs from question 1"
    full = engine.prefill_hidden(torch.cat([state, q1], dim=1), [S + p for p in pos])
    pccs = [pcc(full[i], outs[0][i]) for i in range(len(pos))]
    logger.info(
        f"tail layers={n_layers} S={S} S0={handle.S0} Q={Q} state={t_state:.2f}s "
        f"question={[f'{t:.2f}' for t in times]}s min_pcc={min(pccs):.6f} pccs={[f'{v:.6f}' for v in pccs]}"
    )
    assert min(pccs) > 0.999, f"min pcc {min(pccs):.6f}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_reference_records(device):
    rows_path = REFERENCE_DIR / "rows.json"
    hidden_path = REFERENCE_DIR / "hidden_fp32.pt"
    probs_path = REFERENCE_DIR / "probs_fp32.json"
    if not (rows_path.exists() and hidden_path.exists() and probs_path.exists()):
        pytest.skip(f"reference dumps missing under {REFERENCE_DIR}")
    head_mod = pytest.importorskip("models.autoports.jaredpalmer_kev_9b.tt.head")
    device.enable_program_cache()
    engine = KevEngine(device, max_state_len=MAX_STATE_LEN)
    rows = json.load(open(rows_path))["rows"]
    ref_hidden = torch.load(hidden_path)
    ref_records = json.load(open(probs_path))["records"]
    head = head_mod.PointerHead(os.environ["KEV_RUN"])
    assert isinstance(engine.args, KevModelArgs)
    min_pcc = 1.0
    agree = 0
    max_diff = 0.0
    diffs = []
    flips = []
    near_ties = []
    for row in rows:
        ids = torch.tensor(row["ids"], dtype=torch.long).unsqueeze(0)
        positions = list(row["opt_positions"]) + [row["decide_position"]]
        t0 = time.perf_counter()
        tt = engine.prefill_hidden(ids, positions)
        dt = time.perf_counter() - t0
        ref = ref_hidden[row["row_key"]].float()
        row_pccs = [pcc(ref[j], tt[j]) for j in range(len(positions))]
        min_pcc = min(min_pcc, min(row_pccs))
        probs = head.probs(tt[-1], tt[:-1]).float()
        rq = ref_records[str(row["record"])]["questions"][row["qid"]]["probabilities"]
        rp = torch.tensor([rq[k] for k in row["keys"]], dtype=torch.float32)
        agree += int(probs.argmax() == rp.argmax())
        top2 = rp.topk(2).values
        margin = round(float(top2[0] - top2[1]), 4)
        if probs.argmax() != rp.argmax():
            (flips if margin >= FLIP_MARGIN else near_ties).append((row["row_key"], margin))
        diff = (probs - rp).abs().max().item()
        max_diff = max(max_diff, diff)
        diffs.append(diff)
        logger.info(
            f"row {row['row_key']} {row['type']} T={row['row_tokens']} n={len(positions)} {dt:.2f}s "
            f"min_pcc={min(row_pccs):.6f} argmax tt={int(probs.argmax())} ref={int(rp.argmax())} max_dp={diff:.6f}"
        )
    mean_diff = sum(diffs) / len(diffs)
    logger.info(
        f"reference_records rows={len(rows)} min_pcc={min_pcc:.6f} argmax_agree={agree}/{len(rows)} "
        f"max_abs_prob_diff={max_diff:.6f} mean_abs_prob_diff={mean_diff:.6f} traced={engine.traced} "
        f"flips_margin_ge_{FLIP_MARGIN}={flips} near_tie_flips={near_ties}"
    )
    assert min_pcc > 0.97
    assert not flips, f"argmax flips on rows with an fp32 top-2 margin >= {FLIP_MARGIN}: {flips}"
    assert max_diff <= MAX_ABS_DP, f"max |dp| {max_diff:.6f} > {MAX_ABS_DP}"


def compare_modes(key, mode, outs, bar=0.999):
    _results.setdefault(key, {})[mode] = outs
    if mode == "eager":
        return
    if "eager" not in _results[key]:
        logger.info(f"{mode} {key}: no eager result in this process, cross-mode comparison not run")
        return
    pccs = [pcc(e, t) for e, t in zip(_results[key]["eager"], outs)]
    logger.info(f"{mode} vs eager {key}: min_pcc={min(pccs):.6f} pccs={[f'{v:.6f}' for v in pccs]}")
    assert min(pccs) > bar, f"{mode} vs eager min pcc {min(pccs):.6f}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MODE_CASES)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("n_layers", LAYER_CASES)
def test_slots_interleaved(device, n_layers, mode):
    device.enable_program_cache()
    engine = KevEngine(device, max_state_len=MAX_STATE_LEN, n_layers=n_layers, snapshot_slots=4, **mode_kwargs(mode))
    assert isinstance(engine.args, KevModelArgs) and engine.snapshot_slots == 4
    SA, SB, Q = 2450, 2200, 60
    A, B, q = fixed_ids(SA, SA), fixed_ids(SB, SB), fixed_ids(Q, Q)
    pos = sample_positions(Q, 4, first=0)
    hA = engine.prefill_state(A, slot=0)
    hB = engine.prefill_state(B, slot=1)
    assert (hA.S0, hB.S0) == (2432, 2176)
    a1 = engine.question_hidden(hA, q, pos)
    b1 = engine.question_hidden(hB, q, pos)
    a2 = engine.question_hidden(hA, q, pos)
    assert torch.equal(a1, a2), "question on A changed after a question on B"
    refA = engine.prefill_hidden(torch.cat([A, q], dim=1), [SA + p for p in pos], slot=2)
    refB = engine.prefill_hidden(torch.cat([B, q], dim=1), [SB + p for p in pos], slot=3)
    a3 = engine.question_hidden(hA, q, pos)
    assert torch.equal(a1, a3), "question on A changed after reference rows ran in other slots"
    pa = [pcc(refA[i], a1[i]) for i in range(len(pos))]
    pb = [pcc(refB[i], b1[i]) for i in range(len(pos))]
    logger.info(f"slots layers={n_layers} mode={mode} A={pa} B={pb}")
    assert min(pa) > 0.999 and min(pb) > 0.999
    compare_modes((n_layers, "slots"), mode, [a1, b1])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MODE_CASES)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("n_layers", LAYER_CASES)
def test_tail_buckets(device, n_layers, mode):
    device.enable_program_cache()
    engine = KevEngine(device, max_state_len=MAX_STATE_LEN, n_layers=n_layers, **mode_kwargs(mode))
    S = 2048
    handle = engine.prefill_state(fixed_ids(S, S), slot=0)
    outs = []
    for Q in (50, 200, 450, 1000, 2000):
        q = fixed_ids(Q, Q)
        pos = sample_positions(Q, 4, first=0)
        times = []
        rep = []
        for _ in range(3):
            t0 = time.perf_counter()
            rep.append(engine.question_hidden(handle, q, pos))
            times.append(time.perf_counter() - t0)
        assert torch.equal(rep[0], rep[1]) and torch.equal(rep[1], rep[2])
        assert torch.isfinite(rep[0]).all()
        logger.info(f"bucket layers={n_layers} mode={mode} Q={Q} times={[f'{t * 1000:.1f}' for t in times]} ms")
        outs.append(rep[0])
    compare_modes((n_layers, "buckets"), mode, outs)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_long_state(device):
    import ttnn

    device.enable_program_cache()
    engine = KevEngine(device, max_state_len=65536)
    assert engine.snapshot_slots == 8

    def dram_free_gib():
        view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return int(view.total_bytes_free_per_bank) * int(view.num_banks) / 2**30

    def timed(fn):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        return out, time.perf_counter() - t0

    Q = 40
    q1, q2 = fixed_ids(Q, Q), fixed_ids(Q + 1, Q)
    pos = sample_positions(Q, 3, first=0)
    S = 16384
    state = fixed_ids(S, S)
    handle, t_state = timed(lambda: engine.prefill_state(state, slot=0))
    out = engine.question_hidden(handle, q1, pos)
    ref = engine.prefill_hidden(torch.cat([state, q1], dim=1), [S + p for p in pos], slot=1)
    pccs = [pcc(ref[i], out[i]) for i in range(len(pos))]
    logger.info(f"long S={S} state={t_state:.2f}s pccs={[f'{v:.6f}' for v in pccs]}")
    assert min(pccs) > 0.999
    S = 65536
    state = fixed_ids(S, S)
    free_before = dram_free_gib()
    handle, t_state = timed(lambda: engine.prefill_state(state, slot=7))
    out1, t_q1 = timed(lambda: engine.question_hidden(handle, q1, pos))
    out2, t_q2 = timed(lambda: engine.question_hidden(handle, q2, pos))
    free_after = dram_free_gib()
    logger.info(
        f"long S={S} slot=7 slots={engine.snapshot_slots} state={t_state:.2f}s "
        f"questions={t_q1 * 1000:.1f}/{t_q2 * 1000:.1f}ms dram_free_before={free_before:.2f}GiB after={free_after:.2f}GiB"
    )
    assert torch.isfinite(out1).all() and torch.isfinite(out2).all()
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    engine._forward_body(engine.chunk_size)
    trace = ttnn.graph.end_graph_capture()
    live = peak = 0
    for node in trace:
        params = node["params"]
        if node["node_type"] == "buffer_allocate" and params["type"] == "DRAM":
            live += int(params["size"])
            peak = max(peak, live)
        elif node["node_type"] == "buffer_deallocate" and params["type"] == "DRAM":
            live -= int(params["size"])
    logger.info(f"peak transient DRAM of the {engine.chunk_size}-token forward body: {peak / 2**30:.3f} GiB ({peak} B)")
