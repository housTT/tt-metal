import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
os.environ.setdefault("HF_MODEL", SNAPSHOT)

from models.autoports.cloudflare_clef.scripts import reference_rows
from models.autoports.cloudflare_clef.scripts.clef_paths import read_jsonl, write_jsonl
from models.autoports.cloudflare_clef.tt import encode as clef_encode
from models.autoports.cloudflare_clef.tt.engine import BUCKETS, ClefEngine, tp2_mesh

RECORDS = Path("/home/hous/dev/clef/reports/reference/records_text.jsonl")
REFERENCE = Path("/home/hous/dev/clef/reports/reference/ref_text_bf16.jsonl")
README_EXAMPLES = Path("/home/hous/dev/clef/reports/reference/readme_examples.json")
PARITY_COMPARE = Path(__file__).resolve().parent.parent / "scripts" / "parity_compare.py"
REPORT_DIR = Path("/home/hous/dev/clef/reports")
MAX_STATE_LEN = 16384
LAYER_CASES = [pytest.param(4, id="l4"), pytest.param(64, id="l64", marks=pytest.mark.slow)]
LENGTHS = [300, 1500, 2300, 8192]
GATED_LENGTHS = {300}
FP32_CONTROL_VIEWS = ("tt_vs_fp32", "bf16_vs_fp32", "tt_vs_bf16")
FP32_CONTROL_KEYS = ("min", "p01", "mean", "rows", "rows_below_0_99", "frac_below_0_99", "rows_below_0_80")
LAYER_PROBE_LENGTHS = [pytest.param(300, id="T300"), pytest.param(8192, id="T8192")]
GATE_A_BAND = 0.0005
GATE_A_LAYER_FLOOR = 0.999
FLIP_MARGIN = 0.05
MAX_DP = 0.10
RESULTS = {"hidden_vs_hf": {}, "tail": {}, "records": {}, "timing": {}}
_engine = {}
_hf = {}


def report_path(n_layers):
    tag = os.environ.get("CLEF_REPORT_TAG", "")
    return REPORT_DIR / f"stage1_engine_l{n_layers}{tag}.json"


def write_report(n_layers):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path(n_layers).write_text(json.dumps(RESULTS, indent=2, default=str))


def fp32_control_path(T):
    return REPORT_DIR / f"stage1r_hidden_fp32_control_T{T}.json"


def fp32_control(T):
    path = fp32_control_path(T)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    out = dict(path=str(path), dump=data.get("dump"), n_layers=data.get("n_layers"))
    for view in FP32_CONTROL_VIEWS:
        out[view] = {k: data[view][k] for k in FP32_CONTROL_KEYS if k in data[view]}
    return out


def layer_probe_path(T):
    return REPORT_DIR / f"stage1_layer_pcc_l64_T{T}_gatefp32.json"


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def sample_positions(length, n, first=1):
    pos = torch.linspace(first, length - 1, n).round().long().tolist()
    return sorted(set(pos + [length - 1]))


@pytest.fixture(scope="module")
def submesh():
    with tp2_mesh(os.environ.get("CLEF_PARENT", "1x4")) as sub:
        yield sub


def engine_for(submesh, n_layers):
    if n_layers in _engine:
        return _engine[n_layers]
    if "error" in _engine:
        pytest.fail(f"engine build failed earlier in this process: {_engine['error']}")
    assert not _engine, "one engine per pytest process: run -k l4 and -k l64 separately"
    try:
        engine = ClefEngine(submesh, max_state_len=MAX_STATE_LEN, n_layers=n_layers, snapshot_slots=4)
    except Exception as error:
        _engine["error"] = repr(error)
        raise
    _engine[n_layers] = engine
    RESULTS["engine"] = dict(
        n_layers=engine.args.n_layers,
        layer_types=list(engine.args.attention_type_list),
        precision=engine.precision,
        cache_reload=engine.cache_reload,
        cache_dir=str(engine.cache_dir),
        prefill_only_mlp=engine.prefill_only_mlp,
        timings=engine.timings,
        dram_free_after_weights_bytes=engine.dram_free_after_weights,
        dram_free_after_slots_bytes=engine.dram_free_after_slots,
        snapshot_slots=engine.snapshot_slots,
        slot_bytes=engine.slot_bytes,
        blocks_per_slot=engine.blocks_per_slot,
        max_len=engine.max_len,
        ccl_topology=str(engine.args.ccl_topology()),
        ccl_num_links=engine.model.tt_ccl.get_num_links(),
        submesh_chips=list(submesh.get_device_ids()),
    )
    logger.info(f"engine: {RESULTS['engine']}")
    write_report(n_layers)
    return engine


@pytest.fixture(scope="module")
def records():
    return read_jsonl(RECORDS)


@pytest.fixture(scope="module")
def tokenizer():
    return clef_encode.load_tokenizer(SNAPSHOT)


def request_ids(tokenizer, records, T):
    short = list(clef_encode.encode(tokenizer, records[0]).input_ids)
    if T <= len(short):
        return torch.tensor([short[:T]], dtype=torch.long)
    long_record = dict(records[0])
    long_record["state"] = (" ".join(str(r["state"]) for r in records) + " ") * (1 + T // 2000)
    ids = list(clef_encode.encode(tokenizer, long_record, max_length=T).input_ids)
    assert len(ids) == T, (T, len(ids))
    return torch.tensor([ids], dtype=torch.long)


def hf_model(n_layers):
    if n_layers in _hf:
        return _hf[n_layers]
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(SNAPSHOT)
    config.text_config.num_hidden_layers = n_layers
    config.text_config.layer_types = config.text_config.layer_types[:n_layers]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(SNAPSHOT, config=config, dtype=torch.bfloat16).eval()
    _hf.clear()
    _hf[n_layers] = model
    RESULTS["hf"] = dict(n_layers=n_layers, load_seconds=round(time.perf_counter() - t0, 1), dtype="bfloat16")
    return model


def hf_hidden(ids, n_layers):
    model = hf_model(n_layers)
    with torch.no_grad():
        return model.model.language_model(input_ids=ids, use_cache=False).last_hidden_state[0].float()


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("n_layers", LAYER_CASES)
@pytest.mark.parametrize("T", LENGTHS)
def test_hidden_vs_hf(submesh, tokenizer, records, n_layers, T):
    """Hidden state of prefill_hidden against HF last_hidden_state, two views per length.

    Amended stage 1 gate (/home/hous/.claude/plans/expressive-orbiting-tome.md, 2026 Oct 05): at 64
    layers the sampled-position bar (0.99 over 8 sampled rows plus the last) is asserted at T=300
    only (gate (b)). At T in {1500, 2300, 8192} both views (sampled and all rows) are recorded
    next to the HF bf16 versus fp32 control of the same rows when
    /home/hous/dev/clef/reports/stage1r_hidden_fp32_control_T{T}.json exists, and the test asserts
    only that the report JSON holds the row (gate (c)): HF bf16 itself fails a whole-row 0.99 bar
    against fp32 at those lengths. At 4 layers the 0.97 smoke bar is asserted at every length.
    """
    engine = engine_for(submesh, n_layers)
    ids = request_ids(tokenizer, records, T)
    positions = sample_positions(T, 8)
    times = []
    for _ in range(2):
        t0 = time.perf_counter()
        tt = engine.prefill_hidden(ids, slot=0)
        times.append(round(time.perf_counter() - t0, 3))
    assert tt.shape == (T, engine.args.dim)
    assert torch.isfinite(tt).all(), "non-finite TT hidden"
    t0 = time.perf_counter()
    ref = hf_hidden(ids, n_layers)
    t_hf = round(time.perf_counter() - t0, 1)
    assert ref.shape == tt.shape
    pccs = {p: round(pcc(ref[p], tt[p]), 6) for p in positions}
    all_rows = torch.nn.functional.cosine_similarity(
        ref - ref.mean(1, keepdim=True), tt - tt.mean(1, keepdim=True), dim=1
    )
    bar = 0.99 if n_layers >= 64 else 0.97
    gated = n_layers < 64 or T in GATED_LENGTHS
    control = fp32_control(T) if n_layers >= 64 else None
    worst = all_rows.argsort()[:8].tolist()
    toks = ids[0].tolist()
    row = dict(
        T=T,
        positions=positions,
        pcc=pccs,
        min_pcc=min(pccs.values()),
        all_rows_min_pcc=round(float(all_rows.min()), 6),
        all_rows_mean_pcc=round(float(all_rows.mean()), 6),
        all_rows_p01_pcc=round(float(torch.quantile(all_rows, 0.01)), 6),
        all_rows_below_bar=int((all_rows < bar).sum()),
        all_rows_below_bar_fraction=round(float((all_rows < bar).float().mean()), 6),
        all_rows_below_0_80=int((all_rows < 0.80).sum()),
        sampled_below_bar=int(sum(v < bar for v in pccs.values())),
        worst_rows=[dict(pos=p, pcc=round(float(all_rows[p]), 6), token=tokenizer.decode([toks[p]])) for p in worst],
        tt_seconds_first_call=times[0],
        tt_seconds_second_call=times[1],
        hf_cpu_seconds=t_hf,
        bar=bar,
        gated=gated,
        hf_fp32_control=control,
    )
    RESULTS["hidden_vs_hf"][f"T{T}"] = row
    write_report(n_layers)
    dump = os.environ.get("CLEF_DUMP_HIDDEN")
    if dump:
        torch.save({"T": T, "ids": ids, "tt": tt, "ref": ref, "positions": positions}, f"{dump}_T{T}.pt")
    logger.info(f"hidden_vs_hf l{n_layers} T={T}: {row}")
    logger.info(
        f"hidden_vs_hf l{n_layers} T={T} gate views: sampled min {row['min_pcc']} ({row['sampled_below_bar']} of "
        f"{len(pccs)} sampled rows below {bar}); all rows min {row['all_rows_min_pcc']} mean {row['all_rows_mean_pcc']} "
        f"p01 {row['all_rows_p01_pcc']} ({row['all_rows_below_bar']} of {T} rows below {bar}, "
        f"{row['all_rows_below_0_80']} below 0.80)"
    )
    if gated:
        assert row["min_pcc"] >= bar, f"min pcc {row['min_pcc']} < {bar} at T={T}"
        return
    if control:
        logger.info(
            f"hidden_vs_hf l{n_layers} T={T} HF fp32 control ({control['path']}): rows below 0.99 vs fp32 "
            f"TT {control['tt_vs_fp32'].get('frac_below_0_99')} / HF bf16 {control['bf16_vs_fp32'].get('frac_below_0_99')}, "
            f"min {control['tt_vs_fp32'].get('min')} / {control['bf16_vs_fp32'].get('min')}"
        )
    else:
        logger.info(f"hidden_vs_hf l{n_layers} T={T}: no HF fp32 control at {fp32_control_path(T)}")
    logger.info(f"hidden_vs_hf l{n_layers} T={T}: recorded, not gated (amended gate (c))")
    written = json.loads(report_path(n_layers).read_text())["hidden_vs_hf"].get(f"T{T}")
    assert written and written["min_pcc"] == row["min_pcc"], f"{report_path(n_layers)} does not hold the T={T} row"


@pytest.mark.parametrize("T", LAYER_PROBE_LENGTHS)
def test_layer_floor_gate(T):
    """Gate (a) of the amended stage 1 gate. Host only: reads a probe report, opens no device.

    Reads the post-fix teacher-forced per-layer probe of the 64-layer engine
    (/home/hous/dev/clef/reports/stage1_layer_pcc_l64_T{300,8192}_gatefp32.json, written on the
    device by scripts/layer_pcc_probe.py --T 300,8192 --teacher 1 --tag _gatefp32 with
    QWEN36_GDN_GATE_FP32=1) and asserts, at T=300 and T=8192: the mean over the GDN layers of the
    per-layer teacher-forced delta PCC is within GATE_A_BAND (0.0005) of the same mean over the
    attention layers, and every layer except layer 63 has a per-layer mean above
    GATE_A_LAYER_FLOOR (0.999).

    Layer 63 is excluded because the probe compares the pre-norm residual output of the last
    decoder layer with HF hidden_states[-1], which is the post-final-norm tensor (it equals
    last_hidden_state), so the layer 63 delta PCC in every probe report (0.44 at T=300, 0.51 at
    T=8192, TT row norm about 3x to 6x the reference norm while layer 62 norms agree) is a
    bookkeeping artifact of the probe and not a device fault. The final_norm entry of the same
    report is the valid comparison for layer 63. scripts/layer_probe_summary.py applies the same
    exclusion and prints it.
    """
    path = layer_probe_path(T)
    assert path.exists(), f"probe report missing: {path}"
    report = json.loads(path.read_text())
    assert report["T"] == T and report["n_layers"] == 64, (report["T"], report["n_layers"])
    excluded = report["n_layers"] - 1
    by_layer = {r["layer"]: r for r in report["layers"]}
    rows = [r for r in report["layers"] if r["layer"] != excluded]
    assert len(rows) == excluded and all("teacher_delta_mean_pcc" in r for r in rows), "probe was not teacher-forced"
    kinds = ("gdn", "attn")
    means = {k: statistics.mean(r["teacher_delta_mean_pcc"] for r in rows if r["kind"] == k) for k in kinds}
    worst = {k: min((r for r in rows if r["kind"] == k), key=lambda r: r["teacher_delta_mean_pcc"]) for k in kinds}
    at_or_below_floor = [
        (r["layer"], r["kind"], r["teacher_delta_mean_pcc"])
        for r in rows
        if r["teacher_delta_mean_pcc"] <= GATE_A_LAYER_FLOOR
    ]
    logger.info(
        f"layer floor gate T={T} ({path}): gdn mean {means['gdn']:.6f} over {sum(r['kind'] == 'gdn' for r in rows)} layers "
        f"(worst layer {worst['gdn']['layer']} at {worst['gdn']['teacher_delta_mean_pcc']}), attn mean {means['attn']:.6f} over "
        f"{sum(r['kind'] == 'attn' for r in rows)} layers (worst layer {worst['attn']['layer']} at "
        f"{worst['attn']['teacher_delta_mean_pcc']}), gdn minus attn {means['gdn'] - means['attn']:+.6f} (band {GATE_A_BAND}), "
        f"layers at or below {GATE_A_LAYER_FLOOR}: {at_or_below_floor}; layer {excluded} excluded as the post-norm reference "
        f"artifact (delta PCC {by_layer[excluded]['teacher_delta_mean_pcc']}, tt norm {by_layer[excluded]['tt_norm_mean']} vs "
        f"ref norm {by_layer[excluded]['ref_norm_mean']}); final norm min {report['final_norm']['min_pcc']:.6f} mean "
        f"{report['final_norm']['mean_pcc']:.6f}"
    )
    assert (
        abs(means["gdn"] - means["attn"]) <= GATE_A_BAND
    ), f"T={T}: GDN mean {means['gdn']:.6f} is not within {GATE_A_BAND} of attention mean {means['attn']:.6f}"
    assert not at_or_below_floor, f"T={T}: layers at or below {GATE_A_LAYER_FLOOR}: {at_or_below_floor}"


def long_schema_record(records, n_questions=12):
    record = dict(next(r for r in records if r["id"] == "cfpb/3235868"))
    record["id"] = "long_schema"
    base = list(record["questions"].items())
    record["questions"] = {f"{qid}_{i}": dict(q) for i in range(n_questions) for qid, q in base}
    return record


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("n_layers", LAYER_CASES)
@pytest.mark.parametrize("case", ["cfpb_long_state", "long_schema"])
def test_tail_matches_full_row(submesh, tokenizer, records, n_layers, case):
    engine = engine_for(submesh, n_layers)
    if case == "cfpb_long_state":
        record = next(r for r in records if r["id"] == "cfpb/3235868")
        other = next(r for r in records if r["id"] == "readme_checkout")
    else:
        record = long_schema_record(records)
        other = records[1]
    encoded = clef_encode.encode(tokenizer, record)
    state_part, tail_part, _ = clef_encode.split_for_cache(encoded, tokenizer, record)
    other_encoded = clef_encode.encode(tokenizer, other)
    _, other_tail, _ = clef_encode.split_for_cache(other_encoded, tokenizer, other)
    state = torch.tensor([state_part], dtype=torch.long)
    tail = torch.tensor([tail_part], dtype=torch.long)
    tail2 = torch.tensor([other_tail], dtype=torch.long)
    t0 = time.perf_counter()
    handle = engine.prefill_state(state, slot=1)
    t_state = time.perf_counter() - t0
    assert handle.S0 == (state.shape[1] // 128) * 128
    outs, times = [], []
    for ids in (tail, tail2, tail):
        t0 = time.perf_counter()
        outs.append(engine.schema_hidden(handle, ids))
        times.append(round(time.perf_counter() - t0, 3))
    assert torch.equal(outs[0], outs[2]), "schema call 3 differs from call 1"
    assert outs[0].shape == (state.shape[1] - handle.S0 + tail.shape[1], engine.args.dim)
    full_ids = torch.cat([state, tail], dim=1)
    full = engine.prefill_hidden(full_ids, slot=2)
    tail_rows = torch.nn.functional.cosine_similarity(
        full[handle.S0 :] - full[handle.S0 :].mean(1, keepdim=True), outs[0] - outs[0].mean(1, keepdim=True), dim=1
    )
    prefix = engine.prefix_hidden[1]
    assert prefix.shape[0] == handle.S0
    if handle.S0:
        prefix_rows = torch.nn.functional.cosine_similarity(
            full[: handle.S0] - full[: handle.S0].mean(1, keepdim=True), prefix - prefix.mean(1, keepdim=True), dim=1
        )
        prefix_min = round(float(prefix_rows.min()), 6)
    else:
        prefix_min = None
    row = dict(
        record=record.get("id", "long_schema"),
        S=state.shape[1],
        S0=handle.S0,
        tail_tokens=tail.shape[1],
        tail_chunks=(tail.shape[1] + state.shape[1] - handle.S0 + 1023) // 1024,
        total_tokens=full_ids.shape[1],
        state_seconds=round(t_state, 3),
        schema_seconds=times,
        tail_rows_min_pcc=round(float(tail_rows.min()), 6),
        tail_rows_mean_pcc=round(float(tail_rows.mean()), 6),
        prefix_rows_min_pcc=prefix_min,
        repeat_bit_equal=True,
    )
    RESULTS["tail"][case] = row
    write_report(n_layers)
    logger.info(f"tail l{n_layers} {case}: {row}")
    assert row["tail_rows_min_pcc"] >= 0.999, f"tail rows min pcc {row['tail_rows_min_pcc']}"
    if prefix_min is not None:
        assert prefix_min >= 0.999, f"prefix rows min pcc {prefix_min}"


def parity(candidate, name, out_stem):
    cmd = [
        sys.executable,
        str(PARITY_COMPARE),
        "--reference",
        str(REFERENCE),
        "--candidate",
        str(candidate),
        "--max-dp-bar",
        str(MAX_DP),
        "--margin",
        str(FLIP_MARGIN),
        "--no-margin-flips",
        "--name",
        name,
        "--out-json",
        f"{out_stem}.json",
        "--out-md",
        f"{out_stem}.md",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"parity_compare rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-2000:]}")
    summary = json.loads(Path(f"{out_stem}.json").read_text())
    return proc.returncode, summary["overall"], " ".join(cmd)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("n_layers", LAYER_CASES)
def test_reference_records(submesh, records, n_layers):
    engine = engine_for(submesh, n_layers)
    rows = records if n_layers >= 64 else records[:3]
    outputs = {}
    for mode in ("full", "cached"):
        t0 = time.perf_counter()
        results = reference_rows.run_rows(engine, rows, mode=mode, slot=0 if mode == "full" else 3)
        path = REPORT_DIR / f"stage1_tt_text_l{n_layers}_{mode}.jsonl"
        write_jsonl(path, results)
        outputs[mode] = results
        errors = [r for r in results if "error" in r]
        assert not errors, errors
        RESULTS["records"][mode] = dict(
            path=str(path),
            records=len(results),
            seconds_total=round(time.perf_counter() - t0, 1),
            seconds_per_record={r["id"]: r["seconds"] for r in results},
            device_seconds_per_record={r["id"]: r["timing"]["device_s"] for r in results},
        )
    cross = []
    for a, b in zip(outputs["full"], outputs["cached"]):
        for qid, dist in a["probs"].items():
            cross.append(max(abs(dist[o] - b["probs"][qid][o]) for o in dist))
    RESULTS["records"]["cached_vs_full_max_dp"] = round(max(cross), 6)
    again = reference_rows.run_rows(engine, rows[-1:], mode="cached", slot=3)[0]
    assert "error" not in again, again
    before = outputs["cached"][-1]
    hit_dp = max(abs(dist[o] - again["probs"][qid][o]) for qid, dist in before["probs"].items() for o in dist)
    RESULTS["records"]["cache_hit_rerun"] = dict(
        record=again["id"],
        cache_hit=again["cache_hit"],
        first_cached_seconds=before["seconds"],
        first_cached_device_seconds=before["timing"]["device_s"],
        hit_seconds=again["seconds"],
        hit_device_seconds=again["timing"]["device_s"],
        max_dp_vs_first_cached=round(hit_dp, 6),
    )
    write_report(n_layers)
    logger.info(f"records l{n_layers}: cached vs full max |dp| = {max(cross):.6f}")
    logger.info(f"records l{n_layers}: cache hit rerun {RESULTS['records']['cache_hit_rerun']}")
    assert again["cache_hit"] is True, f"second cached run of {again['id']} did not hit the slot cache"
    assert hit_dp == 0.0, f"cache hit changed the probabilities by {hit_dp}"
    if n_layers < 64:
        assert max(cross) <= 0.05, f"cached vs full max |dp| {max(cross):.6f} at {n_layers} layers"
        return
    readme = json.loads(README_EXAMPLES.read_text())
    expected = {
        "readme_invoice": readme["usage_example"]["printed_probabilities"],
        "readme_checkout": readme["systemone_example"]["raw_probs"],
    }
    for result in outputs["full"]:
        if result["id"] in expected:
            logger.info(f"README example {result['id']}: TT {json.dumps(result['probs'])}")
            logger.info(f"README example {result['id']}: CPU {json.dumps(expected[result['id']])}")
            RESULTS["records"].setdefault("readme_examples", {})[result["id"]] = dict(
                tt=result["probs"], cpu=expected[result["id"]]
            )
    gates = {}
    for mode in ("full", "cached"):
        rc, overall, cmd = parity(
            REPORT_DIR / f"stage1_tt_text_l{n_layers}_{mode}.jsonl",
            f"TT l{n_layers} {mode} vs CPU bf16",
            REPORT_DIR / f"stage1_parity_l{n_layers}_{mode}",
        )
        gates[mode] = dict(rc=rc, overall=overall, command=cmd)
    RESULTS["records"]["parity"] = gates
    write_report(n_layers)
    assert gates["full"]["rc"] == 0, f"parity gate failed (full path): {gates['full']['overall']}"
    assert gates["cached"]["rc"] == 0, f"parity gate failed (cached path): {gates['cached']['overall']}"


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("n_layers", LAYER_CASES)
def test_timing(submesh, tokenizer, records, n_layers):
    import ttnn

    engine = engine_for(submesh, n_layers)
    buckets = {}
    for b in BUCKETS:
        ids = request_ids(tokenizer, records, b)
        times = []
        for _ in range(3):
            ttnn.synchronize_device(submesh)
            t0 = time.perf_counter()
            engine.prefill_hidden(ids, slot=0)
            times.append(round(time.perf_counter() - t0, 3))
        buckets[str(b)] = times
    S = 8192
    state = request_ids(tokenizer, records, S)
    ttnn.synchronize_device(submesh)
    t0 = time.perf_counter()
    handle = engine.prefill_state(state, slot=1)
    t_state = round(time.perf_counter() - t0, 3)
    tail = torch.tensor(
        [clef_encode.split_for_cache(clef_encode.encode(tokenizer, records[0]), tokenizer, records[0])[1]]
    )
    t0 = time.perf_counter()
    engine.schema_hidden(handle, tail)
    t_tail = round(time.perf_counter() - t0, 3)
    from models.autoports.cloudflare_clef.tt.engine import mesh_dram_free_bytes

    RESULTS["timing"] = dict(
        eager_prefill_hidden_seconds_per_bucket=buckets,
        state_prefill_8192_seconds=t_state,
        state_8192_S0=handle.S0,
        schema_after_8192_seconds=t_tail,
        schema_tail_tokens=int(tail.shape[1]),
        dram_free_now_bytes=mesh_dram_free_bytes(submesh),
    )
    write_report(n_layers)
    logger.info(f"timing l{n_layers}: {RESULTS['timing']}")
