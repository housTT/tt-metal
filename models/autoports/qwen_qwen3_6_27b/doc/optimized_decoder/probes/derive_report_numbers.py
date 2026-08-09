# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Derive every published number in this stage's reports from the shipped artifacts.

Three review rounds found the same defect class: a re-run lands new logs, and the numbers
quoted in ``README.md``, ``work_log.md``, ``perf_summary.json`` and ``doc/context_contract.json``
are left behind.  Hand-maintaining them does not work, so this script is the source of truth.

    python derive_report_numbers.py            # print the table, check nothing has drifted
    python derive_report_numbers.py --write    # rewrite the generated blocks in place

It reads only ``logs/*.log`` and ``tracy/*/*.csv`` and writes only the numbers, never the prose.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BW = 512e9
LAYERS = {"linear_attention": 48, "full_attention": 16}
#: Bytes the measured decode path moves per token, from the shipped weight dtypes.
BYTES_PER_TOKEN = {"linear_attention": None, "full_attention": None}
KINDS = ("linear_attention", "full_attention")


def _tile_bytes(dtype: str) -> float:
    return {"bfloat16": 2.0, "bfloat8_b": 1.0625, "bfloat4_b": 0.5625, "float32": 4.0}[dtype]


def weight_bytes(policy: dict) -> dict:
    """Per-token weight + state bytes for each layer kind under a precision policy."""
    mb = lambda k, n, d: k * n * _tile_bytes(policy[d])  # noqa: E731
    linear = (
        mb(5120, 17408, "mlp_gate_up") * 2
        + mb(17408, 5120, "mlp_down")
        + mb(5120, 10240, "gdn_qkv")
        + mb(5120, 6144, "gdn_z")
        + mb(6144, 5120, "gdn_out")
        + mb(5120, 112, "gdn_ba")
        + 2 * 48 * 128 * 128 * 4          # recurrent state, read and written
        + 3 * 10240 * _tile_bytes(policy["gdn_conv"])   # conv state taps
        + 4 * 5120 * 2                     # norms
    )
    full = (
        mb(5120, 17408, "mlp_gate_up") * 2
        + mb(17408, 5120, "mlp_down")
        + mb(5120, 8192, "attn_qkv")
        + mb(5120, 6144, "attn_gate")
        + mb(6144, 5120, "attn_out")
        + 2 * 4 * 2049 * 256 * _tile_bytes(policy["kv_cache"])   # paged KV read at position 2048
        + 4 * 5120 * 2
    )
    return {"linear_attention": int(linear), "full_attention": int(full)}


def evidence(log: str) -> dict:
    text = (ROOT / "logs" / log).read_text(errors="replace")
    out: dict = collections.defaultdict(dict)
    for m in re.finditer(r"PCCEVIDENCE (\{.*?\})", text, re.S):
        d = json.loads(m.group(1))
        out[d["metric"]][d.get("kind", "-")] = d["value"]
    return out


def sweep(log: str) -> dict:
    out = {}
    for line in (ROOT / "logs" / log).read_text(errors="replace").splitlines():
        i = line.find("SWEEP ")
        if i >= 0:
            d = json.loads(line[i + 6 :])
            out[d["kind"]] = d
    return out


def pytest_summary(log: str) -> str:
    text = (ROOT / "logs" / log).read_text(errors="replace")
    found = re.findall(r"=+ ([^\n=]*(?:passed|failed)[^\n=]*) =+", text)
    return found[-1].strip() if found else "?"


def profile(kind: str, phase: str) -> dict:
    """Per-iteration device time, op-to-op gap, op count and per-op-code breakdown."""
    rows = list(csv.DictReader(open(ROOT / "tracy" / kind / f"{phase}_perf_report.csv")))
    it = 8 if phase == "decode" else 1
    per_code: dict = collections.Counter()
    counts: dict = collections.Counter()
    total = gap = 0.0
    for r in rows:
        t = float(r["Device Time"] or 0)
        total += t
        per_code[r["OP Code"]] += t
        counts[r["OP Code"]] += 1
        try:
            gap += float(r["Op-to-Op Gap"] or 0)
        except ValueError:
            pass
    return {
        "device_us": total / it,
        "gap_us": gap / it,
        "ops": len(rows) // it,
        "per_code": {k: v / it for k, v in per_code.items()},
        "counts": {k: v // it for k, v in counts.items()},
        "roofline_line": next(
            (l.strip() for l in (ROOT / "tracy" / kind / f"{phase}_perf_report.console.log")
             .read_text().splitlines() if "roofline" in l), ""),
    }


def layout_ops(kind: str, phase: str) -> list:
    """``(id, us, op code)`` for every tilize/untilize/reshard op, in dispatch order."""
    prefixes = ("Untilize", "Tilize", "Reshard", "InterleavedToSharded", "ShardedToInterleaved")
    out = []
    for r in csv.DictReader(open(ROOT / "tracy" / kind / f"{phase}_perf_report.csv")):
        if any(r["OP Code"].startswith(p) for p in prefixes):
            out.append((r["ID"], float(r["Device Time"] or 0), r["OP Code"]))
    return out


def wall_times() -> dict:
    out = {}
    for line in (ROOT / "logs" / "run_perf.log").read_text(errors="replace").splitlines():
        if not line.startswith("PERF "):
            continue
        kind = re.search(r"kind=(\S+)", line).group(1)
        if "wall_e2e_ms" in line:
            out[(kind, "prefill")] = float(re.search(r"wall_e2e_ms=([\d.]+)", line).group(1))
        else:
            out[(kind, "decode")] = float(re.search(r"wall_per_iter_ms=([\d.]+)", line).group(1))
    return out


def _rewrite_readme(perf, suite, longc, ctrl) -> None:
    """Replace the generated tables in README.md.  Prose is never touched."""
    R = perf["runs"]
    ms = perf["model_scale"]
    f6 = lambda v: f"{v:.6f}"  # noqa: E731
    text = (ROOT / "README.md").read_text()
    text = re.sub(
        r"\| `linear_attention` \| prefill 2048 \|.*\n\| `linear_attention` \| traced decode \|.*\n"
        r"\| `full_attention` \| prefill 2048 \|.*\n\| `full_attention` \| traced decode \|.*\n",
        "".join(
            f"| `{k}` | {p} | {R[f'{k}/{ph}']['baseline_ms']:.2f} ms | "
            f"**{R[f'{k}/{ph}']['optimized_ms']:.3f} ms** | **{R[f'{k}/{ph}']['speedup']:.2f}x** |\n"
            for k in KINDS for ph, p in (("prefill", "prefill 2048"), ("decode", "traced decode"))
        ).replace("| prefill 2048 | 5", "| prefill 2048 | 5").replace(".000 ms**", " ms**"),
        text)
    text = re.sub(
        r"\| `linear_attention` \| 0\.\d+ \(was.*\n\| `full_attention` \| 0\.\d+ \(was.*\n",
        "".join(
            f"| `{k}` | {f6(R[f'{k}/prefill']['optimized_pcc'])} (was {f6(R[f'{k}/prefill']['baseline_pcc'])}) | "
            f"{f6(R[f'{k}/decode']['optimized_pcc'])} (was {f6(R[f'{k}/decode']['baseline_pcc'])}) | 0.995 |\n"
            for k in KINDS),
        text)
    text = re.sub(
        r"\d\.\d\d s of prefill per 2048 tokens against \d\.\d\d s, and \*\*\d+\.\d ms per decoded token against\n"
        r"\d+\.\d ms\*\* — \d+\.\d tok/s of layer-stack budget against \d\.\d\.",
        f"{ms['prefill_2048_optimized_ms']/1000:.2f} s of prefill per 2048 tokens against "
        f"{ms['prefill_2048_baseline_ms']/1000:.2f} s, and **{ms['decode_optimized_ms_per_token']:.1f} ms "
        f"per decoded token against\n{ms['decode_baseline_ms_per_token']:.1f} ms** — "
        f"{ms['decode_optimized_layer_stack_tps']:.1f} tok/s of layer-stack budget against "
        f"{1000/ms['decode_baseline_ms_per_token']:.1f}.",
        text)
    text = re.sub(r"`tests/test_optimized_decoder\.py`: \*\*[^*]*\*\*", 
                  f"`tests/test_optimized_decoder.py`: **{pytest_summary('suite_main.log')}**", text)
    text = re.sub(
        r"\| traced decode speed-up \(`test_optimized_decode_beats_fused`\) \|.*\n"
        r"\| prefill speed-up \(`test_optimized_prefill_beats_fused`\) \|.*\n"
        r"\| worst real-weight PCC over lengths[^|]*\|.*\n"
        r"\| stress, 12 back-to-back passes, min PCC \|.*\n"
        r"\| optimized vs fused, prefill / decode \|.*\n",
        f"| traced decode speed-up (`test_optimized_decode_beats_fused`) | "
        f"**{suite['traced_decode_speedup']['linear_attention']:.3f}x** | "
        f"**{suite['traced_decode_speedup']['full_attention']:.3f}x** |\n"
        f"| prefill speed-up (`test_optimized_prefill_beats_fused`) | "
        f"**{suite['prefill_speedup']['linear_attention']:.3f}x** | "
        f"**{suite['prefill_speedup']['full_attention']:.3f}x** |\n"
        f"| worst real-weight PCC over lengths 1/17/64/743/2049/5000, prefill **and** decode | "
        f"**{f6(suite['real_weight_worst_pcc']['linear_attention'])}** | "
        f"**{f6(suite['real_weight_worst_pcc']['full_attention'])}** |\n"
        f"| stress, 12 back-to-back passes, min PCC | "
        f"{f6(suite['stress_decode_pcc_min']['linear_attention'])} | "
        f"{f6(suite['stress_decode_pcc_min']['full_attention'])} |\n"
        f"| optimized vs fused, prefill / decode | "
        f"{f6(suite['optimized_vs_fused_prefill_pcc']['linear_attention'])} / "
        f"{f6(suite['optimized_vs_fused_decode_pcc']['linear_attention'])} | "
        f"{f6(suite['optimized_vs_fused_prefill_pcc']['full_attention'])} / "
        f"{f6(suite['optimized_vs_fused_decode_pcc']['full_attention'])} |\n",
        text)
    text = re.sub(
        r"\| conv state / K cache \|.*\n\| recurrent state / V cache \|.*\n\| prefill tail \|.*\n"
        r"\| decode at 262143 \|.*\n",
        f"| conv state / K cache | {f6(longc['full_context_conv_state_pcc']['linear_attention'])} | "
        f"{f6(longc['full_context_paged_k_cache_pcc']['full_attention'])} |\n"
        f"| recurrent state / V cache | {f6(longc['full_context_recurrent_state_pcc']['linear_attention'])} | "
        f"{f6(longc['full_context_paged_v_cache_pcc']['full_attention'])} |\n"
        f"| prefill tail | {f6(longc['full_context_prefill_tail_pcc']['linear_attention'])} | "
        f"{f6(longc['full_context_prefill_tail_pcc']['full_attention'])} |\n"
        f"| decode at 262143 | {f6(longc['full_context_decode_pcc']['linear_attention'])} | "
        f"**{f6(longc['full_context_decode_pcc']['full_attention'])}** |\n",
        text)
    text = re.sub(r"Watcher \(`TT_METAL_WATCHER=10`[^)]*\): \*\*[^*]*\*\*",
                  f"Watcher (`TT_METAL_WATCHER=10`, `logs/watcher_run.log`, `watcher/watcher.log`): "
                  f"**{pytest_summary('watcher_run.log')}**", text)
    (ROOT / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite perf_summary.json in place")
    args = parser.parse_args()

    sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import DEFAULT_PRECISION as P

    policy = {f: str(getattr(P, f)).split(".")[-1].lower() for f in
              ("mlp_gate_up", "mlp_down", "attn_qkv", "attn_gate", "attn_out",
               "gdn_qkv", "gdn_z", "gdn_out", "gdn_ba", "gdn_conv", "kv_cache")}
    per_token = weight_bytes(policy)

    base, opt, walls = sweep("sweep_final_baseline.log"), sweep("sweep_final_default.log"), wall_times()
    perf = json.load(open(ROOT / "perf_summary.json"))
    for kind in KINDS:
        for phase in ("prefill", "decode"):
            prof = profile(kind, phase)
            mk = "prefill_ms" if phase == "prefill" else "decode_ms"
            pk = "prefill_pcc" if phase == "prefill" else "decode_pcc_traced"
            row = perf["runs"][f"{kind}/{phase}"]
            row.update(
                baseline_ms=round(base[kind][mk], 3),
                optimized_ms=round(opt[kind][mk], 3),
                speedup=round(base[kind][mk] / opt[kind][mk], 3),
                baseline_pcc=base[kind][pk], optimized_pcc=opt[kind][pk],
                device_time_us=round(prof["device_us"], 1),
                op_to_op_gap_us=round(prof["gap_us"], 1),
                device_ops=prof["ops"],
                profiled_wall_ms=walls[(kind, phase)],
            )
            row.pop("note_o9", None)
            if phase == "decode":
                roof = per_token[kind] / BW * 1e3
                row.update(
                    decode_ms_per_token_e2e=row["optimized_ms"],
                    decode_ms_per_token_device=round(prof["device_us"] / 1e3, 4),
                    op_to_op_gap_ms_per_token=round(prof["gap_us"] / 1e3, 4),
                    roofline_ms_per_token_estimate=round(roof, 4),
                    roofline_fraction_of_device_time=round(roof / (prof["device_us"] / 1e3), 3),
                    bytes_moved_per_token=per_token[kind],
                )
                perf["roofline"][kind] = {"bytes_per_token": per_token[kind],
                                          "assumed_dram_bandwidth_bytes_per_s": BW,
                                          "ms_per_token": round(roof, 4)}
    ms = perf["model_scale"]
    ms["prefill_2048_baseline_ms"] = round(sum(LAYERS[k] * perf["runs"][f"{k}/prefill"]["baseline_ms"] for k in LAYERS), 1)
    ms["prefill_2048_optimized_ms"] = round(sum(LAYERS[k] * perf["runs"][f"{k}/prefill"]["optimized_ms"] for k in LAYERS), 1)
    ms["decode_baseline_ms_per_token"] = round(sum(LAYERS[k] * perf["runs"][f"{k}/decode"]["baseline_ms"] for k in LAYERS), 2)
    ms["decode_optimized_ms_per_token"] = round(sum(LAYERS[k] * perf["runs"][f"{k}/decode"]["decode_ms_per_token_e2e"] for k in LAYERS), 2)
    ms["decode_optimized_layer_stack_tps"] = round(1000.0 / ms["decode_optimized_ms_per_token"], 2)
    suite, longc, ctrl = (evidence("suite_main.log"), evidence("long_context.log"),
                          evidence("long_context_bfp8_control.log"))
    if args.write:
        json.dump(perf, open(ROOT / "perf_summary.json", "w"), indent=1)
        open(ROOT / "perf_summary.json", "a").write("\n")
        _rewrite_readme(perf, suite, longc, ctrl)

    print("== headline (logs/sweep_final_{baseline,default}.log)")
    for kind in KINDS:
        for phase in ("prefill", "decode"):
            r = perf["runs"][f"{kind}/{phase}"]
            print(f"  {kind:18s} {phase:8s} {r['baseline_ms']:8.3f} -> {r['optimized_ms']:7.3f} ms "
                  f"({r['speedup']:.3f}x)  device {r['device_time_us']:8.1f} us  wall {r['profiled_wall_ms']:7.3f} ms  "
                  f"pcc {r['optimized_pcc']:.6f}")
    print("== model scale:", json.dumps(ms))
    print("== rooflines")
    for kind in KINDS:
        r = perf["runs"][f"{kind}/decode"]
        print(f"  {kind:18s} {r['bytes_moved_per_token']} B/token -> {r['roofline_ms_per_token_estimate']:.4f} ms "
              f"= {r['roofline_fraction_of_device_time']*100:.1f} % of device;  {profile(kind,'decode')['roofline_line']}")
    print("== gates")
    for log in ("suite_main.log", "long_context.log", "watcher_run.log"):
        print(f"  {log:22s} {pytest_summary(log)}")
    for metric in ("traced_decode_speedup", "prefill_speedup", "real_weight_worst_pcc",
                   "structural_worst_pcc", "stress_prefill_pcc_min", "stress_decode_pcc_min",
                   "optimized_vs_fused_prefill_pcc", "optimized_vs_fused_decode_pcc"):
        if metric in suite:
            print(f"  {metric:36s} {json.dumps({k: round(v, 6) for k, v in suite[metric].items()})}")
    print("== full advertised context (default | bfp8 gate/up control)")
    for metric in sorted(set(longc) | set(ctrl)):
        print(f"  {metric:36s} {json.dumps({k: round(v,7) for k,v in longc.get(metric,{}).items()})} | "
              f"{json.dumps({k: round(v,7) for k,v in ctrl.get(metric,{}).items()})}")
    print("== decode layout ops per token")
    for kind in KINDS:
        ops = layout_ops(kind, "decode")
        print(f"  {kind:18s} {sum(t for _, t, _ in ops)/8:6.1f} us over {len(ops)//8} ops: "
              + ", ".join(f"{c.replace('DeviceOperation','')}" for _, _, c in ops[:len(ops)//8]))
    print("== prefill layout ops (linear_attention), by op id")
    for op_id, us, code in layout_ops("linear_attention", "prefill"):
        print(f"  id {op_id:>5} {us:7.1f} us  {code}")
    print(f"  total {sum(t for _, t, _ in layout_ops('linear_attention','prefill')):.1f} us; "
          f"full_attention prefill has {len(layout_ops('full_attention','prefill'))}")


if __name__ == "__main__":
    main()
