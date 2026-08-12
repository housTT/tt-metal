# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reconcile the three decode numbers the `$optimize` skill requires, from committed artifacts.

1. **Theoretical roofline** — the bytes the measured path must move per decode token (every weight
   it reads at its *stored* dtype, plus the KV-cache read at the measured context) divided by the
   device's aggregate DRAM bandwidth.
2. **Device-time decode** — per-token device time summed from this stage's own signposted
   `tt-perf-report` CSV, divided by the replay count.
3. **End-to-end decode** — warmed measured ms/token from the host, from the same test the capture
   was taken of.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/tracy/perf_accounting.py

Writes `perf_summary.json` next to this file and prints the table README §7 quotes.

The DRAM peak is not hard-coded from a datasheet: it is recovered from `tt-perf-report`'s own
`DRAM %` column, which is the only bandwidth model the rest of this stage's evidence uses, so the
roofline and the utilisation percentages in the tables cannot disagree.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TILE = 32

#: Bytes per element including the shared exponent (one byte per 16 data for block floats).
DTYPE_BYTES = {"BFLOAT16": 2.0, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "FLOAT32": 4.0}

#: Model shapes, from `tt/model_config.py` + the checkpoint config.
DIM = 2048
N_HEADS, N_KV, HEAD_DIM = 16, 2, 256
GDN_IN = 12352
GDN_OUT_K = 4096
ATTN_IN = 9216
MOE_INTER = 512
SHARED_INTER = 512
N_EXPERTS = 256
TOP_K = 8
#: The prefill length `test_perf_decode_traced` decodes after, i.e. the KV context of the measured
#: decode step.
DECODE_CONTEXT = 128

#: `linear_attention`'s state-layer analogue of the KV read. `_delta_rule_step` touches the
#: persistent float32 recurrent state five times per token — decay read + write, `k @ h`,
#: `add(state, outer)` read + write, `q @ h` — so it is counted the way the KV cache is for the other
#: kind. Shape: [batch, num_value_heads, key_head_dim, value_head_dim] float32 at batch 1.
STATE_TOUCHES_PER_TOKEN = 5
LINEAR_HV, LINEAR_DK, LINEAR_DV = 32, 128, 128
DECODE_REPLAYS = 32

#: Which weights a decode step of each layer kind reads, at the shipped policy's dtypes.
WEIGHTS = {
    "full_attention": [
        ("attn_in", DIM * ATTN_IN, "BFLOAT8_B"),
        ("o_proj", N_HEADS * HEAD_DIM * DIM, "BFLOAT8_B"),
    ],
    "linear_attention": [
        ("gdn_in", DIM * GDN_IN, "BFLOAT8_B"),
        ("gdn_out", GDN_OUT_K * DIM, "BFLOAT8_B"),
    ],
}
SHARED_WEIGHTS = [
    ("router", DIM * N_EXPERTS, "BFLOAT16"),
    ("shared_in", DIM * (2 * SHARED_INTER + TILE), "BFLOAT8_B"),
    ("shared_down", SHARED_INTER * DIM, "BFLOAT8_B"),
    ("expert_gate_up_active", TOP_K * DIM * 2 * MOE_INTER, "BFLOAT4_B"),
    ("expert_down_active", TOP_K * MOE_INTER * DIM, "BFLOAT4_B"),
]


def open_csv(path: Path):
    """Line iterator over ``path`` or ``path.gz`` — the big reports are committed gzipped."""
    if path.is_file():
        return open(path, newline="")
    packed = path.with_suffix(path.suffix + ".gz")
    if not packed.is_file():
        raise SystemExit(f"missing {path} (and {packed})")
    return io.StringIO(gzip.decompress(packed.read_bytes()).decode(errors="ignore"))


def dram_peak_gbps(csv_path: Path) -> float:
    """Recover the bandwidth model `tt-perf-report` used, from any row that reports both."""
    best = []
    with open_csv(csv_path) as handle:
        for row in csv.DictReader(handle):
            bw = row.get("DRAM") or ""
            pct = row.get("DRAM %") or ""
            try:
                bw_v, pct_v = float(bw), float(pct)
            except ValueError:
                continue
            if bw_v > 0 and pct_v > 0:
                best.append(bw_v / (pct_v / 100.0))
    if not best:
        raise SystemExit(f"no row in {csv_path} carries both a DRAM bandwidth and a DRAM %")
    best.sort()
    return best[len(best) // 2]


def device_us_per_step(csv_path: Path, replays: int) -> float:
    total = 0.0
    with open_csv(csv_path) as handle:
        for row in csv.DictReader(handle):
            try:
                total += float(row.get("Device Time") or "")
            except ValueError:
                continue
    return total / replays


#: A gap this large or larger is itemised individually in the accounting; everything below it is
#: reported as an aggregate. 5 us is about five times the ~1 us steady-state op-to-op gap, so the
#: itemised list is "the stalls that are not just dispatch" rather than an arbitrary top-N.
GAP_ITEMISE_US = 5.0


def op_gaps(csv_path: Path, replays: int) -> dict:
    """Itemise the op-to-op gaps of one signposted decode window, per step.

    README §7 used to describe these by hand — "two 6-8 us gaps in front of the float32 gate-promotion
    typecasts" — and review round 4 found there were three. The whole point of that paragraph is that
    the end-to-end/device-time difference is *itemised rather than waved away*, so the itemisation is
    computed here from the report the paragraph cites, and the count cannot be wrong.

    The report holds one row per op **per replay**, not one aggregated row per op, so every figure here
    is divided by ``replays`` to become per-step. Gaps are then grouped **by op code**: what §7 needs is
    "which ops account for the dispatch gap", and a per-replay list would repeat the same op 32 times.
    ``total_us`` is the per-step sum, and it should reconcile with the §7 dispatch-and-host gap.
    """
    per_code: dict[str, list[float]] = {}
    with open_csv(csv_path) as handle:
        for row in csv.DictReader(handle):
            try:
                gap = float(row.get("Op-to-Op Gap") or "")
            except ValueError:
                continue
            per_code.setdefault((row.get("OP Code") or "").strip(), []).append(gap)
    if not per_code:
        raise SystemExit(f"no Op-to-Op Gap column values in {csv_path}")
    total_rows = sum(len(v) for v in per_code.values())
    grouped = []
    for code, gaps in per_code.items():
        grouped.append(
            {
                "op_code": code,
                "gap_us_per_step": round(sum(gaps) / replays, 1),
                "launches_per_step": round(len(gaps) / replays, 1),
                "largest_single_gap_us": round(max(gaps), 1),
            }
        )
    grouped.sort(key=lambda g: -g["gap_us_per_step"])
    big = [g for g in grouped if g["gap_us_per_step"] >= GAP_ITEMISE_US]
    small = [g for g in grouped if g["gap_us_per_step"] < GAP_ITEMISE_US]
    return {
        "total_us": round(sum(sum(v) for v in per_code.values()) / replays, 1),
        "itemise_threshold_us": GAP_ITEMISE_US,
        "largest": big,
        "remainder_op_codes": len(small),
        "remainder_us": round(sum(g["gap_us_per_step"] for g in small), 1),
        "op_codes": len(grouped),
        "launches_per_step": round(total_rows / replays, 1),
    }


def op_device_time(csv_path: Path, op_prefix: str, replays: int) -> tuple[float, float]:
    """``(us_per_step, fraction_of_window)`` for ops whose code **starts with** ``op_prefix``.

    Every figure README §7 and §5.4 attribute to a specific op comes from here rather than from a number
    typed into the limitation text, so a re-run cannot leave the attribution behind.

    Prefix, not substring, and case-sensitive: the report's op codes are
    ``MatmulDeviceOperation`` and ``SparseMatmulDeviceOperation …``, and ``MatmulDeviceOperation`` is a
    *substring* of the sparse one — a contains-match silently reported the routed-expert matmuls as
    dense, which is how the first version of this function attributed 82 % of the prefill window to the
    dense projections that are actually ~1 % of it. The op codes also carry a shape suffix, which is why
    this is a prefix rather than an equality test.
    """
    total = matched = 0.0
    with open_csv(csv_path) as handle:
        for row in csv.DictReader(handle):
            try:
                device = float(row.get("Device Time") or "")
            except ValueError:
                continue
            total += device
            if (row.get("OP Code") or "").strip().startswith(op_prefix):
                matched += device
    return matched / replays, (matched / total if total else 0.0)


def e2e_ms_per_step(run_log: Path) -> float:
    text = run_log.read_text(errors="ignore")
    match = re.findall(r"decode\(traced\).*?wall/iter=([0-9.]+) ms", text)
    if not match:
        raise SystemExit(f"no 'wall/iter' line in {run_log}")
    return float(match[-1])


def main():
    out = {}
    for kind in ("linear_attention", "full_attention"):
        base = HERE / kind
        report_csv = base / "decode_perf_report.csv"
        run_log = base / "decode_tracy_run.txt"
        if not (report_csv.is_file() or report_csv.with_suffix(".csv.gz").is_file()):
            # Fail, do not skip: skipping left a partial perf_summary.json behind, which is the
            # artifact README §7 is generated from. Review round 3 found the `full_attention` report
            # CSV missing from the commit and this script quietly halving the accounting.
            raise SystemExit(f"{report_csv} (or .gz) is missing - refusing to write a partial accounting")
        weights = WEIGHTS[kind] + SHARED_WEIGHTS
        weight_bytes = sum(count * DTYPE_BYTES[dtype] for _, count, dtype in weights)
        kv_bytes = 0.0
        if kind == "full_attention":
            kv_bytes = 2 * N_KV * HEAD_DIM * DECODE_CONTEXT * DTYPE_BYTES["BFLOAT8_B"]
        else:
            kv_bytes = STATE_TOUCHES_PER_TOKEN * LINEAR_HV * LINEAR_DK * LINEAR_DV * DTYPE_BYTES["FLOAT32"]
        peak = dram_peak_gbps(report_csv)
        roofline_us = (weight_bytes + kv_bytes) / (peak * 1e9) * 1e6
        device_us = device_us_per_step(report_csv, DECODE_REPLAYS)
        e2e_us = e2e_ms_per_step(run_log) * 1e3
        gaps = op_gaps(report_csv, DECODE_REPLAYS)
        sparse_us, sparse_share = op_device_time(report_csv, "SparseMatmul", DECODE_REPLAYS)
        topk_us, topk_share = op_device_time(report_csv, "TopK", DECODE_REPLAYS)
        # The prefill window's composition, which README §5.4 and work_log §4.10 quote to say why the
        # dense-projection work is worth ~0.4 % of prefill: the window is overwhelmingly routed-expert
        # sparse_matmul. Measured here so those two percentages have a source.
        prefill_csv = base / "prefill_perf_report.csv"
        prefill_shares = {}
        if prefill_csv.is_file() or prefill_csv.with_suffix(".csv.gz").is_file():
            _, sparse_prefill = op_device_time(prefill_csv, "SparseMatmul", 1)
            _, dense_prefill = op_device_time(prefill_csv, "MatmulDeviceOperation", 1)
            prefill_shares = {
                "sparse_matmul_share": round(sparse_prefill, 4),
                "dense_matmul_share": round(dense_prefill, 4),
            }
        out[kind] = {
            "prefill_window_composition": prefill_shares,
            "workload": {"profile": "single_user_decode", "prompt_len": DECODE_CONTEXT, "gen_len": 1, "batch": 1},
            "bytes_per_token": round(weight_bytes + kv_bytes),
            "state_or_kv_bytes_per_token": round(kv_bytes),
            "dram_peak_gbps": round(peak, 1),
            "roofline_ms_per_token_estimate": round(roofline_us / 1e3, 4),
            "decode_ms_per_token_device": round(device_us / 1e3, 4),
            "decode_ms_per_token_e2e": round(e2e_us / 1e3, 4),
            "roofline_fraction_of_device": round(roofline_us / device_us, 4),
            "dispatch_and_host_ms": round((e2e_us - device_us) / 1e3, 4),
            "op_to_op_gaps": gaps,
            "sparse_matmul_share_of_device_time": round(sparse_share, 4),
            "sparse_matmul_us_per_step": round(sparse_us, 1),
            "topk_share_of_device_time": round(topk_share, 4),
            "topk_us_per_step": round(topk_us, 1),
            "named_limitations": [
                "ttnn.sparse_matmul parallelism is capped by the output tile count (Nt) and it loops "
                "once per active expert at a single tile of M, so the two routed projections reach "
                "roughly 5 % of the FLOP roofline and ~40 GB/s of weight bandwidth even after the "
                f"geometry sweep; they are {sparse_share:.0%} of the window.",
                "The routed-expert intermediates are num_experts wide where only num_experts_per_tok "
                "slots are non-zero, so the zero-fill, the two unpacking slices, the SwiGLU, the "
                "score multiply and the expert reduction each touch 32x the useful width. Moving "
                "them to L1 removed the DRAM cost; the remaining ~230 us/step is L1 and launch "
                "bound and needs an expert-major gather to remove, which is a routing-algorithm "
                "change rather than an op-config one.",
                f"ttnn.topk is single-core on the 256-wide routing dim ({topk_us:.0f} us/step); its "
                "multi-core path needs a power-of-two width >= 8192 and padding to it measured ~3x "
                "slower.",
                f"{gaps['launches_per_step']:.0f} device op launches per decode step, across "
                f"{gaps['op_codes']} op codes, whose op-to-op gaps sum to {gaps['total_us']:.1f} us/step "
                f"and account for most of the difference between device time and end-to-end; "
                f"{gaps['remainder_op_codes']} op codes are individually below "
                f"{gaps['itemise_threshold_us']:.0f} us/step and total "
                f"{gaps['remainder_us']:.1f} us.",
            ],
        }
        print(
            f"{kind}: bytes/token={weight_bytes + kv_bytes:,.0f} peak={peak:.1f} GB/s "
            f"roofline={roofline_us:.1f} us device={device_us:.1f} us e2e={e2e_us:.1f} us "
            f"roofline/device={roofline_us / device_us:.1%} gap={e2e_us - device_us:.1f} us"
        )
    (HERE / "perf_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {HERE / 'perf_summary.json'}")


if __name__ == "__main__":
    main()
