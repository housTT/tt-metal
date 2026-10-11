import json
import sys
from pathlib import Path

E = Path(sys.argv[1])
C = Path(sys.argv[2])


def row(path):
    return json.loads(path.read_text())[0]


def base(directory):
    return row(C / directory / "commands.json")


out = []
out.append("| Layer | Profile | Prefill | Decode | Prefill speedup | Decode speedup | Decode efficiency | PCC vs baseline (prefill / min decode) | HF PCC (prefill / min decode) |")
out.append("|---|---|---:|---:|---:|---:|---:|---|---|")
for kind in ("full", "linear"):
    b = base(f"headline_base_{kind}")
    out.append(f"| {kind} | baseline (Stage 3) | {b['prefill_host_us']:.0f} us | {b['decode_host_us']:.1f} us | 1.00 | 1.00 | | | "
               f"{b['prefill_pcc']:.5f} / {b['decode_pcc']:.5f} |")
    for name, tp in (("p150", 1), ("p150x2", 2), ("p150x4", 4)):
        r = row(E / f"headline_{name}_{kind}" / "commands.json")
        c = r["control_equivalence"]
        sp, sd = b["prefill_host_us"] / r["prefill_host_us"], b["decode_host_us"] / r["decode_host_us"]
        out.append(f"| {kind} | {name} | {r['prefill_host_us']:.0f} us | {r['decode_host_us']:.1f} us | {sp:.2f} | {sd:.2f} | "
                   f"{100 * sd / tp:.0f}% | {c['prefill_pcc']:.5f} / {c['decode_pcc_min']:.5f} | "
                   f"{r['prefill_pcc']:.5f} / {r['decode_pcc']:.5f} |")
full = {n: row(E / f"headline_{n}_full" / "commands.json")["decode_host_us"] for n in ("p150", "p150x2", "p150x4")}
linear = {n: row(E / f"headline_{n}_linear" / "commands.json")["decode_host_us"] for n in ("p150", "p150x2", "p150x4")}
bf, bl = base("headline_base_full")["decode_host_us"], base("headline_base_linear")["decode_host_us"]
out.append("")
out.append(f"40-layer decode estimate (30 linear + 10 full): baseline {(30 * bl + 10 * bf) / 1000:.2f} ms; " +
           "; ".join(f"{n} {(30 * linear[n] + 10 * full[n]) / 1000:.2f} ms" for n in full) + ".")
out.append("")
out.append("| Tokens | Layer | Mesh | Exit | HF prefill | HF min decode | Control prefill | Control min decode | Prefill repeat identical | Prefill |")
out.append("|---|---|---|---|---:|---:|---:|---:|---|---:|")
for spec in ("3", "100", "1000", "2176", "2176_s128", "4095", "4097"):
    for kind in ("full", "linear"):
        for mesh in (1, 2, 4):
            path = E / f"len_m{mesh}_{kind}_{spec}" / "commands.json"
            if not path.exists():
                out.append(f"| {spec} | {kind} | {mesh} | missing | | | | | | |")
                continue
            r = row(path)
            c = r.get("control_equivalence") or {}
            out.append(f"| {spec} | {kind} | {mesh} | {r['exit_code']} | {r['prefill_pcc']:.5f} | {r['decode_pcc']:.5f} | "
                       f"{c.get('prefill_pcc', float('nan')):.5f} | {c.get('decode_pcc_min', float('nan')):.5f} | "
                       f"{r['prefill_repeat_identical']} | {r['prefill_host_us']:.0f} us |")
out.append("")
out.append("| Batch 32, 129 tokens | Layer | Prefill | Decode per step | Decode speedup | Control prefill / min decode | Prefill repeat identical |")
out.append("|---|---|---:|---:|---:|---|---|")
for kind in ("full", "linear"):
    b = base(f"b32_base_{kind}")
    out.append(f"| baseline | {kind} | {b['prefill_host_us']:.0f} us | {b['decode_host_us']:.1f} us | 1.00 | | {b['prefill_repeat_identical']} |")
    for mesh in (1, 2, 4):
        r = row(E / f"b32_m{mesh}_{kind}" / "commands.json")
        c = r["control_equivalence"]
        out.append(f"| mesh {mesh} | {kind} | {r['prefill_host_us']:.0f} us | {r['decode_host_us']:.1f} us | "
                   f"{b['decode_host_us'] / r['decode_host_us']:.2f} | {c['prefill_pcc']:.5f} / {c['decode_pcc_min']:.5f} | "
                   f"{r['prefill_repeat_identical']} |")
print("\n".join(out))
