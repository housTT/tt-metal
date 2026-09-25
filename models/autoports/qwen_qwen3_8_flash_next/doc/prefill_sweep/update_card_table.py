"""Rewrite the 'Endpoint performance' table in tt-model.yaml from a probe_rows.py JSON."""
import json, re, sys
from pathlib import Path
yaml_path = Path(sys.argv[1]); src = Path(sys.argv[2]); date = sys.argv[3]
if src.suffix == ".jsonl":
    # bench-sweeps rows.jsonl (the community latency-sweep harness) -> probe_rows schema
    rows = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") not in ("ok", "slow"):
            continue
        rows.append({"isl": r["isl"], "osl": r["osl"], "num_prompts": r.get("num_prompts") or r.get("completed"),
                     "ttft_ms": r["ttft_s"]["mean"] * 1000, "tpot_ms": r["tpot_s"]["mean"] * 1000, "e2el_s": r["e2el_s"]["mean"],
                     "prefill_tok_s_user": r.get("prefill_tok_s_user")})
    rows.sort(key=lambda r: (r["isl"], r["osl"]))
else:
    rows = json.load(open(src))
def s(v, unit):
    return f"{v/1000:.2f} s" if unit == "ms->s" and v >= 10000 else (f"{v/1000:.2f} s" if unit == "ms->s" else v)
lines = ["    | ISL | OSL | Users | Requests | Mean TTFT | Prefill tok/s | Mean TPOT | Decode tok/s/user | E2EL |",
         "    | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for r in rows:
    ttft = r["ttft_ms"]; ttft_s = f"{ttft/1000:.2f} s" if ttft < 100000 else f"{ttft/1000:.0f} s"
    e2 = r["e2el_s"]; e2s = f"{e2:.1f} s" if e2 < 100 else f"{e2:.0f} s"
    prefill = r.get("prefill_tok_s_user") or (r["isl"] / (ttft / 1000.0))
    lines.append(f"    | {r['isl']:,} | {r['osl']} | 1 | {r['num_prompts']} | {ttft_s} | {prefill:,.0f} | {r['tpot_ms']:.1f} ms | {1000/r['tpot_ms']:.1f} | {e2s} |")
text = yaml_path.read_text()
new = re.sub(r"    \| ISL \| OSL \| Users \| Requests \| Mean TTFT.*?(?=\n\n)", "\n".join(lines), text, count=1, flags=re.S)
new = re.sub(r"Measured \d{4}-\d{2}-\d{2} on two P300 boards", f"Measured {date} on two P300 boards", new, count=1)
assert new != text, "table not found"
yaml_path.write_text(new); print("card table updated with", len(rows), "rows")
