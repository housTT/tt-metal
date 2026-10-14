#!/usr/bin/env bash
set -u
source /home/hous/dev/qwen3.5-35b-a3b/scripts/environment.sh >/dev/null 2>&1
source /home/hous/dev/qwen3.5-35b-a3b/tt-metal/python_env/bin/activate
W=${1:?work dir}
E=${2:?evidence dir}
for m in ${MESHES:-4 2 1}; do
  f=$(find -L $W/profile_m$m -name "ops_perf_results*.csv" 2>/dev/null | head -1)
  [ -z "$f" ] && { echo "missing m$m"; continue; }
  out=$E/m$m/profile
  mkdir -p $out
  gzip -c $f > $out/ops_perf_results.csv.gz
  sha256sum $f > $out/ops_perf_results.sha256
  for ph in PREFILL DECODE; do
    tt-perf-report $f --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --csv $out/${ph,,}_perf_report.csv > $out/${ph,,}_perf_report.console.log 2>&1
    tt-perf-report $f --start-signpost PERF_$ph --end-signpost PERF_${ph}_END --no-summary --no-color > $out/${ph,,}_perf_report.txt 2>&1
  done
  python3 - "$f" "$out/summary.json" <<'PY'
import csv, collections, json, sys
rows = list(csv.DictReader(open(sys.argv[1])))
out = {}
for phase in ("PREFILL", "DECODE"):
    s = [i for i, r in enumerate(rows) if r["OP CODE"] == f"PERF_{phase}"][0]
    e = [i for i, r in enumerate(rows) if r["OP CODE"] == f"PERF_{phase}_END"][0]
    device = min(r["DEVICE ID"] for r in rows[s + 1:e] if r["DEVICE ID"])
    w = [r for r in rows[s + 1:e] if r["DEVICE ID"] == device]
    by_op = collections.defaultdict(float)
    for r in w:
        by_op[r["OP CODE"]] += float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3
    out[phase.lower()] = dict(device=device, ops=len(w), kernel_us=sum(by_op.values()),
                              top_ops_us=dict(sorted(by_op.items(), key=lambda kv: -kv[1])[:15]))
json.dump(out, open(sys.argv[2], "w"), indent=1)
PY
done
echo PROCESS_DONE
