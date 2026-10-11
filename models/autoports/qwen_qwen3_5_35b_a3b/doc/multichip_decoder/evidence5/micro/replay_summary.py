import csv
import collections
import json
import statistics
import sys

path, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(path)))
replay = [r for r in rows if r["METAL TRACE REPLAY SESSION ID"]]
sessions = sorted({int(r["METAL TRACE REPLAY SESSION ID"]) for r in replay})
measured = set(sessions[2:])
devices = sorted({r["DEVICE ID"] for r in replay})
result = {"sessions": len(sessions), "devices": devices, "per_device": {}}
for device in devices:
    per_session = collections.defaultdict(list)
    for r in replay:
        if r["DEVICE ID"] == device and int(r["METAL TRACE REPLAY SESSION ID"]) in measured:
            per_session[int(r["METAL TRACE REPLAY SESSION ID"])].append(r)
    spans, kernel_sums, by_op = [], [], collections.defaultdict(list)
    for session, ops in per_session.items():
        ops.sort(key=lambda r: int(r["DEVICE FW START CYCLE"]))
        start, end = int(ops[0]["DEVICE FW START CYCLE"]), int(ops[-1]["DEVICE FW END CYCLE"])
        spans.append((end - start) / 1.35)
        kernel_sums.append(sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in ops) / 1000)
        counts = collections.Counter()
        for r in ops:
            name = r["OP NAME"]
            counts[name] += 1
            by_op[(name, counts[name])].append(float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1000)
    ops_per_step = len(next(iter(per_session.values())))
    table = sorted(((statistics.median(v), k[0], k[1]) for k, v in by_op.items()), reverse=True)
    grouped = collections.defaultdict(float)
    for value, name, _ in table:
        grouped[name] += value
    result["per_device"][device] = {
        "ops_per_step": ops_per_step,
        "span_us_median_assuming_1350MHz": statistics.median(spans) / 1000,
        "kernel_sum_us_median": statistics.median(kernel_sums),
        "by_op_name_us": dict(sorted(grouped.items(), key=lambda kv: -kv[1])),
        "top_instances": [(round(v, 2), n, i) for v, n, i in table[:25]],
    }
json.dump(result, open(out, "w"), indent=1)
for device, d in result["per_device"].items():
    print(device, d["ops_per_step"], round(d["span_us_median_assuming_1350MHz"], 1), round(d["kernel_sum_us_median"], 1))
first = result["per_device"][devices[0]]
for name, value in list(first["by_op_name_us"].items())[:25]:
    print(f"{value:9.2f}  {name}")
