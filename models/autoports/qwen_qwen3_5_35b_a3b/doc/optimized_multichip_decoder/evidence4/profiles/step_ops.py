import csv, sys, statistics, collections
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["METAL TRACE REPLAY SESSION ID"]]
first_device = min(r["DEVICE ID"] for r in rows)
rows = [r for r in rows if r["DEVICE ID"] == first_device]
sessions = sorted({int(r["METAL TRACE REPLAY SESSION ID"]) for r in rows})[2:]
per = collections.defaultdict(list)
for r in rows:
    s = int(r["METAL TRACE REPLAY SESSION ID"])
    if s in sessions:
        per[s].append(r)
steps = [sorted(v, key=lambda r: int(r["DEVICE FW START CYCLE"])) for v in per.values()]
for i, r in enumerate(steps[0]):
    d = statistics.median(float(s[i]["DEVICE KERNEL DURATION [ns]"] or 0) / 1000 for s in steps)
    g = statistics.median((int(s[i]["DEVICE FW START CYCLE"]) - int(s[i - 1]["DEVICE FW END CYCLE"])) / 1350 if i else 0 for s in steps)
    print(f"{i:3d} {d:7.2f} gap{g:6.2f} {r['CORE COUNT']:>4} {r['OP NAME'][:44]:44s} {r.get('INPUT_0_W_PAD[LOGICAL]','')[:14]:14s} {r.get('OUTPUT_0_W_PAD[LOGICAL]','')[:14]}")
