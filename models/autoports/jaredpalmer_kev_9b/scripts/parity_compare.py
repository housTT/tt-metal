import argparse
import json
import sys


def served(report):
    out = {}
    for row in report["per_record"]:
        for qid, q in row["questions"].items():
            out[(row["record"], qid)] = q["served"]
    return out


def compare(paths):
    reports = {p: json.load(open(p)) for p in paths}
    base_path = paths[0]
    base = served(reports[base_path])
    result = {"base": base_path, "questions": len(base), "runs": {}}
    for p in paths[1:]:
        other = served(reports[p])
        missing = sorted(k for k in base if k not in other)
        differing = sorted(k for k in base if k in other and other[k] != base[k])
        result["runs"][p] = {
            "questions": len(other),
            "missing": [list(k) for k in missing],
            "differing": [{"record": k[0], "question": k[1], "base": base[k], "other": other[k]} for k in differing],
            "identical": not missing and not differing and len(other) == len(base),
            "summary_vs": reports[p].get("summary_vs"),
            "revisits": reports[p].get("revisits"),
        }
    result["all_identical"] = all(r["identical"] for r in result["runs"].values())
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Compare the served answers of several parity_remote.py reports (first file is the base); exit 1 when any answer differs."
    )
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    result = compare(a.reports)
    text = json.dumps(result, indent=1)
    if a.out:
        open(a.out, "w").write(text + "\n")
    for p, r in result["runs"].items():
        print(
            p,
            "identical"
            if r["identical"]
            else f"DIFFERENT: {len(r['differing'])} differing, {len(r['missing'])} missing",
        )
    print("all_identical", result["all_identical"])
    sys.exit(0 if result["all_identical"] else 1)


if __name__ == "__main__":
    main()
