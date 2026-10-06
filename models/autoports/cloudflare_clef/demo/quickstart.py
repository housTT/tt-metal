import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PRESETS_PATH = DEMO_DIR / "presets.json"
DEFAULT_SERVER = "http://127.0.0.1:8008"
BAR_WIDTH = 24
REPLAY_FLOOR = 0.5


def load_presets(path):
    with open(path) as f:
        return json.load(f)


def image_b64(path, base_dir):
    p = Path(path)
    if not p.is_absolute():
        p = base_dir / p
    return base64.b64encode(p.read_bytes()).decode("ascii")


def build_request(case, model, base_dir):
    body = {"model": model, "state": case["state"], "questions": case["questions"]}
    if case.get("images"):
        body["images"] = [image_b64(p, base_dir) for p in case["images"]]
    return body


def post_json(server, path, body, api_key, timeout):
    data = json.dumps(body).encode()
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(server.rstrip("/") + path, data=data, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
            return r.status, payload, (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")
        try:
            payload = json.loads(text)
        except ValueError:
            payload = {"detail": text}
        return e.code, payload, (time.perf_counter() - started) * 1000
    except urllib.error.URLError as e:
        return (
            0,
            {"detail": f"cannot reach {server}: {e.reason}. Start the server first (see README.md)."},
            (time.perf_counter() - started) * 1000,
        )


def get_json(server, path, api_key, timeout=10):
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(server.rstrip("/") + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"detail": e.read().decode(errors="replace")}
    except urllib.error.URLError as e:
        return 0, {"detail": str(e.reason)}


def cache_hits(args):
    status, models = get_json(args.server, "/v1/models", args.api_key)
    if status != 200:
        return None
    try:
        return models["models"][0]["prefix_cache"]["hits"]
    except (KeyError, IndexError, TypeError):
        return None


def predicted(answer):
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    probs = answer["probabilities"]
    return int(max(probs, key=lambda k: probs[k]))


def same(pred, label):
    if isinstance(pred, bool) or isinstance(label, bool):
        return bool(pred) == bool(label)
    return str(pred) == str(label)


def bar(p, width=BAR_WIDTH):
    n = int(round(max(0.0, min(1.0, p)) * width))
    return "#" * n + "." * (width - n)


def format_label(label):
    if isinstance(label, bool):
        return "true" if label else "false"
    return str(label)


def detail_text(payload):
    d = payload.get("detail", payload)
    if isinstance(d, list):
        return "; ".join(f"{'.'.join(str(x) for x in e.get('loc', []))}: {e.get('msg', '')}" for e in d)
    return str(d)


def print_answers(request, response, labels):
    answers = response["answers"]
    for qid, q in request["questions"].items():
        a = answers[qid]
        label = labels.get(qid) if labels else None
        verdict = ""
        if label is not None:
            verdict = "  PASS" if same(predicted(a), label) else "  FAIL"
            verdict += f" (label {format_label(label)})"
        instr = q.get("instructions", "")
        instr = instr if isinstance(instr, str) else json.dumps(instr)
        if a["type"] == "choice":
            print(f"  {qid} [choice] {instr}")
            print(f"    choice {a['choice']}  confidence {a['confidence']:.4f}{verdict}")
            for k, p in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]):
                mark = "*" if k == a["choice"] else " "
                print(f"    {mark} {k[:28]:<28} {bar(p)} {p:.4f}")
        elif a["type"] == "noul":
            p = a["noul"]
            print(f"  {qid} [noul] {instr}")
            print(f"    {'true' if p >= 0.5 else 'false'}  p(true) {p:.4f}{verdict}")
            print(f"      {'true':<28} {bar(p)} {p:.4f}")
        else:
            levels = len(a["probabilities"])
            top = str(predicted(a))
            print(f"  {qid} [score] {instr}")
            print(f"    score {a['score']:.2f} of {levels - 1}  confidence {a['confidence']:.4f}{verdict}")
            for k in sorted(a["probabilities"], key=int):
                p = a["probabilities"][k]
                mark = "*" if k == top else " "
                legend = f"{k} {a['legend'].get(k, '')}"
                print(f"    {mark} {legend[:28]:<28} {bar(p)} {p:.4f}")
            pos = int(round(a["score"] / max(1, levels - 1) * (BAR_WIDTH - 1)))
            print(f"      {'expected value':<28} {' ' * pos}^")


def print_metrics(response, wall_ms, hit):
    usage = response.get("usage", {})
    extra = ""
    if "state_tokens" in usage:
        extra = f"  state tokens {usage['state_tokens']} (used {usage.get('state_tokens_used', '?')})"
    if hit is not None:
        extra += f"  prefix cache {'hit' if hit else 'miss'}"
    print(
        f"  server latency_ms {response['latency_ms']:.1f}  client wall {wall_ms:.0f} ms"
        f"  input_tokens {usage.get('input_tokens', '?')}{extra}"
    )


def run_case(args, case, labels, model):
    request = build_request(case, model, DEMO_DIR)
    if args.show_request:
        print(json.dumps(request, indent=2, ensure_ascii=False))
        return True, None, None
    before = cache_hits(args)
    status, payload, wall_ms = post_json(args.server, "/v1/systemone", request, args.api_key, args.timeout)
    if status != 200:
        print(f"  HTTP {status}: {detail_text(payload)}")
        return False, None, wall_ms
    after = cache_hits(args)
    hit = (after > before) if before is not None and after is not None else None
    print_answers(request, payload, labels)
    print_metrics(payload, wall_ms, hit)
    return True, payload, wall_ms


def select_presets(data, name):
    if name is None:
        return data["presets"]
    exact = [p for p in data["presets"] if p["title"].lower() == name.lower()]
    if exact:
        return exact
    prefix = [p for p in data["presets"] if p["title"].lower().startswith(name.lower())]
    if len(prefix) == 1:
        return prefix
    titles = ", ".join(repr(p["title"]) for p in data["presets"])
    raise SystemExit(f"unknown preset {name!r}; choose one of {titles}")


def run_presets(args, data):
    ok = True
    for preset in select_presets(data, args.preset):
        print(f"== {preset['title']}")
        print(f"   {preset['what_it_shows']}")
        for case in preset["cases"]:
            print(f"-- case {case['name']} ({case['source']})")
            good, _, _ = run_case(args, case, case.get("labels"), data.get("model", "clef"))
            ok = ok and good
        print()
    return ok


def run_replay(args, data):
    model = data.get("model", "clef")
    total = correct = 0
    records_passed = 0
    failures = 0
    latencies, walls = [], []
    started = time.perf_counter()
    for i, rec in enumerate(data["replay"], 1):
        request = build_request(rec, model, DEMO_DIR)
        if args.show_request:
            print(json.dumps(request, indent=2, ensure_ascii=False))
            continue
        status, payload, wall_ms = post_json(args.server, "/v1/systemone", request, args.api_key, args.timeout)
        if status != 200:
            failures += 1
            total += len(rec["labels"])
            print(f"{i:>2} {rec['name']:<36} HTTP {status}: {detail_text(payload)[:120]}")
            continue
        latencies.append(payload["latency_ms"])
        walls.append(wall_ms)
        parts, rec_ok = [], True
        for qid, label in rec["labels"].items():
            pred = predicted(payload["answers"][qid])
            hit = same(pred, label)
            rec_ok = rec_ok and hit
            total += 1
            correct += int(hit)
            parts.append(f"{qid}={format_label(pred)}{'' if hit else ' (label ' + format_label(label) + ')'}")
        records_passed += int(rec_ok)
        acc = correct / total if total else 0.0
        print(
            f"{i:>2} {'PASS' if rec_ok else 'FAIL'} {rec['suite']:<13} {rec['name']:<36} {payload['latency_ms']:>8.1f} ms"
            f"  tokens {payload.get('usage', {}).get('input_tokens', '?'):>5}  acc {acc:.3f}  {'; '.join(parts)}"
        )
    if args.show_request:
        return True
    elapsed = time.perf_counter() - started
    acc = correct / total if total else 0.0
    print()
    print(f"replay: {len(data['replay'])} records, {records_passed} passed, {failures} request failures")
    print(f"questions: {correct}/{total} correct, accuracy {acc:.3f} (smoke floor {REPLAY_FLOOR})")
    if latencies:
        print(
            f"server latency_ms: mean {statistics.mean(latencies):.1f}  p50 {statistics.median(latencies):.1f}"
            f"  max {max(latencies):.1f}; client wall mean {statistics.mean(walls):.0f} ms; total {elapsed:.1f} s"
        )
    return failures == 0 and acc >= REPLAY_FLOOR


def print_status(args):
    status, health = get_json(args.server, "/health", None)
    if status != 200:
        print(f"server {args.server}: unreachable ({health.get('detail')}). Start it first (see README.md).")
        return False
    status, models = get_json(args.server, "/v1/models", args.api_key)
    if status != 200:
        print(f"server {args.server}: health ok, /v1/models HTTP {status}: {detail_text(models)}")
        return status != 401
    card = models["models"][0]
    cache = card.get("prefix_cache", {})
    print(
        f"server {args.server}: {card['name']} backend {card.get('backend')} device {card.get('device')} mesh {card.get('mesh_shape')}"
        f" workers {len(card.get('workers', []))} prefix cache {cache.get('cached_states', '?')}/{cache.get('size', '?')} states"
        f" hits {cache.get('hits', '?')} misses {cache.get('misses', '?')} max state {card.get('max_state_tokens', '?')} tokens"
    )
    return True


def main():
    ap = argparse.ArgumentParser(description="Run the Clef demo presets from a terminal.")
    ap.add_argument("--server", default=DEFAULT_SERVER, help=f"Clef server URL (default {DEFAULT_SERVER})")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--preset", help="run one preset by title (exact or unique prefix)")
    group.add_argument("--all", action="store_true", help="run every preset (the default when nothing else is given)")
    group.add_argument(
        "--replay", action="store_true", help="run the labelled replay set and report accuracy and latency"
    )
    group.add_argument("--list", action="store_true", help="list the presets and exit")
    ap.add_argument("--api-key", default=None, help="bearer token when the server was started with CLEF_API_KEY")
    ap.add_argument("--presets", default=str(PRESETS_PATH), help="path to presets.json")
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    ap.add_argument("--show-request", action="store_true", help="print the request bodies instead of sending them")
    args = ap.parse_args()
    data = load_presets(args.presets)
    if args.list:
        for p in data["presets"]:
            names = ", ".join(c["name"] for c in p["cases"])
            print(f"{p['title']}: {len(p['cases'])} case(s) [{names}]. {p['what_it_shows']}")
        print(f"replay set: {len(data['replay'])} labelled records")
        return 0
    if not args.show_request and not print_status(args):
        return 1
    ok = run_replay(args, data) if args.replay else run_presets(args, data)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
