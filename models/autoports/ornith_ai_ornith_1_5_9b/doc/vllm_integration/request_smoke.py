"""Repeat the exact reduced serving failure with explicit response gates."""

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", default="http://localhost:8000")
    args = parser.parse_args()

    def request(spec):
        token, length = spec
        body = {
            "model": "/home/hous/dev/ornith-1.5-9b/upstream",
            "prompt": [token] * length,
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
        }
        response = requests.post(f"{args.server_url}/v1/completions", json=body, timeout=180)
        return {"request": body, "status": response.status_code, "response": response.json()}

    results = {"single": request((1000, 131))}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results["concurrent"] = list(pool.map(request, [(1000, 131), (2028, 129), (17, 65), (1000, 131)]))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for item in [results["single"], *results["concurrent"]]:
        assert item["status"] == 200, item
        body = item["response"]
        assert "error" not in body and body.get("choices"), body
        assert body["usage"]["prompt_tokens"] == len(item["request"]["prompt"]), body
        assert body["usage"]["completion_tokens"] == 8, body
    print("PASS: 131-token request and four concurrent non-aligned requests")


if __name__ == "__main__":
    main()
