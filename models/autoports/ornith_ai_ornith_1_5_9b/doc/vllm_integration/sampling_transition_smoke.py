"""Target fresh penalty admission and explicit host/device mode transitions."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ("neutral", {}),
        ("presence", {"presence_penalty": 1.0}),
        ("frequency", {"frequency_penalty": 1.0}),
        ("repetition", {"repetition_penalty": 1.5}),
        ("host_logprobs", {"logprobs": 3}),
        ("device_after_host", {}),
    ]
    results = []
    for name, params in cases:
        body = {
            "model": "/home/hous/dev/ornith-1.5-9b/upstream",
            "prompt": [1000] * 131,
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
            **params,
        }
        response = requests.post("http://localhost:8000/v1/completions", json=body, timeout=180)
        result = response.json()
        results.append({"case": name, "request": body, "status": response.status_code, "response": result})
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        assert response.status_code == 200 and "error" not in result, result
        for key, expected in {"prompt_tokens": 131, "completion_tokens": 8, "total_tokens": 139}.items():
            assert result["usage"][key] == expected, result
        assert result["choices"], result
        if name == "host_logprobs":
            assert len(result["choices"][0]["logprobs"]["token_logprobs"]) == 8, result
        print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()
