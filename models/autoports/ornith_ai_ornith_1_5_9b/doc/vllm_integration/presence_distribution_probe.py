# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host/device presence control and certified top-logprob margin analysis.

Run only from the supervising serving lane. This client never imports TTNN.
Raw prompts are continuation stress coverage, not chat-quality evaluation.
"""

import argparse
import json
from pathlib import Path

import requests


def token_id(value):
    if isinstance(value, int):
        return value
    prefix, sep, token = value.partition(":")
    if prefix != "token_id" or not sep:
        raise ValueError(f"Expected token-ID logprobs, got {value!r}")
    return int(token)


def analyze(choice, penalty, top_k):
    """Apply the exact presence formula to raw logprobs (normalizer cancels)."""
    logs = choice["logprobs"]
    seen, steps = set(), []
    for position, (token, candidates) in enumerate(zip(logs["tokens"], logs["top_logprobs"])):
        chosen = token_id(token)
        raw = {token_id(key): float(value) for key, value in candidates.items()}
        adjusted = {key: value - (penalty if key in seen else 0) for key, value in raw.items()}
        best = max(adjusted, key=adjusted.get)
        ranked = sorted(raw.values(), reverse=True)
        # The API can append the chosen token outside the requested raw top-K.
        # Use the Kth-largest value, never the minimum of that extended map.
        cutoff = ranked[top_k - 1] if len(ranked) >= top_k else None
        upper_unreturned = None if cutoff is None else cutoff + max(0.0, -penalty)
        certified = upper_unreturned is not None and adjusted[best] > upper_unreturned + 1e-6
        raw_best = max(raw, key=raw.get)
        unseen = [score for key, score in raw.items() if key not in seen]
        best_unseen = max(unseen) if unseen else cutoff
        steps.append(
            {
                "position": position,
                "chosen": chosen,
                "seen": sorted(seen),
                "raw_best": raw_best,
                "adjusted_best": best,
                "chosen_matches_presence_formula": chosen in adjusted and adjusted[chosen] >= adjusted[best] - 1e-5,
                "topk_certifies_full_vocabulary": certified,
                "raw_winner_gap_to_best_unseen": None if best_unseen is None else raw[raw_best] - best_unseen,
                "unseen_gap_is_lower_bound": not bool(unseen),
                "raw_cutoff": cutoff,
                "adjusted_winner_margin_over_unreturned": (
                    None if upper_unreturned is None else adjusted[best] - upper_unreturned
                ),
            }
        )
        seen.add(chosen)
    return {
        "all_choices_follow_presence_formula": all(row["chosen_matches_presence_formula"] for row in steps),
        "all_steps_certified": all(row["topk_certifies_full_vocabulary"] for row in steps),
        "penalty_changes_raw_argmax_at": [row["position"] for row in steps if row["adjusted_best"] != row["raw_best"]],
        "steps": steps,
    }


def combined_history(choice):
    """Record history without inferring absolute logits from normalized logprobs."""
    prompt = choice["prompt_token_ids"]
    generated, steps = {}, []
    for position, value in enumerate(choice["logprobs"]["tokens"]):
        token = token_id(value)
        steps.append(
            {
                "position": position,
                "chosen": token,
                "in_prompt": token in prompt,
                "generated_count_before": generated.get(token, 0),
            }
        )
        generated[token] = generated.get(token, 0) + 1
    return {
        "formula_certification_available": False,
        "reason": "Raw logprobs omit logsumexp; repetition requires original absolute logit signs and values",
        "validation_scope": "Device/host token-ID and effect parity; arithmetic certified by separate exact-score probe",
        "prompt_token_ids": prompt,
        "history_contract": "Repetition uses prompt union generated; frequency/presence use generated only",
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="/home/hous/dev/ornith-1.5-9b/upstream")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--modes", nargs="+", choices=("device", "host_logprobs"), default=("device", "host_logprobs"))
    parser.add_argument("--penalties", nargs="+", type=float, default=(0.0, -1.5, 2.0))
    parser.add_argument("--combined", action="store_true", help="Neutral vs presence=2, frequency=0.5, repetition=2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = args.prompt or ["a b c a b c a b c"]
    settings = [(0.0, 0.0, 1.0), (2.0, 0.5, 2.0)] if args.combined else [(p, 0.0, 1.0) for p in args.penalties]
    penalties = [p for p, _, _ in settings]
    report = {
        "hf_revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
        "prompt_mode": "raw completion continuation stress; not chat-quality evidence",
        "raw_logprobs_contract": "pinned TT runner constructs vLLM Sampler() with default raw_logprobs",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for prompt in prompts:
        compared = {}
        for penalty, frequency, repetition in settings:
            for mode in args.modes:
                body = {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "presence_penalty": penalty,
                    "frequency_penalty": frequency,
                    "repetition_penalty": repetition,
                    "ignore_eos": True,
                    "return_token_ids": True,
                }
                if mode == "host_logprobs":
                    body.update(logprobs=args.top_logprobs, return_tokens_as_token_ids=True)
                response = requests.post(args.url.rstrip("/") + "/v1/completions", json=body, timeout=300)
                result = response.json()
                case = {"mode": mode, "request": body, "status": response.status_code, "response": result}
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                assert response.status_code == 200 and "error" not in result and result.get("choices"), result
                assert result["usage"]["completion_tokens"] == args.max_tokens, result
                choice = result["choices"][0]
                ids = choice.get("token_ids")
                if ids is None and mode == "host_logprobs":
                    ids = [token_id(value) for value in choice["logprobs"]["tokens"]]
                assert ids is not None and len(ids) == args.max_tokens, choice
                compared[penalty, mode] = ids
                if mode == "host_logprobs":
                    case["analysis"] = (
                        combined_history(choice) if repetition != 1.0 else analyze(choice, penalty, args.top_logprobs)
                    )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "penalty": penalty,
                            "prompt": prompt,
                            "ids": ids,
                            "analysis": {
                                key: value for key, value in case.get("analysis", {}).items() if key != "steps"
                            },
                        }
                    ),
                    flush=True,
                )
        controls = {
            "prompt": prompt,
            "device_matches_host": {
                str(p): compared[p, "device"] == compared[p, "host_logprobs"]
                for p in penalties
                if (p, "device") in compared and (p, "host_logprobs") in compared
            },
            "host_penalties_vary": (
                len({tuple(compared[p, "host_logprobs"]) for p in penalties}) > 1
                if "host_logprobs" in args.modes
                else None
            ),
            "device_penalties_vary": (
                len({tuple(compared[p, "device"]) for p in penalties}) > 1 if "device" in args.modes else None
            ),
        }
        report.setdefault("controls", []).append(controls)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(controls), flush=True)
    # Preserve all evidence before asserting control disagreements.
    assert all(all(control["device_matches_host"].values()) for control in report["controls"]), report["controls"]
    host_cases = [
        case
        for case in report["cases"]
        if case["mode"] == "host_logprobs" and case["request"]["repetition_penalty"] == 1.0
    ]
    assert all(
        case["analysis"]["all_choices_follow_presence_formula"] for case in host_cases
    ), "Host choices disagree with presence formula"
    if args.combined:
        for control in report["controls"]:
            for mode in args.modes:
                key = "host_penalties_vary" if mode == "host_logprobs" else "device_penalties_vary"
                assert control[key], "Combined control must exercise an observable penalty effect"


if __name__ == "__main__":
    main()
