from pathlib import Path

root = Path("/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests")

p = root / "hf_reference.py"
s = p.read_text()
old = """            for row in range(probs.shape[0]):
                boundary = probs[row, selected[row, -1]]
                tied = torch.nonzero(probs[row] == boundary).flatten().tolist()
                keep = [e for e in selected[row].tolist() if probs[row, e] != boundary]
                slots = len(selected[row]) - len(keep)
                if len(tied) <= slots:
                    continue
"""
new = """            for row in range(probs.shape[0]):
                boundary = logits[row, selected[row, -1]].float()
                ulp = 2.0 ** (torch.floor(torch.log2(boundary.abs().clamp_min(1e-30))) - 7)
                near = (logits[row].float() - boundary).abs() <= ulp
                tied = torch.nonzero(near).flatten().tolist()
                keep = [e for e in selected[row].tolist() if not near[e]]
                slots = len(selected[row]) - len(keep)
                if len(tied) <= slots:
                    continue
                exact = bool((probs[row, tied] == probs[row, selected[row, -1]]).all())
"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """                    order = [selected[row].tolist().index(e) if e in selected[row].tolist() else None
                             for e in alternative[0].tolist()]
                    alt_weights = torch.stack([weights[row, o] if o is not None else weights[row, -1] for o in order])
                    delta = experts_forward(hidden[row : row + 1], alternative, alt_weights[None]) - base"""
new = """                    chosen_probs = probs[row, alternative[0]]
                    alt_weights = (chosen_probs / chosen_probs.sum()).to(weights.dtype)
                    delta = experts_forward(hidden[row : row + 1], alternative, alt_weights[None]) - base"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """                alternatives[(row, i)] = dict(tied=tied, slots=slots, deltas=deltas)"""
new = """                alternatives[(row, i)] = dict(tied=tied, slots=slots, deltas=deltas, exact=exact)"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

p = root / "run_layer.py"
s = p.read_text()
old = """        strict_dec = expected_dec
        expected_dec, tie_records = expected_dec.clone(), []
        for (row, step), tie in sorted(router_ties.items()):"""
new = """        strict_dec = expected_dec
        expected_dec, tie_records = (expected_dec.clone() if expected_dec is not None else None), []
        for (row, step), tie in sorted(router_ties.items()):"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """            tie_records.append(dict(row=row, step=step, tied_experts=tie["tied"], open_slots=tie["slots"],
                                    candidates=len(candidates), chosen=chosen,"""
new = """            tie_records.append(dict(row=row, step=step, tied_experts=tie["tied"], open_slots=tie["slots"],
                                    exact=tie["exact"], candidates=len(candidates), chosen=chosen,"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

p = root / "run_positions.py"
s = p.read_text()
old = """    from .hf_reference import decode_reference, prefill_reference"""
new = """    from .hf_reference import decode_reference_with_ties, prefill_reference"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """        expected_decode = decode_reference(reference, rope, cache, x[:, length:], length)"""
new = """        expected_decode, ties = decode_reference_with_ties(reference, rope, cache, x[:, length:], length)"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """        case.update(expected_prefill=expected_prefill, expected_decode=expected_decode, hf_cache=cache)"""
new = """        case.update(expected_prefill=expected_prefill, expected_decode=expected_decode, hf_cache=cache, ties=ties)"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """        for case in cases:
            row = case["row"]
            values = [
                pcc(case["expected_decode"][:, step : step + 1], actual_decode[row : row + 1, step : step + 1])
                for step in range(args.steps)
            ]
            result["decode_rows"].append(
                {
                    "row": row,
                    "slot": case["slot"],
                    "initial_position": case["length"],
                    "pcc_by_token": values,
                    "pcc_min": min(values),
                }
            )"""
new = """        for case in cases:
            row = case["row"]
            strict = [
                pcc(case["expected_decode"][:, step : step + 1], actual_decode[row : row + 1, step : step + 1])
                for step in range(args.steps)
            ]
            values, tie_records = list(strict), []
            for (_, step), tie in sorted(case["ties"].items()):
                base = case["expected_decode"][0, step].float()
                candidates = [base] + [base + d.float().reshape(base.shape) for d in tie["deltas"]]
                actual = actual_decode[row, step].float()
                distance = [float((c - actual).norm()) for c in candidates]
                chosen = min(range(len(candidates)), key=distance.__getitem__)
                values[step] = pcc(candidates[chosen], actual)
                tie_records.append(dict(step=step, tied_experts=tie["tied"], open_slots=tie["slots"],
                                        exact=tie["exact"], candidates=len(candidates), chosen=chosen,
                                        pcc_strict=strict[step], pcc_chosen=values[step]))
            result["decode_rows"].append(
                {
                    "row": row,
                    "slot": case["slot"],
                    "initial_position": case["length"],
                    "pcc_by_token": values,
                    "pcc_by_token_strict": strict,
                    "router_ties": tie_records,
                    "pcc_min": min(values),
                    "pcc_min_strict": min(strict),
                }
            )"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """        result["decode_pcc_min"] = min(row["pcc_min"] for row in result["decode_rows"])"""
new = """        result["decode_pcc_min"] = min(row["pcc_min"] for row in result["decode_rows"])
        result["decode_pcc_min_strict"] = min(row["pcc_min_strict"] for row in result["decode_rows"])"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print("patched")
