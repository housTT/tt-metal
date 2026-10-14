# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import torch

from .run_sampling_audit import PROMPTS

STEPS = 16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import _greedy_params, build_generator
    from .mesh_util import open_mesh

    mesh = open_mesh(args.mesh)
    result = dict(vars(args), output=str(args.output))
    checks = {}
    try:
        gen = build_generator(Path(__file__).parents[1], mesh, decode_batches=(1, 4, 32))
        tok = gen.tokenizer
        prompts = [tok(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                               add_generation_prompt=True), add_special_tokens=False).input_ids
                   for t in PROMPTS]
        greedy = _greedy_params()
        reference = [gen.generate(r, STEPS + 1) for r in prompts]
        width = max(len(r) for r in prompts) + STEPS + 1
        tables = gen.default_page_table(32, width)

        def case(batch, active, traced, hide_at=None):
            rows = sorted(active)
            ids = torch.zeros(len(rows), max(len(prompts[active[r]]) for r in rows), dtype=torch.int64)
            for i, r in enumerate(rows):
                ids[i, : len(prompts[active[r]])] = torch.tensor(prompts[active[r]])
            lens = [len(prompts[active[r]]) for r in rows]
            first = gen.prefill_forward(ids, page_table=tables[rows], prompt_lens=lens,
                                        request_slots=rows)[:, 0].argmax(-1).tolist()
            tokens = torch.zeros(batch, 1, dtype=torch.int64)
            positions = [-1] * batch
            for i, r in enumerate(rows):
                tokens[r, 0] = first[i]
                positions[r] = lens[i]
            out = {r: [first[i]] for i, r in enumerate(rows)}
            rejected = None
            for step in range(STEPS):
                if step == hide_at:
                    try:
                        gen.decode_forward(tokens, [-1] * batch, page_table=tables[:batch], sampling_params=greedy)
                        rejected = False
                    except ValueError:
                        rejected = True
                if traced and step > 0:
                    gen._set_sampling(greedy)
                    gen._replay(batch, sample=True)
                    chosen = gen.read_tokens(batch)
                else:
                    chosen = gen.decode_forward(tokens, positions, page_table=tables[:batch],
                                                sampling_params=greedy).reshape(-1)
                for r in rows:
                    out[r].append(int(chosen[r]))
                tokens = chosen.reshape(batch, 1)
                positions = [p + 1 if p >= 0 else -1 for p in positions]
            gen.release_slots(rows)
            matches = {str(r): out[r] == reference[active[r]] for r in rows}
            return dict(batch=batch, active={str(r): active[r] for r in rows}, traced=traced, matches=matches,
                        hide_at=hide_at, hide_rejected=rejected,
                        all_match=all(matches.values()) and rejected is not False)

        cases = []
        for traced in (False, True):
            gen.reset()
            cases.append(case(4, {0: 0, 2: 2}, traced))
            cases.append(case(32, {0: 0, 2: 2, 17: 1, 31: 3}, traced))
        cases.append(case(4, {1: 1}, True))
        cases.append(case(32, {5: 3}, False))
        cases.append(case(32, {5: 3, 1: 1}, True))
        cases.append(case(4, {2: 2}, True, hide_at=5))
        checks["hidden_live_slot_rejected"] = cases[-1]["hide_rejected"] is True
        result["cases"] = cases
        checks["active_rows_match_b1"] = all(c["all_match"] for c in cases)
        checks["greedy_b1_after_inactive_identical"] = gen.generate(prompts[0], STEPS + 1) == reference[0]
        checks["decode_captures"] = gen.counters["decode_captures"]
        result.update(checks=checks, counters=dict(gen.counters),
                      passed=all(v for v in checks.values() if isinstance(v, bool)))
        print(json.dumps(dict(checks=checks, cases=[(c["batch"], c["traced"], c["all_match"]) for c in cases])),
              flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
