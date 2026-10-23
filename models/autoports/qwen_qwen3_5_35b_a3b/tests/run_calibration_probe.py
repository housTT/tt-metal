# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import torch

REFERENCE = Path("/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/references/readiness_aime24_chat.refpt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--batched", type=int, choices=[0, 1], default=0)
    p.add_argument("--distinct", type=int, choices=[0, 1], default=0)
    args = p.parse_args()
    import ttnn

    from ..tt.generator import build_generator
    from .mesh_util import open_mesh

    entry = torch.load(REFERENCE, weights_only=False)["entries"][0]
    seq = torch.cat([entry["prompt_tokens"][0], entry["generated_tokens"][0]]).long()
    prompt_len = entry["prompt_tokens"].shape[1]
    mesh = open_mesh(args.mesh)
    try:
        batch = args.batch
        gen = build_generator(Path(__file__).parents[1], mesh, decode_batches=tuple(sorted({1, batch})),
                              batched_decode=bool(args.batched))
        prefill = gen.prefill_logits(seq.tolist())[0, prompt_len - 1 : -1].float()
        gen.reset()
        table = gen.default_page_table(batch, len(seq))
        rows_seq = seq[None, :].repeat(batch, 1)
        if args.distinct:
            from .run_full_evidence import BOOK

            book = torch.tensor(gen.tokenizer(BOOK.read_text().strip() * 400, add_special_tokens=False).input_ids)
            for r in range(1, batch):
                rows_seq[r] = book[r * 97 : r * 97 + len(seq)]
        first = gen.prefill_forward(rows_seq[:, :prompt_len], page_table=table,
                                    prompt_lens=[prompt_len] * batch)[:, 0]
        decode, spread = [first[0].float()], []
        for i in range(prompt_len, len(seq) - 1):
            rows = gen.decode_forward(rows_seq[:, i : i + 1], [i] * batch,
                                      page_table=table).float().reshape(batch, -1)
            same = [r for r in range(batch) if torch.equal(rows_seq[r], seq)]
            spread.append(float((rows[same] - rows[0]).abs().max()))
            decode.append(rows[0])
        torch.save(dict(seq=seq, prompt_len=prompt_len, prefill=prefill.clone(), decode=torch.stack(decode)),
                   args.output.with_suffix(".pt"))
        args.output.write_text(json.dumps(dict(mesh=args.mesh, batch=batch, batched=args.batched, distinct=args.distinct,
                                               prompt_len=prompt_len,
                                               positions=len(decode), max_row_spread=max(spread),
                                               counters=dict(gen.counters)), indent=1))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
