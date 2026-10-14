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
    args = p.parse_args()
    import ttnn

    from ..tt.generator import build_generator
    from .mesh_util import open_mesh

    entry = torch.load(REFERENCE, weights_only=False)["entries"][0]
    seq = torch.cat([entry["prompt_tokens"][0], entry["generated_tokens"][0]]).long()
    prompt_len = entry["prompt_tokens"].shape[1]
    mesh = open_mesh(args.mesh)
    try:
        gen = build_generator(Path(__file__).parents[1], mesh)
        prefill = gen.prefill_logits(seq.tolist())[0, prompt_len - 1 : -1].float()
        gen.reset()
        table = gen.default_page_table(1, len(seq))
        first = gen.prefill_forward(seq[None, :prompt_len], page_table=table, prompt_lens=[prompt_len])[0, 0]
        decode = [first.float()]
        for i in range(prompt_len, len(seq) - 1):
            decode.append(gen.decode_forward(seq[i : i + 1].reshape(1, 1), [i], page_table=table)[0].float())
        torch.save(dict(seq=seq, prompt_len=prompt_len, prefill=prefill.clone(), decode=torch.stack(decode)),
                   args.output.with_suffix(".pt"))
        args.output.write_text(json.dumps(dict(mesh=args.mesh, prompt_len=prompt_len, positions=len(decode),
                                               counters=dict(gen.counters)), indent=1))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
