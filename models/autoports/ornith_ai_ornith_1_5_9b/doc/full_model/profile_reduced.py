# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy collection uses only real layers 0/3 and the complete terminal/sampling path."""

import argparse
from pathlib import Path

import tracy

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["decode", "prefill"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    mesh = open_ornith_mesh()
    try:
        gen = build_generator(root, mesh, layer_indices=[0, 3], cache_context=None)
        try:
            gen.generate([100] * 128, 2)
            gen.generate([100] * 128, 2)
            ttnn.synchronize_device(mesh)
            tracy.signpost("start")
            if args.mode == "decode":
                for _ in range(4):
                    gen._replay()
            else:
                output = gen.prefill_forward(
                    [[100] * 128],
                    page_table=gen.page_table,
                    kv_cache=gen.kv_cache,
                    prompt_lens=[128],
                    return_device_logits=True,
                )
                gen._sample_device(output)
            ttnn.synchronize_device(mesh)
            tracy.signpost("stop")
        finally:
            gen.teardown()
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
