# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real hybrid-layer external ownership, continuation and slot-logit checks."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import OrnithGenerator, build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh = open_ornith_mesh()
    report = {}
    try:
        gen = build_generator(
            Path(__file__).resolve().parents[2], mesh, layer_indices=[0, 3], max_batch_size=4, cache_context=2048
        )
        try:
            prompt = list(range(130))

            def prefill(rows, lengths, slots, starts=None):
                device = gen.prefill_forward(
                    rows,
                    page_table=gen.page_table,
                    kv_cache=gen.kv_cache,
                    prompt_lens=lengths,
                    slots=slots,
                    start_pos=starts,
                    return_device_logits=True,
                )
                logits = gen.model.logits_to_host(device, 4)
                ttnn.deallocate(device)
                return logits

            gen.reset()
            baseline = prefill([prompt, prompt], [130, 130], [0, 3])
            assert torch.equal(baseline[0], baseline[3]), "same prompt differs across fixed slots"
            report["identical_logits_slots_0_3"] = True
            gen.reset()
            repeated = prefill([prompt], [130], [3])
            assert torch.equal(baseline[3], repeated[3]), "repeated prompt logits changed"
            gen.reset()
            prefill([prompt[:127]], [127], [3])
            continuation = prefill([prompt[127:]], [3], [3], [127])
            pcc = torch.corrcoef(torch.stack([baseline[3], continuation[3]]).float())[0, 1].item()
            top = int(baseline[3].argmax())
            assert top in continuation[3].topk(5).indices.tolist() and pcc > 0.99, pcc
            report["continuation_127_plus_3"] = dict(
                pcc=pcc, baseline_top1=top, continuation_top1=int(continuation[3].argmax())
            )
            buffers = gen.model.cache_buffers(gen.kv_cache)
            snapshots = [host(tensor).clone() for tensor in buffers]
            external = OrnithGenerator(gen.model, kv_cache=gen.kv_cache, page_table=gen.page_table)
            try:
                external.ensure_traces()
                assert not external.owns_cache
                for index, (tensor, expected) in enumerate(zip(buffers, snapshots)):
                    assert torch.equal(host(tensor), expected), f"external cache buffer {index} changed during warmup"
                report["external_cache_survives_trace_warmup"] = dict(buffers=len(buffers))
            finally:
                external.teardown()
            gen.reset()
            for tensor in gen.model.cache_buffers(gen.kv_cache):
                assert bool((host(tensor) == 0).all()), "reset left nonzero cache state"
            report["explicit_reset_clears_state_and_kv"] = True
            device_tokens = gen.generate(prompt, 8, stop_on_eos=False)
            gen.teardown()
            compatible = OrnithGenerator(gen.model, max_batch_size=1, cache_context=2048, sampling_mode="host")
            try:
                host_tokens = compatible.generate(prompt, 8, stop_on_eos=False)
                assert host_tokens == device_tokens, (host_tokens, device_tokens)
                calls = []

                def choose(logits, **kwargs):
                    calls.append(kwargs["step"])
                    return logits.argmax(-1)

                callback_tokens = compatible.generate(prompt, 8, stop_on_eos=False, host_sample=choose)
                assert callback_tokens == host_tokens and calls == list(range(8))
                report["host_sampling_compatibility"] = dict(
                    tokens=host_tokens, callback_steps=calls, sampling_trace=compatible._sampling_trace
                )
            finally:
                compatible.teardown()
            report["pass"] = True
        finally:
            gen.teardown()
    finally:
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
