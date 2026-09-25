# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stochastic free-run gate for the served default sampling (DEVSTACK-294).

The served default request (temperature 1.0, top_k 20, top_p 0.95) collapsed
into phrase loops on hard prompts under on-device sampling.  This gate runs the
AIME24 chat prompt through the device sampler with explicit seeds, one unseeded
run (which exercises the seed synthesis added to ``set_sampling_params``), and
a host-sampled control, and asserts none of the device runs is degenerate.  It
also checks the per-step seed refresh counter and that every device shard holds
the same sampling parameters and the same sampled token.

Run: ``RUN_QWEN38_STOCHASTIC_GATE=1 pytest -s tests/test_stochastic_free_run.py``
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.demo.full_model import _degeneracy, load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_full_model import REFERENCE, _device_params
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import Qwen38Generator
from models.autoports.qwen_qwen3_8_flash_next.tt.model import Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder

EVIDENCE = Path(os.getenv("QWEN38_EVIDENCE_DIR", str(Path(__file__).parents[1] / "doc/sampling_devstack294")))
NEW_TOKENS = int(os.getenv("QWEN38_STOCHASTIC_TOKENS", "1024"))
TOP_K, TOP_P, TEMPERATURE = 20, 0.95, 1.0
SEEDS = (17, 2024, 90210)


def _shards(tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard).reshape(-1) for shard in ttnn.get_device_tensors(tensor)]


def _all_equal(tensors: list[torch.Tensor]) -> bool:
    return all(torch.equal(tensors[0], other) for other in tensors[1:])


def _run(generator, prompt, *, sampling_mode: str, seeds, label: str) -> dict[str, object]:
    model = generator.model
    before = model.sampling_seed_host_copies
    started = time.perf_counter()
    tokens = generator.generate_batch(
        prompt,
        NEW_TOKENS,
        enable_trace=True,
        sampling_mode=sampling_mode,
        top_k=TOP_K,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        seeds=seeds,
        request_ids=(f"stochastic-{label}",),
        stop_on_eos=False,
    )[0]
    seconds = time.perf_counter() - started
    text = generator.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
    review = _degeneracy(tokens, text)
    return {
        "label": label,
        "sampling_mode": sampling_mode,
        "seeds": None if seeds is None else list(seeds),
        "generated_tokens": int(tokens.numel()),
        "seconds": seconds,
        "seed_host_copies_delta": model.sampling_seed_host_copies - before,
        "review": review,
        "four_gram_ratio": review["repeated_four_grams"] / max(int(tokens.numel()), 1),
        "tail_text": text[-400:],
    }


@pytest.mark.skipif(os.getenv("RUN_QWEN38_STOCHASTIC_GATE") != "1", reason="stochastic free-run gate")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_stochastic_free_run_not_degenerate(bh_1d_mesh_device, device_params, record_property):
    from transformers import AutoTokenizer

    del device_params
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    prompt = torch.as_tensor(load_reference(REFERENCE)["prompt_tokens"], dtype=torch.int64).reshape(1, -1)
    report: dict[str, object] = {
        "prompt_tokens": int(prompt.shape[1]),
        "new_tokens": NEW_TOKENS,
        "sampling": {"top_k": TOP_K, "top_p": TOP_P, "temperature": TEMPERATURE},
        "runs": [],
    }
    try:
        # Parameter and seed buffers must be identical on every device shard.
        model.set_sampling_params(top_k=TOP_K, top_p=TOP_P, temperature=TEMPERATURE, seeds=[123])
        model._advance_sampling_seeds()
        report["shards_equal"] = {
            "seeds": _all_equal(_shards(model.sampling._seeds)),
            "top_k": _all_equal(_shards(model.sampling_k)),
            "top_p": _all_equal(_shards(model.sampling_p)),
            "temperature": _all_equal(_shards(model.sampling_temp)),
        }
        EVIDENCE.mkdir(parents=True, exist_ok=True)

        def record(run: dict[str, object]) -> None:
            report["runs"].append(run)
            print({"stochastic_free_run": {k: v for k, v in run.items() if k != "tail_text"}}, flush=True)
            (EVIDENCE / "stochastic_free_run.json").write_text(json.dumps(report, indent=2))

        for seed in SEEDS:
            run = _run(generator, prompt, sampling_mode="device", seeds=[seed], label=f"device-seed{seed}")
            run["token_input_shards_equal"] = _all_equal(_shards(generator.state.token_input))
            record(run)
        record(_run(generator, prompt, sampling_mode="device", seeds=None, label="device-unseeded"))
        record(_run(generator, prompt, sampling_mode="host", seeds=[SEEDS[0]], label="host-control"))
        for key, value in report["shards_equal"].items():
            record_property(f"shards_equal_{key}", value)
        host = next(run for run in report["runs"] if run["sampling_mode"] == "host")
        device_runs = [run for run in report["runs"] if run["sampling_mode"] == "device"]
        assert all(report["shards_equal"].values()), report["shards_equal"]
        for run in device_runs:
            assert run["generated_tokens"] == NEW_TOKENS, run
            assert run["seed_host_copies_delta"] >= NEW_TOKENS - 1, run
            assert run.get("token_input_shards_equal", True), run
            assert not run["review"]["mechanically_degenerate"], run
            assert run["four_gram_ratio"] <= max(2.0 * host["four_gram_ratio"], 0.05), (run["four_gram_ratio"], host["four_gram_ratio"])
    finally:
        generator.close()
