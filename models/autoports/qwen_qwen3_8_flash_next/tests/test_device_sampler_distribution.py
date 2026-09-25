# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Distribution gate for the on-device stochastic sampler (DEVSTACK-294).

The served model samples its default requests (temperature 1.0, top_k 20,
top_p 0.95) through ``Sampling1D`` -> ``ttnn.sampling``.  Every shipped quality
gate ran greedy, which bypasses that path, and the only test of the stochastic
path checked set membership.  This test draws thousands of tokens from fixed
logit rows on the same (4,1) mesh and vocabulary the model uses and compares
the empirical distribution with the top-k -> top-p -> softmax reference that
vLLM's host sampler implements (HuggingFace warper semantics).

Run: ``RUN_QWEN38_SAMPLER_GATE=1 pytest -s tests/test_device_sampler_distribution.py``
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.test_full_model import _device_params
from models.autoports.qwen_qwen3_8_flash_next.tt.model import VOCAB_SIZE
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import COLLECTIVE_NUM_LINKS, MultichipDecoder
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig

EVIDENCE = Path(os.getenv("QWEN38_EVIDENCE_DIR", str(Path(__file__).parents[1] / "doc/sampling_devstack294")))
BATCH = 32
CALLS = int(os.getenv("QWEN38_SAMPLER_GATE_CALLS", "256"))  # x 32 rows = draws per case
CANDIDATES = 20


def _candidate_probabilities(kind: str) -> torch.Tensor:
    if kind == "lm_like":
        tail = torch.tensor([0.75**i for i in range(CANDIDATES - 1)])
        probs = torch.cat([torch.tensor([0.40]), 0.60 * tail / tail.sum()])
    elif kind == "peaked":
        # distinct tail probabilities: with ties the top-p boundary is a sort-order
        # tie-break and "outside the nucleus" would be meaningless
        tail = torch.tensor([0.8**i for i in range(CANDIDATES - 1)])
        probs = torch.cat([torch.tensor([0.90]), 0.10 * tail / tail.sum()])
    elif kind == "flat":
        probs = torch.full((CANDIDATES,), 1.0 / CANDIDATES)
    else:
        raise ValueError(kind)
    return probs / probs.sum()


def _logit_row(kind: str, vocab: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A vocab-wide logit row whose 20 candidates sit at random ids (they land on different shards)."""

    probs = _candidate_probabilities(kind)
    ids = torch.randperm(vocab, generator=generator)[:CANDIDATES]
    row = torch.full((vocab,), -40.0)
    row[ids] = torch.log(probs) + 10.0
    # bf16 is what the device sees; build the reference from the same rounding
    return row.bfloat16(), ids


def _reference_distribution(row_bf16: torch.Tensor, *, k: int, p: float, temperature: float) -> torch.Tensor:
    """vLLM / HuggingFace semantics: temperature -> top-k -> top-p (keep first past p) -> softmax."""

    scores = row_bf16.float() / temperature
    topk = torch.topk(scores, k)
    kept = torch.full_like(scores, float("-inf"))
    kept[topk.indices] = topk.values
    if 0.0 < p < 1.0:
        sorted_scores, sorted_idx = torch.sort(kept, descending=True)
        probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = (cumulative - probs) > p  # keep every token whose cumulative mass starts below p
        kept[sorted_idx[remove]] = float("-inf")
    return torch.softmax(kept, dim=-1)


def _make_sampler(mesh_device, vocab: int) -> Sampling1D:
    sampler = Sampling1D.from_config(
        Sampling1DConfig(
            vocab_size=vocab,
            valid_vocab_size=vocab,
            mesh_device=mesh_device,
            max_batch_size=BATCH,
            max_top_k=32,
            num_gather_links=COLLECTIVE_NUM_LINKS,
            sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            allow_force_argmax=True,
            pad_to_power_of_2=True,
        )
    )
    sampler.load_device_buffers()
    return sampler


def _replicated(mesh_device, values: torch.Tensor, dtype):
    return ttnn.from_torch(
        values,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _vocab_sharded_logits(mesh_device, rows: torch.Tensor):
    mesh_shape = tuple(mesh_device.shape)
    dims = (-1, None) if mesh_shape[0] >= mesh_shape[1] else (None, -1)
    return ttnn.from_torch(
        rows.reshape(1, 1, BATCH, -1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape),
    )


def _shards_equal(tensor) -> bool:
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
    return all(torch.equal(shards[0], other) for other in shards[1:])


def _chi_square_p_value(observed: torch.Tensor, expected: torch.Tensor) -> float | None:
    try:
        from scipy.stats import chisquare
    except Exception:  # pragma: no cover - scipy optional
        return None
    mask = expected > 0
    observed = observed[mask].double()
    expected = expected[mask].double()
    expected = expected * (observed.sum() / expected.sum())  # scipy requires identical totals
    return float(chisquare(observed.numpy(), expected.numpy()).pvalue)


@pytest.mark.skipif(os.getenv("RUN_QWEN38_SAMPLER_GATE") != "1", reason="device sampler distribution gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_device_sampler_distribution_matches_reference(bh_1d_mesh_device, device_params, record_property):
    del device_params
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh = bh_1d_mesh_device
    cases = [
        # (vocab, distribution, k, p, temperature, TVD limit, gated).  The flat row has 20
        # near-equal probabilities, so which token the p=0.95 cut removes is a tie-break and
        # its TVD is computed on the sorted frequency profile instead of by token id.  It is
        # reported but not gated: the sampler draws one bf16 uniform per row and cuts bf16
        # probabilities (writer_interleaved.cpp), which quantizes a 20-way flat CDF to
        # ~1/256 steps and leaves a ~0.05 TVD that no exp precision removes (DEVSTACK-294).
        (VOCAB_SIZE, "lm_like", 20, 0.95, 1.0, 0.03, True),
        (VOCAB_SIZE, "peaked", 20, 0.95, 1.0, 0.03, True),
        (VOCAB_SIZE, "flat", 20, 0.95, 1.0, 0.03, False),
        (VOCAB_SIZE, "lm_like", 20, 1.0, 1.0, 0.03, True),  # top-k only: isolates the softmax
        (VOCAB_SIZE, "lm_like", 32, 1.0, 1.0, 0.03, True),
        (4096, "lm_like", 20, 0.95, 1.0, 0.03, True),
    ]
    # Draws outside the reference nucleus: the fraction of tokens the host sampler
    # would never emit (bf16 probabilities widen the device's cumulative cut).
    OUTSIDE_LIMIT = 0.005
    report: dict[str, object] = {"draws_per_case": CALLS * BATCH, "cases": []}
    failures = []
    ungated_failures = []
    samplers: dict[int, Sampling1D] = {}
    generator = torch.Generator().manual_seed(2940)
    for vocab, kind, k, p, temperature, tvd_limit, gated in cases:
        sampler = samplers.setdefault(vocab, _make_sampler(mesh, vocab))
        row, ids = _logit_row(kind, vocab, generator)
        reference = _reference_distribution(row, k=k, p=p, temperature=temperature)
        logits = _vocab_sharded_logits(mesh, row.unsqueeze(0).repeat(BATCH, 1))
        k_tt = _replicated(mesh, torch.full((BATCH,), k, dtype=torch.int32), ttnn.uint32)
        p_tt = _replicated(mesh, torch.full((BATCH,), p, dtype=torch.float32), ttnn.bfloat16)
        temp_tt = _replicated(mesh, torch.full((BATCH,), 1.0 / temperature, dtype=torch.float32), ttnn.bfloat16)
        counts = torch.zeros(vocab, dtype=torch.float64)
        shards_equal = True
        started = time.perf_counter()
        for call in range(CALLS):
            seeds = torch.randint(1, 0x7FFFFFFE, (BATCH,), generator=generator, dtype=torch.int32)
            seed_tt = _replicated(mesh, seeds, ttnn.uint32)
            tokens, _ = sampler.decode_forward(logits, k=k_tt, p=p_tt, temp=temp_tt, seeds=seed_tt)
            if call == 0:
                shards_equal = _shards_equal(tokens)
            drawn = ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).reshape(-1)[:BATCH].to(torch.int64)
            counts.index_add_(0, drawn, torch.ones(BATCH, dtype=torch.float64))
            ttnn.deallocate(tokens)
            ttnn.deallocate(seed_tt)
        seconds = time.perf_counter() - started
        draws = counts.sum().item()
        empirical = counts / draws
        support = reference > 0
        def case_tvd(freqs: torch.Tensor) -> float:
            if kind == "flat":
                emp_sorted = torch.sort(freqs[ids], descending=True).values
                ref_sorted = torch.sort(reference.double()[ids], descending=True).values
                return 0.5 * float((emp_sorted - ref_sorted).abs().sum() + freqs[~support].sum())
            return 0.5 * float((freqs - reference.double()).abs().sum())

        tvd = case_tvd(empirical)
        # Sampling-noise floor: the same metric on exact draws from the reference.
        noise = []
        for _ in range(8):
            resample = torch.multinomial(reference.double(), int(draws), replacement=True, generator=generator)
            noise.append(case_tvd(torch.bincount(resample, minlength=vocab).double() / draws))
        tvd_noise_floor = sum(noise) / len(noise)
        outside = float(counts[~support].sum())
        argmax_id = int(torch.argmax(reference))
        tail_ref = float(reference[ids[5:]].sum())
        tail_emp = float(empirical[ids[5:]].sum())
        p_value = _chi_square_p_value(counts[support], reference.double()[support] * draws)
        distinct = int((counts > 0).sum())
        case = {
            "vocab": vocab,
            "distribution": kind,
            "k": k,
            "p": p,
            "temperature": temperature,
            "draws": int(draws),
            "seconds": seconds,
            "tvd": tvd,
            "tvd_limit": tvd_limit,
            "tvd_noise_floor": tvd_noise_floor,
            "tvd_excess_over_noise": tvd - tvd_noise_floor,
            "gated": gated,
            "chi_square_p_value": p_value,
            "outside_support_draws": outside,
            "outside_support_fraction": outside / draws,
            "outside_support_limit": OUTSIDE_LIMIT,
            "reference_support_size": int(support.sum()),
            "argmax_rate_device": float(empirical[argmax_id]),
            "argmax_rate_reference": float(reference[argmax_id]),
            "tail_mass_ratio_ranks6_20": (tail_emp / tail_ref) if tail_ref else None,
            "distinct_tokens": distinct,
            "shards_equal_first_call": shards_equal,
            "reference_probs": [round(float(reference[i]), 5) for i in ids.tolist()],
            "device_freqs": [round(float(empirical[i]), 5) for i in ids.tolist()],
        }
        report["cases"].append(case)
        print({"sampler_distribution": {key: value for key, value in case.items() if key not in ("reference_probs", "device_freqs")}})
        # chi-square is reported, not gated: bf16 probabilities make it reject at 8k draws
        # even when the total-variation distance is ~1-2 %.
        tie_free = kind != "flat"
        ok = tvd <= tvd_limit and (not tie_free or outside / draws <= OUTSIDE_LIMIT) and shards_equal
        if not ok:
            (failures if gated else ungated_failures).append(case)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    mesh_tag = "x".join(str(v) for v in tuple(mesh.shape))
    (EVIDENCE / f"sampler_distribution_{mesh_tag}.json").write_text(json.dumps(report, indent=2))
    record_property("sampler_distribution_cases", len(report["cases"]))
    record_property("sampler_distribution_failures", len(failures))
    record_property("sampler_distribution_ungated_failures", len(ungated_failures))
    assert not failures, json.dumps(
        [{key: case[key] for key in ("distribution", "k", "p", "tvd", "tvd_limit", "chi_square_p_value", "outside_support_fraction", "outside_support_limit", "shards_equal_first_call")} for case in failures],
        indent=2,
    )


@pytest.mark.skipif(os.getenv("RUN_QWEN38_SAMPLER_GATE") != "1", reason="device sampler distribution gate")
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_device_sampler_repeats_with_a_constant_seed(bh_1d_mesh_device, device_params):
    """A constant seed buffer reproduces the same draw every call.

    This is the mechanism behind the fixed-quantile decode that unseeded
    stochastic generation showed before ``set_sampling_params`` synthesized
    seed streams: the seed buffer must change between decode steps.
    """

    del device_params
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    mesh = bh_1d_mesh_device
    sampler = _make_sampler(mesh, 4096)
    generator = torch.Generator().manual_seed(2941)
    row, _ = _logit_row("flat", 4096, generator)
    logits = _vocab_sharded_logits(mesh, row.unsqueeze(0).repeat(BATCH, 1))
    k_tt = _replicated(mesh, torch.full((BATCH,), 20, dtype=torch.int32), ttnn.uint32)
    p_tt = _replicated(mesh, torch.full((BATCH,), 0.95), ttnn.bfloat16)
    temp_tt = _replicated(mesh, torch.ones(BATCH), ttnn.bfloat16)
    constant = _replicated(mesh, torch.arange(BATCH, dtype=torch.int32), ttnn.uint32)
    draws = []
    for _ in range(8):
        tokens, _ = sampler.decode_forward(logits, k=k_tt, p=p_tt, temp=temp_tt, seeds=constant)
        draws.append(ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).reshape(-1)[:BATCH].clone())
        ttnn.deallocate(tokens)
    assert all(torch.equal(draws[0], other) for other in draws[1:]), "constant seeds should reproduce the draw"
    print({"constant_seed_repeats_identical": True, "row0_token": int(draws[0][0])})
