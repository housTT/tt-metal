# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Repeat the same greedy request through the serving path. Same slot, other slot, other batch.

This replaces ``logit_determinism.json``, whose every arm called
``prefill_requests_into_slots(page_table=..., kv_cache=None)``. ``_resolve_page_table`` substituted the
generator's own page table for that call, which in that probe was **all zeros**, so every logical block
of every prompt was written to physical block 0 and the decode step then read the blocks the caller's
table named - blocks nothing had written.
``prefill_determinism_bisect.json`` isolates that, and this probe re-measures the same questions with
the caller's cache passed on every call, which is what the vLLM adapter has always done
(``generator_vllm.py`` prefill/decode both pass ``kv_cache=``).

All arms are greedy, all go through the low-level serving path (traced decode replay, device sampling
turned off only so the logits are readable, the token fed to the next step being this step's greedy
pick - which is what the on-device sampler writes):

* ``batch1_rerun``        - batch 1, the same prompt twice from a wiped state. The reproducibility the
  vLLM suite asserts at ``--max-num-seqs 1``;
* ``batch32_same_slot``   - batch 32, the same prompt into **slot 0** twice from a wiped state. This is
  the same request repeated with the decode geometry vLLM uses at ``--max-num-seqs 32``;
* ``batch32_same_slot_second_pair`` - runs 2 and 3 of the same thing, which separates "the first run
  leaves residual state" from "every run is independently noisy";
* ``batch32_after_ghost`` - slot 0 again, but a previous request has run in slot 5 first, so the idle
  padding rows carry another request's state and tokens. This is what a repeat looks like on a live
  server, where the padding rows are whatever the last requests left;
* ``batch32_other_slot``  - the same prompt in **slot 7**. vLLM does not promise a request the same
  slot on a later run, so this is the comparison a server-level reproducibility assertion is really
  making;
* ``batch1_vs_batch32``   - batch 1 against batch 32, slot 0;
* ``batch32_read_twice``  - one decode step at batch 32 whose device logits are composed to host
  **twice**. The read path has to be controlled *where* the deviation lives, not only at batch 1;
* ``batch32_all_rows_rerun`` - batch 32 with **every one of the 32 rows prefilled and decoding**, twice
  from a wiped state. If the run-to-run deviation comes from idle rows carrying uninitialised state (no
  prefill has ever written their DeltaNet buffers or their KV pages), it should disappear here;
* ``batch{2,4,8,16}_rerun`` - the same prompt twice at each batch size, so the batch at which
  run-to-run bit-identity is lost is located rather than assumed;
* ``batch1_state_rerun`` / ``batch32_state_rerun`` / ``batch32_state_second_pair`` - the same prompt
  prefilled twice (three times at batch 32) with **no decode step**, comparing every DeltaNet
  recurrent/conv buffer of the decode state pack **and the pages of the paged KV cache the request
  writes**, plus the prefill logits. The prefill logits are
  produced before ``_merge_prefill_state_into_slot`` copies the batch-1 prefill state into the decode
  slot, and that merge is a no-op at batch 1, so "logits identical, state not" points at the merge and
  "both identical" clears it.

Index 0 of every arm is the **prefill**'s last-position logits and the rest are the decode steps, so
``first_top1_divergence == 0`` means the prefill already differed and a later index means the decode
introduced it.

Per comparison: max/mean |Δ|, logit PCC, whether top-1 agrees, and the top-1-to-top-2 margin, which is
what decides whether a difference of that size can flip a token. PCC makes these numbers directly
comparable to ``doc/datatype_sweep/README.md`` §9.1, which measured cross-slot decode at max |Δ|
0.28-0.5 / PCC >= 0.9993 / top-1 margins 0.0-0.19 and classified it as reduction-order noise rather
than a state bug.

A reader should conclude: whether repeating one request is bit-identical when the slot is held fixed, at
which batch size that stops being true, whether the read path or the idle rows explain it, and whether
what remains is the same reduction-order effect §9.1 accepted or something larger. Note that §9.1's own
conclusion is about *cross-slot* determinism within a run ("the same slot flips on all three repeats"),
which is a different axis from run-to-run bit-identity: it neither predicts nor excludes what these arms
measure.

    python .../doc/vllm_integration/logs/probe_slot_reproducibility.py --layer-indices all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import OrnithGenerator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import (
    OrnithModel,
    close_ornith_mesh,
    load_text_config,
    open_ornith_mesh,
    resolve_model_path,
)
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import num_blocks_for_context

MODEL_DIR = Path(__file__).resolve().parents[3]
PROBE_LAYERS = [0, 3]
PROMPT = [791, 6864, 315, 9822, 374]
GHOST_PROMPT = [8144, 264, 6520, 39342, 922, 5780, 6975, 13]


def pcc(a, b):
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    if torch.equal(x, y):
        return 1.0
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def build(mesh, batch, layers, context):
    path = resolve_model_path()
    model = OrnithModel.from_pretrained(
        path, mesh_device=mesh, hf_config=load_text_config(path), layer_indices=layers, max_context=context
    )
    blocks_per_user = num_blocks_for_context(context, model.page_block_size)
    # Block 0 is left out of every request's table on purpose: it is the null block a serving warm-up
    # writes into, so a request that accidentally addressed it would read warm-up KV.
    kv_cache = model.allocate_kv_cache(1 + batch * blocks_per_user)
    table = torch.zeros(batch, blocks_per_user, dtype=torch.int32)
    for user in range(batch):
        base = 1 + user * blocks_per_user
        table[user] = torch.arange(base, base + blocks_per_user, dtype=torch.int32)
    generator = OrnithGenerator(
        model,
        max_batch_size=batch,
        cache_context=context,
        sampling_mode="device",
        kv_cache=kv_cache,
        page_table=table,
    )
    generator.ensure_serving_traces()
    generator.ensure_sampling_trace()
    return model, generator, table, kv_cache


def state_fingerprint(model, batch):
    """Every DeltaNet recurrent/conv buffer of the ``batch`` state pack, on host.

    Read straight out of ``model._packs``, so nothing is switched or written: the packs hold the same
    tensors the layers point at. Row 0 of ``recurrent_state`` is decode slot 0's.
    """
    out = []
    for pack in model._packs[batch]:
        buffers = [pack["recurrent_state"], *(pack["conv_state"] or ())]
        for buf in buffers:
            if buf is None:
                continue
            out.append(ttnn.to_torch(ttnn.get_device_tensors(buf)[0]).float())
    return out


def kv_fingerprint(model, blocks: int = 2):
    """The first ``blocks`` pages of every layer's K and V, on host.

    Only the pages the probe's request actually writes are read: the whole paged cache at batch 32 is
    hundreds of megabytes. This closes the "identical prefill logits do not prove identical cache
    writes" gap - for a single-chunk prefill the attention can read its own local K/V rather than the
    pages it fills, so the fill has to be compared directly.
    """
    out = []
    if model.kv_cache is None:
        return out
    for entry in model.kv_cache:
        for tensor in entry:
            shape = [int(d) for d in tensor.shape]
            piece = ttnn.slice(tensor, [0] * len(shape), [min(blocks, shape[0])] + shape[1:])
            out.append(ttnn.to_torch(ttnn.get_device_tensors(piece)[0]).float())
            ttnn.deallocate(piece)
    return out


def compare_state(left, right, batch):
    """Compare two state fingerprints, keeping non-finite entries separate from real differences.

    Rows that no prefill wrote are still carried through every batched op — the recurrent update is a
    dense matmul over all ``batch`` rows, and an inactive row (token 0, position -1) can accumulate
    ``inf``/``NaN`` there. Comparing those with plain arithmetic poisons the summary twice over: ``inf -
    inf`` is ``NaN``, ``NaN != 0`` marks the row as differing, and one ``inf`` makes ``max_abs_diff``
    infinite regardless of what the finite entries did. So this reports both readings:

    * ``bitwise_identical`` — ``torch.equal``, which is NaN-strict (two ``NaN`` rows count as different);
    * ``finite_max_abs_diff`` and ``rows_that_differ`` — computed only where **both** sides are finite,
      which is the arithmetic question;
    * ``nonfinite_*`` — how much of each side is non-finite and whether the two sides put it in the same
      places, so a non-finite pattern that is itself stable is visible as such.
    """
    identical = True
    worst = 0.0
    rows = set()
    nonfinite_left = nonfinite_right = total = 0
    nonfinite_mask_identical = True
    nonfinite_rows = set()
    for a, b in zip(left, right):
        finite_a, finite_b = torch.isfinite(a), torch.isfinite(b)
        nonfinite_left += int((~finite_a).sum())
        nonfinite_right += int((~finite_b).sum())
        total += a.numel()
        if not torch.equal(finite_a, finite_b):
            nonfinite_mask_identical = False
        if not torch.equal(a, b):
            identical = False
        both = finite_a & finite_b
        if a.dim() >= 1 and a.shape[0] == batch:
            bad_rows = (~finite_a).reshape(batch, -1).any(dim=1) | (~finite_b).reshape(batch, -1).any(dim=1)
            nonfinite_rows.update(int(i) for i in bad_rows.nonzero().reshape(-1))
        if not bool(both.any()):
            continue
        diff = torch.where(both, (a - b).abs(), torch.zeros_like(a))
        worst = max(worst, float(diff.max()))
        if a.dim() >= 1 and a.shape[0] == batch:
            per_row = diff.reshape(batch, -1).max(dim=1).values
            rows.update(int(i) for i in (per_row > 0).nonzero().reshape(-1))
    return {
        "bitwise_identical": identical,
        "buffers": len(left),
        "finite_max_abs_diff": worst,
        "rows_that_differ": sorted(rows),
        "nonfinite_fraction_left": nonfinite_left / max(1, total),
        "nonfinite_fraction_right": nonfinite_right / max(1, total),
        "nonfinite_in_same_places": nonfinite_mask_identical,
        "rows_with_nonfinite_entries": sorted(nonfinite_rows),
    }


def prefill_only(generator, table, kv_cache, prompt_ids, slot):
    """One reset + prefill, no decode step, returning the prefill's last-position logits."""
    generator.reset()
    out = generator.prefill_requests_into_slots(
        torch.tensor(prompt_ids, dtype=torch.int64).reshape(1, -1),
        [len(prompt_ids)],
        [slot],
        page_table=table[slot : slot + 1],
        kv_cache=kv_cache,
        sample_on_device=False,
    )
    return out[0, -1].clone()


def run_request(generator, table, kv_cache, prompt_ids, slot, steps):
    """Prefill into ``slot``, then take ``steps`` traced decode steps.

    Returns ``[prefill_logits] + per_decode_step_logits`` so a comparison covers the prefill as well:
    a divergence that is already present at index 0 is the prefill's, and one that starts later is the
    decode's.
    """
    prefill_logits = generator.prefill_requests_into_slots(
        torch.tensor(prompt_ids, dtype=torch.int64).reshape(1, -1),
        [len(prompt_ids)],
        [slot],
        page_table=table[slot : slot + 1],
        kv_cache=kv_cache,
        sample_on_device=False,
    )
    batch = generator.max_batch_size
    tokens = torch.zeros(batch, dtype=torch.int64)
    positions = torch.full((batch,), -1, dtype=torch.int64)
    tokens[slot] = int(torch.argmax(prefill_logits[0, -1]).item())
    positions[slot] = len(prompt_ids)
    per_step = [prefill_logits[0, -1].clone()]
    for _ in range(steps):
        generator.stage_serving_decode_inputs(tokens, positions, table, full_refresh=True)
        device_logits = generator.submit_serving_decode(sample_on_device=False)
        host = generator.logits_from(device_logits)[slot, 0]
        per_step.append(host.clone())
        tokens[slot] = int(torch.argmax(host).item())
        positions[slot] = int(positions[slot]) + 1
    return per_step


def read_twice(generator, table, kv_cache, prompt_ids, slot):
    """One prefill + one traced decode step, with the step's device logits composed to host twice.

    Nothing runs on the device between the two reads, so a difference here is the read path and a match
    means the deviation the rerun arms see is in the computation. This is the batch-32 counterpart of
    ``logit_read_stability.json``'s ``read_twice``, which only ever ran at batch 1.
    """
    generator.reset()
    prefill_logits = generator.prefill_requests_into_slots(
        torch.tensor(prompt_ids, dtype=torch.int64).reshape(1, -1),
        [len(prompt_ids)],
        [slot],
        page_table=table[slot : slot + 1],
        kv_cache=kv_cache,
        sample_on_device=False,
    )
    batch = generator.max_batch_size
    tokens = torch.zeros(batch, dtype=torch.int64)
    positions = torch.full((batch,), -1, dtype=torch.int64)
    tokens[slot] = int(torch.argmax(prefill_logits[0, -1]).item())
    positions[slot] = len(prompt_ids)
    generator.stage_serving_decode_inputs(tokens, positions, table, full_refresh=True)
    device_logits = generator.submit_serving_decode(sample_on_device=False)
    first = generator.logits_from(device_logits)[slot, 0].clone()
    second = generator.logits_from(device_logits)[slot, 0].clone()
    return [first], [second]


def run_all_rows(generator, table, kv_cache, prompt_ids, steps, watch_slot=0):
    """Prefill **every** decode row, then decode with every row active, watching one row's logits.

    The padded-batch arms above leave 31 rows that no prefill ever wrote: their DeltaNet buffers hold
    whatever ``reset()`` zeroed and their KV pages were never filled, so the decode graph computes them
    from a zero state. This arm removes that difference - every row holds a real prompt at a real
    position - so a deviation that survives it is not about uninitialised rows.
    """
    generator.reset()
    batch = generator.max_batch_size
    row = torch.tensor(prompt_ids, dtype=torch.int64).reshape(1, -1)
    tokens = torch.zeros(batch, dtype=torch.int64)
    positions = torch.full((batch,), -1, dtype=torch.int64)
    for slot in range(batch):
        out = generator.prefill_requests_into_slots(
            row,
            [len(prompt_ids)],
            [slot],
            page_table=table[slot : slot + 1],
            kv_cache=kv_cache,
            sample_on_device=False,
        )
        tokens[slot] = int(torch.argmax(out[0, -1]).item())
        positions[slot] = len(prompt_ids)
        if slot == watch_slot:
            watched = [out[0, -1].clone()]
    for _ in range(steps):
        generator.stage_serving_decode_inputs(tokens, positions, table, full_refresh=True)
        device_logits = generator.submit_serving_decode(sample_on_device=False)
        host = generator.logits_from(device_logits)
        watched.append(host[watch_slot, 0].clone())
        for slot in range(batch):
            tokens[slot] = int(torch.argmax(host[slot]).item())
            positions[slot] = int(positions[slot]) + 1
    return watched


def compare(left, right):
    steps = []
    for a, b in zip(left, right):
        diff = (a - b).abs()
        top_a = torch.topk(a, 2)
        top_b = torch.topk(b, 2)
        steps.append(
            {
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "pcc": pcc(a, b),
                "bitwise_identical": bool(torch.equal(a, b)),
                "top1_left": int(top_a.indices[0]),
                "top1_right": int(top_b.indices[0]),
                "top1_agrees": int(top_a.indices[0]) == int(top_b.indices[0]),
                "top1_margin_left": float(top_a.values[0] - top_a.values[1]),
            }
        )
    return {
        "steps": steps,
        "all_bitwise_identical": all(s["bitwise_identical"] for s in steps),
        "top1_agreement": sum(s["top1_agrees"] for s in steps) / max(1, len(steps)),
        "max_abs_diff": max((s["max_abs_diff"] for s in steps), default=None),
        "min_pcc": min((s["pcc"] for s in steps), default=None),
        "median_top1_margin": float(torch.tensor([s["top1_margin_left"] for s in steps]).median()) if steps else None,
        "first_top1_divergence": next((i for i, s in enumerate(steps) if not s["top1_agrees"]), None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--layer-indices", default=",".join(str(v) for v in PROBE_LAYERS))
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--sweep",
        type=lambda v: [int(x) for x in v.split(",") if x],
        default=[2, 4, 8, 16],
        help="batch sizes to locate where run-to-run bit-identity is lost (1 and --batch are already covered)",
    )
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "vllm_integration" / "slot_reproducibility.json"))
    args = ap.parse_args()
    layers = None if args.layer_indices.strip() == "all" else [int(v) for v in args.layer_indices.split(",")]
    report = {
        "layers": layers or "all",
        "steps": args.steps,
        "context": args.context,
        "batch": args.batch,
        "prompt": PROMPT,
        "complete": False,
    }

    def checkpoint():
        """Persist what is measured so far.

        Arms are minutes apart on the full model, and a crash in a later arm must not discard the
        earlier ones. ``complete`` stays False until the run reaches the end, so a partial file is
        never mistaken for a finished one.
        """
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")

    mesh = open_ornith_mesh()
    try:
        model, generator, table, kv_cache = build(mesh, 1, layers, args.context)
        # Prefill-only rounds first: is the state the decode graph will read bit-identical?
        logits_a = prefill_only(generator, table, kv_cache, PROMPT, 0)
        state_a = state_fingerprint(model, 1) + kv_fingerprint(model)
        logits_b = prefill_only(generator, table, kv_cache, PROMPT, 0)
        state_b = state_fingerprint(model, 1) + kv_fingerprint(model)
        report["batch1_prefill_logits_identical"] = bool(torch.equal(logits_a, logits_b))
        report["batch1_state_rerun"] = compare_state(state_a, state_b, 1)
        generator.reset()
        first = run_request(generator, table, kv_cache, PROMPT, 0, args.steps)
        generator.reset()
        second = run_request(generator, table, kv_cache, PROMPT, 0, args.steps)
        report["batch1_rerun"] = compare(first, second)
        report["batch1_substitutions_warned"] = bool(generator._warned_page_table_substitution)
        checkpoint()
        batch1_reference = first
        generator.teardown()
        del generator, model
    finally:
        close_ornith_mesh(mesh)

    mesh = open_ornith_mesh()
    try:
        model, generator, table, kv_cache = build(mesh, args.batch, layers, args.context)
        # Three prefill-only rounds. The prefill *logits* are produced before
        # `_merge_prefill_state_into_slot` runs, so identical logits with a differing state pack
        # localise the divergence to the merge (a batch>1-only operation) rather than to the prefill.
        logits_a = prefill_only(generator, table, kv_cache, PROMPT, 0)
        state_a = state_fingerprint(model, args.batch) + kv_fingerprint(model)
        logits_b = prefill_only(generator, table, kv_cache, PROMPT, 0)
        state_b = state_fingerprint(model, args.batch) + kv_fingerprint(model)
        logits_c = prefill_only(generator, table, kv_cache, PROMPT, 0)
        state_c = state_fingerprint(model, args.batch) + kv_fingerprint(model)
        report["batch32_prefill_logits_identical"] = [
            bool(torch.equal(logits_a, logits_b)),
            bool(torch.equal(logits_b, logits_c)),
        ]
        report["batch32_state_rerun"] = compare_state(state_a, state_b, args.batch)
        report["batch32_state_second_pair"] = compare_state(state_b, state_c, args.batch)
        checkpoint()
        generator.reset()
        clean_a = run_request(generator, table, kv_cache, PROMPT, 0, args.steps)
        generator.reset()
        clean_b = run_request(generator, table, kv_cache, PROMPT, 0, args.steps)
        report["batch32_same_slot"] = compare(clean_a, clean_b)
        generator.reset()
        clean_c = run_request(generator, table, kv_cache, PROMPT, 0, args.steps)
        # Third clean run: if run 1 -> 2 differs but 2 -> 3 does not, the first run leaves residual
        # state behind rather than every run being independently noisy.
        report["batch32_same_slot_second_pair"] = compare(clean_b, clean_c)
        checkpoint()

        generator.reset()
        run_request(generator, table, kv_cache, GHOST_PROMPT, 5, args.steps)
        after_ghost = run_request(generator, table, kv_cache, PROMPT, 0, args.steps)
        report["batch32_after_ghost"] = compare(clean_a, after_ghost)

        generator.reset()
        other_slot = run_request(generator, table, kv_cache, PROMPT, 7, args.steps)
        report["batch32_other_slot"] = compare(clean_a, other_slot)
        checkpoint()
        report["batch1_vs_batch32"] = compare(batch1_reference, clean_a)

        # The read path, at the batch where the deviation lives.
        first_read, second_read = read_twice(generator, table, kv_cache, PROMPT, 0)
        report["batch32_read_twice"] = compare(first_read, second_read)
        checkpoint()

        # Every row occupied, so no row decodes from a state no prefill ever wrote.
        all_rows_a = run_all_rows(generator, table, kv_cache, PROMPT, args.steps)
        all_rows_b = run_all_rows(generator, table, kv_cache, PROMPT, args.steps)
        report["batch32_all_rows_rerun"] = compare(all_rows_a, all_rows_b)
        checkpoint()

        report["counters"] = dict(generator.counters)
        report["trace_recaptures"] = generator.trace_recaptures
        report["batch32_substitutions_warned"] = bool(generator._warned_page_table_substitution)
        generator.teardown()
        del generator

        # Where does run-to-run bit-identity stop? One fresh generator per batch size, same prompt,
        # same slot, twice from a wiped state. The model (and its weights) are reused; only the state
        # pack, the sampler and the traces are rebuilt, which is what a batch size changes.
        sweep = {}
        for batch in args.sweep:
            if batch in (1, args.batch):
                continue  # already measured, as batch1_rerun / batch{args.batch}_same_slot
            logger.info(f"batch sweep: rebuilding the generator at batch {batch}")
            blocks_per_user = num_blocks_for_context(args.context, model.page_block_size)
            sweep_table = torch.zeros(batch, blocks_per_user, dtype=torch.int32)
            for user in range(batch):
                base = 1 + user * blocks_per_user
                sweep_table[user] = torch.arange(base, base + blocks_per_user, dtype=torch.int32)
            sweep_generator = OrnithGenerator(
                model,
                max_batch_size=batch,
                cache_context=args.context,
                sampling_mode="device",
                kv_cache=kv_cache,
                page_table=sweep_table,
            )
            # `state_is_live` is a *model* flag, and the arms above left a prompt in it: capture over
            # live state is refused (it would wipe the prompt it is about to warm against). Wipe
            # first, which is what `generate(reset=True)` does for the same reason.
            sweep_generator.reset()
            sweep_generator.ensure_serving_traces()
            sweep_generator.ensure_sampling_trace()
            try:
                sweep_generator.reset()
                first = run_request(sweep_generator, sweep_table, kv_cache, PROMPT, 0, args.steps)
                sweep_generator.reset()
                second = run_request(sweep_generator, sweep_table, kv_cache, PROMPT, 0, args.steps)
                sweep[f"batch{batch}_rerun"] = compare(first, second)
            finally:
                sweep_generator.teardown()
                del sweep_generator
            report.update(sweep)
            checkpoint()
    finally:
        close_ornith_mesh(mesh)

    report["summary"] = {
        key: {
            "all_bitwise_identical": value["all_bitwise_identical"],
            "top1_agreement": value["top1_agreement"],
            "max_abs_diff": value["max_abs_diff"],
            "min_pcc": value["min_pcc"],
            "median_top1_margin": value["median_top1_margin"],
            "first_top1_divergence": value["first_top1_divergence"],
        }
        for key, value in report.items()
        if isinstance(value, dict) and "all_bitwise_identical" in value
    }
    report["complete"] = True
    checkpoint()
    logger.info(f"wrote {args.output}")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
