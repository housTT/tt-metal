# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The serving primitives, on the reduced two-layer target, without vLLM in the way.

Every arm is one thing the vLLM plugin does to this adapter, with the control that makes the answer
observable:

1. **per-slot prefill** - two prompts prefilled into two slots that are *not* their row order, which
   is what ``prefill_empty_slots`` sends whenever an off-batch request still owns a row;
2. **steady state** - after the first (reset) step, later steps copy **nothing** to the device: the
   token arrives through ``tt_out_tok`` and the position through ``ttnn.plus_one``, inside the trace;
3. **stale host inputs** - three arms over the same prompt and the same slot. ``fresh`` stages the
   correct pair every step; ``stale-merged`` stages the one-token-behind pair vLLM produces under
   async scheduling and marks the row device-authoritative; ``stale-host`` stages the same stale pair
   and lets the host win, which is the bug the merge exists to prevent. ``fresh`` and
   ``stale-merged`` must agree token for token, and ``stale-host`` must not. The step right after a
   prefill is never lagged and never trusts the device: that row's last token came from prefill
   sampling, which writes a scratch buffer, so the decode token buffer is stale for it by
   construction - which is exactly what the adapter's ``_prefilled_rows`` mask encodes;
4. **page-table-only refresh** - a grown page table is copied while tokens and positions are not;
5. **batch-layout change** - a request prefilled into another slot mid-stream leaves the continuing
   row's recurrent state bit-identical, and the continuing row's device token/position pair advances
   by exactly one step across the reset;
6. **slot remap** - a condense moves a request's recurrent state to another row, bit-identically, and
   leaves the untouched rows alone.

Token *streams* are recorded but not compared across arms whose batch content differs: at batch > 1 a
neighbouring row changes the MoE expert union and therefore the last bits of every row's logits, so a
near-tie can flip. ``doc/full_model/batch_slots.json`` measured that property; the mechanical
assertions above are the ones that isolate this stage's own contracts.

    python .../doc/vllm_integration/logs/probe_serving_primitives.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

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
CONTEXT = 4096
BATCH = 4
PROMPT_A = [6, 66, 666, 6666, 66]
PROMPT_B = [9, 99, 999, 9999, 99, 9]
PROMPT_C = [1, 2, 3, 4, 5, 6, 7]


class Serving:
    """The three calls the adapter makes, in the order it makes them."""

    def __init__(self, generator, table):
        self.gen = generator
        self.table = table
        self.batch = generator.max_batch_size
        self.live = {}  # slot -> (last token, next position)

    def prefill(self, prompt, slot):
        tokens = torch.tensor(prompt, dtype=torch.int64).reshape(1, -1)
        out = self.gen.prefill_requests_into_slots(
            tokens,
            [len(prompt)],
            [slot],
            page_table=self.table[slot : slot + 1],
            sample_on_device=True,
        )
        token = int(out[0])
        self.live[slot] = (token, len(prompt))
        return token

    def _host_view(self, *, lag=0):
        tokens = torch.zeros(self.batch, dtype=torch.int64)
        positions = torch.full((self.batch,), -1, dtype=torch.int64)
        for slot, (token, position) in self.live.items():
            tokens[slot] = token
            positions[slot] = position - lag
        return tokens, positions

    def step(self, *, full_refresh, lag=0, trust_device=True, table=None):
        """One decode step. ``lag`` stages the host pair from one token ago, as vLLM does under async
        scheduling; ``trust_device`` is the adapter's per-row device-authority mask."""
        tokens, positions = self._host_view(lag=lag)
        trust = torch.zeros(self.batch, dtype=torch.bool)
        if trust_device:
            for slot in self.live:
                trust[slot] = True
        info = self.gen.stage_serving_decode_inputs(
            tokens,
            positions,
            self.table if table is None else table,
            full_refresh=full_refresh,
            device_token_rows=trust,
        )
        self.gen.submit_serving_decode(sample_on_device=True)
        out = self.gen.read_tokens()
        emitted = {}
        for slot in list(self.live):
            token, position = self.live[slot]
            emitted[slot] = int(out[slot])
            self.live[slot] = (int(out[slot]), position + 1)
        return info, emitted


def _state_rows(model, rows):
    """Every ``linear_attention`` layer's recurrent and conv state, per row, on host.

    Read from every device shard: the DeltaNet heads are split across the mesh, so a row-move has to
    be right on all four, not just on shard 0.
    """
    import ttnn

    out = {}
    for index, layer in enumerate(model.layers):
        if layer.is_full_attention or layer.recurrent_state is None:
            continue
        for name, buf in [("recurrent", layer.recurrent_state)] + [
            (f"conv{i}", b) for i, b in enumerate(layer.conv_state)
        ]:
            for shard, tensor in enumerate(ttnn.get_device_tensors(buf)):
                host = ttnn.to_torch(tensor)
                for row in rows:
                    out[(index, name, shard, row)] = host[row].clone()
    return out


def _pick(state, row):
    return {key: value for key, value in state.items() if key[3] == row}


def _same(left, right):
    """Bit-identical comparison of two state reads, ignoring which row each came from."""
    left_values = [left[key] for key in sorted(left, key=lambda k: (k[0], k[1], k[2]))]
    right_values = [right[key] for key in sorted(right, key=lambda k: (k[0], k[1], k[2]))]
    if len(left_values) != len(right_values):
        return False
    return all(torch.equal(a, b) for a, b in zip(left_values, right_values))


def _page_table(batch, blocks_per_user):
    """A vLLM-shaped table: slot u owns its own run of real blocks, 0 (the null block) elsewhere."""
    table = torch.zeros(batch, blocks_per_user, dtype=torch.int32)
    per_user = blocks_per_user // batch
    for user in range(batch):
        base = 1 + user * per_user
        table[user, :per_user] = torch.arange(base, base + per_user, dtype=torch.int32)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--output", default=str(MODEL_DIR / "doc" / "vllm_integration" / "serving_primitives.json"))
    args = ap.parse_args()

    mesh = open_ornith_mesh()
    report = {"layers": PROBE_LAYERS, "batch": BATCH, "steps": args.steps, "context": CONTEXT}
    try:
        path = resolve_model_path()
        model = OrnithModel.from_pretrained(
            path, mesh_device=mesh, hf_config=load_text_config(path), layer_indices=PROBE_LAYERS, max_context=CONTEXT
        )
        blocks_per_user = num_blocks_for_context(CONTEXT, model.page_block_size)
        kv_cache = model.allocate_kv_cache(1 + BATCH * (blocks_per_user // BATCH))
        gen = OrnithGenerator(
            model,
            max_batch_size=BATCH,
            cache_context=CONTEXT,
            sampling_mode="device",
            kv_cache=kv_cache,
            page_table=torch.zeros(BATCH, blocks_per_user, dtype=torch.int32),
        )
        report["owns_cache"] = gen.owns_cache
        table = _page_table(BATCH, blocks_per_user)
        gen.ensure_serving_traces()
        gen.ensure_sampling_trace()
        serving = Serving(gen, table)

        # ------------------------------------------------------------ 1. per-slot prefill
        first_a = serving.prefill(PROMPT_A, slot=1)
        first_b = serving.prefill(PROMPT_B, slot=3)
        report["prefill"] = {"slots": [1, 3], "tokens": [first_a, first_b]}
        logger.info(f"prefill tokens {[first_a, first_b]} into slots [1, 3]")

        # ------------------------------------------------------------ 2. steady state
        for key in gen.counters:
            gen.counters[key] = 0
        first_info, _ = serving.step(full_refresh=True)
        after_first = dict(gen.counters)
        steady_infos = []
        streams = {1: [], 3: []}
        for _ in range(args.steps - 1):
            info, emitted = serving.step(full_refresh=False)
            steady_infos.append(info)
            for slot, token in emitted.items():
                streams[slot].append(token)
        report["steady"] = {
            "first_step_refresh": first_info,
            "counters_after_first_step": after_first,
            "refreshes": steady_infos,
            "counters_after_steady": dict(gen.counters),
            "streams": {str(k): v for k, v in streams.items()},
            "copies_nothing_in_steady_state": all(
                not i["tokens"] and not i["positions"] and not i["page_table"] for i in steady_infos
            ),
            "one_token_refresh_total": gen.counters["token_refreshes"] == 1,
            "one_position_refresh_total": gen.counters["position_refreshes"] == 1,
        }

        # ------------------------------------------------------------ 3. stale host inputs
        arms = {}
        prefill_tokens = {}
        for arm, (lag, trust) in {
            "fresh": (0, True),
            "stale-merged": (1, True),
            "stale-host": (1, False),
        }.items():
            # Every arm starts from the same device state: at batch > 1 a neighbouring row changes
            # the MoE expert union and therefore the last bits of this row's logits, so two arms can
            # only be compared token for token if the whole batch starts identical.
            gen.reset()
            serving.live = {}
            prefill_token = serving.prefill(PROMPT_C, slot=2)
            prefill_tokens[arm] = prefill_token
            emitted = []
            for index in range(args.steps):
                # full_refresh on every step: this is the reset-heavy path, where a stale host view is
                # actually staged rather than skipped. The first step is the prefill boundary - the
                # host pair is the only authority there, for every arm.
                first = index == 0
                _, step = serving.step(
                    full_refresh=True,
                    lag=0 if first else lag,
                    trust_device=False if first else trust,
                )
                emitted.append(step[2])
            arms[arm] = emitted
            logger.info(f"stale arm {arm}: {emitted}")
        doubled = [i for i in range(1, len(arms["stale-host"])) if arms["stale-host"][i] == arms["stale-host"][i - 1]]
        report["stale_inputs"] = {
            "arms": arms,
            "prefill_tokens": prefill_tokens,
            "arms_started_from_the_same_state": len(set(prefill_tokens.values())) == 1,
            "merged_matches_fresh": arms["stale-merged"] == arms["fresh"],
            "host_wins_differs_from_fresh": arms["stale-host"] != arms["fresh"],
            "host_wins_repeat_positions": doubled,
        }

        # ------------------------------------------------------------ 4. page-table-only refresh
        gen.reset()
        serving.live = {}
        serving.prefill(PROMPT_A, slot=1)
        serving.step(full_refresh=True)
        before = dict(gen.counters)
        grown = table.clone()
        grown[1, blocks_per_user // BATCH] = 1 + BATCH * (blocks_per_user // BATCH) - 1
        info, _ = serving.step(full_refresh=False, table=grown)
        report["page_table_only"] = {
            "refresh": info,
            "token_refreshes_unchanged": gen.counters["token_refreshes"] == before["token_refreshes"],
            "position_refreshes_unchanged": gen.counters["position_refreshes"] == before["position_refreshes"],
            "page_table_refreshes": gen.counters["page_table_refreshes"] - before["page_table_refreshes"],
        }

        # ------------------------------------------------------------ 5. batch-layout change
        gen.reset()
        serving.live = {}
        serving.prefill(PROMPT_A, slot=1)
        stream = [serving.step(full_refresh=True)[1][1] for _ in range(2)]
        before_state = _state_rows(model, [1])
        before_tokens, before_positions = gen.device_decode_state()
        # A second request arrives mid-stream: its prefill writes slot 0's state and its own blocks and
        # must leave slot 1 alone. The step after it is a reset step for the whole batch.
        serving.prefill(PROMPT_B, slot=0)
        after_state = _state_rows(model, [1])
        info, emitted = serving.step(full_refresh=True)
        after_tokens, after_positions = gen.device_decode_state()
        report["batch_layout_change"] = {
            "stream_before": stream,
            "emitted_after_the_new_request": {str(k): v for k, v in emitted.items()},
            "continuing_row_state_bit_identical": _same(before_state, after_state),
            "continuing_row_position_advanced_by_one": int(after_positions[1]) - int(before_positions[1]) == 1,
            "continuing_row_device_token_is_the_emitted_one": int(after_tokens[1]) == int(emitted[1]),
            "refresh": info,
        }

        # ------------------------------------------------------------ 6. slot remap
        gen.reset()
        serving.live = {}
        serving.prefill(PROMPT_B, slot=3)
        for _ in range(2):
            serving.step(full_refresh=True)
        before_state = _state_rows(model, [0, 1, 2, 3])
        remap = torch.tensor([0, 1, 3, 2], dtype=torch.int32)
        moved_layers = gen.remap_serving_slots(remap)
        after_state = _state_rows(model, [0, 1, 2, 3])
        moved_rows = {
            "row2_took_row3": _same(_pick(before_state, 3), _pick(after_state, 2)),
            "row3_took_row2": _same(_pick(before_state, 2), _pick(after_state, 3)),
            "row0_untouched": _same(_pick(before_state, 0), _pick(after_state, 0)),
            "row1_untouched": _same(_pick(before_state, 1), _pick(after_state, 1)),
        }
        # And the moved request keeps generating from its own history in its new row.
        serving.live = {2: serving.live.pop(3)}
        moved_table = table.clone()
        moved_table[2] = table[3]
        serving.table = moved_table
        continued = [serving.step(full_refresh=True)[1][2] for _ in range(3)]
        report["slot_remap"] = {
            "remap": [int(v) for v in remap],
            "layers_moved": moved_layers,
            "state_moved": moved_rows,
            "state_moved_correctly": all(moved_rows.values()),
            "continued_in_new_row": continued,
        }

        report["trace_recaptures"] = gen.trace_recaptures
        report["sampling_trace_captures"] = gen.sampling_trace_captures
        gen.teardown()
    finally:
        close_ornith_mesh(mesh)
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
