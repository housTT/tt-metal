# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in reproduction of persistent allocations overlapping an older trace."""

import json
import os

import pytest
import torch

import ttnn

from ..tt.fused_decoder import FusedDecoder
from . import test_functional_decoder as H

pytestmark = [
    pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True),
    pytest.mark.parametrize(
        "device_params", [{**H.DEVICE_PARAMS[0], "trace_region_size": 32 * 1024 * 1024}], indirect=True
    ),
    pytest.mark.skipif(
        os.environ.get("ORNITH_TRACE_ALLOCATION_REPRO") != "1",
        reason="set ORNITH_TRACE_ALLOCATION_REPRO=1 to reproduce deliberately unsafe allocation ordering",
    ),
]


def test_runtime_trace_overwrites_later_decoder_buffers(mesh_device, monkeypatch):
    """Replay A once and detect changes to untouched decoder B's persistent buffers.

    This is a positive reproduction, not an acceptance gate: a pass means the
    allocation-order hazard was observed with two identical runtime decoders.
    """
    prompt = H.make_activations(1, 2048, seed=71)
    token = H.make_activations(1, 1, seed=82)
    handles, traces, eager_outputs = [], [], []
    try:
        # Deliberately preserve the failing harness order: capture A before
        # constructing B or allocating any of its weights, state, or inputs.
        for _ in range(2):
            with monkeypatch.context() as patch:
                patch.setattr(H, "FunctionalDecoder", FusedDecoder)
                decoder, table, _ = H.build_decoder(mesh_device, H.LINEAR_LAYER, "real")
            x, d = H.to_device(mesh_device, prompt), H.to_device(mesh_device, token)
            ttnn.deallocate(decoder.prefill_forward(x, page_table=table))
            ttnn.deallocate(x)
            saved = H._snapshot_state(decoder)
            pos, rot = H.decode_inputs(mesh_device, torch.tensor([2048]))
            eager = decoder.decode_forward(d, current_pos=pos, rot_idxs=rot, page_table=table)
            eager_outputs.append(ttnn.to_torch(eager))
            ttnn.deallocate(eager)
            H._restore_state(decoder, saved)
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            traces.append(trace)
            out = decoder.decode_forward(d, current_pos=pos, rot_idxs=rot, page_table=table)
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            handles.append((decoder, saved, out, d, pos, rot, table))

        eager_pcc = H.pcc(*eager_outputs)
        assert eager_pcc >= H.PCC_BAR
        second, _, _, second_input, second_pos, second_rot, _ = handles[1]
        probes = {
            "token": second_input,
            "current_pos": second_pos,
            "rot_idxs": second_rot,
            "recurrent_state": second.recurrent_state,
        }
        probes.update({f"conv_state.{i}": buf for i, buf in enumerate(second.conv_state)})
        for key in ("attn_norm", "ff_norm", "gdn_norm", "A_neg", "dt_bias", "kda_norm_vector"):
            probes[f"weight.{key}"] = second.w[key]
        probes.update({f"weight.conv_taps.{i}": buf for i, buf in enumerate(second.w["conv_taps"])})
        before = {name: ttnn.to_torch(buf).clone() for name, buf in probes.items()}

        first, first_saved, first_out, *_ = handles[0]
        H._restore_state(first, first_saved)
        ttnn.execute_trace(mesh_device, traces[0], cq_id=0, blocking=True)
        # B is never replayed. These reads allocate only host tensors.
        changed, measurements = [], {}
        for name, buf in probes.items():
            actual = ttnn.to_torch(buf)
            expected = before[name]
            different = int(torch.count_nonzero(expected != actual))
            error = (expected.double() - actual.double()).abs()
            finite = bool(torch.isfinite(error).all())
            measurements[name] = {
                "buffer_id": buf.buffer_unique_id(),
                "address": buf.buffer_address(),
                "shape": list(buf.shape),
                "elements": actual.numel(),
                "changed_elements": different,
                "max_abs_error": float(error.max()) if finite else None,
                "nonfinite_elements": int(torch.count_nonzero(~torch.isfinite(actual))),
            }
            if different:
                changed.append(name)
        first_replay_equals_eager = torch.equal(ttnn.to_torch(first_out), eager_outputs[0])
        print(
            "TRACE_ALLOCATION_REPRO "
            + json.dumps(
                {
                    "classes": ["FusedDecoder", "FusedDecoder"],
                    "replays": [1, 0],
                    "eager_pcc": eager_pcc,
                    "first_replay_equals_eager": first_replay_equals_eager,
                    "changed_second_decoder_buffers": changed,
                    "buffers": measurements,
                }
            ),
            flush=True,
        )
        assert changed, "No probed B buffer changed; the predicted overlap was not reproduced"
        assert first_replay_equals_eager, "A also diverged; inspect the emitted buffer corruption evidence"
    finally:
        for trace in traces:
            ttnn.release_trace(mesh_device, trace)
