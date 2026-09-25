# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Rectangular or row-wise: which core grid the decode stream should be sharded over.

This chip's compute grid is 11 wide, and the decode stream's core count must divide every role's
output tile count - 160, 192, 256, 320 and 544 tiles, whose GCD is 32 - so the count is a power of two
up to 32.  Those two facts cannot both be satisfied above 8 cores: ``ttnn.num_cores_to_corerangeset``
fills rows, so 32 cores come back as ``{[0-0 - 10-1], [0-2 - 9-2]}`` - two ranges, bounding box 33 -
and the only *rectangles* of 32 cores are 8x4, 16x2 and so on, which no row-wise fill produces.

Both choices cost something, and the costs are of different kinds:

* **row-wise (ragged)**: ops that check *rectangularity* rather than core count quietly degrade.
  ``ttnn.reshape`` on a width-sharded tensor falls back to INTERLEAVED - it logs
  "falling back to INTERLEAVED" - and the two decode norms run on the 33-core bounding box rather
  than the 32 cores that hold data.
* **rectangular (8x4)**: ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` computes its own
  output grid *row-wise* and overrides whatever the caller provides, logging "Mismatch between
  computed MemoryConfig ... and provided MemoryConfig ... Using computed config".  So every
  DRAM-sharded matmul lands its output on the ragged grid anyway, and the next op in the stream has to
  reshard it back - which a rectangular stream turns into three extra reshards per decode step.

So this is not "one is correct"; it is a measurement.  For each mode this probe reports the traced
decode time at both regimes, the number of memory-config conversions the step makes, the number of
INTERLEAVED reshape fallbacks and computed/provided mismatches the device log emits inside the
measured region, and the PCC - so the shipped default is chosen on numbers rather than on which
warning reads worse.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_stream_grid.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import statistics
import sys
import tempfile
import time
from unittest import mock

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import DEFAULT_GEOMETRY, DEFAULT_POLICY, OptimizedDecoder

PREFILL_LEN = 2048
REPLAYS = 8
SAMPLES = 5
RESHARD_OPS = ("interleaved_to_sharded", "sharded_to_interleaved", "to_memory_config")
#: The two device-log lines this trade-off is made of.
FALLBACK_PATTERN = re.compile(r"falling back to INTERLEAVED")
MISMATCH_PATTERN = re.compile(r"Mismatch between computed MemoryConfig")


@contextlib.contextmanager
def _capture_device_log():
    """Capture what the *C++* logger writes, which Python-level redirection cannot see.

    ``tt-metal``'s warnings - "falling back to INTERLEAVED", "Mismatch between computed
    MemoryConfig" - are written to the process's file descriptors by the native library, so
    ``contextlib.redirect_stdout`` never sees them.  This duplicates fds 1 and 2 onto a temp file for
    the duration of the block and yields a callable that returns the text.
    """
    with tempfile.TemporaryFile(mode="w+") as sink:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = (os.dup(1), os.dup(2))
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            yield lambda: (sink.flush(), sink.seek(0), sink.read())[2]
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


class _Counter:
    """Count calls to a set of ``ttnn`` entry points around a block."""

    def __init__(self, names):
        self.names = names
        self.counts = {name: 0 for name in names}
        self._patches: list = []

    def __enter__(self):
        for name in self.names:
            original = getattr(ttnn, name)

            def wrapper(*args, _name=name, _original=original, **kwargs):
                self.counts[_name] += 1
                return _original(*args, **kwargs)

            patch = mock.patch.object(ttnn, name, wrapper)
            patch.start()
            self._patches.append(patch)
        return self

    def __exit__(self, *exc):
        for patch in reversed(self._patches):
            patch.stop()
        self._patches.clear()
        return False

    def total(self) -> int:
        return sum(self.counts.values())


def measure(mesh, geometry, kind, layer_idx, batch) -> dict:
    lut = H.build_layer(
        mesh,
        layer_idx,
        max_batch=batch,
        max_seq_len=4096,
        decoder_cls=OptimizedDecoder,
        policy=DEFAULT_POLICY,
        decode_geometry=geometry,
    )
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, PREFILL_LEN, stats)
    got = H.run_tt_prefill(lut, hidden)
    cache = DynamicCache(config=lut.config)
    golden = H.reference_prefill(lut, hidden, cache)
    out = {
        "prefill_pcc": H.pcc(golden.reshape(1, PREFILL_LEN, -1), got),
        "stream_grid": str(lut.tt_layer.decode_core_range),
    }
    H.prepare_decode(lut)
    # ``decode_forward`` asserts the token's batch equals the layer's ``max_batch``, so the token is
    # built for the whole batch; only user 0 has a reference, since only user 0 was prefilled.
    token = ref.synthetic_hidden_states(lut.config, batch, 1, stats, seed=7)
    positions = torch.full((batch,), PREFILL_LEN)
    golden_decode = H.reference_decode(lut, token[:1], PREFILL_LEN, cache)

    # One eager step with the ttnn entry points counted and the device log captured.
    counter = _Counter(RESHARD_OPS)
    with _capture_device_log() as read_log, counter:
        decoded = H.run_tt_decode(lut, token, positions)
        log = read_log()
    out["decode_pcc"] = H.pcc(golden_decode.reshape(1, 1, -1), decoded[:1])
    out["reshards"] = counter.total()
    out["reshard_breakdown"] = {k: v for k, v in counter.counts.items() if v}
    out["interleaved_reshape_fallbacks"] = len(FALLBACK_PATTERN.findall(log))
    out["computed_vs_provided_mismatches"] = len(MISMATCH_PATTERN.findall(log))

    runner = H.TracedDecode(lut, batch=batch)
    runner.warmup(token, positions)
    runner.capture()
    runner.replay(token, positions)
    ttnn.synchronize_device(mesh)
    samples = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh, runner.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        samples.append((time.perf_counter() - start) * 1e3 / REPLAYS)
    runner.release()
    out["decode_ms"] = statistics.median(samples)
    out["decode_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return out


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=23887872)
    results: list = []
    try:
        for label, geometry in (
            ("rectangular 8x4 stream grid", dataclasses.replace(DEFAULT_GEOMETRY, rectangular_stream=True)),
            ("row-wise stream grid (ragged, bbox 33)", dataclasses.replace(DEFAULT_GEOMETRY, rectangular_stream=False)),
        ):
            for kind, layer_idx in (("linear_attention", H.LINEAR_LAYER_IDX), ("full_attention", H.FULL_LAYER_IDX)):
                for batch in (1, 32):
                    row = {
                        "sweep": "stream_grid",
                        "kind": kind,
                        "candidate": label,
                        "batch": batch,
                        "rectangular_stream": geometry.rectangular_stream,
                    }
                    try:
                        row.update(measure(mesh, geometry, kind, layer_idx, batch))
                    except Exception as exc:  # noqa: BLE001 - a blocker is a result
                        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    finally:
                        H.release_layers()
                    results.append(row)
                    print(
                        f"  {label:40s} {kind:16s} batch={batch:<3d} "
                        f"decode {row.get('decode_ms', float('nan')):7.4f} ms  "
                        f"reshards {row.get('reshards', -1):2d}  "
                        f"reshape_fallbacks {row.get('interleaved_reshape_fallbacks', -1):3d}  "
                        f"mismatches {row.get('computed_vs_provided_mismatches', -1):3d}  "
                        f"pcc {row.get('decode_pcc', float('nan')):.6f}"
                        + (f"  ERROR {row['error']}" if row.get("error") else ""),
                        flush=True,
                    )
    finally:
        for row in results:
            print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
        H.release_layers()
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
