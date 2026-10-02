# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
import threading

from .trex import backends as trex_backends
from .trex import brain as trex_brain

LATENCY_HEADER = "x-clm-latency-ms"
_local = threading.local()


def _record_server_ms(response):
    value = response.headers.get(LATENCY_HEADER)
    try:
        _local.server_ms = float(value) if value is not None else None
    except ValueError:
        _local.server_ms = None


def _hooked_create(kind, **options):
    backend = trex_backends.create(kind, **options)
    backend.client.event_hooks = {"response": [_record_server_ms]}
    return backend


def _recording_build_question(plan, prompt="labeled"):
    state, questions = trex_backends.build_question(plan, prompt)
    action = questions["action"]
    _local.state = state
    _local.criteria = dict(action["criteria"])
    _local.instructions = action["instructions"]
    _local.distance = plan.distance
    return state, questions


_original_think = trex_brain.think


def _demo_think(planner, backend, snap, timing, prompt, guarded):
    _local.server_ms = None
    _local.state = None
    _local.criteria = None
    _local.instructions = None
    _local.distance = None
    result = _original_think(planner, backend, snap, timing, prompt, guarded)
    result["server_ms"] = getattr(_local, "server_ms", None)
    result["state"] = getattr(_local, "state", None)
    result["criteria"] = getattr(_local, "criteria", None)
    result["instructions"] = getattr(_local, "instructions", None)
    result["distance"] = getattr(_local, "distance", None)
    return result


def demo_serve(kind, options, conn):
    trex_brain.create = _hooked_create
    trex_brain.build_question = _recording_build_question
    trex_brain.think = _demo_think
    trex_brain.serve(kind, options, conn)


class DemoBrain(trex_brain.RemoteBrain):
    def __init__(self, kind, **options):
        context = multiprocessing.get_context("spawn")
        self.conn, child = context.Pipe()
        self.process = context.Process(
            target=demo_serve, args=(kind, options, child), name=f"trex-demo-{kind}", daemon=True
        )
        self.process.start()
        child.close()
        reply = self.conn.recv()
        if reply[0] == "error":
            self.process.join(timeout=5)
            raise RuntimeError(reply[1])
        _, self.name, self.model, self.detail, self.usd_per_token = reply
        self.inflight = max(1, options.get("inflight", 1))
        self.lock = threading.Lock()
        self.tickets = 0
        self.waiting = {}
        self.reader = threading.Thread(target=self.read, name=f"brain-demo-{kind}", daemon=True)
        self.reader.start()
