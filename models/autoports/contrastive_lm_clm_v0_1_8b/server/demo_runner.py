# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import statistics
import sys
import time

from .demo_brain import DemoBrain
from .trex.engine import FPS, FRAME_MS, Game
from .trex.pilot import Arena, Pilot
from .trex.planner import snapshot

STATUS_CODES = {"waiting": 0, "running": 1, "jumping": 2, "ducking": 3, "crashed": 4}
KIND_CODES = {"cactusSmall": 0, "cactusLarge": 1, "pterodactyl": 2}
ROW_KEYS = (
    "seed",
    "survived",
    "deaths",
    "best_score",
    "scores",
    "decisions",
    "agreement_with_planner",
    "latency_ms_p50",
    "latency_ms_p95",
    "model_ms_p50",
    "answers_discarded",
    "errors",
    "last_error",
    "best_effort_decisions",
    "shield_interventions",
    "arrival_saves",
    "emergency_saves",
    "input_tokens",
    "game_seconds",
    "wall_seconds",
    "host_stall_seconds_dropped",
    "model",
    "endpoint",
    "server_ms_p50",
    "server_ms_p95",
)
STATS_PERIOD_S = 0.5
DROP_TEXTS = ("Late answer discarded", "Previous run answer discarded", "Answer failed")


class DemoPilot(Pilot):
    def __init__(self, *args, **kwargs):
        self.fresh = []
        super().__init__(*args, **kwargs)

    def note(self, frame, text, decision=None):
        super().note(frame, text, decision)
        self.fresh.append((frame, text, decision))

    def drain(self):
        out, self.fresh = self.fresh, []
        return out


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)


def encode_frame(game: Game, pilot, frame: int, decisions: list, events: list) -> dict:
    t = game.trex
    night = game.night
    return {
        "type": "frame",
        "f": frame,
        "g": {
            "r": game.run_index,
            "t": [t.x, t.y, STATUS_CODES.get(t.status, 1), t.frame, int(t.ducking)],
            "o": [
                [o.id, KIND_CODES[o.kind.name], o.size, round(o.x, 1), o.y, o.frame]
                for o in game.obstacles
                if not o.remove
            ],
            "c": [[round(c.x, 1), c.y] for c in game.clouds],
            "h": [
                round(game.horizon_x[0], 1),
                round(game.horizon_x[1], 1),
                game.horizon_source[0],
                game.horizon_source[1],
            ],
            "n": [
                round(night.opacity, 3),
                round(night.x, 1),
                night.phase,
                [[round(s[0], 1), s[1]] for s in night.stars],
                int(night.draw_stars),
            ],
            "inv": round(game.invert_fade, 3),
            "sc": game.score,
            "shown": game.shown_score,
            "vis": int(game.score_visible),
            "hi": Game.actual_distance(game.high_score),
            "spd": round(game.speed, 3),
            "cr": int(game.crashed),
            "rf": game.restart_frame,
            "pl": int(game.playing),
            "rv": round(game.reveal, 1),
            "dead": game.deaths,
        },
        "d": decisions,
        "e": events,
        "s": {
            "thinking": len(pilot.active),
            "held": pilot.held,
            "last": pilot.last_action,
            "event": pilot.spectator(frame)["event"],
        },
    }


def decision_entry(decision, frame: int, text: str) -> dict:
    probabilities = {a: round(float(v), 4) for a, v in decision.probabilities.items()}
    return {
        "seq": decision.seq,
        "frame": frame,
        "view_frame": decision.view_frame,
        "state": getattr(decision, "state", None),
        "instructions": getattr(decision, "instructions", None),
        "criteria": getattr(decision, "criteria", None),
        "p": probabilities,
        "proposed": decision.proposed,
        "executed": decision.executed,
        "best": decision.best,
        "safe": decision.safe,
        "intervened": bool(decision.intervened or decision.arrival_intervened),
        "arrival": bool(decision.arrival_intervened),
        "agreed": bool(decision.agreed),
        "airborne": bool(decision.airborne),
        "threat": decision.threat,
        "distance": getattr(decision, "distance", None),
        "latency_ms": round(decision.latency_ms, 1),
        "inference_ms": round(decision.inference_ms, 1),
        "server_ms": getattr(decision, "server_ms", None),
        "plan_ms": round(decision.plan_ms, 2),
        "input_tokens": decision.input_tokens,
        "dropped": text if text in DROP_TEXTS else None,
        "error": decision.error,
        "event": text,
    }


def stats_entry(arena, pilot, server_samples, course: int, seeds: int, frames_total: int) -> dict:
    st = pilot.stats
    game = pilot.game
    finished = list(st.scores)
    return {
        "type": "stats",
        "decisions": st.decisions,
        "rate": round(st.rate, 1),
        "inflight_active": len(pilot.active),
        "inflight": pilot.inflight,
        "discarded": st.discarded,
        "errors": st.errors,
        "last_error": st.last_error,
        "interventions": st.interventions,
        "arrival_saves": st.arrival_saves,
        "emergency_saves": st.emergency_saves,
        "agreement": round(st.agreements / st.decisions, 3) if st.decisions else None,
        "latency_p50": percentile(st.all_latency, 0.5),
        "latency_p95": percentile(st.all_latency, 0.95),
        "inference_p50": percentile(st.all_inference, 0.5),
        "server_p50": percentile(server_samples, 0.5),
        "deaths": game.deaths,
        "score": game.score,
        "best_score": max(finished + [game.score]),
        "game_seconds": round(arena.game_frames[0] / FPS, 1),
        "wall_seconds": round(arena.frame * FRAME_MS / 1000, 1),
        "host_stall_s": round(arena.pacer.dropped_ms / 1000, 2) if arena.pacer else 0.0,
        "input_tokens": st.tokens,
        "best_effort": st.best_effort,
        "course": course,
        "seeds": seeds,
        "seconds_left": round(max(0, frames_total - arena.frame) / FPS, 1),
    }


def warm_up(pilot) -> list:
    samples = pilot.brain.warm(snapshot(pilot.game, "run"), pilot.prompt)
    typical = sorted(samples)[len(samples) // 2] / FRAME_MS
    pilot.latency_frames = int(min(samples) / FRAME_MS)
    pilot.typical_frames = typical
    pilot.interval_frames = max(1.0, typical)
    pilot.jitter_frames = max(1.0, max(samples) / FRAME_MS + 1 - pilot.latency_frames)
    return [round(s, 1) for s in samples]


def course_row(arena, pilot, brain, seed: int, server_samples: list) -> dict:
    report = arena.report()
    p = report["players"][brain.name]
    game = pilot.game
    return {
        "seed": seed,
        "survived": game.deaths == 0,
        "deaths": game.deaths,
        "best_score": p["best_score"],
        "scores": p["scores"] + ([game.score] if not game.crashed else []),
        "decisions": p["decisions"],
        "agreement_with_planner": p["agreement_with_planner"],
        "latency_ms_p50": p["latency_ms_p50"],
        "latency_ms_p95": p["latency_ms_p95"],
        "model_ms_p50": p["model_ms_p50"],
        "answers_discarded": p["answers_discarded"],
        "errors": p["errors"],
        "last_error": p["last_error"],
        "best_effort_decisions": p["best_effort_decisions"],
        "shield_interventions": p["shield_interventions"],
        "arrival_saves": p["arrival_saves"],
        "emergency_saves": p["emergency_saves"],
        "input_tokens": p["input_tokens"],
        "game_seconds": round(arena.game_frames[0] / FPS, 1),
        "wall_seconds": report["seconds"],
        "host_stall_seconds_dropped": report["host_stall_seconds_dropped"],
        "model": p["model"],
        "endpoint": p["where"],
        "server_ms_p50": percentile(server_samples, 0.5),
        "server_ms_p95": percentile(server_samples, 0.95),
    }


def summary_of(rows: list, config: dict) -> dict:
    return {
        "model": rows[0]["model"],
        "endpoint": rows[0]["endpoint"],
        "mode": "realtime",
        "shield": bool(config["shield"]),
        "course_style": "original",
        "prompt": "labeled",
        "duration_s": config["duration"],
        "seeds": len(rows),
        "survived": sum(r["survived"] for r in rows),
        "deaths": sum(r["deaths"] for r in rows),
        "mean_best_score": round(statistics.fmean(r["best_score"] for r in rows), 1),
        "mean_decisions": round(statistics.fmean(r["decisions"] for r in rows), 1),
        "mean_agreement_with_planner": round(statistics.fmean(r["agreement_with_planner"] or 0 for r in rows), 3),
        "latency_ms_p50_median": statistics.median(r["latency_ms_p50"] or 0 for r in rows),
        "model_ms_p50_median": statistics.median(r["model_ms_p50"] or 0 for r in rows),
        "server_ms_p50_median": statistics.median(r["server_ms_p50"] or 0 for r in rows),
        "errors": sum(r["errors"] for r in rows),
        "answers_discarded": sum(r["answers_discarded"] for r in rows),
        "shield_interventions": sum(
            r["shield_interventions"] + r["arrival_saves"] + r["emergency_saves"] for r in rows
        ),
        "inflight": config["inflight"],
    }


def run_session(config: dict, conn) -> None:
    sys.setswitchinterval(0.001)
    os.environ["CLM_BASE_URL"] = config["base_url"]
    if config.get("api_key"):
        os.environ["CLM_API_KEY"] = config["api_key"]

    def send(message: dict) -> None:
        conn.send(json.dumps(message, separators=(",", ":")))

    def status(phase: str, course: int, detail: str = "") -> None:
        send({"type": "status", "phase": phase, "course": course, "seeds": config["seeds"], "detail": detail})

    rows = []
    stopped = False
    frames_total = int(config["duration"] * FPS)
    try:
        for i in range(config["seeds"]):
            seed = config["seed"] + i
            status("starting", i, "starting the player process")
            brain = DemoBrain("clm", inflight=config["inflight"])
            pilot = DemoPilot(brain, guarded=bool(config["shield"]), prompt="labeled", seed=seed)
            arena = Arena([pilot])
            server_samples = []
            try:
                status("warming", i, "measuring the answer latency")
                samples = warm_up(pilot)
                send(
                    {
                        "type": "course",
                        "index": i,
                        "seed": seed,
                        "endpoint": brain.detail,
                        "model": brain.model,
                        "warm_ms": samples,
                        "latency_frames": pilot.latency_frames,
                        "instructions": "Choose the best safe action for the dinosaur.",
                        "duration": config["duration"],
                    }
                )
                status("playing", i, f"warm answer {min(samples):.0f} ms")
                arena.start()
                last_stats = time.perf_counter()
                while arena.frame < frames_total:
                    if conn.poll(0):
                        command = conn.recv()
                        if command == "stop":
                            stopped = True
                            break
                    steps = arena.advance()
                    if steps:
                        decisions, events = [], []
                        for frame, text, decision in pilot.drain():
                            if decision is None:
                                events.append({"frame": frame, "event": text})
                            else:
                                entry = decision_entry(decision, frame, text)
                                if entry["server_ms"] is not None and entry["dropped"] is None:
                                    server_samples.append(entry["server_ms"])
                                decisions.append(entry)
                        send(encode_frame(pilot.game, pilot, arena.frame, decisions, events))
                        pilot.game.drain_events()
                    now = time.perf_counter()
                    if now - last_stats >= STATS_PERIOD_S:
                        send(stats_entry(arena, pilot, server_samples, i, config["seeds"], frames_total))
                        last_stats = now
                    time.sleep(0.002)
                if arena.pacer is not None:
                    send(stats_entry(arena, pilot, server_samples, i, config["seeds"], frames_total))
                    row = course_row(arena, pilot, brain, seed, server_samples)
                    rows.append(row)
                    send({"type": "course_end", "row": row})
            finally:
                pilot.close()
            if stopped:
                break
            if i + 1 < config["seeds"]:
                status("between", i + 1, "next course")
        if stopped:
            send({"type": "stopped", "rows": rows})
        else:
            status("finished", config["seeds"], "run complete")
            send({"type": "summary", "summary": summary_of(rows, config) if rows else None, "rows": rows})
    except Exception as exc:
        send({"type": "error", "error": str(exc)[:400]})
    finally:
        try:
            conn.close()
        except OSError:
            pass
