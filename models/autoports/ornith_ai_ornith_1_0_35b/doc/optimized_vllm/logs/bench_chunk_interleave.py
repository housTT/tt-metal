# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Measure decode stalls while a long prompt is prefilling on a live vLLM server.

This is a client-only benchmark: it neither starts a server nor imports TTNN.  Start two
otherwise-identical servers, one with ``TT_INTERLEAVE_PREFILL_CHUNKS=0`` and one with it set to
``1``, and run this client once against each.  A representative steady-state comparison is::

    python bench_chunk_interleave.py run --url http://localhost:8100 \
        --label interleave-off --interleave-prefill-chunks 0 --warmup-long \
        --output /tmp/interleave-off.json
    python bench_chunk_interleave.py run --url http://localhost:8100 \
        --label interleave-on --interleave-prefill-chunks 1 --warmup-long \
        --output /tmp/interleave-on.json
    python bench_chunk_interleave.py compare \
        /tmp/interleave-off.json /tmp/interleave-on.json

The run starts several short-prompt, long-output streams.  Only after *every* stream has emitted
``--launch-after-tokens`` tokens does it submit the long prompt.  Thus this workload, unlike a
single simultaneous-arrival wave, proves that decode requests are active while the long prefill is
scheduled.  Prompts are explicit token-id lists so their logical lengths are exact.

``--warmup-long`` sends the long shape once, with a one-token output, before the measured workload.
Use it for the scheduler A/B so first-use program compilation and decode-trace recapture do not
dominate either arm.  Omit it deliberately when measuring cold first-use behavior.

``--interleave-prefill-chunks`` records the server's declared environment; the client cannot inspect
another process's environment.  Set that same ``TT_INTERLEAVE_PREFILL_CHUNKS`` value on the server,
not merely on this client.  The server needs ``max_num_seqs >= active_requests + 1`` (five for the
defaults), and its model-length limit must admit the requested prompt plus output.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import requests

DEFAULT_MODEL = "ornith-ai/Ornith-1.0-35B"


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """NumPy-compatible linear percentile without a NumPy dependency."""
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p99_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": max(values),
    }


def _prompt_tokens(length: int, request_index: int) -> list[int]:
    # The IDs stay comfortably inside this checkpoint's vocabulary.  Distinct offsets keep active
    # requests from accidentally sharing identical prompt prefixes if server policy changes later.
    offset = request_index * 104729
    return [1000 + ((offset + i * 7919) % 90000) for i in range(length)]


@dataclass
class RequestTrace:
    request_id: str
    role: str
    prompt_tokens: int
    requested_output_tokens: int
    start_s: float
    token_stamps_s: list[float] = field(default_factory=list)
    end_s: float | None = None
    usage: dict[str, int] | None = None

    @property
    def output_tokens(self) -> int:
        return len(self.token_stamps_s)

    def intervals(self) -> list[tuple[float, float, float]]:
        return [
            (previous, current, (current - previous) * 1000.0)
            for previous, current in zip(self.token_stamps_s, self.token_stamps_s[1:])
        ]

    def to_dict(
        self,
        experiment_start_s: float,
        long_prefill_window: tuple[float, float],
    ) -> dict[str, object]:
        if self.end_s is None or not self.token_stamps_s:
            raise RuntimeError(f"incomplete request trace: {self.request_id}")

        first = self.token_stamps_s[0]
        last = self.token_stamps_s[-1]
        e2e_s = self.end_s - self.start_s
        token_window_s = last - first
        itls = [interval_ms for _, _, interval_ms in self.intervals()]
        window_start, window_end = long_prefill_window
        # Include an interval if any part of it overlaps the period from long-request submission
        # through that request's first streamed token.  The baseline's one large blocking gap ends
        # just after the long first token, so filtering only by interval end would hide the stall.
        interference_itls = [
            interval_ms
            for previous, current, interval_ms in self.intervals()
            if previous <= window_end and current >= window_start
        ]

        return {
            "request_id": self.request_id,
            "role": self.role,
            "prompt_tokens": self.prompt_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "observed_output_tokens": self.output_tokens,
            "usage": self.usage,
            "start_after_experiment_ms": (self.start_s - experiment_start_s) * 1000.0,
            "ttft_ms": (first - self.start_s) * 1000.0,
            "e2e_ms": e2e_s * 1000.0,
            "first_to_last_token_ms": token_window_s * 1000.0,
            # Matches vLLM's aggregate output-throughput numerator convention at request scope.
            "output_throughput_tok_s": self.output_tokens / e2e_s,
            # Excludes TTFT but includes any mid-stream prefill stall.
            "decode_throughput_tok_s": (
                (self.output_tokens - 1) / token_window_s if self.output_tokens > 1 and token_window_s > 0.0 else None
            ),
            "itl_ms": _distribution(itls),
            "itl_intersecting_long_prefill_ms": _distribution(interference_itls),
            "token_arrival_ms": [(stamp - self.start_s) * 1000.0 for stamp in self.token_stamps_s],
        }


TokenCallback = Callable[[str, int], None]


def _stream_completion(
    *,
    url: str,
    model: str,
    request_id: str,
    role: str,
    prompt: list[int],
    max_tokens: int,
    read_timeout_s: float,
    on_token: TokenCallback | None = None,
) -> RequestTrace:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    trace = RequestTrace(
        request_id=request_id,
        role=role,
        prompt_tokens=len(prompt),
        requested_output_tokens=max_tokens,
        start_s=start,
    )
    endpoint = f"{url.rstrip('/')}/v1/completions"
    with requests.post(endpoint, json=body, stream=True, timeout=(10.0, read_timeout_s)) as response:
        if not response.ok:
            raise RuntimeError(f"{request_id}: HTTP {response.status_code}: {response.text[:2000]}")
        # requests' default 512-byte chunk can combine several server events and erase client-visible
        # ITL.  One-byte chunks preserve each SSE event's arrival boundary.
        for raw_line in response.iter_lines(chunk_size=1):
            received = time.perf_counter()
            if not raw_line or not raw_line.startswith(b"data: "):
                continue
            data = raw_line[6:]
            if data == b"[DONE]":
                break
            payload = json.loads(data)
            if payload.get("usage") is not None:
                trace.usage = payload["usage"]
            for choice in payload.get("choices", []):
                # Some vLLM versions attach ``finish_reason`` to the same event as the final token;
                # others emit a separate empty-text stop event. Count every non-empty text event,
                # plus an empty decoded token while generation is still live, but never the separate
                # empty stop event.
                if "text" in choice and (choice["text"] != "" or choice.get("finish_reason") is None):
                    trace.token_stamps_s.append(received)
                    if on_token is not None:
                        on_token(request_id, trace.output_tokens)
    trace.end_s = time.perf_counter()

    usage_tokens = trace.usage.get("completion_tokens") if trace.usage is not None else None
    if usage_tokens is not None and usage_tokens != trace.output_tokens:
        raise RuntimeError(
            f"{request_id}: {usage_tokens} completion tokens but {trace.output_tokens} timed SSE events; "
            "per-token ITL would be invalid"
        )
    if trace.output_tokens != max_tokens:
        raise RuntimeError(
            f"{request_id}: expected {max_tokens} streamed tokens with ignore_eos, got {trace.output_tokens}"
        )
    return trace


def _check_server(url: str, read_timeout_s: float) -> None:
    endpoint = f"{url.rstrip('/')}/health"
    response = requests.get(endpoint, timeout=(10.0, min(read_timeout_s, 30.0)))
    response.raise_for_status()


def _aggregate_report(
    traces: list[RequestTrace],
    experiment_start_s: float,
    long_trace: RequestTrace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not long_trace.token_stamps_s:
        raise RuntimeError("long request has no first-token timestamp")
    long_window = (long_trace.start_s, long_trace.token_stamps_s[0])
    rows = [trace.to_dict(experiment_start_s, long_window) for trace in traces]
    active_traces = [trace for trace in traces if trace.role == "active_decode"]

    all_start = min(trace.start_s for trace in traces)
    all_end = max(trace.end_s for trace in traces if trace.end_s is not None)
    active_start = min(trace.start_s for trace in active_traces)
    active_end = max(trace.end_s for trace in active_traces if trace.end_s is not None)
    active_first = min(trace.token_stamps_s[0] for trace in active_traces)
    active_last = max(trace.token_stamps_s[-1] for trace in active_traces)

    pooled_active_itls: list[float] = []
    pooled_interference_itls: list[float] = []
    for trace in active_traces:
        for previous, current, interval_ms in trace.intervals():
            pooled_active_itls.append(interval_ms)
            if previous <= long_window[1] and current >= long_window[0]:
                pooled_interference_itls.append(interval_ms)

    all_tokens = sum(trace.output_tokens for trace in traces)
    active_tokens = sum(trace.output_tokens for trace in active_traces)
    active_decode_intervals = sum(max(0, trace.output_tokens - 1) for trace in active_traces)
    aggregate = {
        "all_requests": {
            "output_tokens": all_tokens,
            "makespan_ms": (all_end - all_start) * 1000.0,
            "output_throughput_tok_s": all_tokens / (all_end - all_start),
        },
        "active_decode_requests": {
            "requests": len(active_traces),
            "output_tokens": active_tokens,
            "makespan_ms": (active_end - active_start) * 1000.0,
            "output_throughput_tok_s": active_tokens / (active_end - active_start),
            "first_to_last_token_span_ms": (active_last - active_first) * 1000.0,
            "decode_throughput_tok_s": active_decode_intervals / (active_last - active_first),
            "pooled_itl_ms": _distribution(pooled_active_itls),
            "pooled_itl_intersecting_long_prefill_ms": _distribution(pooled_interference_itls),
        },
    }
    return rows, aggregate


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.active_requests < 1:
        raise ValueError("--active-requests must be positive")
    if args.launch_after_tokens < 1 or args.launch_after_tokens >= args.active_output_tokens:
        raise ValueError("--launch-after-tokens must be in [1, active-output-tokens)")
    for name in ("active_prompt_tokens", "active_output_tokens", "long_prompt_tokens", "long_output_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    _check_server(args.url, args.read_timeout_s)
    if args.warmup_long:
        print(f"warming exact {args.long_prompt_tokens}-token prompt shape before measurement", flush=True)
        _stream_completion(
            url=args.url,
            model=args.model,
            request_id="warmup-long",
            role="warmup",
            prompt=_prompt_tokens(args.long_prompt_tokens, args.active_requests + 1),
            max_tokens=1,
            read_timeout_s=args.read_timeout_s,
        )

    condition = threading.Condition()
    active_progress = {f"active-{index}": 0 for index in range(args.active_requests)}
    active_failures: dict[str, str] = {}
    start_event = threading.Event()

    def on_active_token(request_id: str, count: int) -> None:
        with condition:
            active_progress[request_id] = count
            condition.notify_all()

    def active_request(index: int) -> RequestTrace:
        request_id = f"active-{index}"
        start_event.wait()
        try:
            return _stream_completion(
                url=args.url,
                model=args.model,
                request_id=request_id,
                role="active_decode",
                prompt=_prompt_tokens(args.active_prompt_tokens, index),
                max_tokens=args.active_output_tokens,
                read_timeout_s=args.read_timeout_s,
                on_token=on_active_token,
            )
        except Exception as error:
            with condition:
                active_failures[request_id] = repr(error)
                condition.notify_all()
            raise

    with ThreadPoolExecutor(max_workers=args.active_requests + 1) as executor:
        active_futures: list[Future[RequestTrace]] = [
            executor.submit(active_request, index) for index in range(args.active_requests)
        ]
        experiment_start = time.perf_counter()
        start_event.set()

        deadline = time.monotonic() + args.activation_timeout_s
        with condition:
            while min(active_progress.values()) < args.launch_after_tokens:
                if active_failures:
                    raise RuntimeError(f"active request failed before long launch: {active_failures}")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"active streams did not all reach {args.launch_after_tokens} tokens; "
                        f"progress={active_progress}"
                    )
                condition.wait(timeout=min(remaining, 1.0))
            progress_at_launch = dict(active_progress)

        print(
            f"all {args.active_requests} decode streams are active at {progress_at_launch}; "
            f"submitting {args.long_prompt_tokens}-token prompt",
            flush=True,
        )
        long_future = executor.submit(
            _stream_completion,
            url=args.url,
            model=args.model,
            request_id="long-prefill",
            role="long_prefill",
            prompt=_prompt_tokens(args.long_prompt_tokens, args.active_requests),
            max_tokens=args.long_output_tokens,
            read_timeout_s=args.read_timeout_s,
        )

        long_trace = long_future.result()
        active_traces = [future.result() for future in active_futures]

    traces = [*active_traces, long_trace]
    request_rows, aggregate = _aggregate_report(traces, experiment_start, long_trace)
    missing_overlap = [
        row["request_id"]
        for row in request_rows
        if row["role"] == "active_decode" and row["itl_intersecting_long_prefill_ms"]["count"] == 0
    ]
    if missing_overlap:
        raise RuntimeError(
            "long request did not overlap an inter-token interval for every active stream; "
            f"invalid mixed-workload evidence for {missing_overlap}"
        )
    long_row = next(row for row in request_rows if row["role"] == "long_prefill")
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "server": {"url": args.url, "model": args.model},
        "declared_server_env": {
            "TT_INTERLEAVE_PREFILL_CHUNKS": str(args.interleave_prefill_chunks),
        },
        "workload": {
            "active_requests": args.active_requests,
            "active_prompt_tokens": args.active_prompt_tokens,
            "active_output_tokens": args.active_output_tokens,
            "launch_after_tokens": args.launch_after_tokens,
            "long_prompt_tokens": args.long_prompt_tokens,
            "long_output_tokens": args.long_output_tokens,
            "warmup_long": args.warmup_long,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
        },
        "launch_evidence": {"progress_at_long_launch": progress_at_launch},
        "headline": {
            "long_prompt_ttft_ms": long_row["ttft_ms"],
            "active_output_throughput_tok_s": aggregate["active_decode_requests"]["output_throughput_tok_s"],
            "total_output_throughput_tok_s": aggregate["all_requests"]["output_throughput_tok_s"],
            "active_interference_itl_p99_ms": aggregate["active_decode_requests"][
                "pooled_itl_intersecting_long_prefill_ms"
            ]["p99_ms"],
            "active_interference_itl_max_ms": aggregate["active_decode_requests"][
                "pooled_itl_intersecting_long_prefill_ms"
            ]["max_ms"],
        },
        "requests": request_rows,
        "aggregate": aggregate,
    }
    return report


def _comparison_metrics(report: dict[str, object]) -> dict[str, float]:
    headline = report["headline"]
    return {
        "long_prompt_ttft_ms": headline["long_prompt_ttft_ms"],
        "active_interference_itl_p99_ms": headline["active_interference_itl_p99_ms"],
        "active_interference_itl_max_ms": headline["active_interference_itl_max_ms"],
        "active_output_throughput_tok_s": headline["active_output_throughput_tok_s"],
        "total_output_throughput_tok_s": headline["total_output_throughput_tok_s"],
    }


def _compare(before_path: Path, after_path: Path) -> dict[str, object]:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    if before["workload"] != after["workload"]:
        raise ValueError("reports have different workloads; refusing an invalid A/B comparison")
    before_metrics = _comparison_metrics(before)
    after_metrics = _comparison_metrics(after)
    metrics: dict[str, object] = {}
    for name, before_value in before_metrics.items():
        after_value = after_metrics[name]
        metrics[name] = {
            "before": before_value,
            "after": after_value,
            "delta": after_value - before_value,
            "delta_percent": ((after_value / before_value) - 1.0) * 100.0 if before_value else None,
        }
    return {
        "schema_version": 1,
        "before": {"path": str(before_path), "label": before["label"]},
        "after": {"path": str(after_path), "label": after["label"]},
        "declared_server_env": {
            "before": before.get("declared_server_env"),
            "after": after.get("declared_server_env"),
        },
        "workload": before["workload"],
        "metrics": metrics,
    }


def _print_headline(report: dict[str, object]) -> None:
    headline = report["headline"]
    print(f"label: {report['label']}")
    print(f"long prompt TTFT: {headline['long_prompt_ttft_ms']:.3f} ms")
    print(f"active output throughput: {headline['active_output_throughput_tok_s']:.3f} tok/s")
    print(f"total output throughput: {headline['total_output_throughput_tok_s']:.3f} tok/s")
    print(
        "active ITL intersecting long prefill: "
        f"p99 {headline['active_interference_itl_p99_ms']:.3f} ms, "
        f"max {headline['active_interference_itl_max_ms']:.3f} ms"
    )
    print(f"{'request':14s} {'TTFT ms':>11s} {'output tok/s':>14s} {'decode tok/s':>14s} {'ITL p99 ms':>12s}")
    for row in report["requests"]:
        decode_tps = row["decode_throughput_tok_s"]
        decode_text = f"{decode_tps:.3f}" if decode_tps is not None else "n/a"
        print(
            f"{row['request_id']:14s} {row['ttft_ms']:11.3f} "
            f"{row['output_throughput_tok_s']:14.3f} {decode_text:>14s} "
            f"{row['itl_ms']['p99_ms']:12.3f}"
        )


def _print_comparison(comparison: dict[str, object]) -> None:
    before_label = comparison["before"]["label"]
    after_label = comparison["after"]["label"]
    print(f"{'metric':43s} {before_label:>16s} {after_label:>16s} {'delta %':>10s}")
    for name, row in comparison["metrics"].items():
        print(f"{name:43s} {row['before']:16.3f} {row['after']:16.3f} {row['delta_percent']:10.2f}")


def _self_test() -> None:
    assert _percentile([], 0.99) is None
    assert _percentile([4.0], 0.99) == 4.0
    assert _percentile([0.0, 10.0], 0.50) == 5.0
    assert _percentile(list(range(1, 101)), 0.99) == 99.01

    trace = RequestTrace("active-0", "active_decode", 8, 4, 0.0, [1.0, 2.0, 6.0, 7.0], 8.0)
    row = trace.to_dict(0.0, (2.5, 5.5))
    assert row["ttft_ms"] == 1000.0
    assert row["output_throughput_tok_s"] == 0.5
    assert row["decode_throughput_tok_s"] == 0.5
    assert row["itl_ms"]["max_ms"] == 4000.0
    # The 2s -> 6s gap spans the entire long-prefill window and must not be filtered out.
    assert row["itl_intersecting_long_prefill_ms"]["count"] == 1
    assert row["itl_intersecting_long_prefill_ms"]["max_ms"] == 4000.0
    print("self-test passed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the staggered mixed workload against a live server")
    run.add_argument("--url", default="http://localhost:8100")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--label", required=True, help="arm label recorded in the report")
    run.add_argument(
        "--interleave-prefill-chunks",
        type=int,
        choices=(0, 1),
        required=True,
        help="TT_INTERLEAVE_PREFILL_CHUNKS value declared for the already-running server",
    )
    run.add_argument("--active-requests", type=int, default=4)
    run.add_argument("--active-prompt-tokens", type=int, default=128)
    run.add_argument("--active-output-tokens", type=int, default=512)
    run.add_argument("--launch-after-tokens", type=int, default=16)
    run.add_argument("--long-prompt-tokens", type=int, default=65536)
    run.add_argument("--long-output-tokens", type=int, default=64)
    run.add_argument("--warmup-long", action="store_true")
    run.add_argument("--activation-timeout-s", type=float, default=300.0)
    run.add_argument("--read-timeout-s", type=float, default=900.0)
    run.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare", help="compare two reports with identical workloads")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--output", type=Path)

    subparsers.add_parser("self-test", help="exercise percentile, throughput, and overlap math locally")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "self-test":
        _self_test()
        return
    if args.command == "compare":
        comparison = _compare(args.before, args.after)
        _print_comparison(comparison)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(comparison, indent=2) + "\n")
        return

    report = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _print_headline(report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
