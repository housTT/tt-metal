from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

READINESS_DIR = Path(__file__).parents[1] / "readiness_vllm"
sys.path.insert(0, str(READINESS_DIR))

from derive_serving_host_metrics import MARKER
from derive_serving_host_metrics import main as derive_main  # noqa: E402
from derive_serving_host_metrics import parse_markers, select_benchmark_windows, window
from run_host_serving_lifecycle import lifecycle_metric_evidence  # noqa: E402


def _runtime(slots: dict[str, object], *, trace_replays: int = 0) -> dict[str, object]:
    return {
        "declared_host_work": {
            "model_load_exact_expert_prepack": True,
            "expert_route_id_read_and_exact_weight_dma": True,
            "ple_ngram_hash_row_lookup_and_dma": True,
            "caller_visible_compact_token_readback": True,
            "explicit_non_greedy_seed_control_h2d": False,
        },
        "prohibited_host_work": {
            "expert_projection": False,
            "ple_projection": False,
            "activation_roundtrip": False,
            "kv_or_recurrence": False,
            "optimized_sampling_or_argmax": False,
            "token_feedback_reconstruction": False,
            "per_token_position_refresh": False,
            "unchanged_page_table_refresh": False,
        },
        "ownership": {
            "kv_cache": "vllm",
            "recurrence": "model",
            "page_table": "state with stable model buffer",
            "tokens_and_positions": "device feedback after request reset",
            "expert_store": "per-layer exact mmap source, model-load packed-host preload, and fixed TT slots",
            "ple_store": "shared exact mmap table with request-isolated two-token history",
        },
        "counters": {
            "trace_replays": trace_replays,
            "model_only_trace_replays": 0,
            "sampling_seed_host_copies": 0,
            "virtual_decode_slots": slots,
        },
        "host_sampling_compatibility_calls": 0,
    }


def _slots(
    assignments: int,
    releases: int,
    *,
    commits: int = 0,
    restores: int = 0,
    resets: int = 0,
    trace_invalidations: int = 0,
) -> dict[str, object]:
    logical_bytes = 100
    return {
        "enabled": True,
        "physical_batch": 1,
        "capacity": 2,
        "active_slots": 0,
        "resident_slot": None,
        "valid_slots": 0,
        "assignments": assignments,
        "releases": releases,
        "stale_rejections": 0,
        "prefill_admissions_while_trace_live": assignments,
        "prefill_trace_invalidations": trace_invalidations,
        "sampling_trace_mode_switches": 0,
        "bank": {
            "enabled": True,
            "capacity": 2,
            "closed": False,
            "allocated_logical_bytes": 300,
            "logical_bytes_per_slot": logical_bytes,
            "zero_template_logical_bytes": logical_bytes,
            "commits": commits,
            "restores": restores,
            "resets": resets,
            "commit_logical_bytes": commits * logical_bytes,
            "restore_logical_bytes": restores * logical_bytes,
            "commit_submit_seconds": commits / 10,
            "restore_submit_seconds": restores / 10,
        },
    }


def _marker(
    completed: int,
    slots: dict[str, object],
    *,
    trace_replays: int = 0,
    host_scale: int = 0,
) -> dict[str, object]:
    host = {
        "expert_requests": float(host_scale),
        "expert_hits": float(host_scale),
        "expert_misses": float(host_scale),
        "expert_h2d_bytes": float(host_scale * 100),
        "ple_lookup_calls": float(host_scale),
        "ple_selected_rows": float(host_scale),
        "ple_device_h2d_bytes": float(host_scale * 10),
    }
    return {
        "event": "prefill_start",
        "completed_requests": completed,
        "attention_cache_owner": "vllm",
        "attention_cache": {
            "standalone_allocations": 48,
            "standalone_tensors_released": 48,
            "vllm_adoptions": 1,
        },
        "host_service": host,
        "host_gauges": {"ple_history_entries": 0.0},
        "decode_timing": {"replay_tokens": float(trace_replays)},
        "runtime_fallback": _runtime(slots, trace_replays=trace_replays),
        "request_counters": {},
    }


def test_parse_markers_accepts_archived_server_log(tmp_path: Path):
    marker = _marker(7, _slots(7, 7))
    path = tmp_path / "server.log.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(f"prefix {MARKER}{json.dumps(marker)}\n")
    assert parse_markers(path) == [marker]


def test_parse_markers_tolerates_only_a_trailing_live_partial_write(tmp_path: Path):
    marker = _marker(7, _slots(7, 7))
    path = tmp_path / "server.log"
    path.write_text(f'{MARKER}{json.dumps(marker)}\n{MARKER}{{"event":')
    assert parse_markers(path) == [marker]


def test_select_windows_uses_logical_counts_for_grouped_virtual_prefills():
    # The equal-replay markers at counts 7 and 39 create a later, shifted
    # +32/+3168 candidate.  Only the 6 -> 38 window shares the independently
    # verified +1/+127 primary boundary and therefore represents the benchmark.
    counts = [0, 5, 6, 7, *range(9, 38, 2), 38, 39]
    replays = [0, 100, 227, 227, *range(425, 3198, 198), 3395, 3395]
    markers = [
        _marker(count, _slots(count, count), trace_replays=replay)
        for count, replay in zip(counts, replays, strict=True)
    ]
    # Later lifecycle traffic must not shadow the canonical benchmark window.
    markers.extend(
        [
            _marker(43, _slots(43, 43, commits=2, restores=2), trace_replays=3412),
        ]
    )
    primary_start, ci_start, ci_end = select_benchmark_windows(markers, primary_requests=1, ci_requests=32)
    assert counts[primary_start] == 5
    assert counts[ci_start] == 6
    assert counts[ci_end] == 38


def test_select_windows_counts_trace_recapture_after_prefill_invalidation():
    # Program-cache growth releases the live trace.  The first subsequent
    # decode step captures its replacement, so one canonical decode step is an
    # invalidation/recapture rather than a replay in each workload.
    markers = [
        _marker(5, _slots(5, 5), trace_replays=100),
        _marker(6, _slots(6, 6, trace_invalidations=1), trace_replays=226),
        _marker(38, _slots(38, 38, trace_invalidations=2), trace_replays=3393),
    ]
    primary_start, ci_start, ci_end = select_benchmark_windows(markers, primary_requests=1, ci_requests=32)
    assert (primary_start, ci_start, ci_end) == (0, 1, 2)


def test_select_windows_does_not_accept_missing_replays_without_invalidation(expect_error):
    markers = [
        _marker(5, _slots(5, 5), trace_replays=100),
        _marker(6, _slots(6, 6), trace_replays=226),
        _marker(38, _slots(38, 38), trace_replays=3393),
    ]
    with expect_error(ValueError, "no markers delimit"):
        select_benchmark_windows(markers, primary_requests=1, ci_requests=32)


def test_window_reports_virtual_bank_deltas_and_exact_ownership():
    start = _marker(9, _slots(9, 9), trace_replays=100, host_scale=10)
    end = _marker(10, _slots(10, 10, trace_invalidations=1), trace_replays=227, host_scale=20)
    result = window(
        "primary",
        {"prompt_tokens": 128, "output_tokens": 128, "requests": 1},
        start,
        end,
        physical_batch=1,
        virtual_slot_capacity=2,
    )
    assert result["virtual_slot_delta"]["assignments"] == 1
    assert result["virtual_slot_delta"]["releases"] == 1
    assert result["virtual_slot_delta"]["prefill_trace_invalidations"] == 1
    assert result["virtual_bank_delta"]["commits"] == 0
    assert result["attention_cache_owner_end"] == "vllm"
    assert result["runtime_counter_delta"]["trace_replays"] == 127


def test_lifecycle_metrics_prove_active_banking_and_four_releases():
    baseline = _marker(20, _slots(20, 20), host_scale=20)
    start = _marker(21, _slots(21, 21), host_scale=21)
    # Physical-prefill grouping advances logical request count by two.
    grouped = _marker(23, _slots(23, 21, commits=1), trace_replays=4, host_scale=23)
    end = _marker(25, _slots(25, 25, commits=3, restores=2, resets=1), trace_replays=20, host_scale=30)
    evidence = lifecycle_metric_evidence(
        [baseline, start, grouped, end],
        first_new_index=1,
        baseline_start_count=20,
        physical_batch=1,
        virtual_slot_capacity=2,
    )
    assert evidence["measured_logical_requests"] == 4
    assert evidence["virtual_slot_delta"]["assignments"] == 4
    assert evidence["virtual_slot_delta"]["releases"] == 4
    assert evidence["active_overlap_bank_evidence"] is True
    assert evidence["finish_cancel_release_evidence"] is True
    assert evidence["runtime_contract_evidence"] is True


def test_derivation_main_accepts_grouped_ci_window(tmp_path: Path, monkeypatch, capsys):
    markers = [
        _marker(5, _slots(5, 5), trace_replays=100, host_scale=10),
        _marker(6, _slots(6, 6), trace_replays=227, host_scale=20),
        _marker(8, _slots(8, 6, commits=1), trace_replays=425, host_scale=25),
        _marker(38, _slots(38, 38, commits=30, restores=29, resets=28), trace_replays=3395, host_scale=100),
    ]
    log = tmp_path / "server.log"
    log.write_text("".join(f"{MARKER}{json.dumps(marker)}\n" for marker in markers))
    output = tmp_path / "serving_host_metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "derive_serving_host_metrics.py",
            "--server-log",
            str(log),
            "--output",
            str(output),
        ],
    )
    derive_main()
    capsys.readouterr()
    artifact = json.loads(output.read_text())
    assert artifact["primary_single_user"]["completed_requests_delta"] == 1
    assert artifact["ci_serving_burst"]["completed_requests_delta"] == 32
    assert artifact["ci_serving_burst"]["virtual_bank_delta"]["commits"] == 30
    assert artifact["marker_selection"]["primary_trace_replays"] == 127
    assert artifact["marker_selection"]["primary_prefill_trace_invalidations"] == 0
    assert artifact["marker_selection"]["ci_trace_replays"] == 3168
    assert artifact["marker_selection"]["ci_prefill_trace_invalidations"] == 0
    assert artifact["serving_capacity"] == {
        "max_num_seqs": 2,
        "physical_decode_batch": 1,
        "virtual_slot_capacity": 2,
    }
