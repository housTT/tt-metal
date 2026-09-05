#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

import pytest

from tracy import process_ops_logs


# class for mocking creation of npe data
class _FakeNpeResult:
    def __init__(self, noc_util, mcast_noc_util, dram_bw_util, cong_impact):
        self.overall_avg_link_util = noc_util
        self.overall_avg_mcast_write_link_util = mcast_noc_util
        self.dram_bw_util = dram_bw_util
        self._cong_impact = cong_impact

    def getCongestionImpact(self):
        return self._cong_impact


class _FakeNpeDatapoint:
    def __init__(self, result):
        self.result = result


class _FakeNpeStats:
    def __init__(self, op_to_result):
        self._op_to_result = op_to_result

    def getDatapointByID(self, op_id):
        result = self._op_to_result.get(op_id)
        if result is None:
            return None
        return _FakeNpeDatapoint(result)


@pytest.mark.skip(reason="Missing mock for device log file; needs fix to properly stub _enrich_ops_from_device_logs")
def test_append_device_data_populates_multicast_noc_util(monkeypatch, tmp_path):
    ops = {
        1: {
            "global_call_count": 1,
            "device_id": 0,
        }
    }
    trace_replays = {}

    fake_stats = _FakeNpeStats(
        {
            1: _FakeNpeResult(
                noc_util=91.24,
                mcast_noc_util=44.44,
                dram_bw_util=38.88,
                cong_impact=12.345,
            )
        }
    )
    monkeypatch.setattr(process_ops_logs, "analyzeNoCTraces", lambda _log_folder: fake_stats)

    process_ops_logs.append_device_data(
        ops=ops,
        traceReplays=trace_replays,
        logFolder=tmp_path,
        analyze_noc_traces=True,
        device_analysis_types=[],
    )

    assert ops[1]["NOC UTIL (%)"] == 91.2
    assert ops[1]["MULTICAST NOC UTIL (%)"] == 44.4
    assert ops[1]["DRAM BW UTIL (%)"] == 38.9
    assert ops[1]["NPE CONG IMPACT (%)"] == 12.35


def test_generate_reports_writes_sub_device_id_column(tmp_path):
    log_folder = tmp_path / "logs"
    report_folder = tmp_path / "reports"
    log_folder.mkdir(parents=True, exist_ok=True)

    device_log = log_folder / "profile_log_device.csv"
    device_log.write_text(
        "\n".join(
            [
                "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64",
                "PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],data,run host ID,trace id,trace id counter,zone name,type,source line,source file,meta data",
                '0,0,0,BRISC,1,100,0,42,,,BRISC-FW,ZONE_START,1,k.cpp,{"sub_device_id":1;"sub_device_manager_id":7}',
            ]
        )
    )

    ops = {
        42: {
            "global_call_count": 42,
            "device_id": 0,
            "host_time": {"ns_since_start": 10, "exec_time_ns": 20},
            "metal_trace_id": None,
            "input_tensors": [],
            "output_tensors": [],
        }
    }

    sub_device_lookup = process_ops_logs.build_sub_device_id_lookup_from_device_csv(device_log)
    host_ops_by_device = {0: [ops[42].copy()]}
    process_ops_logs.attach_sub_device_ids_to_ops(host_ops_by_device, sub_device_lookup)
    ops[42]["sub_device_id"] = host_ops_by_device[0][0]["sub_device_id"]

    process_ops_logs.generate_reports(
        ops=ops,
        deviceOps={},
        traceOps={},
        signposts={},
        logFolder=log_folder,
        outputFolder=report_folder,
        date=False,
        nameAppend=None,
    )

    report_csv = Path(report_folder) / "ops_perf_results.csv"
    assert report_csv.is_file()

    with report_csv.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)
        assert "SUB DEVICE ID" in reader.fieldnames
        assert row["SUB DEVICE ID"] == "1"
        assert "SUB DEVICE MANAGER ID" not in reader.fieldnames


def test_get_op_sub_device_lookup_key_prefers_device_perf_row():
    op = {
        "global_call_count": 1,
        "device_id": 0,
        "metal_trace_id": None,
        "_device_perf_row": {
            "GLOBAL CALL COUNT": 2048,
            "DEVICE ID": 0,
            "METAL TRACE ID": "",
            "METAL TRACE REPLAY SESSION ID": "",
        },
    }
    assert process_ops_logs.get_op_sub_device_lookup_key(op, 0) == (0, 2048, -1, -1)


def test_build_sub_device_id_lookup_ignores_manager_id_only_rows(tmp_path):
    device_log = tmp_path / "profile_log_device.csv"
    device_log.write_text(
        "\n".join(
            [
                "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64",
                "PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],data,run host ID,trace id,trace id counter,zone name,type,source line,source file,meta data",
                '0,0,0,BRISC,1,100,0,42,0,1,BRISC-FW,ZONE_START,1,k.cpp,{"sub_device_id":0;"sub_device_manager_id":7}',
            ]
        )
    )

    lookup = process_ops_logs.build_sub_device_id_lookup_from_device_csv(device_log)
    assert lookup[(0, 42, 0, 1)] == 0


def test_generate_reports_writes_multicast_noc_util_column(tmp_path):
    log_folder = tmp_path / "logs"
    report_folder = tmp_path / "reports"
    log_folder.mkdir(parents=True, exist_ok=True)

    ops = {
        1: {
            "global_call_count": 1,
            "device_id": 0,
            "host_time": {"ns_since_start": 10, "exec_time_ns": 20},
            "metal_trace_id": None,
            "input_tensors": [],
            "output_tensors": [],
            "NOC UTIL (%)": 50.0,
            "MULTICAST NOC UTIL (%)": 25.0,
            "DRAM BW UTIL (%)": 75.0,
            "NPE CONG IMPACT (%)": 1.25,
        }
    }

    process_ops_logs.generate_reports(
        ops=ops,
        deviceOps={},
        traceOps={},
        signposts={},
        logFolder=log_folder,
        outputFolder=report_folder,
        date=False,
        nameAppend=None,
    )

    report_csv = Path(report_folder) / "ops_perf_results.csv"
    assert report_csv.is_file()

    with report_csv.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)
        assert "MULTICAST NOC UTIL (%)" in reader.fieldnames
        assert row["MULTICAST NOC UTIL (%)"] == "25.0"


def test_perf_csv_keeps_replays_and_ignores_unexecuted_capture_definitions():
    host_ops = {
        1: [
            {"global_call_count": 256001, "metal_trace_id": 0},
            {"global_call_count": 300001, "metal_trace_id": 2},
        ]
    }
    first = {"METAL TRACE ID": 2, "METAL TRACE REPLAY SESSION ID": 1, "DEVICE FW DURATION [ns]": 100}
    second = {"METAL TRACE ID": 2, "METAL TRACE REPLAY SESSION ID": 2, "DEVICE FW DURATION [ns]": 120}
    perf = {1: {(300001, 2, 1): first, (300001, 2, 2): second}}
    result = process_ops_logs._enrich_ops_from_perf_csv(host_ops, perf, {1: {2: [1000, 2000]}})
    assert len(result[1]) == 2
    assert [op["_device_perf_row"] for op in result[1]] == [first, second]
    assert [op["tracy_time"] for op in result[1]] == [1000, 2000]


@pytest.mark.parametrize(
    "trace_id,trace_replays",
    [(0, {1: {0: [1000]}}), (0, None), (None, {}), ("", {})],
)
def test_perf_csv_missing_executed_or_unknown_operation_still_fails(trace_id, trace_replays):
    host_ops = {1: [{"global_call_count": 256001, "metal_trace_id": trace_id}]}
    with pytest.raises(  # allow-pytest.raises: CPU parser test runs without device conftest.
        AssertionError, match="Device data missing: Op 256001"
    ):
        process_ops_logs._enrich_ops_from_perf_csv(host_ops, {1: {}}, trace_replays)


def test_perf_csv_retains_device_rows_even_without_host_replay_metadata():
    row = {"METAL TRACE ID": 0, "METAL TRACE REPLAY SESSION ID": 1, "DEVICE FW DURATION [ns]": 100}
    result = process_ops_logs._enrich_ops_from_perf_csv(
        {1: [{"global_call_count": 256001, "metal_trace_id": 0}]},
        {1: {(256001, 0, 1): row}},
        {},
    )
    assert [op["_device_perf_row"] for op in result[1]] == [row]
