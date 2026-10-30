#!/usr/bin/env python3
"""Mixed-shape trace reuse and request-lifecycle check against a live Qwen3-8B vLLM server.

Sends unseen prompt lengths, concurrent bursts, prefix-cache resumes at nonzero start positions while
another request decodes, a prompt longer than one prefill chunk, sampling-mode switches on one prompt,
reordered repeats and host-sampled requests; then reads the adapter's ``qwen3-vllm`` log lines written
during the workload (prefill starts and ends, decode buckets, trace captures, program-cache entries,
DRAM allocation) and writes one JSON record with the per-request results and the trace summary.
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

BOOK = Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
PREFILL_RE = "qwen3-vllm prefill "
DECODE_RE = "qwen3-vllm decode "


def parse_kv(line: str) -> dict:
    out = {}
    for key in ("starts", "ends"):
        match = re.search(key + r"=(\[[^\]]*\])", line)
        if match:
            out[key] = json.loads(match.group(1))
            line = line.replace(match.group(0), "")
    for part in line.split(" "):
        if "=" in part:
            key, _, value = part.partition("=")
            out[key] = value
    return out


def stream_completion(server, model, ids, max_tokens, **params):
    payload = {"model": model, "prompt": ids, "max_tokens": max_tokens, "stream": True, "temperature": 0.0}
    payload.update(params)
    started = time.perf_counter()
    ttft = None
    gaps = []
    last = started
    text = []
    finish = None
    n_chunks = 0
    with requests.post(f"{server}/v1/completions", json=payload, stream=True, timeout=1800) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            piece = choices[0].get("text") or ""
            finish = choices[0].get("finish_reason") or finish
            now = time.perf_counter()
            if piece or choices[0].get("logprobs"):
                n_chunks += 1
                if ttft is None:
                    ttft = (now - started) * 1000.0
                else:
                    gaps.append((now - last) * 1000.0)
                last = now
            text.append(piece)
    return {
        "prompt_tokens": len(ids),
        "max_tokens": max_tokens,
        "params": params,
        "ttft_ms": ttft,
        "e2e_ms": (time.perf_counter() - started) * 1000.0,
        "chunks": n_chunks,
        "max_gap_ms": max(gaps) if gaps else None,
        "mean_gap_ms": statistics.mean(gaps) if gaps else None,
        "finish_reason": finish,
        "text": "".join(text),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--server-log", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    server = args.server_url.rstrip("/")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    text = bz2.open(BOOK, "rt", encoding="utf-8").read()
    book = tok.encode(text[:400000], add_special_tokens=False)

    def ids(length, offset=0):
        return book[offset : offset + length]

    log_lines_before = len(args.server_log.read_text(errors="replace").splitlines())
    record = {"server_url": server, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phases": {}}

    def run(label, jobs, concurrent):
        t0 = time.perf_counter()
        if concurrent:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                results = list(pool.map(lambda j: stream_completion(server, args.model, *j[0], **j[1]), jobs))
        else:
            results = [stream_completion(server, args.model, *j[0], **j[1]) for j in jobs]
        record["phases"][label] = {
            "concurrent": concurrent,
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "requests": results,
        }
        print(
            f"{label}: {len(results)} requests, wall {(time.perf_counter() - t0):.1f} s, ttft {[round(r['ttft_ms'] or -1) for r in results]}",
            flush=True,
        )
        return results

    # 1. unseen logical lengths, solo (first encounters)
    run("solo_unseen_lengths", [((ids(n, 1000), 16), {}) for n in (129, 130, 131, 5, 33, 257, 1025)], concurrent=False)
    # 2. concurrent burst of mixed lengths (multi-row prefill, concurrent decode)
    burst = [31, 127, 128, 640, 1000, 1536, 2049, 4097]
    run("concurrent_mixed_lengths", [((ids(n, 2000 + 7 * i), 64), {}) for i, n in enumerate(burst)], concurrent=True)
    # 3. prefix-cache resumes at nonzero start while another request decodes
    base = ids(1000, 9000)
    long_decode = [((ids(300, 30000), 400), {})]
    resumes = [((base, 32), {}), ((base + ids(200, 10000), 32), {}), ((base + ids(1, 10200), 32), {}), ((base, 8), {})]
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut = pool.submit(run, "background_decode", long_decode, False)
        time.sleep(2.0)
        run("prefix_resume_cold_then_hits", resumes, concurrent=False)
        run("multi_chunk_prompt_during_decode", [((ids(5000, 12000), 16), {})], concurrent=False)
        fut.result()
    # 4. sampling-mode switches on one prompt; greedy must reproduce after sampled and seeded requests
    p129 = ids(129, 1000)
    modes = [
        ((p129, 32), {}),
        ((p129, 32), {"temperature": 0.7, "top_k": 20, "top_p": 0.95}),
        ((p129, 32), {}),
        ((p129, 32), {"temperature": 0.7, "top_k": 20, "top_p": 0.95, "seed": 7}),
        ((p129, 32), {"temperature": 0.7, "top_k": 20, "top_p": 0.95, "seed": 7}),
        ((p129, 32), {}),
    ]
    res = run("sampling_mode_switches", modes, concurrent=False)
    record["sampling_mode_checks"] = {
        "greedy_repeats_identical": res[0]["text"] == res[2]["text"] == res[5]["text"],
        "same_seed_identical": res[3]["text"] == res[4]["text"],
        "sampled_differs_from_greedy": res[1]["text"] != res[0]["text"],
    }
    # 5. known lengths in a new order and batch combination, concurrently
    run(
        "reordered_repeat_burst",
        [((ids(n, 2000 + 7 * i), 48), {}) for i, n in reversed(list(enumerate(burst)))]
        + [((ids(n, 1000), 16), {}) for n in (131, 129)],
        concurrent=True,
    )
    # 6. host-sampled requests (logprobs, penalties) beside a device-sampled one
    run(
        "host_sampling_beside_device",
        [
            ((ids(129, 1000), 24), {"logprobs": 1}),
            ((ids(200, 5000), 24), {"repetition_penalty": 1.2, "temperature": 0.7}),
            ((ids(257, 7000), 24), {}),
        ],
        concurrent=True,
    )
    # 7. repeat the first solo lengths: must be identical to the first encounter
    again = run(
        "solo_unseen_lengths_repeat",
        [((ids(n, 1000), 16), {}) for n in (129, 130, 131, 5, 33, 257, 1025)],
        concurrent=False,
    )
    first = record["phases"]["solo_unseen_lengths"]["requests"]
    record["repeat_identical"] = [a["text"] == b["text"] for a, b in zip(first, again)]

    # server-log evidence written during the workload
    lines = args.server_log.read_text(errors="replace").splitlines()[log_lines_before:]
    prefill = [parse_kv(l[l.index(PREFILL_RE) :]) for l in lines if PREFILL_RE in l]
    decode = [parse_kv(l[l.index(DECODE_RE) :]) for l in lines if DECODE_RE in l]

    def total(entries, key):
        return sum(int(e.get(key, 0)) for e in entries)

    caches = [int(e["program_cache"]) for e in prefill + decode if "program_cache" in e]
    dram = [float(e["dram_alloc_mb"]) for e in prefill if "dram_alloc_mb" in e]
    all_requests = [r for ph in record["phases"].values() for r in ph["requests"]]
    ttfts = sorted(r["ttft_ms"] for r in all_requests if r["ttft_ms"] is not None)
    gaps = [r["max_gap_ms"] for r in all_requests if r["max_gap_ms"] is not None]
    pct = lambda p: ttfts[min(len(ttfts) - 1, int(round(p * (len(ttfts) - 1))))] if ttfts else None
    record["summary"] = {
        "requests": len(all_requests),
        "completed": sum(1 for r in all_requests if r["finish_reason"] in ("length", "stop")),
        "ttft_ms_p50": pct(0.5),
        "ttft_ms_p95": pct(0.95),
        "ttft_ms_p99": pct(0.99),
        "max_decode_gap_ms": max(gaps) if gaps else None,
        "prefill_calls": len(prefill),
        "prefill_rows_per_call": [int(e["rows"]) for e in prefill],
        "prefill_starts": [e.get("starts") for e in prefill],
        "prefill_ends": [e.get("ends") for e in prefill],
        "nonzero_start_rows": sum(1 for e in prefill for s in (e.get("starts") or []) if s > 0),
        "decode_steps": len(decode),
        "decode_buckets_used": sorted({int(e["bucket"]) for e in decode}),
        "decode_modes": {m: sum(1 for e in decode if e.get("mode") == m) for m in ("device", "host")},
        "reset_batch_steps": sum(1 for e in decode if e.get("reset_batch") == "True"),
        "decode_trace_captures": total(decode, "decode_trace_captures") + total(prefill, "decode_trace_captures"),
        "sampling_trace_captures": total(decode, "sampling_trace_captures") + total(prefill, "sampling_trace_captures"),
        "eager_decode_steps": total(decode, "eager_decode_steps"),
        "seeded_eager_sampling_steps": total(decode, "seeded_eager_sampling_steps"),
        "program_cache_entries_first": caches[0] if caches else None,
        "program_cache_entries_last": caches[-1] if caches else None,
        "program_cache_entries_max": max(caches) if caches else None,
        "dram_alloc_mb_first": dram[0] if dram else None,
        "dram_alloc_mb_last": dram[-1] if dram else None,
        "dram_alloc_mb_max": max(dram) if dram else None,
        "out_of_vocab_lines": sum(1 for l in lines if "sampled ids outside" in l),
        "trace_release_lines": sum(1 for l in lines if "release_trace" in l or "retire" in l.lower()),
        "error_lines": [l[:200] for l in lines if "Traceback" in l or "ERROR" in l][:10],
    }
    record["log_lines_before"] = log_lines_before
    record["log_lines_after"] = log_lines_before + len(lines)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["summary"], indent=2))
    print(
        json.dumps(
            {"sampling_mode_checks": record["sampling_mode_checks"], "repeat_identical": record["repeat_identical"]}
        )
    )


if __name__ == "__main__":
    main()
