# Optimized decoder work log

## Scope and hardware discipline

Only `tt/optimized_decoder.py`, its tests, `doc/context_contract.json`, and `doc/optimized_decoder/` were changed. No multichip, full-model, LM-head, serving, or vLLM work was started.

`tt-smi -ls --local` showed four healthy Blackhole p300c chips before candidate work and final gates. All TT commands were serialized and isolated with:

```bash
TT_VISIBLE_DEVICES=0000:01:00.0,0000:02:00.0
TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal
```

An unrelated Laguna process on the other board was not touched. Watcher and Tracy were never enabled together. No reset/recovery was required.

## Topology-first audit

The initial fused measured path was inspected for packed same-input projections, repeated matmuls, SDPA/composites, layout boundaries, recurrent GDN operations, cache traffic, and the dense MLP. The compact decisions are in `candidates/index.csv`; detailed earlier rounds are retained under `candidates/`, `AUTODEBUG.md`, and `AUTOFIX.md`.

The final independent review round found seven gaps and is frozen in `stage_review_3.md`: phase inference incorrectly treated full seq-32 prefill as decode; 8/16/64-core MLP and linear input projection had stopped too early; mixed precision had not been crossed with selected sharding; fidelity labels were stale; GDN prefill/update matmuls lacked explicit configs; and representative larger prefill lacked profiler evidence.

AutoFix repaired them as follows:

- Added explicit `prefill_forward`/`decode_forward` phase scopes. Full seq-32 now uses prefill BFP4/BFP4/BFP8, while decode uses BFP8/BFP8/BFP8.
- Added per-role core/block controls and padded 6144/18432 support. Eight-core block-1 and 16-core block-1 became legal but lost at 1749.951 and 1689.438 us full decode. Padded 64-core block-1 and 3/3/9 retries both reached the installed program and failed with `bad optional access`; neither was rejected on the original padding error.
- Repaired the padded 40-core linear input through shape slicing and required interleaved-consumer conversion. It passed at 1561.389 us and lost to output-only at 1555.930 us.
- Crossed final projection fidelity: LoFi/HiFi2/auto were 1557.698/1597.812/1555.930 us linear and 1214.390/1341.780/1213.799 us full. Auto is selected.
- Crossed selected full decode sharding with mixed BFP4/BFP4/BFP8. Real-weight traced PCC was 0.994284, so BFP8 decode weights remain selected.
- Added actual GDN update programs. `reuse48` passed at 1501.806 us; `reuse96m` won at 1497.917 us; `reuse96n` reached validation and failed because N=4 did not equal per-core N=2.
- Reimplemented the fused chunk recurrence exactly while routing all eight matmuls through legal FP32 reuse configs. The first retry exposed a register constraint (`subblock_h * subblock_w <= 4`); after repair, PCC passed and seq-32 prefill improved 6693.005 -> 5742.163 us.
- Added optimized-only warmed seq-128 linear and 257-logical/384-physical full chunked profiler tests. Final kernels are 6628.735 us / 307 ops and 5339.789 us / 112 ops.
- AutoFix round 5 addressed review 4: the ten inherited inverse matmuls now use the selected FP32 reuse program. Seq-32 host latency improved from 6643.707 us with automatic inverse matmuls to 5604.958 us, and final linear-prefill kernels fell from 3580.546 to 2752.344 us.
- Implemented a phase-scoped generic 2D prefill-program candidate and crossed legal block widths. Linear seq-128 2D candidates were 11848.602-12502.804 us versus auto 10619.324; block 8 was invalid. Full chunked 2D was 12276.011 versus auto 9363.200 us. Auto is selected for generic projection/MLP rows.
- AutoFix round 6 addressed review 5 by moving the decode residual to a rectangular 32-core width-sharded L1 layout before input RMSNorm. Final profiles replace the approximately 85.2-us one-core norm with a 1.4-us conversion and 6.7-us 32-core norm. The same residual layout is carried through projection input, post-attention norm, and MLP; redundant conversions are equality-guarded.
- A real full-decode Q/K candidate kept the height-sharded head output into native RMSNorm. The installed device op rejected it with `Height sharded inputs are not supported`. Because head creation and decode RoPE/cache require height sharding, the interleaved normalization boundary is unavoidable in this runtime. The exact failure is `candidates/autofix6/sharded_qk_norm_rope_full_attention.log`.

Issue #50475 remains an open upstream request for optimized integrated GDN and gated-attention prefill/decode kernels. The installed generic `gated_delta_attn_seq` adapter was exercised after initial API failures: its kernel row was 141.937 us, but conversion, inverse, padding, tilize/untilize, and transpose traffic made the whole path 344 ops / 6482.619 us. It was rejected on end-to-end evidence.

## Final correctness and Watcher commands

```bash
cd /tmp
env TT_VISIBLE_DEVICES=0000:01:00.0,0000:02:00.0 \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
  TT_METAL_CACHE=/tmp/qwen36_opt_final_v5_correctness_cache \
  PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  pytest -q .../tests/test_optimized_decoder.py -k 'not perf' \
  --junitxml=.../doc/optimized_decoder/final_selected_v5/correctness/junit.xml -s
```

Result: 11 passed, 6 deselected in 306.16 s. The optimized path includes non-aligned lengths, representative linear/full layers, paged trace decode, forced chunking, batch 32, deterministic repeated replay, and the 262,144-token context contract. Decode tests assert that the sharded input-norm counter is nonzero.

```bash
cd /tmp
env TT_VISIBLE_DEVICES=0000:01:00.0,0000:02:00.0 \
  TT_METAL_WATCHER=10 TT_METAL_LOGGER_LEVEL=Info \
  TT_METAL_LOGS_PATH=.../final_selected_v5/watcher/generated \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
  TT_METAL_CACHE=/tmp/qwen36_opt_final_v5_watcher_cache \
  PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  pytest -q .../tests/test_optimized_decoder.py \
  -k 'full_attention_non_aligned_prefill_decode or full_attention_paged_decode_trace or full_attention_forced_chunked_prefill or linear_attention_non_aligned_prefill_decode or real_weight_batch32_prefill_decode' -s
```

Result: 6 passed, 11 deselected in 144.19 s. The 1,107-line generated Watcher log contains no Watcher error/assert/hang/timeout, NoC error/timeout, kernel assert, device hang, unreachable, or `TEST FAILED` match. Both devices detached and closed normally. Binding-level nanobind leak diagnostics occur after passing teardown and are not a device fault or runtime fallback.

## Final E2E and profiler commands

Five-sample unprofiled runs used `test_optimized_decoder_perf` with ten decode replays. Exact logs are in `final_selected_v5/e2e/`. Medians are 5629.998/1456.961 us linear and 1622.408/1135.458 us full (prefill/decode). The unchanged fused-control medians are 7319.387/2911.526 and 2461.871/2301.998 us.

Profiler template (phase, replays, layer, and output varied):

```bash
cd /tmp
env TT_VISIBLE_DEVICES=0000:01:00.0,0000:02:00.0 \
  QWEN36_PERF_PHASE=PHASE QWEN36_DECODE_REPLAYS=REPLAYS \
  PANDAS_FUTURE_INFER_STRING=0 \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_CACHE=/tmp/qwen36_final_v5_tracy_cache \
  PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler:/home/ttuser/dev/qwen-perf/tt-metal \
  python -m tracy -p -r -v --check-exit-code --no-runtime-analysis \
  --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
  -o ARTIFACT_DIR -m pytest -q TEST_NODE -s
```

`test_optimized_linear_attention_large_prefill_perf` and `test_optimized_full_attention_large_prefill_perf` were also captured. Each raw operation CSV was filtered with advice enabled:

```bash
tt-perf-report RAW.csv --start-signpost START --end-signpost END \
  --no-color --csv report.csv --summary-file report_summary > report_generation.log
tt-perf-report RAW.csv --start-signpost START --end-signpost END \
  --no-color --no-summary > report.txt
```

| Capture | Kernels | Gaps | Ops | Host under Tracy |
|---|---:|---:|---:|---:|
| linear prefill | 2751.686 us | 3050.375 us | 159 | 6171.125 us |
| linear decode | 1218.077 us | 237.904 us | 73 | 1625.254 us |
| full prefill | 1196.264 us | 416.399 us | 31 | 1848.420 us |
| full decode, 10 replays | 10671.908 us | 725.679 us | 470 | 1155.818 us/replay |
| linear seq-128 prefill | 6628.735 us | 3835.312 us | 307 | 10885.080 us |
| full chunked prefill | 5339.789 us | 337.034 us | 112 | 9556.945 us |

Raw `ops_perf_results*.csv`, advice-enabled `report.csv`, human-readable `report.txt`, `report_summary.csv/png`, generation logs, and console logs are retained in `final_selected_v5/profiler/`. Reproducible duplicate `.logs`, `profile_log_device.csv`, and host `.tracy` databases were removed after compact artifacts were verified, reducing the profiler artifacts to 5.8 MB.

## Advice and movement audit

Advice to use DRAM-sharded decode matmuls was implemented for MLP and selected projections. Explicit LoFi and HiFi2 were both measured on final topology; automatic fidelity won. GDN matmul subblocks were increased to the largest legal FP32 register footprint. Advice suggesting decode-only DRAM-sharded programs on height-64/128 prefill is incompatible with the installed tile-height contract; legal recurrent reuse configs and the padded 64-core attempts provide the before/after evidence.

The final report still names framework layout operations. The decode residual now converts once to width-sharded L1; input RMSNorm is 6.7 us on 32 cores, and the residual stays sharded through the tail. Full head creation emits HEIGHT_SHARDED Q/K; installed RMSNorm rejects that layout, so its two layout conversions are an exact runtime boundary before height-sharded cache/SDPA. GDN convolution/state uses fixed recurrent formats. The lower-movement dedicated GDN kernel was carried past API failures, but its required preparation/inverse/padding/tilize/untilize/transpose adapter expanded to 344 ops and 6482.619 us. Candidate-only padded linear-input slicing/conversion is disabled by the selected output-only policy. Measured runtime methods contain no torch, `from_torch`, `to_torch`, host fallback, explicit tilize/untilize, or explicit reshard call.

## Artifacts and review

- `stage_review_1.md` through `stage_review_5.md`: independent findings and dispositions.
- `final_selected_v5/correctness/`: exact-source JUnit and console.
- `final_selected_v5/watcher/`: JUnit, console, and generated Watcher log.
- `final_selected_v5/e2e/`: five samples per layer.
- `final_selected_v5/profiler/`: six raw/filtered profiler captures with CSV and text tables.
- `final_selected_v5/blackhole_required_byte_floor.csv`: explicit Blackhole 512 GB/s lower bounds for the final precision policy.
- `final_selected_v5/same_run_accounting.csv`: host, device-kernel, operation-gap, and residual accounting from each final Tracy capture.
- `candidates/autofix4/`: review-3 geometry, fidelity, mixed-precision, GDN, and padded-projection evidence.
- `stage_review_1.md`, `stage_review_2.md`, `stage_review_3.md`: independent findings and prior dispositions.

The final independent `$stage-review` verdict and local stage commit SHA are recorded after review at the end of this file.
