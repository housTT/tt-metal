# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The prior wording nit is fixed in the current `work_log.md`: it now says five BFP-named XML files cover eight measured precision-candidate rows. Local artifact enumeration and `optimization_matrix.csv` agree.
- Final watcher evidence uses `TT_METAL_WATCHER_DISABLE_ETH=1`. This is documented as the inherited P300 instrumentation control; Tensix/NoC/CB/stack watcher coverage remains enabled, the watcher XML is 4/4 passing, and the retained log has clean device close with no TT fault signature.
- Final `tt-perf-report` tables still emit generic advice on some GDN/QSA matmul rows. This is not an open gate: inherited `optimized_decoder` evidence records all six material DRAM-sharded role trials under final fidelity, and this stage adds the fractured-layout QSA DRAM-sharded HiFi2/HiFi4 reruns plus the fabric/lower-movement candidates.

## Hard-Check Gaps

No blocking gaps found.

- Final default reproduction is coherent. `final_default_perf_count7.xml` parses as 84 tests, 0 failures/errors, 42 intentional diagnostic skips, and 42 final-path passes. Re-derived medians match `baseline_perf_medians.csv`, `final_default_perf_medians.csv`, `before_after.csv`, `README.md`, and `work_log.md`: layer 0 `158.877355 ms / 2.082220 ms`, layer 1 `120.853601 ms / 2.613159 ms`, layer 3 `162.370719 ms / 3.027236 ms`.
- Source defaults match the final run contract: `COLLECTIVE_NUM_LINKS = 2`, `FABRIC_PACKET_BYTES = 8192`, `HOST_EXPERT_SLOTS = 10`, `HOST_PACKED_EXPERTS = 512`, and QSA decode defaults to `qsa_input:110,attn_out:20` on the fixed `1x2` mesh.
- Correctness gates are current and accepted. Baseline performance XML is 21/21 passing; final static/fallback is 32/32 passing; PLE exactness is 5/5 passing; fabric contract is 1/1 passing. PCC before/after remains identical for all representative layer kinds.
- The delivered path is a true fractured multichip decoder, not a single-chip/replicated fallback. Code and tests enforce `TARGET_MESH == (1, 2)`, `FABRIC_1D`, fractured residual `S = [1,1,4*M,1280]`, stack ingress/exit only, no inter-layer gather/reshard/all-reduce, and rank-local QSA/shared-expert/EP2 execution. `final_static_and_fallback.xml` covers the host-boundary whitelist and no-host-fallback collective audit.
- Host-backed semantics are exact and measured. `host_weight_contract.json` declares the permitted expert/PLE boundaries, generation-checked LRU slots, stable indexed route row, exact zero peer shard, mmap PLE row lookup, and forbidden host compute. CPU and hardware artifacts cover stale-slot, eviction/reload, failure invalidation, PLE EOS/history/reset/isolation/non-aligned rows, and real service metrics.
- Optimization candidate coverage is adequate for this stage. `optimization_matrix.csv` and candidate artifacts cover stable indexed host cache, packed capacity, device-slot capacity, threaded H2D, one/two links, 4,352/8,192-byte payloads, async CCL with adapted retries, fused MMRS with AutoFix/triage, fractured residual consumer, QSA 1D/2D/DRAM-sharded variants, row-parallel FP32, and real-activation BFP8/BFP4 precision probes.
- Profiler provenance is coherent. The three retained raw `.csv.xz` reports unpack to the documented content hashes; packed hashes also match. All six final prefill/decode `tt-perf-report` CSVs and human tables exist under `tracy_final/`, and `profiler_family_summary.csv` / `profiler_operation_audit.csv` support the claimed dense, sparse, collective, copy, and layout/TM families.
- Context and capability contracts are preserved. `context_contract.json` retains 262,144 tokens, BFP8 paged QSA KV/index cache, non-aligned logical lengths through 262,144, max-tested batch 32 for the resident path, and the scoped batch-one host-backed decode mode. All checked evidence paths resolve locally.
- No full-model or vLLM work appears in the optimized-stage source/artifacts beyond explicit “out of scope” documentation.

## Anomaly Ledger

- Observed anomaly: `final_default_perf_count7.xml` contains intentional skips for focused residual diagnostics.
  Evidence: 42 skips with messages requiring `QWEN38_MC_RUN_GDN_RESIDUAL_DIAG=1` or `QWEN38_MC_RUN_RESIDUAL_TOPOLOGY=1`.
  Affected path: diagnostic-only performance tests inside the count-seven final-default run.
  Control or comparison: separate residual artifacts (`residual_topology_current.xml`) and final static/fallback plus watcher gates cover the residual contract; final-path host-backed perf cases passed.
  Likely subsystem: pytest selection/provenance packaging.
  Investigation performed: parsed JUnit counts and checked candidate/residual artifact coverage.
  Resolution: controlled, non-gating.

- Observed anomaly: final watcher log reports nanobind leak warnings at Python shutdown.
  Evidence: `final_watcher_stress100.log.gz` tail shows nanobind leak warnings after `4 passed`.
  Affected path: Python binding teardown after completed watcher stress.
  Control or comparison: watcher XML passed, watcher stopped, device/UMD close completed, and signature scan found no NoC/assert/panic/fatal/ERISC/heartbeat/traceback/device-close failure.
  Likely subsystem: Python/nanobind object lifetime reporting.
  Investigation performed: verified line count/hash and scanned retained log.
  Resolution: controlled, non-gating.

- Observed anomaly: some passing async/QSA BFP8 candidate XMLs lack embedded stdout.
  Evidence: `candidate_async_ccl_recovered_provenance.txt` and `candidate_qsa_bfp8_recovered_provenance.txt` explain the `-s`/`junit_logging` gap.
  Affected path: candidate provenance only.
  Control or comparison: recovered command/result transcript hashes are retained; final default, profiler, watcher, static/fallback, and fabric evidence were regenerated independently.
  Likely subsystem: pytest/JUnit capture configuration.
  Investigation performed: inspected recovered provenance files and matched rows to `optimization_matrix.csv`.
  Resolution: controlled, non-gating.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md`, `.agents/skills/optimize/SKILL.md`, `.agents/skills/host-weight-cache/SKILL.md`, `.agents/skills/tt-device-usage/SKILL.md`, and the optimized multichip decoder contract in the task.
- Artifact paths: `doc/optimized_multichip_decoder/{README.md,work_log.md,optimization_matrix.csv,before_after.csv,baseline_perf_medians.csv,final_default_perf_medians.csv,baseline_perf_count7.xml,final_default_perf_count7.xml,final_static_and_fallback.xml,final_fabric_contract.xml,final_watcher_stress100.xml,final_watcher_stress100.log.gz,host_ple_exactness.xml,profiler_provenance.txt,profiler_family_summary.csv,profiler_operation_audit.csv,tracy_final/**,candidate_*.xml,*_recovered_provenance.txt,*hang_triage.txt,ple_row_cache_sweep.csv,host_pinning_probe.txt}` plus `doc/context_contract.json` and `doc/host_weight_contract.json`.
- Code paths: `tt/multichip_decoder.py`, `tt/host_weight_cache.py`, `tt/optimized_decoder.py`, `tests/test_multichip_decoder.py`, `tests/test_multichip_decoder_perf.py`, and inherited optimized-decoder work-log evidence for DRAM-sharded role coverage.
- Commands run: local read-only `sed`, `find`, `rg`, `git status --short --branch`, `git diff --name-only --stat`, Python XML/CSV/JSON parsers for pass counts, medians, and evidence paths, `sha256sum`/`xz -dc` for raw profiler hashes, and `gzip -dc` scans of the final watcher log. No TT hardware, server, vLLM, reset, or implementation test was run during this review.

## Residual Risk

- The stage satisfies the stated optimized multichip decoder contract. Remaining limitations are scoped and documented: batch-one exact host-backed decode, serialized expert/PLE host service with zero selected overlap, CPU-only Torch pinning unavailable on this host, intra-layer collectives/layout work still present, and full-model/vLLM deliberately out of scope.
