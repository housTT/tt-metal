# Functional-decoder work log

## Startup, 2026-10-09/10 UTC

- Read parent `AGENTS.md`, `../source-audit.json`, pinned HF config, installed
  Transformers 5.12.1 model source, model-bringup startup, functional-decoder,
  tt-device-usage, tt-enable-tracing, AutoDebug/AutoFix and telemetry contracts.
- Enabled plugin inventory verified in `/home/hous/.codex/config.toml`.
- All model commands source `../scripts/environment.sh` (local `python_env`).
- Existing untracked autoport files were an initial handoff; no device evidence.
- `timeout 60 tt-smi -ls --local`: four Blackhole ASICs; `device_list.log`.
- `timeout 90 python` opening/closing `MeshShape(1,1), trace_region_size=0`:
  passed; `mesh_smoke.log`, grid11×10. No reset or process termination.
- Nonfatal startup diagnostics: unknown B850M-C grouping falls back to PCI bus;
  memory-statistics shared-memory object belongs to another user and cannot be
  opened. Mesh and device operations still run; watcher classification pending.

## Initial validation and repairs

- Inspected rank3 single-device attention versus rank4 sparse MoE consumer.
  Added an autoport-local reshape adapter; no shared-kernel source modifications.
- The inherited GDN `valid_len` route constructs torch masks inside prefill.
  Aligned chunks now omit it; arbitrary tails advance exact real tokens via
  the recurrent/paged decode path. Added device RoPE lookup and request slots.
- `python -u -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer --layer 3
  --output .../full_smoke.json`: real weights, length128; PCC .998624742/.998600841.
- Same command with `--layer 0`: PCC .996776879/.998699129.
- Both layer commands with `--length 129 --trace`, tracking enabled and program
  cache allocations included: passed PCC and identical replay; `*_trace129.log`
  and `*_trace129.json`. This is a preliminary check, not all stage gates.
- AutoDebug launcher preflight failed because unprivileged namespaces are disabled.
  Retried per skill with `AUTODEBUG_SKIP_CHILD_SANDBOX=1` under existing unrestricted
  user authorization. Logs: `autodebug_launcher.log`, `autodebug_retry.log`.
  Inspection-only investigation ongoing; no device use by child.

## Required work still active

Long-context and near-context tails; same-instance increasing requests and prefix
continuation; batch>1/32; updated trace inputs and ownership; runtime audit;
synthetic weight statistics; watcher; 4096/128 warmed device profiles and human
tables; final context contract, telemetry, independent clean-pass and local commit.

No local stage checkpoint commit yet. Never push.

## Exact GDN and HF norm repair, 2026-10-10

- AutoDebug finished (`AUTODEBUG.md`); the isolated AutoFix hypothesis agent
  added `tt/functional_gdn.py`. All device experiments remain serialized here.
- Native exact chunk recurrence, FP32 recurrent state, BF16 real-checkpoint
  projections and FP32 FIR accumulation improved GDN4096/128, but two individual
  decode tokens still failed. BF16 expert weights did not resolve either.
- HF zero-centered RMSNorm computes `(1 + weight)` and normalization in FP32;
  the inherited path rounded the offset in BF16. Matching the HF norm order
  resolves all128 per-token failures: `linear_native_fp32norm.json`, prefill
  .999698281 and minimum decode .995265305. The corresponding full-attention
  final run passes at .998588681 / .995969355 (`full_final_4096_128.json`).
- Both commands: `TT_METAL_TRACE_ALLOC_TRACKING=1 python -m
  models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer --layer {0,3}
  --length 4096 --steps 128 --trace --audit --output <artifact>.json`.
  Real weights, inherited sparse expert precision, repeated identical replay.
- CPU contract tests: `python -m pytest -q
  models/autoports/qwen_qwen3_5_35b_a3b/tests/test_functional_contract.py`: 9 passed.
  The target config fixture now works without checkpoint downloads; real-index
  checks skip only when the pinned checkpoint is unavailable.
- Native-context GDN: `run_layer --layer 0 --length 262143 --steps 1
  --sparse-reference --trace --audit`, allocation tracking on, passes prefill
  .999336362 and decode .997513533 at position262143. All TT tokens execute;
  HF retains every recurrent contribution, compares16 selected prefill outputs.
  `linear_context262143.{json,log}`. Full-attention counterpart in progress.

## Stronger acceptance and review repairs, 2026-10-10

- Initial independent review is `stage_review_initial.md`, verdict more-work-needed.
  Unseen logical length36 and continuation start36 inside prepared physical128
  buckets exposed a static slice cache miss. Resident device indices, embedding
  reads, fixed output writes and tensor-valued chunk starts repair this. Same
  trace stays live across17 B2 cases, including withheld36, multiple chunks,
  nonzero slots, page permutation, and a return to a short request.
- New heterogeneous-position B32 test exposed two GDN outliers despite the older
  aligned4K pass. `positions_linear_b32.json` records both failures; component
  controls reproduce row18 in isolation. Router precision, BF16 normalization,
  FP32 SiLU/sigmoid alone, and native recurrent prefill do not solve it. Their
  `components_positions18*.json` artifacts remain as failed controls.
- The local FP32 SFPU single-token recurrence fixes both decode and arbitrary
  prefill tails; it avoids repeated native FPU TF32 state quantization. Detailed
  source proof and attribution limits are in `AUTOFIX.md`. `positions_linear_sfpu`
  passes all32x8 outputs (minimum .998062253), replay equality and runtime audit.
  `linear_accept_4096_128` passes .999701798 prefill / .997864842 minimum decode.
- Flexible chunk SDPA requires32-byte page-table sticks. The132-column table in
  the full4K128 test exposed this (`full_page_table_unaligned_failure.log`).
  Device-side padding repaired the common path: `full_accept_4096_128.json`
  passes .998596847 prefill / .995969236 minimum decode. A follow-up review found
  the small-pool update/read edge case; its narrow repair and watcher tests are
  still pending at this log entry.
- `python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_acceptance`
  runs hardware checks serially and records each exact command, environment,
  exit code and console path in `acceptance_commands.json`. First failure is
  preserved separately in `acceptance_first_failure.json`. The source remains
  within the functional autoport; no device reset or process kill was needed.


## Final page-pool repairs and instrumentation, 2026-10-10

- The full-attention cache update now receives the original logical page table;
  only chunked attention reads receive a device-padded table. Odd multiples of
  32 tokens use a 32-token SDPA decode chunk to avoid reading a nonexistent page.
  Exact pools of three pages at B1/B2 and nine pages at B2 pass real-weight PCC,
  trace allocation tracking, repeated replay, runtime audit and watcher checks.
  Artifacts: `small_pool3_b1`, `small_pool3_b2`, `small_pool9_b2` JSON/log pairs.
- `format_provenance.json` records AST-equivalent Black formatting. Afterwards,
  the new odd-pool adapter exposed unsupported TTNN Shape slicing (`x.shape[:2]`);
  this was changed to indexed dimensions (`x.shape[0]`, `x.shape[1]`). The first
  failure remains in `small_pool_adapter_shape_failure.log`. Thus the current
  attention source hash intentionally differs from the formatting record.
  `full_reuse_postreview.json` and the small-pool runs validate the fixed source.
- `cpu_contract_postreview.log`: 9 passed. `pytest_device_accept.log`: 4 passed
  (real and statistics-matched synthetic weights, both layer kinds). Nanobind
  shutdown reference-count warnings follow successful device/driver close;
  these concern CoreRangeSet/MemoryConfig bindings, not a reported live tensor
  or trace allocation leak. Watcher classification is in `watcher_summary.json`.
- The first final linear profile passed numerical checks but overflowed the
  default 1000-program profiler buffer (12000 uint32 entries per RISC), dropping
  markers. Postprocessing correctly rejected missing device op 1032195. It is
  NOT latency evidence. Preserved as `linear_profile_overflow.{json,log}` and
  `tracy/linear_attention_overflow/`; the original command/exit 1 remains in
  `completion_commands.json` (its original output paths precede this rename).
- Read-only source diagnosis counted 4866 programs per measured prefill and
  166 programs per replay, with 130 replays including two determinism controls.
  `run_profiles.py` retries serially with `--op-support-count 32768` and
  `--dump-device-data-mid-run`. Explicit `ReadDeviceProfiler` calls drain between
  complete passes, outside each timed window and outside trace capture. No
  missing marker is filtered or substituted. `profile_commands.json` records
  retry commands, exits and paths. Production decoder source is unchanged.

- The repaired linear profile exits 0: 4866 measured prefill programs and 166
  programs in each of 128 measured replay windows, all with device timestamps.
  Its whole-layer times are 1743870.348148 us prefill and 2190.255232 us mean decode.
  Current source again passes .999701798 / .997864842 PCC and identical replay.
- Full profiling then encountered a real 16-bit instrumentation hash collision:
  SDPA `compute_streaming.hpp` labels `Reduce max` at line 1691 and
  `Softmax(Q@KT)@V` at line 1770 both hash to 2450 under this checkout path.
  The failed command exits 4; device drivers close. Logs/capture are preserved as
  `full_profile_hash_collision.log` and `tracy/full_attention_hash_collision/`.
  `profile_commands_first_retry.json` retains both the linear success and full
  failure. No profiling assertion was disabled or source marker removed.
- `TT_METAL_RUNTIME_ROOT=/tmp/q35-functional-profiler-root` points through a
  symlink to the identical checkout; `TT_METAL_CACHE=/tmp/q35-functional-profiler-cache`
  isolates recompilation. `profiler_path_alias.json` records the resolved path,
  source hash and collision precheck. All 31 known markers have distinct hashes
  with the alias. Command: `python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_profiles
  --layer-type full_attention` with those two environment overrides after sourcing
  the normal environment. This changes `__FILE__` identities only, not kernels.


## Completed execution gates, 2026-10-10

- The first alias attempt retained CWD-resolved SDPA include paths and failed
  again; preserved in `full_profile_alias_cwd_failure.log`, its matching Tracy
  directory and `profile_commands_second_retry.json`. The final attempt uses
  both runtime-root and kernel-path aliases, an evidence-directory CWD and an
  absolute checkpoint path. Actual emitted markers are distinct (27031/34925).
  The source diagnosis is `profiler_hash_diagnosis.md`.
- Both final captures exit 0 and reproduce the accepted headline PCC values and
  identical replay. Full-attention source hashes match the final implementation.
  `profile_commands_first_retry.json` contains accepted linear provenance;
  `profile_commands.json` contains accepted full provenance. Both contain the
  exact workload 4096/128, batch 1, one request. No watcher runs overlap profiling.
- Complete-layer device latency (prefill / mean traced decode), in microseconds:
  - linear_attention: 1743870.348148 / 2190.255232.
  - full_attention: 1720531.702222 / 1608.711806.
- `python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.process_profiles
  --layer-type {linear_attention,full_attention}` separately generates each
  layer's human-readable tables, filtered CSVs, lossless source CSV archive,
  whole-layer measurement and roofline estimates. Exact executable commands
  appear in each `processing_provenance.json`; tt-perf-report version 1.4.1.
  `functional_decoder.md` records units, all estimates and known tool warnings.
- `python -m black --target-version py312 --include '\.py$' --line-length 120
  --check <autoport>/tt <autoport>/tests`: 20 files unchanged (`format_check.log`).
  `python -m compileall -q <autoport>/tt <autoport>/tests`: passed. The explicit
  include override is needed because the repo default omits autoport directories.
- `source_manifest.json` records final source/test hashes. Prior numerical
  failures remain diagnostic artifacts; README acceptance tables identify the
  final gates. Advertised 262144-token context is preserved without reduction.
  Final independent review and local commit bookkeeping remain pending here.

- Review classified the profiler's stale `PROGRAM CACHE HIT=False` metadata.
  `profile_integrity.json` proves zero new measured prefill program hashes
  (105 linear / 84 full unique hashes already warmed), plus successful traced
  replay, PCC and zero dropped-marker warnings. Source lines and limits are
  documented in `functional_decoder.md`; no assertion was weakened or rerun
  selected by its timing outcome. Both complete CSV archives decompress exactly.


## Stage closure

The fresh independent final review returned **clean-pass**, with no required work
remaining: `stage_review_final.md`. It independently re-derived correctness,
context, trace lifetime, watcher, whole-layer timings, roofline inputs and packet
consistency. Earlier findings are fixed and covered by targeted reruns.

The compact packet is saved at
`/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/9c898890-7c79-49e8-a4b8-ff3c6515e5a9.json`.
It uses the exact supplied template identifiers and workload; both layer types
have measured device times and explicitly estimated rooflines. Source and report
files are committed locally; raw oversized Tracy captures remain on disk, with
lossless exported CSV archives in Git. Only functional-decoder work is included.
No optimized-decoder, multichip, full-model or serving stage has been started.
No push, reset, firmware change or unrelated process termination occurred.

Local commit SHAs are recorded in the following bookkeeping entry and in the
external packet after the commits exist.

Git initially rejected the commit because no author identity is configured in
this checkout or the user's inspected development checkouts. These authorized
local commits use command-scoped `user.name=Codex`, `user.email=codex@localhost`;
no global identity configuration is changed.


### Commit-hook evidence preservation

The initial commit-hook run attempted to trim original logs/tables and format
historical source snapshots, and rejected required evidence larger than 500 KB.
Those evidence changes were restored byte-for-byte from the reviewed staged
snapshot. Current code keeps hook-driven formatting and unused-import cleanup.
`commit_format_provenance.json` proves all production ASTs unchanged and test
changes limited to imports; no layer arithmetic, control flow or test gate changed.
`source_manifest.json` maps the final bytes through this provenance to measured
source. The target-config edit only adds its missing final newline.

`python -m pre_commit run --files <all 20 current tt/tests Python files>
<target_config.json> <models/autoports/__init__.py>` passes all applicable hooks
(`precommit_source.log`). Final commit commands set
`SKIP=trailing-whitespace,end-of-file-fixer,black,autoflake,isort,check-large-files`
only to preserve immutable raw evidence/snapshots and retain the required full
profiler artifacts. These six checks passed on current code separately; all other
applicable commit hooks remain enabled. This is a local artifact exception, not
a global hook/configuration change. The independent reviewer checks this narrow
formatting update before the commit proceeds.

Post-hook CPU contracts: `python -m pytest -q <autoport>/tests/test_functional_contract.py`: 9 passed (`cpu_contract_commit_format.log`).

The final reviewer independently checked the commit-hook cleanup and preserved **clean-pass** in the final report addendum. No required work or hardware rerun remains.
