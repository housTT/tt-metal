# Fused-decoder work log

## Scope and device

- Branch start: `85b1099e34b` (`Record Qwen functional decoder evidence checkpoint`).
- Functional starting point: `tt/functional_decoder.py` and its completed
  correctness/performance evidence.
- Stage-owned code: `tt/fused_decoder.py` and
  `tests/test_fused_decoder{,_perf}.py`.
- Stage-owned documentation/evidence: this directory.
- Hardware: one local P300c Blackhole, logical chip 0, 1x1 mesh.
- Every TT command sourced `doc/functional_decoder/ttenv.sh`, which pins
  `TT_VISIBLE_DEVICES=0`.
- No optimized-decoder, multichip, full-model, generator, or vLLM work began.

The starting context contract advertised context 262144, page size 64,
prefill chunk 128, and decode batch 32. None changed.

## Functional baseline

| Layer | Prefill PCC | Decode PCC | Prefill host ms | Decode host ms | Prefill device us | Decode device us |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.99871051 | 0.99997765 | 24.047593 | 5.730349 | 23790 | 5396 |
| 1 | 0.99910986 | 0.99991739 | 27.598879 | 6.701143 | 27269 | 5922 |
| 3 | 0.99679226 | 0.99988198 | 408.372929 | 11.341126 | 408094 | 11081 |

Host values are medians of seven complete runs using ten traced-decode
replays. The unedited transcript is
`functional_perf_10replay_count7.raw.log` (21 passed).

## Fusion passes

1. Added `FusedDecoder` with a source-visible manifest and no fallback switch.
2. Packed every profitable shared-LHS projection group; setup-folded scales
   and the GDN packed bias.
3. Replaced the primitive router normalization with top-k logits plus selected
   softmax; batched all expert-down groups in one A-sparse/B-dense call and
   moved routed/shared gates before their down projections.
4. Added persistent split GDN/PLE decode state, recurrent norm-scale folds,
   ternary arithmetic, and the fused chunk GDN kernel.
5. Adapted `qkv_causal_conv1d_silu` for GDN prefill, swept channel chunks
   320/640/1280, and retained 640.
6. Packed PLE projections, removed its dead decode reshape, and retained split
   dilation taps after a batched-dot candidate regressed.
7. Packed QSA projections, adopted HF partial RoPE, removed inverse V/Q/head
   transforms and concat-heads, and used dedicated prefill/decode head split.
8. Replaced primitive gathered attention with native 24:2 GQA SDPA and
   dedicated decode SDPA; swept K chunks and retained prefill 64/decode 32.
9. Replaced packed two-head cache gather with per-head gather, eliminating
   padded dimension-2 intermediates and 24-head K/V materialization.
10. Promoted fused K/V paged update, static compressed address/rotary caches,
    and a persistent normalized/RoPE-applied compressed index-key cache.
11. Exhausted dedicated KDA/indexer/sparse attention, sharding, structural,
    activation, bias/scale, sparse indexed, and movement alternatives. The
    exact disposition matrix is `graph_inventory.md`; every executable
    candidate is indexed in `candidates/README.md`.

## Last bounded candidate sweep

The independent review supplied a bounded final remainder; each item was
resolved before final gates:

- Fixed-320 on-device indexed sparse MoE: correct but
  78.541340/14.560467 ms on layer 3; rejected.
- Direct attention output reshape and V/Q inverse-TM cancellation: promoted;
  concat-heads removed.
- Hyper scalar MAC: promoted. Output-weight factor-2 fold was materially worse
  for layer-0 decode PCC and not uniformly faster.
- Redundant QSA validity multiply, dead PLE reshape, static compressed address
  precompute, router algebra, packed GDN bias, RMSNorm scale weights: promoted.
- GDN z packing plus typecast: correct but slower; rejected.
- HF partial RoPE, decode SDPA, fused paged K/V update, PLE split state, legal
  broadcast skips, persistent block RoPE/compressed index keys: promoted.
- PLE batched dot: correct but 32.126572 ms prefill; rejected.
- KDA fused sigmoid-gated RMSNorm: exact prefill math and about 0.60/0.65 ms
  faster on GDN prefill, but the valid path that retains caller-owned decode
  input tensors failed layer-1 eager/traced decode at about 0.86. AutoFix
  proved saved/prepared state equality and ruled out kernel OOB, then tested
  L1 output, deferred lifetimes and program-cache clearing without repair.
  Throwaway inputs masked the failure, demonstrating allocator/address
  sensitivity rather than a semantic state change. Rejected as an in-scope
  TTNN composition/runtime limitation; exact patches/raw artifacts are linked
  from the candidate log.
- KDA fused causal convolution: correct and about 1 ms faster; chunk 640
  promoted.
- `indexer_score_dsa`: correct but 5.588385 ms median decode versus the
  5.540206-ms retained candidate; rejected.
- Direct `sparse_sdpa`: exact math adaptation found, but no row-major arbitrary
  paged cache writer exists. Hard scoped TTNN blocker.
- Dedicated MoE/fused-router/minimal-matmul/KDA affine/fused-QK-RoPE families:
  exact binding/structure mismatches recorded in `graph_inventory.md`.

No applicable pattern remains unassessed. “Fewer ops” was never sufficient:
correct candidates were retained only when the current hardware timing won.

## Correctness and public capability

Final PCC on real checkpoint weights:

| Layer | Prefill PCC | Traced decode PCC |
| ---: | ---: | ---: |
| 0 | 0.99842572 | 0.99702853 |
| 1 | 0.99911219 | 0.99988902 |
| 3 | 0.99668270 | 0.99977344 |

Final exact-source non-long command:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -q -s \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py \
  -m 'not long_context' \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder/final_correctness_trace_alloc.xml
```

Result: 35 passed in 55.413 seconds. The first attempt after the KDA promotion
found three batch-only contract errors: sharded V padding accepted one slice
only, and index matmul did not head-broadcast at batch >1. The source was fixed
with batch fallbacks, the three focused tests passed, and the complete 35-test
gate then passed. The delivered fast batch-one graph is unchanged.

Final exact-source context command:

```bash
pytest -q -s --long-context -m long_context \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder/final_long_context.xml
```

Result: seven passed, covering exact/non-aligned maximum prefill for all three
layer kinds and traced maximum-position QSA decode. `context_contract.json`
remains byte-for-byte unchanged.

## Final host performance

```bash
QWEN38_FUSED_PERF_DECODE_REPLAYS=10 pytest -q -s --count=7 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder_perf.py \
  2>&1 | tee models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder/fused_perf_final_count7.log
```

The final robust sample used seven complete runs (21 tests, 56.61 seconds).
The functional baseline was rerun with the same seven-run/ten-replay protocol;
both raw transcripts are retained. The earlier exact-source three-run JUnit
also passes but is not used for the before/after host table.

| Layer | Fused prefill median ms | Fused traced decode median ms | Prefill reduction vs functional | Decode reduction vs functional |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 17.015016 | 3.883350 | 29.244% | 32.232% |
| 1 | 20.517265 | 4.321484 | 25.659% | 35.511% |
| 3 | 43.610069 | 5.539352 | 89.321% | 51.157% |

The rejected indexer candidate's three-run median was 5.588385 ms. The earlier
5.540206-ms observation came from the same retained compressed-cache lineage,
not a distinct rejected graph; the final seven-run median is 5.539352 ms and
does not regress it. All final and functional samples are retained in the raw
logs above.

## Tracy and tt-perf-report

Watcher and profiler were never enabled together. Each of the six captures
used one measured execution/replay and explicit signposts. Example (layer and
mode substituted for all six):

```bash
QWEN38_FUSED_PERF_MODE=decode \
QWEN38_FUSED_PERF_DECODE_REPLAYS=1 \
QWEN38_FUSED_PERF_FLUSH_PROFILER=1 \
python -m tracy -r --check-exit-code \
  -o models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder/tracy/layer3_qsa/decode_final_raw \
  -m pytest -q -s \
  'models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder_perf.py::test_warmed_prefill_and_traced_decode[3]'
```

The copied raw CSV was transformed offline:

```bash
tt-perf-report <mode>_ops.csv \
  --start-signpost FUSED_PERF_<MODE>_L<LAYER> \
  --end-signpost FUSED_PERF_<MODE>_L<LAYER>_END \
  --no-color --no-advice --no-host-ops \
  --csv <mode>_perf_report.csv \
  --summary-file <mode>_perf_report_stacked
```

| Layer | Prefill device us | Decode device us | Prefill reduction | Decode reduction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 16736.260 | 3848.370 | 29.650% | 28.681% |
| 1 | 20161.470 | 4269.910 | 26.065% | 27.898% |
| 3 | 43401.980 | 5338.730 | 89.365% | 51.821% |

The final reports show no host conversion/fallback. `tt-perf-report` warns
that sparse-matmul active-expert FLOPs and newer KDA/QSA ops are unclassified;
that affects category/roofline labeling, not device-time sums.

## Stress, watcher, and health

```bash
pytest -q -s --count=3 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py::test_decode_trace_replay_and_determinism \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder/final_stress.xml
```

Result: nine passed in 23.073 seconds.

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder/watcher_final_v5 \
pytest -q -s models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py \
  -m 'not long_context' \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder/final_watcher.xml
```

Result: 35 passed in 94.476 seconds. The error-signature audit returned no
matches; the watcher log ends with dump completion and device-0 detach.
Post-run `tt-smi` reported healthy DRAM with zero corrected/uncorrected errors.
No reset/recovery was performed.

## Static checks

```bash
black --check --target-version py312 \
  models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder_perf.py

python -m py_compile <the same three files>
```

Both pass. The formatting-only source change was made before the final
acceptance rerun so all final artifacts match the delivered hash.

## Review and commits

The first independent reviews returned `more-work-needed`. Their findings were
treated as work: group-major sparse down was implemented, replay counts were
made like-for-like, candidate/source provenance was retained, every newly
identified dedicated/structural/folding candidate was tested or contract-
blocked, batch fallbacks were repaired, and all final artifacts were
regenerated on the delivered source. The final bounded independent report is
`stage_review_clean.md` with verdict `clean-pass`; it independently reconciles
the frozen source, correctness/context/stress/watcher/Tracy evidence, all three
last candidate records, the KDA AutoFix disposition, and equal seven-run host
timing. Local commit SHAs are recorded in the follow-up checkpoint entry below.
No push is performed.

## Local checkpoint

- Functional parent: `85b1099e34bb89307dd77b913f4b74f5fdc71283`.
- Fused implementation, tests, final evidence, candidate provenance, and clean
  review: `09b97d35f4a42db1d286787ab37d0c3019ded594`.
- The checkpoint was local only; no push was performed. Black, merge-conflict,
  include, global-torch, pytest-usage, and source-policy hooks passed. The
  whitespace/EOF/autoflake/isort hooks were explicitly skipped to preserve the
  hash-reviewed byte identity of raw terminal/XML/watcher evidence and the
  frozen fused source. The generic 500 KiB hook was skipped for the six
  required raw Tracy op CSVs; their stable filtered reports and exact hashes
  are recorded in `provenance_manifest.md`.
