# AutoFix: TP4 mixed stacked decoder contract

## Starting evidence

The final stage review found that TP4 sliding and full attention had only been
validated as isolated layers.  The existing mixed layer-0 to layer-5 stack gate
covered TP2, so it could not prove the TP4-only R22 sliding to R0 full-attention
boundary, TP4 packed experts, full-KV rank-pair duplication, or shared ring CCL
resources.  A fresh source-only AutoDebug confirmed that the gap was real and
recommended a dedicated 1x4 test.  It made no edits and opened no hardware.

## Hypothesis experiment and fix

**Hypothesis:** the selected TP4 policies compose correctly when a real-weight
sliding layer feeds a real-weight full-attention layer, including through
capture/replay with one shared set of persistent all-reduce resources.

**Experiment:** add
`test_tp4_stacked_mixed_attention_shared_persistent_ccl_trace`, derived from the
TP2 stacked gate.  A separate one-chip fixture first captures the exact
`OptimizedDecoder` two-layer oracle, avoiding a fifth chip or an overlapping
submesh.  The 1x4 run uses the full P300C proxy mesh, `FABRIC_1D_RING`, a 128 MiB
trace region, real weights, and `throw_exception_on_fallback=true`.

**Verdict:** verified.  The durable test checks the final instantiated policies,
replicated tiled DRAM-interleaved public boundary, local caches, replicated
control tensors, actual CCL slot reuse, both traced layer outputs, and 20 replay
iterations.  No implementation change was required.

## Correctness and contract results

The deterministic regime is sequence length 32, decode batch 1, and current
position 32.  Layer 0 is selected BF16-attention/R22/folded/row-major-routing;
layer 5 is selected BFP8-attention/R0/raw-router/raw-FFN-norm/non-row-major
routing.  Both retain folded expert scale, fused final scalar, and packed
gate-selected expert decode.

| Comparison | layer 0 sliding | layer 5 full | Gate |
| --- | ---: | ---: | ---: |
| Same-input prefill vs optimized | 0.997792 | 0.998964 | 0.99 |
| Same-input eager decode vs optimized | 0.999117 | 0.995288 | 0.99 |
| Chained prefill vs optimized chain | 0.997792 | 0.991588 | 0.99 |
| Chained eager decode vs optimized chain | 0.999117 | 0.986078 | 0.99 / 0.98 |

The final traced chained decode PCC is 0.986078 at the documented 0.98
divergent-input stress threshold.  Layer 5 receives approximate TP4 layer-0
output in that comparison; its separately gated identical-input decode remains
0.995288.

- Public prefill/decode input and both layer outputs are bit-exact replicas on
  all four ranks, tiled, DRAM, and interleaved.  Layer 0 output is passed directly
  to layer 5 without a host conversion or test-side reshard.
- Sliding cache per rank is `(4, 2, 64, 256)`; full cache per rank is
  `(2, 1, 128, 512)`.  Both K/V caches are BF16 tiled DRAM-interleaved, updated,
  and retain a zero untouched block tail.
- Full K and V cache ranks 0/1 are bit exact, ranks 2/3 are bit exact, and the two
  rank pairs differ.  Sliding local-head cache contents do not collapse across
  all ranks.
- Per-layer page tables are replicated int32 row-major tensors with shapes
  `(1, 4)` and `(1, 2)`; current positions are replicated int32 row-major
  `(1,)` tensors containing 32.
- Both decoders share the same three BF16 tiled persistent buffers and semaphore
  list.  Each buffer is `(1, 1, 32, 11264)`.  The observed eager slot sequence is
  layer 0 `(0,1,2)`, then layer 5 `(0,1,2)`, proving reuse.  The shared index is
  zero initially and after prefill, same-input calls, eager decode, capture, and
  replay.
- Capture counters are exactly `{all_reduce, attention_tp, dense_tp, expert_tp}`
  = `{9,3,3,3}` for layer 0 and `{15,5,5,5}` for layer 5.  All counters and the
  resource index remain unchanged through 20 device-side replays.
- Eager versus first replay, all replay pairs, and all four replicas are bit
  exact for both layer outputs.

## Commands and health

Pre-run and post-run device checks were serialized; no reset or recovery was
needed:

```bash
timeout 60 tt-smi -ls --local

GEMMA4_CAPTURE_TP4_STACKED_REFERENCE=1 GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
timeout 1800 python_env/bin/python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_capture_optimized_tp4_stacked_mixed_reference \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/stacked_tp4_reference_capture.xml

GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
timeout 1800 python_env/bin/python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_tp4_stacked_mixed_attention_shared_persistent_ccl_trace \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/stacked_tp4_mixed_trace.xml

timeout 60 tt-smi -s
```

The reference and TP4 JUnits each report 1/1 passed.  Final health showed four
P300C devices with DRAM healthy, zero corrected and uncorrected GDDR errors,
maximum ASIC temperature 34.7 C, and maximum GDDR temperature 38 C.

Host verification:

```bash
pre-commit run --files \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/AUTOFIX_TP4_STACK.md \
  models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/stacked_tp4_mixed_trace.xml \
  models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/stacked_tp4_mixed_trace.json \
  models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/stacked_tp4_optimized_reference.pt.gz \
  models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/stacked_tp4_reference_capture.xml
python_env/bin/python -m py_compile \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py
git diff --check -- \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/AUTOFIX_TP4_STACK.md
```

## Artifacts and provenance

- `artifacts/stacked_tp4_mixed_trace.xml`:
  `01b9a3a90ac07c3d0430801bb23947b8354a28d7bab0ff08df98eb09e9b9b6d6`
- `artifacts/stacked_tp4_mixed_trace.json`:
  `a1d500cebacce7b8969b40c173fc69af459db70590c3f49dfb54b11d57d8e22f`
- `artifacts/stacked_tp4_optimized_reference.pt.gz`:
  `0385dd042d370971b751a45868831745ed960e3ed304cfede4a3ce85d2bccd2e`
- `artifacts/stacked_tp4_reference_capture.xml`:
  `e5fd8926afa02a8ae44e55b5048d5749cf1e02851f7f4e2e086308401eeef844`
- Final tested `multichip_decoder.py`:
  `7279e13a379bd9260006f2ab220a1fa7b35d01ccec1ccc79dddd17bab058f3bf`
- Final tested `test_multichip_decoder.py`:
  `5bd55e7231499c17c8e3a77c2711efb860b9412f4e9d9d80d7c82b3e12344abe`

## Remaining uncertainty

This focused gate proves the requested batch-1, S=32 layer-0 to layer-5 TP4
boundary.  It does not independently repeat longer-sequence or batch-32 stacked
chains; those shapes remain covered by the stage's isolated-layer and cache
evidence.  Watcher evidence is also kept in the stage's separate watcher runs,
as required by the device-usage rule separating instrumentation modes.
