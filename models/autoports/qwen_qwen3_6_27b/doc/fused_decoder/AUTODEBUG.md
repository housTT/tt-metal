# AutoDebug: fused-decoder review remediation

## Scope and starting evidence

This is a source-only investigation of `stage_review.md` (`more-work-needed`). No TT device command was run. The implementation/test worktree was not changed.

Authoritative model geometry from the cached Qwen config and decoder source is:

- convolution width/groups: `2 * 16 * 128 + 48 * 128 = 10,240`;
- kernel: 4, stride 1, BF16 projected input and BF16 stored weights;
- chunk prefill: `valid_tokens` is 1..64, with a four-row rolling state;
- token decode: batch is 1 normally and can be 32, input length is one, and the same four-row state is updated in place.

## Finding 1: split depthwise `conv1d` is a viable candidate

### Facts

The current chunk convolution forms `[B,1,4+S,10240]`, updates state from rows `[S:S+4]`, and computes output row `t` from context rows `t+1:t+5`. The decode path first forms the new `[B,1,4,10240]` state and reduces its four weighted rows. These are ordinary depthwise cross-correlations.

For prefill, either of these expressions is equivalent to the current indexing:

1. valid `conv1d` on all `4+S` rows, producing `S+1` rows, then discard output row 0; or
2. discard context row 0 first and run valid `conv1d` on `S+3` rows, producing exactly `S` rows.

The second avoids an output slice. A source-only Torch check covered `B={1,32}` and `S={1,33,64}`. Both formulations matched the spelled expression within `9.54e-7` (the only difference was floating-point reduction order). Decode `conv1d` on the four-row new state produces one row with the same indexing.

`ttnn.conv1d` accepts N,H,W,C input and PyTorch-format `[out_channels, in_channels/groups, kernel]` weights. It wraps `conv2d`; its device kernel requires row-major sharded activation. The wrapper performs internal sharding when given interleaved row-major input. `Conv1dConfig` supports a fused `UnaryWithParam(SILU)`, BF16 weights/output, tile output, and height sharding. The Mamba implementation proves the required lifecycle: channel slice, per-split `conv1d`, `sharded_to_interleaved`, concatenate, and replace the original host weight with the returned preprocessed device weight for later calls.

The unit test's policy skips BF16 output above 2,560 channels, not at 2,560. Therefore:

- 2 x 5,120 is only a plausible BF8 weight/output experiment and changes numerics;
- **4 x 2,560 is the minimum uniform split compatible with the repository's BF16 test policy**;
- 8 x 1,280 is a fallback if 4 x 2,560 still exhausts Blackhole L1/weight-preprocess resources;
- a semantic 5 x 2,048 split (Q, K, and three V pieces) is legal but adds one convolution and has no mathematical benefit over 4 x 2,560.

### Concrete candidate design

At `from_state_dict`, keep the original checkpoint tensor `linear_attn.conv1d.weight` in its native `[10240,1,4]` orientation. Create four host TTNN row-major BF16 tensors, each `[2560,1,4]`; do **not** use the existing `_as_weight` path, which transposes the weight to `[1,1,4,10240]` for elementwise multiplication. Cache a list of four weights. On the first warmed invocation, request `return_weights_and_bias=True` and replace each list element with the returned preprocessed device weight; later prefill, decode, and trace capture must reuse those prepared tensors.

Start with:

```python
ttnn.Conv1dConfig(
    weights_dtype=ttnn.bfloat16,
    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=True,
)
```

and an architecture-derived HiFi4 compute config. Do not start with LoFi/BF8 because the acceptance question is obscured by precision changes.

Chunk intervention, after the existing state slice/copy:

1. slice the full context to rows `[1:4+S]`, logical shape `[B,1,S+3,10240]`;
2. convert once to row-major;
3. slice channels into four `[B,1,S+3,2560]` tensors;
4. call four depthwise convolutions with `batch_size=B`, `input_length=S+3`, `in_channels=out_channels=groups=2560`, kernel 4, padding 0, stride 1;
5. confirm each `out_length == S`, convert sharded results to interleaved, reshape if the returned logical form is flattened, concatenate channels to `[B,1,S,10240]`.

Decode intervention uses the already updated state as `[B,1,4,10240]`, performs the same row-major/channel preparation, and calls the four convolutions with `input_length=4`; each output must be `[B,1,1,2560]`. Preserve `ttnn.copy(new_conv_state, conv_state)` exactly. Because SiLU is in each convolution config, no trailing standalone `ttnn.silu` should remain.

### Hypotheses and focused experiments

The mathematical mapping is verified; these device properties remain hypotheses:

- H1: 4 x 2,560 BF16 preprocessing fits Blackhole and accepts both `(B=1,L=67)` chunk input and `(B=32,L=4)` decode input.
- H2: returned output is sharded tile data that can be made interleaved and reshaped/concatenated without a host or reshard fallback.
- H3: preprocessed weights are geometry-independent and reusable across prefill and decode, including trace capture/replay.
- H4: fused SiLU meets real-weight PCC; if it does not, convolution without fused activation plus `ttnn.silu` distinguishes convolution numerics from activation approximation.
- H5: the four-convolution topology beats the spelled graph. This is uncertain: four conv setup/conversion paths may outweigh removal of four multiplies, three adds, and SiLU, especially in one-token decode.

Run the smallest geometry probe first (a temporary focused test, not the full decoder): real layer-0 weights, `B/L={1/67,32/4}`, split-4 BF16, compare shape/finite/PCC against Torch grouped `conv1d` and the current TTNN expression, invoke twice to prove prepared-weight reuse, then capture/replay the decode call. Log every tensor's logical/padded shape, layout, dtype, memory config, and whether each weight is host or device before/after call 1. Expected success is out lengths 64 and 1, finite output, and no new preprocessing on call 2. An OOM/fatal after split-4 is a valid exact blocker only with its complete log; then try 8 x 1,280. Also compare explicit `sharded_to_interleaved` against `memory_config=ttnn.DRAM_MEMORY_CONFIG` if the latter is accepted.

If the geometry probe passes, run in order:

```bash
PYTHONPATH=$PWD pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py::test_fused_linear_attention_non_aligned_prefill_decode[blackhole-device_params0-1] -s
PYTHONPATH=$PWD pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py::test_fused_real_weight_batch32_prefill_decode[blackhole-device_params0-1-linear_attention] -s
PYTHONPATH=$PWD pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py::test_fused_linear_attention_advertised_context_prefill[blackhole-device_params0-1] -s
PYTHONPATH=$PWD pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py::test_fused_linear_attention_advertised_context_decode[blackhole-device_params0-1] -s
```

The first command must retain real-weight prefill/decode PCC and state/determinism evidence. The native commands are necessary because they execute 4,096 sequential chunks and expose weight-preprocess leaks, state drift, and trace unsafety that a short test cannot. Only after correctness should warmed prefill and traced decode be profiled with the existing `test_fused_decoder_perf[blackhole-device_params0-1-linear_attention]` entry point, `QWEN36_PERF_PHASE={prefill,decode}`, and `QWEN36_DECODE_REPLAYS=1`. Retain source snapshot/hash, raw Tracy CSV, signpost-filtered `tt-perf-report` CSV, PCC logs, and (if selected) Watcher evidence. Reject a correct candidate if either measured linear phase is not faster.

Likely risks are BF16 preprocessed-weight footprint, internal row-major/shard conversions, a different returned logical shape for `B>1`, first-call weight mutation during trace, and fused-SiLU approximation. None is a source-contract blocker yet.

## Finding 2: missing comparison evidence should be rerun, not inferred

### Facts

The old concat-head and decode-RoPE candidate snapshots and candidate raw reports exist, but their quoted controls (2,356.778 us and 2,339.680 us) do not. The direct `repeat_interleave` claim has no candidate bundle at all. The old candidate snapshots have base hash `2347a7...`; the current source is a later base, so comparing an old candidate to today's final raw CSV is not an identical-base control.

The exact reshape/concat alternatives are already present in `FunctionalDecoder`:

- decode: reshape `[B,16,1,128] -> [B,16,1,1,128]`, concatenate the same tensor three times on dimension 3, reshape to `[B,48,1,128]`;
- prefill: reshape `[B,S,16,128] -> [B,S,16,1,128]`, concatenate three times on dimension 3, reshape `[B,S,48,128]`, permute `[0,2,1,3]`.

These preserve the required head order exactly. The existing dedicated concat-head candidate changes the two full-prefill `permute+reshape` sites to `ttnn.transformer.concatenate_heads`. The existing decode-RoPE candidate changes only Q/K decode rotation to the lane-as-sequence `_partial_rope` formulation shown in its retained source snapshot.

### Exact evidence reconstruction

First settle the convolution candidate and freeze the new final base. Then run all three comparisons from that same source hash; do not reuse the historical scalar controls.

For each family, retain a base snapshot/hash and a one-change candidate snapshot/hash:

1. **head repetition:** replace both `_repeat_linear_qk*` methods with the functional reshape/concat bodies; run linear non-aligned correctness plus linear prefill and traced-decode profiler phases;
2. **concat heads:** change only both full-prefill head-merging sites to `transformer.concatenate_heads`; run full non-aligned correctness plus full prefill profiling;
3. **decode RoPE:** apply only the retained lane-axis change to `_full_qkv_decode`; run full non-aligned and paged-trace correctness plus full traced-decode profiling.

For every profiler comparison, use an A/B/A sequence (base, candidate, restored base) with the same profiler checkout, cache policy, replay count, and signposts. Keep all three raw `ops_perf_results.csv` files and filtered reports. Report the candidate against the median of the two base totals; this resolves the prior 0.16% run-to-run ambiguity. The existing Tracy command template in `work_log.md` is suitable. Use these exact signpost pairs:

- `LINEAR_ATTENTION_PREFILL_START/END`;
- `LINEAR_ATTENTION_DECODE_TRACE_START/END`;
- `FULL_ATTENTION_PREFILL_START/END`;
- `FULL_ATTENTION_DECODE_TRACE_START/END`.

Run `tt-perf-report RAW --start-signpost START --end-signpost END --arch blackhole --no-color --no-advice --csv REPORT.csv --summary-file REPORT_summary` for every raw CSV. Record commands, source hashes, PCC, each A/B/A total and op count, median control, delta, and disposition in each manifest and index all artifacts in both `candidates/index.csv` and `perf/candidates.csv`.

Expected outcomes are not assumed. Direct repetition remains selected only if it is faster than reshape/concat in both applicable end-to-end phases on the frozen base. The concat-head and lane-axis candidates remain rejected only if their fresh candidate totals exceed their fresh A/B/A controls while PCC passes. If a candidate wins, it must become the final path and final correctness/perf/Watcher evidence must be recollected.

Finally, remove the stale `clean-pass` statement now. A fresh independent stage review can restore that verdict only after these artifacts and the convolution disposition exist.
