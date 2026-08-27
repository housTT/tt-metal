# Final source AutoDebug: layer-0 host-backed prefill PCC regression

Date: 2026-08-27

Scope: source-only inspection of the current dirty worktree plus retained
diagnostic artifacts.  No TT hardware was run for this report.

## Headline

The first failing edge is already the first ladder edge:

```text
unchanged OptimizedDecoder baseline
  vs
resident replicated-residual MultichipDecoder TP2
```

That means the deterministic layer-0 prefill miss is not primarily localized to
the new persistent fractured residual ABI, PLE bridge, QSA path, or host-backed
expert service.  The source and retained boundary evidence point to the
rank-local routed-expert MoE computation inside the resident TP2 layer.

The decisive retained artifact is
`gdn_diag_baseline_tp_boundaries.xml`:

```text
MC_GDN_PREFILL_RESIDUAL_DIAG layer=0 edge=baseline_tp seq=33
output_pcc=0.99170667

hyper_mix0.mixed          1.00038743
gdn0.output               1.00002766
hyper_inject0.output      1.00002587
hyper_mix1.mixed          1.00000393
routing0.output           0.99997306
routed0.output_partial_sum 0.93857193
moe0.output               0.98043340
hyper_inject1.output      0.99979049
```

`gdn_diag_expert_weights.xml` proves the represented expert weights are not
mis-sliced:

```text
MC_GDN_EXPERT_WEIGHT name=gate_up max_abs=0.00000000 exact=True
MC_GDN_EXPERT_WEIGHT name=down    max_abs=0.00000000 exact=True
```

`gdn_diag_tp_expert_hifi2.xml` keeps the same failing output after changing the
resident TP2 expert compute config to HiFi2, so the most likely boundary is not
simply LoFi versus HiFi math fidelity.

## Ranked hypotheses

### 1. Rank-local routed-expert sparse-matmul geometry is numerically too different from the unchanged full-width baseline

Confidence: high.

Code facts:

- `MultichipDecoder.from_state_dict` always builds TP-local routed and shared
  expert intermediates through `_rank_local_config` and `_rank_local_state`.
  Layer 0 therefore computes two 320-wide expert halves, not one 640-wide
  expert.
- The unchanged `OptimizedDecoder` baseline uses default
  `expert_bfp4_lofi_g40b16_d40b5`; the TP2 resident path defaults to
  `expert_bfp4_lofi_g20b16_d40b5` because the local gate/up output has only 20
  tiles and cannot use the full-width 40-core gate program.
- The retained boundary artifact shows HC, GDN, MLP-HC input, and router logits
  match.  The first large drop is the routed expert output before final
  injection.

Predicted signature:

- `baseline_tp` fails at about `0.9917` output PCC on the 33-row real
  activation diagnostic, and the full host-backed perf gate reports about
  `0.99116`.
- `routing0.output` remains high (`~0.99997`), proving selected expert weights
  and route weights are essentially the same.
- `routed0.output_partial_sum` remains low (`~0.94`) even when rank partials are
  summed on host in float, so the CCL all-reduce itself is not the first cause.
- Capturing inside `_routed_experts` should show whether the first bad tensor is
  the packed gate/up sparse matmul or the down sparse matmul over split K.

Smallest A/B experiments:

1. Add a test-only capture inside `OptimizedDecoder._routed_experts`/the TP2
   routed path for `gate_up_sparse`, `hidden`, `weighted_hidden`, `down`, and
   final `fast_reduce_nc`.  Run only:

   ```bash
   QWEN38_MC_RUN_GDN_RESIDUAL_DIAG=1 pytest -q -s --tt-arch blackhole \
     models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py::test_gdn_prefill_residual_topology_diagnostic \
     -k baseline_tp
   ```

2. Run the same `baseline_tp` diagnostic with both sides explicitly using
   `optimization_policy="expert_bfp4_lofi_g20b16_d40b5"`.  If the routed PCC
   recovers materially, the full-vs-local gate/up program geometry is the main
   delta.  If it does not, the split-K down/partial-sum boundary is more likely.
3. Run both sides with `optimization_policy="fused_precision"` for layer 0 only.
   If TP2 then passes, the issue is the optimized BFP4 sparse path under TP
   decomposition.  If it still fails, the algebraic split itself needs a more
   exact accumulation strategy.
4. If the down boundary is confirmed, try a test-only routed-expert down output
   dtype/accumulation override to FP32 before reducing experts/ranks.  Passing
   would justify a targeted precision policy for the TP MoE down path rather
   than changing HC, GDN, residual, or host service.

### 2. The regression gate is comparing two different valid expert execution contracts too strictly for layer-0 real activations

Confidence: medium-high.

The current baseline is "unchanged OptimizedDecoder", but TP2 cannot literally
execute the same full 640-wide expert kernel: its legal local gate/up shape is
640 output columns (`2 * 320`) and its default policy must use the 20-core gate
program.  The exact expert tensors reconstruct bitwise, but the executable
contract is not identical.

Predicted signature:

- Random small inputs and layers 1/3 may pass because the layer output is less
  sensitive, while the deterministic layer-0 embedding-row activation exposes
  the MoE split.
- Reusing the same TP-local two-half execution on the baseline side should make
  the comparison pass without changing residual topology.

Smallest A/B experiments:

- Build a one-layer "TP-emulated baseline" that runs two rank-local expert
  halves with the same local policy and host-float/device sum, while keeping HC
  and GDN unchanged.  Compare it to resident TP2.  If this passes and the
  unchanged baseline fails, this is a comparator/contract issue rather than a
  fractured-residual integration bug.
- Conversely, run a one-layer resident control with replicated full-width
  experts on both ranks and no TP expert split, then compare rank 0 to the
  unchanged baseline.  This is memory-expensive but only one layer; if it
  passes, the TP expert split is confirmed.

### 3. Host-backed prefill-wave expert service may introduce an additional drift, but it is downstream of the current first failure

Confidence: medium.

Source facts:

- Host prefill gathers unique route IDs, loads bounded expert waves, evaluates
  `_routed_expert_wave` per wave, and sums wave outputs.
- That changes the resident packed expert bank from one sparse reduction over
  all active experts into multiple sparse reductions plus adds.

Why this is not the headline cause:

- The retained ladder already fails before host backing (`baseline_tp`).
- A host-edge failure cannot explain a resident replicated TP2 failure.

Predicted signature if this is also present after fixing baseline TP:

- `baseline_tp` and `residual` pass, but `host` fails.
- Resident fractured and host-backed fractured have matching routing ids but
  diverge at `_routed_expert_wave` output or after summing waves.
- Increasing `expert_cache_slots` for a single-layer diagnostic enough to reduce
  the number of waves should improve PCC if wave splitting is the cause.

Smallest A/B experiments:

- Run the existing ladder only after `baseline_tp` passes:

  ```bash
  QWEN38_MC_RUN_GDN_RESIDUAL_DIAG=1 pytest -q -s --tt-arch blackhole \
    models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py::test_gdn_prefill_residual_topology_diagnostic \
    -k "residual or host"
  ```

- For the `host` edge, repeat with a larger single-layer `expert_cache_slots`
  and report route union, wave count, per-wave routed output PCC, and final
  host-edge PCC.

### 4. General-M fractured HC or GDN output sharding remains a required validation item, but current evidence demotes it

Confidence: medium-low for the current failure; still important for final signoff.

The earlier integration audit correctly called out unproven areas: M=128
fractured HC, GDN output-N sharding, MoE reduce-scatter, and PLE bridge.  Those
are still necessary gates for production.  But the layer-0 boundary diagnostic
shows:

- `hyper_mix0.*` passes;
- `gdn0.output` passes;
- first bad block is routed expert output.

Predicted signature if this hypothesis becomes active after fixing TP MoE:

- `baseline_tp` passes, but `residual` fails.
- The first bad boundary moves to fractured `_hyper_mix`, `gdn0.output` after
  `gdn_out` sharding, or MoE reduce-scatter, not to `routed0.output_partial_sum`
  on the replicated resident edge.

Smallest A/B experiments:

- After `baseline_tp` passes, run `residual` with the same boundary captures.
- Add a direct GDN output-shard probe: compare resident replicated GDN output to
  `gather_residual(fractured_gdn_output)` immediately after `_gdn_prefill`, with
  M=1, M=33 padded to 128, and M=128.

## Recommended next order

1. Do not tune fractured residual or host backing first.  Re-run/extend the
   `baseline_tp` boundary diagnostic because it is the earliest failing edge.
2. Capture inside routed experts to split gate/up vs down vs fast-reduce.
3. Run the policy controls: same `g20b16` baseline, then `fused_precision` on
   both sides.
4. Only after resident replicated TP2 clears `0.995`, continue the original
   ladder to `residual` and then `host`.

The likely fix boundary is a TP2 routed-expert precision/program/accumulation
choice for layer-0 real activations, not a change to the persistent S residual
ABI.
