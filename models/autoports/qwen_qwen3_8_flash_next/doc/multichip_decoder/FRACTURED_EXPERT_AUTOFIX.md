# Routed-expert EP2 AutoFix

## Failure

After persistent fractured-residual integration, the real-checkpoint layer-0
seq-33 prefill gate failed deterministically below PCC 0.995. The failure was
not dismissed as generic BFP4 noise and the threshold was not lowered.

A fresh-context AutoDebug investigation instrumented the optimized baseline,
resident TP2, and host-backed TP2 at the HC, GDN, router, routed-expert, and
MoE boundaries using the exact failing input.

## Localization

The following observations were proven in isolation:

- hyperconnection, GDN, router logits/top-k, and routing inputs were effectively
  exact;
- TP checkpoint slices reconstructed the optimized expert weights exactly;
- packed gate/up output reconstructed exactly;
- replacing candidate routes with baseline routes did not repair the result;
- HiFi2, FP32 down output, and fused-precision changes did not repair it;
- the first material drift was the routed down projection split at K=640 into
  two K=320 sparse matmuls and summed across ranks.

The failing split-K boundary measured about 0.926 PCC for the down partial sum,
about 0.938 PCC for routed output, and about 0.9917 at the final output. The
diagnostic artifacts are `gdn_diag_*.xml`, `expert_ep2_boundary.xml`, and
`FINAL_SOURCE_AUTODEBUG.md`.

## Tested repair ladder

The isolated repair used deterministic expert ownership:

```text
owner = expert_id % 2
owner slot: full gate/up [1,1,2560,1280], full down [1,1,640,2560]
peer slot:  exact zeros with the same physical shapes
```

The existing MoE collective then sums the one routed owner with both
shared-expert TP320 partials.

Two program configurations were tested separately:

| Candidate | Layer-0 seq-33 prefill PCC | Verdict |
| --- | ---: | --- |
| Full K=640 EP2 with old g20 gate/up program | 0.99064916 | rejected |
| Full K=640 EP2 with g40 gate/up program | 0.99942303 | accepted |

The corresponding isolated artifacts are `expert_ep2_layer0_isolated.xml`,
`expert_ep2_boundary.xml`, and `expert_ep2_g40_boundary.xml`.

## Retained source changes

- `Qwen38ExpertHostSource.load` packs the complete checkpoint expert on
  `expert_id % 2` and exact zeros on the peer.
- `PackedExpert`, device slots, and upload staging use full-K physical shapes.
- Host-backed rank-local shape/config construction preserves routed
  intermediate width 640; the shared expert remains width 320/die.
- Host-backed routed experts select
  `expert_bfp4_lofi_g40b16_d40b5`; the resident negative-control path retains
  its original g20 policy.
- Capacity constants and manifests charge 2,764,800 bytes/rank/expert and
  1,459,814,400 bytes/die for all 48 layers' ten slots plus one staging slot.

No other precision experiment from the diagnostic ladder was retained.

## Final proof

Real-checkpoint correctness against `OptimizedDecoder`:

| Layer | Kind | Prefill PCC | First-token decode PCC |
| --- | --- | ---: | ---: |
| 0 | GDN | 0.99942303 | 0.99996978 |
| 1 | PLE+GDN | 0.99949104 | 0.99984211 |
| 3 | QSA | 0.99972457 | 0.99988294 |

Additional current-source gates:

- exact CPU owner/zero packing and capacity: included in the 32-pass static
  matrix;
- exact physical slot contents, PLE rows, and paged QSA hardware contracts:
  3 passed;
- allocation-tracked acceptance: 12 passed;
- progressing GDN/PLE/QSA stress: 3 cases, 100 changing tokens each;
- warmed performance: 21/21 cases, 100 decode replays each;
- separate watcher: 4/4 cases, clean 556-line raw log;
- separate final Tracy/`tt-perf-report`: all three representative layers,
  zero profiler-buffer overflow messages.

The resident routed split-K implementation remains useful negative evidence
but is not the delivered host-backed EP2 acceptance oracle.
