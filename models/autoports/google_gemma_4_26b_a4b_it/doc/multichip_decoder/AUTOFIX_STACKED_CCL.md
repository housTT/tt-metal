# AutoFix: stacked decoder and shared persistent CCL

## Scope

This pass closes the missing stacked-decoder/resource-reuse coverage without
assembling a full model. It adds
`test_tp2_stacked_mixed_attention_shared_persistent_ccl_trace` at the end of
`tests/test_multichip_decoder.py`. The test opens the complete four-chip QB2 as
`(2, 2)` with `FABRIC_2D`, uses a top-row `(1, 2)` TP2 submesh, and uses a
disjoint bottom-row `(1, 1)` submesh for a two-layer `OptimizedDecoder` oracle.

The exercised chain is layer 0 sliding attention followed by layer 5 full
attention. It runs an exact `S=32` prefill, batch-1 eager decode, chained decode
trace capture, and 20 blocking trace replays. Each layer owns its paged KV
cache, while both multichip layers share the same three rotating
`persistent_all_reduce_resources` buffers and semaphores.

No production file was edited in this pass. The stable selected-policy runs
used:

- `multichip_decoder.py` SHA256
  `377f9f8adec3f8e8b851b1a3006396620d455851bf81acb1d2ff0ce2a9dd1a18`;
- test SHA256 before the diagnostic switch
  `33119f169048d86503db3ab5538b56c65eb3a8888657a5c5fdeab76379912eec`;
- passing R0-control run test SHA256, including the diagnostic-only R0
  reference switch,
  `6e1c8b70eded82777bae8a5e7316e1d8725d1992a7d8e3d2d48e7f946734c531`.

Black subsequently reformatted the same test without changing behavior; its
post-check SHA256 was
`54f9db10909077bee82c9a01ae0ab1497478a26dcb692ccc0435421498e2e852`.

## Experiments

### Shared-resource and trace stress

The selected-policy run completed without a CCL wait. Both decoders dispatched
nine reductions on the Python execution path: three during prefill, three
during eager decode, and three during trace capture. Twenty trace replays were
bit-exact with the eager output and with each other, and both TP replicas were
bit-exact for all checked outputs.

This verifies the original TP2 repair at the new stack boundary. The second
layer safely wraps the shared resource index and reuses the first layer's
buffers and semaphores. The exact `M=32` prefill also stresses the Blackhole
boundary selected by `_matrix_rows(partial) <= TILE_SIZE`; no race or
nondeterminism appeared in this 20-replay run.

The first draft synchronized the parent `(2, 2)` mesh after dispatching on two
child meshes. Its math and all 20 replays completed, but teardown raised:

```text
MeshDevice cq ID 0 is in use by parent mesh ID 0 during close of mesh ID 1
```

Synchronizing the `(1, 1)` and `(1, 2)` child meshes separately fixed teardown.
Trace capture, execution, and release remain correctly owned by the TP2
submesh. This was a test ownership bug, not a decoder or CCL bug.

### Selected Optimized acceptance oracle

The stable default run compares TP2 R0/raw-weight execution with the selected
single-chip Optimized R22/folded oracle. It fails the provisional two-layer
`0.99` PCC gate:

| Phase | After layer 0 | After layer 5 |
| --- | ---: | ---: |
| Prefill | 0.9972393566 | 0.9923597082 |
| Eager decode | 0.9899334009 | 0.9840812211 |
| Trace decode | - | 0.9840812211 |

Layer 0 already fails before stacking can amplify the error. Its routed
top-eight experts still agree 8/8. At layer 5, the perturbed input changes one
expert: the selected oracle chooses expert 94 while TP2 chooses expert 46, with
7/8 active experts otherwise identical. That routing discontinuity explains
the additional final-layer drop.

The result is deterministic rather than a resource race: eager and trace are
bit-exact, all 20 replays are bit-exact, and TP replicas are bit-exact.

### All-BF16 multichip weight control

An isolated control uploaded all multichip attention, dense MLP, and expert
weights as BF16. It did not repair decode:

| Phase | After layer 0 | After layer 5 |
| --- | ---: | ---: |
| Prefill | 0.9971517244 | 0.9935143039 |
| Eager decode | 0.9898748530 | 0.9841939765 |

Routing agreement remained 8/8 then 7/8. Weight quantization is therefore
refuted as the cause of this acceptance failure, and no precision-default
change is justified by this evidence.

### Execution-equivalent R0/raw control

The diagnostic-only environment switch
`GEMMA4_MULTICHIP_STACKED_REFERENCE_R0=1` configures the Optimized reference
with R0, graph folds disabled, packed experts disabled, row-major routing
disabled, and BFP8 expert gate/up weights matching TP2. It is an isolation
control and is not the acceptance oracle.

That equivalent reference passes:

| Phase | After layer 0 | After layer 5 |
| --- | ---: | ---: |
| Prefill | 0.9972469664 | 0.9925726011 |
| Eager decode | 0.9991179995 | 0.9991212503 |
| Trace decode | - | 0.9991212503 |

Both layers agree on all eight routed experts. The same shared-resource trace
again completes 20 bit-exact replays with bit-exact TP replicas. This localizes
the selected-oracle failure to the R22/fold-policy adaptation boundary, not TP
fracturing, paged-cache ownership, CCL resource reuse, trace replay, or weight
precision.

## Evidence

Artifacts are under `doc/multichip_decoder/artifacts/`:

- `stacked_tp2_mixed_trace_stacked_v2.json` and `.xml`: stable selected-policy
  failure with per-layer PCC and routed expert sets;
- `stacked_tp2_mixed_trace_stacked_bf16.json` and `.xml`: BF16 control;
- `stacked_tp2_mixed_trace_stacked_r0_reference.json` and `.xml`: passing
  execution-equivalent isolation control;
- `stacked_tp2_mixed_trace.json` and `.xml`: first complete stress result,
  retained as provenance for the parent-mesh synchronization teardown error.

The stable selected command was:

```bash
timeout 1800 env \
  GEMMA4_MULTICHIP_ARTIFACT_SUFFIX=_stacked_v2 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  python_env/bin/python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k tp2_stacked_mixed_attention_shared_persistent_ccl_trace \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/stacked_tp2_mixed_trace_stacked_v2.xml
```

The passing isolation command added:

```text
GEMMA4_MULTICHIP_STACKED_REFERENCE_R0=1
GEMMA4_MULTICHIP_ARTIFACT_SUFFIX=_stacked_r0_reference
```

Host checks passed with `py_compile` and focused `pytest --collect-only`.
Before device work, bounded `tt-smi` list/reset/list showed all four P300C
Blackhole chips and a `(2, 2)` mesh open/close smoke printed `MESH_SMOKE_OK`.
The selected rerun and both controls closed devices cleanly.

## Verdict

The shared persistent CCL candidate is safe at this focused two-layer TP2
stack boundary: prefill, eager decode, trace capture, 20 replays, resource
wrapping, and replica consistency all pass without a wait or race signature.

The overall stacked decoder is **not accepted** against the required selected
Optimized oracle. Its default test intentionally retains that oracle and
currently fails. The constructor/fold-policy work must adapt the multichip path
to the selected R22/fold semantics (or otherwise prove equivalent selected
behavior) and rerun the unchanged default stacked gate. The passing R0/raw
control must not replace or weaken that acceptance comparison.

## Stage-review P2 repair

The stage review found that a later revision weakened the entire stacked test
to `0.98`. That allowed the first, same-input layer-0 boundary to pass at
`0.9912233636` without making the distinction above explicit. The layer-5
route discontinuity cannot explain a discrepancy that is already present at
layer 0.

The repaired test uses two separate contracts:

- both layer kinds run independently on identical Optimized and TP2 inputs
  with fresh cache state; prefill and decode must each clear `0.99`;
- only the final layer-5 output of the deliberately divergent chain retains a
  `0.98` discontinuity threshold. Layer-0 chained decode and both chained
  prefill boundaries remain at `0.99`.

The same-input selected-policy result is:

| Phase | Layer 0 sliding | Layer 5 full |
| --- | ---: | ---: |
| Prefill | 0.9975631291 | 0.9978789479 |
| Decode | 0.9912233636 | 0.9952972969 |

Independent TP2 real-weight HF controls use a stricter `0.995` threshold and
also pass both layer kinds:

| Layer | HF prefill PCC | HF decode PCC |
| --- | ---: | ---: |
| 0 sliding | 0.9990238215 | 0.9995657036 |
| 5 full | 0.9985337388 | 0.9997109593 |

This evidence refutes a TP2 production-policy regression. The low pairwise
`0.991223` is disagreement between two independently approximate TTNN paths on
this input, while TP2 remains substantially above the HF gate. It does not
justify weakening a same-input boundary to `0.98`.

`artifacts/tp2_hf_same_input_stacked_final.xml` records three passing tests.
`artifacts/stacked_tp2_mixed_trace.json` now labels every threshold and whether
the corresponding layer inputs are identical. The two HF JSON files are under
`artifacts/tp2_hf_oracle/`. The run used no watcher or profiler and enabled
`throw_exception_on_fallback`; post-run health showed all four P300C devices,
healthy DRAM, zero GDDR errors, and temperatures from 31.2 to 36.0 C.

This evidence predates the pending capacity-driven removal of TP2's retained
packed-expert decode copy. The same combined command must be rerun after that
policy change; this report must not be used to waive that final-policy rerun.
