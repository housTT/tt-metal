# Stages 4 and 5: multi-chip encoder (optional for the p150 deliverable)

The deliverable is a single p150 chip. The box has four p150-class chips on two p300c boards, so tensor-parallel
meshes were tried with the stock `models.tt_transformers` TP plan (weights sharded across the mesh, replicated
residual stream, CCL all-gathers after the attention and MLP projections), `FABRIC_1D` set before the mesh opens
(the same fabric config the stock tt_transformers conftest selects for a non-Galaxy Blackhole mesh).

## 1x2 (two chips on one p300c board, device name P300)

Did not open. Fabric initialized on the two devices, then:

```
Fabric Router Sync: Timeout after 10000 ms on Device 0. Expected status 0xa2b2c2d2 (LOCAL_HANDSHAKE_COMPLETE)
```

(`/home/hous/dev/clm-v0.1-8B/logs/fidelity_accuracy_1x2.log`, `bench_accuracy_1x2.log`, 2026 Oct 1 22:02 UTC).
The system mesh discovered by the control plane is the 1x4 of all four chips; opening a 1x2 sub-mesh with 1D
fabric on this box is not supported in this configuration. Not pursued further; the campaign that previously ran on
this box also recorded Ethernet-core timeouts on this board pair, so a physical link problem is not excluded.

## 1x4 (all four chips, device name P150x4)

Fidelity (`fidelity_accuracy_1x4.json`, accuracy policy, 308 texts vs HF fp32):

| metric | 1x4 | 1x1 for comparison |
|---|---|---|
| cosine mean | 0.99893 | 0.99910 |
| cosine min | 0.99307 | 0.99596 |
| single vs batched, mean / min | 0.99954 / 0.99620 | 0.99934 / 0.99644 |

The multi-chip output is slightly further from the reference than one chip (reduction-order differences in the
CCL all-gathers and sharded matmuls). Stage 4 gate from the plan (cosine of the 1x4 output against the one-chip
output >= 0.999), measured from `fidelity_accuracy_1x4_tt_single.npy` against the one-chip
`../datatype_sweep/fidelity_accuracy_tt_single.npy` over the 308 corpus texts: mean 0.99929, p05 0.99805, min 0.99547,
68 of 308 texts below 0.999. The gate is met as a mean and not as a minimum; the texts below 0.999 are the same
near-tie-sensitive ones that move between a single and a batched one-chip run (`../full_model/README.md`), and the
plan amendment of 2026 Oct 2 records the mean form. Decision agreement of the 1x4 vectors with the fp32 reference:
190 of 200 (185 of 188 confident), the same band as one chip. The first 1x4 attempt failed in the host readback
because the tensor returned by the traced prefill (the pre-norm residual of the last layer, read back before any
final all-gather) is width-sharded across the four devices; `ConcatMeshToTensor(dim=-1)` is now used (`tt/encoder.py`).

Latency (`bench_accuracy_1x4.json`, accuracy policy, p50, batch 1 unless noted) against the one-chip accuracy run:

| padded length | 1x1 | 1x4 | speedup | efficiency (speedup / 4) |
|---|---|---|---|---|
| 128 | 57.6 ms | 30.5 ms | 1.89x | 47 % |
| 1024 | 170.5 ms | 105.5 ms | 1.62x | 40 % |
| 2048 | 322.8 ms | 207.9 ms | 1.55x | 39 % |
| 128, batch 8 | 162.0 ms | 106.5 ms | 1.52x | 38 % |

Tensor parallelism over four chips cuts single-text latency by about half at 128 tokens and by a third at 1024
and 2048 tokens; the efficiency (speedup per chip) is 39 to 47 percent, as expected for a weight-bandwidth-light,
CCL-heavy prefill of a model this size. It is published as the `p150x4` serve profile.

## Stage 5 (optimized multi-chip)

Not done beyond the stock TP plan. The inter-layer residual contract is the stock one: inside the decoder stack the
residual is replicated on every device after each layer's all-gather, with no reshards between layers; only the
tensor handed back to the host at the end is the width-sharded pre-norm residual (previous section). No CCL audit
(per-collective latency, link utilization) was done for this profile; it ships as validated by fidelity, bench and
the in-container serve check only, and the card marks the package as an experimental community bring-up. The 1x4 serve profile `p150x4` is declared in `tt-model.yaml` with the accuracy policy and `FABRIC_1D`.

## Stock kernels and the LoFi-MLP policy on 1x4 (2026 Oct 2 00:56 to 00:59 UTC)

Review R found that every 1x4 measurement above ran in a working tree with two uncommitted kernel edits from the
Ornith project (`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`, `.../all_gather_async/device/kernels/minimal_default_writer.cpp`),
which the container builds also shipped. With both files stashed to their committed versions, the 1x4 fidelity run of
the `accuracy_lofi_mlp` policy (`fidelity_accuracy_lofi_mlp_1x4_stock_kernels.json`, log
`/home/hous/dev/clm-v0.1-8B/logs/fidelity_lofi_1x4_stock_kernels.log`) loads in 4.6 s, captures the fifteen trace
variants in 26.8 s and passes: cosine vs the fp32 reference mean 0.99908 / min 0.99588 / p05 0.99763, head-projection
minima 0.9917 (state) and 0.9970 (candidate), single vs batched min 0.99664, no NaN; decision agreement with the fp32
reference 194 of 200 (97.0 percent) and 187 of 188 confident decisions (99.5 percent)
(`agreement_accuracy_lofi_mlp_1x4_stock_kernels.json`). The edits are therefore not required by this port; the
published build is made from a clean worktree of the branch and the `p150x4` profile is re-verified from that image
(`../release/RUN_NOTES.md`). Whether the edits matter for the 1x2 fabric timeout was not tested.

## In-container 1x4 serving

The first `tt-model serve --profile p150x4` of the built image failed at fabric initialization:
`Cannot open kernel source file: /opt/tt-metal/tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`.
The file was in the image but with mode 0660 (it is one of two kernel sources carrying uncommitted edits from the
earlier Ornith project in this working tree, saved with this project's umask 007), and tt-model runs the container
as the invoking host uid, which is not the file owner. Single-chip profiles never JIT-compile fabric kernels, so
only the multi-chip profile hit it. Fix: `chmod -R a+rX` on the tt-metal source tree before packaging; the image was
rebuilt and the profile re-verified (see `../release/RUN_NOTES.md`).
