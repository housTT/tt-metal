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

The multi-chip output is within the stage gate but slightly further from the reference than one chip
(reduction-order differences in the CCL all-gathers and sharded matmuls). The first 1x4 attempt failed in the
host readback because the residual stream comes back width-sharded across the four devices
(`ConcatMeshToTensor(dim=-1)` is now used; `tt/encoder.py`).

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

Not done beyond the stock TP plan. The inter-layer residual contract is the stock one: the residual is replicated
on every device after each all-gather; no reshards between layers. The 1x4 serve profile `p150x4` is declared in `tt-model.yaml` with the accuracy policy and `FABRIC_1D`.

## In-container 1x4 serving

The first `tt-model serve --profile p150x4` of the built image failed at fabric initialization:
`Cannot open kernel source file: /opt/tt-metal/tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`.
The file was in the image but with mode 0660 (it is one of two kernel sources carrying uncommitted edits from the
earlier Ornith project in this working tree, saved with this project's umask 007), and tt-model runs the container
as the invoking host uid, which is not the file owner. Single-chip profiles never JIT-compile fabric kernels, so
only the multi-chip profile hit it. Fix: `chmod -R a+rX` on the tt-metal source tree before packaging; the image was
rebuilt and the profile re-verified (see `../release/RUN_NOTES.md`).
