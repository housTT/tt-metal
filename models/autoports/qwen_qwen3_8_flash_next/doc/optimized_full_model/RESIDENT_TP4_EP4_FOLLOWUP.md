# Resident TP4+EP4 performance follow-up

Date: 2026-09-03

## Delivered path

- All 512 routed experts are resident in device DRAM as 128 BFP4 experts per
  device. Runtime expert-weight H2D and route D2H are both zero.
- The exact 102.4 GB PLE/ngram table remains host-backed. Its selected rows are
  read in parallel and staged asynchronously; expert weights are not part of
  this host path.
- Batch-one decode remains TP4+EP4 on a four-device `4x1` mesh. The recurrent
  GDN state is replicated for correctness and uses the shared per-layer L1
  workspace.
- A native Metal2 `ttnn.experimental.kda.recurrent_gated_delta_rule` operation
  now performs decay, state read, delta calculation, rank-one state update,
  and output read in one device program. It assigns one recurrent head per
  core, writes directly into the fixed recurrent-state trace buffer, and
  supports trace replay on one or four devices.
- The native recurrent operation is the selected default. Setting
  `QWEN38_FUSED_RECURRENT_GDN=0` retains the multi-op comparison path.
- The fused real-weight decode regression was a lifetime bug: `mixed` is a
  reshape alias of `mixed_public`, but the backing tensor was deallocated
  before convolution and recurrent-state consumption. The backing tensor now
  remains live through the recurrence and newest-tap state update.
- TP4's non-EP reference path now uses a legal ten-core gate/up sparse-matmul
  geometry for its 160-wide local expert intermediate. Resident EP4 continues
  to use the selected 40-core full-expert geometry.

## Measurements

The canonical four-device resident workload is batch 1, ISL 128, OSL 128,
with device-side greedy sampling and 126 measured trace replays.

| Metric | Prior resident TP4+EP4 | Final single-program recurrent path |
| --- | ---: | ---: |
| TTFT | 2.174 s | 1.274 s |
| Prefill | 2.159 s | 1.264 s |
| Decode latency | 101.889 ms/token | 99.940 ms/token |
| Decode throughput | 9.815 tok/s/user | 10.006 tok/s/user |
| Runtime expert H2D | 0 | 0 |
| Runtime expert route D2H | 0 | 0 |

This is a 1.9% end-to-end decode-latency improvement. The recurrent operation
is decode-only, so the TTFT and prefill differences are not attributed to it.
The final retained performance artifact is
`final_native_recurrent_performance/full_model_performance.json` and is bound
to source digest
`ec0c96312920962ba37ca7d52f3ea0c1220677d2e6da93761387147ecdde516a`.

Component evidence at production recurrent geometry (`H=48`, `K=V=128`):

- output PCC `0.99999928`;
- updated-state PCC `0.99999988`;
- worst individual head PCC `0.99988890`;
- single-device and TP4 trace replay pass;
- real-weight fused layer-0 prefill/decode PCC `0.99995303/0.99997061`;
- real-weight TP4 PLE+GDN prefill/decode PCC `0.99986899/0.99983633`;
- full 48-layer resident token-out smoke pass;
- full 128+128 benchmark pass with no prohibited host fallback.

### Packaged vLLM endpoint

The final four-device endpoint sweep uses exact random-token lengths, greedy
temperature 0, EOS ignored, and the OpenAI-compatible completion endpoint.
Each concurrency-1 case contains three measured requests after the benchmark
tool's initial probe.

| ISL | OSL | Max concurrency | Median TTFT | Mean TPOT | Decode tok/s/user | Aggregate output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 1 | 504 ms | 181.61 ms | 5.506 | 5.229 |
| 1,024 | 128 | 1 | 3,818 ms | 181.70 ms | 5.504 | 4.625 |
| 4,096 | 128 | 1 | 15,052 ms | 181.81 ms | 5.500 | 3.289 |
| 128 | 128 | 2 | 1,955 ms | 368.36 ms | 2.715 | 5.253 |

All requests succeeded. The concurrency-2 result confirms that virtual slots
provide state-safe admission over physical B1 but no aggregate throughput
gain. Runtime endpoint telemetry retained all 48 expert layers on device with
16,986,931,200 resident expert bytes per device and zero expert H2D, expert
service, or route-read/stall time. The raw results and methodology are in
`../final_resident_vllm/README.md`.

## Reproduction

From the `tt-metal` root:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1,2,3
unset TT_MESH_GRAPH_DESC_PATH QWEN38_FUSED_RECURRENT_GDN
export RUN_QWEN38_PERF=1
export QWEN38_COLLECT_DECODE_TIMELINE=0
pytest -q -s \
  'models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_batch1_prompt128_generate128_performance[blackhole-True-device_params0]'
```

Auto-discovery is required on this host. The legacy `p300` descriptor exposes
only two logical devices, while the checked-in `p150_x4` descriptor requests
four Ethernet channels per edge on hardware with two.

## Next performance work

The QSA compressed page map is now shared across all QSA layers, the full
resident stack was reprofiled, prefill is stack-major in 128-token
microchunks, resident decode uses two stack-level trace submissions, and the
server has a persistent revision-scoped expert tensor cache. See
`../batch1_optimization/README.md` for the measured results. Virtual
concurrency remains outside the selected serving configuration.

1. Continue the batch-one QSA work beyond the delivered shared page-map
   bypass: prefix-bound or tier the compressed-key scorer so advertised 262K
   cache capacity does not score all 32,768 compressed blocks for a short
   request, then fuse the remaining query selection, validity mask, and gather
   path.
2. The endpoint/direct gap is resolved for the selected workload. The direct
   gate used a 4K selector while the old launcher constructed a 262K selector;
   aligning both at 4K changed endpoint TPOT from 125.69 to 86.84 ms and
   matched the 87.48 ms direct result. Runtime telemetry now reports both
   sequence capacity and compressed selector blocks.
3. Measure warm startup from the delivered persistent BFP4 tensor cache and
   retain its exact on-disk size. Cold startup still performs the original
   conversion while populating the cache.
4. Evaluate folding q/k normalization into the recurrent program only with
   the exact real-weight advancing-state trajectory gate.
5. Keep concurrency out of the selected path. Revisit it only after a true
   physical batch-2 trace exists and improves aggregate throughput.
6. Do not enable the checkpoint's single MTP layer until decode can verify a
   multi-token proposal and transactionally roll back GDN, convolution, QSA,
   and PLE state. The current host PLE n-gram lookup provides model features;
   it is not a standalone draft-token proposer. Runtime telemetry records this
   distinction and reports speculation as disabled.
