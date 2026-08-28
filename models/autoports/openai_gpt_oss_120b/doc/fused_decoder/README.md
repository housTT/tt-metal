# OpenAI GPT-OSS 120B fused decoder

Status: fused implementation, graph exhaustion, real-checkpoint correctness,
capacity, stress, watcher, and final profiler gates are complete. The final
fresh independent stage review returned `clean-pass`.

The implementation is `tt/fused_decoder.py`; it never imports or calls the
functional decoder. Prefill uses Blackhole's dedicated
`unified_routed_expert_moe` with fabric-free local regroup. Decode uses a
checkpoint-qualified FullLocal `moe_compute` graph where BF4 precision passes,
and the best correct compact indexed graph everywhere else.

This stage uses one P150-class Blackhole device (`1x1`). P150, P150x2, and
P150x4 are supported target hosts by selecting one device. Tensor/expert
parallelism across x2/x4 belongs to the later multichip-decoder stage and was
deliberately not started.

## Preserved public contract

`FusedDecoder.from_state_dict`, `prefill_forward`, `decode_forward`, `forward`,
and `kv_cache` preserve the functional decoder contract:

- paged BF8 KV cache, device-resident `int32` page tables and positions;
- context 131072, decode at position 131071, page size 64, and batch-32 full
  context capacity;
- exact logical output lengths for every positive HF-valid prefill length;
- no public tile, page, window, routing, or 4096-token chunk divisibility rule;
- complete decode trace capture/replay and bitwise-deterministic replay;
- sliding-attention and full-attention layer kinds.

Prefill privately pads routed chunks to 64 rows and removes the padding before
return. Chunks are at most 4096 tokens. FullLocal is capped at configured
maximum batch 2 because wider whole-decoder replay was not deterministic; batch
3-32 transparently selects the indexed fused path. Therefore the machine-
readable [context contract](../context_contract.json) is unchanged.

## Delivered graph and qualification gates

Prefill fuses all 128 expert projections, biases, OAI-SwiGLU, and down
projections into the dedicated routed-expert operation, surrounded by
device-only count/sort/gather/scatter and `post_combine_reduce`.

Decode uses public FullLocal `moe_compute` and
`deepseek_moe_fast_reduce_nc_fused` only when all four conditions hold:

1. checkpoint revision is `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`;
2. layer is one of 0-13, 21, 23-27, 29, 31, or 33-35;
3. effective matmul ring size is 8;
4. configured maximum batch is 1 or 2.

Setup uses public `quantize_weights_via_host` for higher-accuracy BF4 weights.
Every layer was swept with real checkpoint weights at batch 1 and 2; the 25
qualified layers all meet direct fixed-route PCC >=0.995. Second-seed checks
removed marginal layer 28. The other 11 layers, unknown/mismatched revisions,
ring-7 devices, and larger batches use setup-packed indexed top-4 gate/up and
down sparse matmuls with indexed biases. Both are fused paths; neither falls
back to `FunctionalDecoder`. See
`candidates/full_local_moe_compute/host_quant_layer_qualification.csv`.

## Correctness

Whole-layer acceptance remains 0.95 for batch-one prefill/decode, 0.99 for the
real batch-two stress, and 0.995 for direct fusion equivalence.

| Real checkpoint | Path | Prefill PCC | Traced decode PCC | Deterministic |
| --- | --- | ---: | ---: | --- |
| layer 0, sliding | FullLocal | 0.978123157 | 0.990064440 | yes |
| layer 1, full | FullLocal | 0.990889623 | 0.955780929 | yes |
| layer 1, full, batch 2 | FullLocal | 0.992492223 | 0.992720098 | yes |
| layer 1, full, batch 32 | indexed capacity path | finite/capacity gate | trace exact | yes |
| layer 5, full, forced indexed | 0.998950699 to functional | 1.0 to functional | yes, bitwise decode |

The full-attention FullLocal decode value is 0.010151 below the prior indexed
value (0.965932) because of BF4 expert quantization, but remains above the 0.95
functional acceptance bar. Direct fixed-route FullLocal layer-1 PCC is
0.999758583/0.999741804 at batch 1/2.

The final-source layer-5 fallback A/B forces the production-reachable
unqualified-revision selector and asserts the indexed representation is
constructed. It passed three consecutive whole-layer runs at prefill/decode
PCC 0.998950699/1.0 to `FunctionalDecoder`, with bitwise-equal traced decode
and replay. The earlier 0.355932/failed artifact was generated before the
final `fused_decoder.py`; it is retained with a `before_final_source` suffix,
while `candidates/full_local_moe_compute/indexed_layer5_functional_equivalence.log.gz`
is the SHA-attested final-source rerun.

Boundary coverage includes logical lengths 1, 31/32/33, 63/64/65,
127/128/129, 4095/4096/4097, 131071, and 131072; decode at position 131071;
batch 2 semantics; and batch 32 capacity. The fresh post-FullLocal boundary run
passed both layer kinds. Five hundred complete trace replays per path/kind were
bitwise deterministic.

## Before/after performance

Final real-checkpoint Tracy captures use sequence 128, a warmed prefill, and a
warmed complete decode trace. The functional numbers are the completed
functional-stage device-op baselines. Indexed is the best correct pre-FullLocal
decode candidate, measured in the same exact-revision A/B harness.

| Layer kind | Functional prefill | Final fused prefill | Change | Functional decode | Indexed decode | Final FullLocal decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding | 137.401 ms | 36.220 ms | 73.64% lower | 2.060 ms | 1.312 ms | 0.678 ms |
| full | 137.418 ms | 35.558 ms | 74.12% lower | 2.047 ms | 1.320 ms | 0.689 ms |

FullLocal decode is 48.32%/47.80% lower device time than indexed and
67.10%/66.35% lower than the functional baseline. It uses 34 device operations
versus indexed's 67, with zero host operations. The separate 500-replay wall
A/B measured:

| Layer kind | FullLocal | Indexed | FullLocal change |
| --- | ---: | ---: | ---: |
| sliding | 0.719116 ms | 1.368594 ms | 47.46% lower |
| full | 0.728915 ms | 1.367084 ms | 46.68% lower |

The direct prefill expert A/B independently measured 205.288-to-32.829 ms wall
(6.253x), functional-to-fused PCC 0.999770207, and 205.203-to-32.623 ms Tracy
device sum. Raw CSVs, filtered detailed reports, human-readable tables, and
stacked summaries are retained under `tracy/final_full_local/`; standalone and
rejected-candidate evidence is under `candidates/`.

## Exhaustion, runtime, and hardware evidence

[graph_fusion_assessment.md](graph_fusion_assessment.md) assesses every
dedicated-op, graph-rewrite, and adjacent-op pattern from the skill. It records
applied FullLocal/score-reducer work and rejected fabric, generalized-gate,
output-placement, dummy-index, compact-reducer, wider-batch, attention-fold,
and alternate MoE candidates.

[runtime_fallback_audit.md](runtime_fallback_audit.md) audits the measured call
tree. Final windows contain no Torch, `from_torch`, `to_torch`, host fallback,
fabric, or reshard operation. Required layout adapters total about 12.8 us in
FullLocal decode. The profiler helper was corrected to use 128 distinct expert
cache inputs before all final evidence was regenerated.

The exact-revision watcher run passed both layer kinds, with no watcher,
kernel, NoC, semaphore, or trace-allocation error. A post-profiler smoke command
accidentally inherited slow dispatch and required the documented TT-device
recovery sequence; focused Ethernet/ARC triage, a bounded reset, four-board
enumeration, and a corrected `MESH_SMOKE_OK` run establish clean final health.

See [work_log.md](work_log.md) for exact commands, candidate ledger, evidence
paths, review status, limitations, and commit SHAs. [artifacts.sha256](artifacts.sha256)
covers all retained evidence.
