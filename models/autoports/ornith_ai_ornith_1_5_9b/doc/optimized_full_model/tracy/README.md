# Reduced full-model profiler evidence

The final profiles in `prefill_trace_release/` use real layers 0 and 3 plus
the complete embedding, terminal norm/head, common sampler and token feedback
on four Blackhole chips on two P300c boards. Batch 1, native 262144-token cache,
prompt 128, and selected tensor shapes and precision are preserved. Decode
includes the traced UINT32 output-history variant. Prefill measures the actual
warmed public `generate([100] * 128, 1)` request, including request setup,
first-token readback and post-first-token position/page/history setup. No
all-32-layer stack was profiled. The older files directly in this directory
remain historical evidence for the earlier eager-prefill control.

| Signposted window | Device0 | Device1 | Device2 | Device3 |
|---|---:|---:|---:|---:|
| Decode kernels, ms/token |2.233566 |2.233346 |2.235230 |2.231489 |
| Decode interior gaps, ms/token |0.196449 |0.195973 |0.195222 |0.197405 |
| Decode kernels + interior gaps, ms/token |2.430015 |2.429319 |2.430452 |2.428894 |
| Prefill request kernels, ms |3.096754 |3.103987 |3.096260 |3.092638 |
| Prefill request interior gaps, ms |1.239273 |1.239032 |1.236486 |1.242698 |
| Prefill request device window, ms |4.336027 |4.343019 |4.332746 |4.335336 |

Four decode replays execute positions129–132. The same instrumented host window
measures 2.440547018 ms/token, 0.010094768 ms above the slowest device
(rank 2, 2.430452250 ms; 0.414% of host time). The
same-path optimistic weight/KV bandwidth floor is1.176782ms/token at the installed
512GB/s/chip model; it omits state/activation traffic, cache writes, collectives,
compute and small-kernel overhead. This floor is not a prediction of achievable
latency. [Full accounting](../perf_summary.json) keeps this same-run triplet
separate from uninstrumented all-32-layer token-out timing 12.002670 ms/token.
The full-stack device-time field is null because the skill forbids full profiling;
no synthetic all-layer device measurement is substituted.

Each device has 624 decode operations across four replays: 119 model and 37
sampling/history operations per token. Rank 0 model trace 4 is 1.720115500 ms
kernels + 0.122734250 ms attributed gaps; sampler/history trace 6 is
0.513450750 + 0.073714750 = 0.587165500 ms. This reduced-path sampler/history
measurement is about 4.89% of the separately measured full-model token-out
latency. The canonical TopkLargeIndices/route operations replace generic TopK; only32 candidates per
rank are gathered, with no full-vocabulary gather or argmax path. IndexedFill,
UINT32 Copy and PlusOne implement output history. The final terminal sequence is
L1 FillPad→RMSNorm→Reshard→two heads→logit assembly, with the direct final
decoder boundary.

The prefill [host window](../profile_prefill_prefill_trace_host.json) is
4.634666024 ms wall time. Its request TTFT is separately retained at
4.465411999 ms, including 0.759924995 ms setup. The full window additionally
includes post-TTFT position/page/history uploads and the final profiling
synchronization. Window counters prove one prefill replay and one first-token
sampling replay, zero prefill captures or eager prefill calls, and no decode or
history replay. The unchanged prefill page table is not uploaded again.

The [prefill split](prefill_trace_release/prefill_split_accounting.json)
contains 145 operations per rank: eight eager request-boundary operations,
103 operations in prefill trace 7, and 34 in plain sampling trace 5. Rank 0
prefill trace kernels/attributed gaps are 2.573654/0.188683 ms; sampling trace
kernels/attributed gaps are 0.503750/0.069571 ms. A phase's gap sum includes
the incoming handoff assigned to its first operation. For example, rank 1
prefill trace 7 has 0.188870 ms attributed gaps, including 0.116464 ms before
its first embedding after host token preparation/upload. These are not solely
gaps within the captured body; the complete-window sums retain all gaps once.
The eight eager operations account for only 0.019350 ms kernels but 0.981019 ms of preceding gaps. Those eager rows occur
before trace 7; this is request reset/admission/configuration/validation work
between launches, not a stalled captured decoder body. Post-TTFT setup belongs
to the complete host window and is not represented by these eight kernel rows.

The four seed-merge operations (tilize→where→untilize→copy) total 0.006984 ms
kernels and 0.114530 ms of subsequent gaps on rank 0. The much larger
0.757561 ms gap precedes their first operation and overlaps host request setup.
Coalescing seed uploads is a small, unmeasured request-boundary opportunity;
it must preserve both host RNG admissions and their resulting device stream.
These measurements do not establish either that every boundary wait is
mandatory or that the entire 0.981019 ms can be removed. The earlier controlled
dispatch-gap investigation and integrated exactness gates are in
[AutoFix prefill gaps](../AUTOFIX_prefill_gaps.md) and
[AutoFix integration](../AUTOFIX_prefill_integration.md).

Device clocks are independent and chip times are never summed. The renderer
excludes only each first gap, which begins before the signpost; every kernel and
interior gap remains. All 2496 decode and 580 prefill signposted device rows
are retained in the per-rank window CSV artifacts, compressed by finalization;
original full CSV hashes and paths are in provenance. Host JSON `counters`
are cumulative across warmups, while `window_counters` describes exactly the
signposted region. Prefill is split into `request_boundaries`, `prefill`, and
`sampling`; output-history replay is confined to the decode profile.

## Runtime policy and advice

[Decode contracts](prefill_trace_release/decode_runtime_contracts.json) and
[prefill contracts](prefill_trace_release/prefill_runtime_contracts.json) retain exact tensor shapes,
dtypes, fidelity, memory, program configs, kernel sources and call IDs for every
unique matmul and collective program. Runtime confirms BFP4/LoFi decoder
projections, BFP8/LoFi **decode** QKVG, FP32 recurrent math, and BF16/HiFi4 head.
Prefill QKVG correctly remains BFP4/LoFi. The packed MLP uses32 input cores/K4/R3
for gate/up and8/K6/R2 for down; the64-core terminal usesK1/R2/per_core_N16.

| Report advice or material boundary | Evidence-based conclusion |
|---|---|
| Head chunks marked BOTH; raw modeled FLOPs 143.861% | Report model assumes 8 workers; native two readers across 8 banks means 16 compute workers. Corrected modeled FLOPs is 71.930%. Rank 0 mean head kernel is 539913.5 ns; 268435456 weight bytes / duration gives 497.182 GB/s, 97.11% of the 512 GB/s model. Raw rows are unchanged. [Head classification](prefill_trace_release/head_roofline_classification.json) records source hash and formulas |
| Packed gate/up raw FLOPs can exceed100% | Same worker-count heuristic: three readers imply24 workers, so divide reported compute percentage by3. Down's two-reader model uses16; QKVG's one-reader model uses8. Exact per-program correction metadata is retained |
| SLOW packed GDN4096×3136 and output1024×4096 | The report says K8/subblock1×4 look good. Precision-locked smaller/larger grids, DRAM-sharded and fused/coherent families are already measured in `../../optimized_multichip_decoder/optimization_evidence.md`, `candidate_measurements.csv`, `packed_default_families_resumed.json` and `validated_full_attention_families.json`; native selected geometry remains faster. Final rank 0 decode means are 28.484 us for packed GDN, 9.612 us for its output and 9.848 us for attention output |
| Suggest HiFi2/HiFi4 for BF16 activation accuracy | Generic advice does not override the user's selected BFP4/LoFi policy and real-model rejection ledger. Final AIME top5/top100100%, and prompt-correct qualitative controls pass. No slower fidelity policy is silently introduced |
| Head core/readers/K geometry | `../AUTOFIX_head_geometry.md` closes16/32/64-core precision-locked controls, exact K2 L1 collision and adapted33024-column three-reader packing. Fixed64/K1/R3 is15.1% slower than selected64/K1/R2. Lower head dtypes remain rejected by actual French generation controls |
| Final norm/layout movement | Native L1 padding plus common head selected; independent review found and removed the last decoder DRAM hop. Exact B1/B4/B32 eager/trace and watcher controls plus final generation gates verify the direct boundary |
| Embedding/sampler collective ownership | Fixed decode embedding and named candidate outputs use preallocated buffers, with disposable clones. Paired component measurements preserve exact results and show small gains. Arbitrary prefill sizes stay temporary |
| Decoder collectives/residual | Native two-link ring reduce-scatter/all-gather implementing all-reduce remains selected from adapted carried-shard/fused/packed/persistent whole-layer comparisons. No inter-layer conversion or precision fallback is introduced |

The decode residual above the optimistic bandwidth floor comprises measured
recurrent/elementwise operations, data movement, communication, compute and
interior dispatch gaps. Rank 0 decode kernels total 2.233566250 ms, with
0.196449 ms interior gaps. Final rank 0 kernel-group accounting is:

| Kernel group | Decode, ms/token | Prefill request, ms |
|---|---:|---:|
| Matmul |1.286850 |1.699025 |
| Collective |0.108726 |0.205305 |
| Data movement |0.210993 |0.201385 |
| Other compute |0.626998 |0.991039 |

Every rank's exact groups are retained in the linked rank accounting. The
repeated decoder choices are tied to the predecessor's coherent-family ledger, not rejected from isolated
incompatible-policy probes. No applicable new untested material matmul advice
remains in the final report.

## Collective configuration

All material collective programs are retained verbatim in runtime-contract JSON.

| Boundary | dtype / dimension / topology | Links / memory / persistence |
|---|---|---|
| Decode embedding1024→4096 | BF16,dim3,ring4 |1,DRAM,true |
| Prefill embedding1024→4096 | BF16,dim3,ring4 |1,DRAM,false; avoids persistent request-size allocations |
| Decoder reduce-scatter4096→1024 then all-gather→4096 | BF16,dim3,ring4 |2; decodeL1/prefillDRAM; native packet8192 and fabric1D ring |
| Sample local32→128 candidate values | BF16,dim3,ring4 |1,DRAM,true |
| Sample local32→128 candidate indices | UINT32,dim3,ring4 |1,DRAM,true |

Async gather attributes record `chunks_per_sync`, `num_workers_per_link` and
`num_buffers_per_channel` as nullopt: the native implementation selects them;
no explicit tuning value is claimed. Full-grid cycling semaphores and barriers
are source-backed in `SamplingCCL`/`MeshCCLManager`. Decoder persistent/fused
families were measured at whole-layer scope and rejected as slower; helper
persistence remains enabled where its compatible ownership trial wins.

## Artifacts and reproducibility

- Final decode: [merged text](prefill_trace_release/decode_perf_report.txt),
  [CSV](prefill_trace_release/decode_perf_report.csv),
  [rank0](prefill_trace_release/decode_device0_report.txt),
  [rank1](prefill_trace_release/decode_device1_report.txt),
  [rank2](prefill_trace_release/decode_device2_report.txt),
  [rank3](prefill_trace_release/decode_device3_report.txt),
  [rank accounting](prefill_trace_release/decode_rank_accounting.json),
  [trace split](prefill_trace_release/decode_split_accounting.json),
  [host window](../profile_decode_prefill_trace_host.json),
  [provenance](prefill_trace_release/decode_provenance.json).
- Final prefill request: [merged text](prefill_trace_release/prefill_perf_report.txt),
  [CSV](prefill_trace_release/prefill_perf_report.csv),
  [rank0](prefill_trace_release/prefill_device0_report.txt),
  [rank1](prefill_trace_release/prefill_device1_report.txt),
  [rank2](prefill_trace_release/prefill_device2_report.txt),
  [rank3](prefill_trace_release/prefill_device3_report.txt),
  [rank accounting](prefill_trace_release/prefill_rank_accounting.json),
  [request/trace split](prefill_trace_release/prefill_split_accounting.json),
  [host window and TTFT](../profile_prefill_prefill_trace_host.json),
  [provenance](prefill_trace_release/prefill_provenance.json).
- Historical eager-prefill profile: [report](prefill_perf_report.txt),
  [rank accounting](prefill_rank_accounting.json),
  [provenance](prefill_provenance.json), and [host window](../profile_prefill_host.json).
  Its 576 device rows and 5.115028 ms slowest device window belong to the earlier
  eager prefill-plus-sampling scope, not the final complete public request.
  Historical [decode report](decode_perf_report.txt) and
  [provenance](decode_provenance.json) remain alongside it unchanged.

Exact collection and rendering commands are in [decode collection](../logs/tracy_decode_prefill_trace_v1.provenance.json),
[prefill collection](../logs/tracy_prefill_prefill_trace_v1.provenance.json),
[decode rendering](../logs/render_decode_prefill_trace_v1.provenance.json), and
[prefill rendering](../logs/render_prefill_prefill_trace_v1.provenance.json).
`render_perf.py` invokes canonical tt-perf-report with start/stop signposts,
CSV/text output and tracing mode for both final profiles. `profile_details.py` derives contract,
trace-split and report-heuristic classifications; `summarize_perf.py` writes the
same-run roofline/device/host reconciliation. `finalize_artifacts.py` compresses
signposted CSV and completed logs and indexes local raw captures/tensors.
Both collections, both renderers, and
[detail extraction](../logs/profile_details_prefill_trace_v1.provenance.json)
exit 0 under the `prefill_trace_release` label.

The optional viewer warns about the default capture path while this run uses a
custom output directory. The actual host `.tracy` and required ops/device CSV
exist at the provenance paths; canonical collection/rendering both exit0.
An already-running viewer on port18940 predates this stage; no viewer process
was created or stopped here. Pandas mixed-column inference and allocation-after-
capture advisories are retained; exact timing coverage and separate final
watcher/allocation controls classify them. No device profiler overflow, missing
executed trace timings, host fallback or device-health fault occurred.
