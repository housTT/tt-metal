# Reduced full-model profiling

These profiles use real layers 0 and 3, the real embedding, terminal norm,
selected BF16/HiFi4 head, and canonical split sampler. They preserve TP4 on
four Blackhole chips on two P300c boards, batch 1, native262144 cache, prompt128,
all real tensor shapes, and model/sampler traces. No all-layer stack was profiled.

| Window | Device0 | Device1 | Device2 | Device3 |
|---|---:|---:|---:|---:|
| Decode kernels + interior gaps, ms/replay |2.698191 |2.698374 |2.696221 |2.699233 |
| Prefill kernels + interior gaps, ms |5.277278 |5.279348 |5.270494 |5.276842 |

Decode averages four signposted replays at positions129–132. Prefill measures
one128-token prefill plus first-token sampling. Device clocks are independent;
chip times are never summed. Only each device's first gap, which starts before
the signpost, is excluded from window accounting; all kernel times and interior
gaps are retained. Original rows and hashes remain available.

The decode window contains119 model ops and32 sampler ops per replay per chip.
Model trace2 consumes about1.990 ms kernels plus0.129 ms gaps; sampler trace3
consumes0.507–0.509 ms kernels plus0.071–0.072 ms gaps. The sampler's approximately
0.58 ms is under5% of the measured12.26 ms all-layer token-out latency. It does
not dominate token-out decode. [Split accounting](decode_split_accounting.json)
retains each rank separately. The common-sampler semantic-greedy comparison
also measured0.574 ms versus2.748 ms for force-argmax.

Runtime rows confirm BFP4/LoFi decoder projections, BFP8/LoFi decode QKVG,
FP32 recurrent math, and BF16/HiFi4 terminal projections. The two head chunks
measure approximately658–662 us each. The report's118% FLOPs estimate is a
modeling error: its Blackhole DRAM-sharded model assumes eight compute workers,
while the configured two readers per bank enable sixteen compute workers.
The corrected modeled utilization is about59%; measured bandwidth is408 GB/s,
about80% of the512 GB/s modeled memory limit. No raw timing or rendered heuristic
was edited. [Exact classification](head_roofline_classification.json) records
source hashes, native worker selection, and formulas. This supports a bandwidth
bound head; it does not claim untested lower-fidelity candidates are impossible.

The parser initially demanded timings for initial captured traces0/1 that were
released without replay after prefill program compilation. The repaired parser
excludes only these unused capture definitions. Missing executed operations still
fail. All5772 decode and3920 prefill device rows are retained; the signposted
windows contain2416 and564 rows respectively. See
[AutoFix](../AUTOFIX_tracy_unreplayed.md) and
[decode coverage](../tracy_decode_coverage_final.json).

Reports:

- Decode: [merged text](decode_perf_report.txt), [CSV](decode_perf_report.csv),
  [rank0](decode_device0_report.txt), [rank1](decode_device1_report.txt),
  [rank2](decode_device2_report.txt), [rank3](decode_device3_report.txt),
  [accounting](decode_rank_accounting.json), [provenance](decode_provenance.json).
- Prefill: [merged text](prefill_perf_report.txt), [CSV](prefill_perf_report.csv),
  [rank0](prefill_device0_report.txt), [rank1](prefill_device1_report.txt),
  [rank2](prefill_device2_report.txt), [rank3](prefill_device3_report.txt),
  [accounting](prefill_rank_accounting.json), [provenance](prefill_provenance.json).

Collection commands are in `../logs/tracy_{decode,prefill}_final_v1.provenance.json`.
Decode's initial command correctly retains its report-generation failure; its
successful CPU recovery has separate immutable provenance linked in AutoFix.
`render_perf.py` invokes canonical `tt-perf-report` with start/stop signposts and
tracing mode for decode. Pass `--source-csv` when multiple recovery reports exist.
Full raw captures remain in the persistent workspace and are indexed by
`../artifact_manifest.json`; compact signposted operation archives and reports
are included in the local checkpoint.

The optional Tracy viewer warns when looking in the default capture directory
instead of the custom output path; the actual host capture exists and is hashed.
Pandas also reports mixed-column inference during import. Exact raw-to-report
execution/timing coverage controls both warnings; required report data is intact.
