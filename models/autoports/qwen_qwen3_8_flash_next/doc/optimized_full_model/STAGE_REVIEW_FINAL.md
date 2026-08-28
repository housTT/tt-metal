# Independent optimized-full-model stage review

Verdict: `more-work-needed`

The selected path has substantial positive evidence, including the current
selected-endpoint AIME prefill/autoregressive run, the 99-row teacher-forcing
run, the full-48 Watcher/allocation run, the current 34-test static suite, and
the refreshed five-case selected BFP8 split-sampling/mixed-state suite. The
exact prepack/cache/PLE counters are internally consistent between the lazy and
prepacked timelines, and the sampler is not the dominant token-out cost.
However, the performance-closure gate and several required evidence contracts
are not yet substantiated.

## Required work, ranked

### P1. Replace the cache-service rate with a real physical DMA lower bound and close the avoidable host-service gap

The claimed 133.852526 ms/token mandatory transfer floor is derived from
6.820764 GB/s in `completed_owner_dma_bandwidth.xml`. The benchmark at
`tests/test_multichip_decoder.py:408-425` times `cache.ensure_indexed(...)`
followed by a whole-mesh synchronization. Each ten-expert wave serially issues
the two owner H2D copies and both owner and non-owner D2D copies in
`tt/host_weight_cache.py:654-687`, publishes cache indices, and incurs Python,
directory, command-submission, and synchronization overhead. The elapsed time
therefore measures the current exact cache-service implementation, not a
transport/link physical bound. Its bandwidth denominator counts only
27,648,000 owner-H2D bytes even though the timed interval also performs an equal
27,648,000 bytes of local zero/owner D2D work plus control.

This distinction is material: the selected timeline reports 100.786 ms/token
of route-read/device stall and 143.623 ms/token of cache/control/DMA submission.
Adding the current service rate to the preserved layer and PLE measurements as
a mandatory floor makes the observed implementation its own lower bound; it
does not prove that the requested greater-than-10--15% avoidable gap is closed.
`AUTOFIX.md:61-62` dismisses additional architectures as contract/API changes
without a retained API experiment or minimal blocker.

Required action: retain a transport-only completed H2D/link measurement for the
exact packed owner bytes that excludes D2D, publication, directory, Python, and
final-service overhead; separately retain current cache-service latency. Test
the applicable exact alternatives (at minimum batched/coalesced expert
transfers, rank-parallel service, multiple stable staging buffers, and deeper
DMA/compute overlap), or retain a minimal source/API/hardware blocker for each
inapplicable option. Recompute the lower bound and current full-token gap from
those independent terms, then demonstrate closure or continue optimization.

### P1. Exercise the explicit LM-head profiler advice and finish the terminal policy matrix

The selected LM head remains DRAM-interleaved: its lazy weights and inputs and
outputs use `ttnn.DRAM_MEMORY_CONFIG`, with no explicit program configs or
weight memory configs in `tt/model.py:367-388`. In the selected profiler, all
four terminal matmuls (`32 x 2560 x 32768` three times and
`32 x 2560 x 25856`) report `DRAM Sharded=False`, inner block 2, and explicitly
recommend `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`; see
`final_profiler/layer0_gdn/token_out_report.csv:188-191`. The optimization
matrix sweeps only dtype/fidelity and contains no LM-head sharding, grid,
program-config, K-block, split-width, or output-subblock result. Calling the
terminal work immaterial depends on the unproven physical bound above and does
not satisfy the advice-attempt gate.

The BFP4/LoFi row is also rejected after the first API failure. The retained
failure is in sampler `ttnn.untilize`, which requests a row-major result for a
reduced low-precision row and raises `Only TILE layout is supported for
BFLOAT8_B dtype`; it does not show that a sampler-ready BF16 conversion,
tile-preserving gather, or adjusted terminal output is impossible.

Required action: A/B an applicable DRAM-sharded LM-head configuration and
record the full dominant-matmul geometry/dtype/fidelity/layout/program-config
matrix with real endpoint correctness and full token-out performance. Resolve
the BFP4 sampler boundary with the smallest appropriate conversion/layout
adaptation or retain a focused proof of the exact API limitation. Reprofile the
selected result with advice enabled and explain every still-applicable advice
row with measured evidence.

### P1. Refresh and bind every selected-policy gate to the same current source

Several artifacts do not demonstrate the current selected BFP8 implementation:

- `final_batch32_and_context_capacity.xml` predates endpoint selection and
  records 23,691,657,304 planned bytes/device and 10,533,863,336 bytes of
  headroom. Current `README.md:206-210` and `context_contract.json` claim
  23,393,673,304 and 10,831,847,336. The linked batch-32 artifact therefore
  contradicts the current capacity contract by 297,984,000 bytes/device.
- `final_qualitative_shared_suite.xml` completed at 01:58, before the BFP8
  endpoint sweep at roughly 02:55. `README.md:159-162` inherits it because the
  decoder did not change, but the endpoint producing logits did change. The
  required three-prompt shared suite must follow the selected optimization;
  the one-prompt AIME completion is complementary, not a substitute.
- The prepacked/lazy selected performance JSONs completed by 03:12 and the raw
  selected profiler captures by 03:22, while the current `tt/model.py` was
  modified at 03:24. `profiler_provenance.txt` records only base commit
  `a38187012...`, not a dirty-source digest. `artifact_manifest.sha256` hashes
  current source and old artifacts at finalization, but does not prove which
  source produced a run. It also omits the batch-32, shared qualitative, and
  profiler artifacts.

The refreshed `final_selected_split_sampling_mixed_contracts.xml` is not part
of this finding: it is current, passes 5/5 on the selected endpoint, and its
manifest hash verifies.

Required action: rerun batch-32/context construction, the complete shared
qualitative suite, warmed 128+128 TTFT/token-out performance, and profiler
windows on one frozen current-source snapshot. Record the dirty source hashes
(or an exact patch/tree identity), selected policies, command/environment,
workload, and output hashes in run provenance, and include all primary selected
artifacts in the manifest. Capacity output must agree exactly with the current
contract.

### P2. Substantiate the claimed GDN/LoFi rejection without changing the inherited policy prematurely

The stage claims a retained three-run resident screen tied decode and slowed
prefill, but the linked evidence does not contain that screen.
`optimized_multichip_decoder/candidate_gdn_bfp8_lofi.xml` has one testcase for
each of layers 0 and 1. Its real-weight layer-0 candidate passes PCC
(prefill 0.99977404, decode 0.99986905) and reports 157.846770 ms prefill and
1.987174 ms segmented decode, versus the selected layer-0 medians of
158.877355 ms and 2.082220 ms. In
`tests/test_multichip_decoder_perf.py:712-732`, the printed
`baseline_traced_decode_ms` comes from a separate default `OptimizedDecoder`;
only the host-backed candidate receives `_candidate_layer_kwargs()`. It is not
a resident LoFi control. The one-shot route/miss-mix concern is reasonable but
does not constitute the claimed same-work three-run refutation, and no
full-model LoFi accuracy/token-out artifact exists.

Required action: keep the inherited HiFi2 default while retaining either (a)
the claimed same-input, same-route, same-miss/source-pack, multi-run resident
HiFi2-versus-LoFi A/B plus full-model accuracy/token-out result, or (b) a
model-visible correctness failure that earns rejection. Update the matrix and
prose to cite the actual retained evidence.

### P2. Complete the profiler/performance reporting contract and correct teacher-forcing semantics

The four raw profiler CSV hashes match `profiler_provenance.txt` and
`final_profiler/perf_summary.json`, and the compact window totals are
arithmetically consistent. Nevertheless, `final_profiler/tt_perf_report_table.md`
is only a four-row aggregate, not a retained human-readable operation/advice
table. `perf_summary.json` contains representative reduced windows only; it
does not identify a complete-model workload or reconcile theoretical
roofline, measured device time, warmed TTFT, full end-to-end token-out, and host
overhead for the same workload as required.

The teacher-forcing performance label is also inaccurate. `README.md:8` says
sampling/token feedback/readback are excluded, but
`demo/full_model.py:99-116` calls decode with
`host_sampling_compatibility=True`, `read_from_device=True`, concatenates full
logits, and records `compatibility_mode="explicit_host_logits"`. Sampling and
token feedback are excluded; full-logits readback is included.

Required action: retain the actual text operation/advice tables produced by
`tt-perf-report`, expand the machine-readable summary to the required
same-workload complete-model accounting and provenance, and either relabel the
355.862 ms teacher-forcing number as explicit full-logits-readback compatibility
latency or add a separately named no-readback teacher-forcing measurement.

### P3. Repair the stage-owned evidence/documentation ledger

`optimization_matrix.csv` has a seven-column header but row 19 has six columns:
the top-k/top-p evidence is shifted into `latency_or_evidence`, leaving the
reason/artifact mapping malformed. `README.md:227-228` also prescribes the
result of this independent review before it exists. That is not evidence and
should not appear in a stage deliverable.

Required action: make every CSV row schema-consistent, rerun a CSV parser check,
and change the README to link the independent report without dictating its
verdict.

## Anomaly ledger

### Current service bandwidth presented as physics

- Evidence: `tests/test_multichip_decoder.py:408-425`,
  `tt/host_weight_cache.py:654-687`, and `lower_bound.csv`.
- Why suspicious: the numerator excludes compulsory work inside the timed
  interval, and the software service itself is then charged as an unavoidable
  hardware floor.
- Status: unresolved and stage-blocking.
- Required next action: independent transport bound plus exact service A/Bs as
  specified in P1.

### Profiler advises the untested LM-head layout

- Evidence: `tt/model.py:367-388` and selected profiler LM rows 2697--2700
  rendered at `final_profiler/layer0_gdn/token_out_report.csv:188-191`.
- Why suspicious: the stage declares the matrix complete while the selected
  report identifies a concrete applicable configuration that is absent from
  the matrix.
- Status: unresolved and stage-blocking.
- Required next action: measured DRAM-sharded A/B and selected-policy reprofile.

### Capacity artifact contradicts the selected contract

- Evidence: `final_batch32_and_context_capacity.xml` reports
  23,691,657,304 bytes/device, while current docs/contracts report
  23,393,673,304.
- Why suspicious: the artifact is used as proof for an endpoint policy that it
  predates.
- Status: unresolved and stage-blocking.
- Required next action: current selected-endpoint rerun with frozen-source
  provenance.

### Faster real-weight GDN candidate dismissed by an unretained experiment

- Evidence: `optimized_multichip_decoder/candidate_gdn_bfp8_lofi.xml`,
  `optimized_multichip_decoder/optimization_matrix.csv:42-43`, and
  `tests/test_multichip_decoder_perf.py:712-732`.
- Why suspicious: the linked artifact does not contain the asserted three-run
  same-work control, while layer 0 is faster and passes its isolated PCC gate.
- Status: unresolved; preserve HiFi2 until the candidate is fairly adjudicated.
- Required next action: retained controlled A/B plus full-model gate or a
  demonstrated model-visible blocker.

### Teacher-forcing boundary mislabeled

- Evidence: `README.md:8` versus `demo/full_model.py:75-80,99-116`.
- Why suspicious: full-vocabulary logits readback materially changes the
  meaning of the separately reported latency.
- Status: documentation/performance-contract defect.
- Required next action: correct the label or add a no-readback measurement.

## Hard-check gaps

- No current selected-endpoint batch-32 artifact agrees with the advertised
  capacity arithmetic.
- No post-selection shared three-prompt qualitative run exists.
- No run-time dirty-source identity binds the final full-model performance and
  profiler evidence to the current source.
- No transport-only physical H2D bound or controlled cache-service candidate
  matrix exists.
- No DRAM-sharded LM-head attempt or focused BFP4 sampler-boundary resolution
  exists despite explicit advice and a first-error-only rejection.
- No retained operation-level human-readable advice table or complete-model
  same-workload perf summary exists.

## Scope inspected and checks performed

The review inspected the complete dirty stage-owned source diff for generator,
model, multichip decoder, host cache, tests, contracts, documentation, all
selected JSON/JUnit/CSV evidence, raw and rendered profiler outputs, qualitative
text/reviews, the inherited optimized-decoder ledger, and the historical
`STAGE_REVIEW.md`. It used no vLLM path and ran no hardware workload.

Read-only checks confirmed:

- all stage JSON and all 35 top-level JUnit files parse; selected final gates
  pass, while the retained BFP8/LoFi accuracy rejection and BFP4 layout
  rejection are the two expected nonpassing candidate XMLs;
- the seven touched Python files parse as AST and `git diff --check` is clean;
- every entry in `artifact_manifest.sha256` verifies, including the refreshed
  five-test selected sampling artifact;
- the four retained raw profiler-operation CSVs match their provenance hashes;
- selected sampling is approximately 0.488--0.490 ms and is not dominant;
- lazy/prepacked steady routes, misses, hits, owner H2D, zero D2D, source-pack,
  and PLE counters reconcile, including 24,576 prepacked experts and
  68,080,435,200 packed host bytes.

## Residual risk

Until the items above are resolved on one frozen selected-policy source, the
reported 252.451 ms token-out path may still contain a material avoidable host
service gap, the endpoint may have an unmeasured advised speedup, and the
batch-32/qualitative/performance package cannot be reproduced as one coherent
current release candidate. Watcher, exact-cache semantics, context-length
preservation, split sampling, and sampler non-domination are encouraging but do
not close those independent gates.
