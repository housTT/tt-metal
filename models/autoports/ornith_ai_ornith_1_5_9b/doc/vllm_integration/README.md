# Ornith-1.5-9B vLLM integration

Primary single-user **128 input / 128 output / 1 request / concurrency 1 /
max-num-seqs 1**, greedy temperature0, first request after server readiness:
**TTFT62.613ms; decode87.706 tokens/s/user** from mean TPOT11.402ms;
ITL median11.364ms; aggregate output84.717 tokens/s. The full sampling profile
passed **72 tests, with one canonical skip** in the final all-layer run. Qualitative output is coherent and
on-topic, with a controlled selected full-model haiku constraint error described
below. Independent [stage review](stage_review.md) returned **clean-pass**.
Local checkpoint SHAs are recorded in [the work log](work_log.md).

## Serving configuration

All32 layers, four Blackhole chips on two P300c boards, mesh1x4, software
profile `P150x4`, internal TT tensor parallelism4, data parallel1.
vLLM itself uses `tensor_parallel_size=1`: its single TT worker owns the
four-chip mesh. Served `max_model_len=262144`
matches [the native contract](../context_contract.json); no capability reduction.
The shared KV pool supports a native-length request and32 concurrent short
requests; this does not claim32 simultaneous native-length requests.

The completed datatype-sweep predecessor is
`85710be49fbcce5579aeb6d9571cd3bc1829d953`. The adapter constructs its selected
`head4_lofi_last8_c32_k4_r2` policy unchanged: BFP4/LoFi body and head, BFP8
decode QKVG, last-layer attention/MLP BFP8 exceptions, BF16 activation/residual/
logits, native CCL including FP32 recurrent values, BFP8 paged KV and FP32
recurrent state. Full policy is printed by the server and remains defined by
`tt/model.py` and the datatype selection. Head uses32 cores, K4, two readers.

The task-local interpreter is `../state/serving-env/bin/python`; the pinned
HF snapshot is `../upstream` at
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. Dependency versions, effective
package inventory and TTNN binary hash are in [source pins](serving_source_pins.json),
[dependency lock](serving_runtime_requirements.lock) and [provisioning log](provision_work_log.md).
The preserved original TTNN environment was not replaced.

Launch the shared runner through the reproducible wrapper:

```bash
USER=hous ../state/serving-env/bin/python \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/run_server.py \
  --label full_b32_startup_final --max-num-seqs 32 --async-scheduling \
  --allow-host-sampling --sampling-profile full
```

The exact expanded command and source hashes are in
[full_b32_startup_final.command.json](full_b32_startup_final.command.json). TT configuration:
`sample_on_device_mode=all`, `trace_region_size=100000000`,
`l1_small_size=32768`, `fabric_config=FABRIC_1D_RING`,
`fabric_max_packet_payload_size_bytes=8192`; block size64; decode trace enabled.
`--hf-overrides '{"architectures":["OrnithForCausalLM"]}'` selects the explicit
registration. The plugin registers both `OrnithForCausalLM` and
`TTOrnithForCausalLM` in `platform.py::register_tt_models()`.

`ORNITH_VLLM_ALLOW_HOST_SAMPLING=1` is explicit compatibility for canonical
logprobs/host-only tests. Ordinary requests still use the on-device sampler.
The primary benchmark server disables this compatibility flag. No profiler,
watcher or trace-allocation tracker is enabled in performance runs.

## Adapter and correctness evidence

[generator_vllm.py](../../tt/generator_vllm.py) delegates prefill, model/decode
trace replay, sampling configuration, state remapping and deferred readback to
[generator.py](../../tt/generator.py). The generator receives the exact cache
allocated for vLLM's physical block pool. Constant-size linear-attention state
remains model-owned; full-attention layers share the paged KV group.

`supports_async_decode=True` is backed by
[adapter_device_tracker_startup_final.json](adapter_device_tracker_startup_final.json) and
[adapter_device_v2.json](adapter_device_v2.json): malicious stale host tokens
and positions, two deferred steps versus synchronous output, changed physical
pages across the64-token boundary, exact slot permutations, stable buffers and
cache identity. Steady replay needs no token/current-position/RoPE/page-table
upload; actual scheduler changes refresh the affected state. Decode uses
nonblocking model and canonical sampling traces and persistent device token
feedback. Minimal asynchronous readback copies the sampled token shard for
scheduler output. Explicit host compatibility is separate and is not the
benchmark path. Prefix caching remains disabled.

Full-model non-aligned prompt serving passes131-,129- and65-token requests,
including concurrent requests, in
[full_b32_verified_nonaligned.json](full_b32_verified_nonaligned.json).
The final startup-fixed server also passes ordinary device-sampled prose
requests of131 and65 tokens, singly and concurrently, with exact logical
usage and the established token prefixes
([final requests](full_b32_startup_final_nonaligned_prose.json)).
Meaningful131/65-token prose also passes exact API usage and repeated/permuted
numerical logprob checks in [logit_determinism_vllm.json](logit_determinism_vllm.json).
The matching selected all-layer standalone control has bit-identical complete
logits across device rows0/1/31, and all11 serving comparisons of token IDs,
chosen-token logprobs and top20 maps are exact
([control](logit_determinism_standalone.json), [commands and scope](logit_determinism.md)).
Logprob diagnostics explicitly use optional host compatibility and supply no
performance number. The earlier repeated-token inputs are structural stress
cases; their repetitive continuations are not the qualitative suite.
The original reduced layers0/3 probes are preserved as contract-debugging
evidence and are not reported as full-model quality or speed.

Final canonical sampling status is recorded in
[readiness_vllm/sampling_tests.log](../../readiness_vllm/sampling_tests.log).
The final shared-runner full profile completed exit0:72 passed,1 skipped in
329.38s. The skip is the canonical chat all-vocabulary logprob case.
The three previously failing tests now pass
([targeted result](full_b32_targeted_final.log)). The original69-pass/3-fail/
1-skip run remains in [full_b32_sampling_full_v1.log](full_b32_sampling_full_v1.log).
The bad-word assertion now checks tokenizer-defined forbidden sequences,
including negative controls; alternate tokenizations are not forbidden IDs.
A real plugin multi-token history bug was fixed and proven with exact live
[before](bad_words_history_before_server.json)/[after](bad_words_history_after_server.json)
requests. Presence tests retain their assertions and use a sensitive natural
continuation: exact device sampler math and raw-logit controls prove why the
old repetitive stimulus legitimately stayed unchanged at the tested penalties.
See [bad-word AutoFix](AUTOFIX_bad_words.md) and
[presence AutoFix](AUTOFIX_presence_variation.md).

The final source audit also found and fixed a canonical shared penalty-order
bug: repetition must scale the original logits before frequency and presence
subtraction. The independent pinned host implementation and13 CPU tests prove
the previous combined-penalty mismatch. Exact captured TP4/B32 sampler replay
now matches expected scores and tokens across all vocabulary shards, including
negative logits, zero crossings, prompt-only history and repeated output counts
([device proof](combined_penalty_sampler_exact.json)). Neutral and individual
controls also pass; no new sampler or dtype was introduced. See
[combined-penalty diagnosis](AUTODEBUG_combined_penalty_order.md).

Explicit host-to-device transitions also restore penalty history when request
parameters are unchanged, and initialize fresh host-prefilled rows from their
actual request seeds. Both delegate to the generator; continuing device rows
retain their live state. Exact reduced-device history, token, position and
seed proofs are in [host_resume_seed_device_exact.json](host_resume_seed_device_exact.json).

First-token variety tests now compare actual API token IDs instead of leading
text characters. The helper accepts an empty display only for a real
tokenizer-defined terminal EOS with `finish_reason=stop`. It preserves
response/token requirements and every variety/seed assertion; 17 CPU controls
and the three targeted high-temperature top-k cases pass. See
[first-token AutoFix](AUTOFIX_first_token_variety.md).

Startup now exercises the actual adapter admission path with the plugin's
neutral sampling parameters. This removes the verified first-request merge
compilation and trace-recapture pause. It then resets cache, tokens, positions,
seeds, output history and request flags while retaining warmed programs and
traces. [Native guard and state proof](b1_startup_after_tracker.json),
[before/after evidence](AUTOFIX_b1_latency.md) and
[source-scope check](warmup_source_scope.json) show this changes only startup;
all other adapter methods and the low-level generator/model are unchanged.

## Qualitative verdict and controls

All12 greedy/sampled texts were read in
[the shared output artifact](../../readiness_vllm/vllm_qualitative_outputs.json).
They are coherent and on-topic, without mechanical repetition, gibberish,
wrong-language drift or request contamination. The original HF chat template
and all six rendered prompts/IDs match the pinned controls
([format evidence](qualitative_prompt_format.json)). Thinking text consumes
much of the256-token budget; story and code completion are not established by
these truncated outputs. The greedy French translation is correct. The
mechanical degeneracy check passes.

The haiku request exposes a substantive syllable-count error: selected TT and
serving finish with6/7/5; the pinned BF16 HF control finishes with5/7/5.
This is a **selected full-model quality limitation**, not HF quality parity.
The selected standalone and served continuations match all390 raw token IDs,
including EOS ([exact comparison](haiku_standalone_serving_exact_comparison.json)).
That matching control rules out a new serving regression on this prompt; it
does not identify a quantization culprit or establish overall task accuracy.
The six prior selected-TT128 controls also match the served prefixes, but
ended before the haiku error. [The detailed review](qualitative_extended_control_report.md)
contains the completed HF/TT controls; [the final twelve-text reading](qualitative_final_review.md)
and its [hashed record](qualitative_final_review.json) assess the final server output. The selected
precision policy is preserved as required by this stage.

The separate test banning English greeting tokens produces Chinese and
off-topic continuations for seeds2/3. These remain poor task-quality outputs.
Paired unbanned controls answer normally; both branches have identical prompt,
seven-token prefix and raw distribution until the requested ban removes
` hello` and forces a different continuation. The pinned mask removes at
least97.40% of probability mass at that boundary in the exact-prefix control.
This is a controlled consequence of restricting the thinking text, rather
than evidence of request contamination. See
[raw controls](bad_words_distribution_control.json) and
[classification](AUTODEBUG_bad_words_distribution.md); this result does not
extend the normal-suite quality verdict to arbitrary constrained sampling.

## Benchmarks and cleanup

Primary **128 input / 128 output / 1 request / concurrency1 / max-num-seqs1**,
greedy temperature0, ignore EOS, native262144 context; first real request
after readiness, with no benchmark warmup:

| Metric | Primary128/128/1 |
| --- | ---: |
| Completed requests |1/1 |
| TTFT P50 / P99 |62.613 /62.613ms |
| TPOT mean / P50 / P99 |11.402 /11.402 /11.402ms |
| ITL P50 / P99 |11.364 /11.690ms |
| Aggregate output throughput |84.717tokens/s |
| TPOT-derived decode rate |87.706tokens/s/user |

[Primary normalized metrics](../../readiness_vllm/vllm_benchmark.json) and
[raw result](../../readiness_vllm/vllm_result.json) retain all127 token intervals.
The longest is19.106ms, at interval0. TTFT/TPOT P50 and P99 describe one
request, not a statistically established tail. The same-workload repeat
records50.127ms TTFT and87.898tokens/s/user; it is a consistency check, not
the headline. [All four before/after rows](b1_startup_benchmark_comparison.json)
retain the cold failure and unchanged-server controls.

The primary server command is the wrapper above with
`--label full_b1_startup_final --max-num-seqs 1 --async-scheduling --sampling-profile full`
and without `--allow-host-sampling`
([exact manifest](full_b1_startup_final.command.json)). The attached benchmark
uses `--stages benchmark --no-benchmark-ci-serving --additional-benchmark-args=--save-detailed`;
its full command is in [the runner log](full_b1_startup_final_benchmark.runner.log).

Secondary CI burst, **100 input / 100 output / 32 requests / no explicit
concurrency limit / max-num-seqs32**, greedy temperature0, ignore EOS:

| Metric | CI burst100/100/32 |
| --- | ---: |
| Completed requests |32/32 |
| TTFT P50 / P99 |2059.653 /2060.785ms |
| TPOT mean / P50 / P99 |75.442 /74.824 /88.483ms |
| ITL P50 / P99 |67.644 /304.182ms |
| Aggregate output throughput |337.943tokens/s |
| TPOT-derived rate, secondary only |13.255tokens/s/user |

[Normalized CI metrics](../../readiness_vllm/vllm_ci_serving_benchmark.json)
record the exact workload/command; [raw CI result](../../readiness_vllm/vllm_ci_serving_result.json).
The final B32 server's single-request diagnostic is separately archived in
[b32_startup_final_primary/](b32_startup_final_primary/vllm_benchmark.json); its fixed capacity32
is different from the headline capacity1 server.
The CI burst is serving-capacity context and is never the headline decode
rate: admission and overlapping eager prefill affect its TPOT; chunked prefill
is disabled for this model. Full-model
teacher-forcing87.115 tokens/s/user (selected predecessor, B1 AIME24 chat,
100 reference positions/99 decode replays, physical cache context2048) is a decoder cost
lower bound, not an interchangeable serving metric.

Explicit idempotent worker shutdown releases generator traces before closing
mesh/fabric. The earlier no-op worker shutdown caused repeat post-server
Ethernet-open timeouts; bounded list/reset/list plus mesh smoke recovered the
devices, and the source fix passed host regression and two serving-stop/reopen
cycles without another reset. [Shutdown investigation](AUTODEBUG_serving_shutdown.md)
and [trace-allocation audit](trace_allocation_audit.md) preserve the warnings,
controls and recovery evidence. The final B32 server exited0; the [process audit](full_b32_startup_final_process_cleanup.json)
confirms no remaining vLLM/EngineCore processes. No live-serving profiling was performed.

See [work_log.md](work_log.md) for commands, failed attempts, fixes, raw evidence
and final commit receipts. This stage establishes TP4 text serving; TP1/TP2,
prefix caching, million-token YaRN and complete long-form task accuracy are not
claimed.

Raw logs, original benchmark/qualitative JSON, and byte-sensitive generated text
are stored losslessly in
[raw_evidence.tar.gz](raw_evidence.tar.gz), with member hashes in
[the manifest](raw_evidence_manifest.json). Plain originals remain locally;
restore them in a fresh checkout before following raw-log links or rerunning
the artifact checks:

```bash
tar -xzf models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/raw_evidence.tar.gz \
  -C models/autoports/ornith_ai_ornith_1_5_9b
```
