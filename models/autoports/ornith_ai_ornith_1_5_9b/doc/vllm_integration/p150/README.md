# Ornith-1.5-9B vLLM integration on p150 (one Blackhole chip)

Primary single-user **128 input / 128 output / 1 request / concurrency 1 /
max-num-seqs 1**, greedy temperature 0, ignore EOS, native 262144 context, first
request after readiness: **TTFT 83.3 ms; decode 29.34 tokens/s/user** from
mean TPOT 34.09 ms; ITL median 34.08 ms, P99 34.89 ms; aggregate output 29.01
tokens/s (`readiness_vllm/vllm_benchmark.json`, raw `vllm_result.json` with all
127 intervals; repeat 77.4 ms / 34.10 ms in `b1_primary/repeat/`). Sampling:
73 passed, 1 controlled failure, 4 skips (full profile, B32). Qualitative:
twelve coherent on-topic texts. Mixed-shape trace reuse: zero captures on the
tracked server. Prefix caching: documented `recurrent_state` exclusion.
Batched prefill: burst recorded against its serial reference (advisory).

## Serving configuration

All 32 layers on one Blackhole chip (readiness mesh label `N150`, model card
profile `p150`), precision policy `doc/datatype_sweep/p150_head8_hifi2_kv8_policy.json`
(BFP8 attention, MLP and LM head, HiFi2 attention and gate/up, LoFi down, BFP8
KV cache, FP32 recurrent state, BF16 activations and residual, no CCL). The
adapter resolves and logs this policy at startup (`Serving resolved precision`
in every `server.log` under this directory) and raises if the constructed model
differs. Served `max_model_len=262144` matches `doc/context_contract.json`; the
vLLM-owned pool is 4129 blocks of 64 tokens (`ORNITH_VLLM_POOL_TOKENS=262208`),
batch 32 (`--max-num-seqs 32`) for the sampling, qualitative, capacity and
burst evidence and batch 1 for the primary single-user benchmark.

Stack: upstream vLLM 0.26.0 plus `tenstorrent/vllm-tt-plugin` at local HEAD
`9bab7d527d4adbece91de7338302c99e652fd642` (editable install; the one local
commit is recorded in `../plugin_changes.md`). Registration: `EXTRA_MODELS_DIR`
bundle `vllm_bundle/ornith-1-5-9b` with `TT_VLLM_BUILTIN_MODELS=0`; the resolved
class is this checkout's `tt/generator_vllm.py:TTOrnithForCausalLM` in the API
server and in `EngineCore` (`registration_probe.json`, `Registered TT model`
lines in `server.log`). Launch wrapper: `serve.sh` (sources `p150_env.sh`):

```bash
bash doc/vllm_integration/p150/serve.sh <label> <max_num_seqs> <chip>
# expands to
python -m readiness_check.run_vllm_server --stages serve \
  --model-dir models/autoports/ornith_ai_ornith_1_5_9b --hf-model ornith-ai/Ornith-1.5-9B \
  --mesh-device N150 --max-num-seqs <N> --max-model-len 262144 --block-size 64 \
  --no-enable-prefix-caching --tt-config '{"trace_region_size": 100000000, "l1_small_size": 32768}' \
  --additional-server-args "--hf-overrides '{\"architectures\": [\"TTOrnithForCausalLM\"]}' --async-scheduling"
```

TT config: `sample_on_device_mode=all` (runner default), trace region 100 MB,
L1 small 32 KiB, no fabric config (one chip). `HOST_SAMPLING=1` sets
`ORNITH_VLLM_ALLOW_HOST_SAMPLING=1`, the explicit compatibility mode the
canonical sampling suite needs for logprobs and host-only parameters; the
primary benchmark server runs with it off. `TRACKED=1` sets
`TT_METAL_TRACE_ALLOC_TRACKING=1` in the server environment.

## Adapter, cache ownership and capabilities

`tt/generator_vllm.py` delegates prefill, traced decode replay, sampling
configuration, slot remaps and deferred readback to `tt/generator.py`; the
generator receives the exact cache from `allocate_kv_cache` (vLLM-owned pool,
`gen.owns_cache == False`), and the constant-size recurrent and convolution
state of the 24 linear-attention layers stays model-owned. Declared
`model_capabilities` (every key is in the pinned plugin's capability table):
`supports_prefix_caching=False` (`recurrent_state` exclusion, see below),
`supports_async_decode=True`, `supports_sample_on_device=True`,
`supports_masked_prefill_sampling=True`, `sampled_raw_logprobs_devices=(4,)`
(device logprobs only on the four-chip mesh), `supports_host_sampling` =
compatibility flag. The undocumented `supports_batched_prefill` key was removed.

`supports_async_decode=True` on one chip is backed by
`adapter_device_tracker_p150.json` (tracked, reduced layers 0 and 3, pool of
one native request): stale host tokens and positions are ignored on steady
steps, two deferred steps are submitted before the first wait, a page change
across the 64-token boundary writes the new physical page in every K/V tensor
and leaves the old page untouched, slot remaps are exact, persistent buffer
addresses are unchanged, and the synchronous baseline reproduces the deferred
outputs. The full-model serving runs below keep `--async-scheduling` on.

## Prefix caching: documented exclusion

`supports_prefix_caching` is `not_supported` in `doc/full_model/serving_contract.json`
with reason `recurrent_state`: 24 of 32 layers are Gated DeltaNet
linear-attention layers whose recurrent and 3-token convolution state the
vLLM-owned KV blocks cannot rebuild at `start_pos`; a scheduler cache hit would
resume the 8 full-attention layers from cached KV and the 24 recurrent layers
from zero state. Servers run with `--no-enable-prefix-caching`
(`vllm_run_summary.json`, "Automatic prefix caching is disabled" in `server.log`).
The model-owned hybrid snapshot cache (`doc/prefix_cache/README.md`) stays off
(`ORNITH_PREFIX_CACHE_BYTES=0`).

## Trace reuse and request lifecycle

Measured first on this profile (`prefill_shape_probe_v1.json`): every unseen
prompt length compiled 8 to 50 programs and forced a decode and sampler trace
recapture (TTFT 1.0 to 7.7 s versus 0.1 s warm). Three serving-only sources
were found afterwards: per-slot admission programs, per-source-slot programs
in scheduler slot compaction, and a sampler trace key built from per-row
sampling values. All four are fixed in the model, generator and adapter (see
`work_log.md`); the mixed-shape check now passes on the tracked server:

- `tracked_b32/mixed_shape_check.json`: 41 requests (unseen lengths 129, 130,
  131, 64, 200, 2305, 1, 2, 5, 1000, 1024, 1025; six arrivals of new lengths
  while a 192-token decode runs; repeated lengths in two batch orders; sampled
  requests then greedy repeats; a 4353-token prompt crossing the 2048-token
  chunk twice beside two decoding requests; final singles), 41 complete, greedy
  repeats identical, zero trace captures and zero releases during the
  workload, with `TT_METAL_TRACE_ALLOC_TRACKING=1` in the server environment.
- `tracked_b32/trace_reuse_summary.json`: warmup bucket list and program
  counts, trace ids captured before the workload (model `MeshTraceId(5)`,
  sampler `6`, history `7`, plus the prepared penalty-variant pair), request
  signatures per physical bucket and chunk start, every eager prefill with its
  program count before and after, and the two startup captures.
- the same workload with tracking off, for timing: `b32_final/mixed_shape_check.json`
  41/41 complete, TTFT p50/p95/p99 0.381/1.080/1.082 s (first encounters of
  2305 and 4353 tokens are the tail), ITL max 0.449 s, zero captures;
  `b32_compat_prefinal/mixed_shape_check.json` 0.381/1.083/1.084 s, ITL max
  0.450 s; `b32_recheck/mixed_shape_check.json` and
  `b32_recheck2/mixed_shape_check.json` 0.382 s p50, zero captures.
- `prefill_shape_probe_v8_tracked_committed.json`: the adapter-level tracked
  probe (`TT_METAL_TRACE_ALLOC_TRACKING=1`, program-cache allocations
  included) on the committed sources (`git_head` and `source_sha256` of
  `tt/*.py` recorded in the file): zero new programs and zero captures for
  every length and across the greedy, sampled and penalized switches.

Startup: the servers recorded in `tracked_b32/`, `b32_final/` and
`b1_primary/` capture once, then the warmup's dummy admission compiled the
host-to-device merge programs (one program on the B32 servers, eight on the B1
server) and the family was released and recaptured before `Application
startup complete`. The second follow-up commit warms the host-to-device
merge before the first capture: the `b32_recheck2` server (committed sources,
`b32_recheck2/server.log`, `launch_env.txt` with source hashes) captures once
at 1122 programs with no later growth, release or recapture, and serves the
non-aligned and mixed-shape workloads with zero captures
(`b32_recheck2/trace_reuse_summary.json`: one startup capture, 38 eager
prefills, zero workload captures or releases). In every earlier run the
recapture happened before `Application startup complete` and never after.

Residual: the generator accepts a continuation at a start position that is
not a multiple of 128 through its legacy path, which compiles per-position
programs. Under the pinned plugin with chunked prefill and prefix caching off
every served prefill starts at position 0 (a preempted request is re-prefilled
with its prompt plus accepted tokens, `docs/DECODE_RELOAD_CONTRACT.md`), so
that path is unreachable from vLLM on this profile; preemption itself did not
occur in any served run (`grep -ci preempt server.log` is 0) and is untested.

`ORNITH_ACTIVE8_MODE=active8` is set in `p150_env.sh` because the model card
sets it for every profile; `tt/active8.py::eligible()` requires the four-chip
mesh, so every p150 server logs `effective_mode: original` and the decode
variant is the original 32-row graph.

## Non-aligned prompt lengths

`tracked_b32/nonaligned_requests.json` (`nonaligned_requests.py`): raw
completions of 131, 65, 2305, 129 and 33 tokens singly, as one prompt list
`[A, B, A]`, and as six concurrent requests, all with exact `prompt_tokens`
usage, 12 in-vocabulary token ids each and identical greedy ids for repeated
prompts; plus a chat request whose rendered prompt is 25 tokens. The
mixed-shape workload adds 1, 2, 5, 130, 200, 1000, 1024, 1025, 257 and 4353.

## Logit determinism through vLLM

`tracked_b32/logit_determinism_vllm.json` (host-compatibility logprobs, greedy,
seed 7): prompt A (131 tokens) twice and `[A,B,A]`, `[B,A,A]`, `[A,A,B]`; all
nine repeated and permuted comparisons of emitted token ids, chosen logprobs
and top-20 maps are exact. `logit_determinism_standalone.json`: the standalone
all-layer generator on one chip with the p150 policy reproduces the served
tokens and logprobs for the same prompts (`passed: true`, device rows 0, 1 and
31, raw logits artifact `/tmp/ornith_p150_logit_determinism_standalone.pt`).

## Sampling status

Canonical plugin suite, full profile, against the B32 compatibility server
(`readiness_vllm/sampling_tests.log`, copies in `b32_final/` and
`b32_compat_prefinal/`): **73 passed, 1 failed, 4 skipped** in 662 s. Skips:
the three chunked-prefill needle tests (`supports_chunked_prefill` is not
declared) and `test_chat_logprobs_all_vocab` (all-vocabulary logprobs are
unsupported). The failure, `test_tt_penalties::test_frequency_penalty_changes_output`,
expects `frequency_penalty=2.0` to change the greedy output of all eight
prompts; seven change and `After a while, she` does not. The control
`b32_compat_prefinal/penalty_margin_probe.json` shows that the device output
is the penalized argmax of the raw top-20 logits at every step: the repeated
tokens ` to` (steps 9 and 12, counts 1 and 2, penalized by 2.0 and 4.0 nats)
and ` that` (step 23) remain 3.25 and 7.6 nats ahead of the runner-up, and
step 13 is an exact raw tie. The penalty reaches the device sampler; the
output is legitimately unchanged for this prompt under the p150 policy, and
the API rejects penalties above 2.0, so no stronger stimulus exists. The
plugin test was not modified; the disposition is recorded as a `proposed`
entry in `../plugin_changes.md`. Logprobs, host-only parameters and structured
outputs run through the explicit host-sampling compatibility mode; every
other request uses the device sampler (`sample_on_device_mode=all`).

## Qualitative verdict

`readiness_vllm/vllm_qualitative_outputs.json` (copy in `b32_final/`): six
chat prompts rendered with the checkpoint's chat template (thinking on by
default; prompt-format metadata and HF controls from the p150x4 stage in
`../qualitative_prompt_format.json` and `../qualitative_controls.json` apply
unchanged), greedy and sampled, 256 tokens each. All twelve texts were read:
coherent, on-topic English reasoning in the checkpoint's thinking format, no
mechanical repetition (most frequent bigram "Let me", at most 6 of 144 words),
no gibberish, no wrong-language drift (0 CJK characters; French appears only
as the requested translation, `Bonjour, comment allez-vous aujourd'hui ?`), no
cross-request contamination. Thinking consumes most of the budget: the story,
the sampled French answer and the greedy Fibonacci answer reach the final
answer; the others end inside `<think>`. The greedy haiku reasoning
miscounts syllables ("Pat-tens em-erge = 5", then self-corrected); the HF
control of the p150x4 stage (`../qualitative_extended_control_report.md`,
line 129) counts correctly, so this is a precision-policy or greedy
variance effect of the port, not a serving effect: served tokens and logprobs
equal the standalone one-chip generator exactly on the logit-determinism
prompts. The mechanical degeneracy gate passes (stage check).

## Benchmarks

Primary **128 / 128 / 1, concurrency 1, max-num-seqs 1**, greedy, ignore EOS,
B1 server with host sampling off (`b1_primary/`):

| Metric | Primary 128/128/1 |
| --- | ---: |
| Completed requests | 1/1 |
| TTFT P50 / P99 | 83.30 / 83.30 ms |
| TPOT mean / P99 | 34.09 / 34.09 ms |
| ITL P50 / P99 | 34.08 / 34.89 ms |
| Aggregate output throughput | 29.01 tokens/s |
| TPOT-derived decode rate | 29.34 tokens/s/user |

Secondary CI serving burst **100 / 100 / 32, no concurrency limit, max-num-seqs
32**, greedy, ignore EOS, B32 server (`readiness_vllm/vllm_ci_serving_benchmark.json`):

| Metric | CI burst 100/100/32 |
| --- | ---: |
| Completed requests | 32/32 |
| TTFT P50 / P99 | 3778.4 / 3779.8 ms |
| TPOT mean / P99 | 171.26 / 178.26 ms |
| ITL P50 / P99 | 168.59 / 245.33 ms |
| Aggregate output throughput | 154.57 tokens/s |
| TPOT-derived rate, secondary only | 5.84 tokens/s/user |

Prefill burst **32 x 128 input / 32 output, concurrency 32**
(`readiness_vllm/vllm_prefill_burst_benchmark.json`): 32/32, TTFT p50/p99/mean
3758.7/3760.0/3726.0 ms; prefill is one row per pass on this profile, so the
serial reference (`vllm_prefill_burst_serial_benchmark.json`, same workload,
`config_matches: true`) is the same path and the ratios are 1.00 (advisory in
stage 9; batched prefill is stage 10 work).

Comparison with the full-model teacher-forcing lower bound: the 32-row decode
graph costs 168 ms per step on one chip (4.4 ms per row), so the B32 numbers
are capacity context only; the B1 path replays the model, sampler and
persistent token feedback traces with no host sampling, no per-token
page-table copy and no position refresh in steady state, as the tracked
device probe `adapter_device_tracker_p150.json` shows (token, position and
RoPE refresh deltas 0 on steady steps, page-table refresh only on the changed
page). The adapter's teardown counters are not logged: the runner's SIGTERM
makes vLLM force-kill `EngineCore` before `teardown()` runs
(`[shutdown] Process manager: force killing remaining processes` in every
server log).

One-chip teacher-forcing lower bound (`teacher_forcing_p150.json`,
`doc/datatype_sweep/run_candidate.py --num-devices 1 --config
p150_head8_hifi2_kv8_policy.json --teacher-only --teacher-repeats 3`, all 32
layers, batch 1, AIME24 chat 100-token reference, traced decode with logit
readback, median of three repeats, measured on one chip of a P300c board,
not a P150 board): top-1 0.98, top-5 1.00, decode 29.18 t/s/u (repeats
29.29, 29.18, 29.09), TTFT 81.2 ms warmed. The bound reads logits back every
step and uses a 2048-token cache context, so it cannot resolve sub-millisecond
serving overhead. The packaged runner (`readiness_check.run_teacher_forcing
--mesh-device N150`, `teacher_forcing_packaged_runner.log`, after the
generator's imports were made absolute) reproduces the bound: top-1 0.98,
top-5 1.00, decode 29.38 t/s/u; its TTFT of 956 ms is that process's first,
unwarmed prefill, not a warmed figure. The served primary
B1 path measures 29.34 t/s/u and 83.3 ms TTFT on the first request after
readiness: decode is within 1 percent of the traced generator bound and TTFT
within 2.1 ms, so no avoidable vLLM-specific decode overhead remains in the
measured path (one model replay, one sampler replay and one deferred token
read per step; the B32 server's 168 ms per step is the 32-row decode graph,
the same in the generator and under vLLM). The historical p150x4 generator
bound (87.1 t/s/u at TP4) does not apply to one chip.

## Artifact provenance under readiness_vllm/

`vllm_benchmark.json`, `vllm_result.json`, `vllm_benchmark.log`,
`vllm_run_summary.json` and `server.log` are the B1 primary server
(`b1_primary/`); `sampling_tests.log`, `vllm_qualitative_outputs.json`,
`vllm_ci_serving_benchmark.json`, `vllm_ci_serving_result.json`,
`vllm_prefill_burst_benchmark.json`, `vllm_prefill_burst_result.json` and
`vllm_prefill_burst_serial_benchmark.json` are the final B32 compatibility
server (`b32_final/`). Every server directory under this folder keeps its own
`server.log` and `launch_env.txt`; the `vllm_run_summary.json` copies are
written by the attached runner stages, which merge into the existing file and
keep its `server_command`, so `b32_final/vllm_run_summary.json` still carries
the B1 command of the earlier server and the B1 summary's `tt_config` is the
attached benchmark's default; the server command and TT config of each server
are in its `server.log` and `launch_env.txt`. The plugin's capability table is
`AGENTS.md` section 6 of the pinned checkout (it has no `docs/MODEL_INTERFACE.md`).

## Process cleanup

Every server was stopped through the runner (SIGTERM); after each stop
`pgrep -af 'vllm.entrypoints|EngineCore|run_vllm_server'` returned nothing
(observed in the chained command outputs, not archived; each server
directory's `runner.log` ends with the runner's `Run summary:` line, and the
audit was repeated before each review). Chips 0 and 1 were never touched;
chip 2 served and chip 3 ran the generator-level probes.
