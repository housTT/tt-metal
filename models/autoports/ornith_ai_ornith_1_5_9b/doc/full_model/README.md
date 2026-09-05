# Ornith-1.5-9B full model

**Full-model warm TTFT:47.46 ms. Trace-verified batch-1 token-out decode:
81.56 tokens/s/user (12.26 ms/token).** All32 layers,prompt128/generate128,
native262144 cache, TP4 ring on **four Blackhole chips on two P300c boards**.
`p150x4` names the software profile, not P150 hardware. All required execution gates pass; independent [stage review](STAGE_REVIEW.md) returned **clean-pass**.

## Implementation and preserved policy

[model.py](../../tt/model.py) implements the HF text path: hidden-sharded BF16
embedding, all24 linear and8 full-attention layers, HF final norm (weight+1),
untied vocabulary-sharded LM head, and explicit hybrid/paged caches.
[generator.py](../../tt/generator.py) implements the standard Metal readiness
`Generator` and `build_generator(model_dir, mesh_device, **kwargs)` contract.
The pinned reference is `ornith-ai/Ornith-1.5-9B` revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.

The selected optimized multichip decoder and its
[rejection ledger](../optimized_multichip_decoder/optimization_evidence.md)
remain the starting policy: native1x4 ring, TP4 weights/heads, BFP4/LoFi
projections, BFP8/LoFi decode QKVG, packed GDN and packed MLP gate/up, BF16
activation/CCL, BFP8 paged KV with BF16 updates, and FP32 recurrent state.
The selected replicated hidden4096 residual passes directly from layer to layer:
B1 decode L1 width-sharded on32 cores8x4, B2..32 decode and prefill DRAM
interleaved. There is no inter-layer gather or reshard added by the wrapper.

The selected LM head uses BF16/HiFi4 with FP32 destination accumulation,
two32768-column chunks per rank, Kblock1 and2 readers/bank. It is the fastest
measured candidate that passes the French qualitative regression:1.404 ms
terminal-only versus1.454 ms for16384/K4/two readers and1.859 ms for the original
8192/K4/one reader. Larger K2/K4 at32768 columns exceed measured full-stack L1
limits. Faster BFP4/BFP8 candidates pass aggregate top-k but produce an incorrect
French register label. BF16 chooses a different earlier free-running branch and
correctly labels the original prompt; frozen-hidden controls refute a local
terminal-kernel defect. See [AUTOFIX_french_head.md](AUTOFIX_french_head.md).
Large2048-token GDN prefill caps its
output block at6 to fit all24 persistent recurrent states in L1 while preserving
K16 accumulation and bitwise projection output; prior decoder defaults are
unchanged. See [AUTOFIX_context_l1.md](AUTOFIX_context_l1.md).

## Accuracy and generation

| All-layer gate | Top-1 | Top-5 | Top-100 | Evidence |
|---|---:|---:|---:|---|
| Prefill,100 scored predictions | 96% | 100% | 100% | [prefill_final_v2.json](prefill_final_v2.json) |
| Traced teacher forcing,100 predictions | 94% | 100% | 100% | [teacher_final_v2.json](teacher_final_v2.json) |

Both gates use the fresh [AIME24 reference](../../readiness_aime24_chat.meta.json):
exact HF tokenizer chat template,161 prompt tokens,100 generated tokens,K100.
The pinned CPU HF model loaded427/427 keys without missing or random weights.
Its cached-generation versus full-prefill control scores99/100/100%; the single
rank1 difference is rank2. Reference text is in
[aime24_hf_completion.txt](aime24_hf_completion.txt).

The shared six-prompt suite runs standard `run_autoregressive` for128 tokens per
prompt. [Final artifacts](qualitative_final_v2/qualitative_prompt_format.json)
retain HF and TT completions, token IDs, exact rendered prompts and tokenizer
hashes. The HF controls are reused only after exact revision/template/suite/token
and length metadata checks. All twelve final HF/TT outputs were read: coherent task-related reasoning, no
mechanical degeneration or unexplained language drift, and the French label is
corrected. The128-token windows mostly end during reasoning, as do the HF
controls; completed poem, story and code quality are not claimed. Final
classification and degeneracy results are in the qualitative directory.

## Sampling and trace contract

Both common sampler contracts were inspected before token-out design.
`models/common/sampling/SamplingGenerator` is selected because it owns parameter,
penalty, seed and output state and supports `tt_out_tok`. `Sampling1D` supports
per-call tensors and candidate gathering, but requires the generator to rebuild
those state-management contracts. Its rejection is a contract-fit decision,
not an unmeasured speed claim. No custom sampling algorithm was added.

A model trace produces local sampler-ready BF16 logits and advances position and
RoPE state on device. A separate generator-owned trace invokes the common
sampler, writes the persistent decode input through `tt_out_tok`, and advances
seed counters. This also traces the common sampler's explicit seeded mode,
whose internal trace switch is disabled. Parameters change at scheduling
boundaries with trace recapture as needed. Page tables copy only on change.

The [greedy comparison](sampler_greedy_comparison.json) uses semantic k1/p0/temp1
for both paths, despite the physical top32 tile. Both exactly match CPU greedy
for32 distinct rows. Canonical split sampling measures0.574 ms; force-argmax
2.748 ms,4.79x slower. Canonical sampling is the default. The selected path also
supports the common top-k/top-p, seed and penalty controls.

The optimized127-step decode window records127 model and127 sampler replays,
zero token/position/RoPE/page-table host refreshes,127 output reads/event waits,
and zero global device synchronizations. Reads are pipelined behind the next
queued replay and never become token feedback. See
[perf_final_v2.json](perf_final_v2.json),
[trace_b32_masks_fixed.json](trace_b32_masks_fixed.json) and
[scheduler_sampling_final_v3.json](scheduler_sampling_final_v3.json).

## Public state and context

The generator exposes `prefill_forward(tokens, *, page_table, kv_cache,
prompt_lens, slots, start_pos, ...)` and `decode_forward(tokens, start_pos, *,
page_table, kv_cache, ...)`. Passing `None` for decode tokens/positions uses
persistent on-device feedback. The caller can own cache/page tables and bind
them before capture. Mixed prompts, fixed slots, inactive rows, continuation,
partial prefill and new-slot joining use the same model path. Live sampling
configuration preserves existing cache/token/position state.

`generate(..., enable_trace=True)` owns reset, padding, masks, positions, cache
fill, sampling and output slicing. Logical lengths need no tile/page/chunk
alignment. `sampling_mode="host"` explicitly supports host greedy or a
`host_sample` callback; it is separate from measured on-device token-out.
Seed lists are lane-scoped; a scalar seeds lane0 and omitted seeds use independent
request/lane entropy. New-slot prefill resets only that request's RNG, while
continuations and ongoing slots preserve it. No per-token host RNG update is
introduced. See [AUTOFIX_request_seeds.md](AUTOFIX_request_seeds.md).

EOS slices returned output after the fixed trace window; compute is not reclaimed
early. A future scheduler can inactivate finished rows at explicit boundaries.

The [context contract](../context_contract.json) preserves262144 tokens.
[Native all-layer execution](native_context_final_v3.json) passes262143 and262144
prefills in43.921 and43.844 s, and decode at262143 advances to262144. Public
prefill chunks the full stack at2048 and does not allocate a full-context hidden
activation. Post-prefill DRAM is5,445,806,592 bytes/device against34,078,723,072
allocator bytes;13,369,344 trace bytes reside in the separately reserved100 MB
region. Formula accounting includes all loaded weights,8 KV layers,24 recurrent
states, RoPE, page tables and measured persistent allocations. This is capacity
and position evidence, not a native-length HF quality claim. Batch32 is tested
at shorter contexts;32 simultaneous native-length caches exceed physical DRAM.
Optional million-token YaRN is not advertised by this full-model stage.

## Performance interpretation

| Path | Workload | Result |
|---|---|---:|
| Warm request-inclusive TTFT | B1,prompt128,generate128 | 47.46 ms |
| Prefill/sample/read portion | same request | 43.08 ms |
| Request reset/configuration | same request | 4.38 ms |
| Token-out traced decode |127 feedback steps |81.56 t/s/u |
| Logits-only model trace |128 steps,fixed token,device positions |85.61 t/s |
| Readiness teacher forcing |AIME161+100,explicit host override |80.85 t/s/u |

The first generation call reports463.11 ms TTFT including program warmup and
trace capture; weight loading is outside TTFT. The logits-only number has no
sampler or autoregressive token feedback and is not the token-out headline.
Repeated synthetic token100 prompts are a performance workload, not qualitative
language evidence. At prompt2048, warm TTFT is113.11 ms and token-out decode is80.77 t/s/u
(12.381586 ms/token); logits-only decode is11.798601 ms/token. The prior standalone
layer-sum estimate is24×0.355672+8×0.268965=10.687848 ms. The full model adds
1.110753 ms for logits-only or1.693738 ms for token-out relative to that sum.
This is not a strict additive lower bound: standalone traces start from DRAM
inputs, while the full stack directly reuses each layer's selected sharded
residual and shares trace/dispatch boundaries. See
[performance_accounting.json](performance_accounting.json). [Reduced profiling](tracy/README.md) measures one real layer of each kind plus
the selected terminal path. Traced sampler kernels plus gaps total about0.58 ms,
under5% of full-model token-out latency, and do not dominate decode. Runtime rows
confirm the selected decoder dtype/fidelity and BF16/HiFi4 head. All measured
rows are preserved; the profiler's unused-capture parser defect and theoretical
worker-count display limitation are separately documented.

## Integrity, reproduction and review

[Runtime audit](runtime_audit.md) enumerates cache ownership, reset, sampling,
host-logit boundaries and optimized execution. Worker watcher and trace
allocation checks run separately from profiling. The NoC split-reader defect,
BF16/FP32 state-mask defects and scheduler UINT32 selection defect have focused
before/after controls in the AutoFix reports. Canonical RMSNorm row ordering and the selected BF16 head pass the final
all32-layer batch32 test with exact duplicate/permuted full-vocabulary logits,
worker watcher and allocation tracking. See
[full_batch32_final_v2.json](full_batch32_final_v2.json) and
[AUTOFIX_norm_row_order.md](AUTOFIX_norm_row_order.md).

Commands and immutable source/library hashes live in `logs/*.provenance.json`,
with source snapshots alongside them. The [artifact manifest](artifact_manifest.json)
records hashes, sizes and whether each artifact is committed or retained locally.
All Python commands use the pinned local
snapshot and these environment settings:

```bash
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run UNIQUE_LABEL python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.run_checks teacher --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/UNIQUE_LABEL.json
```

Use modes `prefill`, `teacher` and `perf` for the all-layer gates; `smoke` loads
one real layer of each kind for diagnosis. Never profile the all-layer stack.

The required `.github/scripts/copilot-build.sh --build-ttnn-tests` build was
attempted but Docker is unavailable on this host. Changed device kernels are
JIT-compiled and exercised on hardware; the complete CI wrapper build remains
**unverified**. See the build logs and [work log](work_log.md). All46 host readiness/generator tests and10 Tracy parser tests pass; one
pre-existing Tracy mock test remains skipped. Two generic native RMSNorm tests
pass under worker watcher. Source formatting/lint hooks pass. Exact generated
text/metadata bytes are preserved without whitespace normalization. Independent
stage review and local checkpoint SHAs are recorded in the work log.
