# Exact numerical serving and standalone controls

The supervising hardware lane passed both matching raw-prose controls on
2026-09-08: exact API logprob repeats/permutations, exact standalone full-vocabulary
logits across fixed device rows, and exact standalone/API numerical parity.
The authoring subagent independently inspected the saved evidence and ran only
tokenizer, AST/helper, syntax, formatting, and artifact-hash checks; it made no
requests and did not import TTNN or open devices.

The pinned tokenizer produces exactly 131 tokens for a coherent community-library
story (A), and 65 tokens for a coherent exercise paragraph (B). Both texts round-trip
through tokenization exactly. [The prepared manifest](logit_determinism_prompts.json)
contains the full source text and token IDs. They are nonaligned raw continuation
controls, not chat-format quality evaluations. Four emitted tokens provide a
bounded numerical mechanism test; the final readiness qualitative suite remains
the broader output-quality evidence.

The API script sends A twice, then `[A,B,A]`, `[B,A,A]`, and `[A,A,B]`. Every request
is greedy, seed 7, `ignore_eos=true`, with token-ID output and 20 raw logprobs.
It verifies prompt IDs/lengths, completion counts, finite values, and exact equality
of emitted token IDs, their numerical logprobs, and all alternative ID-to-logprob
maps at each emitted position. It leaves nine repeated/permuted comparisons and
all raw responses. Exact means no tolerance: even a one-ULP float change fails.
API positions are recorded as API positions; they do not independently identify
the scheduler's physical device rows. These requests explicitly select the
authorized host-logits compatibility path. They do not measure default device
sampling performance.

The existing selected-policy
[B32 baseline](../datatype_sweep/selected_batch32_watcher_v2.json) records exact
cross-row final logits and repeated/page-permuted equality with all 32 layers.
Its [actual source](../full_model/full_batch_contract.py) uses token-ID sequences
`range(131)`, `range(127)`, and `[31,57,88]`, with cache context 2048. It therefore
does not supply matching prose logits for this new API control.

The new standalone script constructs the actual selected-policy model and
low-level generator directly, requires all 32 layers and native context 262144,
and matches the API batch geometry (B32). It uses explicit diagnostic raw logits
and host greedy feedback; it does not construct the vLLM adapter or alter the
canonical sampler. The same five cases use fixed rows 0, 1, and 31, saving all
248320 vocabulary logits for the first prediction and three traced decode steps.
It checks full-vector exact equality across repeats/row permutations, exact
numerical logprob equality within standalone, and exact top-20/emitted-token
logprob parity with the saved API results. Logprobs use the pinned host sampler's
`logits.log_softmax(dim=-1, dtype=torch.float32)` definition.

The executed standalone cache has a 4096-column native page table and exactly
5120 physical blocks. `num_blocks_for_context` rounds the three blocks required
for 131+4 tokens up to a 32-block table-alignment unit. Each row therefore owns
32 consecutive 64-token blocks: row 0 maps to blocks 0–31, row 1 to 32–63, and
row 31 to 992–1023. The short requests use only the first three blocks in each
range. All row ranges are disjoint and below the pool size; page-table width is
also below the physical block count required by the paged cache primitive.
The earlier pre-execution estimate of 4192 blocks overlooked that helper's
alignment; the recorded run and this final description use the actual 5120.
Recurrent and convolution state reset between cases. Fresh prefill overwrites all
causal KV positions read by each new request; unused KV contents remain masked.

Run after the attached benchmark/sampling work has ended, against the existing
server with explicitly authorized host sampling:

```bash
USER=hous ../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/logit_determinism_vllm.py \
  --url http://127.0.0.1:8000 --model /home/hous/dev/ornith-1.5-9b/upstream --server-max-num-seqs 32 \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/logit_determinism_vllm.json
```

Then stop the server and close its device mesh before the supervising hardware
lane runs the standalone control with the same runtime/device environment:

```bash
USER=hous ../state/serving-env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.vllm_integration.logit_determinism_standalone \
  --model-path ../upstream --batch 32 \
  --vllm-result models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/logit_determinism_vllm.json \
  --raw-logits-output /home/hous/dev/ornith-1.5-9b/state/vllm_logit_determinism_standalone.pt \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/logit_determinism_standalone.json
```

The external tensor payload is 43,704,320 bytes (about 41.68 MiB), plus small
serialization metadata. Keep it in task state, not the repository. JSON records
its path, size, SHA256, per-case tensor hashes, selected precision, runtime source
hashes, exact comparisons, and cleanup completion after generator teardown and
mesh closure. Each case is saved before the final assertions, so mismatches leave
their evidence. API JSON and logs, standalone JSON and logs, and the retained
external raw tensor are complementary evidence.

CPU checks passed: tokenizer lengths/round trips; both scripts compile; the actual
signature helper accepts FP32 log-softmax output; chosen-token and alternative
logprob changes of one ULP are detected, including a permuted occurrence among
all nine comparisons. [Repository pre-commit](logit_determinism_precommit.log)
passed both scripts with exit 0. No C++/CMake changed, so no build was required.

The completed API run used the
[`full_b32_verified` server manifest](full_b32_verified.command.json): all 32
layers, selected precision, native context 262144, `max_num_seqs=32`, TP4 on
P300c, and async scheduling. That manifest records
`ORNITH_VLLM_ALLOW_HOST_SAMPLING=1`, which explicitly permits the raw-logprobs
requests' host compatibility path. No profiler or watcher was enabled. The server
model/generator hashes match those recorded by the standalone control.

| Completed check | Result |
| --- | --- |
| [API numerical repeats/permutations](logit_determinism_vllm.json) | 9/9 exact comparisons across five calls and eleven completions; exit 0 |
| [Standalone full-vocabulary repeats/rows](logit_determinism_standalone.json) | 9/9 exact comparisons over all 248320 logits at each of four positions; maximum absolute difference 0 |
| Standalone numerical logprob repeats/rows | 9/9 exact comparisons |
| Standalone versus API | 11/11 exact token-ID, chosen-logprob, and top-20 map comparisons |
| Standalone teardown | `cleanup_completed=true`; mesh closure completed 2026-09-08 14:41:15 UTC; exit 0 |

The 131-token story consistently continued with `" and smiled.\n"`, IDs
`[321,29547,13,198]`. The 65-token paragraph continued with `" and better able to"`,
IDs `[321,2577,2858,310]`. These are coherent short continuations for the supplied
prose and establish nonaligned numerical stability without synthetic repeated-ID
stimuli. The four-token scope does not replace longer qualitative evaluation.

The standalone JSON records all layers 0–31 and selected configuration
`head4_lofi_last8_c32_k4_r2`. The retained raw file is
`/home/hous/dev/ornith-1.5-9b/state/vllm_logit_determinism_standalone.pt`, exactly
43,707,383 bytes, SHA256
`b558ff9ffc49e9231d73cabd153bd394d1bed3870dceb5a20f803369e1df0480`.
An independent read-only check matched its size and hash and matched all recorded
script/model/generator/precision source hashes to the files used for review.
The [API log](logit_determinism_vllm.log) and
[standalone log](logit_determinism_standalone.log) preserve the emitted outputs,
comparison summaries, and device-close evidence. These results prove raw-logit
and raw-logprob determinism in the stated controls; default device token-output
sampling, async ownership, and throughput retain their separate evidence gates.
