# Optimized-full-model runtime fallback audit

Verdict: **clean for the measured P150x4 split-token-out path**.

The selected path is the resident 36-layer `MultichipDecoder`, final norm,
TP-sharded LM head, canonical `SamplingGenerator`, persistent decode state, and
split model/sampler trace replay.  It does not select a reduced model, host
sampler, replicated full-vocabulary stream, force-argmax candidate, rejected
decoder policy, or vLLM adapter.

| Boundary | Final behavior | Evidence |
| --- | --- | --- |
| Model and residual | 36 optimized TP4 layers; logical BF16 residual replicated across ranks, decode L1 and prefill DRAM; no inter-layer collective | full resident acceptance plus inherited optimized-multichip policy ledger |
| Cache/page state | Generator-owned paged BFP8 local-head KV, physical 128-token K-chunk rounding, 64-token pages, changed-only table refresh | prompt-8/output-122 boundary, prompt-214 non-aligned gate, and trace counters |
| Explicit serving state | Cache, page table, token position, RoPE position, prompt length, batch, fixed slots, and inactive `-1` rows remain explicit | `mixed_prompt_fixed_slots_inactive_row.json` and generator API |
| Decode replay | Persistent token/position/RoPE/page-table tensors; nonblocking model trace submissions | prompt-128/output-128: 127 submissions, one setup refresh, zero steady refreshes |
| Sampling replay | Canonical vocab-sharded split sampler with `tt_out_tok` feedback | 126 fixed greedy sampling replays; sampler trace ID and output identity evidence |
| Greedy | Device top-k=1/top-p=0 semantic argmax | split sampler selected at 154.2817 t/s/u versus 139.1412 for force-argmax |
| Top-k/top-p | Same traced split contract with temperature 0.8, top-k 20, top-p 0.9 | `top_k_top_p_trace_contract.json`: four model/sampler replays, two boundary reads, no logits reads |
| Position/RoPE | Advanced on device inside the model trace | zero steady position/RoPE host refreshes and exact persistent state in four length-isolation endpoints |
| Page table | Uploaded once, reused while unchanged, refreshed only on content change | 126 reuses and zero steady page refreshes in the measured run |
| Logits | Sampler-ready TP shards remain on device | zero full-logit synchronizations/readbacks and no full-vocabulary all-gather in the measured path |
| Token collection | Caller reads first and final scalar tokens only | exactly two readbacks/synchronizations over 128 output tokens |
| Watcher | Fully enabled separately from profiling | `watcher/final_source/watcher_final_both_headers.log.gz` passed with disabled features `None` |
| Profiler | Device op graph for prefill, teacher forcing, split token-out, and LM-head A/B | five final-source raw CSV/table/CSV-report phases; no CPU/fallback op appears |

## Host-path source audit

The source scan covered `tt/generator.py`, `tt/model.py`, and the shared GPT-OSS
model.  Host conversions are confined to explicit boundaries:

- page-table comparison/setup before replay;
- `read_decode_output`, which implements caller-visible scalar collection;
- full-logit validation helpers and `sampling_mode="host"` compatibility;
- teacher-forcing token injection and qualitative/autoregressive collection.

The two `torch.argmax` calls in `Generator.generate` are guarded by
`sampling_mode == "host"`.  The measured split API always passes device mode
and never reaches them; its evidence counter records zero host argmax calls.
The shared sampler's regular `ttnn.all_gather` fallback name denotes an on-device
CCL implementation, not a CPU fallback, and it gathers local candidates rather
than the full vocabulary.  The final profiler has four small all-gather rows
(33.232 us total) and no full-vocabulary logits tensor.

## Measured trace boundary

For warmed prompt 128 / output 128, the final artifact records:

- 127 model trace replays and 127 device token-out submissions;
- one token, position/RoPE, page-table, and sampling-state setup refresh;
- zero steady token, position/RoPE, and page-table host refreshes;
- 126 unchanged page-table reuses and 126 fixed greedy sampling replays;
- two scalar token reads/collections, zero full-logit reads, and zero host
  argmax calls.

The prompt-214/output-100 like-for-like benchmark uses the same split API and
reports 62.7665 t/s/u.  The 100-token autoregressive quality run is intentionally
reported separately: its caller collects every generated token so it can render
text and stop semantically, and it is not the token-out performance path.

## Recovery and limitations

An earlier exact-replay failure was recovered only by a bounded physical device
reset.  The unchanged exact batch-2 gate then passed all rows/runs bitwise, and
the final 36-layer acceptance passed repeated synchronous runs, the pre-unseen-
prefill control, and split endpoints 7, 8, 122, and 128.  This is external
device-state recovery, not a runtime fallback.

P150 and P150x2 cannot host the fixed resident model state in 32 GiB/device;
their supported resident full-model context is therefore zero.  P150x4 retains
the advertised 131072-token batch-1 context and non-aligned prompts.  vLLM and
the broad datatype frontier remain out of scope.
