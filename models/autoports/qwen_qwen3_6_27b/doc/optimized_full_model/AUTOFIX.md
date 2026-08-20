# Autofix report

Date: 2026-08-20 EDT

Trigger: the first independent stage review returned `more-work-needed` for
the inherited full-vocabulary greedy all-gather and unexplained readiness
performance differences. AutoDebug evidence is in `AUTODEBUG.md`.

## Greedy sampler hypotheses and experiments

| Hypothesis / candidate | Experiment | Result | Decision |
|---|---|---|---|
| Inherited force-argmax is compact | Source/profile audit | Refuted: it Ring-all-gathered all 262,144 BF16 logits before argmax | replace |
| Generic Sampling1D is an acceptable greedy choice | Full64 A/B | Correct token 225721, 10.736105 ms | reject, 7.26x slower |
| Upstream-style local max/argmax is semantically correct | Focused eager + captured trace with persistent feedback | Pass; exact host token and persistent `tt_out_tok` | keep algorithm |
| Compact async Ring all-gather is safe | Uninstrumented and safe-Watcher runs | 1.469 ms uninstrumented, Watcher writer stall | reject |
| Adding the upstream barrier fixes candidate gather | Safe-Watcher run | Same writer stall | reject |
| Synchronous Ring candidate gather is safe | Safe-Watcher run | Same writer stall | reject |
| Compact all-broadcast + concat is safe and fast | Focused Watcher, 4-layer A/B, full64 A/B | Pass; 1.478412 ms full64 | select |

Selected implementation details:

- model-scoped opt-in `distributed_force_argmax`; stochastic routing unchanged;
- local BF16 untilize, local max and argmax;
- exact one-tile candidate packing and hi/lo-byte global index reconstruction;
- candidate value `[1,1,32,128]` and index `[1,1,128,32]` tensors;
- physical-Ring `all_broadcast` of compact candidates, then concat;
- direct copy into persistent RM uint32 `[1,1,1,32]` `tt_out_tok`;
- no full-vocabulary greedy all-gather and no generic TopK.

Final full64 results are 42.014693 ms model, 1.478412 ms sampler,
43.488361 ms combined (22.994658 t/s/u), and 48.272701 ms caller-visible
(20.715642 t/s/u). The 42.127888 ms layer-stack floor leaves a 3.23% terminal
gap. Focused profiler evidence contains `AllBroadcastDeviceOperation`, local
reductions, and argmax, with no all-gather or TopK. Safe Watcher passes the
complete reduced full-model/generator gate.

## Teacher-forcing and TTFT hypotheses

The prior 19.2209 and initial current 15.6365 t/s/u artifacts used different
tt-metal runtime trees. The older raw log itself measured 16.42 t/s/u, showing
material run variance. Autonomous full64 model timings were stable, so the
source diff did not support a decoder regression.

A cold run on the final sampler measured 3.045838 t/s/u, but its log showed
first-use distributed-sampler kernels compiling after the readiness runner had
started its decode timer. This exposed a measurement flaw: teacher timing began
at the prefill token, before first trace capture/compile. The first rereview
additionally found that a warmup reset releases request-owned traces, so the
19.919590 t/s/u result still included trace capture even though kernels were
warm.

The reduced same-process boundary control measured:

- autonomous model + sampler: 5.061 ms/token;
- compact token read + callback-equivalent forced-token copy: 5.318 ms/token;
- the same path with an explicit mesh sync: 5.380 ms/token.

Thus the host boundary is real and intentionally reported separately, but it
does not explain the cold 328 ms/token result; adding synchronization is also
not a fix. `run_teacher_forcing` now supports `--warmup-repeats`, with teardown
on error and unit coverage. Its timing separates the first post-prefill trace
capture interval from subsequent replays. One full-reference warmup followed
by an identical measured AIME run gives 22.558017 t/s/u over 98 steady replay
intervals, 19.934041 t/s/u over 99 capture-inclusive intervals, 622.025 ms trace
setup, TTFT 976.068 ms, top-1 97/100, top-5 100/100, and top-100 100/100.

The canonical warmed prompt-128 autonomous TTFT is measured separately in the
same-process A/B: 679.059 ms inherited policy versus 655.160 ms selected. Cold
readiness TTFT values remain unsuitable for source comparisons because they
include model/trace setup and differ in prompt shape.

## Final disposition

Both review blockers are fixed. Greedy token-out is compact, device-resident,
fully traced, Watcher-clean, and below the gap threshold. Teacher accuracy is
refreshed and its warmed performance is controlled. Rejected protocols and the
cold control remain documented rather than being silently discarded.
