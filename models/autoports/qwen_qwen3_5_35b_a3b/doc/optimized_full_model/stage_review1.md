# Stage Review

Verdict: more-work-needed

Stage: 7, optimized full model, first review. Reviewer: independent stage-review subagent, read-only.
Review date: 2026 Oct 15 (02:53 Eastern).
Source under review: live worktree on branch `feat/qwen3-5-35b-a3b`, HEAD `fb8ce438` (stage 6 checkpoint) plus
uncommitted stage 7 changes. I recomputed the chain `manifest()` over `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/` and `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/`. It is equal to
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/source_manifest_end.txt`.
It is not equal to the chain 1 manifests (see Required Work 8).

Abbreviations: B (decode batch, active rows in one decode step), TTFT (time to first token), t/s/u (tokens per
second per user), HF (Hugging Face CPU reference), TT (Tenstorrent TTNN implementation), KV (key/value cache), GDN
(Gated DeltaNet linear-attention layer), MoE (mixture of experts), EP (expert parallel), TP (tensor parallel), LM
head (final vocabulary projection), BF16 (16-bit brain float), BFP4/BFP8 (4/8-bit block float), DRAM (device
memory), L1 (core SRAM), PCC (Pearson correlation coefficient), KL (Kullback-Leibler divergence), PRNG
(pseudo-random number generator), CPU (host processor), CCL (collective communication library), SDPA (scaled
dot-product attention), ISL (input sequence length). P1 means a correctness or contract gap that blocks the stage.
P2 means required work with lower correctness risk. p150, p150x2 and p150x4 are the 1, 2 and 4 chip profiles.


## Required Work

- P1: Decode buckets 8 and 16 have no correctness, tracker or runtime-audit evidence
  Evidence: The README headline matrix and `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json` (`decode_buckets` "4,8,16,32") advertise
  B8 and B16. Only `run_batch_perf` replays them (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/m*/batch_perf_sampled{0,1}.json`),
  with `TT_METAL_TRACE_ALLOC_TRACKING` unset, no token or logit check and no `audit=True`. Every correctness runner
  builds `decode_batches=(1, 4, 32)`: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_inactive_rows.py:29`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_sampling_audit.py:63`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_batched_decode.py:37`, and `run_full_evidence` records `batches` "1,4,32"
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/m4/full_evidence.json`). B16 is a distinct path: `_PackedGateMoE.forward`
  pads 16 rows to 32, runs the prefill dispatch, then slices (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:880-886`). Any
  `decode_rows` check reached inside that dispatch (`_DecodeNorm` at `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:1045`, `_Projection` at
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py:1364` and `:1372`) returns false for 32 rows at B16 and true at B32, so the two buckets can take
  different program configs inside the MoE. B8 is the largest row-loop bucket
  (`moe_row_loop_max` 8). The B32 HF calibration also uses 32 copies of one prompt
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_calibration_probe.py`, README "every row the same prompt"), so the expert union in that check is
  one token's routing. Diverse-routing dispatch has only the 8-step teacher-forced rank check in
  `run_full_evidence` (10 distinct prompts, rank of the B1 token at most 1, no margin).
  Why this matters: serving buckets that are advertised and measured for speed have never been checked for
  output correctness, trace allocation safety or host fallbacks. The `$tt-enable-tracing` rule requires
  representative repeated-replay evidence with the tracker for each new trace variant.
  Required next step: add buckets 8 and 16 to `run_inactive_rows` and `run_sampling_audit` (both hard-code
  `(1, 4, 32)`) and run them with the tracker (`TT_METAL_TRACE_ALLOC_TRACKING=1`, `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`),
  and `run_full_evidence` with `--batches 1,4,8,16,32` (it uses `audit=True`). Add one distinct-prompt calibration:
  the AIME24 reference in row 0 and distinct filler prompts in the other rows, at B16 and B32, and report row 0
  KL against HF. Run on all three profiles.

- P2: The qualitative suite did not exercise the path this stage optimized
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_qualitative.py:74` calls `gen.generate(...)`, which is the unchanged B1 path. The outputs
  are byte-identical to stage 6 (`cmp` of `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/qualitative/tt_m{1,2,4}_outputs.json`
  against `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/qualitative/`). The only batched text is 48 tokens of four prompts in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/m*/sampling_audit.json` (`greedy_b4`, `sampled_b4`). They are coherent.
  No batched text exists for the B16 or B32 dispatch path with distinct prompts.
  Why this matters: `$qualitative-check` and `$optimize` require the shared suite after the optimized path is
  selected. Cross-request leakage is the main risk of batched GDN state and EP dispatch, and only generated text
  shows it directly.
  Required next step: decode the six shared prompts together in one B8 bucket and in one B32 bucket (fill the
  other rows with distinct prompts), greedy, 128 tokens, chat template, on all profiles. Compare each row with the
  B1 and HF outputs and record the verdict per prompt.

- P2: The token-out path keeps a blocking readback every token, against the goal
  Evidence: The goal says token-out decode uses "nonblocking trace replay, no per-token host sync/readback" and
  "has no host boundary". `generate()` calls `read_tokens()` after every replay (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py` `generate`,
  `predicted = int(self.read_tokens(1)[0])`), and `read_tokens` is a blocking `ttnn.to_torch`. The B4 to B32 timing
  loop also reads every step (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_batch_perf.py`, `gen.read_tokens(batch)`). The counters confirm 1908
  `token_readbacks` per run (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/m4/batch_perf_sampled0.json`). The README defines token-out as including
  the readback and does not address the goal item.
  Why this matters: the headline token-out numbers are measured with a host sync per token, which the goal
  excludes. The cost is not measured, so the claim that B1 needs no work is not supported for this term.
  Required next step: measure token-out with nonblocking replay and no per-step sync (for example, replay N steps,
  then read once, or overlap the read of token N with the replay of step N+1 through a separate token buffer).
  Confirm the tokens equal the per-step path. Report both numbers. Adopt the overlapped read in the generator, or
  state which caller needs the per-step sync and its measured cost.

- P2: Performance accounting and `perf_summary.json` are missing
  Evidence: `$optimize` lists "Performance accounting reconciled: roofline estimate, device-time decode, and
  end-to-end decode reported from the same run ... `perf_summary.json` written when optimizing a complete model"
  as a required item. No `perf_summary.json` exists under `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/` and the stage 7 README has no roofline or
  device-time row. The goal also asks for checklist evidence for full-model sharding, CCL, matmuls, program
  configs and kernels. The README has a terminal topology table but no checklist mapping. My scaling of the
  existing 2-layer profiles by layer count (30 linear, 10 full, plus terminal):
  - B1 p150x4: linear 205.5 us, full 234.6 us, terminal 957.3 us per step (device 0,
    `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence1_work/profile_b1_m4/reports/2026_10_14_23_03_07/ops_perf_results_2026_10_14_23_03_07.csv`). The sum
    is 9.47 ms against 9.19 ms end to end. B1 is device bound. This supports the B1 decision, but it is not in the
    README.
  - B32 p150x4: linear 1266.4 us, full 1043.3 us, terminal 948.1 us, prologue 87.4 us (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence1_work/profile_b32_m4/`).
    The sum is about 49.5 ms against 63.89 ms end to end, an unexplained gap of about 14 ms (23%). That profile is
    also unrepresentative (next item), so neither number can be trusted yet.
  - The README "terminal work" lists only the LM head (390 us) and top-k (286 us). The measured terminal path on
    p150x4 B1 is about 957 us: final norm on 1 core 24.8 us, LM head 391.1 us, and about 536 us of sampler ops
    (fill-pad 26.8, pad 30.1, top-k prep 25.8, top-k 283.6, top-k finish 20.1, two all-gathers 12.8, about 20
    small binary/typecast/reduce ops, manual seed 18.6, sampling 21.9).
  Why this matters: without the reconciliation, the B32 gap cannot be assigned to host dispatch or device work,
  and the sampler cost (5.8% of B1 token-out) is understated.
  Required next step: write `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/perf_summary.json` for B1 and B32 on p150x4 with roofline
  (stage 5 `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/evidence12/roofline.json` per layer plus LM head and KV bytes), device time from a representative reduced
  profile scaled by layer count, and end-to-end time. Explain each gap. Add a short `$optimize` and `$multichip`
  checklist table for the full-model path.

- P2: The B32 profile is unrepresentative, and the dominant batched MoE cost has no measured alternative
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_model_profile.py` prefills `torch.tensor([ids] * batch)`, so all 32 rows hold one prompt.
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/m4/profile_b32_run.json` shows row 0 token 28433 and rows 1 to 31 all
  96424. The expert union in the profiled window is therefore about one token's routing, not the 163 of 256
  experts that `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/batched_decode_audit.md` estimates for B32. The README profile table
  and its MoE conclusions ("MoE SwiGLU 1034") come from this window. In the same window the B>8 MoE path is the
  prefill EP dispatch: `MoeFusedSwiGlu` over 2272 capacity rows for 32 tokens, router `TopKDeviceOperation` on
  1 core (33.4 us per layer), gather and scatter on 1 core (about 30 us each), a 1-core matmul from 2048 to 1 column at
  32 rows (45 us), an untilize of 2272 by 2048 (50.2 us) and a tilize of 8 by 32 by 2048 (99.7 us). The README estimates the dispatch at "about 46 ms per full-model
  step at any B", which is about 70% of the 63.89 ms B32 step. The stage audit names the decode-shaped
  alternative ("qwen36 union-mask `sparse_matmul`", batched B-row expert path on the TP decode copy). The work log
  has no attempt at it. The `$optimize` MoE item requires the active-expert `sparse_matmul` path "with no
  avoidable DRAM round trips through decode intermediates".
  Why this matters: B32 decode is 15.6 t/s/u against 108.8 at B1, and the Stage 13 sweep runs client concurrency
  up to 32. The largest batched cost was analyzed with collapsed routing and its main alternative was not tried.
  Required next step: re-profile B32 (and B16) with distinct prompts per row and record the expert union. Measure
  a decode-shaped batched expert path (B-row union-mask `sparse_matmul` on the TP decode copy, or a dispatch with
  decode-sized capacity), or record the exact op-contract blocker after an adapted attempt. Recheck the
  `moe_row_loop_max` crossover per profile with the result.

- P2: The final default does not reproduce the B16 and B32 candidate, and the dismissal is contradicted
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batched/full_perf3/m4_tiles.json` (tile combine candidate): B16 52.89 ms, B32 60.93 ms, runs
  60.94 / 60.93 / 60.91. Final source: B16 54.61, B32 63.89 ms, runs 63.87 / 63.91 / 63.89
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/m4/batch_perf_sampled0.json`). Two more runs give 63.86
  (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/sampler/m4_topk.json`, its manifest has the final
  `optimized_decoder.py` hash) and 63.89 (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batch_perf2/m4_sampled0.json`,
  no manifest, run between chain 1 and chain 2, which both have the final hash). The work log calls the gap "run-to-run
  spread about 5%". The measured spread is at most 0.1% within a run and 0.05% across the three final-source
  runs. The sources differ: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py` `7facebcc...` (full_perf3) against `92838721...` (final),
  and `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py` `817ba352...` against `56d61689...`. The decode bucket sets also differ (1, 8, 16, 32
  against 1, 4, 8, 16, 32). The README still cites 60.93 in "What changed", "Conclusions and actions" and
  "Rejected or deferred".
  Why this matters: `$optimize` requires the final default to reproduce the selected candidate or the report to
  use the final number and explain the difference. A 4.9% B32 regression with an incorrect explanation can hide a
  wiring or allocation-placement defect.
  Required next step: rerun the full_perf3 configuration on the final source, find the change that costs about
  3 ms at B32, fix it if it is a regression, and correct the three README rows to the reproduced numbers.

- P2: The DRAM-sharded LM head rejection is not earned
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/lmhead/trace.json`: 3 readers per bank, 64 input cores, `in0_block_w` 1, 336.9 us traced including
  reshard and unshard, PCC 0.99998. The in-model interleaved LM head is 387 to 391 us on 108 cores at 65.5 to
  66.2% of DRAM bandwidth (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/m4/profile_b1/decode_perf_report.txt`). The README rejects it because "the
  gain is inside run-to-run spread and needs full requalification". The warmed B1 token-out runs and the medians
  across chains agree within 0.02 ms (for example 9.17 / 9.18 / 9.19), smaller than the 53 us gain. Run 1 of
  each file is about 0.64 ms slower because it includes the B1 trace capture; the median excludes it. Requalification cost is
  not a valid reason under `$optimize`. The p150x2 and p150 vocabulary shards (124160 and 248320) were tried only
  with 1, 2 and 4 readers, eager (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/lmhead/micro.json`). Three readers and an N split were not tried there,
  although the LM head is 738 us and 1480 us on those profiles (9.3% of p150 token-out).
  Why this matters: `$optimize` lists "LM Head is optimized for DRAM-sharded matmuls if present" and the
  1/2/3-reader sweep per material role as required items. A measured, correct win was dropped for convenience.
  Required next step: implement the DRAM-sharded LM head on p150x4 with vocabulary padding and the sampler tail
  mask, measure it in the model, and adopt it if faster. Try 3 readers and an N split for p150x2 and p150, or
  record the exact L1 blocker.

- P2: Chain 1 evidence ran on a different source and the README does not say so
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/source_manifest_end.txt` against `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/source_manifest_start.txt` differ in
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py` (`c91c2521...` against final `56d61689...`) and in `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_batch_perf.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_sampling_audit.py` and `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_seed_advance.py`. The work log shows the change: `seed_mode`
  default "skip" to "host" and a `ValueError` for "skip" when tp > 1. The README presents `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/` (HF gates,
  autoregressive, qualitative, full evidence, calibration, inactive rows, profiles, watcher, split-sampling trace,
  sampler micro) without that note. No chain 1 source archive or diff is preserved.
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/m4/split_sampling_trace.json` records `seed_writes` 2, which is the SKIP behavior; the final source
  writes seeds every sampled step, and that loop has no final-source split-trace run.
  Why this matters: `/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json` requires the dirty
  diff or source archive for uncommitted candidates and preserved semantic lineage before evidence from one
  source is reused for another.
  Required next step: reconstruct and save the chain 1 `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py` diff, state in the README that greedy
  paths are byte-identical in behavior between the two sources (the greedy branch of `_sample` returns before any
  seed handling), and rerun the sampled-path checks that ran only on chain 1 (`run_split_sampling_trace`,
  `run_sampler_micro`) on the final source.

## Other Concerns

- p150 batched B4 changes greedy text. `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/m1/sampling_audit.json` `greedy_b4[0]` diverges from
  `greedy_b1` at "Tone: Open" ("(could be peaceful" against ", but generally"). Stage 6 (per-row B4) matched.
  The work log says p150 B4 is bitwise equal, but that holds only for the 2-layer check
  (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batched/l2_profiles/m1_b{0,1}.pt`). The alternative also appears in sampled rows, so a near tie is likely.
  The margin is not recorded.
- The B>8 decode path uses the prefill expert numerics: BFP4 gate/up with LoFi and `fp32_dest_acc_en=False`
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py:205-212`), while the selected decode policy uses FP32 accumulation for gate/up
  (stage 5 precision table). Down stays BFP8, so no rejected dtype is used. The README does not disclose the
  accumulation difference. The only HF evidence for it is the identical-row B32 calibration.
- The README attributes +1.24 ms at B32 to per-step host seed writes. The cost is 0.01 ms at B4 and B8 and
  appears only on the dispatch buckets, with larger run spread (65.08 to 65.51 ms). Sampled tokens change the
  routing, so the cost is probably data-dependent MoE work, not the 128-byte write. The attribution is unverified.
- Sampled mode still writes seeds from the host every step. This is a host input in the sampled token loop. Only
  the biased Weyl add and SKIP were tried as device-side alternatives.
- `moe_row_loop_max` 8 is tuned on p150x4 only. On p150 the row loop extrapolates to about 144 ms at B16 against
  158.89 ms for dispatch.
- The p150 batched GDN kernel hard-codes 4 workers per head and needs 128 cores. Two blocks per head, or a head
  loop, were not tried. p150 B32 keeps the per-row GDN loop.
- The greedy sampler choice was benchmarked on p150x4 only. On p150 the local top-k is about 1070 us with no
  all-gather in the argmax path, so the trade-off can differ.
- The final norm runs on 1 core (24.8 us per step) after `sharded_to_interleaved` (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/model.py` `logits`).
- The README says inactive rows match "bitwise". `run_inactive_rows` compares token sequences, not logits.
- The README profile table counts 1095 ops for "4 steps" of B32. 219 of those rows are the trace capture with
  no device cycles; each step has 219 ops.
- The README does not report the traced teacher-forcing decode t/s/u (for example 100.07, 82.97, 60.65 t/s/u
  on AIME24 in `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/m*/teacher_forcing_aime24.log`) beside token-out and logits-only.
- The `generic_op` program-cache key ignores per-core runtime-argument length. The workaround (user count and
  span in compile arguments, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/gdn_batch.py`) is sound for these kernels. File the upstream issue.
- `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/calibration_compare.json` has no listed generator command.
- Untracked stage 6 files `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence{1,2}/m*/profile/ops_perf_results.csv.gz` are in the
  worktree. Stage 6 left them out on purpose. Keep them out of the stage 7 checkpoint commit.

## Hard-Check Gaps

- `run_full_evidence` batch checks record argmax agreement and rank but no logit margin. The 5-token row agrees
  at 6 of 8 steps at B4 on p150x4, at 7 of 8 at B4 on p150x2, and at 7 of 8 at B32 on p150x4 and p150.
- `max_row_spread` 0.0 in the B32 calibration only shows determinism across identical rows.
- The runtime fallback audit covers B4 and B32 capture through `run_full_evidence` (`audit=True`), but the README
  does not state it.
- No B1 sampled token-out number exists. `run_batch_perf` B1 is greedy in both files.

## Anomaly Ledger

- Observed anomaly: SKIP seed mode lets device token copies diverge at B32 on meshes.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_copies/m4_skip_b1.json` (128 of 128 steps, 3967 row-copies); negative control
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/m4/sampling_audit_skip_control.json` fails `s32`.
  Affected path: sampled decode with EP dispatch between draws (B>8 and every prefill).
  Control or comparison: per-row decode and MoE row loop give 0 mismatches; host seeds give 0.
  Likely subsystem: core PRNG state shared with dispatch kernels.
  Investigation performed: decode path matrix (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_copies/`), audit check added and proven by the control.
  Resolution: fixed (host default, SKIP raises for tp > 1). The consuming op is not identified; host mode reseeds
  before every draw, so it does not depend on that op.

- Observed anomaly: B32 candidate 60.93 ms is not reproduced (63.89 ms).
  Evidence: Required Work 6.
  Affected path: B16 and B32 decode default.
  Control or comparison: three final-source runs agree within 0.05%.
  Likely subsystem: MoE dispatch wiring or trace/allocation placement after source changes.
  Investigation performed: none; dismissed as run spread.
  Resolution: more-work-needed.

- Observed anomaly: B32 profile rows 1 to 31 decode one identical token.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/m4/profile_b32_run.json` `tokens`.
  Affected path: B32 profile conclusions and MoE cost analysis.
  Control or comparison: none with distinct prompts.
  Likely subsystem: profile harness input construction.
  Investigation performed: none.
  Resolution: more-work-needed.

- Observed anomaly: about 14 ms gap between scaled B32 device time (about 49.5 ms) and end-to-end (63.89 ms).
  Evidence: Required Work 4 scaling.
  Affected path: B32 token-out.
  Control or comparison: B1 has no gap (9.47 ms scaled against 9.19 ms).
  Likely subsystem: unrepresentative routing in the profile, or dispatch gaps in the 40-layer trace.
  Investigation performed: none.
  Resolution: more-work-needed.

- Observed anomaly: p150 batched B4 greedy text diverges from B1 at about token 38.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/m1/sampling_audit.json`.
  Affected path: p150 batched decode.
  Control or comparison: the alternative token appears in sampled rows; KL of batched paths to HF is not worse.
  Likely subsystem: numeric difference of the batched projections and SDPA at a near tie.
  Investigation performed: none recorded.
  Resolution: controlled in part; record the margin.

- Observed anomaly: `generic_op` reuses a cached program with shorter per-core runtime arguments (NaN state at 32
  users after 4).
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batched/layer_check1/m4.json` (32-user rows NaN), fixed in `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batched/layer_check2/m4.json`.
  Affected path: batched GDN kernels.
  Control or comparison: separate processes per user count.
  Likely subsystem: `compute_program_descriptor_hash` in `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp` (line 66 hashes `kernel.runtime_args.size()`, the core count, not per-core lengths).
  Investigation performed: source reading, compile-argument workaround, layer recheck at 4 and 32 users.
  Resolution: fixed by workaround; upstream issue pending.

- Observed anomaly: chain 1 `batch_perf` measured the per-row path.
  Evidence: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/m*/batch_perf.json` (`batched` 0).
  Affected path: evidence only.
  Control or comparison: chain 2 rerun with `--batched 1`.
  Likely subsystem: harness default.
  Investigation performed: default changed, rerun.
  Resolution: fixed; the failed attempt is kept as the per-row baseline.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/07-07-optimized-full-model.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`, `/home/hous/dev/qwen3.5-35b-a3b/experiment-provenance-handoff.json`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`, `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md`, `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/multichip/SKILL.md`, `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/qualitative-check/SKILL.md`,
  tracing and full-model readback rules in `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/tt-enable-tracing/SKILL.md` and `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/full-model/SKILL.md`.
- Artifact paths: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/README.md`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/work_log.md`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/batched_decode_audit.md`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence1/` and `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_full_model/evidence2/` (all JSON summaries, readiness logs, profile summaries and perf reports, watcher
  logs, qualitative and autoregressive outputs, manifests), `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/context_contract.json` diff,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/README.md`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/` (batch perf, full evidence, qualitative,
  autoregressive, sampling audit), `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_multichip_decoder/README.md` (results, roofline, precision),
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain1.sh`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain1.log`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain2.sh`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence_chain2.log`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/process_profiles.sh`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/update_contract7.py`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batched/` (full_perf1 to
  3, l2_profiles, gdn_check2, layer_check1 and 2), `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/sampler/`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/batch_perf2/`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/seed_copies/`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/lmhead/`,
  raw profiler CSVs for p150x4 B1 and B32 in `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage07/evidence1_work/`.
- Code paths: `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/generator.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/model.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/optimized_decoder.py`
  (diff and `_LocalDispatch`, default policies), `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/multichip_decoder.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/gdn_batch.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tt/gdn_decode.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_batch_perf.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_model_profile.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_full_evidence.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_inactive_rows.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_sampling_audit.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/run_calibration_probe.py`, `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/tests/test_generator_sampling.py` (diffs).
- Commands run: `git status`, `git log`, `git diff HEAD` on the changed files; manifest recomputation and `diff`
  against chain manifests; `cmp` of qualitative and autoregressive outputs against stage 6; read-only Python over
  JSON evidence (batch perf, calibration, full evidence, inactive rows, sampling audit, seed tests, LM head
  micro, l2 bitwise tensors); per-op and per-step analysis of the raw p150x4 B1 and B32 `ops_perf_results` CSVs;
  `grep` over logs and docs. No device, server or pytest run.

## Residual Risk

- B1 is device bound (scaled profile 9.47 ms against 9.19 ms end to end), so further B1 gains need device work:
  the sampler glue (about 536 us per step) and the LM head (391 us) are the largest non-layer items.
- KV pool capacity is unchanged: p150x4 holds about 4 native-context sequences with 12.96 GB DRAM free after all
  captures. Client concurrency 32 at ISL 131072 cannot be resident at once on any profile. This is a stage 9 to 13
  planning item, not a stage 7 regression.
- Prefill is unchanged. The 4096-token profile shows MoE dispatch data movement and expert SwiGLU dominating
  prefill, the same dispatch family as the B>8 decode cost.
- HF gates, qualitative text and long-context results are equal to stage 6 because the B1 path is unchanged. They
  do not test the batched path.
