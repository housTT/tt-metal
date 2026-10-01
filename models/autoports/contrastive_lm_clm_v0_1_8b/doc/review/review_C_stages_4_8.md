# Stage Review: checkpoint C, stages 4 to 8

Reviewer mode, independent, read-only. Written 2026 Oct 1, 18:50 ET.
Target: Contrastive-LM/CLM-v0.1-8B (frozen Qwen3-8B encoder, last-token pooling after the final RMSNorm, L2 normalized, two MLP heads; score = 100 x cosine of head projections).
Autoport: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b`, branch `hous/clm-v0.1-8b`.
Worktree state: the review started at commit `fe0b69f03e` with `doc/optimized_full_model/README.md` untracked and `doc/optimized_full_model/replay_trace_check_bfp8_attn.json` modified. Commit `add82f9e7d` (22:36:06 UTC) landed during the review and contains exactly those two files plus `doc/release/RUN_NOTES.md` and two version markers. The autoport tree is clean at `add82f9e7d`. All artifacts below were read as they are on disk at `add82f9e7d`.
Times quoted from artifacts are UTC as stamped in the files. Eastern Time is UTC minus 4 h (22:07 UTC = 18:07 ET).

Acronyms: TT = Tenstorrent. HF = Hugging Face. TTNN = the tt-metal neural network library. CPU = central processing unit. KV = key-value. MLP = multilayer perceptron. RMSNorm = root mean square normalization. SDPA = scaled dot-product attention. CCL = collective communication library. TP = tensor parallel. BFP8 / BFP4 = block floating point, 8-bit / 4-bit mantissa. bf16 = bfloat16. fp32 = 32-bit float. HiFi / LoFi = high / low math fidelity. TV = total variation distance between two probability vectors. JSON, CSV, PNG = file formats. SHA = commit hash.

Verdict: more-work-needed

## Required Work

- P1: The stage-6 decision-agreement gate was dropped, and when recomputed from the stage's own saved vectors it fails for every policy, including the selected one
  Evidence:
  `/home/hous/dev/clm-v0.1-8B/PLAN.md` section 4, row 6, sets the stage-6 gate to include "decision agreement (argmax) >= 98 % on Typed Decisions cases scored by both". Row 8 selects "the fastest config that passes the stage 6 gate". Neither `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/full_model/README.md` nor `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/datatype_sweep/README.md` measures it. Both defer it to `doc/release/` ("measured against the served package"). `tests/datatype_sweep.py` lines 44 to 50 gate only on cosine mean, cosine min, head-projection min and NaN count. The sweep's "passes the stage 6 gate" claim therefore uses an incomplete gate.
  I recomputed the metric offline from the saved vectors (`fidelity_*_tt_single.npy`, `fidelity_*_tt_batched.npy`), the head checkpoint `/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt`, the case mapping in `/home/hous/dev/clm-v0.1-8B/reference/typed_decisions_subset.json` and the fp32 reference `/home/hous/dev/clm-v0.1-8B/reference/hf_embeddings.npy`, with the authors' scoring (logit = 100 x cos(state_head(e_s), action_head(e_a)), softmax over a question's candidates, temperature 1). The scorer reproduces the reference artifacts exactly: 74/200 correct vs gold as in `typed_decisions_subset_reference.json`, README customer example noul 0.8419 / billing 0.9880 / frustration 2.0000, tides 0.9935 as in `readme_examples_reference.json`.

  | source (vectors) | mode | argmax agreement vs fp32 reference, all 200 | agreement on the 188 decisions whose reference top-2 margin >= 0.10 | mean TV | max TV |
  |---|---|---|---|---|---|
  | accuracy, 21:51 run (`doc/full_model/*_tt_*.npy`) | single | 193/200 = 96.5 % | 186/188 = 98.9 % | 0.038 | 0.29 |
  | accuracy, 21:51 run | batched | 191/200 = 95.5 % | 185/188 = 98.4 % | 0.039 | 0.36 |
  | accuracy, 22:07 host-norm run (`doc/optimized_full_model/fidelity_accuracy_hostnorm_tt_*.npy`) | single | 191/200 = 95.5 % | 186/188 = 98.9 % | 0.038 | 0.29 |
  | accuracy, 22:07 host-norm run | batched | 191/200 = 95.5 % | 185/188 = 98.4 % | 0.038 | 0.36 |
  | bfp8_attn, 21:59 run (selected policy, `doc/datatype_sweep/fidelity_bfp8_attn_tt_*.npy`) | single | 189/200 = 94.5 % | 183/188 = 97.3 % | 0.046 | 0.32 |
  | bfp8_attn, 21:59 run (selected policy) | batched | 187/200 = 93.5 % | 181/188 = 96.3 % | 0.047 | 0.34 |
  | bfp8_attn_hifi2, 22:06 run | single | 183/200 = 91.5 % | 180/188 = 95.7 % | 0.045 | 0.35 |
  | bfp8_attn_hifi2, 22:06 run | batched | 191/200 = 95.5 % | 183/188 = 97.3 % | 0.043 | 0.32 |
  | performance, 21:53 run | single | 158/200 = 79.0 % | 155/188 = 82.4 % | 0.176 | 0.77 |
  | performance, 21:53 run | batched | 161/200 = 80.5 % | 156/188 = 83.0 % | 0.175 | 0.79 |
  | 1x4 accuracy, 22:14 run (`doc/multichip_decoder/fidelity_accuracy_1x4_tt_*.npy`) | single | 190/200 = 95.0 % | 185/188 = 98.4 % | 0.036 | 0.28 |
  | 1x4 accuracy, 22:14 run | batched | 192/200 = 96.0 % | 186/188 = 98.9 % | 0.034 | 0.27 |

  Reference margin distribution: 8 of 200 decisions have a top-2 probability margin below 0.05, 12 below 0.10, 188 at or above 0.10; the median top-2 logit gap is 2.77 logits.
  The flips are not all near-ties. For the selected policy (single vectors) five reversed decisions had reference margins >= 0.10, for example `invoice_processing_000096/discrepancy_severity`: reference p = [0, 0.329, 0.017, 0.653], TT p = [0, 0.653, 0.008, 0.339], with the state text at cosine 0.9993 vs HF and all candidates at >= 0.9995. The logits moved from [15.5, 24.8, 21.9, 25.5] to [15.7, 25.9, 21.5, 25.2]: a 1.1-logit shift reversed a 0.7-logit margin. `agent_trace_observability_000046/risk` went from p = [0.057, 0.515, 0.166, 0.262] to [0.068, 0.275, 0.215, 0.441]. `customer_service_000027/action` went from [0.025, 0, 0.151, 0.285, 0.539] to [0.014, 0, 0.087, 0.549, 0.35]. The batched vectors (the served path) are the worst of the three bfp8 columns: 93.5 % and seven material reversals. The maximum per-decision logit error for the selected policy is 1.61 (mean 0.59, p95 1.06); for the performance policy it is 4.82 (mean 2.06).
  Why this matters:
  The model's output is a probability vector computed from 100 x cosine. The heads amplify embedding error: decisions reverse at raw cosine 0.9993, far above the stage gate's 0.97 minimum, and even above the measured 0.994 minimum. The stage-6 README justifies the cosine thresholds by anisotropy and a centered-cosine row; it never ties them to decision outcomes. The datatype sweep then ranked policies using only those cosine thresholds. With the gate as written in the plan (98 % argmax agreement) no policy passes, not even the stock accuracy policy (95.5 to 96.5 %). With a margin-aware reading (agreement on decisions with reference margin >= 0.10), the accuracy policy and the 1x4 mesh pass at 98.4 to 98.9 % and the selected bfp8_attn policy does not (97.3 % single, 96.3 % batched). Under either reading the selection in `doc/datatype_sweep/selected_precision_config.json` is not supported by the stage-6 gate. The performance-policy rejection, by contrast, is well supported: 79 % agreement and mean TV 0.176.
  Required next step:
  1. Add a decision-agreement artifact to stages 6 and 8 (single and batched, every policy), recording plain argmax agreement, margin-aware agreement, mean and max TV, and the list of reversed decisions. The script in the appendix of this file reproduces the table above from the existing vectors without a device.
  2. Decide and justify the gate in writing. Either keep 98 % plain argmax, in which case no evaluated policy passes and a higher-fidelity candidate (bf16 MLP weights, HiFi4 MLP math, or fp32 accumulation) must be measured to find out whether 98 % is reachable on this hardware, or adopt a margin-aware gate and justify it with the margin distribution above. Record the choice in `doc/full_model/README.md`.
  3. Re-run the stage-8 selection against the complete stage-6 gate and regenerate `sweep_results.json`, `sweep_results.csv`, `selected_precision_config.json`, the two PNGs and `tt-model.yaml`'s default profile accordingly. With a margin-aware 98 % gate the selection changes to `accuracy`.
  4. Carry the measured agreement and TV numbers into the model card limitations.

- P2: `doc/context_contract.json` is stale, and the 2048-token capability decision is not recorded the way the skills require
  Evidence:
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/context_contract.json` still says `current_supported_context: 1024`, `target_supported_context: 2048`, and `limiting_reason: "stage 6 probe runs with max_seq_len 1024; 2048 requires a traced prefill length entry for Qwen3-8B on P150 (stage 7)"`, with `capacity_evidence: doc/probe/probe_full_encoder.json`. Stages 6, 7 and 8 all serve 2048 (`doc/full_model/README.md` "this port serves 2048"; `tt-model.yaml` `CLM_MAX_TOKENS: "2048"`; every fidelity and bench JSON has `max_seq_len: 2048`). The full-model, optimize and datatype-sweep prompts each require the contract to be recomputed for the stage. It was not touched after the probe.
  On the capability itself: `hf_advertised_context` is 40960. The upstream CLM README (`/tmp/claude-1002/-home-hous-dev-clm-v0-1-8B/0082bb62-0b82-4079-9fa4-87d59d66a985/scratchpad/clm-repo/README.md` lines 65 to 66) states that 2048 is a default and that longer states are supported by raising `--max-model-len` and `clm-serve --max-tokens` together, for example to 8192. The port's limit is fixed in a different way: `tt/encoder.py` takes `trace_lens` from `model_args.trace_prefill_supported_seq_lens` (128, 1024, 2048) and `_traced_prefill` calls `Generator._easy_trace_prefill`, which captures a new trace at first use when a key is missing (`models/tt_transformers/tt/generator.py` lines 826 to 842). A user who sets `CLM_MAX_TOKENS=4096` would get a runtime capture after warmup, which the stage's own trace-safety rule forbids. The justification given (upstream default, training recipe `max_len 2048`) is reasonable for a verifier whose heads were trained at 2048, but it is recorded only in prose in the stage-6 README, and the contract file contradicts it.
  Why this matters: the contract is the artifact later stages and the card consume; it currently advertises a smaller context than the port serves and names a stage-7 blocker that no longer exists. The skill treats a stale or contradicted required artifact as more work.
  Required next step: recompute and rewrite the contract for stages 6 to 8: served context 2048 and why (training recipe, upstream default), traced buckets 128 / 1024 / 2048, batch 8, KV blocks (1024 blocks x 32 tokens), the two truncation semantics (serving keeps the last 2048 tokens for every role; the fidelity harness keeps 2047 with head truncation for candidates), the measured non-aligned lengths (32, 33, 127, 129, 500, 2047), what raising the limit would need (trace length entries and a re-run of the fidelity corpus at that length), and the upstream configurability note. Make `TtQwen3Encoder` reject or clip `CLM_MAX_TOKENS` above the largest traced length instead of capturing at runtime.

- P2: `selected_precision_config.json` is incomplete, and the sweep README misdescribes two of the four measured policies
  Evidence:
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/datatype_sweep/selected_precision_config.json` contains a policy name, an environment variable, the gate and the selected row. The datatype-sweep skill requires weight dtype groups, layer exceptions, compute fidelities, activation/residual dtype, CCL dtype, KV-cache dtype, output dtype assumptions and the runtime flags, and says "A selected config that only says BFP8 weights is incomplete".
  The resolved policies, from `tt/encoder.py` `CUSTOM_POLICIES` merged over `ModelOptimizations._default_settings` (`models/tt_transformers/tt/model_config.py` lines 380 to 414) and the stock `accuracy` / `performance` constructors (lines 224 to 333, Qwen3-8B takes the generic `else` branches):

  | group | accuracy (stock) | bfp8_attn (selected) | bfp8_attn_hifi2 | performance (stock) |
  |---|---|---|---|---|
  | WQKV, WO weights | BF16 | BFP8 | BFP8 | BFP8 (default) |
  | KV_CACHE | BF16 | BFP8 | BFP8 | BFP8 (default) |
  | FF1_FF3 weights | BFP8 (default) | BFP8 | BFP8 | BFP4 |
  | FF2 weights | BFP8 (default) | BFP8 | BFP8 | BFP8 (default) |
  | LI_FF1_FF3 fidelity | HIFI2_FP16 (default) | HIFI2_FP16 | HIFI2_FP16 | LOFI |
  | LI_FF2 fidelity | HIFI2_FP16 (default) | HIFI2_FP16 | HIFI2_FP16 | HIFI2_FP16 (default) |
  | LI_QKV_PREFILL, LI_O_PREFILL | HIFI4 | HIFI2 (default) | HIFI2 | HIFI2 (default) |
  | SDPA_PREFILL | HIFI4 | HIFI4 (default) | HIFI2 | HIFI4 (default) |
  | activation | bf16 (input dtype) | bf16 | bf16 | bf16 |

  The README candidate table (`doc/datatype_sweep/README.md`) states that `performance` uses bf16 attention weights and KV with HiFi4 attention math; the code says BFP8 attention weights and KV with HiFi2 linear projections. It lists `accuracy` MLP math as "HiFi2" and `bfp8_attn` as "HiFi2, fp16 accumulate" as if they differed; both are `HIFI2_FP16`. It describes `bfp8_attn_hifi2` as changing "the attention matmuls" from HiFi4 to HiFi2; the only effective change is `SDPA_PREFILL`, because the QKV and O prefill projections are already HiFi2 by default under `bfp8_attn`. The decode op groups are irrelevant for this encoder and should be recorded as not exercised.
  Why this matters: later stages must be able to construct the exact policy mechanically, and the rejection ledger must describe what was actually measured. The performance row's rejection is attributed to "bfp4 MLP", which is right, but the table says the attention side was bf16/HiFi4 when it was BFP8/HiFi2.
  Required next step: write the resolved table above into `selected_precision_config.json` (plus `layer_exceptions: none`, `ccl_dtype: bf16 on 1x4, none on 1x1`, `final_norm: host fp32`, `heads: host fp32`, `weight_dtype_default: bfp8`, `env: CLM_PRECISION=bfp8_attn`), correct the README table, and cite the propagation path (`DecodersPrecision.get_tensor_dtype` / `get_math_fidelity`, consumed in `models/tt_transformers/tt/attention.py` lines 128 to 150 and `models/tt_transformers/tt/mlp.py` lines 176 and 340) as the consumption evidence.

- P2: The sweep and stage-6/7 fidelity evidence mixes three code generations, one JSON no longer matches its vectors, and the selected policy was never measured on the final path over the full corpus
  Evidence (dated from the log markers in `/home/hous/dev/clm-v0.1-8B/logs/` and the `encoder_stats.calls` field; the warmup token count 28800 = 9 x 3200 identifies two warmup batch sizes, 41600 = 13 x 3200 three):
  - Generation A, commit `418c0e85bf` code: device-side slice and norm after the trace, per-length warmup through `embed_ids`, log line `warmup done ... for lens=`, 386 calls. Produced `fidelity_performance.json` (21:53), `fidelity_bfp8_attn.json` (21:59, the selected policy's row in `sweep_results.json`), the stage-6 README table (`fidelity_accuracy.log`, 21:51: mean 0.99909, min 0.99588, head min 0.99429, batched min 0.99643) and `bench_performance.json` (21:58). The trace tracker later flagged this generation with 43 corruptible buffers (`replay_trace_check.log`, 22:01).
  - Generation B, commit `498325e8ed` code: host-norm readback, per-variant prepare+capture warmup through `embed_ids` for batch 1/4/8, log line `traces captured: [...]`, 389 calls. Produced `fidelity_bfp8_attn_hifi2.json` (22:06) and the 22:07 accuracy run. The tracker flagged this generation with 176 corruptible buffers (`doc/optimized_full_model/replay_trace_check_accuracy.json`, 22:11, `pass: false`).
  - Generation C, commit `fe0b69f03e` code: two-phase warmup, log line `(prepare ... capture ...)`, 380 calls. Produced `fidelity_accuracy_1x4.json` (22:14), `fidelity_bfp8_attn_twophase_limit96.json` (22:23, 96 of 308 texts, no text above 1024 tokens), `bench_bfp8_attn.json` (22:15), `bench_accuracy.json` (22:17), `bench_bfp8_attn_hifi2.json` (22:21), `bench_accuracy_1x4.json` (22:19) and the passing replay check (22:26).
  - `doc/full_model/fidelity_accuracy.json` carries the 22:07 (generation B) numbers (mean 0.999103, min 0.995960), but `doc/full_model/fidelity_accuracy_tt_single.npy` beside it is the 21:51 (generation A) run: recomputed mean 0.999094, min 0.995878, exactly the stage-6 README table. `doc/optimized_full_model/fidelity_accuracy_hostnorm.json` is a byte-identical copy of that JSON and its `.npy` is the real 22:07 run. The stage-6 README table therefore has no backing JSON on disk, only the log line. `doc/datatype_sweep/README.md`'s accuracy row (0.99588 / 0.99429) contradicts `sweep_results.json` (0.99596 / 0.99434).
  - In `sweep_results.json`, the fidelity columns for `bfp8_attn` and `performance` come from generation A, the `accuracy` and `bfp8_attn_hifi2` columns from generation B; the latency columns for `performance` come from generation A, the other three from generation C. The two PNGs plot these mixed points.
  Controls that bound the risk: generation A vs generation B accuracy vectors agree at cosine mean 0.999994, min 0.999989 (the host-norm change is numerically negligible). The 96-text generation-C run of `bfp8_attn` agrees with the generation-A vectors at cosine mean 0.999994, min 0.999988. So the mixed regime did not change any gate outcome, but the artifact set does not show that by itself, and the 96-text file is not mentioned in any README.
  Why this matters: the skill requires that the fidelity used for selection ties to the measured final path, and that paths in reports exist and match the described run. Two of the sweep's four fidelity rows and the stage-6 headline table were produced by code the stage itself later classified as trace-unsafe.
  Required next step: re-run `tests/run_fidelity.py --precision bfp8_attn` (308 texts) on the final path and point the sweep at it; re-run or drop the `performance` bench so all latency columns share a regime; restore the one-to-one correspondence between each fidelity JSON and its `.npy` (re-run or restore the 21:51 JSON next to its vectors); update both README tables from the JSON; either document the 96-text file as the regime bridge or remove it.

- P2: The isolation gate was weakened from per-text to mean without a written justification, and the resulting batch-composition nondeterminism is not recorded
  Evidence: PLAN row 6 says "same text alone vs inside a mixed-length batch gives the same vector (cosine >= 0.999)". `doc/full_model/README.md` reports it as "mean >= 0.999". Recomputed from the vectors: 60 of 308 texts fall below 0.999 for the accuracy policy (min 0.99644) and 56 for `bfp8_attn` (min 0.99630). Decision impact, same vectors scored single vs batched: argmax differs on 5/200 decisions (accuracy), 4/200 (`bfp8_attn`), 8/200 (`bfp8_attn_hifi2`), 6/200 (1x4), with max TV 0.18. The upstream note the README cites (`models/tt_transformers/tt/model_config.py` lines 734 to 745, tt-metal issue 47238) describes the same effect and the stock workaround is `disable_batched_prefill` for the affected models; it is not enabled for Qwen3-8B on P150.
  Why this matters: a verifier that returns different probabilities for the same request depending on how the server batched it is nondeterministic in a way the card must state, and the plan's gate was the place that should have forced the decision. The cost of the batch-invariant alternative is measurable from the existing bench: at 128 tokens, batch 8 takes 142 ms versus 8 x 54 = 432 ms as eight batch-1 calls.
  Required next step: choose one of (a) a batch-invariant path (per-user batch-1 prefill) with its measured cost, or (b) keep batched prefill and record the nondeterminism in the contract and the card with the numbers above, restating the stage-6 gate as it was actually applied (mean), with the per-text distribution.

- P2: The sweep lacks the compute-fidelity and restore-order candidates the skill names, so "fastest passing" is not established
  Evidence: four candidates were evaluated (`sweep_results.json`). There is no BFP8 + LoFi candidate for the dominant prefill matmul groups FF1_FF3 and FF2 (the datatype-sweep skill: "include BFP8+LoFi and BFP8+HiFi2 candidates when both are legal"; "Do not assume HiFi2 is fastest for BFP8"). There is no restore-order candidate after `performance` failed (skill step 5: restore the highest-risk groups until the gate passes, for example BFP4 FF1_FF3 excluding the first and last layer, or FF2 only). There is also no higher-fidelity candidate (bf16 MLP weights or HiFi4 MLP math), which P1 needs to establish whether the decision-agreement bar is reachable.
  Why this matters: the stage claims the fastest passing policy; the skill asks whether that claim is only true because an obvious legal candidate is missing. Here two obvious candidates are missing. Given P1, lower-precision candidates will probably fail a decision-level gate, but that must be measured or excluded with a recorded reason.
  Required next step: after the gate in P1 is settled, add at least BFP8 + LoFi for FF1_FF3/FF2 and one BFP4 layer-exception candidate, plus one higher-fidelity candidate, and record the per-group table (weight dtype, fidelity, latency, decision agreement, kept/rejected, reason).

- P2: `work_log.md` is missing for stages 4 to 8
  Evidence: `ls doc/*/work_log.md` returns only `doc/probe/work_log.md`. PLAN section 4 ("Every stage keeps ... doc/<stage>/work_log.md") and prompts 04 to 08 require it.
  Why this matters: the code-generation history in the previous P2 had to be reconstructed from log sizes, line numbers in log messages and call counts. The work log is where that belongs.
  Required next step: write `doc/multichip_decoder/work_log.md`, `doc/full_model/work_log.md`, `doc/optimized_full_model/work_log.md` and `doc/datatype_sweep/work_log.md` with commands, UTC timestamps, SHAs, which code generation produced each artifact, and the review outcome.

## Other Concerns

- Selection regime and wording. The tie-break rule in `tests/datatype_sweep.py` uses only 128-token batch-1 latency with a 1 % band. At that column `bfp8_attn` (53.96 ms) and `bfp8_attn_hifi2` (53.95 ms) tie and the higher minimum cosine wins, as the README says. At 2048 tokens `bfp8_attn_hifi2` is 3.2 % faster (276.95 vs 286.22 ms) and at 1024 tokens 0.6 % faster, so the README sentence "beats bfp8_attn_hifi2 on fidelity at the same speed" is only true for the short bucket. The short bucket is the dominant serving regime for this model (178 of 308 corpus texts, README examples of 2 to 19 tokens), which is a sound reason, but it is not written down. The README's "13 percent faster at 1024 and 2048" is 13.1 % at 1024 and 11.3 % at 2048 (322.81 to 286.22 ms).
- The stage-7 "lower bound" is above the measurement. `perf_summary_accuracy.json` reports `layer_stack_lower_bound_ms: 57.92` against `p50_ms: 57.64` at 128 tokens (gap minus 0.5 %). Thirty-six times the layer-0 profile exceeds the end-to-end latency, so the profile overestimates traced per-layer time and the README's inference that "the terminal work ... and dispatch are hidden" is not supported by this number. No profile exists under the selected policy ("a policy-matched profile was not captured"). The bfp8_attn rows are 7 to 14 % below the same bound.
- Stage 4 gate not reported, stage 5 not done, profile published anyway. PLAN row 4 gates the 1x4 mesh on "cosine vs the 1-chip TTNN output >= 0.999". `doc/multichip_decoder/README.md` compares 1x4 to HF instead. Recomputed 1x4 vs 1x1 TTNN (both accuracy policy, single vectors): mean 0.99929, min 0.99547, 68 of 308 texts below 0.999. The README's speedups check out against `bench_accuracy_1x4.json` and `bench_accuracy.json` (1.89x, 1.62x, 1.55x, 1.52x). Stage 5 (CCL audit, inter-layer residual contract) is declared "Not done beyond the stock TP plan", yet `tt-model.yaml` publishes `p150x4` as a serve profile. The README also contradicts itself on the layout: "the residual stream comes back width-sharded across the four devices" versus "the residual is replicated on every device after each all-gather". `models/tt_transformers/tt/model.py` line 256 documents the batched prefill output as `[padded_batch, 1, prefill_seq_len, dim_per_device]`, column-sharded, which supports the first statement; `_pool_and_norm` then keeps the first 4096 columns of the host concatenation, which is correct for a sharded output and also for a replicated one, so fidelity cannot tell them apart. Either complete the stage-5 audit before the release stage or mark the `p150x4` profile experimental in the card.
- Head placement is asserted, not measured. `doc/optimized_full_model/README.md` says "Decide head placement (host vs device) by measurement" was satisfied by "warm cache-hit answers take 0.1 ms server-side", which is the cache path, not the head path. The conclusion (host fp32 heads) is almost certainly right for 9.4 M-parameter heads over a few vectors, but no measurement artifact exists.
- A failed check artifact sits unlabeled in the stage directory. `doc/optimized_full_model/replay_trace_check_accuracy.json` is a `pass: false` run with 176 flagged buffers from generation B. It is useful history, but the README only says "earlier runs ... are kept in logs"; label it as superseded in the README or move it under a `history/` name.
- Qualitative substitute measured on the wrong path and policy. The README-example comparison cited by stage 6 comes from `doc/probe/probe_full_encoder.json` (probe code, accuracy policy, bf16 HF control, `max_seq_len 1024`). Recomputed from the stage vectors against the fp32 reference: `bfp8_attn` gives noul 0.873 (reference 0.842), billing 0.993 (0.988), frustration 2.000 (2.000), tides Moon 0.995 (0.993); `accuracy` gives 0.852 / 0.991 / 2.000 / 0.992. Argmax identical in all cases. Record this for the selected policy on the final path.
- Harness and server truncate differently. `tests/run_fidelity.py` uses the training recipe (cap 2047, head truncation for candidates); `TtQwen3Encoder.tokenize` keeps the last 2048 tokens for every role (vLLM semantics). Documented in `/home/hous/dev/clm-v0.1-8B/reference/README.md` item 5 but not in the stage READMEs. Only inputs above 2047 tokens are affected.
- Readback size understated. The README says "readback of up to 8 MB"; that is the 128 x 8 variant. The 2048 x 8 variant reads back 8 x 2048 x 4096 x 2 bytes = 134 MB per call. It is under 1 % of that variant's 2290 ms, so it is not a performance problem, but the number should be right.
- `_pool_and_norm` in `tt/encoder.py` lines 265 to 268 has two branches that do the same thing (`rows[:, :dim]`); the first condition is dead logic. Cosmetic.
- `bench_accuracy.json` and `bench_accuracy_hostnorm.json` are byte-identical (same 22:17:21 timestamp). Keep one or say they are the same run.

## Hard-Check Gaps

- No profiler or `tt-perf-report` rows exist for the selected policy or for the full 36-layer traced path. Policy consumption is shown only by code inspection (the stock `DecodersPrecision` plumbing) and by the fidelity and latency deltas between policies. A reduced-layer profile under `bfp8_attn` would close this cheaply.
- No watcher-clean run or runtime fallback audit for stages 6 to 8 (stage 1 has watcher logs). The prompts for stages 4 to 7 ask for one.
- The selected policy's full-corpus fidelity on the final code path is missing (96 of 308 texts only, no 2048-token text). See the P2 on mixed generations.
- The decision-agreement artifact is missing (P1).
- No qualitative-substitute artifact for the selected policy on the final path (numbers above are from this review).
- Stage 4: no `tt-perf-report`, no CCL audit, no 1x4 vs 1x1 TTNN comparison artifact (optional stage, but the profile ships).
- The fidelity vectors (`*_tt_single.npy`, `*_tt_batched.npy`) are gitignored. Every re-derivation in this review depends on local files. Consider committing the 5 MB vectors for the selected and accuracy policies, or at least their SHA-256 in the JSON.
- The stage-6 README's headline table (0.99909 / 0.99588 / 0.99429) is backed only by `/home/hous/dev/clm-v0.1-8B/logs/fidelity_accuracy.log` line 1700, not by any JSON on disk.

## Anomaly Ledger

- Observed anomaly: the same text embedded alone and inside a batch differs by up to 1 minus cosine = 3.6e-3 (accuracy) / 3.7e-3 (bfp8_attn); 60 of 308 texts below 0.999.
  Evidence: recomputed from `fidelity_*_tt_single.npy` vs `fidelity_*_tt_batched.npy`; by variant on 1x1: (128, batch 8) mean 7.7e-4, max 3.6e-3; (1024, batch 8) mean 5.1e-4, max 1.4e-3; (128, batch 4) 3e-4; (2048, batch 4) 1e-7 in generations A and C, 2.4e-4 in generation B.
  Affected path: `TtQwen3Encoder.embed_ids` batched traced prefill on P150.
  Control or comparison: no dependence on batch position (position 0: 7.4e-4, position 7: 6.5e-4; cross-user causal leakage would spare position 0); exact agreement (<= 2.4e-7) for the (1024, batch 8) variant on 1x4 and for (2048, batch 4) on 1x1 in two generations, so the effect is shape-specific, not a masking defect; replay check shows identical output on repeated replay (min cosine 0.9999997) and cosine 0.77 to 0.99 between different inputs; a leakage probe (|cos| of the batched-minus-single delta against centered in-batch neighbours vs same-bucket controls) gives 0.100 vs 0.096, within the confound of adjacent corpus texts sharing one state.
  Likely subsystem: prefill matmul and SDPA program configs that depend on the flattened token count B x S (tt-metal issue 47238, cited in `model_config.py` line 734).
  Investigation performed: this review's recomputation; the stage attributed it in prose and reinterpreted the gate as a mean.
  Resolution: controlled as far as cross-request leakage is concerned; more-work-needed for the gate and the serving-determinism decision (P2).

- Observed anomaly: decisions reverse against the fp32 reference at state cosine 0.9993 to 0.9994 (for example `invoice_processing_000096/discrepancy_severity`, `invoice_processing_000063/discrepancy_severity`), under every policy including `accuracy`.
  Evidence: table and examples under P1; logit shifts of 0.6 to 1.1 against reference margins of 0.3 to 0.7 logits.
  Affected path: `TtQwen3Encoder` output through the CLM heads (host fp32), all policies and both meshes.
  Control or comparison: the CPU bf16 control in `/home/hous/dev/clm-v0.1-8B/reference/bf16_sensitivity.json` is at cosine 0.99998 vs fp32 over 40 texts, about 50 times closer than the TT output, so the reversals are a property of the TT precision policy, not of bf16 arithmetic alone; the `performance` policy shows the same mechanism at larger scale (33 material reversals).
  Likely subsystem: BFP8 weights and HiFi2 fp16-accumulate MLP math in the encoder stack, amplified by the 100 x cosine head scoring on a dataset with small decision margins.
  Investigation performed: this review's offline scoring; not measured by the stage.
  Resolution: more-work-needed (P1).

- Observed anomaly: fidelity JSON and vectors disagree in `doc/full_model/` (JSON from the 22:07 run, `.npy` from the 21:51 run), and two of the sweep's fidelity rows were produced by code generations the trace tracker later flagged (43 and 176 corruptible buffers).
  Evidence: recomputed means 0.999094 (vectors) vs 0.999103 (JSON); `replay_trace_check.log` (22:01) and `replay_trace_check_accuracy.json` (22:11).
  Affected path: evidence integrity for stages 6 and 8.
  Control or comparison: generation A vs B accuracy vectors cosine 0.999994 / 0.999989; generation C vs A `bfp8_attn` vectors on 96 texts 0.999994 / 0.999988. No corruption is visible in the numbers.
  Likely subsystem: artifact handling (output path reuse across runs), not the model.
  Investigation performed: timestamps, log markers, call-count arithmetic, vector recomputation.
  Resolution: more-work-needed (P2, artifact regeneration); the numerical risk is controlled.

- Observed anomaly: the 1x2 mesh (P300) does not open: "Fabric Router Sync: Timeout after 10000 ms on Device 0".
  Evidence: `/home/hous/dev/clm-v0.1-8B/logs/fidelity_accuracy_1x2.log`, `bench_accuracy_1x2.log` (22:02 to 22:06).
  Affected path: optional 1x2 TP profile.
  Control or comparison: 1x4 opens and passes fidelity on the same boards; the README notes earlier Ethernet-core timeouts on this box.
  Likely subsystem: fabric or control-plane configuration for a sub-mesh of the discovered 1x4 system mesh, or a physical link.
  Investigation performed: documented and not pursued (optional per PLAN row 4).
  Resolution: controlled (out of scope by plan; no P300 profile is published).

- Observed anomaly: stage-7 layer-stack "lower bound" (57.92 ms) exceeds the measured end-to-end latency (57.64 ms).
  Evidence: `perf_summary_accuracy.json` rows for 32 and 128 tokens, `gap_to_lower_bound_pct` minus 0.4 / minus 0.5.
  Affected path: performance reconciliation claim only.
  Control or comparison: none; the layer-0 profile was taken under Tracy in stage 1, not from the traced 36-layer run.
  Likely subsystem: profiler overhead or non-traced per-op dispatch in the layer-0 measurement.
  Investigation performed: arithmetic check.
  Resolution: more-work-needed in documentation only (Other Concerns); no runtime defect is implied.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/clm-v0.1-8B/PLAN.md` (section 4 rows 4 to 8, sections 1, 6, 9); `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/{stage-review,multichip,full-model,optimize,datatype-sweep,tt-enable-tracing,qualitative-check}/SKILL.md`; `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/prompts/model_bringup_multigoal/04-multichip-decoder.txt` through `08-datatype-sweep.txt`.
- Artifact paths (all under `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/`): `doc/context_contract.json`; `doc/full_model/README.md`, `fidelity_accuracy.json`, `fidelity_accuracy_tt_single.npy`, `fidelity_accuracy_tt_batched.npy`; `doc/optimized_full_model/README.md`, `perf_summary.json`, `perf_summary_accuracy.json`, `bench_accuracy.json`, `bench_accuracy_hostnorm.json`, `fidelity_accuracy_hostnorm.json` and its two `.npy`, `replay_trace_check_bfp8_attn.json` (committed and working-tree versions), `replay_trace_check_accuracy.json`; `doc/datatype_sweep/README.md`, `sweep_results.json`, `sweep_results.csv`, `selected_precision_config.json`, `fidelity_{bfp8_attn,bfp8_attn_hifi2,bfp8_attn_twophase_limit96,performance}.json` and their `.npy`, `bench_{bfp8_attn,bfp8_attn_hifi2,performance}.json`; `doc/multichip_decoder/README.md`, `fidelity_accuracy_1x4.json` and `.npy`, `bench_accuracy_1x4.json`; `doc/probe/README.md`, `probe_full_encoder.json`; `doc/review/REVIEW_PROMPT_TEMPLATE.md`; `tt-model.yaml`; `.gitignore`. Reference data: `/home/hous/dev/clm-v0.1-8B/reference/{README.md,fidelity_corpus.json,hf_embeddings.npy,hf_embeddings_meta.json,typed_decisions_subset.json,typed_decisions_subset_reference.json,readme_examples_reference.json}`; `/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt`; `/home/hous/dev/clm-v0.1-8B/STATUS.md`. Logs: `/home/hous/dev/clm-v0.1-8B/logs/fidelity_*.log`, `bench_*.log`, `replay_trace_check*.log` (headers, encoder load and warmup lines, result lines). Upstream: `/tmp/claude-1002/-home-hous-dev-clm-v0-1-8B/0082bb62-0b82-4079-9fa4-87d59d66a985/scratchpad/clm-repo/README.md` and `src/clm/server.py`.
- Code paths: `tt/encoder.py`, `tt/heads.py`, `clm/heads.py`, `clm/embedder.py`, `tests/run_fidelity.py`, `tests/bench_encoder.py`, `tests/replay_trace_check.py`, `tests/datatype_sweep.py`, `tests/perf_summary.py` (all under the autoport); `/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/model_config.py` (lines 73 to 102, 213 to 420, 730 to 745, 4852 to 4943), `common.py` (lines 730 to 755), `generator.py` (lines 583 to 680, 784 to 845, 1057 to 1110), `model.py` (lines 245 to 290, 406 to 520, 1010 to 1033), `attention.py` (lines 1024 to 1290), `mlp.py` (lines 176, 340). Git: `git log`, `git show --stat add82f9e7d`, `git show 498325e8ed -- tt/encoder.py`, `git show fe0b69f03e -- tt/encoder.py`, `git diff` of the replay JSON, `git ls-files`, `git status`.
- Commands run: read-only `cat`, `grep`, `sed`, `ls`, `stat`, `git` queries; three analysis scripts with `/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python` over the `.npy`, JSON and checkpoint files (cosine recomputation per policy and per batch position, leakage probe, 1x4 vs 1x1 comparison, decision scoring through the CLM heads). No device was opened, no `ttnn` import, no server, no test run, no implementation file modified.

## Residual Risk

- Decision stability is bounded by the model, not only by the port. Eight of 200 subset decisions have a reference margin below 0.05 and will flip under any non-bit-exact encoder; a plain 98 % argmax gate may be unreachable even at bf16 fidelity. The gate must be defined with that in mind before more precision work is spent.
- The served path (batched prefill) is both the least faithful column (93.5 % plain agreement for the selected policy) and nondeterministic across batch compositions (4 to 5 of 200 decisions change). Until a decision is recorded, two identical requests can return different answers.
- The 2048-token limit is fixed by trace lengths; a user who raises `CLM_MAX_TOKENS` triggers a runtime trace capture, which the stage's own safety rule forbids. Upstream treats 2048 as configurable.
- The `p150x4` profile ships without the stage-5 audit, with a layout description that contradicts itself, and with 1x4 vs 1x1 TTNN agreement (min 0.99547) that was never evaluated against the plan's gate.
- All fidelity re-derivation depends on gitignored local vectors. If the box is re-imaged, only the JSON summaries survive, and one of them does not match its vectors today.
- The worktree is live. HEAD moved from `fe0b69f03e` to `add82f9e7d` during this review; the findings were checked against the on-disk state at `add82f9e7d` between 22:38 and 22:48 UTC.

## Appendix: reproducing the decision-agreement table

Run with `/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python`. It needs no device. Paths are the ones used in this review.

```python
import json, numpy as np, torch
REF = "/home/hous/dev/clm-v0.1-8B/reference"
AP = "/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b"
ck = torch.load("/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt", map_location="cpu", weights_only=False)
scale = min(100.0, float(torch.as_tensor(ck["logit_scale"]).float().exp()))

def head(sd, x):
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    h = torch.nn.functional.gelu(x @ sd["inp.weight"].T + sd["inp.bias"])
    h = h @ sd["hidden.0.weight"].T + sd["hidden.0.bias"]
    h = torch.nn.functional.layer_norm(h, (h.shape[-1],), sd["norms.0.weight"], sd["norms.0.bias"])
    h = torch.nn.functional.gelu(h)
    return torch.nn.functional.normalize(h @ sd["out.weight"].T + sd["out.bias"], dim=-1).numpy()

corpus = json.load(open(f"{REF}/fidelity_corpus.json"))
idx = {c["id"]: i for i, c in enumerate(corpus)}
l2 = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
ref = l2(np.load(f"{REF}/hf_embeddings.npy").astype(np.float32))
sub = json.load(open(f"{REF}/typed_decisions_subset.json"))
dec = [(p["state_id"], p["candidate_ids"]) for c in sub["cases"] for p in c["pairs"].values()]

def probs(E):
    out = []
    for s, cands in dec:
        zs = head(ck["state_head"], E[idx[s]][None])
        za = head(ck["action_head"], E[[idx[c] for c in cands]])
        lg = scale * (za @ zs[0]); lg -= lg.max(); p = np.exp(lg); out.append(p / p.sum())
    return out

rp = probs(ref); ra = np.array([p.argmax() for p in rp])
margin = np.array([np.sort(p)[-1] - np.sort(p)[-2] for p in rp]); M = margin >= 0.10
for base in ["doc/optimized_full_model/fidelity_accuracy_hostnorm", "doc/datatype_sweep/fidelity_bfp8_attn",
             "doc/datatype_sweep/fidelity_bfp8_attn_hifi2", "doc/datatype_sweep/fidelity_performance",
             "doc/multichip_decoder/fidelity_accuracy_1x4"]:
    for mode in ("single", "batched"):
        pp = probs(np.load(f"{AP}/{base}_tt_{mode}.npy"))
        a = np.array([p.argmax() for p in pp]); ag = a == ra
        tv = np.array([0.5 * np.abs(p - r).sum() for p, r in zip(pp, rp)])
        print(base, mode, f"{ag.sum()}/200", f"{(ag & M).sum()}/{M.sum()}", f"meanTV {tv.mean():.4f} maxTV {tv.max():.3f}")
```
