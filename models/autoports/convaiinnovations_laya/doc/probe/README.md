# Probe, Track R: CPU fp32 reference, corpus, E0 vendor equivalence, outlier report

Date: 2026 Oct 5. Host `qb2-120-p11t01` (AMD Ryzen 7 9700X, 8 cores / 16 hardware threads, AVX-512). tt-metal venv: Python 3.10.21, torch 2.11.0+cpu, transformers 5.12.1. Eval venv: Python 3.12.3, torch 2.14.1+cpu, transformers 5.18.0, pip laya 0.3.27. No device was opened by this track.

Acronyms: HF = Hugging Face. PCC = Pearson correlation coefficient. CLS, SEP, MASK, PAD = the tokenizer's special tokens. RoPE = rotary position embedding. GeGLU = gated GELU feed-forward. SDPA = scaled dot-product attention (the fused torch kernel). AST = abstract syntax tree.

## What exists now

| path | content |
|---|---|
| `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/convaiinnovations_laya/common.py` | pins (`HF_MODEL`, `LAYA_REVISION`), `model_dir()`, `load_config()`, `rope_theta()`, `sliding_window_half()`, `load_tokenizer()`, `load_state_dict()`, `load_torch_model()` (HF `ModernBertModel` with the Laya encoder weights, eager by default), typed-decisions loader, `gate_subset()`, `case_items()`, `collate()`, `build_batch()`, `build_inputs()` |
| `.../reference/laya_reference.py` | `LayaReference` (vendored `DecisionModel`, strict safetensors load, fp16 to fp32, eager attention, `forward`, `forward_with_hidden`, `forward_explicit_head`, `shapes()`, `encode`, `collate`, `decide`), `head_layer_explicit`, `scorer_explicit`, host post-processing functions (`temperature_for`, `scaled_probs`, `confidence_from_probs`, `answer_confidence`, `expected_score`, `p_true`, `decode_answer`, `decode_items`) |
| `.../reference/corpus.py` | typed-decisions and `parity_fast.py` corpus builder; `python -m models.autoports.convaiinnovations_laya.reference.corpus` writes the npz, the index and the typed-decisions CPU outputs |
| `.../tests/test_reference_parity.py` | pytest: config pins, strict key map, HF versus from-scratch reference, eager versus sdpa, six negative controls, LayaReference bit-for-bit, explicit head, host post-processing mirror, fp32 padding invariance |
| `.../tests/outlier_report.py` | per-layer fp32 outlier report writer |
| `.../doc/context_contract.json` | 512 / 192 contract, sibling 1024 / 256, bucket lists from Appendix A.5 |
| `.../doc/probe/outliers.json` | outlier report (section 6) |
| `/home/hous/dev/laya/reference/parity_corpus.npz`, `parity_corpus_index.json` | fp32 reference corpus (section 4) |
| `/home/hous/dev/laya/reference/typed_decisions_cpu/answers.jsonl`, `summary.json` | CPU fp32 outputs for the 40 gate cases |
| `/home/hous/dev/laya/evals/equivalence/e0_vendor_side.py`, `e0_pip_side.py`, `e0_compare.py`, `e0_cases.json`, `e0_vendor.json`, `e0_pip.json`, `e0_results.json`, `E0.md` | E0 (section 5) |

## 1. Checkpoint facts (verified from the files)

- `model.safetensors` (843 MB, sha256 recorded in `parity_corpus_index.json`): 206 tensors. 205 are **float16** and one (`temperature`, shape [3]) is float32. PLAN.md section 1 says bf16; that is wrong. The reference converts fp16 to fp32 exactly.
- Key groups: `encoder.*` 170 tensors (`embeddings.tok_embeddings.weight` [50368, 1024], `embeddings.norm.weight`, `final_norm.weight`, per layer `attn.Wqkv.weight` [3072, 1024], `attn.Wo.weight` [1024, 1024], `mlp.Wi.weight` [5248, 1024], `mlp.Wo.weight` [1024, 2624], `mlp_norm.weight`, and `attn_norm.weight` for layers 1 to 27 only); `head.layers.{0,1}.*` 24 tensors (`self_attn.in_proj_weight` [3072, 1024], `self_attn.in_proj_bias` [3072], `self_attn.out_proj.weight` [1024, 1024] and bias, `linear1.weight` [4096, 1024] and bias, `linear2.weight` [1024, 4096] and bias, `norm1`, `norm2` weight and bias); `scorer.0` (LayerNorm weight and bias), `scorer.1` (Linear 1024 to 1024 with bias), `scorer.3` (Linear 1024 to 1 with bias); `act_head.0` (Linear 1028 to 256 with bias), `act_head.2` (Linear 256 to 2 with bias); `type_emb.weight` [3, 1024]; `temperature` [3].
- Encoder config (transformers 5.12.1 `ModernBertConfig` from `encoder/config.json`): hidden 1024, 28 layers, 16 heads (head_dim 64), intermediate 2624, vocab 50368, `norm_eps` 1e-5, no norm, attention or MLP biases, `local_attention` 128, `sliding_window` 64, `layer_types` full at layers 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, `rope_parameters = {"full_attention": {"rope_theta": 160000.0}, "sliding_attention": {"rope_theta": 10000.0}}`, pad 50283, cls 50281, sep 50282. The tokenizer's mask id is 50284. Field names: the thetas live only in `rope_parameters`; `global_rope_theta` and `local_rope_theta` do not exist on this config.
- `rl_agent_config.json`: `max_len` 512, `head_max_len` 192, `head_layers` 2, `temperature` [1.6369, 1.2514, 1.9834] (choice, score, noul), `temperature_by_options` {choice:2 1.9064, choice:3-5 1.7602, choice:6-10 1.0000, choice:11+ 0.1006, score:3-5 1.2514, noul:2 1.9834}. The sibling `laya-typed-decisions` has `max_len` 1024, `head_max_len` 256, the same `temperature_by_options` and per-type temperatures [1.0148, 1.0374, 1.0575].
- Hub attention masks in transformers 5.12.1: `ModernBertModel.forward` builds a dict of two masks with `create_bidirectional_mask` and `create_bidirectional_sliding_window_mask`; the sliding overlay is `abs(q - kv) <= config.sliding_window` with `sliding_window` 64, so the band is 129 wide (`+/-64`). The upstream from-scratch reference uses `local_attention // 2` and matches (section 7).

## 2. Vendored files

`vendor/rl_common.py`, `vendor/rl_agent_api.py` and `vendor/email_utils.py` are AST-identical to the Hub originals at revision `7b928d82` after removing two unused imports that the vendored copies lack (`import torch.nn.functional as F` in rl_common.py, `import math` in rl_agent_api.py); the rest of the byte difference is formatting (the repository's formatter hook). `vendor/rl_agent_config.json` differs only by a trailing newline. The sha256 values in `VENDORED.md` are those of the Hub originals and match the snapshot files.

`rl_agent_api.py` imports `rl_common` as a top-level module. `common.vendor_module("rl_agent_api")` inserts the vendor directory into `sys.path` and imports it; `common.to_internal` is `RLAgent._to_internal` (a staticmethod, no model load).

## 3. Reference model and backend contract

`LayaReference` builds `rl_common.DecisionModel(AutoModel.from_config(encoder_config, attn_implementation="eager"), head_layers=2, n_act=2)`, loads all 206 tensors with an explicit missing / unexpected / shape check followed by `load_state_dict(strict=True)`, casts to fp32, sets `requires_grad=False`, and runs every forward under `torch.inference_mode()` with `torch.set_num_threads(common.DEFAULT_THREADS)`, 6 by default after the orchestrator capped per-track CPU use (`LAYA_CPU_THREADS` overrides).

Contract (`shapes()` reports it): `forward(input_ids int64 [B,S], attention_mask int64 [B,S], marker_pos int64 [B,kmax], marker_mask bool [B,kmax], qtype int64 [B]) -> (logits float32 [B,kmax], act_logits float32 [B,2])`. Logits are `-1e4` where `marker_mask` is False. `act_logits` are pre-softmax; `act_probability = softmax(act_logits)[0]`.

Measured on this host (16 threads, box otherwise idle): load 9.5 s; one forward of 5 rows x 512 tokens 10.0 s; two calls bit-identical.

Head layer math (what the device must reproduce; `head_layer_explicit` is the unit-test oracle): `h = x + out_proj(softmax(q k^T / 8 + keypad) v)` with `q, k, v = split(in_proj(LN1(x)))`, 16 heads of 64, the key padding mask only (no causal, no RoPE, no band); then `h = h + linear2(relu(linear1(LN2(h))))`. LayerNorm eps 1e-5 with bias. Scorer: `Linear(1024,1) (GELU(erf) (Linear(1024,1024) (LN(h)))))` read at the marker positions. The vendored module uses the fused `torch._transformer_encoder_layer_fwd` path in eval; it does not zero padded rows (probe on a toy layer and on the full model); the fused path differs from the explicit math by 2.4e-7 on the logits and 4.9e-4 on the act logits (full model, 5 x 512).

Reference caveat (Appendix A.4): `nn.TransformerEncoder`'s nested-tensor path is never entered because `DecisionModel` iterates `self.head.layers` itself; the only fast-path effect is the fused kernel above. Attaching forward hooks to a head layer's submodules switches that layer to the explicit module path (torch rule), which changes its output by about 1e-7; the outlier report records the resulting logit delta.

Equalities proven (`tests/test_reference_parity.py`, numbers from the smoke run; the pytest log is listed in section 8):

| comparison | result |
|---|---|
| `LayaReference(attn="sdpa").forward` versus the vendored `RLAgent.model` (built by `rl_common.build_model`, sdpa) on a padded 5 x 512 batch | logits and act logits bit-identical (`torch.equal`) |
| `LayaReference(attn="eager")` versus `RLAgent.model` (sdpa) | max abs logit delta 3.7e-6, act logits 1.7e-3 |
| `forward_with_hidden` versus `forward` | bit-identical |
| `forward_explicit_head` versus `forward` | logits 2.4e-7, act logits 4.9e-4 |
| natural length (181) versus padded to 512, same 5 rows | logits 4.3e-6 (the fp32 noise floor for the stage 6 alone-versus-in-batch gate) |
| host decode mirror (`decode_items(shape="hub")`) versus `RLAgent.system_one` answers | identical dicts |

## 4. Corpus (`/home/hous/dev/laya/reference/parity_corpus.npz`)

Gate subset: the typed-decisions test parquet (`/home/hous/dev/laya/evals/typed_decisions/data/test-00000-of-00001.parquet`, 400 cases, 5 questions each) is stored in four blocks of 100 by workflow, so "the first 40 cases" would be one workflow only. The subset is instead 10 cases per workflow drawn with `numpy.random.RandomState(13).choice(..., replace=False)` (seed 13 is the authors' benchmark seed), sorted by index: [9, 14, 31, 37, 43, 44, 57, 62, 69, 83, 100, 103, 123, 157, 164, 180, 181, 187, 195, 196, 208, 209, 230, 259, 262, 268, 276, 287, 289, 297, 304, 307, 321, 340, 364, 366, 369, 370, 384, 393]. 200 decisions: choice 4 options (40), choice 5 (20), score 4 (70), score 5 (10), noul (60). Token lengths at 512 / 192: min 130, mean 301, max 510. This is a plan deviation (the plan said "the first 40 cases"); the case ids are in the index JSON, and E0 and E1 use the same list.

parity_fast states: `TEXTS` and `PRESETS` are executed from the AST of `/home/hous/dev/laya/evals/vendor/laya/benchmarks/parity_fast.py` with the presets module of the clone, and `states()` is replicated: 5 presets x 12 texts = 60 states with the preset's questions (triage 5, moderation 5, guard 5, router 4, email 5) = **288 questions**, not "about 480" as the plan says; 200 + 288 = 488 items in total, which is the figure the plan most likely meant. These add the `choice:6-10` temperature bucket (6-option intents) that typed-decisions lacks. `choice:2` and `choice:11+` are not covered by either corpus.

Protocol: one forward per case (its 5 questions) or per state (4 to 5 questions), rows padded to the longest row of the call, `max_len` 512, `head_max_len` 192, fp32, eager attention, 16 threads. The npz holds `input_ids` [N,512] int32 (-1 beyond the real length), `attention_mask` [N,512] int8, `marker_pos` and `marker_mask` [N,kmax], `qtype`, `k`, `seq_len`, `source` (0 typed-decisions, 1 parity_fast), `group_index`, `temperature`, `logits_fp32` [N,kmax] (-1e4 beyond k), `act_logits_fp32` [N,2], `act_probs`, `probs` (shipped temperatures), `probs_raw` (temperature 1, the `p_fp32` of parity_fast), `gold_idx`. `parity_corpus_index.json` holds the per-item metadata (case id, workflow, qid, type, option keys, temperature bucket, gold, both decoded answer shapes, markers, timings) and the pins (revision, safetensors sha256, parquet sha256, library versions). `typed_decisions_cpu/answers.jsonl` holds one line per gate case with the pip-shaped answers, logits, probabilities and gold.

Run numbers (`/home/hous/dev/laya/logs/p1_corpus_20261005T211106Z.log`, 16 threads, while three other CPU jobs ran, load average about 30 to 37):

| item | value |
|---|---|
| model load | 12.4 s |
| typed-decisions, 40 calls x 5 rows | 702 s (median 10.8 s per call under contention; 10.0 s for 5 x 512 on an idle box) |
| parity_fast, 60 calls x 4 to 5 rows | 609 s |
| items | 488 (200 + 288), kmax 6, `parity_corpus.npz` 52 KB, index 760 KB |
| typed-decisions token lengths | min 130, mean 301, max 510; call padding 176 to 510 |
| parity_fast token lengths | min 62, mean 152, max 424; call padding 83 to 424 |
| question shapes | typed: score 80, choice 60, noul 60 (k 4: 110, k 2: 60, k 5: 30); parity_fast: noul 180, score 60, choice 48 (k 2: 180, k 6: 48, k 4: 48, k 3: 12) |
| temperature buckets | noul:2 240, score:3-5 140, choice:3-5 60, choice:6-10 48 |
| scorer logits (valid markers) | finite, range -5.46 to 20.73; masked entries exactly -1e4 |
| act logits | range -3980 to 4862; `act_probability` is exactly 1.0 on all 488 items (the act head is saturated; the authors' issue #185 says it carries no usable signal) |
| confident decisions (top-1 minus top-2 >= 0.10 after temperature) | 403 of 488; 149 of the 200 gate decisions (the population of the stage 6 and stage 8 confident-agreement gate) |
| max probability deciles (10 / 50 / 90) | 0.384 / 0.650 / 0.978 |
| gate subset sanity count | argmax equals the gold label on 74 of 200 decisions (0.37; the published typed-decisions accuracy of this checkpoint is 0.362; the E2 metrics proper come from the evals harness) |

Pins recorded in the index: `model.safetensors` sha256 `891102d372688fc2...`, parquet sha256 `4f294f218ea1da27...`, torch 2.11.0+cpu, transformers 5.12.1, numpy 1.26.4.

## 5. E0 vendor equivalence

Full write-up: `/home/hous/dev/laya/evals/equivalence/E0.md`; data `e0_results.json`. Inputs: the 40 gate cases and the 60 parity_fast states (100 calls, 488 questions); the vendored side writes `e0_cases.json` and the pip side reads it, so both see byte-identical inputs. Vendored side: `RLAgent.system_one` (Hub `rl_agent_api.py` + `rl_common.py`, sdpa, fp32, torch 2.11.0, transformers 5.12.1). pip side: `laya.load(<snapshot>, device="cpu").system_one` (laya 0.3.27, sdpa, fp32, autocast off, torch 2.14.1, transformers 5.18.0). A forward hook on each agent's `model` captures the exact tensors each `system_one` used.

Result (`p1_e0_vendor_*.log`, `p1_e0_pip_*.log`, `p1_e0_compare_*.log`; both sides at 6 threads): **PASS** on all six gates.

| type | n | ids identical | markers identical | max abs logit delta | max abs act logit delta | JSON identical (common fields, 4 dp) | argmax agree | pip-shape mirror identical | Hub-shape mirror identical |
|---|---|---|---|---|---|---|---|---|---|
| choice | 108 | 108 | 108 | 0.0 | 0.0 | 108 | 108 | 108 | 108 |
| score | 140 | 140 | 140 | 0.0 | 0.0 | 140 | 140 | 140 | 140 |
| noul | 240 | 240 | 240 | 0.0 | 0.0 | 240 | 240 | 240 | 240 |
| all | 488 | 488 | 488 | 0.0 | 0.0 | 488 | 488 | 488 | 488 |

`usage.input_tokens` identical on 100 of 100 states; no rounding-boundary flips. The logits are bit-identical between torch 2.11.0 / transformers 5.12.1 and torch 2.14.1 / transformers 5.18.0 when both run with the same thread count (6); the 4-state smoke run with mixed thread counts (8 and 8 on a loaded box) showed a 6.6e-6 max delta, which is the reduction-order noise, not a modelling difference. Timings: vendored side 194 s, pip side 166 s for 100 calls (load 7 s and 9 s).

Temperature rule (plan amendment A6): pip laya clamps every temperature to [0.5, 5.0] at load (`clamp_temperature`; bool, non-numeric, NaN and inf give 1.0); the Hub `rl_agent_api.py` divides by the raw value. For the shipped tables the two rules differ only in the `choice:11+` bucket (0.1006 raw, 0.5 clamped), which no corpus question uses, so every stored probability is the same under both rules. `temperature_for`, `decode_answer`, `decode_items` and `LayaReference.decide` take `clamp` (default True, the pip rule); `clamp=False` is the Hub rule. The corpus stores `probs` and `answer_pip` under the clamped rule and `probs_hub` and `answer_hub` under the raw rule (`temperature` and `temperature_raw` arrays); `typed_decisions_cpu/answers.jsonl` stores `answers` / `probs` (clamped) and `answers_hub` / `probs_hub` (raw). Logits are unaffected.

pip laya 0.3.27 is byte-identical to the pinned clone `8a6e1328` for `common.py`, `presets.py`, `agent.py`, `confidence.py`. The pip `build_sequence` is restructured (`build_head`, tokenizer-side truncation to 48 option tokens, per-call token cache) but produced identical ids and markers on every corpus sequence, so the Hub `rl_common.py` stays the vendored sequence builder (no plan amendment needed for sequences). pip-only answer fields: `answer_confidence` (= max p, clipped to [0, 1]) on every answer, `confidence` = max(p_true, 1 - p_true) on noul, `action.act_probability` rounded to 4 decimals (Hub: `rl_agent.act_probability`, unrounded), and the score `legend` rendered through `render_criterion` (equal to `str` for string levels). `reference/laya_reference.py: decode_answer(..., shape="pip")` reproduces the pip answer dict exactly from the vendored logits; the server (Track S) can import it.

## 6. Per-layer fp32 outlier report (`doc/probe/outliers.json`)

Inputs: the 8 longest gate sequences from distinct cases (397 to 510 tokens: customer_service 000087, 000081, 000064, 000095, 000003, 000080, 000096 and invoice_processing 000076), one batch of 8 padded to 512, fp32, eager. Statistics are over real token positions only, all 8 sequences pooled. Max abs values come from forward hooks on every `nn.LayerNorm` (input and output); residual-stream values come from `forward_with_hidden`. The hooks switch the two head layers to the explicit module path, which moved the logits by 9.5e-7 and the act logits by 4.9e-4 relative to the plain forward (recorded in the JSON).

| layer | type | pre attn_norm max abs | post attn_norm | pre mlp_norm | post mlp_norm | residual out max abs | residual mean token L2 | top channel |
|---|---|---|---|---|---|---|---|---|
| 0 | full | n/a (Identity) | n/a | 13.5 | 5.25 | 49.1 | 30.5 | 379 |
| 1 | sliding | 49.1 | 3.74 | 48.4 | 1.26 | 83.6 | 39.6 | 379 |
| 2 | sliding | 83.6 | 3.63 | 84.5 | 1.41 | 81.8 | 47.7 | 379 |
| 3 | full | 81.8 | 3.44 | 80.9 | 1.47 | 74.7 | 53.4 | 379 |
| 4 | sliding | 74.7 | 3.43 | 72.9 | 1.51 | 75.9 | 59.5 | 379 |
| 5 | sliding | 75.9 | 4.90 | 76.4 | 2.03 | 1404.0 | 75.7 | 379 |
| 6 | full | 1404.0 | 5.03 | 1404.8 | 3.96 | 1522.8 | 87.9 | 379 |
| 7 | sliding | 1522.8 | 3.93 | 1521.2 | 3.70 | 2281.2 | 111.6 | 379 |
| 8 | sliding | 2281.2 | 4.94 | 2281.4 | 3.99 | 2395.1 | 120.8 | 379 |
| 9 | full | 2395.1 | 7.77 | 2394.0 | 4.21 | 2456.4 | 132.8 | 379 |
| 10 | sliding | 2456.4 | 5.33 | 2455.5 | 3.61 | 2520.2 | 138.2 | 379 |
| 11 | sliding | 2520.2 | 6.25 | 2520.9 | 3.54 | 2512.0 | 141.8 | 379 |
| 12 | full | 2512.0 | 8.26 | 2510.4 | 4.61 | 4006.9 | 211.7 | 379 |
| 13 | sliding | 4006.9 | 6.28 | 4006.9 | 9.37 | 3984.5 | 224.9 | 379 |
| 14 | sliding | 3984.5 | 6.79 | 3979.0 | 11.53 | 11696.3 | 352.9 | 382 |
| 15 | full | 11696.3 | 8.50 | 11698.7 | 12.15 | 11700.8 | 375.2 | 382 |
| 16 | sliding | 11700.8 | 7.46 | 11701.0 | 13.02 | 11696.8 | 390.3 | 382 |
| 17 | sliding | 11696.8 | 8.19 | 11696.9 | 13.58 | 11695.3 | 397.0 | 382 |
| 18 | full | 11695.3 | 10.21 | 11696.0 | 15.41 | 11690.2 | 518.8 | 382 |
| 19 | sliding | 11690.2 | 7.35 | 11690.3 | 22.70 | 33109.1 | 1654.6 | 379 |
| 20 | sliding | 33109.1 | 15.93 | 33111.8 | 19.56 | 33112.7 | 1665.6 | 379 |
| 21 | full | 33112.7 | 12.64 | 33118.4 | 19.72 | 33115.8 | 1705.1 | 379 |
| 22 | sliding | 33115.8 | 20.31 | 33124.1 | 20.04 | 33127.3 | 1761.0 | 379 |
| 23 | sliding | 33127.3 | 15.14 | 33127.9 | 20.31 | 33130.6 | 1818.8 | 379 |
| 24 | full | 33130.6 | 10.84 | 33131.0 | 20.71 | 33134.9 | 1877.4 | 379 |
| 25 | sliding | 33134.9 | 18.11 | 33137.8 | 23.06 | 33138.7 | 1906.7 | 379 |
| 26 | sliding | 33138.7 | 17.10 | 33144.2 | 21.11 | 33143.6 | 1958.5 | 379 |
| 27 | full | 33143.6 | 17.70 | 33152.1 | 19.56 | 35.7 (after final_norm) | 34.7 | 382 |

Other norms (input max abs, output max abs, input mean token L2, top input channel): `encoder.embeddings.norm` 1.9 / 13.19 / 2.2 / 963; `encoder.final_norm` 33144.6 / 35.72 / 1960.0 / 379; `head.layers.0.norm1` 36.1 / 20.50 / 45.4 / 382; `head.layers.0.norm2` 35.1 / 21.03 / 54.0 / 382; `head.layers.1.norm1` 1001.9 / 3.79 / 6225.0 / 963; `head.layers.1.norm2` 1049.8 / 3.73 / 6635.7 / 963; `scorer.0` (input = gathered marker rows) 1270.8 / 3.37 / 11009.0 / 963.

Reading: the encoder residual stream carries channel-localised outliers that grow in steps: channel 379 reaches 1404 after layer 5, 4007 after layer 12, channel 382 reaches 11696 after layer 14, and channel 379 reaches 33109 after layer 19, then stays flat to layer 26 (median channel max 76 to 79 there). Every post-norm tensor stays under 24. The head's own residual stream is also large: after head layer 0 the mean token L2 norm is 6225 (channel 963 at 1002), after head layer 1 it is 9139 (1271), while the scorer's LayerNorm brings it back to 3.4. For the device port: bf16 has 8 bits of mantissa, so a residual value of 33,000 is quantised in steps of 256 and a value of 11,700 in steps of 64; the per-layer updates in those channels are of the same order as the quantum, which is PLAN.md risk 4. The post-norm activations and the final-norm output (max 35.7) are benign. Compare TT-versus-reference per layer at real positions and watch channels 379, 382, 195 and 963.

## 7. HF versus from-scratch reference (`tests/test_reference_parity.py`)

Run: `cd /home/hous/dev/ornith-1.5-9b/tt-metal && python -m pytest models/autoports/convaiinnovations_laya/tests/test_reference_parity.py -v -s --noconftest -p no:cacheprovider` in the tt-metal venv (`--noconftest` skips the device fixtures of the root and autoport conftest files; the test needs none). Log `p1_pytest_reference_parity_20261005T213946Z.log`: 16 passed (the clamp test of amendment A6 was added afterwards and is covered by `p1_pytest_reference_parity_final_*.log`). The batch is `common.build_batch(512, 4, seed=0)`: real lengths 191, 266, 320, 338, so every row is padded.

| test | measured |
|---|---|
| `test_config_pins` | thetas 160000 / 10000, half window 64, 28 layers, full every third layer, rl config 512 / 192 / 2 head layers |
| `test_state_dict_maps_exactly` | from-scratch reference `load_state_dict(hf.state_dict(), strict=True)` passes; 170 encoder keys, 206 total, layer 0 has no `attn_norm`, dtypes fp16 + fp32 |
| `test_reference_matches_hf` | PCC at real positions 1.0000000000 for the final output and for all 29 hidden states (worst 1.0000000000 at index 16); whole-tensor PCC including padded positions 0.8313 with max abs 34.7, so padded query rows are implementation-defined and must be excluded from any comparison |
| `test_hf_eager_vs_sdpa` | PCC 1.0000000000, max abs 7.8e-4 on the last hidden state |
| NC1 GeGLU gate swapped | PCC 0.4228 |
| NC2 window 65 // 2 | PCC 0.9191 |
| NC3 norm at layer 0 | rejected by the strict key map |
| NC4 Q/K permuted | PCC 0.3647 |
| NC5 single RoPE theta | PCC 0.9311 |
| NC6 band removed | PCC 0.8918 |
| `test_laya_reference_bit_identical_to_vendored` | eager versus vendored sdpa: logits 3.1e-6, act logits 4.9e-4; `LayaReference(attn="sdpa")` bit-identical to the vendored model on logits and act logits; two calls bit-identical; `forward_with_hidden` bit-identical |
| `test_explicit_head_matches_fused` | logits 2.4e-7, act logits 4.9e-4, PCC 1.0; per head layer PCC 1.0 at real positions, max abs 3.1e-4 and 1.2e-4; dropping the pad mask moves the layer outputs by 29.1 and 0.54 |
| `test_host_postprocessing_matches_vendored_system_one` | sequences equal the captured tensors; Hub-shape mirror (raw temperatures) equals `RLAgent.system_one` answers |
| `test_padding_invariance_fp32` | natural 181 versus padded 512: logits 5.7e-6, act logits 9.8e-4, same argmax |
| `test_temperature_clamp_rule` | choice:11+ gives 0.1006 raw and 0.5 clamped; all corpus buckets equal under both rules; clamp edge cases |

## 8. Evidence

- `/home/hous/dev/laya/logs/p1_corpus_20261005T211106Z.log` (corpus build)
- `/home/hous/dev/laya/logs/p1_e0_vendor_20261005T213316Z.log`, `p1_e0_pip_20261005T213646Z.log`, `p1_e0_compare_20261005T213930Z.log` (first full E0 run), `p1_outliers_20261005T213931Z.log`, `p1_pytest_reference_parity_20261005T213946Z.log` (16 tests, before amendment A6)
- `/home/hous/dev/laya/logs/p1_pytest_reference_parity_final_20261005T214343Z.log` (17 tests, final code), `p1_e0_vendor_final_20261005T214512Z.log`, `p1_e0_compare_final_20261005T214909Z.log` (E0 re-run with the flagged decode mirrors; the files under `evals/equivalence/` are from this run)
- `/home/hous/dev/laya/logs/p1_chain.log`, `p1_chain2.log` (job sequencing)
- `work_log.md` (timeline and decisions)

## 9. Deviations from PLAN.md and open items for the other tracks

- Checkpoint dtype is fp16, not bf16 (section 1). Device weight conversion must start from fp16 tensors.
- The gate subset is stratified by workflow with seed 13, not "the first 40 cases" (section 4).
- parity_fast yields 288 questions, not about 480; the full corpus is 488 items.
- The vendored `.py` files differ from the Hub bytes by formatting and two unused imports (section 2); AST-identical otherwise.
- `common.build_inputs` now returns real, right-padded typed-decisions sequences with a real attention mask (zeros at the tail). Track T tests that assumed an all-ones mask must compare at real positions or pass the mask through.
- Thread policy: the corpus job ran with 16 threads before the orchestrator capped per-track CPU use; every later job and the library default use 6 threads (`LAYA_CPU_THREADS` overrides).
- For Track T: head layer math, scorer, fp32 final linear, `type_emb` add and the host tail are specified in section 3 and `doc/context_contract.json`; the fused head path does not zero padded rows, so a device head that computes padded rows normally matches; compare at real positions.
- For Track S: `LayaReference.forward` is the CPU backend; `decode_items(..., shape="pip")` (clamp default on, amendment A6) is the verified pip-format decode; `usage.input_tokens` equals the attention-mask sum of the call.
- Padded query rows of the encoder differ between HF and the from-scratch reference (whole-tensor PCC 0.83 at S=512 B=4 while real positions give 1.0); any TT-versus-reference comparison must index real positions.
- The act head is saturated (act logits up to 4862 in magnitude; `act_probability` 1.0 on every corpus item); a PCC on act logits is dominated by their scale and the act argmax never changes. Do not gate on it.
- The residual stream reaches 33,000 in channel 379 from layer 19 (section 6); the bf16 quantum there is 256.
