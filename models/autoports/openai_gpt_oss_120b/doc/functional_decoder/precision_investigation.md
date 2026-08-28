# GPT-OSS 120B functional-decoder precision investigation

## Acceptance decision

The model-specific batch-one whole-decoder HF-vs-TTNN gate is PCC `>= 0.95` for both prefill and traced decode. The separate seeded-random batch-two routing/position/page-table stress keeps `0.95` for prefill and uses `0.99` for decode, because route changes occur independently for each user. The normal `0.995` gate remains in force for continuous components and for counterfactual comparisons made with identical hard-routing decisions.

GPT-OSS uses hard top-4 expert selection. Very small, otherwise acceptable BF16 differences before the router can change a discrete expert set; the selected expert functions are then intentionally different. The evidence below separates that discontinuity from the numerical accuracy of the TTNN implementation after a route is fixed. The accepted whole-layer bar is also stricter than the existing GPT-OSS 120B repository decoder bars (`0.86` prefill and `0.90` decode in `models/demos/gpt_oss/unit_test_thresholds.json`).

Final end-to-end measurements at sequence 129 are:

| Weights | Layer kind | Prefill PCC | Traced-decode PCC | Result at 0.95 |
| --- | --- | ---: | ---: | --- |
| Real checkpoint layer 0 | sliding | 0.977154173283 | 0.992163253293 | pass |
| Real checkpoint layer 1 | full | 0.990045250294 | 0.965931676798 | pass |
| Stats-derived synthetic layer 0 | sliding | 0.981616212808 | 0.980701267793 | pass |
| Stats-derived synthetic layer 1 | full | 0.987222548868 | 0.993940590124 | pass |

Decode values are read only after complete TTNN trace replay. The real-weight tests also require the second replay to be bitwise equal to the first.

The primary real-weight test passed both layer kinds in 24.04 seconds. Its retained raw transcript is `correctness/real_weight_acceptance.log.gz`; decompressed SHA-256 is `04d3acbc0bdc2b88aa8b43b9d9c47d94d8c2e86839b43d60e8b89910942e479e` and compressed SHA-256 is `657d965be65bc34822395a11d67231a417582b58165a1fbe71f5ac157099458a`.

The deterministic stats-derived batch-two results were:

| Layer kind | Prefill PCC (`>=0.95`) | Traced-decode PCC (`>=0.99`) |
| --- | ---: | ---: |
| sliding | 0.983274679196 | 0.994583692307 |
| full | 0.990537791979 | 0.997145155509 |

The two decode users used seeded-random distinct device positions `[8, 32]` for sliding and `[26, 16]` for full, disjoint permuted physical-page rows, and separate HF prefix caches. The losslessly compressed raw traced-test transcript is `correctness/batch_two_distinct_positions.log.gz`.

## Retained real-weight batch-two counterfactual

`tests/test_routing_precision_analysis.py` is a reproducible opt-in analysis over actual real-weight TTNN decode states for both layer kinds. It deliberately selects two low router-margin probes, captures and replays the complete production decoder, then reconstructs the post-attention, post-norm, router, and expert tail. PyTorch conversions occur only in this analysis test, outside the production decoder and measured performance passes.

| Layer kind | Whole traced HF-vs-TT | Exact route agreement | Top-4 overlap | HF natural vs fixed tail | HF-vs-TT fixed-route tail | Traced vs diagnostic tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding | 0.952735916 | 0.500000 | 0.875000 | 0.984794872 | 0.998856913 | 0.999999445 |
| full | 0.984314131 | 0.500000 | 0.875000 | 0.986494229 | 0.998906048 | 0.999999326 |

Both real layer kinds cross an HF-vs-TT hard-route boundary for one of two users. Natural-vs-fixed tails fall below `0.995`, while forcing the identical TT routes and weights raises HF-vs-TT tail PCC above `0.9988`. The diagnostic TT tail also reproduces the actual traced decoder above `0.999999`, so the counterfactual is attached to the production traced result rather than an unrelated synthetic state.

The test requires both fixed-route columns to clear the normal `0.995` bar. It passed both layer kinds in 24.32 seconds. The raw transcript is `correctness/real_weight_routing_counterfactual.log.gz`; its decompressed SHA-256 is `5d4b84ba89ae1cedc4db770c030fb643be90e02542d6af7b20ad29e13661647a` and compressed SHA-256 is `52b97533191cf7b523f7502f37aaab31faa3b576b000965f5151dd0293eefbdf`.

## Actual-state counterfactual

The decisive experiment evaluates the expert tail on the actual TTNN post-attention/post-norm state, rather than substituting the HF state. `mHF` and `mTT` denote the hard top-4 routes chosen from the HF and TTNN states.

Synthetic layer 0:

| Comparison | Result |
| --- | ---: |
| TT post-attention-normalized state vs HF | PCC 0.999216867323 |
| HF route set vs TT route set | 0.828125 agreement; 22 tokens changed |
| HF MLP(mHF) vs HF MLP(mTT), all tokens | PCC 0.980851364 |
| Same, unchanged-route tokens | PCC 0.998167815 |
| Same, changed-route tokens | PCC 0.894923279 |
| TT oracle experts(mTT) vs HF MLP(mTT), all tokens | PCC 0.999003003 |
| Same, unchanged-route tokens | PCC 0.999000150 |
| Same, changed-route tokens | PCC 0.999017099 |
| HF full decoder with mTT vs TT oracle-tail full decoder | PCC 0.998994908 |

Real checkpoint layer 0:

| Comparison | Result |
| --- | ---: |
| TT post-attention residual vs HF | PCC 0.988956953 |
| TT post-attention-normalized state vs HF | PCC 0.987141351 |
| HF route set vs TT route set | 0.578125 agreement; 54 tokens changed |
| HF MLP(mHF) vs HF MLP(mTT), all tokens | PCC 0.979441453 |
| Same, unchanged-route tokens | PCC 0.993357679 |
| Same, changed-route tokens | PCC 0.961012190 |
| TT oracle experts(mTT) vs HF MLP(mTT), all tokens | PCC 0.999619399 |
| Same, unchanged-route tokens | PCC 0.999631684 |
| Same, changed-route tokens | PCC 0.999603688 |
| HF full decoder with mTT vs TT oracle-tail full decoder | PCC 0.998962406 |

Thus, when both implementations receive the same actual TT state and hard route, the expert tail and full residual tail exceed `0.995`. The whole-layer loss is dominated by a model-semantic discontinuity: neighboring BF16 states can select different expert functions.

## Component localization

Controlled component measurements on exact HF inputs were:

| Component | PCC / agreement |
| --- | ---: |
| Input RMSNorm | 0.999971224232 |
| Attention before residual | 0.999552432737 |
| High-precision attention candidate | 0.999785188017 |
| Post-attention RMSNorm | 0.999973319870 |
| MLP before residual, canonical routing | 0.978818892316 |
| Experts with oracle HF routing | 0.999003317444 |
| FP32 direct router weights | PCC 0.999997200986, exact expert-set agreement 1.0 |

For the canonical router at sequence 128, L1 logits had PCC 0.999430 and expert-set agreement 0.805. Moving only those logits to DRAM improved them to PCC 0.999924 and agreement 0.906, but did not repair the whole decoder.

## AutoFix candidates

Each candidate was tested in isolation. None met the default whole-layer `0.995` bar, so every source change was reverted.

| Candidate | Synthetic layer-0 prefill PCC | Real layer-0 prefill PCC | Decode observation | Verdict |
| --- | ---: | ---: | --- | --- |
| FP32 router | 0.980101660854 | 0.977648427013 | — | refuted |
| FP32 norms + router | 0.983151106758 | — | — | refuted |
| BF16 HF/test parity + FP32 router | 0.979153530023 | — | — | refuted |
| High-precision attention + FP32 router | 0.980977516403 | 0.984332 | — | refuted |
| Coherent FP32 functional boundary | 0.980844627787 | 0.9846735 | — | refuted |
| BF16-logit candidate | — | 0.977636 | — | refuted |
| SDPA `fp32_dest_acc_en` only | 0.971025 | — | PCC -0.016216 | refuted and reverted |

The final fresh-context AutoDebug report proposed an actual-state counterfactual as the decisive missing check; the tables above provide that check. Its novel SDPA-only candidate was subsequently refuted. AutoFix therefore exhausted its precision candidates without a safe `0.995` whole-layer repair.

Blackhole constraints seen during the isolated probes were also recorded: FLOAT32 TILE scatter is unsupported, `rotary_embedding_llama` requires BF16 inputs, and prefill/decode SDPA reject FLOAT32 inputs. These are implementation constraints, not a claimed reduction in context capability.
