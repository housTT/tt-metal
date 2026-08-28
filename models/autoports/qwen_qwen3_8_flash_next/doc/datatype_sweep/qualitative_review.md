# Selected-policy qualitative review

Artifact: `post_selection/qualitative/qualitative_shared_suite_final.json`

The shared suite used the checkpoint chat template (`tokenizer.apply_chat_template(add_generation_prompt=True)`), three prompt IDs (`explanation`, `coding`, and `summarization`), fresh HF controls from the checked-in reference, 128 generated tokens, greedy device sampling, and traced TT decode. The selected config ID is recorded and all 61 precision-policy leaves are consumed. Every prohibited host-work flag is false and neither HF nor TT output triggers the mechanical-degeneracy gate.

| Prompt | Prompt-format check | HF control | Selected TT | Manual classification |
|---|---|---|---|---|
| `explanation` | 76 tokens; system/user/assistant markers and `<think>` prefix are present | Reaches a scientifically correct explanation but is truncated mid-analogy at the 128-token cap | Reaches a short final sentence but omits the scattering explanation | Coherent but incomplete at the fixed cap; limitation, not degeneration |
| `coding` | 89 tokens; the extra Python-programmer system message is correctly rendered | Begins a correct set-based implementation but is truncated before `return`, closing fence, and example | Remains in reasoning and does not reach a final answer before the cap | Incomplete; the HF control is also capped, while TT is materially less useful |
| `summarization` | 94 tokens; the requested passage is present verbatim inside the user turn | First final sentence is correct, then the fixed reference spills into another turn marker | Produces one complete, accurate sentence and stops | Pass |

The shared qualitative suite is diagnostic evidence, not the datatype accuracy gate. The prompt rendering is valid and the behavior is not mechanically degenerate, but the incomplete explanation/coding responses must not be presented as a clean semantic-quality pass. The AIME24 full-model top-1/top-5/top-100 gate and traced teacher-forcing ranking remain the selection criteria.
