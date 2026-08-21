# Selected-config qualitative verdict

Status: **pass**.

The exact local `Qwen/Qwen3.6-27B` checkpoint and tokenizer chat template were
used for both controls. Six shared readiness prompts were rendered with
`tokenizer.apply_chat_template(add_generation_prompt=True)` and generated
greedily for up to 64 new tokens.

All six TT completions are coherent, prompt-relevant, nondegenerate, and in the
expected language. There is no prompt echo, cross-request leakage, malformed
control token, or replacement character. The HF and TT controls both expose
the base checkpoint's characteristic "thinking process" continuation style;
that shared behavior is not a TT regression. Matching HF/TT prefixes range from
11 to 63 tokens before benign greedy divergence. The automatic degeneracy
report also passes every prompt: maximum identical-token run is two and the
largest repeated-trigram fraction is 0.113.

Artifacts:

- `qualitative_prompt_format.json`
- `qualitative_hf_outputs.json`
- `qualitative_tt_outputs.json`
- `qualitative_degeneracy_report.json`
