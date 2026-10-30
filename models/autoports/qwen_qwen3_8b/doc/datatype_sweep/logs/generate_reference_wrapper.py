"""Forward the command line to `readiness_check.generate` with the stage-6 transformers-5.12 chat-template fix.

Only `_chat_or_plain_prompt_tokens` is replaced (see doc/full_model/logs/generate_reference_wrapper.py): the
packaged helper iterates the BatchEncoding that `apply_chat_template(tokenize=True)` returns on
transformers 5.12; this renders the template to text and encodes it without special tokens.
"""
import sys

from readiness_check import generate as g


def _chat_or_plain_prompt_tokens(tokenizer, prompt_text, *, chat_template):
    if chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False
        )
        prompt_tokens = tokenizer.encode(text, add_special_tokens=False)
    else:
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    if not prompt_tokens:
        raise ValueError("Prompt tokenization produced no tokens")
    return [int(t) for t in prompt_tokens]


g._chat_or_plain_prompt_tokens = _chat_or_plain_prompt_tokens
sys.argv = ["readiness_check.generate"] + sys.argv[1:]
g._main()
