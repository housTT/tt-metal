"""Same arguments as `python -m readiness_check.generate --hf-model Qwen/Qwen3-8B --prompt-source aime24
--chat-template --gen-len 100 --top-k 100 --output bringup/references/qwen3_8b_aime24_chat.refpt`.

Only `_chat_or_plain_prompt_tokens` is replaced: with transformers 5.12 `apply_chat_template(tokenize=True)`
returns a BatchEncoding, which the packaged helper iterates as token ids. The replacement renders the
template to text and encodes it without special tokens, which is the same token sequence."""
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
sys.argv = [
    "readiness_check.generate",
    "--hf-model",
    "Qwen/Qwen3-8B",
    "--prompt-source",
    "aime24",
    "--chat-template",
    "--gen-len",
    "100",
    "--top-k",
    "100",
    "--output",
    "bringup/references/qwen3_8b_aime24_chat.refpt",
]
g._main()
