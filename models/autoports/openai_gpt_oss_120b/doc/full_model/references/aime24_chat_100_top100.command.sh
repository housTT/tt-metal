/home/ttuser/dev/gpt-oss-20b/tt-metal/python_env/bin/python - <<'PY'
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.common.readiness_check.generate import (
    DEFAULT_AIME24_PROMPTS_FILE,
    _chat_or_plain_prompt_tokens,
    _generate_continuation_tokens,
    _generate_one_entry,
    _generation_stop_ids,
    _load_aime24_prompt,
    _safe_pad_id,
)
from models.common.readiness_check.schema import Reference, save_reference

snapshot = '/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
output = Path('/home/ttuser/dev/gpt-oss-20b/tt-metal/models/autoports/openai_gpt_oss_120b/doc/full_model/references/aime24_chat_100_top100.refpt')
device_map = {
    'model.embed_tokens': 'cpu',
    'model.norm': 'cpu',
    'model.rotary_emb': 'cpu',
    'lm_head': 'cpu',
}
for layer_idx in range(36):
    device_map[f'model.layers.{layer_idx}'] = 'cpu' if layer_idx < 27 else 'disk'
print('Loading pinned HF model: layers 0-26 CPU, layers 27-35 NVMe...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    snapshot,
    trust_remote_code=True,
    device_map=device_map,
    offload_folder='/tmp/gpt_oss_120b_hf_explicit_offload_20260829',
    offload_state_dict=True,
    offload_buffers=True,
    low_cpu_mem_usage=True,
).eval()
prompt_text = _load_aime24_prompt(DEFAULT_AIME24_PROMPTS_FILE, 0)
prompt_tokens = torch.tensor(_chat_or_plain_prompt_tokens(tokenizer, prompt_text, chat_template=True), dtype=torch.long)
print(f'Chat-template prompt tokens: {prompt_tokens.numel()}', flush=True)
gen_tokens = _generate_continuation_tokens(model, tokenizer, prompt_tokens, 100, torch.device('cpu'))
if gen_tokens.numel() != 100:
    raise RuntimeError(f'Expected exactly 100 continuation tokens, got {gen_tokens.numel()}')
print('Generated exactly 100 continuation tokens; extracting top-100...', flush=True)
entry = _generate_one_entry(model, tokenizer, prompt_tokens, gen_tokens, 100, torch.device('cpu'))
stop_ids = _generation_stop_ids(tokenizer, model)
reference = Reference(
    k=100,
    hf_model_id='openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a',
    entries=[entry],
    token_ids_meta={
        'bos_id': int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None,
        'eos_id': int(stop_ids[0]),
        'pad_id': _safe_pad_id(tokenizer, stop_ids),
    },
)
print(f'Reference saved to: {save_reference(reference, output)}', flush=True)
PY
