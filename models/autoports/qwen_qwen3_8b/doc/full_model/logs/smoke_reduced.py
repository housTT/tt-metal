import sys
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator, greedy_sampling_params

layers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
t0 = time.perf_counter()
gen = build_generator(
    "models/autoports/qwen_qwen3_8b",
    mesh,
    num_layers=layers,
    **({"decode_buckets": (1, 8), "warmup_prefill_lengths": (32, 128), "num_blocks": 256} if layers < 36 else {}),
)
print("BUILD s", round(time.perf_counter() - t0, 1), flush=True)
tok = gen.tokenizer
prompt = tok.encode(tok.apply_chat_template([{"role": "user", "content": "What is 2+2?"}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)
print("prompt len", len(prompt), flush=True)
t0 = time.perf_counter()
out = gen.generate(prompt, 12, next_input=None, enable_trace=True)
print("GENERATE s", round(time.perf_counter() - t0, 2), out, repr(tok.decode(out)), flush=True)
print("counters", gen.counter_snapshot(), flush=True)
t0 = time.perf_counter()
out2 = gen.generate(prompt, 12, next_input=None, enable_trace=True)
print("GENERATE2 s", round(time.perf_counter() - t0, 2), out2, "same", out == out2, flush=True)
print("counters", gen.counter_snapshot(), flush=True)
logits = gen.prefill_logits(prompt + out[:5])
print("prefill_logits", tuple(logits.shape), logits[0, len(prompt) - 1 :].argmax(-1).tolist(), flush=True)
ref = gen.prefill_logits_continuation(prompt + out[:5], 0, positions=[len(prompt) - 1, len(prompt) + 3])
cand = gen.prefill_logits_continuation(prompt + out[:5], len(prompt) - 3, positions=[len(prompt) - 1, len(prompt) + 3])
ctrl = gen.prefill_logits_continuation(prompt + out[:5], len(prompt) - 3, positions=[len(prompt) - 1, len(prompt) + 3], prefill_prefix=False)
def pcc(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))
print("continuation pcc", pcc(ref, cand), "control pcc", pcc(ref, ctrl), flush=True)
gen.reset()
blocks = gen._ensure_blocks(0, len(prompt) + 4)
table = gen._host_page_table([0])
last = gen.prefill_forward(torch.tensor([prompt]), page_table=table, kv_cache=gen.kv_cache, prompt_lens=[len(prompt)])
print("prefill_forward logits", tuple(last.shape), int(last[0, 0].argmax()), flush=True)
toks = torch.tensor([[int(last[0, 0].argmax())]])
pos = torch.tensor([len(prompt)])
host_logits = gen.decode_forward(toks, pos, page_table=table, kv_cache=gen.kv_cache)
print("decode host logits", tuple(host_logits.shape), int(host_logits[0].argmax()), flush=True)
dev = gen.decode_forward(torch.tensor([[int(host_logits[0].argmax())]]), pos + 1, page_table=table, kv_cache=gen.kv_cache, sampling_params=greedy_sampling_params(1))
print("decode device token", dev.tolist(), flush=True)
dev2 = gen.decode_forward(torch.tensor([[0]]), pos + 2, page_table=table, kv_cache=gen.kv_cache, sampling_params=greedy_sampling_params(1))
print("decode device token 2", dev2.tolist(), "counters", gen.counter_snapshot(), flush=True)
print("traces", gen.trace_summary(), flush=True)
gen.teardown()
ttnn.close_mesh_device(mesh)
print("SMOKE_OK", flush=True)
