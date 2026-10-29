"""Tracy-signposted profile of the reduced full model (one real decoder layer, real terminal path).

Windows: PERF_PREFILL_128 (warmed 128-token prompt through embedding, the layer, last-row gather, final norm,
LM head and eager greedy sampling), PERF_DECODE (one blocking replay of the decode trace plus the sampling
trace after two warm replays). Run under `python -m tracy -r -p -v -o <dir> <this file>`.
"""
import bz2
import os

import ttnn

try:
    from tracy import signpost
except ImportError:
    signpost = lambda *_, **__: None

from models.autoports.qwen_qwen3_8b.tt.generator import build_generator

layers = int(os.environ.get("QWEN3_PROFILE_LAYERS", "1"))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=256 * 1024 * 1024)
gen = build_generator(
    "models/autoports/qwen_qwen3_8b",
    mesh,
    num_layers=layers,
    decode_buckets=(1,),
    warmup_prefill_lengths=(32, 128),
    num_blocks=256,
)
tok = gen.tokenizer
text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read(50000)
prompt = tok.encode(text, add_special_tokens=False)[:128]
gen.reset()
gen.generate_prefill_only(prompt)
gen.reset()
ttnn.synchronize_device(mesh)
signpost("PERF_PREFILL_128")
gen.generate_prefill_only(prompt)
ttnn.synchronize_device(mesh)
signpost("PERF_PREFILL_128_END")
state = gen.buckets[1]
for _ in range(2):
    gen._replay(state, "device", gen.kv_cache)
    gen._read_tokens(state, 1)
ttnn.synchronize_device(mesh)
signpost("PERF_DECODE")
gen._replay(state, "device", gen.kv_cache)
tokens = gen._read_tokens(state, 1)
signpost("PERF_DECODE_END")
print("profiled token", tokens.tolist(), "counters", gen.counter_snapshot(), flush=True)
gen.teardown()
ttnn.close_mesh_device(mesh)
print("PROFILE_OK", flush=True)
