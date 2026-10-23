# Optional YaRN bringup

Implementation preparation only. No million-token device capacity, generation,
accuracy, serving performance, or profile support is claimed by this change.
The native profile remains 262144 tokens with the original RoPE configuration.

The pinned model card recommends factor 4.0, original context 262144, and a
1000000-token serving window. `config.json` records that explicit opt-in.
`OrnithModel.from_pretrained(..., max_context=1000000,
rope_parameters=<rope_scaling from config.json>)` now constructs an effective
copy of the pinned text config and passes it through every decoder's RoPE setup.
It preserves the checkpoint's partial rotary dimension, theta, and text M-RoPE
settings. It does not change the checkpoint files or infer scaling merely from
a larger requested context.

Serving reads both legacy top-level `rope_scaling` and current text-level
`rope_parameters`. Conflicting fields fail before model allocation. A YaRN
request additionally requires its own `doc/context_contract.json` entry under
`optional_yarn.profiles.<p150|p150x2|p150x4>` with `status: pass`, `max_context`,
and the complete tested `rope_parameters`. The entry must be written only after
capacity and accuracy validation on that profile. Every supplied RoPE parameter
must match that entry. A TP4 entry does not authorize TP1 or TP2.

Required device work before recording a passing entry:

- Allocate the complete selected-precision model, KV pool, recurrent state,
  RoPE tables, and trace buffers; record actual per-device memory.
- Check both attention kinds and full-model generation against the pinned HF
  YaRN reference. Cover ordinary context and positions beyond 262144.
- Prefill non-aligned long prompts and reach position 999999 through actual
  cache-continuous inference. Table parity alone does not establish this.
- Exercise served streaming, cancellation, reuse, and cache behavior with the
  same opt-in; measure startup, prefill, TTFT, and decode separately.
- If capacity fails, preserve allocator/device evidence and validate the maximum
  that fits. Keep unsupported profiles absent from the passing contract.

Host verification: `tests/test_yarn_context_host.py` has seven passing tests
covering default immutability, explicit scaling, invalid overrides, HF rotary
parity at 0/262143/262144/999999, version-field resolution, evidence isolation
across profiles, and adapter-to-model forwarding. These tests open no devices.

YaRN is static: selecting it also changes ordinary-length requests. Native and
YaRN accuracy/performance results must be reported separately.
