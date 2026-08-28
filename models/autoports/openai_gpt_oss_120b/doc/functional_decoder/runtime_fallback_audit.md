# Runtime fallback audit

Verdict: clean for one measured `prefill_forward` or `decode_forward` pass.

## Audited call tree

The public runtime path is:

```text
FunctionalDecoder.prefill_forward / decode_forward
  -> FunctionalDecoder._forward
     -> RMSNorm.forward
     -> Attention.__call__ -> attention.prefill_forward / decode_forward
     -> _FunctionalMLP.__call__
        -> TopKRouter.__call__
        -> Experts.__call__ -> experts.prefill_forward / decode_forward
     -> ttnn.add
```

The audit searched the implementation and all reusable modules in that call tree for `torch`, `ttnn.from_torch`, and `ttnn.to_torch`.

- `models/autoports/openai_gpt_oss_120b/tt/functional_decoder.py` contains none of the forbidden APIs. Its runtime methods use TTNN tensors and operations only.
- `models/demos/gpt_oss/tt/attention/prefill.py`, `decode.py`, and `operations.py` contain no forbidden APIs. PyTorch in `attention/weights.py` and `attention/kv_cache.py` is constructor-time weight/cache creation.
- `models/demos/gpt_oss/tt/experts/prefill.py`, `decode.py`, and `operations.py` contain no forbidden APIs. `Experts._create_prefill_sparsity` uses PyTorch and `ttnn.from_torch` once in `Experts.__init__`, before any measured pass.
- `RMSNorm` uses the host state dict only in `__init__`; `forward` is TTNN-only.
- `TopKRouter` uses host tensors for weight/bias setup. Its lazy fused-op setup also contains a host path, but `_FunctionalMLP` always calls the router with `use_throughput_experts=False`; therefore `_fused_call` and `_init_fused_op` are unreachable from this functional decoder's prefill/decode runtime. The reachable `TopKRouter.__call__` path is TTNN-only.

Input construction, RoPE construction, HF reference execution, and final output conversion are test-harness boundaries outside the `PERF_PREFILL`/`PERF_DECODE` signposts. `from_state_dict` is the documented host-to-device construction boundary.

As a second check, each signpost-filtered `tt-perf-report` table reports only device operations and `0 host ops`:

| Layer kind | Prefill | Traced decode |
| --- | --- | --- |
| sliding attention | 58 device ops, 0 host ops | 64 device ops, 0 host ops |
| full attention | 58 device ops, 0 host ops | 64 device ops, 0 host ops |

Audit command:

```bash
rg -n 'torch|ttnn\.from_torch|ttnn\.to_torch' \
  models/autoports/openai_gpt_oss_120b/tt/functional_decoder.py \
  models/demos/gpt_oss/tt/attention \
  models/demos/gpt_oss/tt/experts \
  models/demos/gpt_oss/tt/rms_norm.py \
  models/demos/gpt_oss/tt/topk.py
```
