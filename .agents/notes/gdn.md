# Qwen3.6-27B — GDN and gated attention: prior analysis for the optimization stage

Written 2026-08-08, before this stage started, from the stage-1 functional-decoder profile.
Issue: https://github.com/tenstorrent/tt-metal/issues/50475 (P0 — "Optimization of Gated Delta
Net and Gated Attention layers in qwen36").

Read this before the operation-topology audit. It names the single largest win and the file
that already implements it, so the audit can start from a hypothesis instead of a blank sheet.

## The measurement

From `models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/perf_summary.json`, one layer at
seq 2048, one Blackhole chip:

| layer kind | prefill | ops | decode / token | ops |
|---|---|---|---|---|
| `linear_attention` (GDN) | **190.6 ms** | 832 | 3.04 ms | 96 |
| `full_attention` (gated attn) | 18.5 ms | 44 | 2.42 ms | 50 |

48 GDN + 16 full-attention layers => ~9.4 s for a 2k prefill, ~185 ms/token (~5.4 tok/s). GDN
prefill is 10x a full-attention layer at the same length, which is backwards: issue #50475's own
data has GDN as the cheaper of the two below ~16k.

## Root cause 1 — the autoport hand-rolled an op that already exists

`ttnn.transformer.gated_delta_attn_seq` is **already compiled into the installed ttnn** at
`/home/ttuser/.local/lib/model-bringup/tt-metal`. It is a fused C++ GDN kernel doing blocked
forward substitution plus the sequential inter-chunk scan. No rebuild is needed to call it —
verify with `hasattr(ttnn.transformer, "gated_delta_attn_seq")`.

`tt/functional_decoder.py` instead composes that work from ttnn primitives:

- `_unit_tri_inverse` (line 655) recursively block-inverts the triangular matrix down to
  `TRI_INV_BASE = 16`. Those 16x16 matrices are padded into 32x32 tiles and dispatched as
  `b={12288} x 32x32x32` batched matmuls at **HiFi4 FP32 on 1.0 core** — 14.5 ms each,
  ~65 ms of the 190 ms, ~75% of every tile wasted.
- `_linear_attention_prefill_chunk` (line 836) then runs a **32-iteration Python loop** over
  chunks, ~6 matmuls plus slices each. That is most of the remaining ~125 ms, most of the 832
  op count, and the 11.6 ms of op-to-op gap.

Both collapse into one kernel call.

### Where to find the reference

These are in the **built** tree (commit `559921b40a8`), not in this checkout (`d58cb341c70`),
so they cannot be imported by path — port the logic explicitly.

- `ttnn/cpp/ttnn/operations/transformer/gated_delta_attn/gated_delta_attn.hpp` — the signature.
  Takes `L_unit, v_beta_sc, k_bd_sc, intra_attn, q_decay, k_decay_t, dl_exp, L_inv` plus
  optional `initial_state`; returns `(output, final_state)`. All inputs float32 / TILE / DRAM,
  which is the dtype the functional decoder already produces.
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py:604` — the
  canonical caller. Shows how all 8 inputs are prepared, including how `L_inv`'s 4 diagonal
  block inverses per chunk are packed. This is the input-prep reference to mine.
- `models/demos/blackhole/qwen36/tt/` — a complete Blackhole Qwen3.6 reference implementation:
  `gdn/`, `attention/`, `mlp.py`, `layer.py`, `model.py`, `tp.py`, `qwen36_vllm.py`. Useful
  well beyond this stage.

Most of the 8 inputs already exist in `_linear_attention_prefill_chunk` under other names
(`v_beta`, `k_beta * exp_gcum`, `q_decayed`, `k_decayed`, `exp_g_last`). The new work is the
`D^{-1}` unit-diagonal normalisation and packing `L_inv`.

### Caveat that gates the whole thing

`models/demos/blackhole/qwen36/tt/model_config.py:145` sets `gdn_chunk_size = 128` with the
comment that the kernel **requires** 128. The autoport's `DELTA_CHUNK` is 64, matching HF.

The chunked gated delta rule is an exact reformulation, so chunk size should change float
association only, not semantics — but verify that in torch on CPU against
`reference/hf_reference.py` **before** building anything on the kernel. If PCC regresses at 128,
the kernel is unusable as-is and the fallback is root cause 2 plus batching the chunk loop,
which recovers much less. `doc/functional_decoder/probes/probe_capture_attn0.py` is the existing
CPU-only probe pattern for exactly this kind of check (no device needed).

## Root cause 2 — HiFi4 on everything

`tt/functional_decoder.py:215` sets `self.compute_cfg = _hifi4(fp32_dest_acc_en=True)` and uses
it for **every** matmul in the layer, including the bf16 weight matmuls (`wqkv`, `wgate`,
`o_proj`, `mlp_gate_up`, `mlp_down`). HiFi4 costs 4 math passes where LoFi costs 1.

The reference demo makes the opposite choice:

- `models/demos/blackhole/qwen36/tt/mlp.py:96` — LoFi
- `models/demos/blackhole/qwen36/tt/attention/gated_attention.py:27` — LoFi
- `models/demos/blackhole/qwen36/tt/tp_common.py:32` — HiFi2

This is the `32 x 5120 x 34816` matmul at 875 us that tops the decode profile for **both** layer
kinds. Retain HiFi4 + fp32 dest acc only where the numerics need it — the delta-rule state path,
where `_unit_tri_inverse`'s docstring records real cancellation problems.

Headroom is thin: the current PCC floor is 0.99788 against a 0.995 bar, so revert individual
sites rather than the whole change if a downgrade costs too much.

## Not in scope here

The `full_attention` prefill PCC gap at 262143 tokens (0.9838 vs the 0.995 bar) is a
`chunked_scaled_dot_product_attention` accuracy limitation, localised in
`doc/functional_decoder/work_log.md` section 7. Stage 1 owns it.
