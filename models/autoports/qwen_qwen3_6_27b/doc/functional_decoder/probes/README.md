# Diagnostic probes

Standalone scripts used to verify or refute each hypothesis during this stage. They are kept
verbatim so the findings in `../work_log.md` can be reproduced. They are **not** part of the
test suite: they take minutes, print rather than assert, and some of them exist only to
characterise a TTNN op.

Run them from the repo root with the stage environment loaded:

```bash
cd /home/ttuser/dev/qwen/tt-metal
source models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh
python models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/probes/<script>.py
```

(`ttenv.sh` activates this checkout's own venv, asserts `ttnn` resolves inside the checkout,
points `TT_METAL_HOME` at the repo — which is where the loaded `_ttnn.so` was built from — and
selects the healthy PCI device.)

| script | question it answers | verdict |
|---|---|---|
| `probe_batched.py` | which users/sequence lengths fail at batch 32 | failures exactly where `padded - logical < 32` |
| `probe_boundary.py` | where in the output does the divergence start | at position 0 → the layer *input* is wrong |
| `probe_pad.py` | does `ttnn.pad` zero the padded rows | yes, always correct on its own |
| `probe_pad_alias.py` | does `ttnn.pad` alias its input | **yes** when the padding fits in the tile padding; freeing the input then frees the result |
| `probe_real_linear.py` | is the real-weight gap in prefill, state or decode | recurrent state diverges (2.76e18 vs 3.36) and grows with length |
| `probe_inv.py` | is the delta-rule inverse or its inputs wrong | inverse: TTNN error 3.27 vs torch fp32 2.45e-4 on the same captured `attn0` |
| `probe_capture_attn0.py` | how big do the doubling product's intermediates get (CPU only) | `\|A^8\|` peaks at 9.9e2, partial product at 4.3e2, result 1.0 — ~3 orders of cancellation |
| `probe_matmul_prec.py` | how accurate is `ttnn.matmul` under each compute config | ~1.4e-3 relative at fp32/HiFi4; fidelity/packer changes do not rescue it |
| `probe_blockinv.py` | does block-recursive inversion fix the accuracy | yes: 3.269 → 3.4e-2 (base 32) → 8.8e-3 (base 16) → 1.8e-3 (base 8) |
| `probe_capacity.py` | device DRAM and the byte budget at full context | feeds `../context_contract.json` |
| `probe_tri_inv_base.py` | model-level PCC and warmed prefill time for `TRI_INV_BASE` in {8,16,32} | same PCC at all three; 204.6 / 160.5 / 135.8 ms -> base 16 selected |
| `probe_sdpa_precision.py` | how does prefill tail PCC move with context and `fp32_dest_acc_en` | 0.970 -> 0.984 at 262143; the cliff between 131071 and 262143 is not smooth |
| `probe_sdpa_localise.py` | which stage of `full_attention` prefill loses the accuracy at 262143 | the SDPA op alone: it scores 0.9991 on its own Q/K/V while the layer scores 0.9838 |
| `probe_amplification.py` | is the layer error propagation of the SDPA error, or something downstream (host only) | pure propagation: torch-continuation of the *device* attention output reproduces the layer PCC to 0.9838 |
| `probe_sdpa_synthetic.py` | model-free `chunked_scaled_dot_product_attention` vs float32, with the fitted output scale | the output is scaled by 1.465 at 1024 k chunks; the scale tracks the chunk count, not the context |
| `probe_sdpa_decode_synthetic.py` | model-free `paged_scaled_dot_product_attention_decode` vs float32, by position | 37.7x too large at position 262143; explicit program configs fix that but break other positions |
| `run_perf.sh` | the Tracy + `tt-perf-report` command sequence used for `../tracy/` | — |

`probe_capture_attn0.py` writes `/tmp/attn0_real.pt`, which `probe_blockinv.py` consumes; run
it first. It is the only script here that needs no device.

`probe_sdpa_synthetic.py` and `probe_sdpa_decode_synthetic.py` need no model and no weights;
they are the artefacts to attach to the upstream tt-metal issues described in `../work_log.md`
sections 8 and 9. Both report a fitted scale factor `alpha` alongside PCC, because the defect
they characterise is a pure scale on the softmax denominator and PCC is scale-invariant.
`probe_sdpa_localise.py` writes its tensors to `$DUMP_DIR/sdpa_localise_<len>.pt` (default
`/home/ttuser/dev/qwen/tt-metal/generated`), which `probe_amplification.py` then reads on the
host with no device at all.

`probe_tri_inv_base.py` was added after the first stage review, which pointed out that
`TRI_INV_BASE` had been chosen purely on inverse-level accuracy with no runtime or
model-level measurement. Its output (`../logs/tri_inv_base_sweep.log`) is what
`tt/functional_decoder.py` now cites for `TRI_INV_BASE = 16`.
