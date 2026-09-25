# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why the synthetic weights are an adversarial case for block-float.  Host only, no device.

``probe_real_weight_policy.py`` finds a large discrepancy: a BFP4 gate/up policy sits at PCC
0.999081 on the real checkpoint and 0.987883 on the suite's synthetic weights.  OPT-012 says such a
discrepancy must be *explained* - by inspecting the synthetic distribution - and not used to veto a
real-weight win. This is that inspection, and it is arithmetic rather than opinion.

The suite's synthetic weights are, by construction
(``reference/hf_reference.py::synthetic_state_dict_from_stats``), an i.i.d. normal draw with **one**
recorded per-tensor mean and standard deviation. A block-float format shares one exponent across each
16-element sub-tile and spends its mantissa bits on the ratio of each element to that block's
maximum, so its accuracy depends entirely on *how unequal the elements of a block are*:

* in an i.i.d. Gaussian every block has nearly the same maximum and the elements inside a block are
  spread over the full distribution, so the shared exponent carries no information and every element
  pays the full mantissa error;
* real projection weights have per-output-channel scale structure and heavy tails, so a large share
  of blocks are dominated by one element the shared exponent represents exactly, and the small
  elements around it contribute little to the product.

This probe quantises the real and the synthetic tensor with the same model of the format and reports
the relative error of each, per tensor. It writes ``PROBEROW`` lines like the other probes.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_blockfloat_distribution.py
"""

from __future__ import annotations

import json
import sys

import torch

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref

#: Elements sharing one exponent in tt-metal's block-float tile formats.
BLOCK = 16
#: Mantissa bits, including the implicit one, of each format.
MANTISSA_BITS = {"bfp8_b": 8, "bfp4_b": 4}
TENSORS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def quantise(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Model of a block-float round trip: one shared exponent per :data:`BLOCK` elements."""
    flat = tensor.reshape(-1)
    pad = (-flat.numel()) % BLOCK
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, BLOCK)
    # The shared exponent is the block maximum's exponent; each element is then a sign plus
    # (bits - 1) mantissa bits scaled by 2**shared_exponent.
    peak = blocks.abs().amax(dim=1, keepdim=True)
    exponent = torch.where(peak > 0, torch.floor(torch.log2(peak)), torch.zeros_like(peak))
    step = torch.ldexp(torch.ones_like(exponent), exponent.to(torch.int64) - (bits - 2))
    quantised = torch.round(blocks / step) * step
    return quantised.reshape(-1)[: tensor.numel()].reshape(tensor.shape)


def relative_error(tensor: torch.Tensor, bits: int) -> float:
    q = quantise(tensor.to(torch.float32), bits)
    return float((q - tensor).norm() / tensor.norm())


def systematic_gain(tensor: torch.Tensor, bits: int) -> float:
    """Best-fit scale of the quantised weight onto the original: ``<w, q> / <w, w>``.

    Separate from :func:`relative_error` because they answer different questions and only this one is
    relevant to a *scale* failure.  Relative error is symmetric noise; this is the systematic part - a
    shared exponent rounds elements far below the block maximum toward zero, so a block with a large
    intra-block dynamic range loses magnitude rather than just gaining noise.

    It is here because the real-weight full-context scale failures needed this ruled out, and it is:
    the shrink is at most ~1e-3 (BFP4) and ~2e-5 (BFP8), while the failures are 3-4 %.  Real weights do
    shrink several times more than the stand-in, which is consistent with their larger intra-block
    range, but several times a hundredth of a percent is still a hundredth of a percent.
    """
    q = quantise(tensor.to(torch.float32), bits)
    a = tensor.to(torch.float64).flatten()
    b = q.to(torch.float64).flatten()
    return float((a @ b) / (a @ a))


def block_inequality(tensor: torch.Tensor) -> float:
    """Mean ratio of a block's maximum to its RMS - 1.0 means every element is the maximum.

    This is the single number that separates the two distributions: the larger it is, the more of
    each block's energy sits in elements the shared exponent represents well.
    """
    flat = tensor.reshape(-1).to(torch.float32)
    pad = (-flat.numel()) % BLOCK
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, BLOCK)
    rms = blocks.pow(2).mean(dim=1).sqrt()
    peak = blocks.abs().amax(dim=1)
    return float((peak / rms.clamp_min(1e-30)).mean())


def main() -> int:
    config = ref.load_text_config()
    stats = ref.load_weight_stats()
    for kind, layer_idx in (("linear_attention", 0), ("full_attention", 3)):
        real = ref.load_real_layer_state_dict(layer_idx)
        synthetic = ref.synthetic_state_dict_from_stats(stats, layer_idx, config, seed=0)
        for name in TENSORS:
            row = {"sweep": "blockfloat_distribution", "kind": kind, "tensor": name}
            for label, tensor in (("real", real[name].to(torch.float32)), ("synthetic", synthetic[name])):
                row[f"{label}_std"] = float(tensor.std())
                row[f"{label}_kurtosis"] = float(((tensor - tensor.mean()) ** 4).mean() / (tensor.var() ** 2))
                row[f"{label}_block_peak_over_rms"] = round(block_inequality(tensor), 4)
                for fmt, bits in MANTISSA_BITS.items():
                    row[f"{label}_{fmt}_rel_err"] = round(relative_error(tensor, bits), 6)
                    row[f"{label}_{fmt}_gain"] = round(systematic_gain(tensor, bits), 8)
            row["bfp4_err_ratio_synthetic_over_real"] = round(
                row["synthetic_bfp4_b_rel_err"] / row["real_bfp4_b_rel_err"], 3
            )
            print("PROBEROW " + json.dumps(row, sort_keys=True), flush=True)
            print(
                f"  {kind:17s} {name:24s} "
                f"bfp4 rel-err real {row['real_bfp4_b_rel_err']:.6f} vs synthetic "
                f"{row['synthetic_bfp4_b_rel_err']:.6f} ({row['bfp4_err_ratio_synthetic_over_real']:.2f}x)  "
                f"bfp8 real {row['real_bfp8_b_rel_err']:.6f} vs {row['synthetic_bfp8_b_rel_err']:.6f}  "
                f"block peak/rms real {row['real_block_peak_over_rms']:.3f} vs "
                f"{row['synthetic_block_peak_over_rms']:.3f}  "
                f"kurtosis real {row['real_kurtosis']:.2f} vs {row['synthetic_kurtosis']:.2f}  "
                f"bfp8 gain real {row['real_bfp8_b_gain']:.8f} bfp4 gain real {row['real_bfp4_b_gain']:.8f}",
                flush=True,
            )
    return 0


def output_share(layer_idx: int, config, state_dict, tokens: int = 128) -> dict:
    """How much of the HF layer's output the MLP and the mixer actually contribute.

    This is the second half of the explanation, and the half the quantisation numbers above rule
    *in* by ruling themselves out: if the block-float error of the weight tensor is the same for the
    real and the synthetic draw, then a layer-level PCC that differs by an order of magnitude can
    only come from how much of the layer's output those weights produce.  A trained MLP's
    contribution is a modest correction on top of the residual stream; a random one's is not
    constrained to be.
    """
    import torch as _torch

    layer = ref.build_reference_layer(layer_idx, state_dict={k: v.clone() for k, v in state_dict.items()})
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(config, 1, tokens, stats)
    captured: dict = {}

    def hook(_module, _inputs, output):
        captured["mlp"] = output.detach()

    handle = layer.mlp.register_forward_hook(hook)
    with _torch.no_grad():
        positions = _torch.arange(tokens)
        cos, sin = ref.text_position_embeddings(ref.make_rotary(config), positions, batch=1)
        text_position_ids = ref.build_text_position_ids(positions, 1)[0]
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=config)
        mask = ref.build_causal_mask(config, hidden, cache, text_position_ids) if layer_idx == 3 else None
        out = layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            position_ids=text_position_ids,
            past_key_values=cache,
            use_cache=True,
        )
    handle.remove()
    return {
        "mlp_norm_over_output_norm": round(float(captured["mlp"].norm() / out.norm()), 4),
        "residual_norm_over_output_norm": round(float(hidden.norm() / out.norm()), 4),
        "output_norm": round(float(out.norm()), 3),
    }


def main_share() -> int:
    config = ref.load_text_config()
    stats = ref.load_weight_stats()
    for kind, layer_idx in (("linear_attention", 0), ("full_attention", 3)):
        row = {"sweep": "output_share", "kind": kind}
        for label, sd in (
            ("real", ref.load_real_layer_state_dict(layer_idx)),
            ("synthetic", ref.synthetic_state_dict_from_stats(stats, layer_idx, config, seed=0)),
        ):
            for key, value in output_share(layer_idx, config, sd).items():
                row[f"{label}_{key}"] = value
        row["mlp_share_ratio_synthetic_over_real"] = round(
            row["synthetic_mlp_norm_over_output_norm"] / row["real_mlp_norm_over_output_norm"], 3
        )
        print("PROBEROW " + json.dumps(row, sort_keys=True), flush=True)
        print(
            f"  {kind:17s} MLP/output norm: real {row['real_mlp_norm_over_output_norm']:.4f} vs "
            f"synthetic {row['synthetic_mlp_norm_over_output_norm']:.4f} "
            f"({row['mlp_share_ratio_synthetic_over_real']:.2f}x);  residual/output: real "
            f"{row['real_residual_norm_over_output_norm']:.4f} vs "
            f"{row['synthetic_residual_norm_over_output_norm']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc or main_share())
