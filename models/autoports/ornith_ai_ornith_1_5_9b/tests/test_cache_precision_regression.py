# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-input cache precision localization; mixed caches use separate update ops."""

import gc
import json

import pytest
import torch

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder, PrecisionPolicy
from . import test_functional_decoder as H
from .test_optimization_experiments import recorded_activations

pytestmark = H.pytestmark
_ORIGINAL_FUSED_UPDATE = ttnn.experimental.paged_fused_update_cache


class CachePrecisionCandidate(OptimizedDecoder):
    key_dtype = "bfloat4_b"
    value_dtype = "bfloat4_b"
    precise_attention = False

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        if cls.precise_attention:
            kwargs["policy"] = PrecisionPolicy(attention="bfloat16", attention_fidelity="HiFi4")
        decoder = super().from_state_dict(*args, **kwargs)
        if cls.precise_attention:
            decoder.decode_sdpa_compute = ttnn.init_device_compute_kernel_config(
                decoder.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        return decoder

    def allocate_kv_cache(self, num_blocks, dtype=None):
        result = super().allocate_kv_cache(num_blocks, dtype=getattr(ttnn, self.key_dtype))
        if self.is_full_attention and self.key_dtype != self.value_dtype:
            original = self.v_cache
            self.v_cache = ttnn.typecast(original, getattr(ttnn, self.value_dtype))
            ttnn.deallocate(original)
        return result


def candidate(name):
    key, value, precise = {
        "k4v4": ("bfloat4_b", "bfloat4_b", False),
        "k8v8": ("bfloat8_b", "bfloat8_b", False),
        "k8v4": ("bfloat8_b", "bfloat4_b", False),
        "k4v8": ("bfloat4_b", "bfloat8_b", False),
        "k4v4_attention16": ("bfloat4_b", "bfloat4_b", True),
    }[name]
    return type(
        "Cache_" + name,
        (CachePrecisionCandidate,),
        {"key_dtype": key, "value_dtype": value, "precise_attention": precise},
    )


def separate_update(k_cache, k, v_cache, v, **kwargs):
    if k_cache.dtype == v_cache.dtype:
        return _ORIGINAL_FUSED_UPDATE(k_cache, k, v_cache, v, **kwargs)
    ttnn.experimental.paged_update_cache(k_cache, k, **kwargs)
    ttnn.experimental.paged_update_cache(v_cache, v, **kwargs)


def inputs():
    source = recorded_activations(H.FULL_LAYER)[0]
    prefix = source[(39 + torch.arange(32)[:, None] * 137 + torch.arange(96)[None, :]) % len(source)]
    token = source[(131 + torch.arange(32)[:, None] * 137) % len(source)]
    return prefix, token


@pytest.mark.parametrize("name", ["k4v4", "k8v8", "k4v4_attention16", "k8v4", "k4v8"])
def test_real_cache_precision(mesh_device, monkeypatch, name):
    cls = candidate(name)
    monkeypatch.setattr(H, "FunctionalDecoder", cls)
    if cls.key_dtype != cls.value_dtype:
        monkeypatch.setattr(ttnn.experimental, "paged_fused_update_cache", separate_update)
    prefix, token = inputs()
    ref = H.reference_layer(H.FULL_LAYER, "real")
    with torch.no_grad():
        golden_prefix, cache = H.R.reference_prefill(ref, H.hf_config(), prefix.float(), start_pos=0)
        golden_token = H.R.reference_decode(ref, H.hf_config(), token.float(), torch.full((32,), 96), cache)
    decoder, table, _ = H.build_decoder(mesh_device, H.FULL_LAYER, "real", batch=32, max_context=1024)
    out = decoder.prefill_forward(H.to_device(mesh_device, prefix), page_table=table)
    prefill = ttnn.to_torch(out)
    ttnn.deallocate(out)
    pos, rot = H.decode_inputs(mesh_device, torch.full((32,), 96))
    out = decoder.decode_forward(H.to_device(mesh_device, token), current_pos=pos, rot_idxs=rot, page_table=table)
    actual = ttnn.to_torch(out)
    ttnn.deallocate(out)
    prefill_pcc = [H.pcc(golden_prefix[u], prefill[u]) for u in range(32)]
    decode_pcc = [H.pcc(golden_token[u], actual[u]) for u in range(32)]
    print(
        "CACHE_PRECISION "
        + json.dumps(
            {
                "name": name,
                "batch": 32,
                "prefix": 96,
                "prefill_pcc": prefill_pcc,
                "decode_pcc": decode_pcc,
                "failed_users": [u for u, value in enumerate(decode_pcc) if value < H.PCC_BAR],
                "key_dtype": str(decoder.k_cache.dtype),
                "value_dtype": str(decoder.v_cache.dtype),
                "policy": vars(decoder.policy),
                "cache_shape": list(decoder.k_cache.shape),
                "block_size": decoder.page_block_size,
                "page_table_shape": list(table.shape),
            }
        ),
        flush=True,
    )
    del decoder, table
    gc.collect()
    # The all-B1 extraction separates each original user's distribution from B32 kernels.
    decoder, table, _ = H.build_decoder(mesh_device, H.FULL_LAYER, "real", max_context=1024)
    pos, rot = H.decode_inputs(mesh_device, torch.tensor([96]))
    extracted = []
    for u in range(32):
        decoder.reset_state()
        ttnn.deallocate(decoder.prefill_forward(H.to_device(mesh_device, prefix[u : u + 1]), page_table=table))
        out = decoder.decode_forward(
            H.to_device(mesh_device, token[u : u + 1]), current_pos=pos, rot_idxs=rot, page_table=table
        )
        actual = ttnn.to_torch(out)
        ttnn.deallocate(out)
        extracted.append(H.pcc(golden_token[u], actual[0]))
    print(
        "CACHE_PRECISION_EXTRACTED "
        + json.dumps(
            {
                "name": name,
                "batch": 1,
                "decode_pcc": extracted,
                "failed_users": [u for u, value in enumerate(extracted) if value < H.PCC_BAR],
            }
        ),
        flush=True,
    )
    # Failed candidates are diagnostic observations, not weakened acceptance gates.
    assert all(torch.isfinite(torch.tensor(prefill_pcc + decode_pcc + extracted)))
    if name in ("k8v8", "k8v4"):
        assert min(prefill_pcc + decode_pcc + extracted) >= H.PCC_BAR


@pytest.mark.parametrize("permuted", [False, True])
def test_exact_cache_rows(mesh_device, monkeypatch, permuted):
    """Actual B32 fill/update geometry preserves every untouched physical row."""
    monkeypatch.setattr(H, "FunctionalDecoder", candidate("k4v4"))
    decoder, table, blocks = H.build_decoder(mesh_device, H.FULL_LAYER, "real", batch=32, max_context=1024)
    table_host = torch.arange(32 * blocks, dtype=torch.int32).reshape(32, blocks)
    if permuted:
        table_host = (
            torch.randperm(32 * blocks, generator=torch.Generator().manual_seed(819))
            .to(torch.int32)
            .reshape(32, blocks)
        )
        ttnn.deallocate(table)
        table = H.to_device(mesh_device, table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    prefix, token = inputs()
    ttnn.deallocate(decoder.prefill_forward(H.to_device(mesh_device, prefix), page_table=table))
    previous = [ttnn.to_torch(buf) for buf in (decoder.k_cache, decoder.v_cache)]
    original_update = ttnn.experimental.paged_fused_update_cache
    captured = []

    def update(k_cache, k, v_cache, v, **kwargs):
        for cache, value in ((k_cache, k), (v_cache, v)):
            scratch = ttnn.clone(cache, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.experimental.paged_update_cache(scratch, value, **kwargs)
            captured.append(ttnn.to_torch(scratch))
            ttnn.deallocate(scratch)
        return original_update(k_cache, k, v_cache, v, **kwargs)

    monkeypatch.setattr(ttnn.experimental, "paged_fused_update_cache", update)
    original_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode
    sdpa_records = []

    def sdpa(q, k, v, **kwargs):
        result = original_sdpa(q, k, v, **kwargs)
        query = ttnn.to_torch(q).float()
        keys, values = (ttnn.to_torch(cache) for cache in (k, v))
        actual = ttnn.to_torch(result).float()
        oracle = []
        for user in range(32):
            logical = []
            for cache in (keys, values):
                rows = cache[table_host[user].long()].permute(1, 0, 2, 3).reshape(4, -1, 256)[:, :97].float()
                logical.append(rows.repeat_interleave(4, dim=0))
            scores = (query[0, user].unsqueeze(1) @ logical[0].transpose(-1, -2)) * (256**-0.5)
            oracle.append((scores.softmax(dim=-1) @ logical[1]).squeeze(1))
        oracle = torch.stack(oracle).unsqueeze(0)
        sdpa_records.append({"pcc": H.pcc(oracle, actual), "max_abs": (oracle - actual).abs().max().item()})
        return result

    monkeypatch.setattr(ttnn.transformer, "paged_scaled_dot_product_attention_decode", sdpa)
    pos, rot = H.decode_inputs(mesh_device, torch.full((32,), 96))
    ttnn.deallocate(
        decoder.decode_forward(H.to_device(mesh_device, token), current_pos=pos, rot_idxs=rot, page_table=table)
    )
    checks = []
    for kind, cache, before, packed in zip(("k", "v"), (decoder.k_cache, decoder.v_cache), previous, captured):
        expected = before.clone()
        for user in range(32):
            block = table_host[user, 96 // decoder.page_block_size]
            expected[block, :, 96 % decoder.page_block_size, :] = packed[block, :, 96 % decoder.page_block_size, :]
        actual = ttnn.to_torch(cache)
        checks.append(
            {
                "kind": kind,
                "exact": torch.equal(expected, actual),
                "mismatch_elements": torch.count_nonzero(expected != actual).item(),
                "exact_unfused_control": torch.equal(packed, actual),
            }
        )
    print(
        "EXACT_CACHE_ROWS "
        + json.dumps(
            {
                "permuted": permuted,
                "checks": checks,
                "sdpa_dequantized_cache_oracle": sdpa_records,
                "position": 96,
                "page": 1,
                "offset": 32,
                "rounded_read_end": 256,
                "backed_tokens_per_user": blocks * decoder.page_block_size,
                "shape": list(decoder.k_cache.shape),
                "page_table_shape": list(table.shape),
            }
        ),
        flush=True,
    )
    assert all(row["exact"] for row in checks)
    assert min(row["pcc"] for row in sdpa_records) >= 0.999


@pytest.mark.parametrize("name", ["k8v8", "k8v4"])
def test_cache_precision_pair(mesh_device, monkeypatch, name):
    from . import test_optimization_experiments as pair

    cls = candidate(name)
    monkeypatch.setattr(pair, "selected_candidate", lambda: cls)
    if cls.key_dtype != cls.value_dtype:
        monkeypatch.setattr(ttnn.experimental, "paged_fused_update_cache", separate_update)
    print("CACHE_PAIR_POLICY " + name, flush=True)
    pair.test_optimized_pair(mesh_device, monkeypatch, H.FULL_LAYER)


@pytest.mark.parametrize("name", ["k8v8", "k8v4"])
def test_cache_precision_restored_trace(mesh_device, monkeypatch, name):
    from .test_optimized_trace_regression import all_equal, differences

    cls = candidate(name)
    monkeypatch.setattr(H, "FunctionalDecoder", cls)
    if cls.key_dtype != cls.value_dtype:
        monkeypatch.setattr(ttnn.experimental, "paged_fused_update_cache", separate_update)
    prefix, token = inputs()
    ref = H.reference_layer(H.FULL_LAYER, "real")
    with torch.no_grad():
        golden_prefix, cache = H.R.reference_prefill(ref, H.hf_config(), prefix.float(), start_pos=0)
        golden_token = H.R.reference_decode(ref, H.hf_config(), token.float(), torch.full((32,), 96), cache)
    decoder, table, _ = H.build_decoder(mesh_device, H.FULL_LAYER, "real", batch=32, max_context=1024)
    x, d = H.to_device(mesh_device, prefix), H.to_device(mesh_device, token)
    out = decoder.prefill_forward(x, page_table=table)
    prefill = ttnn.to_torch(out)
    ttnn.deallocate(out)
    saved = H._snapshot_state(decoder)
    pos, rot = H.decode_inputs(mesh_device, torch.full((32,), 96))

    def forward():
        return decoder.decode_forward(d, current_pos=pos, rot_idxs=rot, page_table=table)

    out = forward()
    eager, state = ttnn.to_torch(out), H._snapshot_state(decoder)
    ttnn.deallocate(out)
    H._restore_state(decoder, saved)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    checks = {}
    try:
        for i in range(4):
            H._restore_state(decoder, saved)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            checks[f"output_{i}"] = differences(eager, ttnn.to_torch(out))
            checks[f"state_{i}"] = differences(state, H._snapshot_state(decoder))
        for _ in range(32):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        H._restore_state(decoder, saved)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        checks["poststress_output"] = differences(eager, ttnn.to_torch(out))
        checks["poststress_state"] = differences(state, H._snapshot_state(decoder))
        checks["input"] = differences({"x": prefix, "d": token}, {"x": ttnn.to_torch(x), "d": ttnn.to_torch(d)})
    finally:
        ttnn.release_trace(mesh_device, trace)
    prefill_pcc = [H.pcc(golden_prefix[u], prefill[u]) for u in range(32)]
    decode_pcc = [H.pcc(golden_token[u], eager[u]) for u in range(32)]
    print(
        "CACHE_PRECISION_TRACE "
        + json.dumps(
            {
                "name": name,
                "batch": 32,
                "min_prefill_pcc": min(prefill_pcc),
                "min_decode_pcc": min(decode_pcc),
                "checks": checks,
            }
        ),
        flush=True,
    )
    assert min(prefill_pcc + decode_pcc) >= H.PCC_BAR
    assert all_equal(checks)
