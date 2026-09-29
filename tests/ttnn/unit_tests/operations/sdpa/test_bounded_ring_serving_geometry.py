# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bounded sliding-window ring in the gpt-oss serving geometry: a prompt shorter than the
ring is filled through a cyclic page table, then decode writes and reads walk past the
prompt end, past the first ring wrap, and into windows made only of decode-written
positions. Every step is compared with an unbounded torch reference (PCC >= 0.99)."""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

BLOCK = 64
WINDOW = 128
NUM_KV_HEADS = 2
NUM_Q_HEADS = 16
HEAD_DIM = 64
PADDED_HEADS = 32
TABLE_COLUMNS = 512
RING_BASE = 40


def _sharded_input(device, x_padded):
    num_users = x_padded.shape[1]
    xt = ttnn.Tensor(x_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    grid = ttnn.num_cores_to_corerangeset(num_users, device.compute_with_storage_grid_size(), True)
    spec = ttnn.ShardSpec(
        grid,
        [xt.volume() // xt.padded_shape[-1] // num_users, xt.padded_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return xt.to(device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec))


def _reference(k_hist, v_hist, q, pos, scale):
    lo = max(0, pos - WINDOW + 1)
    k = k_hist[lo : pos + 1].float().repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    v = v_hist[lo : pos + 1].float().repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    scores = torch.einsum("hd,shd->hs", q.float(), k) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,shd->hd", probs, v)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("ring_tokens", [256, 768])
@pytest.mark.parametrize("prompt_len", [699, 799])
def test_ring_decode_walk_after_a_short_prompt(device, ring_tokens, prompt_len):
    torch.manual_seed(0)
    ring_blocks = ring_tokens // BLOCK
    total_blocks = RING_BASE + ring_blocks
    scale = HEAD_DIM**-0.5
    steps = ring_tokens + WINDOW + 200

    k_cache = ttnn.Tensor(torch.zeros(total_blocks, NUM_KV_HEADS, BLOCK, HEAD_DIM).bfloat16(), ttnn.bfloat16)
    v_cache = ttnn.Tensor(torch.zeros(total_blocks, NUM_KV_HEADS, BLOCK, HEAD_DIM).bfloat16(), ttnn.bfloat16)
    k_cache = k_cache.to(ttnn.TILE_LAYOUT).to(device)
    v_cache = v_cache.to(ttnn.TILE_LAYOUT).to(device)

    cyclic = RING_BASE + (torch.arange(TABLE_COLUMNS, dtype=torch.int32) % ring_blocks)
    decode_table = ttnn.Tensor(cyclic.reshape(1, TABLE_COLUMNS), ttnn.int32).to(device)

    prompt_blocks = (prompt_len + BLOCK - 1) // BLOCK
    fill_len = prompt_blocks * BLOCK
    prefill_table = cyclic[:prompt_blocks].reshape(1, prompt_blocks).clone()
    prefill_table_tt = ttnn.Tensor(prefill_table, ttnn.int32).to(device)

    total = prompt_len + steps
    k_hist = torch.randn(total, NUM_KV_HEADS, HEAD_DIM).bfloat16()
    v_hist = torch.randn(total, NUM_KV_HEADS, HEAD_DIM).bfloat16()

    k_fill = k_hist[:fill_len].permute(1, 0, 2).reshape(1, NUM_KV_HEADS, fill_len, HEAD_DIM)
    v_fill = v_hist[:fill_len].permute(1, 0, 2).reshape(1, NUM_KV_HEADS, fill_len, HEAD_DIM)
    fill_kwargs = {"cache_position_modulo": ring_tokens} if fill_len > ring_tokens else {}
    ttnn.experimental.paged_fill_cache(
        k_cache,
        ttnn.Tensor(k_fill, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        prefill_table_tt,
        batch_idx=0,
        **fill_kwargs,
    )
    ttnn.experimental.paged_fill_cache(
        v_cache,
        ttnn.Tensor(v_fill, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        prefill_table_tt,
        batch_idx=0,
        **fill_kwargs,
    )

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
    )
    failing = []
    for pos in range(prompt_len, total):
        k_new = k_hist[pos].reshape(1, 1, NUM_KV_HEADS, HEAD_DIM)
        v_new = v_hist[pos].reshape(1, 1, NUM_KV_HEADS, HEAD_DIM)
        kt = _sharded_input(device, torch.nn.functional.pad(k_new, (0, 0, 0, PADDED_HEADS - NUM_KV_HEADS)).float())
        vt = _sharded_input(device, torch.nn.functional.pad(v_new, (0, 0, 0, PADDED_HEADS - NUM_KV_HEADS)).float())
        pos_tt = ttnn.Tensor(torch.tensor([pos], dtype=torch.int32), ttnn.int32).to(device)
        ttnn.experimental.paged_update_cache(
            k_cache, kt, update_idxs_tensor=pos_tt, page_table=decode_table, cache_position_modulo=ring_tokens
        )
        ttnn.experimental.paged_update_cache(
            v_cache, vt, update_idxs_tensor=pos_tt, page_table=decode_table, cache_position_modulo=ring_tokens
        )
        q = torch.randn(NUM_Q_HEADS, HEAD_DIM).bfloat16()
        qt = ttnn.Tensor(q.reshape(1, 1, NUM_Q_HEADS, HEAD_DIM).float(), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qt,
            k_cache,
            v_cache,
            cur_pos_tensor=pos_tt,
            page_table_tensor=decode_table,
            scale=scale,
            sliding_window_size=WINDOW,
            program_config=program_config,
            cache_position_modulo=ring_tokens,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        got = ttnn.to_torch(out)[0, 0, :NUM_Q_HEADS, :]
        ok, msg = comp_pcc(_reference(k_hist, v_hist, q, pos, scale), got, pcc=0.99)
        if not ok:
            failing.append((pos, msg))
    if failing:
        first = failing[0][0]
        pytest.fail(
            f"{len(failing)}/{steps} decode steps below PCC 0.99 (first at position {first}, prompt {prompt_len}, "
            f"ring {ring_tokens}):\n" + "\n".join(f"  pos={p}: {m}" for p, m in failing[:8])
        )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("ring_tokens", [768])
@pytest.mark.parametrize("users", [8, 16, 32])
def test_ring_decode_walk_with_many_users(device, ring_tokens, users):
    torch.manual_seed(1)
    ring_blocks = ring_tokens // BLOCK
    total_blocks = RING_BASE + users * ring_blocks
    scale = HEAD_DIM**-0.5
    prompt_lens = ([699, 799, 199, 1500, 128, 40, 2000, 900] * 4)[:users]
    steps = 450 if users <= 8 else 200

    k_cache = ttnn.Tensor(torch.zeros(total_blocks, NUM_KV_HEADS, BLOCK, HEAD_DIM).bfloat16(), ttnn.bfloat16)
    v_cache = ttnn.Tensor(torch.zeros(total_blocks, NUM_KV_HEADS, BLOCK, HEAD_DIM).bfloat16(), ttnn.bfloat16)
    k_cache = k_cache.to(ttnn.TILE_LAYOUT).to(device)
    v_cache = v_cache.to(ttnn.TILE_LAYOUT).to(device)

    columns = torch.arange(TABLE_COLUMNS, dtype=torch.int32) % ring_blocks
    tables = torch.stack([RING_BASE + user * ring_blocks + columns for user in range(users)])
    decode_table = ttnn.Tensor(tables, ttnn.int32).to(device)

    total = max(prompt_lens) + steps
    k_hist = torch.randn(users, total, NUM_KV_HEADS, HEAD_DIM).bfloat16()
    v_hist = torch.randn(users, total, NUM_KV_HEADS, HEAD_DIM).bfloat16()

    for user, prompt_len in enumerate(prompt_lens):
        prompt_blocks = (prompt_len + BLOCK - 1) // BLOCK
        fill_len = prompt_blocks * BLOCK
        table_tt = ttnn.Tensor(tables[user, :prompt_blocks].reshape(1, prompt_blocks).clone(), ttnn.int32).to(device)
        fill_kwargs = {"cache_position_modulo": ring_tokens} if fill_len > ring_tokens else {}
        for cache, hist in ((k_cache, k_hist), (v_cache, v_hist)):
            chunk = hist[user, :fill_len].permute(1, 0, 2).reshape(1, NUM_KV_HEADS, fill_len, HEAD_DIM)
            ttnn.experimental.paged_fill_cache(
                cache,
                ttnn.Tensor(chunk, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
                table_tt,
                batch_idx=0,
                **fill_kwargs,
            )

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
    )
    failing = []
    for step in range(steps):
        positions = torch.tensor([prompt_len + step for prompt_len in prompt_lens], dtype=torch.int32)
        k_new = torch.stack([k_hist[user, int(positions[user])] for user in range(users)]).reshape(
            1, users, NUM_KV_HEADS, HEAD_DIM
        )
        v_new = torch.stack([v_hist[user, int(positions[user])] for user in range(users)]).reshape(
            1, users, NUM_KV_HEADS, HEAD_DIM
        )
        kt = _sharded_input(device, torch.nn.functional.pad(k_new, (0, 0, 0, PADDED_HEADS - NUM_KV_HEADS)).float())
        vt = _sharded_input(device, torch.nn.functional.pad(v_new, (0, 0, 0, PADDED_HEADS - NUM_KV_HEADS)).float())
        pos_tt = ttnn.Tensor(positions, ttnn.int32).to(device)
        ttnn.experimental.paged_update_cache(
            k_cache, kt, update_idxs_tensor=pos_tt, page_table=decode_table, cache_position_modulo=ring_tokens
        )
        ttnn.experimental.paged_update_cache(
            v_cache, vt, update_idxs_tensor=pos_tt, page_table=decode_table, cache_position_modulo=ring_tokens
        )
        q = torch.randn(users, NUM_Q_HEADS, HEAD_DIM).bfloat16()
        qt = (
            ttnn.Tensor(q.reshape(1, users, NUM_Q_HEADS, HEAD_DIM).float(), ttnn.bfloat16)
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qt,
            k_cache,
            v_cache,
            cur_pos_tensor=pos_tt,
            page_table_tensor=decode_table,
            scale=scale,
            sliding_window_size=WINDOW,
            program_config=program_config,
            cache_position_modulo=ring_tokens,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        got = ttnn.to_torch(out)[0, :, :NUM_Q_HEADS, :]
        for user in range(users):
            pos = int(positions[user])
            ok, msg = comp_pcc(_reference(k_hist[user], v_hist[user], q[user], pos, scale), got[user], pcc=0.99)
            if not ok:
                failing.append((user, pos, msg))
    if failing:
        pytest.fail(
            f"{len(failing)} (user, step) pairs below PCC 0.99 with {users} users and ring {ring_tokens}:\n"
            + "\n".join(f"  user={u} pos={p}: {m}" for u, p, m in failing[:10])
        )
