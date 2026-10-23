# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only partition and ownership checks; no device capacity/performance claims."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.autoports.ornith_ai_ornith_1_5_9b.tt import model
from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as decoder


def config():
    # Preserve head/group counts, using compact head/hidden dimensions for host tests.
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        intermediate_size=12288,
    )


def weights(full_attention):
    def values(*shape):
        return torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)

    state = {
        "mlp.gate_proj.weight": values(12288, 8),
        "mlp.up_proj.weight": values(12288, 8) + 1,
        "mlp.down_proj.weight": values(8, 12288),
        "input_layernorm.weight": values(8),
    }
    if full_attention:
        for role, rows in (("q_proj", 128), ("k_proj", 16), ("v_proj", 16)):
            state[f"self_attn.{role}.weight"] = values(rows, 8)
        state["self_attn.o_proj.weight"] = values(8, 64)
    else:
        for name, shape in {
            "in_proj_qkv.weight": (128, 8),
            "conv1d.weight": (128, 1, 4),
            "in_proj_z.weight": (64, 8),
            "in_proj_a.weight": (32, 8),
            "in_proj_b.weight": (32, 8),
            "A_log": (32,),
            "dt_bias": (32,),
            "out_proj.weight": (8, 64),
        }.items():
            state[f"linear_attn.{name}"] = values(*shape)
    return state


class MeshProfilesHostTest(unittest.TestCase):
    def test_complete_head_groups_and_nonmutating_config(self):
        cfg = config()
        before = copy.deepcopy(vars(cfg))
        for tp, expected in ((1, (16, 4, 16, 32, 12288)), (2, (8, 2, 8, 16, 6144)), (4, (4, 1, 4, 8, 3072))):
            with self.subTest(tp=tp):
                local = decoder.local_decoder_config(SimpleNamespace(text_config=cfg), tp)
                self.assertEqual(
                    (
                        local.num_attention_heads,
                        local.num_key_value_heads,
                        local.linear_num_key_heads,
                        local.linear_num_value_heads,
                        local.intermediate_size,
                    ),
                    expected,
                )
                self.assertEqual(local.num_attention_heads // local.num_key_value_heads, 4)
                self.assertEqual(local.linear_num_value_heads // local.linear_num_key_heads, 2)
        self.assertEqual(vars(cfg), before)

    def test_partition_reconstructs_every_weight_without_loss_or_duplication(self):
        for full in (False, True):
            original = weights(full)
            for tp in (1, 2, 4):
                with self.subTest(full_attention=full, tp=tp):
                    shards = [decoder.partition_state_dict(original, config(), rank, tp) for rank in range(tp)]
                    for key, value in original.items():
                        if key in ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight"):
                            groups = [shard[key].split((32 // tp, 32 // tp, 64 // tp)) for shard in shards]
                            restored = torch.cat([torch.cat([parts[g] for parts in groups]) for g in range(3)])
                        elif key == "input_layernorm.weight":
                            for shard in shards:
                                torch.testing.assert_close(shard[key], value, rtol=0, atol=0)
                            continue
                        else:
                            axis = (
                                1
                                if key
                                in ("mlp.down_proj.weight", "self_attn.o_proj.weight", "linear_attn.out_proj.weight")
                                else 0
                            )
                            restored = torch.cat([shard[key] for shard in shards], dim=axis)
                        torch.testing.assert_close(restored, value, rtol=0, atol=0)

    def test_query_and_gate_packing_preserves_global_head_order(self):
        state = weights(True)
        global_qg = state["self_attn.q_proj.weight"].reshape(16, 2, 4, 8)
        for tp in (1, 2, 4):
            local_cfg = decoder.local_decoder_config(config(), tp)
            for rank in range(tp):
                local = decoder.partition_state_dict(state, config(), rank, tp)
                packed = decoder._projection_weights(local, local_cfg)["qkvg"].T
                q, k, v, gate = packed.split((64 // tp, 16 // tp, 16 // tp, 64 // tp))
                heads = global_qg[rank * (16 // tp) : (rank + 1) * (16 // tp)]
                torch.testing.assert_close(q, heads[:, 0].reshape(-1, 8), rtol=0, atol=0)
                torch.testing.assert_close(gate, heads[:, 1].reshape(-1, 8), rtol=0, atol=0)
                torch.testing.assert_close(k, local["self_attn.k_proj.weight"], rtol=0, atol=0)
                torch.testing.assert_close(v, local["self_attn.v_proj.weight"], rtol=0, atol=0)

    def test_rejects_unsupported_geometry_before_device_operations(self):
        for shape in ((2, 2), (1, 3), (4, 1), (1,), (1, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                decoder.mesh_parallel_size(SimpleNamespace(shape=shape))
        for tp in (1, 2, 4):
            self.assertEqual(decoder.mesh_parallel_size(SimpleNamespace(shape=(1, tp))), tp)
        cfg = config()
        cfg.num_key_value_heads = 3
        with self.assertRaises(ValueError):
            decoder.local_decoder_config(cfg, 2)
        with self.assertRaises(ValueError):
            decoder.partition_state_dict(weights(True), config(), 2, 2)

    def test_single_chip_gather_owns_its_result_without_ccl(self):
        source, owned = object(), object()
        with (
            patch.object(model, "MeshCCLManager") as manager,
            patch.object(model.ttnn, "clone", return_value=owned) as clone,
        ):
            ccl = model.SamplingCCL(SimpleNamespace(shape=(1, 1)))
            result = ccl.line_all_gather(source, dim=3)
            self.assertIs(result, owned)
            self.assertIsNot(result, source)
            manager.assert_not_called()
            clone.assert_called_once_with(source, memory_config=model.ttnn.DRAM_MEMORY_CONFIG)

    def test_single_chip_projection_bypasses_reduction(self):
        layer = object.__new__(decoder.MultichipDecoder)
        layer.tp = 1
        layer.mesh_config = decoder.MeshConfig(decode_grid=None)
        source = SimpleNamespace(shape=(1, 1, 8))
        owned = object()
        with (
            patch.object(decoder.OptimizedDecoder, "_linear", return_value=owned),
            patch.object(decoder.ttnn, "all_reduce") as reduce,
        ):
            self.assertIs(layer._linear(source, "down_proj"), owned)
            reduce.assert_not_called()

    def test_single_chip_open_disables_fabric(self):
        with (
            patch.object(model.ttnn, "set_fabric_config") as fabric,
            patch.object(model.ttnn, "open_mesh_device") as open_mesh,
        ):
            model.open_ornith_mesh(num_devices=1)
            fabric.assert_called_once_with(model.ttnn.FabricConfig.DISABLED)
            self.assertEqual(tuple(open_mesh.call_args.args[0]), (1, 1))
        with patch.object(model.ttnn, "set_fabric_config") as fabric, self.assertRaises(ValueError):
            model.open_ornith_mesh(num_devices=3)
        fabric.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)


class StackStateL1Budget(unittest.TestCase):
    """Pinning the 24 recurrent states in L1 is a TP4-validated optimization only."""

    BANKS = 110  # Blackhole compute grid used for the per-bank estimate
    LAYERS = 24  # linear_attention layers in Ornith-1.5-9B

    def layer_bytes_per_bank(self, tp):
        # B1 prefill copy of one layer's recurrent state: [1, 32 // tp, 128, 128] fp32.
        tiles = (32 // tp) * 128 * 128 // 1024
        return -(-tiles // self.BANKS) * 4096

    def test_geometry_matches_measured_tp4_pool(self):
        self.assertEqual(self.layer_bytes_per_bank(4), 8192)  # 24 x 8 KiB = 192 KiB measured
        self.assertEqual(self.layer_bytes_per_bank(2), 12288)  # 24 x 12 KiB = 288 KiB
        self.assertEqual(self.layer_bytes_per_bank(1), 20480)  # 24 x 20 KiB = 480 KiB

    def test_only_tp4_pins_the_stack_in_l1(self):
        self.assertTrue(decoder.stack_state_fits_l1(self.layer_bytes_per_bank(4), self.LAYERS))
        # TP2 fit the old 300 KiB bound and collided with the LM head's circular buffers.
        self.assertFalse(decoder.stack_state_fits_l1(self.layer_bytes_per_bank(2), self.LAYERS))
        self.assertFalse(decoder.stack_state_fits_l1(self.layer_bytes_per_bank(1), self.LAYERS))

    def test_budget_is_the_working_set_bound(self):
        self.assertEqual(decoder.STACK_STATE_L1_BUDGET, 224 * 1024)
        self.assertLessEqual(self.LAYERS * self.layer_bytes_per_bank(4), decoder.STACK_STATE_L1_BUDGET)
        self.assertGreater(self.LAYERS * self.layer_bytes_per_bank(2), decoder.STACK_STATE_L1_BUDGET)
