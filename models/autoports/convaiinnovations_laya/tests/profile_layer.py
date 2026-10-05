# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import time

import torch

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--port", default="{}", help="JSON of PortConfig overrides")
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-head-layer", action="store_true")
    ap.add_argument("--signpost", action="store_true")
    a = ap.parse_args()

    import ttnn
    from models.autoports.convaiinnovations_laya.tests import laya_inputs as LI
    from models.autoports.convaiinnovations_laya.tt import model_config as mc
    from models.autoports.convaiinnovations_laya.tt.laya_head import TtnnHeadLayer
    from models.autoports.convaiinnovations_laya.tt.modernbert_layer import TtnnModernBertEncoderLayer
    from models.autoports.convaiinnovations_laya.tt.modernbert_masks import TtnnMaskBuilder, deallocate_masks
    from models.autoports.convaiinnovations_laya.tt.modernbert_rope import TtnnModernBertRotary
    from models.autoports.convaiinnovations_laya.tt.weights import (
        deallocate_weights,
        load_state_dict,
        prepare_head_weights,
        prepare_weights,
        split_state_dict,
    )

    policy = mc.policy_from_name(a.policy)
    port = mc.DEFAULT_PORT.with_(**json.loads(a.port))
    config = LI.load_config()
    sd = load_state_dict()
    parts = split_state_dict(sd)
    b = LI.build_inputs(batch_size=a.batch, seq_len=a.seq, fill=a.batch > 1)

    device = ttnn.open_device(device_id=0, l1_small_size=79104, trace_region_size=0)
    report = {"batch": a.batch, "seq": a.seq, "layer": a.layer, "policy": policy.describe(), "port": port.describe()}
    try:
        params = prepare_weights(parts["encoder"], config, device, policy, layers=[a.layer])
        head_params = prepare_head_weights(sd, config, device, policy) if a.with_head_layer else None
        plan = mc.bucket_plan(device, config, a.batch, a.seq, policy, port)
        report["plan"] = mc.describe_plan(plan)
        rotary = TtnnModernBertRotary(config, device, a.seq, batch_size=a.batch, port=port, attention_memory=plan.attention_memory)
        builder = TtnnMaskBuilder(config, device, a.seq, a.batch)
        pad = builder.upload_pad_row(b["attention_mask"])
        masks = builder.build(pad)
        layer = TtnnModernBertEncoderLayer(params["layers"][a.layer], config, a.layer, plan, device, policy, port)
        head_layer = TtnnHeadLayer(head_params["layers"][0], config, plan, device, policy) if head_params else None
        torch.manual_seed(0)
        x = torch.randn(a.batch, a.seq, config.hidden_size) * 0.5
        times = []
        for i in range(a.repeats):
            tt_x = ttnn.from_torch(x, dtype=mc.ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
            if layer.resident:
                sh = ttnn.to_memory_config(tt_x, plan.mlp_shard.hidden_memory)
                ttnn.deallocate(tt_x)
                tt_x = sh
            if a.signpost and i == a.repeats - 1:
                ttnn.synchronize_device(device)
                try:
                    from tracy import signpost

                    signpost("layer_start")
                except ImportError:
                    pass
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = layer(tt_x, rotary, masks[layer.layer_type])
            if head_layer is not None:
                out2 = head_layer(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG) if layer.resident else out, masks[mc.FULL_ATTENTION])
                ttnn.deallocate(out)
                out = out2
            ttnn.synchronize_device(device)
            times.append((time.perf_counter() - t0) * 1000.0)
            ttnn.deallocate(out)
        report["layer_ms"] = times
        report["layer_ms_min"] = min(times)
        report["layer_ms_median"] = sorted(times)[len(times) // 2]
        deallocate_masks(masks)
        ttnn.deallocate(pad)
        builder.deallocate()
        rotary.deallocate()
        deallocate_weights(params)
        if head_params:
            deallocate_weights({"layers": head_params["layers"], "scorer": head_params["scorer"], "type_emb": head_params["type_emb"]})
    finally:
        ttnn.close_device(device)
    print("PROFILE_LAYER", json.dumps(report))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
