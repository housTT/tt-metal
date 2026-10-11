# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-instance request, page ownership, continuation, and decode trace reuse checks."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding

from ..tt.functional_decoder import FunctionalDecoder
from .hf_reference import decode_reference, prefill_reference
from .layer_weights import load_layer_state_dict
from .run_layer import pcc
from .runtime_audit import device_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, choices=(0, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--no-kv-scratch", action="store_true")
    parser.add_argument("--expert-precision", choices=("inherited", "bf16"), default="inherited")
    parser.add_argument("--decoder", choices=("functional", "fused", "optimized", "multichip"), default="functional")
    parser.add_argument("--mesh", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--fusions")
    parser.add_argument("--optimization-json")
    parser.add_argument("--input-activations", type=Path)
    args = parser.parse_args()
    torch.manual_seed(913)
    torch.set_num_threads(8)
    checkpoint = Path("../weights/Qwen3.5-35B-A3B")
    cfg = AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    state = load_layer_state_dict(checkpoint, args.layer)
    with torch.device("meta"):
        hf = Qwen3_5MoeDecoderLayer(cfg, args.layer)
    prefix = f"model.language_model.layers.{args.layer}."
    hf.load_state_dict({k.removeprefix(prefix): v for k, v in state.items()}, assign=True)
    hf.eval()
    rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
    specifications = [(n, ()) for n in (1, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1153, 33, 35, 36)]
    specifications += [(1153, (1, 3, 127, 129, 160, 513)), (256, (35,)), (256, (36,)), (2047, (1152,))]
    cases = []
    capacity = 2048
    pages = capacity // args.page_size
    total_pages = args.batch * pages
    recorded = None
    if args.input_activations:
        recorded = torch.load(args.input_activations, weights_only=True)
        assert recorded["metadata"]["layer"] == args.layer
        recorded = recorded["activations"][0]
    for index, (length, cuts) in enumerate(specifications):
        if recorded is not None:
            starts = [((index * args.batch + row) * 997) % (recorded.shape[0] - length) for row in range(args.batch)]
            x = torch.stack([recorded[start : start + length + 1] for start in starts]).contiguous()
        else:
            x = torch.randn(args.batch, length + 1, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
        ref, cache, _ = prefill_reference(hf, cfg, rope, x[:, :length], args.layer)
        dec = decode_reference(hf, rope, cache, x[:, length:], length)
        boundaries = (0,) + cuts + (length,)
        table = torch.randperm(total_pages, dtype=torch.int32).reshape(args.batch, pages)
        cases.append(dict(length=length, cuts=cuts, x=x, ref=ref, dec=dec, table=table, boundaries=boundaries))
    print("REFERENCES_READY", flush=True)
    from ttnn.tools import trace_allocation_tracker

    import ttnn

    from .mesh_util import ReplicaReader, decoder_class as select_decoder, open_mesh, shards

    if args.mesh != 1 and args.decoder != "multichip":
        raise ValueError("Only the multichip decoder opens a multi-device mesh")
    mesh = open_mesh(args.mesh)
    replica = ReplicaReader()

    def put(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=True):
        kw = dict(device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) if device else {}
        return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=layout, **kw)

    def get(x):
        return replica(x)

    def nonzero_any_device(x, value=None):
        return any(bool(torch.count_nonzero(s if value is None else s != value).item()) for s in shards(x))

    try:
        decoder_class, fusion_kwargs = select_decoder(args.decoder), {}
        if args.decoder in ("fused", "optimized", "multichip"):
            if args.fusions is not None:
                fusion_kwargs["fusions"] = tuple(filter(None, args.fusions.split(",")))
        if args.decoder in ("optimized", "multichip"):
            if args.optimization_json is not None:
                fusion_kwargs["optimization"] = json.loads(args.optimization_json)
        model = decoder_class.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            expert_precision=args.expert_precision,
            **fusion_kwargs,
        )
        slots = tuple(range(32 - args.batch, 32))
        kwargs = dict(request_slots=slots)
        decode_kwargs = dict(request_slots=slots)
        dx = put(cases[0]["x"][:, -1:])
        if args.layer == 3:
            kv_host = torch.zeros((total_pages + 4, model.config.n_kv_heads, args.page_size, cfg.head_dim),
                                  dtype=torch.bfloat16)
            kv_host[total_pages + 3].fill_(5.0)
            kv_dtype = (ttnn.bfloat8_b if getattr(getattr(model, "optimization", None), "kv_dtype", "bf16") == "bfp8"
                        else ttnn.bfloat16)
            kc, vc = put(kv_host, kv_dtype), put(kv_host, kv_dtype)
            if args.decoder in ("optimized", "multichip") and not args.no_kv_scratch:
                model.bind_paged_kv_cache(kc, vc, scratch_block=total_pages + 3)
            else:
                model.bind_paged_kv_cache(kc, vc)
            pt = put(cases[0]["table"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            pos = put(torch.ones(args.batch, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            kwargs["page_table"] = pt
            decode_kwargs.update(page_table=pt, position_tensor=pos)
        for case in cases:
            bounds = case["boundaries"]
            case["pieces"] = [
                (
                    a,
                    put(
                        torch.nn.functional.pad(case["x"][:, a:b], (0, 0, 0, ((b - a + 127) // 128) * 128 - (b - a)))
                        if case["length"] in (35, 36, 256)
                        else case["x"][:, a:b]
                    ),
                    b - a,
                )
                for a, b in zip(bounds[:-1], bounds[1:])
            ]
            case["dx_host"] = put(case["x"][:, -1:], device=False)
            case["pt_host"] = put(case["table"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device=False)
            case["pos_host"] = put(
                torch.full((args.batch,), case["length"], dtype=torch.int32),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
                device=False,
            )

        def prefill(case):
            model.reset_request_state(slots)
            if args.layer == 3:
                ttnn.copy_host_to_device_tensor(case["pt_host"], pt)
                ttnn.copy_host_to_device_tensor(case["pos_host"], pos)
            ttnn.copy_host_to_device_tensor(case["dx_host"], dx)
            outputs = []
            for start, px, valid in case["pieces"]:
                with device_only():
                    y = model.prefill_forward(px, valid_len=valid, start_pos=start, **kwargs)
                outputs.append(get(y))
                y.deallocate(True)
            return torch.cat(outputs, dim=1)

        # Every eager signature/state copy is prepared before the one lifetime trace.
        for case in cases:
            if case["length"] == 36 or case["cuts"] == (36,):
                continue  # A new logical length in the already prepared 128-token input bucket.
            prefill(case)
            with device_only():
                y = model.decode_forward(dx, **decode_kwargs)
            get(y)
            y.deallocate(True)
        ttnn.synchronize_device(mesh)
        mesh.set_program_cache_misses_allowed(False)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            trace_out = model.decode_forward(dx, **decode_kwargs)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        trace_allocation_tracker.acknowledge_corruptible(trace_out)
        results = []
        for case in cases:
            actual = prefill(case)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            actual_dec = get(trace_out)
            entry = dict(
                length=case["length"],
                padded_input=case["length"] in (35, 36, 256),
                warmed=case["length"] != 36 and case["cuts"] != (36,),
                cuts=case["cuts"],
                batch=args.batch,
                prefill_pcc=pcc(case["ref"], actual),
                prefill_pcc_min=min(pcc(case["ref"][r], actual[r]) for r in range(args.batch)),
                decode_pcc=pcc(case["dec"], actual_dec),
                decode_pcc_min=min(pcc(case["dec"][r], actual_dec[r]) for r in range(args.batch)),
                trace_id=str(trace_id),
            )
            results.append(entry)
            print(json.dumps(entry), flush=True)
            assert entry["prefill_pcc_min"] >= 0.995 and entry["decode_pcc_min"] >= 0.995
        ttnn.release_trace(mesh, trace_id)
        mesh.set_program_cache_misses_allowed(True)
        guards_clean = None
        if args.layer == 3:
            guards_clean = (not nonzero_any_device(kc[total_pages : total_pages + 3])
                            and not nonzero_any_device(vc[total_pages : total_pages + 3]))
            assert guards_clean
        args.output.write_text(
            json.dumps(
                dict(
                    decoder_class=type(model).__name__,
                    input_activations=str(args.input_activations) if args.input_activations else "random_normal_0.1",
                    optimization=(__import__("dataclasses").asdict(model.optimization)
                                  if args.decoder in ("optimized", "multichip") else None),
                    mesh_shape=list(mesh.shape),
                    device_ids=list(mesh.get_device_ids()),
                    tensor_parallel=getattr(model, "tensor_parallel", None),
                    output_replicas_identical=replica.all_identical(),
                    fusions=getattr(model, "fusions", None),
                    cases=results,
                    captures=1,
                    request_releases=0,
                    teardown_releases=1,
                    guards_clean=guards_clean,
                    kv_scratch_block=getattr(model, "kv_scratch_block", None),
                    kv_scratch_written=(nonzero_any_device(kc[total_pages + 3 : total_pages + 4], 5.0)
                                        if args.layer == 3 and getattr(model, "kv_scratch_block", None) is not None
                                        else None),
                    runtime_audit=True,
                    request_slots=slots,
                    page_size=args.page_size,
                    expert_precision=args.expert_precision,
                    source_sha256={
                        p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__).resolve().parents[1] / "tt").glob("*.py")
                    },
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
