# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Three remaining candidates, measured at Ornith's real shapes.

1. **``ttnn.conv1d`` for the depthwise causal conv.** ``models/demos/blackhole/qwen36`` drives its
   Gated-DeltaNet conv through ``ttnn.conv1d(groups=C)``; this checks whether Ornith's untensor-
   parallel ``conv_dim = 8192`` fits (the sibling implementation in
   ``models/experimental/gated_attention_gated_deltanet`` falls back to the FIR form above
   ``D = 2048`` with the note "native conv1d CBs overflow L1 at D=4096").

2. **Width-sharded RMSNorm for the residual-stream norms.** Decode normalises a single 32x2048
   tile, which the interleaved kernel runs on one core (20 us each, twice per step).

3. **Folding the SiLU into ``Conv1dConfig(activation=...)``** — the $graph-fusing "conv + activation"
   op-merging pattern. ``models/demos/blackhole/qwen36`` rejects it by comment; this measures it.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/probe_conv1d_and_norm.py
"""

import re
import time

import torch

import ttnn

CONV_DIM = 8192
KERNEL = 4
HIDDEN = 2048


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, memory_config=mc, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def pcc(a, b):
    a = a.double().flatten() - a.double().flatten().mean()
    b = b.double().flatten() - b.double().flatten().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def probe_conv1d(mesh, seq_len=2048, slices=0, shard=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, channels=CONV_DIM):
    """One conv1d call over ``channels`` of the depthwise stream.

    A depthwise conv is separable over channels, so the full 8192-channel stream can also be run as
    ``CONV_DIM // channels`` independent calls. That is the shape adaptation worth trying before
    concluding the op cannot serve this layer: the sibling implementation in
    ``models/experimental/gated_attention_gated_deltanet`` works up to ``D = 2048``.
    """
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(1, seq_len + KERNEL - 1, channels, generator=gen).to(torch.bfloat16)
    w = torch.randn(channels, 1, KERNEL, generator=gen).to(torch.bfloat16)
    ref = torch.nn.functional.conv1d(x.float().transpose(1, 2), w.float(), groups=channels).transpose(1, 2)
    try:
        xin = dev(mesh, x, layout=ttnn.ROW_MAJOR_LAYOUT)
        xin = ttnn.reshape(xin, (1, seq_len + KERNEL - 1, 1, channels))
        conv_cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=shard)
        slice_cfg = (
            ttnn.Conv2dL1FullSliceConfig
            if slices == 0
            else ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dSliceConfig.SliceTypeEnum.DRAMSliceWidth, num_slices=slices
            )
        )
        cc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        wprep = ttnn.prepare_conv_weights(
            # prepare_conv_weights requires a HOST weight tensor (prepare_conv2d_weights.cpp:1056).
            weight_tensor=ttnn.from_torch(
                w.reshape(channels, 1, 1, KERNEL), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            weights_format="OIHW",
            in_channels=channels,
            out_channels=channels,
            batch_size=1,
            input_height=1,
            input_width=seq_len + KERNEL - 1,
            kernel_size=(1, KERNEL),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            has_bias=False,
            groups=channels,
            device=mesh,
            input_dtype=ttnn.bfloat16,
            conv_config=conv_cfg,
            compute_config=cc,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        out = ttnn.conv1d(
            input_tensor=xin,
            weight_tensor=wprep,
            device=mesh,
            in_channels=channels,
            out_channels=channels,
            batch_size=1,
            input_length=seq_len + KERNEL - 1,
            kernel_size=KERNEL,
            stride=1,
            padding=0,
            dilation=1,
            groups=channels,
            dtype=ttnn.bfloat16,
            conv_config=conv_cfg,
            compute_config=cc,
            slice_config=slice_cfg,
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        got = (
            ttnn.to_torch(ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG))
            .float()
            .reshape(1, seq_len, channels)
        )
        print(f"CONV1D channels={channels} slices={slices} shard={shard} ok pcc={pcc(ref, got):.6f}")
    except Exception as exc:  # noqa: BLE001 - this probe exists to record the failure mode
        msg = str(exc).strip().splitlines()
        detail = msg[0] if msg else repr(exc)
        # Surface the requested allocation size, which is what shows whether slicing is helping.
        size = re.search(r"allocate (\d+) B", "\n".join(msg))
        if size:
            detail += f"  [requested {size.group(1)} B]"
        print(f"CONV1D channels={channels} slices={slices} shard={shard} FAILED: {detail}")


def probe_conv1d_split_timing(mesh, seq_len=2048, channels=4096, iters=5):
    """Time the working channel-split ``ttnn.conv1d`` against the shipped 4-tap FIR.

    ``channels=4096`` is the widest split that runs, so the full 8192-wide stream needs
    ``8192 // channels`` calls. Whether that is *worth* doing is a latency question, and this is
    where it gets answered rather than assumed.
    """
    calls = CONV_DIM // channels
    gen = torch.Generator().manual_seed(9)
    conv_cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    cc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    w = torch.randn(channels, 1, KERNEL, generator=gen).to(torch.bfloat16)
    wprep = ttnn.prepare_conv_weights(
        weight_tensor=ttnn.from_torch(
            w.reshape(channels, 1, 1, KERNEL), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        weights_format="OIHW",
        in_channels=channels,
        out_channels=channels,
        batch_size=1,
        input_height=1,
        input_width=seq_len + KERNEL - 1,
        kernel_size=(1, KERNEL),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        has_bias=False,
        groups=channels,
        device=mesh,
        input_dtype=ttnn.bfloat16,
        conv_config=conv_cfg,
        compute_config=cc,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    x_rm = dev(
        mesh,
        torch.randn(1, seq_len + KERNEL - 1, channels, generator=gen).to(torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    x_rm = ttnn.reshape(x_rm, (1, seq_len + KERNEL - 1, 1, channels))
    taps = [dev(mesh, torch.randn(1, 1, CONV_DIM, generator=gen).to(torch.bfloat16)) for _ in range(KERNEL)]
    padded_rm = dev(
        mesh,
        torch.randn(1, seq_len + KERNEL - 1, CONV_DIM, generator=gen).to(torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    # The FIR arm below must be the fallback that actually ships (fused_decoder.py `_gdn_prefill`),
    # not a generic 4-tap FIR: the shipped form accumulates with `addcmul` (one LLK op for these
    # dtypes, where `ttnn.mac` is always add(multiply()) — work_log §4.14) and reuses the already-TILE
    # input for the last tap instead of slicing and tilizing a fourth window. Timing anything else
    # overstates the arm this conv1d replaces.
    qkv_tile = dev(mesh, torch.randn(1, seq_len, CONV_DIM, generator=gen).to(torch.bfloat16))

    def one_conv():
        out = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=wprep,
            device=mesh,
            in_channels=channels,
            out_channels=channels,
            batch_size=1,
            input_length=seq_len + KERNEL - 1,
            kernel_size=KERNEL,
            stride=1,
            padding=0,
            dilation=1,
            groups=channels,
            dtype=ttnn.bfloat16,
            conv_config=conv_cfg,
            compute_config=cc,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.to_layout(
            ttnn.reshape(out, (1, seq_len, channels)), ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # The shipped path applies SiLU to the conv output, and so does the FIR arm below. Timing the
        # conv without it made the two arms non-comparable (round 15).
        activated = ttnn.silu(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return activated

    def fir():
        acc = None
        for tap in range(KERNEL):
            if tap == KERNEL - 1:
                piece, owned = qkv_tile, False
            else:
                piece, owned = (
                    ttnn.to_layout(
                        padded_rm[:, tap : tap + seq_len, :], ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    ),
                    True,
                )
            acc = (
                ttnn.multiply(piece, taps[tap], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if acc is None
                else ttnn.addcmul(acc, piece, taps[tap], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            if owned:
                ttnn.deallocate(piece)
        out = ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(acc)
        return out

    for name, fn, scale in (
        (f"conv1d x{calls} @{channels}ch", one_conv, calls),
        ("shipped-fallback FIR @8192ch", fir, 1),
    ):
        ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        start = time.time()
        for _ in range(iters):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        per = (time.time() - start) / iters * scale
        print(f"CONV1DTIME {name:24s} {per * 1e3:7.3f} ms per full 8192-channel conv over {seq_len} tokens", flush=True)


def probe_conv1d_fused_activation(mesh, seq_len=2048, channels=4096, iters=5):
    """Fold the SiLU into ``Conv1dConfig(activation=...)`` — the $graph-fusing "conv + activation" pattern.

    The shipped path runs ``conv1d`` and then a separate ``ttnn.silu``. ``Conv2dConfig`` (which
    ``Conv1dConfig`` aliases) carries an ``activation`` field, so the op-merging pattern is
    *expressible*; ``models/demos/blackhole/qwen36/tt/gdn/tp.py:367`` records that folding it into
    this depthwise conv "drops PCC to ~0.84". That is another model's note, not this stage's
    measurement, so it is measured here at Ornith's own shapes: same input, same weights, PCC of the
    folded output against ``silu(conv1d(x))`` computed on the same device, plus the latency of both
    arms. A rejection then rests on this stage's own evidence.
    """
    gen = torch.Generator().manual_seed(11)
    cc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    w = torch.randn(channels, 1, KERNEL, generator=gen).to(torch.bfloat16)
    x = torch.randn(1, seq_len + KERNEL - 1, channels, generator=gen).to(torch.bfloat16)
    ref = torch.nn.functional.silu(
        torch.nn.functional.conv1d(x.float().transpose(1, 2), w.float(), groups=channels).transpose(1, 2)
    )

    def build(activation):
        cfg = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            # Conv2dConfig.activation is a UnaryWithParam, not a string.
            **({} if activation is None else {"activation": ttnn.UnaryWithParam(activation)}),
        )
        wprep = ttnn.prepare_conv_weights(
            weight_tensor=ttnn.from_torch(
                w.reshape(channels, 1, 1, KERNEL), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            weights_format="OIHW",
            in_channels=channels,
            out_channels=channels,
            batch_size=1,
            input_height=1,
            input_width=seq_len + KERNEL - 1,
            kernel_size=(1, KERNEL),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            has_bias=False,
            groups=channels,
            device=mesh,
            input_dtype=ttnn.bfloat16,
            conv_config=cfg,
            compute_config=cc,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return cfg, wprep

    x_rm = dev(mesh, x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x_rm = ttnn.reshape(x_rm, (1, seq_len + KERNEL - 1, 1, channels))

    def run(cfg, wprep, fold):
        out = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=wprep,
            device=mesh,
            in_channels=channels,
            out_channels=channels,
            batch_size=1,
            input_length=seq_len + KERNEL - 1,
            kernel_size=KERNEL,
            stride=1,
            padding=0,
            dilation=1,
            groups=channels,
            dtype=ttnn.bfloat16,
            conv_config=cfg,
            compute_config=cc,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.to_layout(
            ttnn.reshape(out, (1, seq_len, channels)), ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if fold:
            return out
        activated = ttnn.silu(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return activated

    for name, activation, fold in (
        ("conv1d + separate silu (shipped)", None, False),
        ("conv1d(activation=silu) folded", ttnn.UnaryOpType.SILU, True),
    ):
        try:
            cfg, wprep = build(activation)
            got = ttnn.to_torch(run(cfg, wprep, fold)).float().reshape(1, seq_len, channels)
            ttnn.synchronize_device(mesh)
            start = time.time()
            for _ in range(iters):
                ttnn.deallocate(run(cfg, wprep, fold))
            ttnn.synchronize_device(mesh)
            per = (time.time() - start) / iters
            print(
                f"CONV1DACT {name:34s} pcc={pcc(ref, got):.6f}  {per * 1e3:7.3f} ms per {channels}-channel call",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - this probe exists to record the failure mode
            msg = str(exc).strip().splitlines()
            print(f"CONV1DACT {name:34s} FAILED: {msg[0] if msg else repr(exc)}", flush=True)


def probe_norm(mesh, batch=32):
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(1, 1, batch, HIDDEN, generator=gen).to(torch.bfloat16)
    w = torch.randn(1, 1, 1, HIDDEN, generator=gen).to(torch.bfloat16)
    xt, wt = dev(mesh, x), dev(mesh, w)
    ref = ttnn.to_torch(ttnn.rms_norm(xt, weight=wt, epsilon=1e-6)).float()

    def interleaved():
        return ttnn.rms_norm(xt, weight=wt, epsilon=1e-6)

    grid = mesh.compute_with_storage_grid_size()
    results = [("interleaved", interleaved, None)]
    for cores in (8, 16, 32, 64):
        width = HIDDEN // cores
        if width % 32:
            continue

        def sharded(cores=cores, width=width):
            shard_grid = ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(shard_grid, [batch, width], ttnn.ShardOrientation.ROW_MAJOR),
            )
            xs = ttnn.to_memory_config(xt, mc)
            rows = (cores + grid.x - 1) // grid.x
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(min(cores, grid.x), rows),
                subblock_w=1,
                block_h=max(1, batch // 32),
                block_w=width // 32,
                inplace=False,
            )
            out = ttnn.rms_norm(xs, weight=wt, epsilon=1e-6, program_config=pc, memory_config=mc)
            res = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(xs)
            ttnn.deallocate(out)
            return res

        results.append((f"width-sharded x{cores}", sharded, None))

    for name, fn, _ in results:
        try:
            out = fn()
            ttnn.synchronize_device(mesh)
            p = pcc(ref, ttnn.to_torch(out).float())
            iters = 50
            start = time.time()
            for _ in range(iters):
                o = fn()
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            per = (time.time() - start) / iters
            print(f"RMSNORM {name:22s} pcc={p:.6f} wall={per * 1e6:7.1f} us/call")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).strip().splitlines()
            print(f"RMSNORM {name:22s} FAILED: {msg[0] if msg else exc!r}")


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    try:
        # Sweep the DRAM slice count until the allocator's demand stops shrinking or the op
        # succeeds: a single failure at one slice count would not be an exhausted rejection, and
        # the required L1 buffer shrinks with the slice count.
        probe_conv1d(mesh)
        for slices in (2, 4, 8, 16, 32, 64, 128, 256):
            probe_conv1d(mesh, slices=slices)
        probe_conv1d(mesh, slices=8, shard=ttnn.TensorMemoryLayout.BLOCK_SHARDED)
        probe_conv1d(mesh, slices=64, shard=ttnn.TensorMemoryLayout.BLOCK_SHARDED)
        # A depthwise conv is separable over channels, so try running the 8192-channel stream as
        # several narrower calls -- 2048 is the width the sibling implementation is known to serve.
        for ch in (4096, 2048, 1024):
            probe_conv1d(mesh, channels=ch)
            probe_conv1d(mesh, channels=ch, slices=8)
        probe_conv1d_split_timing(mesh)
        probe_conv1d_fused_activation(mesh)
        probe_norm(mesh)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
