import os
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn
from models.autoports.jaredpalmer_kev_9b.tt.loader import KevModelArgs
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.experimental.gated_attention_gated_deltanet.tt import ttnn_gated_deltanet as gdn_ops
from models.tt_transformers.tt.common import Mode

BLOCK_SIZE = 64
ALIGN = 128
MAX_QUESTION_LEN = 2048
BUCKETS = tuple(Qwen36Model._PREFILL_MASK_BUCKETS)
READ_ROWS = 32
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT
GDN_L1_SEQ_THRESHOLD = 256
MATMUL_POLICY = {
    (256, 4096, 12288, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 8), 8, 1, 1, 1, 35),
    (512, 4096, 12288, ttnn.bfloat8_b, ttnn.bfloat16): ("minimal",),
    (1024, 4096, 12288, ttnn.bfloat8_b, ttnn.bfloat16): ("minimal",),
    (2048, 4096, 12288, ttnn.bfloat8_b, ttnn.bfloat16): ("minimal",),
    (256, 4096, 12288, ttnn.bfloat4_b, ttnn.bfloat16): ("2d", (11, 10), 16, 1, 1, 1, 35),
    (512, 4096, 12288, ttnn.bfloat4_b, ttnn.bfloat16): ("2d", (11, 10), 16, 2, 1, 2, 35),
    (1024, 4096, 12288, ttnn.bfloat4_b, ttnn.bfloat16): ("minimal",),
    (2048, 4096, 12288, ttnn.bfloat4_b, ttnn.bfloat16): ("minimal",),
    (128, 12288, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 16, 1, 4, 1, 12),
    (256, 12288, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 8), 16, 1, 4, 1, 12),
    (512, 12288, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 16, 1, 4, 2, 12),
    (1024, 12288, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 16, 1, 4, 4, 12),
    (2048, 12288, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 16, 1, 4, 7, 12),
    (256, 4096, 12352, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 8, 1, 2, 1, 36),
    (512, 4096, 12352, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 8, 1, 4, 2, 36),
    (1024, 4096, 12352, ttnn.bfloat8_b, ttnn.bfloat16): ("minimal",),
    (2048, 4096, 12352, ttnn.bfloat8_b, ttnn.bfloat16): ("minimal",),
    (256, 4096, 8192, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 8, 1, 2, 1, 24),
    (512, 4096, 8192, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 16, 1, 4, 2, 24),
    (1024, 4096, 8192, ttnn.bfloat8_b, ttnn.bfloat16): ("minimal",),
    (2048, 4096, 8192, ttnn.bfloat8_b, ttnn.bfloat16): ("minimal",),
    (128, 4096, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 16, 1, 2, 1, 12),
    (256, 4096, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 8), 8, 1, 2, 1, 12),
    (512, 4096, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 16, 2, 2, 2, 12),
    (1024, 4096, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 8), 16, 1, 4, 4, 12),
    (2048, 4096, 4096, ttnn.bfloat8_b, ttnn.bfloat16): ("2d", (11, 10), 16, 1, 4, 7, 12),
}


@dataclass
class StateHandle:
    S: int
    S0: int
    suffix_ids: torch.Tensor
    slot: int = 0


def aligned_pieces(length):
    out = []
    for b in sorted(BUCKETS, reverse=True):
        while length >= b:
            out.append(b)
            length -= b
    assert length == 0, f"length not a multiple of {ALIGN}"
    return out


_original_linear = ttnn.linear


def policy_linear(a, b, *args, **kw):
    pol = None
    if (
        not args
        and kw.get("bias") is None
        and kw.get("program_config") is None
        and len(a.shape) == 3
        and a.shape[0] == 1
        and len(b.shape) == 2
    ):
        pol = MATMUL_POLICY.get((a.shape[1], a.shape[2], b.shape[1], b.dtype, a.dtype))
    if pol is None:
        return _original_linear(a, b, *args, **kw)
    kw = dict(kw)
    kw.pop("program_config", None)
    kw.pop("bias", None)
    activation = kw.pop("activation", None)
    if a.shape[1] >= 256:
        kw["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    if pol[0] == "minimal":
        out = ttnn.experimental.minimal_matmul(a, b, **kw)
    else:
        _, grid, in0, sub_h, sub_w, pcm, pcn = pol
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=in0,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_M=pcm,
            per_core_N=pcn,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        out = _original_linear(a, b, program_config=cfg, **kw)
    if activation is not None:
        assert activation == "silu", activation
        act = ttnn.silu(out, memory_config=kw.get("memory_config"))
        ttnn.deallocate(out)
        out = act
    return out


class KevEngine:
    def __init__(
        self,
        device,
        args_cls=KevModelArgs,
        max_state_len=65536,
        n_layers=None,
        chunk_size=2048,
        snapshot_slots=8,
        traced=True,
        read_rows=READ_ROWS,
        kv_reserve_bytes=2 << 30,
        matmul_policy=None,
    ):
        assert chunk_size == max(BUCKETS), f"chunk_size must be {max(BUCKETS)}"
        self.device = device
        self.chunk_size = chunk_size
        self.traced = traced
        self.read_rows = read_rows
        self.matmul_policy = os.environ.get("KEV_MATMUL_POLICY", "1") == "1" if matmul_policy is None else matmul_policy
        ttnn.linear = policy_linear if self.matmul_policy else _original_linear
        if self.matmul_policy:
            os.environ.setdefault("QWEN9B_MLP_DOWN_AUTO", "1")
        self.max_len = max_state_len + MAX_QUESTION_LEN
        args = args_cls(mesh_device=device, max_batch_size=1, max_seq_len=self.max_len)
        if n_layers is not None:
            args.n_layers = n_layers
            args.attention_type_list = args.attention_type_list[:n_layers]
        self.args = args
        logger.info(
            f"KevEngine args={type(args).__name__} adapter_sha8={getattr(args, 'adapter_sha8', None)} "
            f"cache={args.weight_cache_path()} max_len={self.max_len} traced={traced} matmul_policy={self.matmul_policy}"
        )
        state_dict = args.load_state_dict()
        self.model = Qwen36Model(device, args, state_dict, tensor_cache_path=args.weight_cache_path())
        del state_dict
        self.blocks_per_slot = self.max_len // BLOCK_SIZE
        self.snapshot_slots = self._fit_slots(snapshot_slots, kv_reserve_bytes)
        kv_shape = [self.snapshot_slots * self.blocks_per_slot, args.n_kv_heads, BLOCK_SIZE, args.head_dim]
        self.model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
        self.page_tables = [
            torch.arange(s * self.blocks_per_slot, (s + 1) * self.blocks_per_slot, dtype=torch.int32).unsqueeze(0)
            for s in range(self.snapshot_slots)
        ]
        self.page_table = self.page_tables[0]
        self.gdn_layers = [layer.attention for layer in self.model.layers if not layer.is_full_attention]
        for dn in self.gdn_layers:
            dn._chunk_inplace_state = True
        self.model._init_dn_zero_buffers()
        self.snapshots = [
            [(self._zeros_like(dn.recurrent_state), self._zeros_like(dn.fused_conv_state)) for dn in self.gdn_layers]
            for _ in range(self.snapshot_slots)
        ]
        self.select_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.traces = {}
        if traced:
            self._setup_traces()

    def _fit_slots(self, requested, reserve):
        n_attn = sum(1 for layer in self.model.layers if layer.is_full_attention)
        per_slot = 2 * n_attn * self.blocks_per_slot * self.args.n_kv_heads * BLOCK_SIZE * self.args.head_dim * 2
        view = ttnn.get_memory_view(self.device, ttnn.BufferType.DRAM)
        free = int(view.total_bytes_free_per_bank) * int(view.num_banks)
        slots = max(1, min(requested, (free - reserve) // per_slot))
        logger.info(
            f"KV per slot {per_slot / 2**30:.2f} GiB, DRAM free {free / 2**30:.2f} GiB, "
            f"reserve {reserve / 2**30:.2f} GiB: {slots} slot(s) of {requested} requested"
        )
        return slots

    def _zeros_like(self, t):
        return ttnn.zeros(
            list(t.shape), dtype=t.dtype, layout=TILE, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _dev(self, t, dtype, layout):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device)

    def _write(self, t, buf, dtype, layout):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dtype, layout=layout), buf)

    def _setup_traces(self):
        dev = self.device
        gdn_ops._L1_SEQ_THRESHOLD = min(gdn_ops._L1_SEQ_THRESHOLD, GDN_L1_SEQ_THRESHOLD)
        rd = self.args.rope_head_dim
        dim = self.args.dim
        R = self.read_rows
        self.model._build_request_rope(torch.zeros(1, 1, dtype=torch.long), None)
        self.csi = self._dev(torch.zeros(1, dtype=torch.int32), ttnn.int32, RM)
        self.pt_full = self._dev(self.page_tables[0], ttnn.int32, RM)
        self.pt_host = [ttnn.from_torch(pt, dtype=ttnn.int32, layout=RM) for pt in self.page_tables]
        self.bufs = {}
        for b in BUCKETS:
            self.bufs[b] = {
                "tok": self._dev(torch.zeros(1, b, dtype=torch.int32), ttnn.uint32, RM),
                "cpt": self._dev(self.page_tables[0][:, : b // BLOCK_SIZE].contiguous(), ttnn.int32, RM),
                "cos": self._dev(torch.zeros(1, b, rd, dtype=torch.bfloat16), ttnn.bfloat16, TILE),
                "sin": self._dev(torch.zeros(1, b, rd, dtype=torch.bfloat16), ttnn.bfloat16, TILE),
                "sel": self._dev(torch.zeros(1, R, b, dtype=torch.bfloat16), ttnn.bfloat16, TILE),
                "hidden": ttnn.zeros(
                    [1, b, dim], dtype=ttnn.bfloat16, layout=TILE, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
                ),
                "rows": ttnn.zeros(
                    [1, R, dim], dtype=ttnn.bfloat16, layout=TILE, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
                ),
            }
        self.model._reset_dn_state_inplace()
        self._copy_state(0, to_live=False)
        self._copy_state(0, to_live=True)
        for b in BUCKETS:
            logger.info(f"warming bucket {b}")
            self._forward_body(b)
            self._gather_body(b)
        ttnn.synchronize_device(dev)
        dev.set_program_cache_misses_allowed(False)
        for b in BUCKETS:
            self.traces[("fwd", b)] = self._capture(lambda: self._forward_body(b))
            self.traces[("gather", b)] = self._capture(lambda: self._gather_body(b))
        for slot in range(self.snapshot_slots):
            self.traces[("restore", slot)] = self._capture(lambda: self._copy_state(slot, to_live=True))
            self.traces[("save", slot)] = self._capture(lambda: self._copy_state(slot, to_live=False))
        self.traces[("zero",)] = self._capture(self.model._reset_dn_state_inplace)
        ttnn.synchronize_device(dev)
        view = ttnn.get_memory_view(dev, ttnn.BufferType.TRACE)
        self.trace_bytes = int(view.total_bytes_allocated_per_bank) * int(view.num_banks)
        logger.info(f"{len(self.traces)} traces captured, trace region used {self.trace_bytes / 2**20:.1f} MiB")

    def _capture(self, body):
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        body()
        ttnn.end_trace_capture(self.device, tid, cq_id=0)
        return tid

    def _replay(self, key):
        ttnn.execute_trace(self.device, self.traces[key], cq_id=0, blocking=False)

    def _forward_body(self, b):
        B = self.bufs[b]
        x = self.model._forward_prefill_chunk(B["tok"], B["cos"], B["sin"], self.csi, self.pt_full, B["cpt"])
        ttnn.copy(x, B["hidden"])
        ttnn.deallocate(x)

    def _gather_body(self, b):
        B = self.bufs[b]
        x = ttnn.matmul(B["sel"], B["hidden"], compute_kernel_config=self.select_kernel_config)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.model.norm(x, mode=Mode.PREFILL)
        ttnn.copy(x, B["rows"])
        ttnn.deallocate(x)

    def _copy_state(self, slot, to_live):
        for dn, (rec, conv) in zip(self.gdn_layers, self.snapshots[slot]):
            if to_live:
                ttnn.copy(rec, dn.recurrent_state)
                ttnn.copy(conv, dn.fused_conv_state)
            else:
                ttnn.copy(dn.recurrent_state, rec)
                ttnn.copy(dn.fused_conv_state, conv)

    def _begin_sequence(self, token_ids):
        if self.traced:
            self._replay(("zero",))
        else:
            self.model._reset_gdn_state_for_new_sequence()
        self.model._build_request_rope(token_ids, None)

    def _segment(self, token_ids, chunk_start, slot):
        length = token_ids.shape[1]
        b = Qwen36Model._mask_bucket_for(length)
        assert b in self.bufs, f"segment of {length} tokens has no bucket"
        assert chunk_start % ALIGN == 0
        B = self.bufs[b]
        tok = torch.zeros(1, b, dtype=torch.int32)
        tok[:, :length] = token_ids.to(torch.int32)
        self._write(tok, B["tok"], ttnn.uint32, RM)
        self._write(torch.tensor([chunk_start], dtype=torch.int32), self.csi, ttnn.int32, RM)
        ttnn.copy_host_to_device_tensor(self.pt_host[slot], self.pt_full)
        blk0 = chunk_start // BLOCK_SIZE
        self._write(self.page_tables[slot][:, blk0 : blk0 + b // BLOCK_SIZE].contiguous(), B["cpt"], ttnn.int32, RM)
        cos, sin = self.model.rope.prefill_cos_sin_torch(chunk_start, b)
        self._write(cos.unsqueeze(0).contiguous(), B["cos"], ttnn.bfloat16, TILE)
        self._write(sin.unsqueeze(0).contiguous(), B["sin"], ttnn.bfloat16, TILE)
        self._replay(("fwd", b))
        return b

    def _gather(self, b, rows):
        B = self.bufs[b]
        R = self.read_rows
        out = torch.empty(len(rows), self.args.dim, dtype=torch.float32)
        for i in range(0, len(rows), R):
            group = rows[i : i + R]
            sel = torch.zeros(1, R, b, dtype=torch.float32)
            sel[0, torch.arange(len(group)), torch.tensor(group)] = 1.0
            self._write(sel, B["sel"], ttnn.bfloat16, TILE)
            self._replay(("gather", b))
            out[i : i + len(group)] = ttnn.to_torch(B["rows"])[0, : len(group)].float()
        return out

    def _run_segment(self, token_ids, chunk_start, slot):
        length = token_ids.shape[1]
        bucket = Qwen36Model._mask_bucket_for(length)
        buf = torch.zeros(1, bucket, dtype=torch.int32)
        buf[:, :length] = token_ids.to(torch.int32)
        return self.model._forward_prefill_chunk_masked(buf, length, chunk_start, self.page_tables[slot], bucket)

    def _read_rows(self, hidden, rows):
        bucket = hidden.shape[1]
        sel = torch.zeros(1, len(rows), bucket, dtype=torch.float32)
        sel[0, torch.arange(len(rows)), torch.tensor(rows)] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=hidden.dtype, layout=TILE, device=self.device)
        x = ttnn.matmul(sel_tt, hidden, compute_kernel_config=self.select_kernel_config)
        ttnn.deallocate(sel_tt)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.model.norm(x, mode=Mode.PREFILL)
        out = ttnn.to_torch(x)[0, : len(rows)].float()
        ttnn.deallocate(x)
        return out

    def _segment_rows(self, token_ids, chunk_start, slot, rows):
        if self.traced:
            b = self._segment(token_ids, chunk_start, slot)
            return self._gather(b, rows) if rows else None
        hidden = self._run_segment(token_ids, chunk_start, slot)
        out = self._read_rows(hidden, rows) if rows else None
        ttnn.deallocate(hidden)
        return out

    def prefill_hidden(self, token_ids, positions, slot=0):
        T = token_ids.shape[1]
        assert T <= self.max_len, f"T={T} exceeds max_len={self.max_len}"
        assert 0 <= slot < self.snapshot_slots, f"slot {slot} out of range"
        self._begin_sequence(token_ids)
        out = torch.empty(len(positions), self.args.dim, dtype=torch.float32)
        for cs in range(0, T, self.chunk_size):
            ce = min(cs + self.chunk_size, T)
            idx = [i for i, p in enumerate(positions) if cs <= p < ce]
            got = self._segment_rows(token_ids[:, cs:ce], cs, slot, [positions[i] - cs for i in idx])
            if idx:
                out[idx] = got
        return out

    def prefill_state(self, state_ids, slot=0):
        S = state_ids.shape[1]
        S0 = (S // ALIGN) * ALIGN
        assert S <= self.max_len - MAX_QUESTION_LEN, f"S={S} exceeds max_state_len"
        assert 0 <= slot < self.snapshot_slots, f"slot {slot} out of range"
        self._begin_sequence(state_ids)
        if self.traced:
            full = (S0 // self.chunk_size) * self.chunk_size
            cs = 0
            for b in [self.chunk_size] * (full // self.chunk_size) + aligned_pieces(S0 - full):
                self._segment(state_ids[:, cs : cs + b], cs, slot)
                cs += b
            self._replay(("save", slot))
        else:
            for cs in range(0, S0, self.chunk_size):
                ce = min(cs + self.chunk_size, S0)
                ttnn.deallocate(self._run_segment(state_ids[:, cs:ce], cs, slot))
            self._copy_state(slot, to_live=False)
        return StateHandle(S, S0, state_ids[:, S0:S].clone(), slot)

    def question_hidden(self, handle, question_ids, positions_in_question):
        Q = question_ids.shape[1]
        assert handle.S + Q <= self.max_len, f"S+Q={handle.S + Q} exceeds max_len={self.max_len}"
        tail = torch.cat([handle.suffix_ids, question_ids], dim=1)
        offset = handle.S - handle.S0
        rows = [offset + p for p in positions_in_question]
        if self.traced:
            self._replay(("restore", handle.slot))
        else:
            self._copy_state(handle.slot, to_live=True)
        return self._segment_rows(tail, handle.S0, handle.slot, rows)
