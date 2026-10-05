import contextlib
import hashlib
import os
import pathlib
import time
import traceback
from dataclasses import dataclass

import torch
from loguru import logger

os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")

import ttnn
from models.autoports.cloudflare_clef.tt import precision_defaults

precision_defaults.apply()

from models.autoports.cloudflare_clef.tt import encode as clef_encode
from models.autoports.cloudflare_clef.tt import head as clef_head
from models.autoports.cloudflare_clef.tt import vision as clef_vision
from models.autoports.cloudflare_clef.tt.loader import ClefModelArgs
from models.demos.blackhole.qwen36.tt import mlp as qwen36_mlp
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import Mode

BLOCK_SIZE = 64
ALIGN = 128
MAX_BUCKET = 1024
BUCKETS = tuple(b for b in Qwen36Model._PREFILL_MASK_BUCKETS if b <= MAX_BUCKET)
TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT
LM_HEAD_STUB_ROWS_PER_DEVICE = 32
CACHE_RELOAD_MODES = ("off", "host_to_device", "stock")
VISION_MAX_SEQ_LEN = 4096
MEDIA_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
SPARE_BLOCKS = MAX_BUCKET // BLOCK_SIZE
REPLAY_COST_S = {128: 0.173, 256: 0.187, 512: 0.220, 1024: 0.338}
EAGER_COST_S = {128: 0.172, 256: 0.191, 512: 0.226, 1024: 0.345}
TRACE_GUARD_BYTES = 2 << 30
REFERENCE_GRIDS = "1,16,20;1,22,38;1,26,36;1,28,36;1,28,38;1,40,50;2,16,20"
WARM_GRID_HELP = (
    "CLEF_TRACED=1 serves only image grids compiled before the traces were captured: set "
    "CLEF_VISION_WARM_GRID to a ;-separated list of t,h,w grids (for the 8 reference images and the video: "
    f"{REFERENCE_GRIDS}), or run eager with CLEF_TRACED=0 (the default), which accepts any grid"
)


def single_grid_message(n_grids):
    return (
        f"this request carries {n_grids} image or video grids; a traced server (CLEF_TRACED=1) takes one grid "
        "per request, because the tower joins several images with a concat program compiled per image count "
        "and no vision program may compile while traces are live: send one image (or one video) per request, "
        "or serve with CLEF_TRACED=0 (the default), which accepts several"
    )


def parse_grids(text):
    return [tuple(int(g) for g in part.split(",")) for part in text.split(";") if part.strip()]


def aligned_pieces(length):
    out = []
    for b in sorted(BUCKETS, reverse=True):
        while length >= b:
            out.append(b)
            length -= b
    assert length == 0, f"length {length + sum(out)} is not a multiple of {ALIGN}"
    return out


@dataclass
class VisionRequest:
    tokens: object
    grid: torch.Tensor
    video: bool
    n_rows: int
    n_patches: int
    seconds: float
    rows_host: torch.Tensor = None

    def release(self):
        if self.tokens is not None:
            ttnn.deallocate(self.tokens)
            self.tokens = None
        self.rows_host = None

    @property
    def marker(self):
        return self.tokens if self.tokens is not None else self.rows_host


@dataclass
class StateHandle:
    S: int
    S0: int
    suffix_ids: torch.Tensor
    slot: int
    key: str = None
    state_ids: torch.Tensor = None
    vision: VisionRequest = None


def media_key(media):
    if not media:
        return ""
    digest = hashlib.sha1()
    for name in MEDIA_KEYS:
        value = media.get(name)
        if value is None:
            continue
        tensor = torch.as_tensor(value).contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_media_paths(record):
    from PIL import Image

    def load(item):
        if isinstance(item, (str, os.PathLike)):
            return Image.open(item).convert("RGB")
        return item

    request = dict(record)
    if request.get("images"):
        request["images"] = [load(item) for item in request["images"]]
    if request.get("videos"):
        request["videos"] = [[load(frame) for frame in frames] for frames in request["videos"]]
    return request


@contextlib.contextmanager
def as_tensor_cache_policy(mode):
    if mode not in CACHE_RELOAD_MODES:
        raise ValueError(f"cache_reload={mode!r}; expected one of {CACHE_RELOAD_MODES}")
    if mode == "stock":
        yield
        return
    original = ttnn.as_tensor

    def patched(
        tensor,
        dtype=None,
        *,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=None,
        memory_config=None,
        cache_file_name=None,
        preprocess=None,
        mesh_mapper=None,
    ):
        if mode == "off":
            cache_file_name = None
        elif cache_file_name is not None and device is not None:
            dtype_name = dtype.name if dtype is not None else "None"
            layout_name = layout.name if layout is not None else "None"
            full = pathlib.Path(f"{cache_file_name}_dtype_{dtype_name}_layout_{layout_name}.tensorbin")
            if full.is_file():
                host = ttnn.load_tensor(full)
                return ttnn.to_device(host, device, memory_config=memory_config)
        return original(
            tensor,
            dtype,
            layout=layout,
            device=device,
            memory_config=memory_config,
            cache_file_name=cache_file_name,
            preprocess=preprocess,
            mesh_mapper=mesh_mapper,
        )

    ttnn.as_tensor = patched
    try:
        yield
    finally:
        ttnn.as_tensor = original


@contextlib.contextmanager
def prefill_only_mlp_weights(enabled):
    if not enabled:
        yield
        return
    original = qwen36_mlp.load_mlp_weights

    def load_prefill_only(mesh_device, state_dict, tensor_cache_path=None, args=None, use_gateup_agmm=True):
        tp = getattr(args, "num_devices", 1) if args is not None else 1
        interleaved = tp > 1 and getattr(args, "mlp_1d_decode", False)
        if not (interleaved and use_gateup_agmm and tpc.mlp_gateup_agmm_enabled(tp)):
            return original(mesh_device, state_dict, tensor_cache_path, args=args, use_gateup_agmm=use_gateup_agmm)

        def cache(name, tag=""):
            return str(tensor_cache_path / f"mlp.{name}.weight{tag}.tp") if tensor_cache_path else None

        gate_up = qwen36_mlp._build_gate_up(
            state_dict["gate_proj.weight"], state_dict["up_proj.weight"], mesh_device, tp, cache("gate_up", ".swiglu")
        )
        down = tpc.shard_w(
            state_dict["down_proj.weight"],
            mesh_device,
            dim=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=cache("down_proj"),
            dtype=qwen36_mlp.MLP_DOWN_DTYPE,
        )
        return qwen36_mlp.MLPWeights(w1=None, w2=down, w3=None, w_gate_up=gate_up)

    qwen36_mlp.load_mlp_weights = load_prefill_only
    try:
        yield
    finally:
        qwen36_mlp.load_mlp_weights = original


PARENTS = {"1x4": ("FABRIC_1D", (1, 4)), "2x2": ("FABRIC_2D", (2, 2))}


@contextlib.contextmanager
def tp2_mesh(parent="1x4", trace_region_size=0, l1_small_size=24576):
    fabric_name, shape = PARENTS[parent]
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric_name))
    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*shape),
        l1_small_size=l1_small_size,
        num_command_queues=2,
        trace_region_size=trace_region_size,
    )
    sub = parent_mesh.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(0, 0))
    sub.enable_program_cache()
    logger.info(
        f"tp2_mesh: {fabric_name} parent {list(parent_mesh.shape)} chips {list(parent_mesh.get_device_ids())}, "
        f"submesh {list(sub.shape)} chips {list(sub.get_device_ids())}"
    )
    try:
        yield sub
    finally:
        for child in parent_mesh.get_submeshes():
            ttnn.close_mesh_device(child)
        ttnn.close_mesh_device(parent_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@contextlib.contextmanager
def tp_proj_dtype(dtype):
    if dtype is None or dtype == ttnn.bfloat8_b:
        yield
        return
    original = tpc.shard_w

    def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b, _forced=dtype):
        chosen = _forced if dtype == ttnn.bfloat8_b else dtype
        return original(torch_tensor, mesh, dim, memory_config, cache_path, dtype=chosen)

    tpc.shard_w = shard_w
    try:
        yield
    finally:
        tpc.shard_w = original


def mesh_dram_free_bytes(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(view.total_bytes_free_per_bank) * int(view.num_banks)


class ClefEngine:
    def __init__(
        self,
        mesh,
        args_cls=ClefModelArgs,
        max_state_len=16384,
        chunk_size=1024,
        snapshot_slots=4,
        n_layers=None,
        max_tail_len=4096,
        cache_reload=None,
        kv_reserve_bytes=2 << 30,
        prefill_only_mlp=True,
        vision=None,
        traced=None,
        trace_guard_bytes=None,
        vision_warm_grid=None,
        planner=None,
    ):
        assert chunk_size == MAX_BUCKET, f"chunk_size must be {MAX_BUCKET} at TP=2 (stage 0 finding 2)"
        self.mesh = mesh
        self._rep = ttnn.ReplicateTensorToMesh(mesh)
        self.num_devices = mesh.get_num_devices()
        self.chunk_size = chunk_size
        self.max_state_len = max_state_len
        self.max_tail_len = max_tail_len
        self.max_len = max_state_len + max_tail_len
        self.cache_reload = cache_reload or os.environ.get("CLEF_CACHE_RELOAD", "off")
        self.prefill_only_mlp = prefill_only_mlp
        self.precision = precision_defaults.active()
        self.traced = (os.environ.get("CLEF_TRACED", "0") == "1") if traced is None else bool(traced)
        self.planner = (os.environ.get("CLEF_PLANNER", "1") == "1") if planner is None else bool(planner)
        if self.traced:
            os.environ.setdefault("QWEN_GDN_CONV", "fir")
        self.gdn_conv_impl = os.environ.get("QWEN_GDN_CONV", "kda") if self.traced else "fir"
        self.cost_table = REPLAY_COST_S if self.traced else EAGER_COST_S
        self.trace_guard_bytes = (
            int(os.environ.get("CLEF_TRACE_GUARD_BYTES", str(TRACE_GUARD_BYTES)))
            if trace_guard_bytes is None
            else int(trace_guard_bytes)
        )
        self.vision_warm_grid = vision_warm_grid
        self.vision_warmed_grids = []
        self.traces = {}
        self.trace_bytes = 0
        self.bufs = {}
        self.counters = {"replays": 0, "host_writes": 0, "readbacks": 0, "tower_runs": 0}
        self.timings = {}
        t0 = time.perf_counter()
        args = args_cls(mesh_device=mesh, max_batch_size=1, max_seq_len=self.max_len)
        if n_layers is not None:
            args.n_layers = n_layers
            args.attention_type_list = args.attention_type_list[:n_layers]
        self.args = args
        self.cache_dir = args.weight_cache_path()
        logger.info(
            f"ClefEngine args={type(args).__name__} n_layers={args.n_layers} max_len={self.max_len} "
            f"cache={self.cache_dir} cache_reload={self.cache_reload} prefill_only_mlp={prefill_only_mlp} "
            f"precision={self.precision}"
        )
        state_dict = args.load_state_dict()
        self.timings["host_state_dict_s"] = time.perf_counter() - t0
        state_dict["output.weight"] = torch.zeros(
            self.num_devices * LM_HEAD_STUB_ROWS_PER_DEVICE, args.dim, dtype=torch.bfloat16
        )
        t1 = time.perf_counter()
        from models.demos.blackhole.qwen36.tt.precision import PROJ_DTYPE

        with as_tensor_cache_policy(self.cache_reload), prefill_only_mlp_weights(prefill_only_mlp), tp_proj_dtype(
            PROJ_DTYPE
        ):
            self.model = Qwen36Model(mesh, args, state_dict, tensor_cache_path=self.cache_dir)
        del state_dict
        ttnn.synchronize_device(mesh)
        self.timings["model_build_s"] = time.perf_counter() - t1
        self.timings["load_total_s"] = time.perf_counter() - t0
        self.model._lm_head = self._no_lm_head
        self.device_dtypes = self._device_dtypes()
        logger.info(f"ClefEngine device dtypes and fidelities: {self.device_dtypes}")
        self.dram_free_after_text_weights = mesh_dram_free_bytes(mesh)
        self.vision = None
        self.last_vision = None
        if vision is None:
            vision = os.environ.get("CLEF_VISION", "1") == "1"
        if vision:
            t2 = time.perf_counter()
            vision_args = clef_vision.ClefVisionArgs(
                mesh, args.snapshot, max_batch_size=1, max_seq_len=VISION_MAX_SEQ_LEN
            )
            self.vision = clef_vision.ClefVision(mesh, vision_args)
            self.vision.attach(self.model)
            self.model._alloc_vision_merge_buffers(mesh, MAX_BUCKET)
            ttnn.synchronize_device(mesh)
            self.timings["vision_build_s"] = time.perf_counter() - t2
            self.timings["load_total_s"] = time.perf_counter() - t0
            logger.info(
                f"ClefEngine vision tower attached in {self.timings['vision_build_s']:.1f} s: {self.vision.describe()}"
            )
        if self.vision is not None and self.traced:
            self._warm_vision_tower()
        self.dram_free_after_weights = mesh_dram_free_bytes(mesh)
        self.blocks_per_slot = self.max_len // BLOCK_SIZE
        self.spare_blocks = 0
        if self.traced:
            padded = ((self.blocks_per_slot + SPARE_BLOCKS + 31) // 32) * 32
            self.spare_blocks = padded - self.blocks_per_slot
        self.gdn_layers = [layer.attention for layer in self.model.layers if not layer.is_full_attention]
        self.n_attention_layers = sum(1 for layer in self.model.layers if layer.is_full_attention)
        self.snapshot_slots = self._fit_slots(snapshot_slots, kv_reserve_bytes)
        kv_shape = [
            self.snapshot_slots * self.blocks_per_slot + self.spare_blocks,
            args.n_local_kv_heads,
            BLOCK_SIZE,
            args.head_dim,
        ]
        self.model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
        self.page_tables = [
            torch.arange(s * self.blocks_per_slot, (s + 1) * self.blocks_per_slot, dtype=torch.int32).unsqueeze(0)
            for s in range(self.snapshot_slots)
        ]
        self.snapshots = [
            [(self._zeros_like(dn.rec_state), self._zeros_like(dn.conv_carry)) for dn in self.gdn_layers]
            for _ in range(self.snapshot_slots)
        ]
        self.device_dtypes["gdn_rec_state"] = str(self.gdn_layers[0].rec_state.dtype)
        self.prefix_hidden = [None] * self.snapshot_slots
        self.handles = [None] * self.snapshot_slots
        ttnn.synchronize_device(mesh)
        self.dram_free_after_slots = mesh_dram_free_bytes(mesh)
        self._tokenizer = None
        self._head = None
        self._processor = None
        if self.traced:
            self._setup_traces()
            self.timings["load_total_s"] = time.perf_counter() - t0
        logger.info(
            f"ClefEngine ready: load {self.timings['load_total_s']:.1f} s (state dict "
            f"{self.timings['host_state_dict_s']:.1f} s, build {self.timings['model_build_s']:.1f} s), "
            f"DRAM free after weights {self.dram_free_after_weights / 2**30:.2f} GiB, after slots "
            f"{self.dram_free_after_slots / 2**30:.2f} GiB, slots {self.snapshot_slots}, "
            f"blocks/slot {self.blocks_per_slot}, traced={self.traced} traces={len(self.traces)} "
            f"trace_region_used={self.trace_bytes / 2**20:.1f} MiB, gdn_conv={self.gdn_conv_impl}, "
            f"planner={self.planner}, warm_grids={[g for g, _ in self.vision_warmed_grids]}"
        )

    def _dev(self, t, dtype, layout):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=self._rep)

    def _host(self, t, dtype, layout, mapper=None):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mapper or self._rep)

    def _write(self, t, buf, dtype, layout, mapper=None):
        ttnn.copy_host_to_device_tensor(self._host(t, dtype, layout, mapper), buf)
        self.counters["host_writes"] += 1

    @contextlib.contextmanager
    def _misses_allowed(self):
        if not self.traces:
            yield
            return
        self._misses_depth = getattr(self, "_misses_depth", 0) + 1
        if self._misses_depth == 1:
            self.mesh.set_program_cache_misses_allowed(True)
        try:
            yield
        finally:
            self._misses_depth -= 1
            if self._misses_depth == 0:
                self.mesh.set_program_cache_misses_allowed(False)

    def _warm_vision_tower(self):
        grids = self.vision_warm_grid
        if grids is None:
            text = os.environ.get("CLEF_VISION_WARM_GRID")
            if text is None:
                raise ValueError(WARM_GRID_HELP)
            grids = parse_grids(text)
        elif grids and isinstance(grids[0], int):
            grids = [tuple(grids)]
        if not grids:
            raise ValueError(WARM_GRID_HELP)
        t0 = time.perf_counter()
        warmed = []
        for grid in grids:
            grid_t = torch.tensor([list(grid)], dtype=torch.long)
            n_patches = int(grid_t.prod())
            pixels = torch.randn(
                n_patches, 3 * 2 * 16 * 16, dtype=torch.float32, generator=torch.Generator().manual_seed(0)
            )
            if int(grid[0]) > 1:
                tokens = self.model.get_video_features(pixels, grid_t)
            else:
                tokens = self.model.get_image_features(pixels, grid_t)
            ttnn.synchronize_device(self.mesh)
            ttnn.deallocate(tokens)
            warmed.append((tuple(grid), n_patches))
        self.model._req_image_grid_thw = None
        self.model._req_video_grid_thw = None
        self.timings["vision_warm_s"] = time.perf_counter() - t0
        self.vision_warmed_grids = warmed
        logger.info(f"vision tower warmed on grids {warmed} in {self.timings['vision_warm_s']:.1f} s")

    def _alloc_guard(self, nbytes):
        if nbytes <= 0:
            return None
        rows = max(32, (nbytes // (2 * 1024) // 32) * 32)
        try:
            guard = ttnn.zeros(
                [1, 1, rows, 1024],
                dtype=ttnn.bfloat16,
                layout=TILE,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        except Exception as error:
            logger.warning(f"ttnn.zeros guard failed ({error}); uploading host zeros instead")
            guard = self._dev(torch.zeros(1, 1, rows, 1024, dtype=torch.bfloat16), ttnn.bfloat16, TILE)
        self.trace_guard = {"bytes": rows * 1024 * 2, "address": int(guard.buffer_address())}
        return guard

    def _setup_traces(self):
        mesh = self.mesh
        t0 = time.perf_counter()
        self._rep = ttnn.ReplicateTensorToMesh(mesh)
        self.model._build_request_rope(torch.zeros(1, 1, dtype=torch.long), None)
        self.csi = self._dev(torch.zeros(1, dtype=torch.int32), ttnn.int32, RM)
        first_spare = self.snapshot_slots * self.blocks_per_slot
        spare = torch.arange(first_spare, first_spare + self.spare_blocks, dtype=torch.int32).unsqueeze(0)
        self.full_tables = [torch.cat([pt, spare], dim=1).contiguous() for pt in self.page_tables]
        self.pt_full = self._dev(self.full_tables[0], ttnn.int32, RM)
        for b in BUCKETS:
            cos, sin = self.model._rope_tp_cos_sin_torch(0, b)
            self.bufs[b] = {
                "tok": self._dev(torch.zeros(1, b, dtype=torch.int32), ttnn.uint32, RM),
                "cpt": self._dev(self.full_tables[0][:, : b // BLOCK_SIZE].contiguous(), ttnn.int32, RM),
                "cos": self._dev(cos.contiguous(), ttnn.bfloat16, TILE),
                "sin": self._dev(sin.contiguous(), ttnn.bfloat16, TILE),
                "out": None,
            }
        guard = self._alloc_guard(self.trace_guard_bytes)
        for b in BUCKETS:
            for _ in range(2):
                self.model._reset_gdn_state_for_new_sequence()
                self._forward_body(b)
        self._copy_state(0, to_live=False)
        self._copy_state(0, to_live=True)
        self.model._reset_gdn_state_for_new_sequence()
        ttnn.synchronize_device(mesh)
        self.timings["trace_warm_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        mesh.set_program_cache_misses_allowed(False)
        for b in BUCKETS:
            self.traces[("fwd", b)] = self._capture(lambda b=b: self._forward_body(b))
        for slot in range(self.snapshot_slots):
            self.traces[("restore", slot)] = self._capture(lambda s=slot: self._copy_state(s, to_live=True))
            self.traces[("save", slot)] = self._capture(lambda s=slot: self._copy_state(s, to_live=False))
        self.traces[("zero",)] = self._capture(self.model._reset_gdn_state_for_new_sequence)
        ttnn.synchronize_device(mesh)
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        self.trace_bytes = int(view.total_bytes_allocated_per_bank) * int(view.num_banks)
        self.timings["trace_capture_s"] = time.perf_counter() - t1
        if guard is not None:
            ttnn.deallocate(guard)
            probe = self._dev(torch.zeros(1, 1024, dtype=torch.int32), ttnn.int32, RM)
            self.trace_guard["probe_address_after_release"] = int(probe.buffer_address())
            ttnn.deallocate(probe)
        self.model._reset_gdn_state_for_new_sequence()
        ttnn.synchronize_device(mesh)
        self.dram_free_after_traces = mesh_dram_free_bytes(mesh)
        logger.info(
            f"{len(self.traces)} traces captured in {self.timings['trace_capture_s']:.1f} s after a "
            f"{self.timings['trace_warm_s']:.1f} s warmup; trace region used {self.trace_bytes / 2**20:.1f} MiB; "
            f"guard {getattr(self, 'trace_guard', None)}; DRAM free {self.dram_free_after_traces / 2**30:.2f} GiB"
        )

    def _capture(self, body):
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            body()
        except Exception:
            logger.error(f"trace capture failed:\n{traceback.format_exc()}")
            try:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                ttnn.release_trace(self.mesh, tid)
            except Exception as error:
                logger.error(f"could not end the failed capture: {error}")
            raise
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        return tid

    def _replay(self, key):
        ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
        self.counters["replays"] += 1

    def _forward_body(self, b):
        B = self.bufs[b]
        self.model._chunked_chunk_size = b
        hidden = self.model._forward_prefill_chunk_tp(B["tok"], B["cos"], B["sin"], self.csi, self.pt_full, B["cpt"])
        x = self.model.norm(hidden, mode=Mode.PREFILL)
        ttnn.deallocate(hidden)
        if B["out"] is None:
            B["out"] = x
            return
        ttnn.copy(x, B["out"])
        ttnn.deallocate(x)

    def _copy_state(self, slot, to_live):
        for dn, (rec, carry) in zip(self.gdn_layers, self.snapshots[slot]):
            if to_live:
                ttnn.copy(rec, dn.rec_state)
                ttnn.copy(carry, dn.conv_carry)
            else:
                ttnn.copy(dn.rec_state, rec)
                ttnn.copy(dn.conv_carry, carry)

    def _stage_merge(self, tok_host, chunk_start, full_ids, vision):
        model = self.model
        if model._vis_buf is None:
            return
        if vision is None or vision.rows_host is None:
            ttnn.copy_host_to_device_tensor(model._vis_zero_mask_host, model._vis_mask_buf)
            self.counters["host_writes"] += 1
            return
        offset = model._vis_row_offset_for(full_ids, chunk_start)
        cs = int(model._vis_buf.shape[-2])
        flat = tok_host.reshape(-1)
        pos = torch.nonzero(flat[:cs] == model._vision_placeholder_token_id(), as_tuple=False).reshape(-1)
        n = int(pos.numel())
        if n == 0:
            ttnn.copy_host_to_device_tensor(model._vis_zero_mask_host, model._vis_mask_buf)
            self.counters["host_writes"] += 1
            return
        assert offset + n <= int(
            vision.rows_host.shape[0]
        ), f"vision splice out of range: {offset}+{n} > {vision.rows_host.shape[0]}"
        vis_full = torch.zeros(1, 1, cs, self.args.dim, dtype=torch.bfloat16)
        vis_full[0, 0, pos] = vision.rows_host[offset : offset + n]
        mask = torch.zeros(1, 1, cs, 1, dtype=torch.bfloat16)
        mask[0, 0, pos, 0] = 1.0
        shard = ttnn.ShardTensor2dMesh(self.mesh, dims=(None, -1), mesh_shape=self.args.cluster_shape)
        self._write(vis_full, model._vis_buf, ttnn.bfloat16, TILE, mapper=shard)
        self._write(mask, model._vis_mask_buf, ttnn.bfloat16, TILE)

    def _replay_chunk(self, token_ids, chunk_start, slot, full_ids=None, vision=None):
        length = token_ids.shape[1]
        bucket = self.bucket_for(length)
        assert chunk_start % ALIGN == 0, f"chunk_start {chunk_start} is not {ALIGN}-aligned"
        B = self.bufs[bucket]
        tok = torch.zeros(1, bucket, dtype=torch.int32)
        tok[:, :length] = token_ids.to(torch.int32)
        self._write(tok, B["tok"], ttnn.uint32, RM)
        self._write(torch.tensor([chunk_start], dtype=torch.int32), self.csi, ttnn.int32, RM)
        self._write(self.full_tables[slot], self.pt_full, ttnn.int32, RM)
        blk0 = chunk_start // BLOCK_SIZE
        self._write(
            self.full_tables[slot][:, blk0 : blk0 + bucket // BLOCK_SIZE].contiguous(), B["cpt"], ttnn.int32, RM
        )
        cos, sin = self.model._rope_tp_cos_sin_torch(chunk_start, bucket)
        self._write(cos.contiguous(), B["cos"], ttnn.bfloat16, TILE)
        self._write(sin.contiguous(), B["sin"], ttnn.bfloat16, TILE)
        self._stage_merge(tok, chunk_start, full_ids, vision)
        self._replay(("fwd", bucket))
        out = ttnn.to_torch(B["out"], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        self.counters["readbacks"] += 1
        return out[0].reshape(-1, self.args.dim)[:length].float()

    def _device_dtypes(self):
        out = {}
        gdn = next(layer for layer in self.model.layers if not layer.is_full_attention)
        attn = next((layer for layer in self.model.layers if layer.is_full_attention), None)
        mlp = gdn.feed_forward.weights
        out["mlp_gate_up_packed"] = str(mlp.w_gate_up.dtype) if mlp.w_gate_up is not None else None
        out["mlp_w1"] = str(mlp.w1.dtype) if mlp.w1 is not None else None
        out["mlp_down"] = str(mlp.w2.dtype)
        out["mlp_fidelity"] = str(gdn.feed_forward.compute_kernel_config.math_fidelity)
        tw = gdn.attention.tw
        out["gdn_qkvzab"] = str(tw["qkvz"].dtype)
        out["gdn_out_colpar"] = str(tw["out_colpar"].dtype)
        out["gdn_fidelity"] = str(gdn.attention.cfg.math_fidelity)
        out["gdn_gate_fp32"] = bool(gdn.attention._gate_fp32)
        out["gdn_dt_bias"] = str(tw["dt_bias"].dtype)
        out["gdn_rec_state"] = None
        if attn is not None:
            atw = attn.attention.tw
            key = "wqkv_fused" if "wqkv_fused" in atw else "wqkv"
            out["attn_qkv"] = str(atw[key].dtype)
            out["attn_wo"] = str(atw["wo"].dtype)
            out["attn_fidelity"] = str(attn.attention.compute_cfg.math_fidelity)
        out["embedding"] = str(self.model.embd.weights.dtype)
        return out

    def _no_lm_head(self, x):
        raise RuntimeError("ClefEngine never runs the LM head; output.weight is a stub")

    def _fit_slots(self, requested, reserve):
        kv_per_slot = (
            2
            * self.n_attention_layers
            * self.blocks_per_slot
            * self.args.n_local_kv_heads
            * BLOCK_SIZE
            * self.args.head_dim
            * 2
        )
        rec_bytes = 4 if os.environ.get("QWEN35_GDN_STATE_BF16") != "1" else 2
        gdn_per_slot = sum(
            dn.Nv * dn.Dk * dn.Dv * rec_bytes + ttnn.TILE_SIZE * dn.qkv_dim_tp * 2 for dn in self.gdn_layers
        )
        per_slot = kv_per_slot + gdn_per_slot
        free = self.dram_free_after_weights
        slots = max(1, min(requested, (free - reserve) // per_slot))
        self.slot_bytes = {
            "kv_per_slot": kv_per_slot,
            "gdn_per_slot": gdn_per_slot,
            "fit": int((free - reserve) // per_slot),
        }
        logger.info(
            f"KV per slot {kv_per_slot / 2**30:.3f} GiB + GDN snapshot {gdn_per_slot / 2**20:.1f} MiB, DRAM free "
            f"{free / 2**30:.2f} GiB, reserve {reserve / 2**30:.2f} GiB: {slots} slot(s) of {requested} requested "
            f"(max fit {self.slot_bytes['fit']})"
        )
        return slots

    def _zeros_like(self, t):
        return ttnn.from_torch(
            torch.zeros(list(t.shape), dtype=torch.float32 if t.dtype == ttnn.float32 else torch.bfloat16),
            dtype=t.dtype,
            layout=TILE,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = clef_encode.load_tokenizer(self.args.snapshot)
        return self._tokenizer

    @property
    def head(self):
        if self._head is None:
            self._head = clef_head.load_head(self.args.snapshot)
        return self._head

    @property
    def processor(self):
        if self._processor is None:
            self._processor = clef_encode.load_processor(self.args.snapshot)
        return self._processor

    def _vision_request(self, media):
        if not media:
            return None
        pixels = media.get("pixel_values")
        video_pixels = media.get("pixel_values_videos")
        if pixels is None and video_pixels is None:
            return None
        if self.vision is None:
            raise ValueError("this engine was built without the vision tower (CLEF_VISION=0)")
        if pixels is not None and video_pixels is not None:
            raise ValueError("one media kind per request: the backbone stages one grid (images or videos)")
        video = video_pixels is not None
        grid = torch.as_tensor(media["video_grid_thw" if video else "image_grid_thw"]).reshape(-1, 3)
        if self.traced:
            self.check_warm_grids(grid)
        t0 = time.perf_counter()
        with self._misses_allowed():
            if video:
                tokens = self.model.get_video_features(video_pixels, grid)
            else:
                tokens = self.model.get_image_features(pixels, grid)
            ttnn.synchronize_device(self.mesh)
            n_rows = int(tokens.shape[0])
            rows_host = None
            if self.traced:
                rows_host = ttnn.to_torch(tokens, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=1)).to(
                    torch.bfloat16
                )
                ttnn.deallocate(tokens)
                tokens = None
        self.counters["tower_runs"] += 1
        request = VisionRequest(
            tokens=tokens,
            grid=grid,
            video=video,
            n_rows=n_rows,
            n_patches=int(grid.prod(dim=1).sum()),
            seconds=time.perf_counter() - t0,
            rows_host=rows_host,
        )
        self.last_vision = dict(
            video=video,
            grid=grid.tolist(),
            n_patches=request.n_patches,
            n_rows=request.n_rows,
            seconds=round(request.seconds, 4),
            tower=dict(self.vision.last_run),
        )
        return request

    def check_warm_grids(self, grid):
        rows = torch.as_tensor(grid).reshape(-1, 3).tolist()
        if len(rows) > 1:
            raise ValueError(single_grid_message(len(rows)))
        warmed = {tuple(g) for g, _ in self.vision_warmed_grids}
        for row in rows:
            if tuple(row) not in warmed:
                raise ValueError(
                    f"image grid {tuple(row)} is not in this traced server's warm list {sorted(warmed)}: "
                    "resize the image to a warmed grid, add the grid to CLEF_VISION_WARM_GRID, "
                    "or serve with CLEF_TRACED=0"
                )

    def _stage_rope(self, token_ids, vision):
        if vision is None:
            self.model._build_request_rope(token_ids, None)
            return
        self.model._req_image_grid_thw = None if vision.video else vision.grid
        self.model._req_video_grid_thw = vision.grid if vision.video else None
        self.model._build_request_rope(token_ids, vision.marker)

    def _release_slot(self, slot):
        handle = self.handles[slot]
        self.handles[slot] = None
        if handle is not None and handle.vision is not None:
            handle.vision.release()

    @staticmethod
    def bucket_for(length):
        bucket = Qwen36Model._mask_bucket_for(length)
        assert bucket in BUCKETS, f"segment of {length} tokens needs bucket {bucket} > {MAX_BUCKET}"
        return bucket

    def _begin_sequence(self, token_ids, vision=None):
        if self.traced:
            self._replay(("zero",))
        else:
            self.model._reset_gdn_state_for_new_sequence()
        self._stage_rope(token_ids, vision)

    def _run_chunk(self, token_ids, chunk_start, slot, full_ids=None, vision=None):
        if self.traced:
            return self._replay_chunk(token_ids, chunk_start, slot, full_ids, vision)
        length = token_ids.shape[1]
        bucket = self.bucket_for(length)
        assert chunk_start % ALIGN == 0, f"chunk_start {chunk_start} is not {ALIGN}-aligned"
        buf = torch.zeros(1, bucket, dtype=torch.int32)
        buf[:, :length] = token_ids.to(torch.int32)
        if self.model._vis_buf is not None:
            if vision is None:
                self.model._set_vision_merge(buf, None)
            else:
                offset = self.model._vis_row_offset_for(full_ids, chunk_start)
                self.model._set_vision_merge(buf, vision.tokens, offset)
        return self.model._forward_prefill_chunk_masked_tp(buf, length, chunk_start, self.page_tables[slot], bucket)

    def _read_hidden(self, hidden, length):
        x = self.model.norm(hidden, mode=Mode.PREFILL)
        ttnn.deallocate(hidden)
        out = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        ttnn.deallocate(x)
        return out[0].reshape(-1, self.args.dim)[:length].float()

    def _segments(self, token_ids, start, exact=False):
        T = token_ids.shape[1]
        if exact and self.traced:
            full = (T // self.chunk_size) * self.chunk_size
            cs = 0
            for b in [self.chunk_size] * (full // self.chunk_size) + aligned_pieces(T - full):
                yield token_ids[:, cs : cs + b], start + cs
                cs += b
            return
        for cs in range(0, T, self.chunk_size):
            ce = min(cs + self.chunk_size, T)
            yield token_ids[:, cs:ce], start + cs

    def _run_rows(self, token_ids, start, slot, full_ids=None, vision=None, exact=False):
        rows = []
        for piece, chunk_start in self._segments(token_ids, start, exact):
            if self.traced:
                rows.append(self._replay_chunk(piece, chunk_start, slot, full_ids, vision))
                continue
            hidden = self._run_chunk(piece, chunk_start, slot, full_ids, vision)
            rows.append(self._read_hidden(hidden, piece.shape[1]))
        if not rows:
            return torch.empty(0, self.args.dim, dtype=torch.float32)
        return torch.cat(rows, dim=0)

    def _save_state(self, slot):
        if self.traced:
            self._replay(("save", slot))
            return
        self._copy_state(slot, to_live=False)

    def _restore_state(self, slot):
        if self.traced:
            self._replay(("restore", slot))
            return
        self._copy_state(slot, to_live=True)

    def prefill_hidden(self, token_ids, slot=0, media=None):
        T = token_ids.shape[1]
        assert token_ids.shape[0] == 1
        assert T <= self.max_len, f"T={T} exceeds max_len={self.max_len}"
        assert 0 <= slot < self.snapshot_slots, f"slot {slot} out of range"
        self._release_slot(slot)
        self.prefix_hidden[slot] = None
        vision = self._vision_request(media)
        try:
            self._begin_sequence(token_ids, vision)
            return self._run_rows(token_ids, 0, slot, token_ids, vision)
        finally:
            if vision is not None:
                vision.release()

    def _chunk_cost(self, length, exact):
        if length <= 0:
            return 0.0
        if exact:
            pieces = [self.chunk_size] * (length // self.chunk_size) + aligned_pieces(length % self.chunk_size)
        else:
            full, rest = divmod(length, self.chunk_size)
            pieces = [self.chunk_size] * full + ([self.bucket_for(rest)] if rest else [])
        return sum(self.cost_table[p] for p in pieces)

    def plan_prefix(self, S, tail_len=None):
        S0_max = (S // ALIGN) * ALIGN
        if not self.planner or tail_len is None:
            return S0_max
        exact = self.traced
        base_tail = self._chunk_cost(S - S0_max + tail_len, exact=False)
        best, best_cost = S0_max, self._chunk_cost(S0_max, exact=exact) + base_tail
        S0 = S0_max - ALIGN
        while S0 >= 0 and S0 > S0_max - self.chunk_size:
            tail_cost = self._chunk_cost(S - S0 + tail_len, exact=False)
            if tail_cost > base_tail:
                break
            cost = self._chunk_cost(S0, exact=exact) + tail_cost
            if cost < best_cost:
                best, best_cost = S0, cost
            S0 -= ALIGN
        return best

    def prefill_state(self, state_ids, slot=0, key=None, media=None, tail_len=None):
        S = state_ids.shape[1]
        S0 = self.plan_prefix(S, tail_len)
        assert state_ids.shape[0] == 1
        assert S <= self.max_state_len, f"S={S} exceeds max_state_len={self.max_state_len}"
        assert 0 <= slot < self.snapshot_slots, f"slot {slot} out of range"
        self._release_slot(slot)
        vision = self._vision_request(media)
        self._begin_sequence(state_ids, vision)
        self.prefix_hidden[slot] = self._run_rows(state_ids[:, :S0], 0, slot, state_ids, vision, exact=True)
        self._save_state(slot)
        handle = StateHandle(S, S0, state_ids[:, S0:S].clone(), slot, key, state_ids.clone(), vision)
        self.handles[slot] = handle
        return handle

    def schema_hidden(self, handle, schema_ids):
        tail = torch.cat([handle.suffix_ids, schema_ids], dim=1)
        L = tail.shape[1]
        assert handle.S0 + L <= self.max_len, f"S0+L={handle.S0 + L} exceeds max_len={self.max_len}"
        assert self.handles[handle.slot] is handle, "slot no longer holds this state"
        self._restore_state(handle.slot)
        full_ids = torch.cat([handle.state_ids[:, : handle.S0], tail], dim=1)
        self._stage_rope(full_ids, handle.vision)
        return self._run_rows(tail, handle.S0, handle.slot, full_ids, handle.vision)

    def cached_hidden(self, state_ids, tail_ids, slot=0, media=None):
        key = clef_encode.cache_key(state_ids[0].tolist()) + media_key(media)
        handle = self.handles[slot]
        hit = handle is not None and handle.key == key
        if not hit:
            handle = self.prefill_state(state_ids, slot, key, media=media, tail_len=int(tail_ids.shape[1]))
        tail_hidden = self.schema_hidden(handle, tail_ids)
        return torch.cat([self.prefix_hidden[slot], tail_hidden], dim=0), hit

    def probs_for_request(self, record, mode="full", slot=0, max_length=None):
        if mode not in ("full", "cached"):
            raise ValueError(f"mode={mode!r}; expected 'full' or 'cached'")
        t0 = time.perf_counter()
        request = load_media_paths(record)
        has_media = bool(request.get("images") or request.get("videos"))
        encoded = clef_encode.encode(
            self.tokenizer,
            request,
            processor=self.processor if has_media else None,
            max_length=max_length or min(16384, self.max_len),
        )
        media = encoded.media if has_media else None
        ids = torch.tensor([list(encoded.input_ids)], dtype=torch.long)
        t_enc = time.perf_counter() - t0
        hit = None
        self.last_vision = None
        if mode == "full":
            hidden = self.prefill_hidden(ids, slot, media=media)
        else:
            state_part, tail_part, _ = clef_encode.split_for_cache(encoded, self.tokenizer, request)
            hidden, hit = self.cached_hidden(
                torch.tensor([state_part], dtype=torch.long),
                torch.tensor([tail_part], dtype=torch.long),
                slot,
                media=media,
            )
        t_dev = time.perf_counter() - t0 - t_enc
        probs = clef_head.probs_for_record(self.head, hidden, ids[0], encoded, self.args.load_lm_head_rows)
        answers = {
            qid: clef_encode.release_module().systemone_answer(request["questions"][qid], dist)
            for qid, dist in probs.items()
        }
        total = time.perf_counter() - t0
        timing = {
            "encode_s": round(t_enc, 3),
            "device_s": round(t_dev, 3),
            "head_s": round(total - t_enc - t_dev, 3),
        }
        if self.last_vision is not None:
            timing["vision_s"] = self.last_vision["seconds"]
        result = {
            "model": request.get("model", "clef"),
            "answers": answers,
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
            "probs": probs,
            "input_tokens": len(encoded.input_ids),
            "seconds": round(total, 3),
            "timing": timing,
            "mode": mode,
            "cache_hit": hit,
        }
        if self.last_vision is not None:
            result["vision"] = dict(self.last_vision)
        return result
