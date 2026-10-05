import contextlib
import json
import os
import shutil
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.cloudflare_clef.tt import encode as clef_encode
from models.autoports.cloudflare_clef.tt.loader import LM_HEAD_KEY, ClefModelArgs, read_tensors
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
RECORDS = "/home/hous/dev/clef/reports/reference/records_text.jsonl"
REPORT_DIR = Path("/home/hous/dev/clef/reports")
PARENTS = {"1x4": ("FABRIC_1D", (1, 4)), "2x2": ("FABRIC_2D", (2, 2))}
BLOCK_SIZE = 64
NUM_BLOCKS = 64
MAX_SEQ_LEN = NUM_BLOCKS * BLOCK_SIZE
PCC_BAR = 0.97
CASES = [
    pytest.param(128, "masked", id="T128-masked"),
    pytest.param(256, "masked", id="T256-masked"),
    pytest.param(512, "masked", id="T512-masked"),
    pytest.param(1024, "masked", id="T1024-masked"),
    pytest.param(2048, "chunked1024", id="T2048-chunked1024"),
    pytest.param(2048, "masked_conv_dram", id="T2048-masked-conv-dram"),
    pytest.param(2048, "masked", id="T2048-masked"),
]
RESULTS = {"cases": {}}

os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
os.environ.setdefault("HF_MODEL", SNAPSHOT)


def n_layers_under_test():
    return int(os.environ.get("CLEF_N_LAYERS", "4"))


def report_path():
    return REPORT_DIR / f"stage0_tp2_sanity_l{n_layers_under_test()}_{os.environ.get('CLEF_PARENT', '1x4')}.json"


def write_report():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path().write_text(json.dumps(RESULTS, indent=2, default=str))


@contextlib.contextmanager
def gdn_conv_in_dram():
    orig_fir = gdn_tp._causal_conv1d_fir
    orig_proj = gdn_tp.TPGatedDeltaNet._project_qkvzab

    def fir_dram(*args, **kwargs):
        kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        return orig_fir(*args, **kwargs)

    def proj_dram(self, x, S, out_mc=None):
        return orig_proj(self, x, S, out_mc=None)

    gdn_tp._causal_conv1d_fir = fir_dram
    gdn_tp.TPGatedDeltaNet._project_qkvzab = proj_dram
    try:
        yield
    finally:
        gdn_tp._causal_conv1d_fir = orig_fir
        gdn_tp.TPGatedDeltaNet._project_qkvzab = orig_proj


@pytest.fixture(scope="module")
def submesh():
    name = os.environ.get("CLEF_PARENT", "1x4")
    fabric_name, shape = PARENTS[name]
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric_name))
    parent = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*shape),
        l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
        num_command_queues=2,
        trace_region_size=int(os.environ.get("CLEF_TRACE_REGION", "0")),
    )
    sub = parent.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(0, 0))
    if hasattr(sub, "enable_program_cache"):
        sub.enable_program_cache()
    RESULTS["mesh"] = dict(
        parent=name,
        fabric=fabric_name,
        parent_shape=list(parent.shape),
        parent_device_ids=list(parent.get_device_ids()),
        submesh_shape=list(sub.shape),
        submesh_device_ids=list(sub.get_device_ids()),
        cluster_type=str(ttnn.cluster.get_cluster_type()),
    )
    logger.info(f"mesh: {RESULTS['mesh']}")
    yield sub
    for child in parent.get_submeshes():
        ttnn.close_mesh_device(child)
    ttnn.close_mesh_device(parent)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def tt_model(submesh):
    n = n_layers_under_test()
    t0 = time.perf_counter()
    args = ClefModelArgs(mesh_device=submesh, max_batch_size=1, max_seq_len=MAX_SEQ_LEN)
    args.n_layers = n
    args.attention_type_list = args.attention_type_list[:n]
    raw = args.raw_language_state_dict()
    raw.update(read_tensors(args.snapshot, args.weight_map, [LM_HEAD_KEY]))
    state_dict = remap_qwen36_state_dict(raw)
    t_host = time.perf_counter() - t0
    if os.environ.get("CLEF_WEIGHT_CACHE", "0") == "1":
        cache_dir = args.weight_cache_path()
    else:
        cache_dir = Path(f"/home/hous/dev/clef/tt_cache/_scratch_tp2_sanity_{os.getpid()}")
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True)
    model = Qwen36Model(submesh, args, state_dict, tensor_cache_path=cache_dir)
    del raw, state_dict
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)
    kv_shape = (NUM_BLOCKS, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    RESULTS["model"] = dict(
        n_layers=n,
        layer_types=list(args.attention_type_list),
        host_state_dict_seconds=round(t_host, 1),
        build_seconds=round(time.perf_counter() - t0, 1),
        ccl_topology=str(args.ccl_topology()),
        ccl_num_links=model.tt_ccl.get_num_links(),
        prefill_tuning=dict(args.prefill_tuning),
        n_local_heads=args.n_local_heads,
        n_local_kv_heads=args.n_local_kv_heads,
        kv_replication=args.kv_replication,
        gdn_qkv_dim_tp=args.gdn_qkv_dim_tp,
        weight_cache=str(cache_dir),
        weight_cache_is_fresh_scratch=os.environ.get("CLEF_WEIGHT_CACHE", "0") != "1",
        lm_head_vocab_sharded=model._lmhead_vocab_sharded,
    )
    logger.info(f"model: {RESULTS['model']}")
    write_report()
    yield model, page_table
    if os.environ.get("CLEF_WEIGHT_CACHE", "0") != "1":
        shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def requests():
    tokenizer = clef_encode.load_tokenizer(SNAPSHOT)
    records = [json.loads(line) for line in open(RECORDS)]
    short = list(clef_encode.encode(tokenizer, records[0]).input_ids)
    long_record = dict(records[0])
    long_record["state"] = (" ".join(str(r["state"]) for r in records) + " ") * 8
    out = {"tokenizer": tokenizer}
    kinds = {}
    for T in sorted({T for T, _ in (case.values for case in CASES)}):
        if T <= len(short):
            out[T] = short[:T]
            kinds[T] = f"first {T} tokens of the {len(short)}-token {records[0]['id']} request"
        else:
            out[T] = list(clef_encode.encode(tokenizer, long_record, max_length=T).input_ids)
            kinds[T] = f"full {records[0]['id']} schema with a long state, encode_record max_length={T}"
        assert len(out[T]) == T, (T, len(out[T]))
    RESULTS["requests"] = dict(record_id=records[0]["id"], short_total_tokens=len(short), per_bucket=kinds)
    return out


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    n = n_layers_under_test()
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(SNAPSHOT)
    config.text_config.num_hidden_layers = n
    config.text_config.layer_types = config.text_config.layer_types[:n]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(SNAPSHOT, config=config, dtype=torch.bfloat16).eval()
    RESULTS["hf"] = dict(n_layers=n, load_seconds=round(time.perf_counter() - t0, 1), dtype="bfloat16", device="cpu")
    return model


def run_tt(model, page_table, ids, T, mode):
    if mode == "chunked1024":
        previous = model._chunked_chunk_size
        model._chunked_chunk_size = 1024
        try:
            return model.prefill_traced_chunked(ids, page_table, actual_len=T)
        finally:
            model._chunked_chunk_size = previous
    if mode == "masked_conv_dram":
        with gdn_conv_in_dram():
            return model.prefill_masked_bucket(ids, page_table, actual_len=T)
    return model.prefill_masked_bucket(ids, page_table, actual_len=T)


@pytest.mark.timeout(3000)
@pytest.mark.parametrize("T, mode", CASES)
def test_tp2_prefill_matches_hf(submesh, tt_model, hf_model, requests, T, mode):
    model, page_table = tt_model
    ids = torch.tensor([requests[T]], dtype=torch.long)
    assert ids.shape == (1, T)
    vocab = model.args.vocab_size
    composer = ttnn.ConcatMeshToTensor(submesh, dim=0)
    key = f"T{T}-{mode}"

    timings = []
    try:
        for _ in range(2):
            t0 = time.perf_counter()
            logits_dev = run_tt(model, page_table, ids, T, mode)
            replicas = ttnn.to_torch(logits_dev, mesh_composer=composer).reshape(-1, vocab).float()
            timings.append(round(time.perf_counter() - t0, 3))
    except RuntimeError as exc:
        message = str(exc)
        first_line = next(
            (ln for ln in message.splitlines() if "Out of Memory" in ln or "TT_FATAL" in ln), message[:300]
        )
        RESULTS["cases"][key] = dict(T=T, mode=mode, passed=False, error=first_line[:600])
        write_report()
        logger.error(f"TP2 sanity {key}: {first_line}")
        if mode == "masked" and T == 2048 and "Out of Memory" in message:
            pytest.xfail(f"known L1 OOM in the TP GDN prefill conv at bucket 2048 on 2 devices: {first_line[:200]}")
        raise
    tt = replicas[0]
    replica_gap = float((replicas[0] - replicas[-1]).abs().max())

    t0 = time.perf_counter()
    with torch.no_grad():
        ref = hf_model(input_ids=ids, use_cache=False).logits[0, -1].float()
    t_hf = round(time.perf_counter() - t0, 1)

    assert torch.isfinite(tt).all(), "non-finite TT logits"
    assert ref.shape == tt.shape, (ref.shape, tt.shape)
    _, pcc = comp_pcc(ref, tt, PCC_BAR)
    pcc = float(pcc)
    tt_arg, ref_arg = int(tt.argmax()), int(ref.argmax())
    tokenizer = requests["tokenizer"]
    row = dict(
        T=T,
        mode=mode,
        bucket=1024 if mode == "chunked1024" else Qwen36Model._mask_bucket_for(T),
        pcc=round(pcc, 6),
        tt_argmax=tt_arg,
        hf_argmax=ref_arg,
        tt_argmax_text=tokenizer.decode([tt_arg]),
        hf_argmax_text=tokenizer.decode([ref_arg]),
        tt_top5=tt.topk(5).indices.tolist(),
        hf_top5=ref.topk(5).indices.tolist(),
        tt_seconds_first_call=timings[0],
        tt_seconds_second_call=timings[1],
        hf_cpu_seconds=t_hf,
        replica_max_abs_gap=replica_gap,
        passed=bool(pcc >= PCC_BAR and tt_arg == ref_arg),
    )
    RESULTS["cases"][key] = row
    write_report()
    logger.info(f"TP2 sanity {key}: {row}")
    assert pcc >= PCC_BAR, f"{key}: logits PCC {pcc:.6f} < {PCC_BAR}"
    assert tt_arg == ref_arg, f"{key}: argmax TT {tt_arg} != HF {ref_arg}"
