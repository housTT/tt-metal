# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch

import ttnn
from models.autoports.convaiinnovations_laya import common
from models.autoports.convaiinnovations_laya.reference import corpus
from models.autoports.convaiinnovations_laya.tt import engine as eng
from models.autoports.convaiinnovations_laya.tt import model_config as mc

SIBLING_DIR = "/home/hous/dev/laya/state/laya_models/laya-typed-decisions"
SIBLING_REVISION = "e929ae5cf69bc34259cd2f95c9e91145b818b1f0"
SIBLING_CORPUS = "/home/hous/dev/laya/reference/parity_corpus_td.npz"
ENGLISH_INDEX = "/home/hous/dev/laya/reference/parity_corpus_index.json"
BY_SEQ = {1024: (1, 2, 4, 5, 8, 10, 16)}


class _Device:
    def __init__(self, x=11, y=10):
        self._grid = type("G", (), {"x": x, "y": y})()

    def compute_with_storage_grid_size(self):
        return self._grid

    def arch(self):
        return ttnn.device.Arch.BLACKHOLE


P150 = _Device(11, 10)


class _Config:
    hidden_size = 1024
    intermediate_size = 2624
    num_hidden_layers = 28
    num_attention_heads = 16


def test_sibling_bucket_constants():
    assert mc.ROW_BUCKETS_AT_1024 == (1, 2, 4, 5, 8, 10, 16)
    assert mc.SEQ_BUCKETS_SIBLING == (128, 256, 512, 1024)
    assert mc.ROW_BUCKETS_BY_SEQ_SIBLING == {1024: mc.ROW_BUCKETS_AT_1024}
    sib = mc.sibling_deployment_buckets()
    assert (
        len(sib) == 37
        and sib[:30] == mc.deployment_buckets()
        and sib[30:] == [(b, 1024) for b in mc.ROW_BUCKETS_AT_1024]
    )
    assert mc.parse_buckets("sibling") == sib and mc.parse_buckets("sibling1024") == sib[30:]
    assert mc.parse_buckets("all") == mc.deployment_buckets() and len(mc.deployment_buckets()) == 30


def test_english_defaults_are_unchanged():
    assert mc.ROW_BUCKETS == (1, 2, 4, 5, 8, 10, 16, 32, 50, 64) and mc.SEQ_BUCKETS == (128, 256, 512)
    assert mc.DEFAULT_PORT.sdpa_full_grid_buckets == ((5, 1024), (10, 1024))
    for b, s in mc.deployment_buckets():
        g = mc.sdpa_program_config(P150, s, b * s).compute_with_storage_grid_size
        assert (g.x, g.y) == (8, 8), (b, s)
    assert mc.rows_for_seq(512) == mc.ROW_BUCKETS and mc.rows_for_seq(1024, row_buckets_by_seq=None) == mc.ROW_BUCKETS
    assert mc.select_bucket(5, 205) == (5, 256) and mc.select_bucket(50, 512) == (50, 512)
    assert mc.deployment_buckets(row_buckets_by_seq=None) == [(b, s) for s in mc.SEQ_BUCKETS for b in mc.ROW_BUCKETS]
    assert eng.parse_warmup_shapes(None, mc.ROW_BUCKETS, mc.SEQ_BUCKETS) == mc.deployment_buckets()


def test_per_seq_selection():
    kw = dict(seq_buckets=mc.SEQ_BUCKETS_SIBLING, row_buckets_by_seq=BY_SEQ)
    assert mc.rows_for_seq(1024, mc.ROW_BUCKETS, BY_SEQ) == (1, 2, 4, 5, 8, 10, 16)
    assert mc.rows_for_seq(512, mc.ROW_BUCKETS, BY_SEQ) == mc.ROW_BUCKETS
    assert mc.select_bucket(5, 597, **kw) == (5, 1024)
    assert mc.select_bucket(16, 1024, **kw) == (16, 1024)
    assert mc.select_bucket(3, 513, **kw) == (4, 1024)
    assert mc.select_bucket(50, 300, **kw) == (50, 512)
    assert mc.select_bucket(64, 128, **kw) == (64, 128)
    with pytest.raises(ValueError):
        mc.select_bucket(17, 700, **kw)
    with pytest.raises(ValueError):
        mc.select_bucket(1, 1025, **kw)


def test_env_parsing_of_per_seq_row_lists(monkeypatch):
    monkeypatch.delenv("LAYA_ROW_BUCKETS_1024", raising=False)
    assert eng.row_buckets_by_seq_from_env((128, 256, 512, 1024)) == {}
    monkeypatch.setenv("LAYA_ROW_BUCKETS_1024", "16,1,2,4,5,8,10")
    assert eng.row_buckets_by_seq_from_env((128, 256, 512, 1024)) == {1024: (1, 2, 4, 5, 8, 10, 16)}
    warm = eng.parse_warmup_shapes("all", mc.ROW_BUCKETS, mc.SEQ_BUCKETS_SIBLING, {1024: (1, 2, 4, 5, 8, 10, 16)})
    assert warm == mc.sibling_deployment_buckets()
    assert eng.parse_warmup_shapes("none", mc.ROW_BUCKETS, mc.SEQ_BUCKETS_SIBLING, BY_SEQ) == []
    assert eng.parse_warmup_shapes("5x1024,1x256", mc.ROW_BUCKETS, mc.SEQ_BUCKETS_SIBLING, BY_SEQ) == [
        (5, 1024),
        (1, 256),
    ]


@pytest.mark.parametrize("batch", mc.ROW_BUCKETS_AT_1024)
def test_every_1024_bucket_has_a_plan(batch):
    plan = mc.bucket_plan(P150, _Config, batch, 1024)
    rows = batch * 1024
    assert plan.rows == rows and plan.mlp_width == 2816
    assert plan.qkv_minimal and plan.minimal_config is not None
    assert plan.wo_minimal == (rows >= 4096)
    assert (plan.attention_memory == ttnn.L1_MEMORY_CONFIG) == (rows <= 4096)
    cfg = plan.sdpa_program_config
    assert cfg is not None and 1024 % cfg.q_chunk_size == 0 and 1024 % cfg.k_chunk_size == 0
    grid = (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y)
    assert grid == ((11, 10) if batch in (5, 10) else (8, 8))
    assert cfg.q_chunk_size == (128 if rows == 4096 else 256)
    assert (plan.mlp_shard is not None) == (batch in (1, 2))
    if plan.mlp_shard is None:
        up = mc.mlp_up_projection_program_config(P150, batch, 1024, 1024, 2816, True)
        assert up is not None and up.compute_with_storage_grid_size.x == 11
        assert up.compute_with_storage_grid_size.y == (10 if (rows // 32) % 10 == 0 else 8)
    assert mc.rotary_shard_config((batch, 16, 1024, 64)) is None
    assert mc.describe_plan(plan)["rows"] == rows


def test_engine_chunks_calls_above_the_largest_row_bucket_of_the_seq():
    e = eng.LayaEngine.__new__(eng.LayaEngine)
    e.seq_buckets = mc.SEQ_BUCKETS_SIBLING
    e.model = type("M", (), {"max_rows_for_seq": staticmethod(lambda seq: 16 if seq == 1024 else 64)})()
    calls = []

    def fake_once(ids, att, qt):
        n, L = ids.shape
        calls.append(n)
        return {
            "logits": ids.float()[:, :L],
            "cls": torch.full((n, 1024), float(n)),
            "bucket": (n, 1024 if L > 512 else 512),
            "device_ms": 10.0,
        }

    e._run_once = fake_once
    ids = torch.arange(40 * 600).reshape(40, 600)
    out = e._run_device(ids, torch.ones(40, 600, dtype=torch.long), torch.zeros(40, dtype=torch.long))
    assert calls == [16, 16, 8] and out["buckets"] == [(16, 1024), (16, 1024), (8, 1024)]
    assert out["bucket"] == (16, 1024) and out["device_ms"] == 30.0
    assert (
        torch.equal(out["logits"], ids.float()) and out["cls"].shape == (40, 1024) and float(out["cls"][39, 0]) == 8.0
    )
    calls.clear()
    out = e._run_device(
        torch.zeros(40, 300, dtype=torch.long), torch.ones(40, 300, dtype=torch.long), torch.zeros(40, dtype=torch.long)
    )
    assert calls == [40] and out["buckets"] == [(40, 512)]


def test_checkpoint_pins_from_snapshot_paths():
    if not os.path.isdir(SIBLING_DIR):
        pytest.skip("sibling checkpoint not present")
    pins = common.checkpoint_pins(SIBLING_DIR)
    assert pins["hf_model"] == "convaiinnovations/laya-typed-decisions" and pins["revision"] == SIBLING_REVISION
    english = common.checkpoint_pins("/home/hous/dev/laya/state/laya_models/laya")
    assert english["hf_model"] == common.HF_MODEL and english["revision"] == common.LAYA_REVISION
    assert common.checkpoint_pins("/tmp") == {
        "hf_model": common.HF_MODEL,
        "revision": common.LAYA_REVISION,
        "model_dir": "/tmp",
    }


def test_corpus_file_names():
    assert corpus.corpus_names("") == {
        "npz": "parity_corpus.npz",
        "index": "parity_corpus_index.json",
        "typed_dir": "typed_decisions_cpu",
    }
    assert corpus.corpus_names("td") == {
        "npz": "parity_corpus_td.npz",
        "index": "parity_corpus_td_index.json",
        "typed_dir": "typed_decisions_cpu_td",
    }


def test_sibling_corpus_matches_the_english_gate_subset():
    index_path = SIBLING_CORPUS.replace(".npz", "_index.json")
    if not (os.path.exists(SIBLING_CORPUS) and os.path.exists(index_path) and os.path.exists(ENGLISH_INDEX)):
        pytest.skip("sibling corpus not built")
    import numpy as np

    td = json.load(open(index_path))
    en = json.load(open(ENGLISH_INDEX))
    assert td["hf_model"] == "convaiinnovations/laya-typed-decisions" and td["revision"] == SIBLING_REVISION
    assert (td["max_len"], td["head_max_len"]) == (1024, 256) and (en["max_len"], en["head_max_len"]) == (512, 192)
    assert td["gate_subset"]["case_ids"] == en["gate_subset"]["case_ids"] and td["gate_subset"]["seed"] == 13
    assert td["n_items"] == en["n_items"] == 488 and td["n_typed"] == 200
    assert [it["qid"] for it in td["items"]] == [it["qid"] for it in en["items"]]
    assert [it["ids"] for it in td["items"][:0]] == []
    z = np.load(SIBLING_CORPUS)
    assert z["input_ids"].shape == (488, 1024) and z["logits_fp32"].shape[0] == 488
    assert int(z["seq_len"].max()) <= 1024 and (z["marker_mask"].sum(1) == z["k"]).all()
    assert np.isfinite(z["logits_fp32"][z["marker_mask"]]).all()
    assert td["temperature"] != en["temperature"] and td["temperature_by_options"] == en["temperature_by_options"]
