# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections import Counter
from dataclasses import MISSING, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

import torch

from models.autoports.qwen_qwen3_5_35b_a3b.tt.model import BLOCK_SIZE, PREFILL_CHUNK, Qwen35Model

try:
    from readiness_check.contract import Generator as _ContractGenerator
except ImportError:
    _ContractGenerator = object

CHECKPOINT = Path(os.environ.get("QWEN35_CHECKPOINT", "/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B"))
SAMPLER_ROWS = 32
DEFAULT_KV_BLOCKS = {1: 8256, 2: 16448, 4: 32832}
DEFAULT_REQUEST_SLOTS = {1: 32, 2: 32, 4: 32}
SEED_MODES = ("skip", "host")
PREFILL_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)
WARMUP_FIRST = (1, 2, 128, 129, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 4096, 4097, 8192)
WARMUP_CONTINUATION = WARMUP_FIRST


def _greedy_params():
    from models.common.sampling.generator import SamplingParams

    return SamplingParams(temperature=1.0, top_k=1, top_p=0.0)


class Qwen35Generator(_ContractGenerator):
    def __init__(self, model: Qwen35Model, tokenizer, *, kv_blocks: Optional[int] = None,
                 decode_batches: Sequence[int] = (1,), max_top_k: int = 32, audit: bool = False,
                 warmup_first: Sequence[int] = WARMUP_FIRST, warmup_continuation: Sequence[int] = WARMUP_CONTINUATION,
                 seed_mode: str = "host", greedy_sampler: str = "topk"):
        import ttnn

        if seed_mode not in SEED_MODES:
            raise ValueError(f"seed_mode must be one of {SEED_MODES}")
        self.seed_mode = seed_mode
        if greedy_sampler not in ("topk", "argmax"):
            raise ValueError("greedy_sampler must be topk or argmax")
        self.greedy_sampler = greedy_sampler

        if seed_mode == "skip" and model.tp > 1:
            raise ValueError("seed_mode='skip' lets device token copies diverge on multichip meshes")
        self.model = model
        model.audit = audit
        self.warmup_first, self.warmup_continuation = tuple(warmup_first), tuple(warmup_continuation)
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.tp = model.tp
        self.vocab_size = model.vocab_size
        self.local_vocab = model.lm_head.shape[-1]
        self.decode_batches = tuple(sorted(set(int(b) for b in decode_batches)))
        if not self.decode_batches or self.decode_batches[0] < 1 or self.decode_batches[-1] > model.max_batch_size:
            raise ValueError("decode batches must be within the configured request slots")
        self.blocks_per_seq = -(-model.max_seq_len // BLOCK_SIZE)
        self.kv_pool = model.kv_pool or model.allocate_kv_cache(kv_blocks or DEFAULT_KV_BLOCKS[self.tp])
        self.counters = Counter()
        self._replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        self._composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        self.tt_tokens = self._device(torch.zeros(1, 1, 1, SAMPLER_ROWS, dtype=torch.int32), ttnn.uint32)
        self.tt_positions = {b: self._device(torch.zeros(b, dtype=torch.int32), ttnn.int32)
                             for b in self.decode_batches}
        self.tt_page_tables = {b: self._device(torch.full((b, self.blocks_per_seq), self.kv_pool.scratch_block, dtype=torch.int32), ttnn.int32)
                               for b in self.decode_batches}
        self.tt_prefill_table = self._device(torch.full((1, self.blocks_per_seq), self.kv_pool.scratch_block, dtype=torch.int32), ttnn.int32)
        self._page_table_host = {b: None for b in self.decode_batches}
        self._prefill_table_host = None
        self._decode_traces = {}
        self._live_slots = set()
        self._sampling_mode = None
        self._greedy_lanes = True
        self._seeds_written = False
        self._prepared = False
        self.sampler = self._build_sampler()

    def _device(self, tensor, dtype):
        import ttnn

        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device,
                               mesh_mapper=self._replicate)

    def _host(self, tensor, dtype):
        import ttnn

        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._replicate)

    def _build_sampler(self):
        from models.common.sampling.generator import SamplingGenerator

        tt_ccl = None
        if self.tp > 1:
            from models.common.modules.tt_ccl import TT_CCL

            tt_ccl = TT_CCL(self.mesh_device)
        import ttnn

        model_config = {}
        if self.greedy_sampler == "argmax":
            model_config["SAMPLING_AG_CONFIG"] = dict(
                allow_force_argmax=True, num_links=1,
                topology=ttnn.Topology.Ring if self.tp == 4 else ttnn.Topology.Linear)
        args = SimpleNamespace(vocab_size=self.vocab_size, padded_vocab_size=self.local_vocab * self.tp,
                               cluster_shape=tuple(self.mesh_device.shape), max_batch_size=SAMPLER_ROWS,
                               max_top_k=32, pad_logits_to_power_of_2=self.tp > 1, model_config=model_config)
        return SamplingGenerator(args=args, mesh_device=self.mesh_device, tt_ccl=tt_ccl)

    def _sample(self, logits, **kwargs):
        seeds = self.sampler.seed_manager
        if self._greedy_lanes:
            self.sampler.sample(logits, tt_out_tok=self.tt_tokens, **kwargs)
            return
        if self.seed_mode == "host":
            seeds.write_device_seed_values([seeds._next_unseeded_device_seed() for _ in range(seeds.max_batch_size)])
            self.counters["seed_writes"] += 1
        else:
            if not self._seeds_written:
                seeds.reset_seed(None, list(range(seeds.max_batch_size)))
                self._seeds_written = True
            if seeds.get_new_values() is not None:
                self.counters["seed_writes"] += 1
        self.sampler.sample(logits, tt_out_tok=self.tt_tokens, **kwargs)

    def _set_sampling(self, params):
        from models.common.sampling.generator import SamplingParams, format_sampling_params

        def values(name, default):
            value = getattr(params, name, default)
            return [default if v is None else v for v in (value if isinstance(value, (list, tuple)) else [value])]

        if any(s is not None for s in values("seed", None)):
            raise ValueError("explicit request seeds run the sampler outside its trace; send seed=None")
        for name, default in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
            if any(v != default for v in values(name, default)):
                raise ValueError(f"{name} is not supported: penalty programs and token state are not prepared")
        if any(values("enable_log_probs", False)) or any(values("num_logprobs", 0)):
            raise ValueError("log-probabilities are not supported on 1, 2 or 4 devices")
        if not isinstance(params.temperature, list):
            params = replace(params, temperature=[params.temperature] * SAMPLER_ROWS)
        key = tuple(tuple(values(f.name, None if f.default is MISSING else f.default)) for f in fields(SamplingParams))
        if key != self._sampling_mode:
            formatted = format_sampling_params(params, SAMPLER_ROWS)
            self.sampler.reset_sampling_params(formatted)
            self._greedy_lanes = all(k == 1 for k in formatted.top_k)
            self._sampling_mode = key
            self.counters["sampling_param_refreshes"] += 1

    def _sampler_logits(self, logits, batch):
        import ttnn

        rows = ttnn.reshape(logits, (1, 1, batch, self.local_vocab))
        if batch == SAMPLER_ROWS:
            return rows
        return ttnn.pad(rows, ((0, 0), (0, 0), (0, SAMPLER_ROWS - batch), (0, 0)), 0.0)

    def _decode_device(self, batch):
        import ttnn

        tokens = ttnn.reshape(ttnn.slice(self.tt_tokens, (0, 0, 0, 0), (1, 1, 1, batch)), (batch, 1))
        logits = self.model.decode(tokens, self.tt_positions[batch], self.tt_page_tables[batch],
                                   request_slots=tuple(range(batch)))
        ttnn.plus_one(self.tt_positions[batch], skip_negative_entries=True)
        return self._sampler_logits(logits, batch)

    def _write_tokens(self, tokens):
        import ttnn

        host = torch.zeros(1, 1, 1, SAMPLER_ROWS, dtype=torch.int32)
        host[0, 0, 0, : len(tokens)] = torch.as_tensor(tokens, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(self._host(host, ttnn.uint32), self.tt_tokens)
        self.counters["token_refreshes"] += 1

    def _write_positions(self, batch, positions):
        import ttnn

        ttnn.copy_host_to_device_tensor(self._host(torch.as_tensor(positions, dtype=torch.int32).reshape(batch),
                                                   ttnn.int32), self.tt_positions[batch])
        self.counters["position_refreshes"] += 1

    def _write_page_table(self, batch, table):
        import ttnn

        full = torch.full((batch, self.blocks_per_seq), self.kv_pool.scratch_block, dtype=torch.int32)
        full[:, : table.shape[1]] = table.to(torch.int32)
        previous = self._page_table_host[batch]
        if previous is not None and torch.equal(previous, full):
            return
        ttnn.copy_host_to_device_tensor(self._host(full, ttnn.int32), self.tt_page_tables[batch])
        self._page_table_host[batch] = full
        self.counters["page_table_refreshes"] += 1

    def _write_prefill_table(self, row):
        import ttnn

        full = torch.full((1, self.blocks_per_seq), self.kv_pool.scratch_block, dtype=torch.int32)
        full[0, : row.shape[-1]] = row.to(torch.int32).reshape(-1)
        if self._prefill_table_host is not None and torch.equal(self._prefill_table_host, full):
            return
        ttnn.copy_host_to_device_tensor(self._host(full, ttnn.int32), self.tt_prefill_table)
        self._prefill_table_host = full
        self.counters["prefill_table_refreshes"] += 1

    def read_tokens(self, batch):
        import ttnn

        self.counters["token_readbacks"] += 1
        host = ttnn.to_torch(ttnn.get_device_tensors(self.tt_tokens)[0])
        return host.reshape(-1)[:batch].to(torch.int64)

    def _read_tokens_async(self):
        import ttnn

        return self.tt_tokens.cpu(blocking=False), ttnn.record_event(self.mesh_device, 0)

    def _wait_tokens(self, pending, batch):
        import ttnn

        host, event = pending
        ttnn.event_synchronize(event)
        self.counters["token_readbacks"] += 1
        return ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[:batch].to(torch.int64)

    def decode_steps(self, batch, steps, sampling_params, stop=None):
        self._set_sampling(sampling_params)
        tokens, pending = [], None
        for _ in range(steps):
            self._replay(batch, sample=True)
            current = self._read_tokens_async()
            if pending is not None:
                tokens.append(self._wait_tokens(pending, batch))
                if stop is not None and stop(tokens[-1]):
                    pending = None
                    break
            pending = current
        if pending is not None:
            tokens.append(self._wait_tokens(pending, batch))
        return tokens

    def _logits_to_host(self, logits, rows=None):
        import ttnn

        self.counters["logit_readbacks"] += 1
        host = ttnn.to_torch(logits, mesh_composer=self._composer).float()
        host = host.reshape(-1, host.shape[-1])[:, : self.vocab_size]
        return host if rows is None else host[:rows]

    def default_page_table(self, rows: int, tokens_per_row: int) -> torch.Tensor:
        blocks = -(-tokens_per_row // BLOCK_SIZE)
        if rows * blocks > self.kv_pool.num_blocks:
            raise ValueError(f"{rows} rows x {tokens_per_row} tokens exceed the KV pool of {self.kv_pool.num_blocks} blocks")
        return torch.arange(rows * blocks, dtype=torch.int32).reshape(rows, blocks)

    def prepare(self):
        if self._prepared:
            return
        import ttnn

        self.counters["warmup_runs"] += 1
        greedy = _greedy_params()
        table = self.default_page_table(1, 2 * PREFILL_CHUNK)
        self._write_prefill_table(table[0])
        slot = (0,)
        self._set_sampling(greedy)
        for length in self.warmup_first:
            self.model.reset_request_state(slot)
            logits = self.model.prefill(torch.randint(0, self.vocab_size, (1, length)), page_table=self.tt_prefill_table,
                                        request_slots=slot)
            self.sampler.sample(self._sampler_logits(logits, 1), enable_trace=False, tt_out_tok=self.tt_tokens)
            logits.deallocate(True)
        for length in self.warmup_continuation:
            logits = self.model.prefill(torch.randint(0, self.vocab_size, (1, length)), start_pos=PREFILL_CHUNK,
                                        page_table=self.tt_prefill_table, request_slots=slot)
            logits.deallocate(True)
        for bucket in PREFILL_BUCKETS:
            hidden = ttnn.zeros((1, bucket, self.model.hidden_size), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.model.all_logits(hidden, 1)
            hidden.deallocate(True)
        top_k = replace(greedy, top_k=32, top_p=0.9, temperature=0.7)
        for batch in self.decode_batches:
            self._write_page_table(batch, self.default_page_table(batch, 256))
            self._write_positions(batch, [3] * batch)
            for params in (greedy, top_k):
                self._set_sampling(params)
                logits = self._decode_device(batch)
                self.sampler.set_trace_bucket(batch)
                self.sampler.sample(logits, enable_trace=False, tt_out_tok=self.tt_tokens)
                logits.deallocate(True)
        self.model.zero_kv_cache()
        self.model.reset_request_state()
        self._write_tokens([0])
        ttnn.synchronize_device(self.mesh_device)
        self._prepared = True
        if os.environ.get("QWEN35_STRICT_CACHE"):
            self.mesh_device.set_program_cache_misses_allowed(False)

    def _decode_trace(self, batch):
        import ttnn
        from ttnn.tools import trace_allocation_tracker

        if batch not in self._decode_traces:
            if batch not in self.decode_batches:
                raise ValueError(f"decode batch {batch} is not a prepared variant {self.decode_batches}")
            self.prepare()
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            with self.model._guard():
                logits = self._decode_device(batch)
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            trace_allocation_tracker.acknowledge_corruptible(logits)
            self._decode_traces[batch] = (trace_id, logits)
            self.counters["decode_captures"] += 1
        return self._decode_traces[batch]

    def _replay(self, batch, sample=True):
        import ttnn

        trace_id, logits = self._decode_trace(batch)
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        self.counters["decode_replays"] += 1
        if sample:
            self.sampler.set_trace_bucket(batch)
            self._sample(logits, enable_trace=True, skip_precompile=True)
            self.counters["sampling_replays"] += 1
        return logits

    def _prefill_rows(self, tokens, page_table, prompt_lens, start_pos, request_slots, return_all_logits,
                      reset=True):
        outputs = []
        for row, length in enumerate(prompt_lens):
            slot = (request_slots[row],)
            start = int(start_pos[row])
            if reset and start == 0:
                self.model.reset_request_state(slot)
            self._write_prefill_table(page_table[row])
            ids = tokens[row : row + 1, :length]
            outputs.append(self.model.prefill(ids, start_pos=start, page_table=self.tt_prefill_table,
                                              request_slots=slot, return_all_logits=return_all_logits))
        return outputs

    def prefill_forward(self, tokens, *, page_table, kv_cache=None, prompt_lens, return_all_logits=False,
                        start_pos=None, request_slots=None, empty_slots=None):
        self._check_kv(kv_cache)
        batch = tokens.shape[0]
        if empty_slots is not None:
            empty_slots = [int(s) for s in empty_slots]
            if request_slots is not None and [int(s) for s in request_slots] != empty_slots:
                raise ValueError("empty_slots and request_slots name different slots")
            request_slots = empty_slots
        request_slots = list(range(batch)) if request_slots is None else [int(s) for s in request_slots]
        if len(request_slots) != batch or len(set(request_slots)) != batch:
            raise ValueError("prefill needs one distinct request slot per row")
        if not all(0 <= s < self.model.max_batch_size for s in request_slots):
            raise ValueError("request slot outside the configured request slots")
        self.prepare()
        start_pos = [0] * batch if start_pos is None else list(start_pos)
        outputs = self._prefill_rows(tokens, page_table, list(prompt_lens), start_pos, request_slots,
                                     return_all_logits)
        self._live_slots.update(request_slots)
        if not return_all_logits:
            rows = [self._logits_to_host(logits, 1) for logits in outputs]
            for logits in outputs:
                logits.deallocate(True)
            return torch.stack(rows)
        result = [torch.cat(chunks) for chunks in outputs]
        width = max(p.shape[0] for p in result)
        return torch.stack([torch.nn.functional.pad(p, (0, 0, 0, width - p.shape[0])) for p in result])

    def decode_forward(self, tokens, start_pos, *, page_table, kv_cache=None, sampling_params=None,
                       enable_trace=True, request_slots=None, slot_remap=None):
        if not enable_trace:
            raise ValueError("decode runs only through the captured trace")
        self._check_kv(kv_cache)
        batch = tokens.shape[0]
        if batch not in self.decode_batches:
            raise ValueError(f"decode batch {batch} is not prepared; prepared batches: {self.decode_batches}")
        for name, slots in (("request_slots", request_slots), ("slot_remap", slot_remap)):
            if slots is not None and [int(s) for s in torch.as_tensor(slots).reshape(-1).tolist()] != list(range(batch)):
                raise ValueError(f"{name}: decode row r uses request slot r; prefill each request into the slot of its row")
        positions = [int(p) for p in torch.as_tensor(start_pos).reshape(-1).tolist()]
        hidden = [row for row, position in enumerate(positions) if position < 0 and row in self._live_slots]
        if hidden:
            raise ValueError(f"inactive decode rows {hidden} hold live requests; an inactive row overwrites its "
                             "slot's recurrent state. Keep the row active or call release_slots first")
        table = torch.as_tensor(page_table).to(torch.int32).clone()
        for row, position in enumerate(positions):
            if position < 0:
                positions[row] = 0
                table[row] = self.kv_pool.scratch_block
        self._write_tokens(tokens.reshape(-1).tolist())
        self._write_positions(batch, positions)
        self._write_page_table(batch, table)
        if sampling_params is None:
            logits = self._replay(batch, sample=False)
            return self._logits_to_host(logits, batch)
        self._set_sampling(sampling_params)
        self._replay(batch, sample=True)
        return self.read_tokens(batch)

    def _check_kv(self, kv_cache):
        if kv_cache is None or kv_cache is self.kv_pool.caches:
            return
        if len(kv_cache) != len(self.kv_pool.caches) or any(
            a is not b for pair_a, pair_b in zip(kv_cache, self.kv_pool.caches) for a, b in zip(pair_a, pair_b)
        ):
            raise ValueError("pass the generator-owned kv_cache (generator.kv_pool.caches)")

    def prefill_logits(self, prompt_token_ids: List[int]) -> torch.Tensor:
        self.prepare()
        ids = torch.tensor([prompt_token_ids], dtype=torch.int64)
        table = self.default_page_table(1, len(prompt_token_ids))
        logits = self.prefill_forward(ids, page_table=table, prompt_lens=[len(prompt_token_ids)],
                                      return_all_logits=True)
        return logits[:, : len(prompt_token_ids)]

    def generate(self, prompt_token_ids: List[int], max_new_tokens: int, *, next_input=None,
                 enable_trace: bool = True, sampling_params=None, host_sampling: bool = False,
                 stop_on_eos: bool = False, **kwargs) -> List[int]:
        if not enable_trace:
            raise ValueError("generate decodes only through the captured trace")
        self.prepare()
        params = _greedy_params() if sampling_params is None else sampling_params
        length = len(prompt_token_ids)
        if not 0 < length <= self.model.max_seq_len:
            raise ValueError(f"prompt length {length} is outside the supported context {self.model.max_seq_len}")
        max_new_tokens = min(max_new_tokens, self.model.max_seq_len - length + 1)
        table = self.default_page_table(1, min(length + max_new_tokens, self.model.max_seq_len))
        ids = torch.tensor([prompt_token_ids], dtype=torch.int64)
        (logits,) = self._prefill_rows(ids, table, [length], [0], [0], False)
        self._live_slots.add(0)
        if host_sampling:
            predicted = int(self._logits_to_host(logits, 1)[0].argmax())
            self._write_tokens([predicted])
        else:
            self._set_sampling(params)
            self.sampler.set_trace_bucket(None)
            self._sample(self._sampler_logits(logits, 1), enable_trace=False)
            predicted = int(self.read_tokens(1)[0])
        logits.deallocate(True)
        self._write_page_table(1, table)
        self._write_positions(1, [length])
        predictions = [predicted]
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if next_input is None and not host_sampling:
            if stop_on_eos and eos is not None and predicted == eos:
                return predictions
            stop = (lambda t: int(t[0]) == eos) if stop_on_eos and eos is not None else None
            tokens = self.decode_steps(1, max_new_tokens - 1, params, stop=stop)
            predictions += [int(t[0]) for t in tokens]
            return predictions
        for step in range(max_new_tokens - 1):
            feed = predicted if next_input is None else int(next_input(step, predicted))
            if feed != predicted:
                self._write_tokens([feed])
            if stop_on_eos and eos is not None and feed == eos:
                break
            if host_sampling:
                logits = self._replay(1, sample=False)
                predicted = int(self._logits_to_host(logits, 1)[0].argmax())
                self._write_tokens([predicted])
            else:
                self._set_sampling(params)
                self._replay(1, sample=True)
                predicted = int(self.read_tokens(1)[0])
            predictions.append(predicted)
        if next_input is not None and len(predictions) == max_new_tokens:
            next_input(max_new_tokens - 1, predicted)
        return predictions

    def release_slots(self, slots) -> None:
        self._live_slots.difference_update(int(s) for s in slots)

    def reset(self) -> None:
        self._live_slots.clear()
        self._seeds_written = False
        self.model.zero_kv_cache()
        self.model.reset_request_state()
        for batch in self.decode_batches:
            self._page_table_host[batch] = None
        self._prefill_table_host = None
        self.counters["resets"] += 1


def build_generator(model_dir, mesh_device, **kwargs) -> Qwen35Generator:
    from transformers import AutoTokenizer

    checkpoint = Path(kwargs.pop("checkpoint", CHECKPOINT))
    layer_ids = kwargs.pop("layer_ids", None)
    cache = kwargs.pop("tensor_cache_path", None)
    kv_blocks = kwargs.pop("kv_blocks", None) or DEFAULT_KV_BLOCKS[mesh_device.get_num_devices()]
    model = Qwen35Model.from_checkpoint(checkpoint, mesh_device, layer_ids=layer_ids,
                                        max_batch_size=kwargs.pop("max_batch_size", None)
                                        or DEFAULT_REQUEST_SLOTS[mesh_device.get_num_devices()],
                                        tensor_cache_path=None if cache is None else Path(cache), kv_blocks=kv_blocks,
                                        lm_head_dram=kwargs.pop("lm_head_dram", False))
    batched = kwargs.pop("batched_decode", True)
    moe_overrides = kwargs.pop("moe_policy", None) or {}
    for layer in model.layers:
        layer.batched_decode = batched
        moe = getattr(layer.layer.feed_forward, "module", None)
        if moe_overrides and hasattr(moe, "policy"):
            moe.policy = replace(moe.policy, **moe_overrides)
            dispatch = getattr(getattr(moe.moe, "experts", None), "dispatch", None)
            if dispatch is not None and hasattr(dispatch, "policy"):
                dispatch.policy = replace(dispatch.policy, **moe_overrides)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    generator = Qwen35Generator(model, tokenizer, **kwargs)
    generator.prepare()
    return generator
