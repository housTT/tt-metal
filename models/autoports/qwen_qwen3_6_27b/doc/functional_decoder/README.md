# Qwen/Qwen3.6-27B — functional decoder

TTNN implementation of the Qwen3.6-27B (HF `model_type: qwen3_5`) decoder layers, validated
against the HuggingFace reference on one Blackhole chip (1x1 mesh).

Hardware, for the record: this host has **two p300c boards of two Blackhole chips each**
(`/sys/class/tenstorrent/*/tt_card_type`, `tt_serial`). Board `000004613192404C` holds
`/dev/tenstorrent/{0,1}` and chip 0's ARC is wedged; board `0000046131924022` holds
`/dev/tenstorrent/{2,3}` and both are healthy. Everything here runs on **device 2**, a chip of
the intact board. Earlier revisions of this document said "p150"; that was wrong.

* Implementation: [`../../tt/functional_decoder.py`](../../tt/functional_decoder.py)
* Shape contract: [`../../tt/model_config.py`](../../tt/model_config.py)
* HF reference harness: [`../../reference/hf_reference.py`](../../reference/hf_reference.py)
* Tests: [`../../tests/test_functional_decoder.py`](../../tests/test_functional_decoder.py),
  [`../../tests/test_functional_decoder_perf.py`](../../tests/test_functional_decoder_perf.py)
* Bringup narrative, including the three bugs found and fixed: [`work_log.md`](work_log.md)
* Context capability: [`../context_contract.json`](../context_contract.json)
* Every measured number: [`pcc_evidence.json`](pcc_evidence.json) (262 records, 258 numeric)

## Layer kinds

`text_config.layer_types` alternates three `linear_attention` layers and one
`full_attention` layer (`full_attention_interval = 4`) over 64 layers, so **both** kinds are
implemented by the same `FunctionalDecoder` and both are covered by every test:

| kind | example layer | mixer | state |
|---|---|---|---|
| `linear_attention` | 0 | `Qwen3_5GatedDeltaNet` — fused qkv/z/b/a projections, depthwise causal conv1d (width 4) over 10240 channels, L2-normalised Q/K, gated delta rule, z-gated RMS norm | conv state `[4, 10240]` + recurrent state `[48, 128, 128]`, float32 |
| `full_attention` | 3 | `Qwen3_5Attention` — GQA 24/4 heads, `head_dim` 256, **output gate** (`q_proj` emits `2*heads*head_dim`, second half is a sigmoid gate), per-head Q/K RMS norms, **partial** RoPE over 64 of 256 channels | paged KV cache |

Both share the residual structure, the two 1-centred RMS norms (`y = normed * (1 + w)`) and
the SwiGLU MLP. `Qwen3_5RMSNormGated` inside the delta net is *not* 1-centred.

## API

```python
class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device,
                        max_batch=1, max_seq_len=None, block_size=64, max_num_blocks=None,
                        weight_dtype=ttnn.bfloat16, cache_dtype=ttnn.bfloat16,
                        state_dtype=ttnn.float32) -> "FunctionalDecoder": ...

    def prefill_forward(self, hidden_states, *, user_id=0, page_table=None,
                        page_tables_per_chunk=None, rot_mats=None): ...

    def decode_forward(self, hidden_states, *, current_pos=None, page_table=None,
                       rot_mats=None): ...

    def prefill_chunk_plan(self, seq_len) -> list[tuple[int, int, int]]: ...
    def prepare_decode_state(self) -> None: ...
```

* `state_dict` keys are exactly HF's `Qwen3_5DecoderLayer` submodule-relative names
  (`input_layernorm.weight`, `self_attn.q_proj.weight`, `linear_attn.in_proj_qkv.weight`, …).
  `from_state_dict` is the **only** place `torch` is used.
* `max_seq_len` defaults to `hf_config.max_position_embeddings` (262144).
* **prefill** takes `[1, 1, seq_len, hidden]` for one user. Any `1 <= seq_len <= max_seq_len`
  is accepted — the layer owns all chunking, padding and masking and slices the output back
  to `seq_len`. Batched prefill is a loop over `user_id`.
* **decode** takes `[1, 1, batch, hidden]` with `batch == max_batch`, and a device int32
  `current_pos` tensor of shape `[batch]` so the pass is traceable. `linear_attention`
  ignores `current_pos`/`page_table`/`rot_mats`: the recurrence carries the whole history.
* `prepare_decode_state()` folds the per-user `linear_attention` prefill state into the
  batch-wide buffers the traced decode updates in place. Call it once after prefilling all
  users and before capturing the trace.

The full contract, including the per-chunk page-table slices and the cache-reset semantics,
is the module docstring of `tt/functional_decoder.py`.

## Running

All device jobs run from `/home/ttuser/dev/qwen/rundir` with
[`ttenv.sh`](ttenv.sh) sourced — see [`work_log.md`](work_log.md) §0 for why (one wedged
board is excluded, and kernel sources must come from the built tree).

```bash
cd /home/ttuser/dev/qwen/rundir && source ./ttenv.sh

# full functional suite (42 tests, ~7 min)
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py -v -s

# full advertised context, 262144 tokens (2 tests, ~3 min)
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py \
    -k test_full_advertised_context --long-context -v -s

# collect every recorded number into pcc_evidence.json
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
    $REPO/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/logs/*.log
```

Tests use synthetic weights generated deterministically from
[`weight_stats.json`](weight_stats.json) (real per-tensor name/shape/mean/std, with
`linear_attn.A_log` and `linear_attn.dt_bias` stored **verbatim** so the SSM gating is
realistic) at the **real** config shapes; `test_real_weights` loads the actual checkpoint.
Regenerate the stats with
`python -m models.autoports.qwen_qwen3_6_27b.scripts.extract_weight_stats`.

## Correctness

Acceptance bar: **PCC >= 0.995** (the skill default; no model-specific reason to move it).
Minimum over the 258 numeric records, **excluding the one known gap below**: **0.998817**.
The other 4 records are the boolean determinism flags. Exactly one measurement in the whole
stage is under the bar.

**One known gap, blocked on a tt-metal kernel defect:** `full_attention` **decode** at
position 262143 reaches only 0.5503. Prefill at that context was the same defect and *is*
fixed (0.9838 → 0.9988, see below); the decode op cannot be worked around from Python.
`work_log.md` sections 7–9 have the root cause, the fix, and why the decode half is stuck.
Every other measurement in this stage clears the bar.

| measurement | `linear_attention` | `full_attention` |
|---|---|---|
| prefill vs HF, seq 1 / 17 / 128 / 2048 / 2049 / 4096 / 5000 | min 0.999888 | min 0.998870 |
| prefill vs HF, longest reference-checkable length | 0.999942 @ 16385 | 0.998897 @ 8191 |
| decode vs HF, 4 steps after prefill 17 / 2048 / 2049 / 5000 | min 0.999927 | min 0.997878 |
| batch 32, 32 unequal prompts 64..3071, permuted page table — prefill | min 0.999940 | min 0.998878 |
| batch 32 — decode | min 0.999931 | min 0.998362 |
| **real checkpoint weights** — prefill @ 2049 | **0.999968** | **0.999952** |
| **real checkpoint weights** — decode @ 2049 | **0.999989** | **0.999972** |
| traced decode, replay output vs HF (3 replays) | min 0.999935 | min 0.998490 |
| on-device state vs HF cache after prefill 2049 | conv 0.999995, recurrent 0.999981 | K 0.999989, V 0.999993 |
| page block size 32 and 128 instead of 64 | 0.999942 | 0.998912 |
| full context 262143 — prefill tail vs HF | 0.999942 (last 8192) | **0.998817** (last 256) |
| full context 262143 — state / paged cache vs HF | conv 0.999995, recurrent 0.999981 | K 0.999989, V 0.999993 |
| full context 262143 — decode at position 262143 | 0.999946 | **0.550293** — see the gap note above |
| BFP8 KV cache — prefill / decode @ 2049 | — | 0.998850 / 0.998766 |
| traced decode at batch 4, per-user positions | min 0.999939 | min 0.998465 |
| pad-below-one-tile lengths 735..768 — prefill / decode | min 0.999942 / 0.999934 | min 0.999078 / 0.998707 |

`full_attention` sits systematically ~1e-3 below `linear_attention` because its SDPA path
cannot use fp32 destination accumulation (`head_dim` 256 > 128); the value drifts slowly with
sequence length (0.99997 at 17 → 0.99887 at 8191), which is ordinary bf16 accumulation, not a
correctness cliff. All values clear the bar with margin.

### Sequence-length coverage

| class | lengths | test |
|---|---|---|
| sub-tile smoke | 1, 17 | `test_prefill_pcc` |
| below one chunk | 128 | `test_prefill_pcc` |
| exactly at chunk / page / SDPA-chunk boundaries | 2048, 4096 | `test_prefill_pcc` |
| one token past a boundary | 2049 | `test_prefill_pcc`, `test_decode_pcc` |
| long, divisible by none of {32, 64, 256, 2048} | 5000, 8191, 16385 | `test_prefill_pcc`, `test_prefill_pcc_long` |
| pad amount **below one tile** (`ttnn.pad` aliasing regression) | 735, 736, 737, 743, 767, 768 | `test_prefill_decode_pad_below_one_tile` — prefill **and** decode |
| every batch-32 prompt | 64, 161, 258, …, 3071 (24 of the 32 are non-64-divisible) | `test_batched_users` |
| full advertised context | 262143 | `test_full_advertised_context` |

No divisibility requirement is imposed on the public prefill API. Every length in the table is
asserted by a test; none of them is probe-only.

### Paged KV cache and state

* Page tables are a **shuffled permutation** of all `max_batch * blocks_per_user` blocks, not
  an identity map, so any assumption of contiguous or zero-based slots shows up as a PCC
  failure (`tests/harness.py::make_page_table`).
* Prefill uses `ttnn.experimental.paged_fill_cache` per chunk with a per-chunk page-table
  slice; decode uses `ttnn.experimental.paged_update_cache` with a device `current_pos`
  tensor and `ttnn.transformer.paged_scaled_dot_product_attention_decode`.
* `test_linear_state_and_kv_cache_match_reference` un-pages the device cache and compares it
  against HF's own cache object, so the cache contents are checked, not just the layer output.
* Block size is a constructor parameter, tested at 32, 64 and 128 by
  `test_alternate_page_block_size`. That test is `full_attention` only —
  `linear_attention` has no paged cache, so parametrising it there would assert nothing.
* `cache_dtype=ttnn.bfloat8_b` is covered by `test_bfloat8_kv_cache`, which is what exercises
  the asymmetric dtype contract: prefill casts K/V to the cache dtype before
  `paged_fill_cache`, while decode must **not** cast the `paged_update_cache` update tensors.
* A per-chunk page table that is shorter than the padded chunk is an assertion, not a silent
  trim — dropping cache writes for the tail of a prompt would only surface much later as a
  wrong decode.

## Determinism

`test_determinism` asserts **bit-identical** outputs for repeated identical inputs, for
prefill and for decode, for both layer kinds (`prefill_bit_identical` /
`decode_bit_identical` = `true` in `pcc_evidence.json`).

## Runtime fallback audit

`test_no_runtime_host_fallback` does both halves of the audit:

1. a source scan of `tt/functional_decoder.py` over the module-level helpers, `__init__`
   **and** everything after the setup section, rejecting `torch`, `from_torch`, `to_torch`,
   `as_tensor` and `.cpu()`. Only the module docstring and the body of `from_state_dict` —
   the documented setup-time boundary — are excluded, and each scanned region is asserted
   non-empty so a moved anchor cannot silently disable the check;
2. a live run of one prefill and one decode pass with `ttnn.from_torch`, `ttnn.to_torch` and
   `ttnn.as_tensor` replaced by raising stubs.

Both layer kinds pass. Weight transposition, dtype selection, cache/state allocation and the
`eye`/zero-prefix constants are all built in `from_state_dict`.

## Watcher

`TT_METAL_WATCHER=10` over both layer kinds' paged prefill, paged decode and traced decode:
**4 passed**, watcher log clean. Full audit, including the exact command, the grep that finds
no fatal/assert/sanitize/corruption lines, and the line-category census:
[`watcher/WATCHER_AUDIT.md`](watcher/WATCHER_AUDIT.md). Raw log:
`watcher/generated/watcher/watcher.log`. Watcher and the device profiler were run separately.

## Performance

Warmed measurements on one Blackhole chip, batch 1, from a **Tracy device-profiler** run with
the measured window delimited by signposts. All host work — input construction, upload, trace
capture — happens before the start signpost, and the profiler's marker buffers are drained
(`ttnn.ReadDeviceProfiler`) between warm-up and the measured window. Markers *are* dropped
during setup, before that drain and before the start signpost; what matters is that the
measured windows are complete, and they provably are — `tt-perf-report`'s op list repeats with
an exact period (96 and 50 ops), so all replays were captured whole.
Decode is **traced**: capture once, then replay `execute_trace` inside the window.

| layer kind | phase | ops in one pass | device kernel time | host wall |
|---|---|---|---|---|
| `linear_attention` | prefill, 2048 tokens | 805 | **150.4 ms** | 161.1 ms |
| `linear_attention` | traced decode, 1 token | 96 | **3.031 ms** | 3.408 ms |
| `full_attention` | prefill, 2048 tokens | 44 | **18.6 ms** | 20.2 ms |
| `full_attention` | traced decode, 1 token | 50 | **2.420 ms** | 2.480 ms |

Re-measured after the review fixes (`TRI_INV_BASE` 8 → 16, fp32 destination accumulation on
the SDPA kernels, and the `SDPA_MAX_K_CHUNKS` k-chunk cap), so these are the numbers the
current code produces.

Decode numbers are the mean of 8 trace replays inside the window; `tt-perf-report`'s op list
repeats with an exact period (96 and 50 ops), which confirms all 8 replays were captured
whole. Device time is the `Device Time` column of the `tt-perf-report --csv` output, **in
microseconds** (the raw Tracy ops CSV exposes the same quantity as
`DEVICE KERNEL DURATION [ns]`).

Artifacts, per layer kind, under [`tracy/`](tracy/):

* `<phase>_ops.csv` — the post-processed Tracy ops CSV, copied verbatim
* `<phase>_ops.csv.provenance` — source path and copy timestamp
* `<phase>_perf_report.txt` — the human-readable `tt-perf-report` table
* `<phase>_perf_report.csv` — the same data as CSV
* `<phase>_perf_report.console.log` — `--csv` mode stdout (roofline summary, file paths)
* `<phase>_perf_report_stacked.{csv,png}` — the stacked breakdown `tt-perf-report` emits
* [`perf_summary.json`](perf_summary.json) — the table above plus the top ops by device time

Commands: [`probes/run_perf.sh`](probes/run_perf.sh), e.g.
`./run_perf.sh linear_attention decode`. It wraps

```bash
python -m tracy -r -p -v -m pytest \
  $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder_perf.py::test_perf_decode_traced[linear_attention] -s -q
tt-perf-report <ops.csv> --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-summary --no-advice
```

Profiling runs use [`ttenv_profiler.sh`](ttenv_profiler.sh), not `ttenv.sh`: the shared
tt-metal install was built with `ENABLE_TRACY=OFF`, so a profiler-enabled copy of the same
commit was built for these four runs. See [`work_log.md`](work_log.md) §3.

### Observations for the optimization stage (not acted on here)

This stage is about correctness, so the implementation is deliberately unoptimised — BF16,
tile layout, DRAM interleaved everywhere. The profile already shows where the work is:

* `linear_attention` prefill is **op-count bound**, not FLOP bound: 832 ops for 2048 tokens
  versus 44 for `full_attention`, because the gated delta rule is a Python loop over the 32
  sub-chunks of the recurrence plus a recursive block inverse.
* The single largest cost is that block inverse. Re-aggregating
  `tracy/linear_attention/prefill_perf_report.csv` by op, the `_unit_tri_inverse` batched
  `32 x 32 x 32` matmuls — its 8x8 and 16x16 blocks stored in 32x32 tiles — run on **one core**
  and accounted for **43 %** of the prefill window at `TRI_INV_BASE = 8`. The two large
  matmuls (`2048 x 5120 x 10240`, `2048 x 17408 x 5120`) that an earlier revision of this
  document called dominant were 2.9 %.
* `TRI_INV_BASE` is now 16, chosen by measurement on real weights rather than by inverse-level
  error alone: 8, 16 and 32 give the same layer PCC to six decimals, and a warmed 2048-token
  prefill takes 204.6 / 160.5 / 135.8 ms respectively
  (`probes/probe_tri_inv_base.py`, `logs/tri_inv_base_sweep.log`). Base 32 is faster still and
  is the first base whose recurrent-state PCC moves; that trade is handed to the optimization
  stage with the numbers attached, not taken here.
* Traced decode is DRAM bound as expected, with the MLP matmul near its own roofline.

## Known limitations

* **`full_attention` decode at long positions** — the one open gap. At position 262143 the
  PCC is 0.5503. Root cause (`work_log.md` §8–9): ttnn's SDPA kernels keep the
  flash-attention running max, softmax denominator and output accumulator in `Float16_b`
  regardless of `fp32_dest_acc_en` (`sdpa_program_factory.cpp:657`, fp32 variant commented out
  citing tt-metal issue #13364). The denominator is a sum of positive terms, so once a chunk's
  contribution falls below bfloat16's half-ULP the denominator stops growing and the output
  comes out uniformly **too large** — a pure scale error, invisible to PCC on the attention
  output itself but exposed by the residual and `o_proj`.
  **Prefill has been fixed**: `_sdpa_program_config` grows the k chunk so at most
  `SDPA_MAX_K_CHUNKS = 512` are merged, taking 262143-token prefill from 0.9838 to 0.9988.
  **Decode cannot be fixed from Python**: the decode op needs a program config whose validity
  condition depends on `ceil((cur_pos+1)/k_chunk)`, a *runtime* quantity, while the config is
  compile-time; violating it is catastrophic (position 12287 → 780440x scale, 261887 → NaN).
  Model-free reproducers for both are in [`probes/`](probes/)
  (`probe_sdpa_synthetic.py`, `probe_sdpa_decode_synthetic.py`) and are what an upstream issue
  needs. Largest `full_attention` decode position verified against HF: **5003**;
  `linear_attention` decode is correct at 262143.
* **`prepare_decode_state()` rewrites every batch slot** from that user's post-prefill
  snapshot, so it cannot insert one newly-prefilled user into a live decode batch without
  rewinding the others. Fine for the prefill-all-then-decode pattern this stage tests; a
  per-slot fold is a strided `ttnn.copy` into one slice of the user-major buffer and belongs
  with the serving stage's slot-eviction policy.
* **Trace capture is exercised at batch 1 and batch 4**, not at batch 32.
* **Prefill is single-user per call** by construction; batched prefill is a loop over
  `user_id`.
* Batch 32 is tested at `max_seq_len` 8192, not at 262144: a batch-32 `full_attention` layer
  at full context would need 32 x 1.07 GB of KV cache, above the 31 GiB measured. That is a
  batch x context product limit for later stages to schedule, not a reduction of the
  advertised context, which is tested in full at batch 1.
* The implementation is deliberately unoptimised — BF16, tile layout, DRAM interleaved. See
  the observations above for where the optimization stage should start.
