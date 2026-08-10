# Qwen/Qwen3.6-27B — functional decoder

TTNN implementation of the Qwen3.6-27B (HF `model_type: qwen3_5`) decoder layers, validated
against the HuggingFace reference on one Blackhole chip (1x1 mesh).

Hardware: this host has **two p300c boards of two Blackhole chips each**
(`/sys/class/tenstorrent/*/tt_card_type`, `tt_serial`). Board `000004613192404C` holds
`/dev/tenstorrent/{0,1}` and chip 0's ARC is wedged; board `0000046131924022` holds
`/dev/tenstorrent/{2,3}` and both are healthy. Everything here runs on **device 2**.

* Implementation: [`../../tt/functional_decoder.py`](../../tt/functional_decoder.py)
* Shape contract: [`../../tt/model_config.py`](../../tt/model_config.py)
* HF reference harness: [`../../reference/hf_reference.py`](../../reference/hf_reference.py)
* Tests: [`../../tests/test_functional_decoder.py`](../../tests/test_functional_decoder.py),
  [`../../tests/test_functional_decoder_perf.py`](../../tests/test_functional_decoder_perf.py)
* Bringup narrative, provenance and the two upstream defects: [`work_log.md`](work_log.md)
* Context capability: [`../context_contract.json`](../context_contract.json)
* Every measured number: [`pcc_evidence.json`](pcc_evidence.json) (268 records: 260 PCC, 4 full-context scale ratios, 4 determinism booleans)

## Layer kinds

`text_config.layer_types` alternates three `linear_attention` layers and one `full_attention`
layer (`full_attention_interval = 4`) over 64 layers, so **both** kinds are implemented by the
same `FunctionalDecoder` and both are covered by every test:

| kind | example layer | mixer | state |
|---|---|---|---|
| `linear_attention` | 0 | `Qwen3_5GatedDeltaNet` — fused qkv/z/b/a projections, depthwise causal conv1d (width 4) over 10240 channels, L2-normalised Q/K, gated delta rule, z-gated RMS norm | conv state `[4, 10240]` + recurrent state `[48, 128, 128]`, float32 |
| `full_attention` | 3 | `Qwen3_5Attention` — GQA 24/4 heads, `head_dim` 256, **output gate** (`q_proj` emits `2*heads*head_dim`, second half is a sigmoid gate), per-head Q/K RMS norms, **partial** RoPE over 64 of 256 channels | paged KV cache |

Both share the residual structure, the two 1-centred RMS norms (`y = normed * (1 + w)`) and
the SwiGLU MLP. `Qwen3_5RMSNormGated` inside the delta net is *not* 1-centred.

`decoder_shapes` fails loudly on any `layer_types` value other than these two, so a third kind
cannot be silently mis-handled.

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

```bash
cd /home/ttuser/dev/qwen/tt-metal
source models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh

# full functional suite (57 tests + 2 long-context skips, ~7.5 min)
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py -v -s

# full advertised context, 262143-token prompt + decode at 262143 (2 tests, ~6.5 min)
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py \
    -k test_full_advertised_context --long-context -v -s

# collect every recorded number into pcc_evidence.json
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
    models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/logs/*.log
```

`ttenv.sh` activates this checkout's own `python_env` and **asserts** that `ttnn` resolves
inside the checkout; other tt-metal trees on this host would otherwise capture the import and
every number here would describe code that is not under test. It prints
`ttnn OK: /home/ttuser/dev/qwen/tt-metal/ttnn/ttnn/__init__.py`.

Tests use synthetic weights generated deterministically from
[`weight_stats.json`](weight_stats.json) (real per-tensor name/shape/mean/std, with
`linear_attn.A_log` and `linear_attn.dt_bias` stored **verbatim** so the SSM gating is
realistic) at the **real** config shapes; `test_real_weights` loads the actual checkpoint.
Regenerate the stats with
`python -m models.autoports.qwen_qwen3_6_27b.scripts.extract_weight_stats`.

## Correctness

Acceptance bar: **PCC >= 0.995** (the skill default; no model-specific reason to move it).
**Minimum over all 260 PCC records: 0.998031.** There is no exception, no waiver and no open
gap: every measurement of the shipped configuration clears the bar. The remaining records are
the boolean determinism flags, all `true`, and the four full-context scale ratios (0.99493
to 0.99853, all inside the ±2 % `SCALE_TOLERANCE` the long-context tests assert — see below).

`pcc_evidence.json` is collected from the three run logs directly under `logs/` — the suite, the
long-context pair and the watcher run. `logs/controls/` holds the two deliberately-failing
reverted-build controls behind the decode fix (see [`work_log.md`](work_log.md) §3.4); they are
one directory down precisely so the flat `logs/*.log` glob cannot fold a run that is *supposed*
to fail into the stage's evidence.

| measurement | `linear_attention` | `full_attention` |
|---|---|---|
| prefill vs HF, seq 1 / 17 / 128 / 2048 / 2049 / 4096 / 5000 | min 0.999913 | min 0.999388 |
| prefill vs HF, longest single-shot reference length | 0.999947 @ 16385 | 0.999415 @ 8191 |
| decode vs HF, 4 steps after prefill 17 / 2048 / 2049 / 5000 | min 0.999933 | min 0.999211 |
| batch 32, 32 unequal prompts 64..3071, permuted page table — prefill | min 0.999944 | min 0.999383 |
| batch 32 — decode | min 0.999938 | min 0.999240 |
| **real checkpoint weights** — prefill @ 2049 | **0.999969** | **0.999964** |
| **real checkpoint weights** — decode @ 2049 | **0.999990** | **0.999988** |
| traced decode, replay output vs HF (3 replays) | min 0.999940 | min 0.999434 |
| traced decode at batch 4, per-user positions | min 0.999947 | min 0.999472 |
| on-device state vs HF cache after prefill 2049 | conv 0.999995, recurrent 0.999984 | K 0.999989, V 0.999993 |
| page block size 32 and 128 instead of 64 — prefill / decode | — | 0.999392 / 0.999490 |
| BFP8 KV cache — prefill / decode @ 2049 | — | 0.999346 / 0.999439 |
| pad-below-one-tile lengths 735..768 — prefill / decode | min 0.999948 / 0.999941 | min 0.999448 / 0.999334 |
| **full context 262143** — prefill tail vs HF | **0.999947** (last 8192) | **0.998031** (last 256) |
| **full context 262143** — state / paged cache vs HF | conv 0.999995, recurrent 0.999984 | K 0.999989, V 0.999993 |
| **full context 262143** — decode at position 262143 | **0.999955** | **0.999201** |
| **full context 262143** — best-fit *scale* vs HF, prefill tail / decode | 0.99853 / 0.99836 | 0.99744 / 0.99493 |

The scale row exists because PCC is scale-invariant and both SDPA defects this stage works
around are *pure scale errors*. `test_full_advertised_context` asserts `H.scale_ratio` inside
`SCALE_TOLERANCE = (0.98, 1.02)` alongside PCC, so the exact failure mode the implementation is
built to avoid is now a gate rather than a narrative. The 1.29 figure quoted for the stock
decode kernel elsewhere in this document is the **op-level** probe `alpha`
(`logs/controls/sdpa_decode_stock_baseline.log`); the stock kernel's *layer-level* scale is not
a recorded number, because the reverted-build control fails the decode **PCC** assertion
(0.977888, already below the bar) before the scale assertion is reached.

`full_attention` sits systematically ~5e-4 below `linear_attention`; that is ordinary bfloat16
attention accumulation, and it drifts slowly with context (0.99997 at seq 1 → 0.998031 at
262143), not a correctness cliff.

### The two upstream tt-metal defects behind those last rows

Both are characterised, both have model-free reproducers in [`probes/`](probes/), and neither
reduces the advertised capability. [`work_log.md`](work_log.md) §2.2 and §3 have the full
narrative; in short:

1. **Prefill SDPA loses a little softmax denominator per k-chunk merge**, one-sidedly, so the
   output comes out uniformly too large by a factor that tracks the merge count (1.09x at
   262144 keys in 512 merges, on synthetic worst-case inputs). `SDPA_MAX_K_CHUNKS = 512` caps
   the merge count, which is also the floor — 512 is the largest k chunk that fits L1 at
   `head_dim` 256. Reproducer: `probes/probe_sdpa_synthetic.py`.
2. **Decode SDPA had the same defect far worse (37.7x at position 262143), plus a broken
   cross-core tree reduction** that is correct only when `num_k_chunks` is 1 or a multiple of
   `2 * cores_per_head` — a runtime quantity against a compile-time config, with catastrophic
   violations (3705x at position 1023, NaN at 261887). **This is what the previous pass could
   not fix and is fixed here**: `sdpa_decode_program_factory.cpp` now keeps the core-local
   flash accumulators in fp32 under `fp32_dest_acc_en` at one core per head, and the layer
   pins `k_chunk = 512, max_cores_per_head_batch = 1`, which takes the tree reduction out of
   the picture. The scale is then 0.995–1.017 at every position from 1023 to 262143.
   Reproducer: `probes/probe_sdpa_decode_synthetic.py`.

The decode fix is **opt-in**: it fires only when the caller explicitly sets
`max_cores_per_head_batch = 1` *and* `fp32_dest_acc_en`. The derived `num_cores_per_head == 1`
alone is not enough to trigger it, because that value also collapses to 1 for ordinary batched
decode (B x KV heads >= the core grid) — an earlier revision of this stage gated on the derived
value and a stage review correctly rejected it. Blast radius is measured three ways: no caller
outside this autoport passes `1` (repo-wide grep), the `max_cores_per_head_batch = 16` probe row
is digit-for-digit identical before and after, and every non-nightly unit-test file under
`tests/ttnn/unit_tests/operations/sdpa/` that reaches this program factory - the two
`sdpa_decode`-named files plus `test_bounded_sliding_kv_cache.py` and `test_mla_decode.py`,
31 collected - is **30 passed, 1 skipped** (`logs/ttnn_sdpa_decode_op_tests.log`).

The fix is also load-bearing rather than precautionary, shown by a stock-main control: with the
`.cpp` change reverted and everything else identical, `full_context_decode_pcc` at position
262143 is **0.977888 — below the bar** (`logs/controls/long_context_stock_control.log`), against
**0.999201** with it, while both `full_context_prefill_tail_pcc` and
`full_context_prefill_tail_scale` are bit-identical in the two builds — the invariance check that
the change touches decode and only decode.

### Capability-contract evidence

| claim | evidence | remaining risk |
|---|---|---|
| Both HF layer kinds implemented and correct | 57 functional tests; both kinds in every parametrised case except the five that are `full_attention`-only by construction (the two block-size tests, the block-size guard and the BFP8 cache test - `linear_attention` has no paged KV cache); `logs/suite_main.log` = `57 passed, 2 skipped` | Only layers 0 and 3 are instantiated. `layer_types` has exactly two distinct values and `decoder_shapes` rejects a third. |
| Advertised context 262144 supported, not reduced | `test_full_advertised_context` prefills 262143 and decodes at 262143 for both kinds, against a real HF reference; all four PCCs >= 0.998031 | The reference is built segmentally (`linear_attention`) or by projection-only cache fill validated `torch.equal` against a real short prefill (`full_attention`); a whole-prompt HF forward is impossible at this length. |
| Paged KV cache correct under a non-trivial page table | Page tables are a shuffled permutation of all `batch * blocks_per_user` blocks; `test_linear_state_and_kv_cache_match_reference` un-pages the device cache and compares it against HF's own cache object (K 0.999989, V 0.999993) | Cache compare is at one length (2049) and batch 1; batched addressing covered indirectly by `test_batched_users[32]` and directly at full context. |
| Page/block geometry is a parameter | `test_alternate_page_block_size` at 32 and 128 | Three block sizes tested. |
| Non-aligned logical lengths work on the public API | 1, 17, 128, 2049, 5000, 8191, 16385, 262143, the 735..768 pad-below-one-tile range, and 24 non-64-divisible batch-32 prompts | — |
| Decode is traceable and correct from replay | `test_traced_decode_pcc` and `test_traced_decode_batched` compare **replay** output against HF (min 0.999434) | Trace captured at batch 1 and 4, not 32. |
| Batch > 1 works | batch 4 and 32, unequal prompts, per-user page tables and current positions | Batch 32 tested at `max_seq_len` 8192, not 262144 — 32 x 1.07 GB of KV cache exceeds the 31 GiB measured. A batch x context product limit, not a context reduction. |
| Real checkpoint weights load and pass | `test_real_weights`, prefill 0.999969 / 0.999964, decode 0.999990 / 0.999988 | One layer per kind, one length. |
| No host fallback in a measured pass | `test_no_runtime_host_fallback`: source scan **and** a live run with `from_torch`/`to_torch`/`as_tensor` stubbed to raise | — |
| Deterministic for repeated inputs | bit-identical prefill and decode output, both kinds | — |
| Watcher clean | `watcher/WATCHER_AUDIT.md`, 9 passed, zero fatal/assert/sanitize lines in 1712 | Watcher subset is 9 tests, not the whole suite. |
| Warmed prefill and traced warmed decode measured | `perf_summary.json` + `tracy/*/*_perf_report.txt`; measured windows provably complete (exact op-count periodicity) | Single chip, batch 1, unoptimised config; this stage claims no perf target. |

### Sequence-length coverage

| class | lengths | test |
|---|---|---|
| sub-tile smoke | 1, 17 | `test_prefill_pcc` |
| below one chunk | 128 | `test_prefill_pcc` |
| exactly at chunk / page / SDPA-chunk boundaries | 2048, 4096 | `test_prefill_pcc` |
| one token past a boundary | 2049 | `test_prefill_pcc`, `test_decode_pcc` |
| long, divisible by none of {32, 64, 256, 2048} | 5000, 8191, 16385 | `test_prefill_pcc`, `test_prefill_pcc_long` |
| pad amount **below one tile** (`ttnn.pad` aliasing regression) | 735, 736, 737, 743, 767, 768 | `test_prefill_decode_pad_below_one_tile` — prefill **and** decode |
| every batch-32 prompt | 64, 161, 258, …, 3071 (`64 + 97*u`; `gcd(97, 64) = 1`, so 31 of the 32 are non-64-divisible) | `test_batched_users` |
| full advertised context | 262143 | `test_full_advertised_context` |

No divisibility requirement is imposed on the public prefill API. Every length in the table is
asserted by a test; none is probe-only.

### Paged KV cache and state

* Page tables are a **shuffled permutation** of all `max_batch * blocks_per_user` blocks, not
  an identity map, so any assumption of contiguous or zero-based slots shows up as a PCC
  failure (`tests/harness.py::make_page_table`).
* Prefill uses `ttnn.experimental.paged_fill_cache` per chunk with a per-chunk page-table
  slice; decode uses `ttnn.experimental.paged_update_cache` with a device `current_pos`
  tensor and `ttnn.transformer.paged_scaled_dot_product_attention_decode`.
* `test_linear_state_and_kv_cache_match_reference` un-pages the device cache and compares it
  against HF's own cache object, so the cache *contents* are checked, not just the output.
* Block size is a constructor parameter, tested at 32, 64 and 128 by
  `test_alternate_page_block_size`, through **prefill and a following decode step** — decode is
  where `paged_update_cache` and the decode SDPA do their own page-table indexing, and the
  decode k chunk is a fixed 512 tokens regardless of block size. That test is `full_attention`
  only — `linear_attention` has no paged cache, so parametrising it there would assert nothing.
* `cache_dtype=ttnn.bfloat8_b` is covered by `test_bfloat8_kv_cache` at the same 0.995 bar as
  everything else (measured 0.999346 prefill / 0.999439 decode), which exercises the
  asymmetric dtype contract: prefill casts K/V to the cache dtype before `paged_fill_cache`,
  while decode must **not** cast the `paged_update_cache` update tensors.
* A per-chunk page table shorter than the padded chunk is an assertion, not a silent trim —
  dropping cache writes for the tail of a prompt would only surface much later as a wrong
  decode.

## Determinism

`test_determinism` asserts **bit-identical** outputs for repeated identical inputs, for
prefill and for decode, for both layer kinds (`prefill_bit_identical` / `decode_bit_identical`
= `true` in `pcc_evidence.json`).

## Runtime fallback audit

`test_no_runtime_host_fallback` does both halves:

1. a source scan of `tt/functional_decoder.py` over the module-level helpers, `__init__` **and**
   everything after the setup section, rejecting `torch`, `from_torch`, `to_torch`, `as_tensor`
   and `.cpu()`. Only the module docstring and the body of `from_state_dict` — the documented
   setup-time boundary — are excluded, and each scanned region is asserted non-empty so a moved
   anchor cannot silently disable the check;
2. a live run of one prefill and one decode pass with `ttnn.from_torch`, `ttnn.to_torch` and
   `ttnn.as_tensor` replaced by raising stubs.

Both layer kinds pass. Weight transposition, dtype selection, cache/state allocation and the
`eye`/zero-prefix constants are all built in `from_state_dict`.

## Watcher

`TT_METAL_WATCHER=10` over both layer kinds' paged prefill, paged decode, traced decode at
batch 1 and 4, the BFP8 cache path and both alternate page block sizes: **9 passed**, watcher
log clean. Full audit, with the
exact command and the grep that finds no fatal/assert/sanitize/corruption lines:
[`watcher/WATCHER_AUDIT.md`](watcher/WATCHER_AUDIT.md). Raw log:
`watcher/generated/watcher/watcher.log`. Watcher and the device profiler were run separately.

## Performance

Warmed measurements on one Blackhole chip, batch 1, from **Tracy device-profiler** runs with
the measured window delimited by signposts. All host work — input construction, upload, trace
capture — happens before the start signpost, and the profiler's marker buffers are drained
(`ttnn.ReadDeviceProfiler`) between warm-up and the measured window. Markers *are* dropped
during setup, before that drain and before the start signpost; what matters is that the
measured windows are complete, and they provably are — the op list repeats with an exact
period (92 and 50 ops), so all replays were captured whole. Decode is **traced**: capture
once, then replay `execute_trace` inside the window.

| layer kind | phase | ops in one pass | device kernel time | host wall |
|---|---|---|---|---|
| `linear_attention` | prefill, 2048 tokens | 801 | **151.24 ms** | 162.26 ms |
| `linear_attention` | traced decode, 1 token | 92 | **3.034 ms** | 3.409 ms |
| `full_attention` | prefill, 2048 tokens | 44 | **18.63 ms** | 20.15 ms |
| `full_attention` | traced decode, 1 token | 50 | **2.271 ms** | 2.333 ms |

Decode numbers are the mean of 8 trace replays inside the window. Device time is the
`Device Time` column of the `tt-perf-report --csv` output, **in microseconds** (the raw Tracy
ops CSV exposes the same quantity as `DEVICE KERNEL DURATION [ns]`).

`tt-perf-report`'s `Cores` column shows 64 for `SdpaDecodeDeviceOperation`: that is the op's
program *grid*, not the number of cores doing work. `num_active_cores = num_cores_per_head *
num_kv_heads * B = 1 * 4 * 1 = 4`; the other 60 are idle. That is the "one core per head" cost
referred to below, and it is why the column and the prose disagree.

Artifacts, per layer kind, under [`tracy/`](tracy/):

* `<phase>_ops.csv.gz` — the post-processed Tracy ops CSV, copied verbatim and gzipped
  (the raw CSVs are 0.9–3.2 MB, over this repo's 500 KB commit limit; they are left
  uncompressed next to it in a live run tree and are gitignored)
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
  models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder_perf.py::test_perf_decode_traced[linear_attention] -s -q
tt-perf-report <ops.csv> --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-summary --no-advice
```

This checkout is built in-tree with `ENABLE_TRACY=ON`, so the profiling runs use the same
environment as the correctness runs — there is no separate profiler tree.

### Observations for the optimization stage (not acted on here)

This stage is about correctness, so the implementation is deliberately unoptimised — BF16,
tile layout, DRAM interleaved everywhere, HiFi4 on every matmul. The profile shows where the
work is:

* `linear_attention` prefill is **op-count bound**: 801 ops for 2048 tokens versus 44 for
  `full_attention`, because the gated delta rule is a Python loop over the 32 sub-chunks of
  the recurrence plus a recursive block inverse. `ttnn.transformer.gated_delta_attn` and
  `ttnn.transformer.chunk_gated_delta_rule` exist in this tree and collapse most of that into
  one kernel call; `models/demos/blackhole/qwen36/tt/` is a working reference for how to feed
  them. That is the single largest win available and is squarely optimization-stage work.
* The largest single cost inside that loop is `_unit_tri_inverse`'s batched `32 x 32 x 32`
  matmuls, which run on one core at HiFi4/FP32. `TRI_INV_BASE` is 16, chosen by measurement
  (`probes/probe_tri_inv_base.py`).
* `full_attention` decode currently runs its SDPA on **4 cores** (one per KV head) because of
  the upstream tree-reduction defect above. At the 2048-token position measured here that
  costs nothing visible, but it will dominate at long context. Fixing the upstream defect, or
  extending the fp32 promotion to the cross-core packet formats, unlocks 16x more cores.
* Traced decode is DRAM bound as expected, with the MLP matmul near its own roofline.

## Known limitations

* **`prepare_decode_state()` rewrites every batch slot** from that user's post-prefill
  snapshot, so it cannot insert one newly-prefilled user into a live decode batch without
  rewinding the others. Fine for the prefill-all-then-decode pattern this stage tests; a
  per-slot fold is a strided `ttnn.copy` into one slice of the user-major buffer and belongs
  with the serving stage's slot-eviction policy.
* **Trace capture is exercised at batch 1 and batch 4**, not at batch 32.
* **Prefill is single-user per call** by construction; batched prefill is a loop over
  `user_id`.
* **Batch 32 is tested at `max_seq_len` 8192**, not 262144: a batch-32 `full_attention` layer
  at full context would need 32 x 1.07 GB of KV cache, above the 31 GiB measured. That is a
  batch x context product limit for later stages to schedule, not a reduction of the
  advertised context, which is tested in full at batch 1.
* **The decode SDPA runs on one core per head.** See the performance section; it is a
  correctness-first choice forced by an upstream defect, recorded with its reproducer.
* The implementation is deliberately unoptimised. See the observations above.
