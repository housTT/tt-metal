# Qwen/Qwen3.6-27B — functional decoder work log

Model dir: `models/autoports/qwen_qwen3_6_27b` (slug of the HF model id).
Branch: `agentic-research/hous/qwen3.6-27b-v2`.

## 0. Provenance — what this stage started from

This is the **second** functional-decoder pass on this model. An earlier pass on branch
`agentic-research/hous/qwen3.6-27b` (merge base `107623bb9dd`) produced a complete
implementation but could **not** close the stage: `full_attention` decode at position 262143
scored PCC 0.5503 against a 0.995 bar, and that pass concluded there was no Python-level
workaround. This pass:

* starts from that branch's functional-decoder sources (`tt/functional_decoder.py`,
  `tt/model_config.py`, `reference/`, `scripts/`, `tests/`, `doc/functional_decoder/probes/`),
  restored with `git checkout agentic-research/hous/qwen3.6-27b -- <paths>`;
* re-verifies every one of them against **current main** (`837e8da3e9b`), which is ~1500
  commits newer than the branch they were written against, and fixes what that broke (§2);
* **closes the long-position decode gap** (§3), which is the reason this pass exists;
* regenerates every number, log and artifact on this branch. Nothing in
  `doc/functional_decoder/` is carried over from the earlier pass except the probe scripts and
  `weight_stats.json` (a pure function of the checkpoint).

The earlier pass's own narrative for the three bugs it found and fixed —
the `ttnn.pad` aliasing hazard, the zero-padded prefill tokens corrupting the gated-delta-net
state, and the Neumann doubling product being unusable at the real weights — is not repeated
here; those fixes are in the code and are covered by the tests listed in §5. What is new on
this branch is §2 and §3.

## 1. Environment

The stage runs entirely inside this checkout. `doc/functional_decoder/ttenv.sh`:

```bash
export REPO=/home/ttuser/dev/qwen/tt-metal
export TT_METAL_HOME=$REPO
export TT_VISIBLE_DEVICES=2
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
source $REPO/python_env/bin/activate
export PYTHONPATH=$REPO
# ... then asserts ttnn resolves inside the checkout
```

The assertion at the end of that file is not decoration. This host has other tt-metal trees
and a `ttnn-custom.pth`-style editable install that binds `ttnn` elsewhere; `PYTHONPATH` alone
does not undo that, and every number below would then describe code that is not under test.
Sourcing the file prints:

```
ttnn OK: /home/ttuser/dev/qwen/tt-metal/ttnn/ttnn/__init__.py
python : /home/ttuser/dev/qwen/tt-metal/python_env/bin/python
```

This also removes the earlier pass's constraint that device jobs run from outside a tt-metal
checkout: there, `TT_METAL_HOME` pointed at a *different* built tree, so kernel sources
resolved from the wrong tree. Here `TT_METAL_HOME` is the repo, the loaded `_ttnn.so` was
built from it (`build_Release`, `ENABLE_TRACY=ON`), and the working directory is the repo root.
There is no separate profiler tree either — the in-tree build already has Tracy.

### Hardware

`/sys/class/tenstorrent/*/tt_card_type` and `tt_serial` report **two p300c boards of two
Blackhole chips each**:

| board serial | chips | state |
|---|---|---|
| `000004613192404C` | `/dev/tenstorrent/0`, `/dev/tenstorrent/1` | chip 0's ARC is wedged (`tt-smi -ls` fails with `Read 0xffffffff over PCIe ID 0`); chip 1 healthy |
| `0000046131924022` | `/dev/tenstorrent/2`, `/dev/tenstorrent/3` | both healthy |

Chip 0 has been wedged since before this pass and needs an operator-side host power cycle; the
earlier pass recorded two bounded `tt-smi -r` resets, a driver reload and a PCIe FLR all
failing to restart its ARC. `tt-smi -ls` still cannot enumerate because it probes chip 0.
`TT_VISIBLE_DEVICES=2` makes UMD topology discovery skip it, and this stage needs only a 1x1
mesh. Mesh smoke on that chip, `device_recovery/v2_mesh_smoke.log`:

```
opened in 2.3s MeshShape([1, 1])
matmul ok Shape([1, 1, 512, 512])
MESH_SMOKE_OK
```

Exposing a single chip of a 2-chip p300 makes metal classify the cluster as `CUSTOM` and
demand an explicit fabric mesh graph descriptor; the stock single-Blackhole-chip descriptor is
the right shape for a 1x1 mesh. Its `p150_` file name names the board the descriptor ships
for, not the board in this host.

No hardware recovery was needed during this pass.

## 2. What current main broke, and the fixes

### 2.1 The SDPA prefill program config no longer fits L1

First run of the restored suite: every `full_attention` prefill at 2048 tokens or more died
with

```
TT_THROW: Statically allocated circular buffers on core range [0-0 - 7-7] grow to 1676672 B
which is beyond max L1 size of 1572864 B
```

Cause: current main's `sdpa_program_factory.cpp` promotes the QK intermediate and the row-sum
circular buffers to fp32 under `fp32_dest_acc_en` (`fp32_dest_intermediate_dataformat`), which
the older tree did not. The layer's `q_chunk_size = 256, k_chunk_size = 256` was sized against
the older, all-bfloat16 buffers.

Measured the whole feasible space at `head_dim` 256 rather than guessing
(`probes/probe_sdpa_synthetic.py`, `logs/sdpa_fit_sweep_v2.log`):

| q / k | fp32 dest acc | result |
|---|---|---|
| 256 / 256 | on | 1676672 B — rejected |
| 128 / 256 | on | fits |
| 128 / 128 | on | fits |
| 64 / 512 | on | fits |
| 128 / 512 | on | 1815936 B — rejected |
| 512 / 128 | on | 2184576 B — rejected |
| 64 / 1024, 32 / 1024 | on | 2672000 / 2444672 B — rejected |

So 512 is the largest usable k chunk and it needs `q <= 64`. `_SDPA_Q_FOR_K` is now
`{256: 128, 512: 64}`; `SDPA_CHUNK` and `SDPA_MAX_K_CHUNKS` are unchanged.

### 2.2 The long-context SDPA merge error is worse on current main

The earlier pass established that `chunked_scaled_dot_product_attention` loses a little of its
softmax denominator on every k-chunk merge, one-sidedly, so the normalised output comes out
uniformly too large by a factor that tracks the *merge count*. That is still true, and the
per-merge loss is larger on current main. `logs/sdpa_long_sweep_v2.log`, device/float32-golden
scale `alpha` on bit-identical bfloat16 inputs:

| q / k | fp32 dest acc | 131072 keys | 262144 keys |
|---|---|---|---|
| 128 / 256 | on | 1.090 (512 chunks) | 1.204 (1024 chunks) |
| 64 / 512 | on | 1.045 (256 chunks) | **1.091 (512 chunks)** |
| 256 / 256 | off | 1.160 | 1.692 |
| 64 / 512 | off | 0.972 | 1.176 |

fp32 destination accumulation is kept (it is uniformly better than the streaming kernel that
clearing it selects), and `SDPA_MAX_K_CHUNKS = 512` still caps the merge count, which is also
the floor: 512 is both the largest k chunk that fits and the fewest merges a 262144-key call
can do. The layer-level consequence is in §5 — `full_attention` full-context prefill tail PCC
is 0.99803, against 0.99882 on the older tree. Both clear the bar; the difference is this
regression.

The synthetic probe is the **worst case** for this defect: random Q/K/V give a maximally flat
softmax, so every chunk contributes equally. Real attention is peakier, which is why a scale
of 1.09 on the op costs only ~8e-4 of layer PCC.

### 2.3 The repo-root 300 s pytest timeout

`pytest.ini` sets `timeout = 300`. The earlier pass ran from outside any tt-metal checkout, so
it never saw that setting. Three tests are legitimately longer than 300 s because their **HF
reference** is the slow part — the 262143-token segmented reference, 32 users of eager
attention, and the 16385-token single-shot reference — and they now carry
`@pytest.mark.timeout(0)` individually. The global setting still guards everything else. This
is recorded as a deliberate per-test exemption, not a weakened assert: no assertion changed.

### 2.4 De-generalised the test harness

The restored `tests/harness.py` carried hooks for decoder implementations that do not exist on
this branch — a `DECODER_CLS`/`DECODER_KWARGS` indirection, `expand_rot_mats`,
`rope_permutation` and `value_head_permutation`, all no-ops for `FunctionalDecoder`. They were
removed, along with the `decoder_cls`/`decoder_kwargs` parameters of `build_layer`. Fake
generality for one supported shape is a code-quality defect in its own right, and here it also
misdescribed the layer under test.

## 3. The long-position decode defect, and its fix

This is the gate the earlier pass could not close.

### 3.1 It reproduces unchanged on current main

`probes/probe_sdpa_decode_synthetic.py` is model-free: synthetic Q/K/V at the layer's shapes
(24 q heads / 4 kv heads / `head_dim` 256 / page 64), a shuffled page table, and a float32
torch golden on bit-identical bfloat16 inputs. It reports `alpha`, the fitted device/golden
scale, because a softmax-denominator defect is a pure scale and PCC cannot see it.

With the **default** (auto) program config — `logs/sdpa_decode_default_v2.log`:

| position | 1023 | 4095 | 16383 | 65535 | 131071 | 262143 |
|---|---|---|---|---|---|---|
| alpha | 0.996 | 0.995 | 1.211 | 4.765 | 12.659 | **37.659** |

Same 37.7x at 262143 the earlier pass recorded. Current main's decode kernel gained a tree
reduction over cores, but not fp32 statistics.

### 3.2 Two independent defects, separated

Sweeping explicit program configs (`logs/sdpa_decode_cfg_sweep_v2.log`) separates them.

**Defect A — the cross-core tree reduction is wrong for most positions.** With
`k_chunk = 128, max_cores_per_head_batch = 16` the scale is 0.997–1.006 at positions 4095,
12287, 16383, 65535, 131071 and 262143 — and 3705x at position 1023 and **NaN** at 261887.
Repeating over `max_cores_per_head_batch` 1, 4, 8, 16 and k chunk 64, 128, 256 fits one rule
exactly, with no exceptions in 40 measurements:

> the result is correct only when `num_k_chunks == 1` or `num_k_chunks % (2 * cores_per_head) == 0`

and `num_k_chunks = ceil((cur_pos + 1) / k_chunk)` is a **runtime** quantity while the program
config is compile-time. Violations are catastrophic, not small (3705x, 780440x at
`k_chunk = 256` position 12287, NaN at 261887). There is therefore no core count above 1 that
is correct at every position. `max_cores_per_head_batch = 1` removes the tree reduction
entirely (`num_tree_reduction_rounds` is 0 at one core per head) and showed **no** catastrophic
position in the sweep.

**Defect B — the flash statistics are bfloat16.** With the tree reduction out of the way the
remaining error is the same one-sided denominator loss as prefill, and it is large:
`k_chunk = 128, cores = 1` gives alpha 5.223 at position 262143 (2048 sequential merges).

### 3.3 The fix

`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
promotes the **core-local** flash accumulators — running max, running softmax denominator,
`exp_max_diff`, and the output accumulator (`c_27`–`c_31`, `c_21`, `c_22`, `c_25`, `c_26`,
`c_23`) — to `Float32` when the caller asked for `fp32_dest_acc_en` **and** there is one core
per head:

```cpp
const bool fp32_local_accumulators = fp32_dest_acc_en && num_cores_per_head == 1 &&
                                     program_config.has_value() &&
                                     program_config->max_cores_per_head_batch == 1;
const tt::DataFormat acc_df       = fp32_local_accumulators ? Float32 : im_df;
const tt::DataFormat acc_stats_df = fp32_local_accumulators ? Float32 : stats_df;
```

The third clause is not redundant, and a stage review caught its absence. `num_cores_per_head`
is *derived*:

```cpp
num_cores_per_batch_uncapped = min(num_cores_available, max_cores_per_head * B * num_kv_heads) / B;
num_cores_per_head           = max(1u, num_cores_per_batch_uncapped / num_kv_heads);
```

so it collapses to 1 whenever `num_cores_available / (B * num_kv_heads) < 2` — on a 64-core
grid that is ordinary batched decode (B=16 with 4 KV heads, B=32 with 4 or 8, B=8 with 8), not
an exotic corner, and `models/tt_transformers` reaches it with `fp32_dest_acc_en=True` from its
HiFi2/HiFi4 defaults. Keying only off the derived value would have changed the CB footprint and
the numerics of callers that never asked for it. Requiring the caller to have *explicitly* set
`max_cores_per_head_batch == 1` makes the promotion strictly opt-in.

Three things are deliberately *not* promoted:

* `c_24` (the QK intermediate). Promoting it as well was tried and measurably breaks the op —
  a roughly constant 3.8–5.5x error at **every** position, including position 1023 where only
  8 merges happen, so it cannot be an accumulation effect. Backed out and left at `im_df`.
* `c_16`/`c_17`/`c_18`/`c_19`. Those are the packet formats shared with the reader and writer
  kernels and with sibling cores; widening them would change the wire format and the reduction
  CB footprint. The `num_cores_per_head == 1` guard is what makes that safe: at one core per
  head none of them is used.
* Anything at all unless the caller explicitly asked for one core per head with
  `fp32_dest_acc_en` set.

Blast radius, measured rather than asserted:

| check | result |
|---|---|
| repo-wide `grep -rn max_cores_per_head_batch models/ tests/ ttnn/` | no caller outside this autoport passes `1`; the values in the repo are the struct default `16`, plus explicit `16` (`models/demos/gemma4`) and `4` (`tests/.../test_mla_decode.py`, `test_sdpa_decode_cache.py`, `test_mla_decode_stress.py`) |
| `k_chunk = 128, max_cores_per_head_batch = 16` probe row, before vs after | digit-for-digit identical (0.99671, 0.99893, 0.99975, 1.00195, 1.00178, 1.00613, and the same NaN at 261887) |
| `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py` + `test_paged_sdpa_decode_flexible_geometry.py` | **21 passed, 1 skipped** (`logs/ttnn_sdpa_decode_op_tests.log`) |

The first row is what makes the gate safe: a caller has to opt in by name, and nothing in the
repo does.

Rebuild: `ninja -C build_Release install` (`logs/rebuild_sdpa_decode.log`).

### 3.4 Measured result

`logs/sdpa_decode_fp32acc_v2.log`, one core per head, after the fix:

| k chunk | 1023 | 4095 | 12287 | 16383 | 65535 | 131071 | 261887 | 262143 |
|---|---|---|---|---|---|---|---|---|
| 128 | 0.994 | 0.996 | 0.997 | 0.998 | 0.999 | 0.954 | 1.284 | 1.336 |
| 256 | 0.995 | 0.996 | 0.998 | 0.998 | 0.999 | 1.002 | 0.993 | 0.989 |
| **512** | **0.995** | **0.997** | **0.997** | **0.998** | **0.999** | **1.000** | **1.017** | **1.006** |

The stock-main control for exactly this configuration — the row a stage review pointed out was
missing — is `logs/sdpa_decode_stock_baseline.log`, measured on a rebuild with the `.cpp`
change reverted:

| k chunk / cores | 1023 | 65535 | 131071 | 261887 | 262143 |
|---|---|---|---|---|---|
| 512 / 1, **stock main** | 0.995 | 0.993 | 0.979 | **1.310** | **1.290** |
| 512 / 1, with the fix | 0.995 | 0.999 | 1.000 | **1.017** | **1.006** |

and at the layer level, same code, same test, only the `.cpp` differing
(`logs/long_context_stock_control.log` vs `logs/long_context.log`):

| `test_full_advertised_context[full_attention]` | stock main | with the fix |
|---|---|---|
| `full_context_prefill_tail_pcc` @ 262143 | 0.9980307630901388 | 0.9980307630901388 (bit-identical — prefill does not touch this factory) |
| `full_context_decode_pcc` @ 262143 | **0.977888 — FAILS the 0.995 bar** | **0.999201 — passes** |

So the C++ change is load-bearing for the configuration the layer actually ships, not just for
the `k_chunk = 128` configuration that exposed the defect.

`k_chunk = 512` at one core per head is scale-correct at every position tested, including the
two that were catastrophic before, so that is what the layer uses
(`SDPA_DECODE_K_CHUNK`, `SDPA_DECODE_CORES_PER_HEAD`). 512 is also the largest k chunk that
fits L1 here, and the fewest merges available.

The cost is real and is not hidden: one core per head means **4 active cores instead of 64**
for the decode attention op. This is a correctness-first choice for a functional decoder, and
it is the item to revisit in the optimization stage — either by fixing defect A upstream, or
by extending the fp32 promotion to the cross-core packet formats so a larger core count is
usable.

### 3.5 Layer-level effect

`logs/long_context.log`, prompt 262143 and decode at
position 262143 against the HF reference:

| | earlier pass | this pass |
|---|---|---|
| `full_attention` decode @ 262143 | 0.550293 | **0.999201** |
| `linear_attention` decode @ 262143 | 0.999946 | 0.999955 |
| `full_attention` prefill tail @ 262143 | 0.998817 | 0.998031 |
| `linear_attention` prefill tail @ 262143 | 0.999942 | 0.999947 |

### 3.6 Worth filing upstream

Both defects have model-free reproducers in `probes/`, which is what an upstream issue needs:

1. `probe_sdpa_decode_synthetic.py` — defect A, the tree reduction. The
   `num_k_chunks % (2 * cores_per_head)` rule, 3705x at position 1023, NaN at 261887. This is
   the one that still has no general fix.
2. `probe_sdpa_synthetic.py` — the prefill half of defect B, which is *not* fixed by this
   change (it lives in `sdpa_program_factory.cpp` and its cross-core buffers) and is what caps
   `SDPA_MAX_K_CHUNKS`.

## 4. Capability contract evidence

The claim/evidence/risk table lives in [`README.md`](README.md) ("Capability-contract
evidence"), so it sits next to the numbers it refers to rather than being duplicated here.
`../context_contract.json` is the machine-readable form and passes the runner-side guardrail:

```
$ python .agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_6_27b \
      --hf-model Qwen/Qwen3.6-27B --stage functional_decoder --require-contract --strict-caps
Context contract OK for models/autoports/qwen_qwen3_6_27b: target=262144, supported=262144 (full HF context).
```

DRAM is not the constraint: `probes/probe_capacity.py` (`logs/capacity_probe.log`) allocates
**31 GiB** before the allocator refuses, and the worst-case single layer at batch 1 and the
full 262144 context is **1.82 GB** (744 MB of `full_attention` weights + 1.07 GB of paged KV
cache) — about 5.5 % of the measured DRAM, ~18x headroom. No capability reduction was made or
needed.

## 5. Final state

| gate | result | artifact |
|---|---|---|
| functional suite | **55 passed, 2 skipped** in 7:33 (the 2 skips are the `--long-context` cases, run separately) | `logs/suite_main.log` |
| full advertised context, both layer kinds | **2 passed** in 6:24 | `logs/long_context.log` |
| watcher (`TT_METAL_WATCHER=10`) | **9 passed**, log clean, 0 fatal/assert/sanitize lines in 1720 | `logs/watcher_run.log`, `watcher/WATCHER_AUDIT.md` |
| tt-metal `sdpa_decode` op suites (blast-radius control for the `.cpp` change) | **21 passed, 1 skipped** | `logs/ttnn_sdpa_decode_op_tests.log` |
| profiling | 4 Tracy runs, marker-drop-free windows (exact op-count periodicity) | `perf_summary.json`, `tracy/*/*_perf_report.txt` |
| recorded measurements | 264 records, 260 numeric, **minimum 0.998031**, **0 below the 0.995 bar** | `pcc_evidence.json` |
| context contract | OK, target = supported = 262144 | `../context_contract.json` |
| runtime fallback audit | source scan + live stubbed-`ttnn` run, both layer kinds | `test_no_runtime_host_fallback` |
| determinism | bit-identical prefill and decode, both layer kinds | `test_determinism` |

Performance, warmed, batch 1, one Blackhole chip (`perf_summary.json`):

| layer kind | phase | ops/pass | device kernel time | host wall |
|---|---|---|---|---|
| `linear_attention` | prefill 2048 | 801 | 151.07 ms | 161.98 ms |
| `linear_attention` | traced decode | 92 | 3.040 ms | 3.413 ms |
| `full_attention` | prefill 2048 | 44 | 18.63 ms | 20.05 ms |
| `full_attention` | traced decode | 50 | 2.271 ms | 2.329 ms |

### Repo changes owned by this stage

Two areas, both deliberate:

* `models/autoports/qwen_qwen3_6_27b/**` — the stage's own tree.
* `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp` —
  the fp32 local flash accumulators of §3.3. This is a tt-metal source change, made here
  because the alternative was to ship a decoder that is silently wrong at long decode
  positions, which the stage contract does not allow. It is gated so that no existing caller's
  program changes, and that gating is measured, not asserted.

## 6. First stage review — findings and what changed

An independent `$stage-review` subagent returned `more-work-needed` on the state described
above. Both P-findings were real defects in this stage's own work and both were treated as
work, not argued down.

### 6.1 P1 — the "zero blast radius" claim for the C++ change was false

The reviewer re-derived the core allocation and showed that `num_cores_per_head` collapses to
1 whenever `num_cores_available / (B * num_kv_heads) < 2`, i.e. for ordinary batched decode
(B=16 or 32 with 4 KV heads on a 64-core grid), and that `models/tt_transformers` reaches that
with `fp32_dest_acc_en=True` from its HiFi2/HiFi4 defaults. So the gate as written *would* have
changed the CB footprint and the numerics of callers that never asked for it — the exact
failure class this stage had already hit on the prefill side (§2.1). The "verified rather than
asserted" evidence was also circular: the `max_cores_per_head_batch = 16` row that was compared
before and after does not reach the modified branch at this layer's shape.

Fixed by making the promotion strictly opt-in — the caller must set
`max_cores_per_head_batch == 1` by name — and by replacing the claim in `README.md`,
`work_log.md` §3.3 and `doc/context_contract.json` with three measured checks (repo-wide grep,
the unchanged 16-core row, and tt-metal's own `sdpa_decode` op suites at 21 passed / 1 skipped).
See §3.3.

### 6.2 P2 — the C++ change's necessity was never measured at the shipped config

The pre-fix sweep had no `k_chunk = 512, cores = 1` row: the "defect B is large" conclusion came
from the `128 / 1` row (2048 merges), while the layer ships 512 merges. Fixed by rebuilding with
the `.cpp` change reverted and measuring both the probe and the layer test at exactly the
shipped configuration. Stock main: probe scale 1.290 at position 262143, layer
`full_context_decode_pcc` **0.977888 — below the bar**. With the fix: 1.006 and **0.999201**,
with `full_context_prefill_tail_pcc` bit-identical in both. The change is load-bearing. See
§3.4.

### 6.3 Other review findings, all fixed

* `test_alternate_page_block_size` asserted prefill only, while the README claimed block size
  was "tested at 32, 64 and 128". It now runs a following decode step as well, which is where
  `paged_update_cache` and the decode SDPA do their own page-table indexing
  (`alt_block_size_decode_pcc`, min 0.999490).
* `test_bfloat8_kv_cache` carried its own relaxed `BFP8_PCC_BAR = 0.99` while the README said
  there was "no exception, no waiver". The measured values (0.999346 / 0.999439) clear the
  stage bar, so the constant is gone and the test asserts `H.PCC_BAR` like every other test.
* `WATCHER_AUDIT.md` mis-stated the dump count and line census. Recounted from the regenerated
  log, with the census command quoted so it can be re-run.
* `perf_summary.json`'s `tracy_run_log` paths were malformed and did not resolve. Fixed, and a
  `core_count_note` now explains why `tt-perf-report` shows `Cores 64` for an op that has 4
  active cores.
* `tests/conftest.py` still carried the earlier pass's "must run device jobs from outside a
  tt-metal checkout" rationale, which §1 retired. Corrected to the real reason the file exists.
* `scripts/collect_evidence.py` de-duplicated with "later logs win", so the reported minimum
  depended on argument order. It now keeps the *worst* numeric value per measurement, which
  makes the "0 records below the bar" claim order-independent.
* Leftover later-stage artifact trees (`doc/fused_decoder/`, `doc/optimized_decoder/`) and
  `__pycache__` for deleted sources were removed from the model directory; this goal owns the
  functional decoder only.

Everything above was re-run after the fixes: the suite, the long-context pair, the watcher run
and all four Tracy runs in §5 are from the final build, not from the reviewed state.

### 6.4 Not done, and why

The reviewer noted that no upstream issue was filed for the cross-core tree-reduction defect.
Filing one is an outward-facing action, and this stage runs unattended with an explicit
never-push constraint, so it is handed off instead: the model-free reproducer
(`probes/probe_sdpa_decode_synthetic.py`), the exact failing rule
(`num_k_chunks % (2 * cores_per_head) != 0` and `num_k_chunks < cores_per_head`), and the
measured signatures (3705x at position 1023, NaN at 261887, both with `k_chunk = 128,
max_cores_per_head_batch = 16`) are all recorded here and in `doc/context_contract.json`.
