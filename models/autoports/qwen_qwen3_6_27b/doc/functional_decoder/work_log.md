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
The sweep asks for eight configurations and gets seven: `k 128` at cores 1, 4, 8 and 16, plus
`k 64` and `k 256` at 16 cores, plus an `exp_approx_mode = 1` repeat of `k 128 / 16 cores` (it
changes nothing, which is itself the finding); `k 512` at 16 cores throws on L1 and produces no
rows. Seven rows at eight positions each is 56 measurements. `k 64` and `k 256` were swept at
16 cores only, which is enough to show the rule is not a property of one k chunk. One rule fits
all 56 with no exception:

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

One consequence to carry forward: for a caller that *does* opt in, the promotion roughly doubles
the L1 footprint of `c_21`-`c_23` and `c_25`-`c_31`. At this layer's shape that is ~70 KB and it
fits with room; a future opt-in caller with a larger `head_dim` or `vDHt` could hit an L1
overflow it would not have hit before, and would see it as a build-time `TT_THROW` rather than
as a wrong answer.

Blast radius, measured rather than asserted:

| check | result |
|---|---|
| repo-wide `grep -rn max_cores_per_head_batch models/ tests/ ttnn/` | no source literal `1` outside this autoport; the values in the repo are the struct default `16`, plus explicit `16` (`models/demos/gemma4`) and `4` (`tests/.../test_mla_decode.py`, `test_sdpa_decode_cache.py`, `test_mla_decode_stress.py`). Four sweep-framework loaders parse the value out of a config *string*, so a data-driven sweep config could in principle select 1; that is an explicit opt-in by the same definition, and the promotion only raises precision. |
| `k_chunk = 128, max_cores_per_head_batch = 16` probe row, before vs after | digit-for-digit identical (0.99671, 0.99893, 0.99975, 1.00195, 1.00178, 1.00613, and the same NaN at 261887) |
| every non-nightly unit-test file under `tests/ttnn/unit_tests/operations/sdpa/` that reaches `SdpaDecodeDeviceOperation::create_descriptor`: the two `sdpa_decode`-named files, plus `test_bounded_sliding_kv_cache.py` (calls `paged_scaled_dot_product_attention_decode`) and `test_mla_decode.py` (calls `flash_multi_latent_attention_decode`, same factory) - 31 collected | **30 passed, 1 skipped** (`logs/ttnn_sdpa_decode_op_tests.log`); the skip needs a (10,11) grid. None of them can trigger the gate: `test_bounded_sliding_kv_cache.py` passes no `program_config`, `test_mla_decode.py` passes `max_cores_per_head_batch=4`. The larger `tests/ttnn/nightly/.../test_sdpa_decode.py` was *not* run; its cases also omit `program_config`. |

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
missing — is `logs/controls/sdpa_decode_stock_baseline.log`, measured on a rebuild with the `.cpp`
change reverted:

| k chunk / cores | 1023 | 65535 | 131071 | 261887 | 262143 |
|---|---|---|---|---|---|
| 512 / 1, **stock main** | 0.995 | 0.993 | 0.979 | **1.310** | **1.290** |
| 512 / 1, with the fix | 0.995 | 0.999 | 1.000 | **1.017** | **1.006** |

and at the layer level, same code, same test revision, only the `.cpp` differing
(`logs/controls/long_context_stock_control.log` vs `logs/long_context.log`):

| `test_full_advertised_context[full_attention]` | stock main | with the fix |
|---|---|---|
| `full_context_prefill_tail_pcc` @ 262143 | 0.9980307630901388 | 0.9980307630901388 |
| `full_context_prefill_tail_scale` @ 262143 | 0.9974432795186233 | 0.9974432795186233 |
| `full_context_decode_pcc` @ 262143 | **0.977888 — FAILS the 0.995 bar** | **0.999201 — passes** |
| `full_context_decode_scale` @ 262143 | not reached (see below) | 0.9949287878196464 |

Both prefill rows are **bit-identical** across the two builds, which is the invariance check
that the change touches decode and only decode. The control's decode row is where it fails, and
it fails on PCC, so the run aborts before the decode *scale* assertion is evaluated — the
layer-level stock decode scale is therefore not a recorded number. The 1.29 figure quoted for
the stock kernel is the **op-level** probe `alpha` from
`logs/controls/sdpa_decode_stock_baseline.log`, not a layer-level scale.

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

The earlier pass's numbers are extracted from that branch's own committed `pcc_evidence.json`
into [`earlier_pass_reference.json`](earlier_pass_reference.json), with the commit and the
extraction command, so this comparison quotes an artifact rather than a memory.

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
| functional suite | **57 passed, 2 skipped** in 07:40 (the 2 skips are the `--long-context` cases, run separately) | `logs/suite_main.log` |
| full advertised context, both layer kinds | **2 passed** in 06:25 | `logs/long_context.log` |
| watcher (`TT_METAL_WATCHER=10`) | **9 passed**, log clean, 0 fatal/assert/sanitize lines in 1712 | `logs/watcher_run.log`, `watcher/WATCHER_AUDIT.md` |
| tt-metal op suites reaching the modified factory (blast-radius control for the `.cpp` change) | **30 passed, 1 skipped** | `logs/ttnn_sdpa_decode_op_tests.log` |
| profiling | 4 Tracy runs, marker-drop-free windows (exact op-count periodicity) | `perf_summary.json`, `tracy/*/*_perf_report.txt` |
| recorded measurements | 268 records (260 PCC, 4 scale, 4 booleans), **minimum PCC 0.998031**, **0 below the 0.995 bar**; scale ratios 0.99493-0.99853 | `pcc_evidence.json` |
| context contract | OK, target = supported = 262144 | `../context_contract.json` |
| runtime fallback audit | source scan + live stubbed-`ttnn` run, both layer kinds | `test_no_runtime_host_fallback` |
| determinism | bit-identical prefill and decode, both layer kinds | `test_determinism` |

Performance, warmed, batch 1, one Blackhole chip (`perf_summary.json`):

| layer kind | phase | ops/pass | device kernel time | host wall |
|---|---|---|---|---|
| `linear_attention` | prefill 2048 | 801 | 151.24 ms | 162.26 ms |
| `linear_attention` | traced decode | 92 | 3.034 ms | 3.409 ms |
| `full_attention` | prefill 2048 | 44 | 18.63 ms | 20.15 ms |
| `full_attention` | traced decode | 50 | 2.271 ms | 2.333 ms |

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
the unchanged 16-core row, and the tt-metal op suites that reach the modified factory - widened
in round 6 to four files, 30 passed / 1 skipped).
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
* Leftover later-stage artifact trees from the earlier pass - a fused-decoder and an
  optimized-decoder doc directory - plus `__pycache__` for sources that no longer exist were
  deleted from the model directory; this goal owns the functional decoder only, so those paths
  are deliberately absent now.

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

## 7. Repo checkpoint

```
repo   /home/ttuser/dev/qwen/tt-metal
branch agentic-research/hous/qwen3.6-27b-v2
base   837e8da3e9b
first  9d18c856aaa9b203804d6e5dd5e726fcec65b772   the stage itself (92 files)
```

The authoritative list is `git rev-list 837e8da3e9b..HEAD`; enumerating it here cannot work,
because the commit that records a SHA never contains its own, and every later
documentation-only commit reopens the same gap. Two earlier revisions of this block tried and
lagged by one and then by two. What is checkable, and is what matters, is that **every** commit
in that range touches only `models/autoports/qwen_qwen3_6_27b/**` plus the one
`ttnn/.../sdpa_decode_program_factory.cpp`:

```bash
for c in $(git rev-list 837e8da3e9b..HEAD); do
  git show --name-only --format= "$c" | grep -v '^models/autoports/qwen_qwen3_6_27b\|sdpa_decode_program_factory'
done   # prints nothing
```

Committed with an explicit pathspec so the pre-existing dirty `.agents/` and `scripts/` files —
which this stage did not touch and which the runner staged before it started — stayed out of the
checkpoint. `git show --name-only` on that commit contains only
`models/autoports/qwen_qwen3_6_27b/**` and the one `sdpa_decode_program_factory.cpp`. Not pushed.

Two artifact-shape notes for anyone reading the tree:

* the repo's pre-commit hooks reformatted the Python and the C++ (black/isort/autoflake,
  clang-format). The reformat is cosmetic — line reflow and import order — but `ttnn` was
  rebuilt afterwards and **every** run in §5 was repeated against the reformatted sources and
  the rebuilt binary, so no recorded number predates the formatting;
* three of the four raw Tracy ops CSVs (0.95, 3.03 and 3.28 MB) exceed the repo's 500 KB
  commit limit, so `tracy/*/` carries `<phase>_ops.csv.gz` plus the uncompressed
  `<phase>_perf_report.csv`. `full_attention/prefill_ops.csv` at 0.18 MB is under the limit and
  is gzipped only so all four artifacts have the same shape. The `.provenance` files still
  record the original absolute path and copy timestamp.

## 8. Second stage review — findings and what changed

The second `$stage-review` confirmed both round-1 P-findings resolved and returned
`more-work-needed` on four new ones, all of which were real.

* **`H.pcc()` scored 1.0 for a degenerate output.** A tensor with zero variance has no
  direction, and the old guard returned `1.0` whenever the denominator vanished — so an
  all-zero output, a saturated gate or a trace replay that never ran would have passed every
  assertion in the suite with a perfect record. Latent (nothing produces a constant today) but
  it undermined the metric behind every PCC record in the stage. `pcc()` now returns `1.0` only when *both*
  sides are constant and `0.0` when one is.
* **`block_size` had a documented but unenforced invariant.** `PREFILL_CHUNK` must be a whole
  number of pages and of padded chunks; nothing checked it. `block_size = 96` is legal as far
  as `paged_update_cache` is concerned, gives `lcm(256, 96) = 768`, and would have made chunk 1
  start writing K/V 32 tokens before the 2048 chunk boundary — silent cache corruption that every existing
  length assertion still passes, and that neither tested block size (32, 128, both dividing
  2048) could see. `from_state_dict` now rejects it, with
  `test_block_size_incompatible_with_prefill_chunk_is_rejected` as the guard.
* **The documented evidence command contradicted the headline claim.** `collect_evidence
  logs/*.log` picked up the deliberately-failing reverted-build control and, with the round-1
  min-wins de-duplication, reported a record below the bar. The two control logs moved to
  `logs/controls/`, one level below the flat glob, and both the collector's docstring and the
  README now say why. Re-running the documented command now reproduces the committed artifact
  exactly.
* **Stale, missing and corrupt artifacts.** `logs/tri_inv_base_sweep.log` was cited but did not
  exist; it has been regenerated on real weights against this branch, and `TRI_INV_BASE`'s
  docstring now quotes those numbers (204.7 / 161.1 / 136.4 ms, recurrent-state PCC
  0.999993 / 0.999992 / 0.999988) instead of the earlier pass's. `logs/sdpa_fit_sweep_v2.log`
  had been corrupted to NUL-padded binary by a `tee` race and was re-captured as clean text
  (it reproduces the same table). `probes/README.md` carried earlier-branch numbers (1.465,
  0.9838) and pointed at work-log sections that do not exist here; it now quotes this branch's
  measurements with the backing log named, and says so explicitly at the bottom.

Smaller items also fixed: the perf test's "~55 device ops" comment (it is 92), a work-log
cross-reference for the `ttnn.pad` hazard, `release_layers` swallowing every `RuntimeError`
during teardown (the one error class the `_free()` design exists to prevent), and the traced
decode's host-side update dropping the mesh mapper that its allocation used.

**The reviewer's "no magnitude assertion anywhere" concern is now closed rather than noted.**
`H.scale_ratio` was added and `test_full_advertised_context` asserts it inside
`SCALE_TOLERANCE = (0.98, 1.02)` for both prefill tail and decode, on both layer kinds, so the
stage's own headline defect class is gated rather than only narrated. Measured:
0.99853 / 0.99836 (`linear_attention`) and 0.99744 / 0.99493 (`full_attention`).

An earlier revision of this paragraph said this was "the check that fails on the stock decode
kernel (scale 1.29) while PCC alone still reads 0.978". Both halves were wrong and §10 records
the correction: 1.29 is the **op-level** probe `alpha`
(`logs/controls/sdpa_decode_stock_baseline.log`), and the reverted-build control fails on
decode **PCC** (0.977888, already below the 0.995 bar) before the decode scale assertion is
evaluated, so the layer-level stock decode scale is not a recorded number. PCC does catch the
stock kernel at the layer level; the scale gate is a second, independent check of a quantity PCC
cannot see, not a rescue of one PCC would miss.

Everything in §5 was re-run after these changes.

## 9. Third stage review — findings and what changed

The third `$stage-review` verified all four round-2 findings resolved and re-derived every
headline claim from the raw artifacts independently. It raised three new P2s, all of them
documentation accuracy in stage-owned files, and all fixed:

* **`doc/context_contract.json`'s acceptance counts were stale.** They still carried the totals
  from before the four scale ratios were added, while `pcc_evidence.json`, `README.md` and this
  log carried the current ones. The block is now derived from `pcc_evidence.json` and
  spells the breakdown out (268 = 260 PCC + 4 scale + 4 booleans), with the scale range and
  tolerance alongside. The runner guardrail does not check these fields, so nothing but a reader
  would have caught it.
* **`README.md` still cited `logs/controls/long_context_stock_control.log`** after the round-2 fix moved
  it to `logs/controls/`. That is the citation for the single most load-bearing claim in the
  stage — that the tt-metal change is necessary rather than precautionary — so a dead link there
  matters more than most. Fixed, and a path-resolution sweep over every link and backticked
  artifact path in the five stage documents now comes back clean.
* **`probes/README.md` had a dead `../context_contract.json` link and an over-broad provenance
  claim.** The round-2 fix corrected the rows whose numbers had changed and then asserted that
  *every* row was measured on this branch — which is not true of the nine rows that diagnose the
  earlier pass's three bugs and have no log here. Those rows are now marked *(earlier pass)*
  with an explicit note that the probes are kept as the reproduction recipe, not because the
  figures were re-measured. The `3.4e-2` / `3.3e-2` disagreement with the code docstring is
  reconciled.

Smaller items also fixed: the sweep in §3.2 is described accurately (56 measurements over six
`(k_chunk, cores)` combinations, not "40" over an implied 4x3 grid, and `k 64`/`k 256` were only
swept at 16 cores); the blast-radius table now notes that four sweep-framework loaders parse
`max_cores_per_head_batch` out of a config string, so the "no caller passes 1" claim is a
static-literal claim; `TRI_INV_BASE`'s docstring no longer reads as though the measurement
selected 16 when it favours 32 on speed; the "both kinds in every parametrised case" row names
the five `full_attention`-only tests; and two suite timings were off by a second or two.

No number, test or measurement changed in this round — every fix was to a document describing
them — so §5's runs stand as recorded.

## 10. Fourth stage review — findings and what changed

The fourth `$stage-review` verified the round-3 findings resolved and re-derived every headline
claim from the raw artifacts, including reproducing `pcc_evidence.json` field-for-field from the
logs. It raised four more P2s, all documentation accuracy, all fixed — three by correcting the
document, one by going back to the device.

* **The necessity control had run a pre-round-2 revision of the test.** §3.4 described it as
  "same code, same test", but the committed control log predated the `scale_ratio` assertion
  added in round 2, so it emitted four evidence records where the shipped run emits six. The
  fix was to **re-run the control** rather than reword: the `.cpp` was reverted to
  `837e8da3e9b`, rebuilt, and `test_full_advertised_context[full_attention]` re-run at the
  shipped test revision, then the fix restored, rebuilt, and the shipped long-context pair
  re-run to confirm. The control now reproduces `full_context_prefill_tail_pcc` **and**
  `full_context_prefill_tail_scale` bit-identically and fails at the decode PCC assertion. The
  log carries a command/date/base-SHA header it previously lacked.
  The reviewer also caught that **1.29 was being attributed to the layer-level scale gate** when
  it is the op-level probe `alpha`; the control fails on decode PCC before the decode scale
  assertion is evaluated, so the layer-level stock decode scale is genuinely not a recorded
  number. §3.4, §8 and the `SCALE_TOLERANCE` docstring now all say that.
* **`README.md` undercounted the non-64-divisible batch-32 prompts.** The lengths are
  `64 + 97*u` and `gcd(97, 64) = 1`, so it is 31 of 32. Corrected, with the derivation inline so
  the number is checkable rather than asserted.
* **`README.md` advertised the full-context command as "~13 min"** against a cited log that says
  6:23. Corrected.
* **§3.2's sweep arithmetic did not multiply out.** Six combinations at eight positions is 48,
  not 56. The sweep actually asks for eight configurations and gets seven — the six distinct
  `(k_chunk, cores_per_head)` combinations plus an `exp_approx_mode = 1` repeat that the prose
  had dropped, with `k 512 / 16 cores` throwing on L1. Seven rows at eight positions is 56.

Two other concerns also fixed: `_prefill_alignment`'s comment claimed a padded chunk is always a
whole number of SDPA k chunks, which stops being true once the k chunk grows to 512 — the op
rounds its own key extent up internally and only needs `k_chunk % TILE_WIDTH == 0`, and the
comment now says that; and §3.3 now records that the promotion roughly doubles the L1 footprint
of the promoted CBs for a future opt-in caller, which would surface as a build-time `TT_THROW`
rather than a wrong answer.

`logs/long_context.log` and `logs/controls/long_context_stock_control.log` in §5 are from this
round's re-runs; every other run in §5 is unchanged, because nothing outside these documents and
one comment changed.

## 11. Fifth stage review — findings and what changed

The fifth `$stage-review` found that §10's own write-up over-claimed: it said the "1.29 is the
op-level probe, not the layer-level gate" correction had been applied in three places when it
had reached two, and it recorded a §3.3 edit about the promotion's L1 cost that had never been
written. Both are exactly the failure mode four consecutive rounds have been catching, now
committed by the section that was supposed to be closing it. Fixed, and this time each edit was
grepped back out of the file before being written up:

* **`README.md` and §8 still attributed 1.29 to the layer-level scale gate.** Both now say it is
  the op-level probe `alpha` from `logs/controls/sdpa_decode_stock_baseline.log`, and that the
  layer-level stock decode scale is not a recorded number because the control fails the decode
  **PCC** assertion first. §8's second error is corrected too: PCC alone *does* catch the stock
  kernel at the layer level (0.977888 is already below the bar) — the scale gate is an
  independent check of a quantity PCC cannot see, not a rescue of one PCC would miss.
* **The §3.3 L1-cost note now exists**, in §3.3 *and* next to `fp32_local_accumulators` in
  `sdpa_decode_program_factory.cpp`: the promotion roughly doubles `c_21`-`c_23` and
  `c_25`-`c_31` (~70 KB at this shape), and a future opt-in caller with a larger `head_dim` or
  `vDHt` would see a build-time `TT_THROW` from the CB allocator rather than a wrong answer.
* **Both control artifacts were regenerated with provenance headers** — command, ISO date and
  the base SHA of the reverted file — in one revert/rebuild cycle, so neither depends on prose
  or file mtime any more. Both reproduce their previous numbers exactly (op-level alpha 1.30960
  / 1.28984 at 261887 / 262143; layer-level prefill tail PCC and scale bit-identical to the
  shipped run, decode PCC 0.977888). The fix was then restored, rebuilt, and the suite and
  long-context pair re-run on the final binary.
* **§7's checkpoint block no longer enumerates SHAs.** Two revisions of it lagged by one and
  then by two, because the commit recording a SHA cannot contain its own. It now names the first
  commit, points at `git rev-list 837e8da3e9b..HEAD` as authoritative, and gives the one-line
  loop that checks the property that actually matters — that every commit in the range touches
  only this model directory plus the one `.cpp`.
* **The op-suite blast-radius claim is scoped precisely**: the two non-nightly `sdpa_decode` op
  files only, with a note that the larger nightly file was not
  run and cannot reach the new branch (its cases omit `program_config`).
* §5's two timings are now taken from the logs they cite (07:40 and 06:25).

## 12. Sixth stage review — findings and what changed

The sixth `$stage-review` re-derived every headline number from raw artifacts and found none
stale, no dead paths and no corrupt artifacts — the first round with no numeric finding. It
returned three scope/attribution findings, all fixed:

* **The round-5 misattribution survived in one more place: the shipped test source.** The
  comment on the *prefill* scale assertion still said "the check that would have caught the
  stock-kernel decode (0.9779 PCC but a 1.29x attention scale) at the op level" — pairing a
  layer-level PCC with an op-level alpha, and re-asserting the framing §11 had just corrected.
  §11 said each edit was "grepped back out of the file", and that is exactly the flaw: the check
  was per-file. It is now repo-wide. The exact check, and its result at the time of writing:

  ```bash
  grep -rn '1\.29\|0\.9779\|0\.977' models/autoports/qwen_qwen3_6_27b \
       --include='*.py' --include='*.md' --include='*.json' \
       ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp
  ```

  22 hits, each one either the correction itself or a correctly attributed op-level /
  layer-level statement. The `--include` filters matter: without them the same pattern also
  matches numeric coincidences in the Tracy CSVs (`21.29`, `1.292`) and a log timestamp, which
  is what made an earlier revision of this paragraph quote a number nobody could reproduce.
* **The blast-radius op-suite claim said "the two non-nightly `sdpa_decode` op files" as though
  that were the complete set.** It is not: `test_bounded_sliding_kv_cache.py` calls
  `paged_scaled_dot_product_attention_decode`, and `test_mla_decode.py` calls
  `flash_multi_latent_attention_decode` — both go through the edited
  `SdpaDecodeDeviceOperation::create_descriptor`. Rather than reword the scope down, the
  **coverage was widened**: all four files now run, 31 collected, **30 passed, 1 skipped**. None
  can trigger the gate (`test_bounded_sliding_kv_cache.py` passes no `program_config`,
  `test_mla_decode.py` passes `max_cores_per_head_batch=4`), which is the point.
* **`probes/README.md`'s provenance sentence was still over-broad.** The round-3 fix tagged nine
  earlier-pass rows and then re-asserted completeness without re-checking the rest;
  `probe_sdpa_localise.py` and `probe_amplification.py` were untagged, and the former's own
  verdict text said "earlier-pass localisation" while the row was not marked. Both are tagged
  now, and the closing sentence is written so it can be checked row by row - it names which rows
  carry an inline log, which feed a document that does, and states that any row with a number
  and neither is a defect in that file.

Two smaller items: `logs/sdpa_decode_cfg_sweep_v2.log` — the *before* side of the invariance
check, and the one blast-radius artifact without a header — now carries one that is explicit
about being added after the fact and points at the file's own embedded device timestamps
(earliest 15:11:09, before the first build of the fix) as the dating evidence; and §8 now names
the log the 1.29 figure comes from.

The reviewer also noted that `pcc_evidence.json` was not re-emitted by the previous commit even
though the logs it derives from were re-run. That is because re-running the collector against
the final logs produces a byte-identical file, which is the intended property; it has been
re-run again here and is again identical.

## 13. Seventh stage review — and closing the class instead of the instance

The seventh review again re-derived every headline number clean and again found three
single-place stragglers of multi-place corrections: `README.md`'s capability table still said
"24 non-64-divisible batch-32 prompts" while its own sequence-length table two sections later
said 31; `doc/context_contract.json`'s `blast_radius` still carried the round-5 op-suite scope and
count that round 6 had superseded; and §12 quoted a grep count nobody could reproduce
because the command it described had no `--include` filters.

All three are fixed. But seven rounds have now produced findings in exactly this class, and the
last three reviews each ended with the same Hard-Check Gap: *nothing mechanically ties a number
in these documents back to the artifact it cites*. So this round adds the check rather than only
the fix — `scripts/check_docs.py`:

```bash
python -m models.autoports.qwen_qwen3_6_27b.scripts.check_docs
```

```
ok   every link and artifact path in 5 documents resolves
ok   268 records (260 PCC, 4 scale), min PCC 0.998031, 0 below the bar; context_contract.json agrees
ok   4 perf measurements re-derived from their tt-perf-report CSVs
ok   test counts in the documents match their run logs: suite_main=57, long_context=2, watcher_run=9, ttnn_sdpa_decode_op_tests=30
ok   every '<n> passed' figure in the documents is produced by a committed run log

all document checks passed
```

It re-derives the record counts, the PCC minimum and the scale range from `pcc_evidence.json`
and asserts `context_contract.json`'s acceptance block matches; re-derives every perf number by
summing the `Device Time` column of the `tt-perf-report` CSVs and re-checks the replay
periodicity; re-derives the test counts from the run logs; and asserts that every markdown link
and backticked artifact path resolves.

It earned its keep immediately: on its first run it found a **fourth** straggler this round had
missed — §10 still citing the long-context control log at its pre-move path, i.e. without the
`controls/` component — and then
a fifth, a superseded op-suite pass count surviving in §12's own narrative. Neither would have been
caught by re-reading. Both are fixed, and the checker is green.

It also constrains how this log may write history: a superseded figure cannot be quoted
verbatim, because the checker cannot tell a historical quote from a live claim. Two sentences
in §12 and §13 are phrased around that ("a superseded op-suite pass count" rather than the
number). That is the right trade - a reader who wants the old number can read the review
history in git, and no reader can mistake it for a current one.

Two smaller items from the same review: the gzip rationale claimed the raw Tracy ops CSVs are
"0.9-3.2 MB" when one of the four is 0.18 MB and under the commit limit (it is gzipped only so
all four artifacts have the same shape), and the "two `sdpa_decode`-named files" phrase now
names them. One item is recorded rather than changed: the four Tracy runs and the watcher run
predate the last revert/restore cycle by about an hour. The sources they exercised are identical
— that cycle only reverted and restored the same `.cpp` — and §11's re-run claim was already
scoped to the suite and the long-context pair, but the perf and watcher artifacts are from the
earlier build of identical source, and that is worth saying plainly rather than leaving to
timestamp archaeology.

## 14. Eighth stage review — the checker was checking less than it claimed

The eighth review confirmed again that no numeric, correctness or performance claim in the stage
is wrong — it re-derived the whole chain, raw Tracy CSV through `tt-perf-report` through
`perf_summary.json` to the README table, and reproduced `pcc_evidence.json` element-for-element
from the logs. It found two things, and the first is the important one.

**`scripts/check_docs.py` did not implement the check it advertised.** Its docstring claimed "one
value per figure … the same value in all of them"; the function that was supposed to do it built
a dictionary of figures and never read it, and nothing compared a derived number against the
*prose* in `README.md` or `work_log.md` — only against other JSON. The reviewer proved it: with the
work log's perf row and the README's headline PCC minimum both edited to wrong values, the
checker still printed five `ok` lines and exited 0.
A stage-owned check that overstates its own guarantee is the same defect class as round 1's false
blast-radius claim, and it would have misled every later stage inheriting this tree.

The checker now actually does it. For each figure it derives — the record counts, the PCC
minimum, the scale range, each phase's `ops_per_pass` and `device_kernel_time_ms`, and each run
log's pass count — it finds every occurrence in the prose and asserts the derived value, and it
binds each `<n> passed` to the log that produced it by requiring the sentence to name that run.
Two further claims are now true rather than aspirational: the perf check verifies **real op-code
periodicity** (op *i* equals op *i mod period* across all 8 replays) rather than only that the
op count divides by 8, and the docstring no longer claims to cover every backticked path — it
covers paths under the stage artifact directories, which is what the code matches.

And it now proves it is not vacuous. `--self-test` copies the tree to a temporary directory,
applies five mutations — a wrong perf row, a wrong PCC minimum, a wrong record count, one run's
pass count attributed to another run, a dead link — and asserts the checker rejects each:

```
$ python -m models.autoports.qwen_qwen3_6_27b.scripts.check_docs --self-test
ok   rejected: a wrong perf row in the work log
ok   rejected: a wrong PCC minimum in the README
ok   rejected: a wrong record count
ok   rejected: the suite's count attributed to watcher
ok   rejected: a dead link

self-test passed: the checker rejects every mutation it claims to catch
```

The rewritten checker immediately earned it, failing on four live stragglers in the committed
tree that seven rounds of reading had not caught, including the review's second finding: §7 still
carried the superseded "0.9-3.2 MB" Tracy-CSV range that round 7 corrected in `README.md` only —
a range wrong at both ends, whose "over the 500 KB limit" rationale is false for the 0.18 MB
file. Fixed here, and it is now a checked statement rather than a remembered one.

Three smaller review points are recorded rather than changed: `check_paths` no longer claims
coverage it does not have; `check_evidence` validates `pcc_evidence.json` against its own summary
and the contract, not against the logs (the reviewer re-derived that hop by hand and it
reproduces element-for-element, and `collect_evidence` regenerating identically is the standing
check); and the op-suite blast-radius control was run on the fixed build only, which is the right
control for "breaks no existing caller".

## 15. Ninth stage review — the checker now gates the class, and proves it

The ninth review again re-derived the entire chain clean — raw Tracy CSV through
`tt-perf-report` to the README rows, and the logs through `collect_evidence` to every cell of
the correctness table — and found no wrong number anywhere in the stage. Its two findings were
both about the checker's honesty and reach, and one was measured rather than argued: the
reviewer ran a battery of 19 realistic drift mutations drawn from what rounds 1-8 actually
found, and **18 of them passed**. The checker gated only the handful of figures it enumerated,
while `README.md` described it as checking "every number".

That is fixed by making it check the class, not a list. The new `check_quoted_numbers` takes
every committed artifact — all the run logs, `pcc_evidence.json`, `perf_summary.json`, the
`tt-perf-report` CSVs, `context_contract.json` — as the corpus, and requires **every decimal
with three or more fraction digits in the prose** to be some artifact number, *rounded*: a
quoted `1.310` is accepted because `1.30960` rounds to it at three places, `0.998031` because
`0.9980307630901388` does at six. Anything invented or edited fails. Two exemptions, both
narrow and both documented in the function: a row tagged *(earlier pass)* in `probes/README.md`,
which that file already declares unbacked, and the earlier pass's own full-context numbers,
which now have a committed artifact of their own (see below). The scale range and the watcher
line census are pinned to their artifacts as well, closing the reviewer's specific finding that
the docstring listed the scale range among the checked figures while the code never compared it.

`--self-test` grew from five mutations to ten, covering the classes the reviewer used: a wrong
perf row, a wrong PCC minimum, a wrong record count, a wrong scale range at either end, an
edited cell of the README correctness table, an edited probe alpha, a wrong watcher census, one
run's pass count attributed to another, and a dead link. All ten are rejected.

Building it found two more live items, neither of which any amount of re-reading had produced:

* §3.5's before/after table quoted the earlier pass's `0.550293` and `0.998817` with no artifact
  behind them at all. Rather than delete the comparison — it is the clearest statement of what
  this stage changed — those numbers now come from
  [`earlier_pass_reference.json`](earlier_pass_reference.json), extracted from that branch's own
  committed `pcc_evidence.json` with the commit and the extraction command recorded.
* The self-test itself found a hole in the first draft of the corpus check: it matched only
  numbers starting `0.` or `1.`, so an edited two-digit alpha sailed through. Widened to any
  decimal with three or more fraction digits.

The review's second finding is fixed too: `SDPA_MAX_K_CHUNKS`'s docstring attributed its whole
alpha table to `logs/sdpa_long_sweep_v2.log`, which holds only the 131072- and 262144-key rows;
the three 8192-key rows are in `logs/sdpa_fit_sweep_v2.log`. Both logs are now named.

What the checker still does **not** cover, stated plainly rather than implied away: it reads the
five stage documents, so numbers in test sources, in `tt/functional_decoder.py` docstrings and
in `probes/README.md`'s untagged rows are outside it; and it cannot catch a *swap* of two
numbers that both exist somewhere in the artifacts. `README.md`'s description of the command and
the module docstring both say exactly this now.

## 16. Tenth stage review — measuring the gate instead of describing it

The tenth review confirmed every goal-contract requirement met and every headline number
independently re-derived, and returned one finding: the checker's *description* had again
outrun the checker. The reviewer measured it — 33 realistic drift mutations, 10 caught, 23
missed — and named the two classes. Figures outside the three-decimal shape were not matched at
all (`37.7x`, the `1.29` attribution that rounds 4-6 each found wrong somewhere, host-wall
milliseconds, `3705x`); and inside the matched shape, a *plausible* edit could land on a value
that happens to exist in a corpus of thousands of numbers.

The description is now true, and the gate is wider:

* `check_quoted_numbers` matches **every** decimal at any precision, and every integer of three
  digits or more, rather than only three-decimal values. Integers must appear verbatim and not
  embedded in a longer word, so a commit SHA cannot launder one. Artifact byte sizes and the
  watcher first-token histogram are in the corpus, because the documents legitimately quote
  them.
* The batch-32 non-64-divisible count — the claim that regressed in round 4 and again in round 7,
  and is too small an integer for the general net — is now **derived from the test's own
  `seq_lens` formula** and pinned.
* The docstring says what the check does and does not buy, in those words: it catches a number
  that exists in no artifact; it does not catch a value that happens to exist somewhere in a
  large corpus, and `check_prose` is what pins the specific figures to one derived value. The
  README command comment and this section say the same.

`--self-test` now covers 15 drift classes, including every class the reviewer used to defeat the
previous version: an edited one-decimal alpha, an edited host-wall time, an edited integer scale
blow-up, an edited `TRI_INV_BASE` timing, and the round-4 batch-32 divisibility regression
itself. All 15 are rejected:

```
$ python -m models.autoports.qwen_qwen3_6_27b.scripts.check_docs --self-test
...
ok   rejected: an edited one-decimal alpha
ok   rejected: an edited host-wall time
ok   rejected: an edited integer scale blow-up
ok   rejected: an edited TRI_INV_BASE timing
ok   rejected: the round-4 batch-32 divisibility regression

self-test passed: the checker rejects every mutation it claims to catch
```

Building it turned up one more live item, again by construction rather than by reading: the
gzip rationale quoted artifact sizes that were in no artifact, so the sizes are now derived into
the corpus. And it re-imposed the history rule twice more — §10's account of the batch-32
finding and §12's of the op-suite count now describe the superseded figures instead of quoting
them, because the checker cannot tell a historical quote from a live claim and should not try.

What remains outside it, measured rather than intended: the five stage documents only, so
numbers in `tt/`, `tests/` and `probes/*.py` are not gated; a swap of two numbers that both
exist in the corpus; and digit runs inside an identifier (a commit SHA, `p300c`,
`p150_mesh_graph_descriptor`), which are skipped by design because they are not quantities.

## 17. Eleventh stage review — the gate now holds on a fresh clone

The eleventh review again re-derived every substantive claim clean and returned two findings,
both on the checker, both demonstrable:

* **It passed here and failed on the committed tree.** `artifact_corpus` built its size facts by
  `stat`-ing the *uncompressed* Tracy ops CSVs, which are gitignored. The reviewer ran the
  checker against a `git archive HEAD` export and it exited 1: one of the four sizes the gzip
  rationale quotes had no committed backing at all, and two others matched only by coincidence
  against unrelated numbers. So the README's instruction — run this to verify these documents —
  did not work on a fresh clone, and round 10's fix for exactly this class was incomplete.
  The sizes now come from the **gzip trailer of the committed `.gz`** (ISIZE, the last four
  bytes), which recovers all four uncompressed sizes exactly with nothing untracked involved.
  And the guard is now standing rather than remembered: `--self-test` exports the model
  directory with `git archive HEAD`, overlays the checker under test, and runs the whole check
  suite there, so a dependency on an untracked file fails the self-test.
* **The integer net had three syntactic blind spots.** The old rule refused any digits adjacent
  to a dot or a word character, which correctly skipped commit SHAs but also skipped
  `735..768`, `64..3071` and any sentence-final figure — the non-aligned-length coverage claims,
  which is precisely what the checker exists to protect. The rule is now "take the maximal
  alphanumeric run around the digits; if it is longer and contains a letter it is an identifier,
  otherwise it is a figure", with two explicit carve-outs: the fraction of a decimal (already
  checked as a decimal, including when written as a regex like `0\.9779`) and a trailing `x`
  multiplier. The `--self-test` battery now contains the round-4 batch-32 regression and the
  integer scale blow-up, and all 15 classes are rejected.

Both were found by measurement, not reading — the reviewer's 43-mutation drift battery and a
16-case syntactic-position battery using a token first proved absent from the corpus. That is
the right way to check a checker, and it is why the "outside its reach" list in §16 is now
written from measurement.
