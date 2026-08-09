# Qwen/Qwen3.6-27B — functional decoder work log

Model dir: `models/autoports/qwen_qwen3_6_27b` (slug of the HF model id).
`models/autoports/qwen3.6-27b` is a symlink to it so runner-side gates scoped with
`MODEL_DIR=models/autoports/qwen3.6-27b` resolve to the same artifacts.

## 0. Environment / hardware recovery (infrastructure, not a model result)

Host shows 4 Tenstorrent Blackhole PCI functions (`lspci -d 1e52:`, `/dev/tenstorrent/{0,1,2,3}`).
**Correction, see section 6.3:** these are not four p150s. They are two **p300c** boards of two
chips each; the wedged chip 0 is the partner of chip 1. The original text below said p150 and
is left as written so the correction is traceable.

### Failure signature

First device-facing command of the stage:

```
timeout 60 tt-smi -ls --local      # exit 124 (hang; --local is Wormhole-only)
```

then a TTNN open-device probe:

```
python -c "import ttnn; ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0)"
```

```
UMD | Setting power state failed on device 0: Invalid argument (pci_device.cpp:1073)
RuntimeError: ARC startup error at core 8-0 over NOC0: scratch_status=0x2,
postcode=0xc0de001c, smc_init_status=0x0 (No errors) (Timed out after 300000 ms)
  tt::umd::BlackholeTTDevice::wait_arc_core_start(...)
```

### Recovery attempts (each bounded, one at a time)

| # | command | result | log |
|---|---|---|---|
| 1 | `timeout 400 tt-smi -r` | reset issued, `Error when re-initializing chips! ARC startup error ... postcode=0xc0de001c` | `/tmp/ttdev/reset2.log` |
| 2 | `timeout 400 tt-smi -r` (second bounded reset per skill) | same ARC startup error | `/tmp/ttdev/reset3.log` |
| 3 | `tt-smi -r --no_reinit` + TTNN open probe | same ARC startup error | `/tmp/ttdev/reset_all_nr.log`, `/tmp/ttdev/smoke3.log` |
| 4 | `modprobe -r tenstorrent && modprobe tenstorrent` + TTNN open probe | same ARC startup error | `/tmp/ttdev/smoke4.log` |
| 5 | PCIe FLR `echo 1 > /sys/bus/pci/devices/0000:01:00.0/reset` + TTNN open probe | same ARC startup error | `/tmp/ttdev/smoke5.log` |

`/sys/class/tenstorrent/tenstorrent!0/device -> 0000:01:00.0`, i.e. the wedged ARC is
**PCI device 0 only**. Devices 1..3 initialise cleanly.

### Resolution — restrict the stage to a healthy single device

`TT_VISIBLE_DEVICES` makes UMD topology discovery skip the wedged chip.  This originally
selected device 1; section 6.3 moves it to device 2, on the fully intact board. Metal then
classifies the cluster as `CUSTOM` and requires an explicit fabric mesh graph descriptor
(`tt_cluster.cpp:281`), satisfied with the stock single-Blackhole-chip descriptor, whose file
name happens to be `p150_mesh_graph_descriptor.textproto` - it describes one chip, which is
what a 1x1 mesh exposes, and does not imply the host has p150s.

Additionally, kernel sources are resolved relative to the current working directory when a
`tt_metal/` tree exists there. The active `ttnn` is the **built** tree
`/home/ttuser/.local/lib/model-bringup/tt-metal`, whose
`tt_metal/impl/dispatch/kernels/cq_dispatch.cpp` differs from the (unbuilt) repo copy at
`/home/ttuser/dev/qwen/tt-metal`; running with cwd inside the repo made metal compile the
repo's dispatch kernel against the installed headers and fail:

```
cq_dispatch.cpp:1486:5: error: 'init_telemetry' was not declared in this scope
```

So every device job in this stage is launched from `/home/ttuser/dev/qwen/rundir`, with the
repo on `PYTHONPATH`. The environment file is `models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh`:

```bash
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export TT_VISIBLE_DEVICES=1
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export REPO=/home/ttuser/dev/qwen/tt-metal
export PYTHONPATH=$REPO
```

### Mesh smoke + compute smoke (post-recovery)

```
cd /home/ttuser/dev/qwen/rundir && source ttenv.sh
python /tmp/ttdev/smoke2.py        -> opened in 1.5s, MeshShape([1, 1]), MESH_SMOKE_OK
python /tmp/ttdev/compute_smoke.py -> matmul ok Shape([1, 1, 512, 512])
```

Recorded as infrastructure recovery. No stage was declared blocked; the wedged board 0 is
excluded and bringup runs on a healthy 1x1 mesh as the skill requires. Board 0 needs an
operator-side host power cycle (host reboot was deliberately not performed because it would
kill the unattended pipeline runner; two bounded resets, a driver reload and a PCIe FLR all
failed to restart its ARC).

## 1. HF architecture read

`Qwen/Qwen3.6-27B` is `model_type: qwen3_5`, a text+vision model; only the text stack
(`Qwen3_5TextModel`, `text_config.model_type == qwen3_5_text`) is in scope for the decoder.
64 decoder layers alternate two **kinds** via `text_config.layer_types`
(`full_attention_interval = 4`): three `linear_attention` layers then one `full_attention`
layer, repeating. Layer 0 and layer 3 are therefore the two representative layer kinds and
are what the tests instantiate.

| item | value |
|---|---|
| `hidden_size` / `intermediate_size` | 5120 / 17408 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 4 / 256 |
| `attn_output_gate` | `true` — `q_proj` emits `2 * heads * head_dim`, second half is a sigmoid gate |
| `partial_rotary_factor` | 0.25 — only 64 of 256 head channels rotate |
| `rope_parameters.mrope_section` | `[11, 11, 10]`, `mrope_interleaved: true` |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 48 (v-per-k = 3) |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 |
| `linear_conv_kernel_dim` | 4 (depthwise causal conv over `2*key_dim + value_dim = 10240` channels) |
| `rms_norm_eps` | 1e-6, and `Qwen3_5RMSNorm` is **1-centred**: `y = normed * (1 + w)` |
| `Qwen3_5RMSNormGated` | **not** 1-centred (`weight` initialised to ones), norm before gate |
| `max_position_embeddings` | 262144 |

mRoPE: `Qwen3_5TextModel.forward` builds `position_ids` of shape `(4, batch, seq)` whose
rows are identical for pure text. `reference/hf_reference.py::text_position_embeddings`
constructs cos/sin exactly the way the real model does **and asserts** that the result is
bit-identical to plain partial RoPE, so the TT layer can consume a single `(cos, sin)` pair.
That assertion runs in every `full_attention` test.

## 2. Bugs found and fixed (each verified in isolation before the fix was kept)

### 2.1 `ttnn.pad` returns a view when the padding fits inside the tile padding

*Symptom.* `test_batched_users[32-*]` failed for both layer kinds; a length sweep showed
prefill PCC collapsing to ~0.08 for exactly those logical lengths with
`ceil(len/32)*32 == round_up(len, alignment)` (743, 1519, 2295, 3071), while neighbouring
lengths passed at 0.999. The error was already present at output position 0, i.e. the layer
*input* was wrong.

*Experiment.* `rundir/probe_pad.py` showed `ttnn.pad` itself is correct (tail zeroed, head
preserved). `rundir/probe_pad_alias.py` added the one thing the layer does and the probe did
not — deallocating the pad input — and reproduced it exactly:

```
logical=737 padded=768 padamt=31 pad_is_alias=True  head_ok_after_dealloc=False absmax=7.0
logical=736 padded=768 padamt=32 pad_is_alias=False head_ok_after_dealloc=True
```

`ttnn.pad` returns an **alias** of its input when the requested padding fits inside the
existing tile padding (`pad < 32`). `_pad_seq` then freed the input, so the padded tensor
pointed at freed DRAM and the next allocation (`absmax=7.0`, the junk tensor) overwrote it.

*Fix.* `_free(tensor, *live)` in `tt/functional_decoder.py` — a `ttnn.deallocate` that is a
no-op when the target shares a buffer with a still-live tensor. The same hazard exists for
`ttnn.reshape`, `ttnn.typecast` to the same dtype, full-range `ttnn.slice`,
`ttnn.concat` of a single tensor and `ttnn.to_memory_config` into the current config; every
deallocation in the file whose target can alias a live tensor now goes through `_free`.
Latent instances this also removed: `prepare_decode_state` freeing the per-user state at
`max_batch == 1` (`concat` of one tensor), `_linear_attention_decode` freeing the caller's
`normed` via `reshape`, `flat = reshape(gated)` followed by `deallocate(gated)` before the
`out_proj` matmul, and `_causal_conv` freeing its own returned conv state at `seq_len == 1`.

*Verification.* `rundir/probe_boundary.py`, lengths 735/736/737/743/767/768 for both kinds:
all now 0.999083–0.999085 (`full_attention`) and 0.999942 (`linear_attention`).

### 2.2 Zero-padded prefill tokens corrupted the gated-delta-net state

*Symptom.* `test_decode_pcc[{17,2049,5000}-linear_attention]` and
`test_linear_state_and_kv_cache_match_reference[linear_attention]` failed while the prefill
*output* PCC passed. The read-back recurrent state had ~1e-29 entries where the HF reference
had ~1e-3.

*Cause.* HF's `torch_chunk_gated_delta_rule` pads `q/k/v/beta/g` with **zeros** to the
64-token chunk, so a padded position is an exact no-op (`beta = 0` contributes nothing,
`g = 0` leaves the decay at 1). This implementation pads the *hidden states* instead, which
leaves `beta = sigmoid(0) = 0.5` and `g = -exp(A_log) * softplus(dt_bias) != 0`, so up to 63
phantom tokens decayed and updated the recurrent state. Separately, the conv state was
sliced at the padded end rather than at the last logical token.

*Fix.* `_zero_after_seq` masks `beta` and `g` beyond the logical length, and `_causal_conv`
takes the logical length and slices the conv window at `logical - 1`.

### 2.3 Neumann doubling product for `(I - A)^-1` is unusable at the real weights

*Symptom.* Everything passed with synthetic weights, but `test_real_weights[linear_attention]`
gave prefill PCC 0.937 and the recurrent state diverged to `absmax 2.76e18`
(reference: 3.36), growing with sequence length (`rundir/probe_real_linear.py`).

*Experiment.* `rundir/probe_inv.py` captured the real `attn0` matrices out of HF's own
`torch_chunk_gated_delta_rule` and ran only the inverse on device:

```
attn0 absmax=0.7377   exact |inv|max=1.0
TTNN doubling product : |got|max=3.28   max_abs_err=3.27
torch float32 doubling:                 max_abs_err=2.45e-04
torch float64 doubling:                 max_abs_err=1.08e-07
```

so the algorithm is right and the *precision* is the problem. Tracing the intermediates in
torch showed why: `|A^8|` peaks at 9.9e2 and the partial product at 4.3e2 before cancelling
back to 1.0 — about three orders of catastrophic cancellation. TTNN's fp32/HiFi4 matmul was
measured at ~1.4e-3 relative error (`rundir/probe_matmul_prec.py`), which after that
cancellation leaves an absolute error of ~3 on a result of magnitude 1.

*Fix.* `FunctionalDecoder._unit_tri_inverse` inverts the unit-lower-triangular matrix by
recursive 2x2 block inversion, `[[L11,0],[L21,L22]]^-1 = [[X11,0],[X22 a21 X11, X22]]`, with
both diagonal blocks inverted in one batched call, falling back to the doubling product only
at `TRI_INV_BASE = 8`. Every intermediate is then itself a well-conditioned unit-triangular
inverse. Measured on the same captured `attn0` (`rundir/probe_blockinv.py`):

| method | max abs error |
|---|---|
| doubling product at 64 | 3.269 |
| block recursion, base 32 | 3.44e-2 |
| block recursion, base 16 | 8.81e-3 |
| **block recursion, base 8 (selected)** | **1.76e-3** |

*Verification.* Real weights, `linear_attention`, `rundir/probe_real_linear.py`:
prefill PCC 0.999956 / 0.999964 / 0.999968 at seq 64 / 128 / 2049, recurrent-state PCC
0.999986 / 0.999988 / 0.999991, decode PCC 0.999983 / 0.999990 / 0.999989.

## 3. Profiling environment: the shared install has no Tracy

The first `python -m tracy -r -p -v -m pytest ...` attempt died before the test even
collected:

```
TT_FATAL: TT_METAL_DEVICE_PROFILER requires a Tracy-enabled build of tt-metal. (assert.hpp:104)
RuntimeError: TT_FATAL @ tt_metal/llrt/rtoptions.cpp:816
```

```
$ grep ENABLE_TRACY ~/.local/lib/model-bringup/tt-metal/build_Release/CMakeCache.txt
ENABLE_TRACY:BOOL=OFF
```

tt-metal enables the profiler by default (`build_metal.sh --disable-profiler` turns it off),
so the shared install was deliberately built without it. The skill's device-only fallback
(`TT_METAL_DEVICE_PROFILER=1 pytest ...` + `tools/tracy/process_ops_logs.py`) hits the same
assert, because it is the same env var and the same check.

Rebuilding the shared install in place would have swapped `ttnn/ttnn/_ttnn.so` underneath the
rest of the unattended pipeline, so instead the same source tree (same commit
`559921b40a8b7b21c807d5592323c2fa5e8c7ecb`) was copied to
`~/.local/lib/model-bringup/tt-metal-profiler` and rebuilt with the profiler enabled:

```bash
tar -C ~/.local/lib/model-bringup/tt-metal --exclude=./build --exclude=./build_Release \
    --exclude=./.git --exclude=./generated -cf - . \
  | tar -C ~/.local/lib/model-bringup/tt-metal-profiler -xf -
cd ~/.local/lib/model-bringup/tt-metal-profiler && ./build_metal.sh --release
```

`doc/functional_decoder/ttenv_profiler.sh` points `TT_METAL_HOME` and `PYTHONPATH` at that
tree; it is used **only** for the two profiling runs. Every correctness number in this stage
was produced with `ttenv.sh` against the untouched shared install. Build log:
`/home/ttuser/dev/qwen/rundir/logs/build_profiler.log`.

## 4. Capability contract evidence table

| claim | evidence | remaining risk |
|---|---|---|
| Both HF layer kinds are implemented and correct | 42 functional tests, both kinds in every parametrised case; `logs/suite_final.log` = `42 passed, 2 skipped` | Only layers 0 and 3 are instantiated. `layer_types` contains exactly two distinct values, and `decoder_shapes` fails loudly on any third value, so a new kind cannot be silently mis-handled. |
| Advertised context 262144 is supported, not reduced | `test_full_advertised_context` prefills 262144 and decodes at 262143 for both kinds; prefix PCC 1.0000000000000 / 0.9999999999996; `doc/context_contract.json` | No HF reference exists at that length; correctness there rests on prefix consistency plus the reference-checked lengths up to 16385. |
| Paged KV cache behaves under a non-trivial page table | Page tables are a shuffled permutation; `test_linear_state_and_kv_cache_match_reference` un-pages the device cache and compares against HF's cache (K 0.999989, V 0.999993) | Cache compare is at one length (2049) and batch 1; batched addressing is covered indirectly by `test_batched_users[32]`. |
| Page/block geometry is a parameter | `test_alternate_page_block_size` at 32 and 128 | Only three block sizes tested. |
| Non-aligned logical lengths work on the public API | 1, 17, 128, 2049, 5000, 8191, 16385, plus the 735..768 pad-below-one-tile range and 24 non-64-divisible batch-32 prompts | — |
| Decode is traceable and correct from replay | `test_traced_decode_pcc` compares **replay** output against HF (min 0.998490) | Trace captured at batch 1 only. |
| Batch > 1 works | batch 4 and 32, unequal prompt lengths, per-user page tables and current positions | Batch 32 tested at max_seq_len 8192, not at 262144 (34 GB of KV cache > 31 GiB DRAM); recorded in `context_contract.json`. |
| Real checkpoint weights load and pass | `test_real_weights`, prefill 0.999968 / 0.999952 and decode 0.999989 / 0.999972 | One layer per kind, one length. |
| No host fallback in a measured pass | `test_no_runtime_host_fallback`: source scan of the helpers **and** the runtime path, plus a live run with `ttnn.from_torch`/`to_torch`/`as_tensor` stubbed to raise | — |
| Deterministic for repeated inputs | bit-identical prefill and decode outputs | — |
| Watcher clean | `watcher/WATCHER_AUDIT.md`, 4 passed, no fatal/assert/sanitize lines | Watcher subset is 4 tests, not the whole suite. |
| Warmed prefill and traced warmed decode measured | `perf_summary.json` + `tracy/*/…_perf_report.txt`, marker-drop-free windows verified by exact op-count periodicity | Single-chip, batch 1, unoptimised config; this stage does not claim a perf target. |

## 5. Final state

* Functional suite: `logs/suite_final.log` — **42 passed, 2 skipped** (the 2 skips are the
  `--long-context` tests, run separately).
* Full advertised context: `logs/long_context.log` — **2 passed** in 161 s.
* Watcher: `logs/watcher_run.log` — **4 passed**, log clean (`watcher/WATCHER_AUDIT.md`).
* Profiling: 4 Tracy runs, `tracy/<kind>/<phase>_perf_report.txt`, summarised in
  `perf_summary.json`.
* All 218 recorded values: `pcc_evidence.json`; minimum 0.99788 against a 0.995 bar.

## 6. First stage review — findings and what changed

An independent `$stage-review` subagent returned `more-work-needed`. Every finding was
treated as work; nothing was argued away.

### 6.1 P1 — the full-context correctness check was a tautology

The reviewer showed that comparing the first 4096 outputs of a 262144-token prefill against a
fresh 4096-token prefill proves nothing: a causal prefill computes those outputs in chunks 0-1
and nothing the later 250k tokens do can change them. The recorded values (1.0000000000000138
and 0.9999999999996415) are bit-identical precisely because the check cannot fail. The
reviewer also refuted the recorded reason for not running a real reference: the eager-attention
`num_heads * L^2 * 4` cost argument applies to `full_attention` only — the `linear_attention`
reference is O(L) and supports cache continuation.

`test_full_advertised_context` was rewritten to check the full context against the HF
reference, using the only constructions that stay O(seq_len):

* `linear_attention` — the HF layer runs in 16384-token segments with a carried
  `DynamicCache`. `Qwen3_5GatedDeltaNet` continues exactly from a populated cache (it
  prepends `conv_state` and passes `recurrent_state` as `initial_state`), so this is the same
  computation as one long call. Compared: the last 8192 outputs, the final conv and recurrent
  state, and a decode step continuing from that state.
* `full_attention` — the K/V cache is built from `k_proj`/`v_proj` + `k_norm` + partial RoPE
  only, with **no attention**, then the real HF layer runs on the last 8192 queries against
  that cache. The construction is first validated to be `torch.equal` to a genuine short
  `reference_prefill` cache, so it cannot silently diverge from what
  `Qwen3_5Attention.forward` writes. Compared: the last 8192 outputs, the un-paged device KV
  cache over all 262143 positions, and a decode step at position 262143.

The prompt length is now 262143 so that prompt + one decoded token occupy exactly the
advertised 262144 positions and decode runs at the last addressable position. 262143 is also
divisible by none of the tile, page, delta chunk, SDPA chunk or prefill chunk.

### 6.2 P2 — silent cache truncation and an under-provisioned block count

`_full_attention_prefill` trimmed K/V when the caller's per-chunk page table was shorter than
the padded chunk, which would silently drop cache writes for the tail of a prompt. It now
asserts. Relatedly, `from_state_dict` sized `max_num_blocks` from `max_seq_len` while the
layer pads the final chunk up to `lcm(SDPA_CHUNK, block_size)`, so the documented
`seq_len <= max_seq_len` contract was false whenever `max_seq_len` was not a multiple of 256
(e.g. `max_seq_len=100000` needs block 1564 of 1563). `max_num_blocks` is now derived from
the padded context via the shared `_prefill_alignment` helper.

### 6.3 P2 — the hardware was misidentified, and the run sat on the damaged board

`/sys/class/tenstorrent/*/tt_card_type` reports **p300c**, not p150, and `tt_serial` shows
two boards of two chips each:

| board serial | chips | state |
|---|---|---|
| `000004613192404C` | `/dev/tenstorrent/0`, `/dev/tenstorrent/1` | chip 0's ARC is wedged; chip 1 healthy |
| `0000046131924022` | `/dev/tenstorrent/2`, `/dev/tenstorrent/3` | both healthy |

So the wedged chip is the **partner of the chip the stage had been running on**. Every
"p150" reference in the stage evidence was wrong, and a fully intact board was available.
`ttenv.sh` now selects `TT_VISIBLE_DEVICES=2`, every device run in this stage was repeated on
that chip, and the board layout is recorded in both env files. The `p150_mesh_graph_descriptor`
is still the right descriptor — it describes a single Blackhole chip, which is what a 1x1 mesh
exposes — and the env file now says so instead of implying the host has p150s.

### 6.4 P2 — the optimization notes pointed at 2.9 % of the time, and `TRI_INV_BASE` was unmeasured

Re-aggregating `tracy/linear_attention/prefill_perf_report.csv` showed that the single-core
32x32 matmuls of `_unit_tri_inverse` were **43 %** of `linear_attention` prefill device time,
while the two matmuls the README called dominant were 2.9 %. `TRI_INV_BASE = 8` had been
chosen purely on the inverse-level error table with no runtime or model-level measurement.

`probes/probe_tri_inv_base.py` measures both on real checkpoint weights
(`logs/tri_inv_base_sweep.log`):

| `TRI_INV_BASE` | real-weight prefill PCC | recurrent state PCC | decode PCC | warmed 2048 prefill |
|---|---|---|---|---|
| 8 | 0.999968 | 0.999991 | 0.999986 | 204.6 ms |
| **16 (selected)** | **0.999968** | **0.999991** | **0.999986** | **160.5 ms** |
| 32 | 0.999967 | 0.999986 | 0.999988 | 135.8 ms |

Base 16 matches base 8 to six decimals on every metric and is 21 % faster, so it is the new
default. Base 32 is faster still but is the first base whose recurrent-state PCC moves; the
measured trade is handed to the optimization stage rather than taken here.

### 6.5 Other review findings, all fixed

* `test_no_runtime_host_fallback`'s source scan skipped `__init__`; it now scans the module
  helpers, `__init__` and the whole post-setup region, and asserts each region is non-empty so
  a moved anchor cannot silently disable the check.
* The pad-below-one-tile lengths (735, 736, 737, 743, 767, 768) were only in a probe, not in
  any assertion, while the README listed them as covered. They are now a real parametrised
  regression test over both layer kinds, checking prefill PCC **and** a following decode step
  (prefill output alone did not catch the original state bug).
* `test_alternate_page_block_size` was parametrised over both kinds although
  `linear_attention` has no paged cache; it is now `full_attention` only.
* `cache_dtype=ttnn.bfloat8_b` was a constructor parameter no test exercised. `test_bfloat8_kv_cache`
  now covers it, which is what proves the asymmetric dtype contract (prefill casts K/V to the
  cache dtype before `paged_fill_cache`; decode must not cast the `paged_update_cache` update
  tensors).
* Trace capture/replay was only tested at batch 1. `test_traced_decode_batched` adds batch 4
  with per-user positions and page tables.
* Dead code removed: the `chunk_start & -chunk_start` idiom in `_sdpa_program_config` and
  `_prefill_alignment` could only ever produce 256 (chunk starts are `PREFILL_CHUNK`
  multiples), and `_rms_norm_heads` was a pure passthrough.
* `prepare_decode_state`'s all-slots-at-once behaviour blocks continuous batching. It is now
  documented as an explicit limitation on the method, with the shape of the fix.
* README arithmetic corrected: the PCC minimum is over the numeric records (6 of the records
  are booleans), and the `full_attention` "0.99997" value is at seq_len 1, not 17.
* The README claimed the profiler drain meant "nothing is dropped". Markers *were* dropped
  during setup, before the drain and before the start signpost; the measured windows are
  provably complete (the op sequences are exactly periodic), and the wording now says that.

## 7. The long-context bug the rewritten test found

Replacing the tautological prefix check (6.1) with a real HF-referenced check immediately
found what it was built to find. At the full advertised context, `full_attention` prefill is
**wrong**, and the failure is invisible to every shorter test:

```
PCCEVIDENCE full_context_paged_k_cache_pcc  seq_len=262143  0.9999887
PCCEVIDENCE full_context_paged_v_cache_pcc  seq_len=262143  0.9999926
PCCEVIDENCE full_context_prefill_tail_pcc   seq_len=262143  0.9699976   <-- bar is 0.995
```

`linear_attention` is unaffected at the same length (tail 0.9999415, conv state 0.9999946,
recurrent state 0.9999809, decode at position 262143 0.9999458).

### Localisation

The paged K/V cache matches the HF cache at 0.99999 over all 262143 positions, so everything
*written* is right and the loss is inside `chunked_scaled_dot_product_attention` itself.
`probes/probe_sdpa_precision.py` swept the context length
(`logs/sdpa_precision_sweep.log`):

| context | tail PCC, `fp32_dest_acc_en=False` | tail PCC, `True` |
|---|---|---|
| 8191 | 0.998966 | 0.999473 |
| 32767 | 0.999349 | 0.999401 |
| 65535 | 0.999340 | 0.999098 |
| 131071 | 0.994978 | 0.998944 |
| 262143 | **0.969998** | **0.983820** |

Two things follow. First, fp32 destination accumulation in the SDPA kernels **is** accepted at
`head_dim = 256` — the code's comment claiming it was forbidden was wrong — and it halves the
error, so it is now the default (`sdpa_compute_cfg`). Second, the error does not grow
smoothly: doubling the context from 131071 to 262143 multiplies it by 15, which no
accumulation model explains, so the remaining knobs were swept directly.

### Knobs swept, all with fp32 destination accumulation on

| lever | result | log |
|---|---|---|
| `packer_l1_acc` True vs False | identical 0.983820 | `logs/sdpa_config_sweep.log` |
| compute grid 8x8 vs 7x7 | identical 0.983820 | `logs/sdpa_config_sweep.log` |
| q/k chunk 512, 1024, 2048 (square) | all fail: `TT_THROW program.cpp:1582` (L1) | `logs/sdpa_chunk_sweep.log` |
| asymmetric q=128/k=512, q=256/k=512, q=64/k=1024 | all fail the same way | `logs/sdpa_config_sweep.log` |
| KV block size 64 / 128 / 256 (4096 / 2048 / 1024 page-table entries) | identical 0.983820 | `logs/sdpa_blocksize_sweep.log` |

Identical to six decimals across block sizes rules out a page-table-size effect, and no
k-chunk above 256 fits L1 at `head_dim = 256`, which is what would reduce the number of
online-softmax merge steps.

| fp32 K/V cache, with and without casting q to fp32 | rejected: `TT_FATAL @ ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:43` | `logs/sdpa_dtype_sweep.log` |

So bf16 operands with fp32 destination accumulation is the maximum precision the op offers,
and that is the 0.983820 number. Every lever the TTNN API exposes has been measured.

### Status

This is a TTNN `chunked_scaled_dot_product_attention` accuracy limitation at ~2.6e5 keys with
`head_dim = 256`, not a model-code bug: the inputs and the cache are provably right, and no
API knob changes the result. DRAM is *not* the constraint (31 GiB measured, 1.8 GB used), so
this does not qualify as the hard-physical-device-limit exception that would allow reducing
the advertised context. It was therefore escalated to `$autofix` with the full evidence above,
a pointer at the kernel sources in the built tree, and the instruction that an on-branch
workaround is preferred over a capability reduction.

The largest `full_attention` context that currently meets the PCC >= 0.995 bar is **131071**
(tail PCC 0.998944). `linear_attention` meets it at the full 262143.

## 8. Root cause and fix of the long-context bug ($autofix)

Section 7's conclusion — "a TTNN accuracy *limitation* that no API knob changes" — was wrong on
both counts. The mechanism is specific, it is a plain bfloat16 accumulation defect, and one API
knob does change it. Two of section 7's premises were also wrong and are corrected below.

### The mechanism

`chunked_scaled_dot_product_attention` keeps its flash-attention state — running max, running
softmax denominator, running output accumulator — in `Float16_b` **regardless of
`fp32_dest_acc_en`**. In the built tree,
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:657`:

```cpp
tt::DataFormat im_df = tt::DataFormat::Float16_b;  // need to disable fp32 cbs (Issue #13364) fp32_dest_acc_en ?
                                                   // tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
tt::DataFormat stats_df = im_df;
```

`fp32_dest_acc_en` only widens DST *within* one op; every k chunk the state is packed back out
to a bfloat16 CB (`:757-764`, merged in `compute_common.hpp` `sdpa_inner_loop`).

The denominator is a sum of positive terms. After `m` chunks each new chunk adds about `1/m` of
the running total, so once `1/m` falls below bfloat16's half-ULP (`2**-9`, i.e. `m > 512`) whole
chunks are swallowed by round-to-nearest, the denominator stops growing, and the *normalised*
output comes out uniformly **too large**. The output accumulator does not suffer the same way,
because its terms have mixed signs and it grows like `sqrt(m)` rather than `m`.

### What the number actually was

The device attention output at 262143 tokens is not "noisy", it is **scaled**:

| | 131071 tokens | 262143 tokens |
|---|---|---|
| attention output vs float32 golden, PCC | 0.999529 | 0.999076 |
| ... same tensors, best-fit scale `alpha` | **1.041** | **1.318** |
| ... relative L2 error | 0.052 | 0.323 |
| residual relative error after removing `alpha` | 0.032 | 0.057 |

PCC is scale-invariant, which is exactly why a 32 % error read as 0.9991. The scale survives
`o_proj` and the residual, and that is the 0.9838 layer PCC.

Two corrections to section 7:

* **The paged K/V cache being right did not prove the kernel was wrong in a diffuse way.**
  `probes/probe_sdpa_localise.py` intercepts the last chunk's SDPA call and re-runs float32
  attention on the *device's own* Q and un-paged K/V. It scores 0.999101 — the kernel is fine
  apart from the scale — while the layer scores 0.983820. Pushing the device attention output
  through the rest of the layer in torch (`probes/probe_amplification.py`) reproduces the layer
  PCC to 0.983784, and the device layer output matches that torch continuation at 0.999967. So
  100 % of the layer error is the SDPA scale; nothing downstream contributes.
* **"No k chunk above 256 fits L1" was wrong.** The earlier sweep only tried `q>=128` with
  `k=512`. `q_chunk_size=64, k_chunk_size=512` fits, and the q chunk has *no* effect on the
  result (q 64 and q 256 give bit-identical output).

### The proof that it is the merge count, not the context

`probes/probe_sdpa_synthetic.py` is a model-free reproducer: synthetic Q/K/V rounded to
bfloat16 on the host, so the torch float32 golden and the device see identical inputs
(`logs/sdpa_synthetic_merge_sweep.log`).

| keys | k chunk | k chunks merged | alpha |
|---|---|---|---|
| 262144 | 256 | 1024 | 1.465 |
| 131072 | 128 | 1024 | 1.370 |
| 65536 | 64 | 1024 | 1.286 |
| 262144 | 512 | 512 | 1.039 |
| 131072 | 256 | 512 | 1.023 |

65536 keys in 1024 chunks is worse than 262144 keys in 512. The context length is irrelevant;
the number of sequential bfloat16 merges is everything, and the knee is at 512 as the half-ULP
argument predicts.

### The fix

`tt/functional_decoder.py`: `_sdpa_program_config` now takes the call's KV length and grows the
k chunk until at most `SDPA_MAX_K_CHUNKS = 512` of them are merged, pairing each k chunk with
the largest q chunk that fits L1 (`_SDPA_Q_FOR_K = {256: 256, 512: 64}`). Below 131072 keys
nothing changes, so every previously passing test runs the identical program.

At 262143 tokens (`logs/sdpa_localise_262143_fixed.log`, `logs/long_context_fixed.log`):

| | before | after |
|---|---|---|
| attention output `alpha` | 1.318 | **1.032** |
| attention output relative error | 0.323 | 0.049 |
| SDPA vs float32 on its own inputs | 0.999101 | 0.999368 |
| `full_context_prefill_tail_pcc` | 0.983820 | **0.998817** |

The advertised context is unchanged at 262144.

### Upstream

Two separate tt-metal defects, both worth filing, with `probes/probe_sdpa_synthetic.py` and
`probes/probe_sdpa_decode_synthetic.py` as the reproducers:

1. `sdpa_program_factory.cpp:657` — flash-attention statistics pinned to `Float16_b`. The
   symptom is a *scale* error, invisible to PCC. Issue #13364 is cited as the reason the fp32
   variant is commented out. Related: `packer_l1_acc` from the compute config is destructured at
   `:310` and never forwarded (`:1310-1315`), and `exp_approx_mode` never reaches the dominant
   `exp(QK-max)`, which hard-codes `approx=true` (`compute_common.hpp:301,325`).
2. The decode defect in section 9.

## 9. A second defect the fix uncovered: decode at long positions

With prefill fixed, `test_full_advertised_context[full_attention]` reaches its next assertion
for the first time and fails there:

```
PCCEVIDENCE full_context_decode_pcc  position=262143  0.5502930
```

This is **not** a regression — the decode assertion sits after the prefill assertion that used
to abort the test, so it had never run at this position.

`probes/probe_sdpa_decode_synthetic.py` reproduces it without the model
(`logs/sdpa_decode_synthetic_sweep.log`). `paged_scaled_dot_product_attention_decode` has the
same bfloat16-denominator disease, far worse: with the default program config the output is
**37.7x too large** at position 262143, and already 1.21x at 16383.

Unlike prefill, it cannot be fixed from Python. The lever — `SDPAProgramConfig`'s grid and
`max_cores_per_head_batch`, which decide how many cores split the KV — does cure the long
position (`k_chunk 256`, 16 cores per head: alpha 1.00044 at 262143), but any explicit config
turns out to be correct **only** when

```
num_k_chunks == 1   or   num_k_chunks % (2 * cores_per_head) == 0
```

and `num_k_chunks = ceil((cur_pos+1)/k_chunk)` is a runtime quantity while the program config is
compile-time. Violations are not small errors: position 12287 returns alpha 780440, position
261887 returns NaN. Verified for `max_cores_per_head_batch` 1/2/4/8/16 and k chunk 64/128/256;
independent of cache contents (zeroing the cache past `cur_pos` changes nothing) and of cache
size. `max_cores_per_head_batch=1` is the only setting with no bad band, and it leaves alpha at
2.56 at position 262143 — still failing.

So the honest position: the model's decode config was left alone. `full_attention` decode is
accurate to roughly position 8191 (alpha 0.969) and degrades smoothly past it. Fixing it needs
the same kernel change as section 8 plus a fix to the decode reduction over idle cores.

### Cost of the fix

The k chunk is free; the q chunk it forces is not. One 2048-token prefill chunk attending
262144 keys, warmed, median of 3:

| q chunk | k chunk | time |
|---|---|---|
| 256 | 256 | 262.3 ms |
| 128 | 256 | 260.4 ms |
| 64 | 256 | 514.2 ms |
| 64 | 512 | 514.4 ms |

Every q chunk re-streams the whole KV, so quartering the q chunk doubles the DRAM traffic;
`k_chunk_size` itself costs nothing. Only prefill chunks past 131072 keys select
`q 64 / k 512`, so a full-context prefill pays roughly 1.75x on its SDPA time and nothing at
all below 128k. `q 128 / k 512` would be free, and misses the L1 budget by 7 %
(1684224 B against 1572864 B) — worth raising with the SDPA owners alongside issue #13364.

### Suite state after the fix

`pytest tests/test_functional_decoder.py --long-context` → **56 passed, 1 failed** in 14:13
(`logs/suite_after_sdpa_fix.log`). The single failure is the section 9 decode assertion, which
this stage had never reached before. No test regressed: below 131072 keys the SDPA program
config is byte-for-byte what it was.

## 10. Final state of the stage

Re-run on device 2 (intact p300c board) against the final code — `TRI_INV_BASE = 16`, fp32
destination accumulation on the SDPA kernels, and the `SDPA_MAX_K_CHUNKS` k-chunk cap.

| gate | result | artifact |
|---|---|---|
| functional suite | **56 passed, 1 failed** | `logs/suite_after_sdpa_fix.log` |
| full advertised context (262143) | 1 passed (`linear_attention`), 1 failed (`full_attention` decode) | `logs/long_context_fixed.log` |
| watcher | **5 passed**, log clean, no fatal/assert/sanitize lines | `watcher/WATCHER_AUDIT.md` |
| profiling | 4 Tracy runs, marker-drop-free windows | `perf_summary.json`, `tracy/*/*_perf_report.txt` |
| recorded measurements | 262 records, 258 numeric | `pcc_evidence.json` |

Minimum PCC over every numeric record **except the one open gap**: **0.998817**. Exactly one
measurement in the stage is under the 0.995 bar.

### The one failing gate

`test_full_advertised_context[full_attention]` fails its **decode** assertion at position
262143 with PCC 0.550293. Its prefill assertion, its paged-cache assertions and the whole
`linear_attention` half all pass.

This is not a model-code defect and not a capability reduction. The layer is constructed at
the full 262144, prefills it correctly (0.998817), and DRAM is not the constraint (31 GiB
measured, 1.8 GB used). The cause is the bfloat16 flash-attention statistics in ttnn's SDPA
kernels (§8), which `$autofix` fixed for prefill by capping the merge count and could **not**
fix for decode: `paged_scaled_dot_product_attention_decode` needs a program config whose
validity depends on `ceil((cur_pos+1)/k_chunk)`, a runtime quantity, while the config is
compile-time, and violating it is catastrophic (position 12287 → 780440x scale; 261887 → NaN).
No setting is correct for every position, so there is no Python-level workaround.

Largest `full_attention` decode position verified against the HF reference: **5003**.
`linear_attention` decode is correct at 262143.

### Upstream

Model-free reproducers, which are what an upstream issue needs:

* `probes/probe_sdpa_synthetic.py` — prefill; the scale error tracks the **merge count**, not
  the context (65536 keys in 1024 chunks is worse than 262144 keys in 512).
* `probes/probe_sdpa_decode_synthetic.py` — decode; 37.7x scale at position 262143, already
  1.21x at 16383.

Worth filing: (1) `sdpa_program_factory.cpp:657` bfloat16 flash statistics — the symptom is a
scale error that PCC on the op output cannot see; `packer_l1_acc` is also destructured at
`:310` and never forwarded, and `exp_approx_mode` never reaches the dominant `exp(QK-max)`,
which hard-codes `approx=true` (`compute_common.hpp:301,325`); (2) the decode reduction over
idle cores.
