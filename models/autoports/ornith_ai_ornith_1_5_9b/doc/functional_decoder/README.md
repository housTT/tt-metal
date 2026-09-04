# Ornith-1.5-9B functional decoder

**Functional-decoder stage complete. Independent stage review: `clean-pass`.**

Target `ornith-ai/Ornith-1.5-9B` revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, Transformers 5.12.1,
source-built tt-metal based on `e7638d2859b6a1ef30eb984781cbddf9872a8d62`.
Single 1×1 Blackhole mesh: one chip on physical **P300c** boards. The profile name
`p150` denotes one chip; these are not measurements on a P150 board.

## Decoder contract

`tt/functional_decoder.py::FunctionalDecoder` subclasses `LightweightModule`.
Shapes derive from the exact checkpoint text config: hidden 4096, MLP 12288,
32 layers (24 linear-attention, 8 full-attention). Representative layers 0 and 3
exercise the two kinds with actual checkpoint weights. Both apply zero-centered
RMSNorm, mixer, residual, zero-centered RMSNorm, dense SwiGLU and residual.

- `from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device,
  max_context=None, page_block_size=64, prefill_chunk=2048, dtype=ttnn.bfloat16)`
  loads module-relative HF keys. Projection weights are transposed; residual/QK
  norm weights have the HF `+1` folded at setup. Unknown kwargs fail directly.
- `allocate_state(batch_size)` creates persistent per-batch state, before forward
  or capture. `allocate_kv_cache(num_blocks, dtype=ttnn.bfloat16)` or
  `attach_kv_cache(k_cache, v_cache)` binds full-attention paged state.
- `prefill_forward(x, *, start_pos=0, page_table=None, chunk_size=None)` accepts
  BF16/TILE/DRAM `[B,T,4096]`, any positive logical T with `start_pos+T` within
  configured context. Internal chunks pad to 128 and outputs trim to logical T.
  Non-aligned continuation consumes the first partial 128-token block with device
  decode operations; remaining blocks use chunked SDPA. All users share T/start.
- `decode_forward(x, *, current_pos=None, rot_idxs=None, page_table=None)` accepts
  `[B,1,4096]`. Full attention needs device INT32 ROW_MAJOR `[B]` current positions,
  UINT32 ROW_MAJOR `[1,B]` RoPE indices (same positions), and INT32 ROW_MAJOR
  `[B,num_blocks_per_user]` page table. Positions name the slot written by this
  token. Callers validate position bounds before submitting a trace; forward
  does not read positions back to the host. Returns `[B,1,4096]`.

Full attention uses 16 query heads, 4 KV heads, head width 256, per-head Q/K
zero-centered RMSNorm, partial RoPE on 64 dimensions, sigmoid output gating,
and paged causal attention. Cache shape `[physical_pages,4,64,256]`; page table
rows route batch lanes into disjoint physical pages. `num_blocks_for_context`
rounds page-table width to 32 entries, covering the padded SDPA read window.

DeltaNet uses 16 key heads / 32 value heads, dimensions 128/128, convolution width 4,
and standard gated RMSNorm at the mixer output. Its state is FP32
`[B,32,128,128]` plus three BF16 convolution-history buffers `[B,1,8192]`.
The recurrent state carries across prefill/decode and is updated in place.
Padding has beta=0 and decay exponent=0 so it cannot advance recurrent state;
convolution history uses the last real tokens. Positions/page tables are unused
for this kind. A batch lane owns its recurrent history for the request lifetime.
`reset_state()` zeros it in place for a new request. Request scheduling is outside
this layer API; batch lanes must not be silently reassigned without state reset.

All weights/constants/state are prepared before measured forwards. Runtime uses
TTNN operations only. The HF oracle and explicit input/output transfer helpers
are test boundaries. The trace harness captures stable input/state addresses;
replay refreshes input contents outside capture. No generation or serving is
implemented in this stage.

## Context and optional YaRN

Native target is 262144 tokens for both kinds. No capability reduction is selected.
[context_contract.json](../context_contract.json) records 262144 tested prefill and
decode context, including decode position 262143.
Optional HF config `rope_type=yarn`, factor1000000/262144,
original_max_position_embeddings262144, max_position_embeddings1000000 is
accepted and uses HF's setup-time frequency/attention-factor initializer. The
million-position table parity test does not establish a million-token full-layer
prefill or serving capability; native-context validation remains the stage gate.

## Evidence and reproduction

See [work_log.md](work_log.md) for commands, failures, investigations, and results.
`hf_config.json` is the pinned config; `weight_stats_layer{0,3}.json` records exact
checkpoint tensor names/shapes/dtypes/moments. Tests default to deterministic
synthetic weights generated from these statistics, with the same full shapes.
`ORNITH_WEIGHTS=real` selects pinned local checkpoint weights. `embedding_stats.json`
records the measured embedding scale used for input construction.

Environment for commands in this container:

```bash
source python_env/bin/activate
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8
export ORNITH_WEIGHTS=real
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests -m 'not long' -v -s
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests -m long -v -s
```

Hardware jobs must run serially. Watcher and Tracy must use separate processes.


## Correctness and capability evidence

Acceptance is PCC >= 0.995 without a model-specific exception. The final real-weight
input distribution uses the measured checkpoint embedding standard deviation
0.0142854163. HF runs in FP32; TT activations/weights are BF16 with FP32 DeltaNet
state and HiFi4 projection matmuls. Full-attention decode SDPA uses HiFi2. The
HF CPU fallback warning refers to the reference implementation only.

| Real-weight comparison | Linear attention (layer 0) | Full attention (layer 3) | Artifact |
| --- | ---: | ---: | --- |
| Prefill, 300 tokens | 0.999366 | 0.999459 | `logs/final_functional.log` |
| Decode after that prefill | 0.999433 | 0.998906 | `logs/final_functional.log` |
| Prefill, 8001 tokens | 0.999393 | 0.999530 | `logs/final_long.log` |
| Decode after 8001 tokens | 0.999319 | 0.999734 | `logs/final_long.log` |
| Traced decode, native final position, exact-cache HF oracle | N/A | 0.99965012 | `logs/final_long.log` |
| Unaligned continuation through final allocated page, 4096 tokens | N/A | 0.99951140 | `logs/continuation_capacity_after.log` |

`logs/final_functional.log`: **76 passed**, 9 long cases deselected. This includes
synthetic-weight comparisons, real-weight comparisons, B=1/4/32, ragged positions,
shuffled and nonzero physical pages, changed inputs/positions during trace replay,
repeated-input determinism, and guarded forwards. `logs/final_long.log`: **9 passed**,
76 shorter cases deselected. `logs/continuation_capacity_after.log`: **11 passed**
after the final continuation alignment repair, covering both layer kinds and the
expanded runtime guard. The last repair changes only nonzero-start prefill; the
native fresh-prefill and decode paths measured by the long suite are unchanged.

| Capability claim | Evidence | Remaining scope or risk |
| --- | --- | --- |
| Both target layer kinds | Real layer 0/3 weights, exact target shapes, all suites above | Other layers share these architectures; each layer's individual weights are not separately tested |
| Native 262144-token prefill | Both kinds run full decoder at 262143 and 262144, with finite/nonconstant outputs; 2048 vs 1024 chunk tail PCC 1.000000 | Full-length HF prefill is not computed; HF parity is measured through 8001 tokens |
| Native final-position decode | Both kinds replay after actual 262143-token prefill, eager/replay PCC 1.0; full attention additionally compares against HF with exact paged history at position 262143 | Exact-cache fixture is synthetic history with real projection weights |
| Arbitrary valid prompt lengths | 1, 7, 32, 64, 128, 129, 250, 2048, 2049, 3000, 8001, 262143; continuation splits 63/65/128/129 and split 63 through capacity | Shared prefill length/start across batch lanes |
| Paged cache and device positions | Shuffled/nonzero page slots, B=4/13 ragged decode, B=32 prefill/decode, changed table contents in one trace | Caller must supply disjoint valid physical pages and in-range position tensors |
| Traced decode | B=1/4/32, three changed-input steps, per-user HF checks; stable input/state buffers refreshed outside replay | Caller updates stable device buffers outside capture |
| Determinism | Three bit-identical eager runs and three bit-identical trace runs after restoring identical state; repeated page-table request is bit-identical | A recurrent state must be reset/restored before comparing repeated requests |
| No host fallback | TorchFunctionMode and patched transfer APIs forbid Torch/from_torch/to_torch/as_tensor/from_device/to_device/copy_host_to_device_tensor in fresh prefill, continuation, decode; positive controls prove guards fire | Setup/reference/explicit test input-output boundaries use Torch |
| Optional YaRN | HF setup frequencies and million-position cos/sin table parity > 0.99999 | Million-token full-layer capacity and accuracy remain unvalidated; no million-token layer capability is claimed |

The original native oracle failure at PCC 0.94535 is preserved and investigated in
[AUTOFIX_long_decode.md](AUTOFIX_long_decode.md). Exact original-scale reruns pass
under normal execution, watcher and cold JIT; exact-cache controls pass. No numerical
fix is claimed for that unreproduced historical result. Any recurrence requires a
new failing-output/cache capture. The ramp alias and end-of-cache continuation
failures have isolated before/after evidence and retained fixes.

## Warmed performance

One Blackhole chip on physical P300c boards, batch 1, real weights. These are
functional-stage baselines; no optimization or generation-throughput claim.

| Layer kind | Prefill 2048 tokens, kernel sum | Traced decode, kernel sum per replay | Measured prefill/decode HF PCC |
| --- | ---: | ---: | --- |
| Linear attention | 40.076 ms | 1.610 ms | .99948959 / .99980134 |
| Full attention | 35.104 ms | 1.422 ms | .99949539 / .99902570 |

The metric sums `Device Time` in **microseconds** from filtered tt-perf-report
1.2.9 CSVs and divides by 1000; decode additionally divides by four replays.
Prefill has two warmups; decode has compile/capture and four warm replays before
measurement. Decode uses a 128-token prompt and position 128, holding full-attention
position fixed for repeatable overwrites. DeltaNet advances its state on each replay
and is checked against four corresponding HF recurrent steps. Unprofiled 32-replay
wall measurements remain in `logs/final_functional.log`; the table above excludes
host/dispatch gaps and uses complete kernel records.

Human-readable tables and CSVs:

- [Linear prefill](tracy/linear_attention/prefill_perf_report.txt), [CSV](tracy/linear_attention/prefill_perf_report.csv)
- [Linear traced decode](tracy/linear_attention/decode_perf_report.txt), [CSV](tracy/linear_attention/decode_perf_report.csv)
- [Full prefill](tracy/full_attention/prefill_perf_report.txt), [CSV](tracy/full_attention/prefill_perf_report.csv)
- [Full traced decode](tracy/full_attention/decode_perf_report.txt), [CSV](tracy/full_attention/decode_perf_report.csv)

[performance.json](performance.json) records metrics, SHA256 and coverage audit:
all four measured decode sessions contain exactly 78 linear or 69 full-attention
ops, matching the warm trace template. `tools/summarize_perf.py` checks this and
cross-checks report microseconds against raw nanoseconds. Raw ops CSVs are preserved
losslessly as `tracy/<kind>/<mode>_ops.csv.gz`; reporting argv is in neighboring
`<mode>_provenance.json`. The earlier 32-replay profiler captures dropped markers;
`rejected_v2/` and `performance_v2_rejected.json` retain those explicitly invalid
results. Final decode uses `--tracing-mode` to preserve replay ordering.

## Watcher, archives and review

`TT_METAL_WATCHER=10`, without the profiler: `logs/watcher_final.log` records
**23 passed** in 169.82 s, including both kinds, B=1/4/32 traced decode, native
exact-cache decode, continuation through capacity, changed page tables, determinism
and host-fallback guards. [watcher_audit.json](watcher_audit.json) records no suspicious
watcher messages and hashes of lossless watcher artifacts. Pytest reopens the device
between parameterized cases; the generated watcher.log retains the last fixture,
while suite stdout records all periodic checks and test outcomes. The separate
native original-scale watcher control is retained too. Both task-owned Tracy GUI
processes were stopped after collection; there is no serving process for this stage.

To preserve exact bytes and avoid the repository's 500 KB text-file limit, logs and
raw CSVs are committed compressed. Local uncompressed originals remain available.
After a fresh checkout, restore each adjacent original with:

```bash
python - <<'PYTHON'
from pathlib import Path
import gzip
root = Path('models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder')
for archive in root.rglob('*.gz'):
    archive.with_suffix('').write_bytes(gzip.decompress(archive.read_bytes()))
PYTHON
```

Use `watcher_audit.json` to map archived watcher filenames back to their original
`generated/watcher/` paths if needed. Full Tracy binary captures and device CSVs
remain under ignored `tracy/<kind>/raw/`; the committed lossless ops CSVs are sufficient
to reproduce the rendered reports. Independent verdict: [STAGE_REVIEW.md](STAGE_REVIEW.md).
