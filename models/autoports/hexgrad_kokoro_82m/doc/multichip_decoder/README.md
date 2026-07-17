# Kokoro-82M — Multichip Decoder (TTNN, TP=4)

Stage 03 (multichip-decoder) for `hexgrad/Kokoro-82M` on **4× Blackhole p300c**
(`ClusterType.P300_X2`, a physical 4-ring exposed as a `(1, 4)` mesh). Implements
`models/autoports/hexgrad_kokoro_82m/tt/multichip_decoder.py`, the tensor-parallel
counterpart of the stage-02 `tt/optimized_decoder.py`.

Single-chip baseline: **`OptimizedDecoder`** (stage 02) — packed-QKV +
`nlp_create_qkv_heads` + fused SDPA + `nlp_concat_heads`, fused-gelu FF1, explicit
`ffn_output` program config, precision policy bf16 act / BFP8 weights / HiFi2 /
fp32-dest-acc. `PrecisionPolicy` and the op-builder helpers are **imported** from
the optimized decoder so the two stages cannot numerically drift.

Target component (unchanged from stages 01/02): Kokoro's only attention-transformer,
`plbert` = HF `AlbertModel` — 12 weight-tied `AlbertLayer`s, hidden 768, 12 heads,
head_dim 64, intermediate 2048, embedding 128→768, vocab 178, max ctx 512.
**Bidirectional, non-autoregressive, no KV cache, no MoE.**

## Target mesh & why this strategy

4 chips → 1D tensor parallelism `TP=4` (multichip skill default for ≤8-chip 1D
meshes). The chips form a physical 4-ring, so `FABRIC_1D_RING` + `Topology.Ring`
(all 4 links) is used — measured faster than Linear at max context (2.56 vs
2.71 ms @T=512, like-for-like under the final config; `sweeps/topology.log`).

Kokoro's plbert is *tiny* (~30 M-param encoder, already 2.95 ms on one chip) and
**the only token-mixing op is SDPA** — embedding, both LayerNorms, the residual
adds and the entire FFN are per-token. That shapes an unusual-but-optimal hybrid:

| Sub-block | Parallelism | Weights | Collective |
|---|---|---|---|
| Embedding + map | sequence-parallel | replicated | none |
| Attention (QKV/SDPA/WO) | **head-parallel TP** (3 heads/device) | **fractured** | 1 all_gather (residual) + 1 reduce_scatter (WO) |
| FFN (FF1/gelu/FF2) | sequence-parallel | replicated | none |
| LayerNorm ×2, residual adds | sequence-parallel (local) | replicated | none |

**Residual stream is sequence-sharded** `[b, 1, S/TP, H]` — this is the layer
input *and* output contract, so decoders stack with no boundary reshard. Only the
attention block needs the full sequence, so each layer `all_gather`s the residual
to full sequence, runs local-head QKV/SDPA/WO, then `reduce_scatter`s the WO
partial back to the shard (reduce over head-groups + scatter over sequence in one
op). Everything else runs locally on the 1/TP sequence shard → **exactly 1
all_gather + 1 reduce_scatter per layer**, and every op is 1/TP the work.

Row-parallel bias correctness: the WO (`dense`) bias is added *once* to the
already-reduced sharded result after `reduce_scatter` (adding it inside the
partial WO matmul would sum it TP times). Column-parallel QKV / FF1 biases are
naturally sharded and applied in-matmul.

## Per-device tensor / shard table (TP=4)

H=768, heads 12→**3 local**, head_dim 64, local attn width `gw`=192, intermediate 2048.

| Tensor | Single-chip shape | Mesh placement | Per-device shape | Padding |
|---|---|---|---|---|
| word/pos/type emb, LN, `map_w/b` | as stage 02 | **replicate** | same | — |
| `qkv_w` (packed, reordered) | [768, 3·768=2304] | **col-shard** (by head-group) | [768, 3·192=**576**] | — |
| `qkv_b` | [2304] | col-shard | [**576**] | — |
| `dense_w` (WO) | [768, 768] | **row-shard** (concat-heads dim) | [**192**, 768] | — |
| `dense_b` | [768] | replicate (added once post-RS) | [768] | — |
| attn LN, full LN | [768] | replicate | [768] | — |
| `ffn_w` (FF1) | [768, 2048] | **replicate** | [768, 2048] | — |
| `ffn_out_w` (FF2) | [2048, 768] | **replicate** | [2048, 768] | — |
| Activation / residual | [b, S, 768] | **seq-shard (dim=seq)** | [b, 1, **S/4**, 768] | S→mult. of TP·TILE=128 |
| SDPA mask | [b,1,S,S] | replicate (attn on gathered full seq) | [b,1,S,S] | built only when padded/masked |

QKV column reorder: the packed `[Q|K|V]` output columns are reindexed to
`[Q0K0V0 | Q1K1V1 | …]` (head-group interleave) so a contiguous shard-by-TP gives
device *c* exactly its 3 local heads of Q, K and V, which
`nlp_create_qkv_heads(num_heads=3, num_kv_heads=3)` slices in one fused op.

Sequence padding keeps the public contract length-agnostic: any logical length
1..512 is accepted; internally padded to a multiple of `TP·TILE = 128` so each
shard is tile-aligned (S/4 ∈ {32,64,96,128}); padded keys are masked in SDPA and
the output is sliced to the logical length at the model boundary. **No
aligned-only public contract.**

## Rejected / compared alternatives (collective-topology table)

| Candidate | Collectives/layer | Residual | Result | Verdict |
|---|---|---|---|---|
| **Selected: head-TP attn + seq-par FFN, seq-sharded residual** | 1 AG + 1 RS | sharded | **2.56 ms @512** | ✅ |
| Data-parallel + KV all-gather (replicate all weights) | 1 AG (KV) | sharded | 4.05 ms @512 (measured, `/tmp/dp_probe`) | ❌ slower: `[b,24,S/4,64]` KV gather has tiny per-head chunks + 12-head SDPA/device |
| Intermediate-TP FFN (fracture FFN weights) | +1 AG +1 RS | sharded | same FLOPs/device, +2 collectives, 0 memory win | ❌ strictly more collectives, no benefit |
| Replicated residual + all_reduce | 2 all_reduce | replicated | composite all_reduce 172 µs vs RS 67+AG 67; norms not sharded | ❌ costlier collective + full-seq norms |
| Linear topology (FABRIC_1D) | 1 AG + 1 RS | sharded | 2.71 ms @512 (final cfg, `sweeps/topology.log`) | ❌ slower than Ring at max ctx (uses 3 of 4 links) |

## Result summary (warmed, real weights)

Baseline = stage-02 `OptimizedDecoder` on **one** p300c (fresh measurement, same run).

| Metric (T=512) | Single-chip | Multichip TP=4 | Speedup | Efficiency |
|---|---|---|---|---|
| Traced warmed decode | 2.95 ms | **2.56 ms** | **1.15×** | 29 % |
| Traced warmed decode (T=128) | 2.46 ms | **1.95 ms** | **1.26×** | 31 % |
| Worst PCC vs single-chip TTNN | — | **0.99803** | — | — |
| Worst PCC vs HF | 0.9977 | **0.99686** | — | — |

Eager (untraced) prefill on the mesh is host-dispatch-bound by the async-CCL calls
(5.6 ms) and is **not** the production path; the traced decode is. All numbers in
`perf_summary.json` / `pcc_results.json`.

### Why the speedup is modest — and why this is the best practical result

`tt-perf-report` (traced decode T=512, `tracy/decode_perf_report.txt`, merged
4-device, per replay): **CCL 24 %** (ReduceScatter 14 % + AllGather 10 %),
LayerNorm 19 %, Matmul 33 %, SDPA 13 %, head TMs 6 %. **Modeled DRAM roofline
6.9 %.**

The model is already movement/launch-bound on **one** chip (stage-02 roofline
12.8 %). Splitting the work 4 ways makes each op *smaller*, so per-op fixed launch
overhead — which does not shrink — dominates and the ops scale **sub-linearly**
(e.g. per-device Matmul only drops 1.3→0.95 ms, not 4×). Combined with the ~24 %
fixed collective cost (2 collectives/layer × 12), the achievable speedup is
~1.15–1.26×. This is inherent to a ~30 M-param encoder, not an implementation
defect: everything is already 1/TP work, collectives are minimized to the
attention-only floor (1 AG + 1 RS), the physical ring is used, and the dominant
FF1 matmul was given an explicit config (43→33 µs; `in0_block_w` 8/12/24 swept, 8
fastest). The alternatives above were measured and are slower. The value of this
stage is a **correct, stackable TP layer contract** for the full-model stack, not
a large single-utterance speedup.

The profiler's per-matmul advice "place input 0 in L1 (currently DRAM_INTERLEAVED)"
was **considered and not adopted**: the model is launch/movement-bound, not
DRAM-bandwidth-bound (roofline 6.9%, matmul DRAM util 12–37%), so L1 residency
would not lift the bottleneck; and DRAM-interleaved + 2D program configs is the
stage-02-validated choice for these prefill-shaped (16-M-tile) shapes, where L1
height-sharding hits the same ≤16-tile alignment limits documented in stage-02
`advice_disposition.json`. The dominant matmul geometry was swept under the final
BFP8/HiFi2 policy (FF1 `in0_block_w` 8/12/24 → 8; `sweeps/RESULTS.md`); the
profiler marks the selected FF1/FF2/QKV/WO configs "look good 🤷" and explicitly
"not FLOP-bound".

## Correctness / capability (preserved)

- **PCC vs single-chip TTNN ≥ 0.998** and **vs HF ≥ 0.995** on every meaningful
  length: prefill T∈{8,31,32,96,128,256,500,511,512}, traced decode
  T∈{32,64,128,500,511,512} (incl. non-tile-aligned), batch {2,4,8,32}, real IPA
  sentences, masked variable-length batches. Worst vs-single-chip 0.99803, vs-HF
  0.99686.
- **Per-layer-kind component PCC** vs single-chip: embedding and one full
  `AlbertLayer` (attention + FFN sublayers + both collectives) validated
  separately (`test_component_pcc_vs_single_chip`).
- **Non-aligned logical lengths supported** — public API takes any length 1..512;
  internal 128-multiple padding + masking; sliced at the boundary.
- **Stateless (`decode == prefill`)**, **determinism**, **repeated traced replay**
  across revisited shapes (bit-identical) — all validated on the mesh.
- **Warmed trace replay for decode on the target mesh** — `decode_forward` captures
  and replays a per-shape trace; perf measured via `execute_trace`.
- **Watcher clean** (`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`): 0
  fault/assert/tripped/overflow/corruption markers, min stack 1348 B free
  (`watcher/watcher.log`). ETH cores are excluded because instrumenting the
  fabric ETH kernels overflows the ACTIVE_ETH kernel-config buffer (27920 > 25600)
  — a watcher/fabric infra limit, not model code; the CCL kernels are the standard
  upstream `all_gather_async` / `reduce_scatter_minimal_async` ops.
- **Fallback audit clean** (`fallback_audit.txt`): forward/trace path is pure TTNN;
  torch only in `from_state_dict` (setup) + `prepare_inputs` (host input).
- **Context contract**: advertised = supported = 512, no reduction; weights fit on
  each device with a huge margin (`../context_contract.json → multichip_decoder`).
- KV cache / paged cache / current-position / MoE: **N/A** (non-autoregressive
  bidirectional encoder), same as stages 01/02.

## Reproduce

```bash
ENV="TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal"

# Correctness suite (32 tests, opens the (1,4) ring mesh):
env $ENV python -m pytest models/autoports/hexgrad_kokoro_82m/tests/test_multichip_decoder.py -v

# PCC + single-chip-vs-multichip latency / speedup / efficiency:
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/multichip_decoder/gen_evidence.py

# Profiling (one window/run, KOKORO_PERF_ITERS=2):
env $ENV KOKORO_PERF_ITERS=2 python -m tracy -r -p -v -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_perf_multichip.py -k decode   # or prefill
# then tt-perf-report --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END

# Watcher (separate from profiler; ETH excluded — see above):
env $ENV TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 python -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_multichip_decoder.py -k "512 or determinism or stress or masked or component or stateless"
```

## Artifacts

- `pcc_results.json`, `perf_summary.json`, `logs/gen_evidence.log`
- `sweeps/RESULTS.md` (topology, FF1 geometry, DP-KV/collective rejections) +
  preserved probe scripts `sweeps/probe_*.py`, fresh logs `sweeps/topology.log`,
  `sweeps/ffn_in_matmul.log`
- `tracy/{decode,prefill}_perf_report.txt` (human tables), `*_perf_report.csv`,
  `*_perf_report_stacked.csv.{csv,png}`. Raw `*_ops.csv` are provenance-only and
  exceed the 500 KB git gate (regenerate via the tracy command above).
- `watcher/watcher.log` (clean), `watcher/watcher_run.log`,
  `watcher/eth_watcher_overflow.log` (the ACTIVE_ETH config-buffer overflow that
  forces `TT_METAL_WATCHER_DISABLE_ETH=1`), `fallback_audit.txt`
- `work_log.md`

## Limitations / deviations

- Single-utterance latency speedup is modest (~1.15–1.26×) and efficiency ~30 %,
  fundamentally because the encoder is tiny and launch/movement-bound; documented
  and evidence-backed above (not a defect).
- For **throughput** serving (many utterances), pure batch/data parallelism — one
  utterance per device, zero collectives — would give near-4× throughput; that is
  a serving-stage concern, out of scope for this decoder layer-stack stage.
- FFN weights are replicated (per-token sequence-parallel). For this model that is
  optimal (identical FLOPs/device, zero collective); intermediate-TP would only add
  collectives. If a future variant grows the FFN enough that replication does not
  fit, switch FFN to intermediate-TP (the residual contract already supports it).
