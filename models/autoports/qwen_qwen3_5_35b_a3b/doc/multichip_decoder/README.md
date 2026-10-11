# Qwen3.5-35B-A3B multichip decoder: stage 04

- Checkpoint: `Qwen/Qwen3.5-35B-A3B`, revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- Hardware: four Blackhole ASICs on two P300c cards; Ethernet ring 0-1-2-3-0, two links per pair.
- Implementation: `tt/multichip_decoder.py` (`MultichipDecoder`), on top of the Stage 3 `OptimizedDecoder`.
- Single-chip baseline: Stage 3 `OptimizedDecoder` defaults (commit `81485827`), measured again in this stage.
- Profiles: p150 = 1x1 mesh, p150x2 = 1x2 mesh, p150x4 = 1x4 ring. Optimization target: p150x4.
- Final evidence: `evidence7/` (chain `runs/main/stage04/evidence_chain7.sh`, log
  `runs/main/stage04/evidence_chain7.log`, start and end source manifests equal).
- Superseded chains, kept: `evidence1/` (before review 1, `stage_review.md`), `evidence2/` (before the decode
  projection sweep), `evidence3/` (prefill shared-expert regression), `evidence4/` (before the MoE group split,
  review 4), `evidence5/` (before the split rule fix, review 5), `evidence6/` (L1 clash in one row, fixed).
  History: `work_log.md`.

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated
DeltaNet, the linear-attention layer), KV (key/value cache), CCL (collective communication),
PCC (Pearson correlation coefficient), HF (Hugging Face reference), BF16/FP32 (16/32-bit float),
BFP8/BFP4 (8/4-bit block float), DRAM (device memory), L1 (core SRAM), RM (row-major),
SDPA (scaled dot-product attention), TTFT (time to first token).

## Open the meshes

| Profile | Open | Device IDs (logical = PCIe) |
|---|---|---|
| p150 | `open_mesh_device(MeshShape(1,1), physical_device_ids=[3])` | 3 |
| p150x2 | `TT_VISIBLE_DEVICES=2,3`, `FABRIC_1D`, `MeshShape(1,2)` | PCIe 2,3 (one card); artifacts record the renumbered logical IDs [1, 0] |
| p150x4 | `FABRIC_1D_RING`, `MeshShape(1,4)` | 3,2,1,0 |

- A bare `MeshShape(1,2)` on the four-chip host fails fabric router sync. `TT_VISIBLE_DEVICES` is required;
  the full model and the server must set it for p150x2.
- The p150x4 ring uses all four chips, so no alternative placement exists. It includes chip 0, which Stage 3 measured
  16-23 us slower per traced decode replay.
- Harness: `tests/mesh_util.py::open_mesh`; `run_layer.py --decoder multichip --mesh N`.

## Mesh plan

### Strategy

| Part | Strategy | Collective |
|---|---|---|
| Residual stream | replicated [B,T,2048] BF16 on every device | none |
| Norms, router, shared-expert gate | replicated | none |
| Full attention | TP by heads: 16/tp query heads, KV head `rank*2//tp` | all-reduce of o_proj partials |
| GDN | TP by heads: 16/tp key heads, 32/tp value heads, local state | all-reduce of out_proj partials |
| Routed experts, decode | expert-TP: every expert, intermediate 512/tp, indexed `sparse_matmul` | shares the MoE all-reduce |
| Routed experts, prefill | EP: 256/tp whole experts per device, local dispatch | shares the MoE all-reduce |
| Shared expert | TP: intermediate 512/tp | shares the MoE all-reduce |

- Two all-reduces per layer: after attention and after MoE (routed + shared partials summed locally).
- `ttnn.all_reduce`: reduce-scatter + all-gather; outputs are identical on all devices (checked every run).
- Decoder input and output contract: replicated [B,T,2048] BF16, TILE. Layers stack without conversion.

### Per-device shapes (tp = 1 / 2 / 4)

| Tensor | Single chip | Per device, tp=2 | Per device, tp=4 | Mapping, padding |
|---|---|---|---|---|
| Full attention qkvg (BFP8) | 2048 x 9216 | 2048 x 4608 | 2048 x 2560 | q/gate heads split; K/V head 1 per device (shared by 2 devices at tp=4) |
| Full attention o_proj (BFP8) | 4096 x 2048 | 2048 x 2048 | 1024 x 2048 | rows by local heads |
| KV cache page (BFP8) | [P,2,page,256] | [P,1,page,256] | [P,1,page,256] | local KV head |
| GDN in-projection (BF16) | 2048 x 12352 | 2048 x 6208 | 2048 x 3136 | [q k v a b z] by heads; a and b padded 8/16 -> 32 columns each |
| GDN out_proj (BFP8) | 4096 x 2048 | 2048 x 2048 | 1024 x 2048 | rows by local value heads |
| GDN recurrent state (FP32) | [1,32,128,128] | [1,16,128,128] | [1,8,128,128] | local value heads, 32 configured slots |
| GDN conv state | 3 x 8192 | 3 x 4096 | 3 x 2048 | local q/k/v channels |
| Decode experts gate/up (BFP4) | [1,256,2048,1024] | [1,256,2048,512] | [1,256,2048,256] | [up | gate] halves, intermediate slice |
| Decode experts down (BFP8) | [1,256,512,2048] | [1,256,256,2048] | [1,256,128,2048] | intermediate slice |
| Prefill experts gate/up (BFP4) | shares decode weights | [1,128,2048,1024] | [1,64,2048,1024] | stacked, experts `rank*256/tp ..` |
| Prefill experts down (BFP8) | shares decode weights | [1,128,512,2048] | [1,64,512,2048] | stacked |
| Shared expert gate/up (BFP4) | 2048 x 1024 | 2048 x 512 | 2048 x 256 | interleaved 32-column tiles |
| Shared expert down (BFP8) | 512 x 2048 | 256 x 2048 | 128 x 2048 | DRAM-sharded on tp=1; 1D multicast on tp=2/4 |
| Router + shared gate (BF16) | 2048 x 288 | same | same | replicated |
| RoPE tables (BF16 RM) | 262144 x 64 x 2 | same | same | replicated |

Program configs that change with tp (`tt/multichip_decoder.py::MESH_POLICIES`):

| Op | tp=1 | tp=2 | tp=4 |
|---|---|---|---|
| Decode gate/up `sparse_matmul` | 32 cores, k 32 | 16 cores, k 64 | 8 cores, k 64 |
| Decode down `sparse_matmul` | 32 cores, k 16 | 64 cores, k 8 | 64 cores, k 4 |
| Shared down | DRAM-sharded, 8 cores, k 8 | 1D, 16 cores, k 8 | 1D, 8 cores, k 4 |
| Shared gate/up `minimal_matmul` (SwiGLU fused) | default config | grid 8x2, K block 16, N block 4 | grid 4x2, K block 16, N block 4 |
| Full attention qkvg decode | default config | 1D, 16 cores, k 8 | 1D, 40 cores, k 8 |
| GDN in-projection decode | default config | default config | 1D, 20 cores, k 2 |
| Prefill dispatch weights | stacked decode weights | stacked EP | stacked EP |
| Prefill MoE group (`prefill_moe_pieces`) | pad to next bucket | bucket pieces, at most 2 (last may be padded) | bucket pieces, at most 2 (last may be padded) |
| Fused routed reduce | `cluster_axis=0` | `cluster_axis=1` | `cluster_axis=1` |

tp=1 keeps the Stage 3 configs, so p150 stays bitwise equal to the baseline.
Unchanged: attention_out decode grid 8x4 (N=2048 stays 64 tiles), prefill minimal-matmul configs,
decode norms (hidden 2048), dispatch grids (4x4, 8x8 from 2048 rows).

### Construction

- Each rank builds the Stage 3 decoder from its tensor slice (`tensor_parallel_state`) with a local
  config (`local_hf_config`), replicated on the mesh.
- `merge_replicas` walks the replicas in lockstep. Tensors equal on all ranks stay replicated; others
  become one mesh tensor from per-device host shards (`ttnn.from_host_shards`). No float round trip.
- Expert-parallel dispatch: routes outside the local range map to one dummy slot. Its restore row is
  written as zero (`slice_write`), so non-local routes add exactly zero. `padding_idx` in
  `ttnn.embedding` does not zero rows (`runs/main/stage04/micro/embedding_padding.py`).

### Decode expert geometry (`evidence7/micro/sparse_geometry_sweep.json`, gate and down swept independently)

| Mesh | Op | Best | Next | Previous selection |
|---|---|---|---|---|
| x4 | gate/up N=256 | 8 cores, k 64: 37.4 us | 8 cores, k 32: 41.4 us | same |
| x4 | down N=2048 | 64 cores, k 4: 15.8 us | 32 cores, k 4: 18.7 us | 16 cores, k 4: 25.9 us |
| x2 | gate/up N=512 | 16 cores, k 64: 42.1 us | 16 cores, k 32: 45.3 us | 16 cores, k 16: 56.6 us |
| x2 | down N=2048 | 64 cores, k 8: 21.1 us | 32 cores, k 8: 24.2 us | 64 cores, k 4: 25.6 us |

### Decode projection geometry (`evidence7/micro/projection_geometry_sweep.json`, selected dtype and fidelity)

Single chip, per-device shape, traced, median per call. Max abs difference vs the default config is 0 for every
selected row.

| Mesh | Op (K x N, dtype, fidelity) | Previous | Selected | Candidates |
|---|---|---:|---:|---:|
| x4 | qkvg 2048 x 2560, BFP8, LoFi | 43.1 us (default) | 16.4 us | 46 |
| x4 | GDN in 2048 x 3136, BF16, HiFi4 | 48.8 us (default) | 36.6 us | 31 |
| x2 | qkvg 2048 x 4608, BFP8, LoFi | 41.1 us (default) | 34.1 us | 31 |
| x2 | GDN in 2048 x 6208, BF16, HiFi4 | 64.6 us (default) | 64.6 us (default kept) | 31 |
| x4 | shared gate/up 2048 x 256, BFP4, LoFi | 19.3 us | 11.9 us | 146 |
| x2 | shared gate/up 2048 x 512, BFP4, LoFi | 19.2 us | 13.7 us | 175 |
| x4 | shared down 128 x 2048, BFP8, LoFi | 7.7 us (DRAM-sharded, 4 cores; profile) | 4.7 us | 9 |
| x2 | shared down 256 x 2048, BFP8, LoFi | 11.1 us (DRAM-sharded, 8 cores) | 4.7 us | 15 |
| x4, x2 | attention_out, router | already best within 0.3 us | unchanged | 33-45 |

- Candidates: 1D multicast grids 8-80 cores with every legal k block, DRAM-sharded with 1-2 readers and 8-64
  input cores, minimal-matmul grids and blocks.
- x2 qkvg: DRAM-sharded is 29.9 us but needs a second padded weight copy (9.6 MiB per full layer, prefill uses the
  interleaved copy). Rejected for 4.2 us per full layer.
- Layer screen with the final policy (`evidence7/micro/projection_screen2/`, exploratory): x4 decode full
  380.7 -> 345.9 us, linear 517.3 -> 498.1 us; x2 399.3 -> 373.5 us, 584.0 -> 574.6 us. Prefill and PCC unchanged.
  Final numbers: Results.
- The decode shared gate/up config is `shared_decode_minimal_config`; prefill keeps the default config.

### MoE group split (`evidence7/micro/padding_routes/`)

- The prefill planner groups attention chunks and runs MoE once per group, padded to the next bucket (2176 -> 4096).
- On meshes, EP prefill cost follows the padded row count. Zero padding rows also tie all router logits, so they all
  pick the same 8 experts: rank 0 got 20130 routes vs 3790-4932 on other ranks (2176 tokens, p150x4, full layer).
- Change: padding routes go to the zeroed dummy slot (`valid_rows`), and meshes run a padded group as exact
  bucket pieces when that is cheaper (`_moe_pieces`, at most 2 pieces; the last piece may be padded):
  - split only if the pieces total less than the single bucket (3300 and 3584 would be 2048 + 2048 = 4096: no split);
  - a padded last piece only if the single bucket is the largest (4096): 2400 = 2048 + 512, 2600 = 2048 + 1024;
    1300 stays one 2048-row group.
- Masking alone: rank 0 down to 4770 routes, time unchanged (20.1 ms). The split gives the gain.
- Limit 1 (pad) against limit 2 (pieces), medians of 6 warm runs, ms (`pieces_x*.json`, chain-5 code; limit 2 here
  is the split without the rule):

| Profile | Layer | 1152 | 2176 | 2560 | 3072 |
|---|---|---|---|---|---|
| p150x4 | full | 12.6 -> 11.1 | 21.3 -> 14.7 | 20.3 -> 16.1 | 20.6 -> 18.4 |
| p150x4 | linear | 12.8 -> 11.5 | 20.6 -> 15.8 | 21.4 -> 18.0 | 20.9 -> 19.1 |
| p150x2 | full | 13.0 -> 12.8 | 20.5 -> 17.0 | 20.8 -> 18.5 | 21.4 -> 20.9 |
| p150x2 | linear | 13.7 -> 13.8 | 23.1 -> 19.7 | 24.0 -> 21.6 | 24.3 -> 23.9 |
| p150 | full | 14.8 -> 16.9 | 22.9 -> 21.7 | 23.6 -> 23.9 | 24.5 -> 26.9 |
| p150 | linear | 17.1 -> 19.4 | 27.0 -> 26.2 | 28.3 -> 28.9 | 29.5 -> 31.8 |

- Lengths where 2 pieces leave a padded piece, linear layer, limits 1 / 2 / 3 / 4, ms (`pieces3_x*_layer0.json`):

| Profile | 1300 | 2400 | 2600 | 3300 | 3584 |
|---|---|---|---|---|---|
| p150x4 | 13.9 / 14.4 / 15.0 / 15.0 | 21.6 / 18.2 / 19.8 / 19.5 | 22.8 / 20.4 / 20.7 / 20.5 | 22.2 / 23.2 / 22.7 / 22.7 | 22.6 / 24.1 / 24.6 / 24.4 |
| p150x2 | 15.0 / 16.2 / 17.7 / 17.2 | 24.3 / 22.1 / 24.1 / 24.5 | 24.5 / 23.7 / 24.8 / 24.8 | 25.2 / 28.1 / 28.8 / 28.8 | 26.0 / 29.1 / 30.4 / 30.7 |

- 3 or 4 pieces never beat 2. The rule picks the faster of 1 and 2 in every measured case.
- p150 keeps padding (split slower at 3 of 4 lengths). The probes are exploratory (no source manifest); the final
  timing is the repeated table in "Prefill latency by length".
- PCC of split against padding: >= 0.99998 (different MoE call grouping).

### Batch dimension

- Decode runs the layer once per batch row (`FunctionalDecoder.decode_forward`, inherited from Stage 1).
  Each row does its own two all-reduces: 2 x B collectives per layer per step. Stage 3 has the same per-row cost.
- Batched decode is owned by Stage 5 (layer) and Stage 7 (full model). Plan:
  - one layer pass over [B, 1, 2048]; two all-reduces per layer for any B (B x 4 KiB per device, latency bound);
  - attention and GDN over B slots in one op each; SDPA decode already takes a batch;
  - MoE over the union of selected experts. For B=32 the expected union is 256 x (1 - (1 - 8/256)^32) = 163
    experts. Expert-TP indexed decode then costs about 163/8 = 20x one token's expert cost on every device; masked
    EP needs about 41 local experts per device on x4. Measured masked EP: 195 us (0 active) to 248 us (8 active)
    for one token (`evidence7/micro/moe_ep_decode.json`). The B32 choice needs a B32 measurement in Stage 5.

### Rejected alternatives (measured, `work_log.md`)

| Option | Result |
|---|---|
| Expert-parallel decode (masked `sparse_matmul`) | 195-248 us vs 69 us expert-TP (x4) |
| Expert-TP prefill dispatch | 4859 us vs 1752 us EP at 4096 tokens (x4) |
| Reduce-scatter + all-gather instead of `all_reduce` | equal or slower (16.8 vs 18.4 us, x4) |
| Linear topology on the 1x4 ring | 26.2 vs 16.8 us |
| One CCL link | 20.7 vs 16.8 us |
| Sharded residual (hidden/tp), reduce-scatter + distributed RMSNorm + all-gather | 45.3 vs 40.7 us per boundary (x4), 61.7 vs 42.2 us (x2); without the output all-gather 36.5 us (x4) bounds a fused all-gather-matmul gain at 4 us (`evidence7/micro/residual_contract_x{2,4}.json`) |
| Pipeline parallel (layers per device) | no per-token decode speedup |

## Results (`evidence7/headline_*`, 4096 prefill tokens, 128 traced decode steps, recorded HF inputs)

Host times: warmed synchronized prefill and median of 128 blocking trace replays.
Rows run alone. Baseline and p150 on PCIe 3; p150x2 on PCIe 2,3. Generated by `evidence7/gen_tables.py`
into `evidence7/tables.md`.

| Layer | Profile | Prefill | Decode | Prefill speedup | Decode speedup | Decode efficiency | PCC vs baseline (prefill / min decode) | HF PCC (prefill / min decode) |
|---|---|---:|---:|---:|---:|---:|---|---|
| full | baseline (Stage 3) | 25778 us | 434.7 us | 1.00 | 1.00 | | | 0.99952 / 0.99835 |
| full | p150 | 25414 us | 434.7 us | 1.01 | 1.00 | 100% | 1.00000 / 1.00000 | 0.99952 / 0.99835 |
| full | p150x2 | 21919 us | 377.1 us | 1.18 | 1.15 | 58% | 0.99996 / 0.99936 | 0.99952 / 0.99835 |
| full | p150x4 | 19851 us | 346.5 us | 1.30 | 1.25 | 31% | 0.99996 / 0.99904 | 0.99952 / 0.99838 |
| linear | baseline (Stage 3) | 32521 us | 709.4 us | 1.00 | 1.00 | | | 0.99972 / 0.99874 |
| linear | p150 | 32225 us | 709.1 us | 1.01 | 1.00 | 100% | 1.00000 / 1.00000 | 0.99972 / 0.99874 |
| linear | p150x2 | 24946 us | 575.0 us | 1.30 | 1.23 | 62% | 0.99998 / 0.99845 | 0.99972 / 0.99874 |
| linear | p150x4 | 20959 us | 496.9 us | 1.55 | 1.43 | 36% | 0.99998 / 0.99888 | 0.99972 / 0.99874 |

- Decode efficiency = decode speedup / chip count.
- HF PCC is unchanged by the mesh.
- 40-layer decode estimate (30 linear + 10 full layers, sum of single-layer times): baseline 25.63 ms,
  p150 25.62 ms, p150x2 21.02 ms, p150x4 18.37 ms per token (1.40x). This is an estimate, not a measurement.
- Decode efficiency is low because a B1 decode layer is latency bound: 68-104 small ops per step (profile below).

## Prefill latency by length (`evidence7/micro/prefill_timing7/`, repeated)

Warm prefill host time in ms: one compile run, then the median of 6 warm runs per length. Same frozen sources as
chain 7 (`source_manifest_{start,end}.json` equal to `evidence7/source_manifest_end.json`). Recorded inputs.
The chain rows above are one timed run each and are up to 3.8 ms above the repeated medians on meshes; this table is the latency reference.
Spread (max minus min) per cell is at most 2.1 ms.

| full | 3 | 100 | 1000 | 1300 | 2176 | 2400 | 2600 | 3000 | 3300 | 3968 | 4095 | 4096 | 4097 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3.9 | 6.6 | 11.0 | 15.8 | 23.2 | 23.7 | 24.7 | 24.7 | 25.9 | 25.9 | 25.8 | 25.5 | 26.4 |
| p150 | 3.8 | 6.5 | 10.5 | 15.4 | 22.9 | 23.8 | 24.4 | 24.4 | 25.6 | 25.6 | 25.5 | 25.2 | 25.8 |
| p150x2 | 4.3 | 4.8 | 8.6 | 12.1 | 16.5 | 18.0 | 20.7 | 20.6 | 22.2 | 22.3 | 22.1 | 21.6 | 22.3 |
| p150x4 | 5.1 | 3.9 | 7.5 | 11.2 | 14.5 | 15.4 | 17.6 | 17.8 | 19.4 | 19.3 | 19.1 | 18.8 | 19.4 |

| linear | 3 | 100 | 1000 | 1300 | 2176 | 2400 | 2600 | 3000 | 3300 | 3968 | 4095 | 4096 | 4097 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3.8 | 6.7 | 13.3 | 18.4 | 27.1 | 28.4 | 29.4 | 30.0 | 31.1 | 33.3 | 33.2 | 32.3 | 33.4 |
| p150 | 3.7 | 6.4 | 12.8 | 17.9 | 26.7 | 28.1 | 28.9 | 29.6 | 30.8 | 33.0 | 32.9 | 32.0 | 33.1 |
| p150x2 | 3.5 | 5.0 | 9.1 | 14.2 | 19.2 | 21.4 | 23.5 | 23.5 | 24.9 | 25.7 | 25.5 | 24.8 | 25.8 |
| p150x4 | 3.3 | 4.3 | 8.1 | 13.6 | 15.2 | 17.8 | 19.8 | 18.7 | 21.4 | 21.5 | 21.3 | 20.7 | 21.5 |

- The MoE group split (plan, "MoE group split") removed the mesh step between 2176 and 4096 tokens.
- Small steps remain: p150x4 linear 2600 is 1.1 ms above 3000 (2600 runs a padded 1024-row piece); 3300 and 3968
  are 0.1-0.9 ms above 4096 on meshes. The baseline shows the same attention-plan step (linear 3968 is 1.0 ms
  above 4096).
- 3 tokens: p150x4 full is slower than p150 (5.1 vs 3.8 ms). Each decode-path token pays two all-reduces.

## Non-aligned lengths (`evidence7/len_*`, 4 traced decode steps, recorded inputs)

Control = single-chip baseline with identical inputs (`len_base_*_commands.json`). All 78 rows exit 0
(`CHILD_FAILURES 0` in `runs/main/stage04/evidence_chain7.log`). Row JSONs record the MoE pieces (`moe_pieces`, `[]` on the tiled prefill path).
- The padding mask runs when a MoE piece is padded: on linear layers at start 0, lengths 1281-1408 and 3073-3584
  (one padded bucket) and 2305-2432 and 2561-2688 (split plus padded piece); 896 lengths. Full-attention layers
  never reach it at start 0. Each range has a baseline-controlled row on p150x2 and p150x4 (1300, 3300, 2400, 2600).
Full table: `evidence7/tables.md`. Minimum PCC per row group:

| Tokens | Path | Control PCC, p150 | Control PCC, p150x2 | Control PCC, p150x4 | HF prefill PCC (all meshes) |
|---|---|---|---|---|---|
| 3 | decode-path tail | 1.0 / 1.0 | >= 0.99997 | >= 0.99995 | 0.99977-0.99983 |
| 100 | one masked chunk | 1.0 / 1.0 | >= 0.99998 | >= 0.99996 | 0.99932-0.99970 |
| 1000 | masked chunk | 1.0 / 1.0 | >= 0.99997 | >= 0.99996 | 0.99956-0.99972 |
| 2176 | ladder; mesh MoE pieces 2048 + 128 (p150 pads to 4096) | 1.0 / 1.0 | >= 0.99952 | >= 0.99995 | 0.99953-0.99971 |
| 2176 = 128 + 2048 | continuation at non-zero start | 1.0 / 1.0 | >= 0.99952 | >= 0.99995 | 0.99953-0.99971 |
| 1300 | linear: split rejected (padded piece under 2048): one padded 2048-row group (1312 valid, padding routes masked); full: one exact 2048-row group | 1.0 / 1.0 | >= 0.99996 | >= 0.99995 | 0.99953-0.99971 |
| 2400 | linear: mesh MoE pieces 2048 + 512 (384 valid, padding routes masked) | 1.0 / 1.0 | >= 0.99950 | >= 0.99950 | 0.99955-0.99971 |
| 2600 | linear: mesh MoE pieces 2048 + 1024 (576 valid, padding routes masked) | 1.0 / 1.0 | >= 0.99966 | >= 0.99924 | 0.99953-0.99972 |
| 3000 | non-aligned; mesh MoE pieces 2048 + 1024 | 1.0 / 1.0 | >= 0.99995 | >= 0.99995 | 0.99951-0.99972 |
| 3300 | linear: split rejected (pieces would total 4096): one padded 4096-row group (3328 valid, padding routes masked); full: one exact 4096-row group | 1.0 / 1.0 | >= 0.99995 | >= 0.99994 | 0.99950-0.99972 |
| 3968 | one exact 4096-row MoE group (no split, no mask) | 1.0 / 1.0 | >= 0.99932 | >= 0.99996 | 0.99952-0.99973 |
| 4095 | masked tail | 1.0 / 1.0 | >= 0.99936 | >= 0.99996 | 0.99952-0.99972 |
| 4097 | decode-path tail after 4096 | 1.0 / 1.0 | >= 0.99936 | >= 0.99996 | 0.99952-0.99972 |

## Batch 32 (`evidence7/b32_*`, 129 tokens, 8 traced decode steps, random inputs)

| Layer | Profile | Prefill | Decode per step | Decode speedup | Control PCC (prefill / min decode) |
|---|---|---:|---:|---:|---|
| full | baseline | 231103 us | 12622 us | 1.00 | |
| full | p150 | 217376 us | 12608 us | 1.00 | 1.00000 / 1.00000 |
| full | p150x2 | 142143 us | 10917 us | 1.16 | 0.99996 / 0.99997 |
| full | p150x4 | 107317 us | 9921 us | 1.27 | 0.99996 / 0.99997 |
| linear | baseline | 259938 us | 22252 us | 1.00 | |
| linear | p150 | 246065 us | 22255 us | 1.00 | 1.00000 / 1.00000 |
| linear | p150x2 | 158079 us | 17732 us | 1.25 | 0.99994 / 0.99998 |
| linear | p150x4 | 113775 us | 15222 us | 1.46 | 0.99993 / 0.99998 |

- B32 is correct on every profile.
- B32 decode is 29-31 x the B1 step: the inherited decoder runs the layer once per row
  (section "Batch dimension"). The batched path is Stage 5 and Stage 7 work.
- `prefill_repeat_identical` is false in the p150x4 B32 rows in every chain. See "Nondeterminism".

## Paged KV, positions, request reuse, native context (`evidence7/accept_*`, `evidence7/real_inputs_m*`)

| Gate | p150 | p150x2 | p150x4 |
|---|---|---|---|
| Reuse: 18 cases per layer, slots 30-31, continuations, one trace across requests, page 32 and 128 | pass | pass | pass |
| KV guard pages unchanged on every device; scratch page written | yes | yes | yes |
| Heterogeneous positions, batch 32, prefixes 1-1006, traced decode | pass | pass | pass |
| Same with recorded inputs, layer 3 | pass | pass | pass |
| Same with recorded inputs, layer 0 | row 17 token 4: 0.99393 | 0.99396 | 0.99414 |
| Native context 262143 prefill + decode, 262144 prefill | pass | pass | pass |
| Device pytest (`tests/test_multichip_decoder_device.py`, 12 cases) | 12 passed (one run covers all meshes; chain sets `QWEN35_X2_DEVICES=2,3`, the test default is 0,1) | | |

- Layer-0 row 17 token 4 also fails on the Stage 3 single chip (0.99393,
  `../optimized_decoder/final_evidence15/real_inputs/positions_0.json`). The mesh matches the single chip.
- Native-context PCC (p150x4): linear 262143 prefill 0.99939, decode 0.99987; full 0.99851 / 0.99840
  (single chip: 0.99939 / 0.99988, 0.99853 / 0.99837).
- The pytest log ends with nanobind leak warnings at interpreter exit. They come after the 12 passes and do
  not touch device state.

## Trace, determinism, audit, watcher

- Every timed and correctness row: `--trace --audit`, `TT_METAL_TRACE_ALLOC_TRACKING=1`,
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, program-cache misses forbidden during capture.
- Repeated traced decode replays are bitwise identical in every row (`repeated_identical`).
- Output replicas are bitwise identical across devices in every row (`output_replicas_identical`).
- Runtime fallback audit (`runtime_audit`) is clean in every row.
- First prefill (compiles programs) against the warmed prefill (`prefill_repeat_identical`): bitwise identical in
  113 of 122 rows (92 mesh-decoder rows including 2 placement rows, 30 baseline rows). The 9 exceptions are 7 mesh
  rows and 2 baseline rows (below). Warm against warm: see "Nondeterminism", E0 warm.
- Watcher (`TT_METAL_WATCHER=10`): 16 logs (`accept_watcher_m{1,2,4}`, `watcher_long`), all commands exit 0,
  no errors, asserts, sanitizer hits or hangs.
  - p150x4 needs `TT_METAL_WATCHER_NOINLINE=1`. Without it the watcher build of the ring fabric ERISC router
    is 29264 B, over the 26624 B ETH kernel config buffer (`evidence1/accept_watcher_m4/linear.log`).
    This is a watcher build size limit, not a model fault.

### Nondeterminism (`evidence7/nondeterminism/`, `evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md`)

First (compile) prefill against warmed prefill (`prefill_repeat_identical`). Same ttnn patch (sha256 723ed3a1) in
every chain. Cells: rows differing, max abs.

| Row | evidence1 | evidence2 | evidence3 | evidence4 | evidence5 | evidence6 | evidence7 (final) |
|---|---|---|---|---|---|---|---|
| p150x4 B32 full (of 4128 rows) | 953, 0.0156 | 709, 0.0117 | 1305, 0.0156 | 1430, 0.0078 | 1785, 0.0078 | 1184, 0.0156 | 1083, 0.0156 |
| p150x4 B32 linear | 838, 0.0078 | 1093, 0.0078 | 1184, 0.0078 | 1163, 0.0078 | 1315, 0.0156 | 958, 0.0078 | 947, 0.0078 |
| p150x2 B1 rows drifting (count; rows differing) | 2; 64-159 | 1; 32 | 1; 32 | 2; 32-64 | 3; 31-71 | 7; 31-94 | 1; 31 |
| p150x4 B1 rows drifting (count of rows checked) | 0 of 16 | 0 of 16 | 0 of 16 | 0 of 16 | 3 of 20; 117-127 | 2 of 27; 105-117 | 4 of 28; 74-119 |
| p150 rows drifting | 0 | 0 | 0 | 0 | 0 | 1 (linear 1000; 32, 0.0005) | 0 |
| single-chip baseline rows drifting | 1 (linear 1000; 32, 0.0005) | 1 (full 2176 = 128 + 2048; 1090, 0.0029) | 0 | 1 (linear 1000; 8, 0.0002) | 0 | 0 | 2 (full 2176 = 128 + 2048; 1090, 0.0029; linear 1300; 722, 0.0005) |
| p150 B32 and baseline B32 | bitwise repeat | bitwise repeat | bitwise repeat | bitwise repeat | bitwise repeat | bitwise repeat | bitwise repeat |

- Bound over all 7 chains: max abs 0.0156 in the layer output; repeat PCC >= 0.9999973. All control and HF PCC gates
  pass. Mesh drift rows on B1 differ in 31-159 of 1000-4096 rows.
- Rates by path (all 7 chains; chain 6 is superseded but kept, its sources differ from chain 7 only in the prefill
  all-reduce):
  - p150x4 B32: first and warmed prefill never match (14 of 14 rows, 17-43% of rows differ).
  - p150x4 B1: 0 of 64 rows in chains 1-4; 9 of 75 rows in chains 5-7, all at 2176-2400 tokens. These lengths
    changed path in chain 5 (MoE pieces), but 2176 = 128 + 2048 also drifts and runs no split piece. The cause of
    the rate change is not separated.
  - p150x2 B1: 1-3 rows per chain, except chain 6 with 7. Chains 6 and 7 differ only in the prefill all-reduce
    (`ttnn.all_reduce` vs `all_reduce_async` with construction-time semaphores): 7 rows vs 1 row. The collective
    path can change the rate.
  - Single chip (baseline and p150 share the path): 0-2 rows per chain. p150 differs from the baseline only where
    the baseline itself does not repeat.
- The mesh raises the rate. Bound and gates are unchanged. The mechanism is unknown (below); owner Stage 5.

Localization (E0 and E2 ran on chain-4 sources; the B32 129-token prefill uses one exact 128-row MoE group, so the
MoE split does not change its path; the prefill CCL semaphores changed in chain 7):

| Experiment | Result | Evidence |
|---|---|---|
| Per-op capture, B32 full layer, 4 runs | p150 and baseline repeat bitwise; on p150x2 and p150x4 the first differing capture is `moe_fused_swiglu` or `post_combine_reduce` output | `evidence7/nondeterminism/x*_full_b32_*.json` |
| BFP8 weight lists instead of the stacked extension (p150x4) | drifts the same way | `evidence7/nondeterminism/x4_full_b32_lists.json` |
| E0 cold: probe frees the previous output and synchronizes between runs; captures counts, offsets, expert ids; reference is run 0 (compiles programs) | drift remains (49-73 differing captures per run); all MoE inputs bitwise identical; run 0 is still the odd run in 5 of 19 drifting (call, device) pairs | `evidence4/nondeterminism/e0e2/x4_full_b32_e0e2.json` |
| E0 warm: same probe, reference is warm run 1, warm runs 2-6 compared | every warm run differs from warm run 1 (53-84 differing captures; layer output 151614-255183 elements); all MoE inputs bitwise identical | `evidence4/nondeterminism/e0e2/x4_full_b32_e0e2_warm.json` |
| E2b warm target: call 10 holds an alternate value in warm run 1 on devices 1 and 2 (runs 2-6 agree); its live inputs replayed on the 1x4 mesh 200 times back to back and 200 times with a neutral op between | 1 distinct output per device, equal to the run 6 value; the alternate never appears | `x4_full_b32_e0e2_warm.json`, `replay` |
| E2a warm target: the same inputs replayed on a single chip, 200 times | 1 distinct output, bitwise equal to the run 6 value | `evidence4/nondeterminism/e0e2/x4_full_b32_e2a_single_warm.json` |
| E2a/E2b cold target (call 27, device 3, alternate in cold run 0 only) | same result: 1 distinct output in 600 replays | `x4_full_b32_e0e2.json`, `x4_full_b32_e2a_single.json` |
| Standalone op loops: single chip, x4 mesh with fabric, with interleaved all-reduces | repeatable | `evidence7/nondeterminism/moe_repeat.json`, `evidence2/nondeterminism/followup/` |
| AutoDebug source inspection | no source defect in `moe_fused_swiglu`; fixed reduction order | `evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md` |

Classification:
- Affected path: prefill MoE expert compute and combine. Decode replays are repeatable.
- Warm runs drift too, not only the compiling run (E0 warm).
- For fixed inputs, the op gives one value in 600 isolated replays per target, on one chip and on the mesh,
  for a cold-run alternate and a warm-run alternate (E2a, E2b). The alternate appears only inside a model run.
- The mechanism is unknown. Carried device state from preceding ops and execution timing both fit; E2 does not
  separate them.
- Not specific to the stacked extension (BFP8 lists drift). Not specific to the mesh (rare single-chip rows).
  The mesh EP context raises the rate. The root cause is unknown.
- Resolution for Stage 4: controlled by bound (gates pass, max abs 0.0156). Not fixed.
- Owner: Stage 5 (optimized multichip decoder). Next: AutoDebug E3 (back-to-back calls inside the model
  run), E1 (column-band map of one alternate output; save raw tensors of both outcomes), then E5 (hardware state
  on the MoE cores).
- Risk for later stages: greedy outputs may differ between identical requests. Stage 6 and the serving stages
  must run a repeatability check.

## Profile (`evidence7/profiles/m{2,4}_*`, tt-perf-report 1.4.1)

- Method: `tests/run_profiles.py --mesh N --tracy-arg=--disable-device-data-push-to-tracy`. Without that flag the
  tracy ops CSV drops signposts and measured ops on a mesh (`work_log.md`).
- Tables: `prefill_perf_report.txt.gz`, `decode_perf_report.txt.gz`, CSV, stacked CSV, `decode_replay_summary.json`
  (per-op medians over 128 replays per device from the C++ device report).

Decode, kernel time per step, mesh device 0 (median over replays). Kernel sums are 2-5% above the traced host
medians in Results (profiler overhead):

| Op family | p150x4 linear | p150x4 full | p150x2 linear | p150x2 full |
|---|---:|---:|---:|---:|
| All ops (ops per step) | 522.5 us (104) | 353.4 us (68) | 602.8 us (104) | 379.2 us (68) |
| Matmul | 67.6 | 42.7 | 95.0 | 57.3 |
| Shared gate/up `minimal_matmul` | 11.7 | 11.7 | 13.7 | 13.7 |
| Routed experts `sparse_matmul` | 49.9 | 49.7 | 59.5 | 60.6 |
| RMS norms (`LayerNorm` ops) | 58.2 | 23.6 | 57.6 | 24.0 |
| Eltwise binary | 97.9 | 29.8 | 111.2 | 28.3 |
| Reshape, typecast, slice | 112.5 | 19.3 | 131.1 | 20.1 |
| SDPA decode (4096 context) | | 43.0 | | 41.8 |
| CCL: reduce-scatter + all-gather (two all-reduces) | 32.7 | 32.8 | 32.5 | 32.4 |

Prefill 4096, p150x4 (merged-device tt-perf-report): embedding gathers 4.4 ms (linear) and 4.6 ms (full),
`moe_fused_swiglu` 1.7 / 1.8 ms, scatter 1.6 / 1.6 ms (`evidence7/profiles/`). Warm prefill median is 20.7 / 18.7 ms.

Findings:
- Communication: 5.4-9.3% of decode kernel time and about 4% of prefill. Not the limiter.
- DRAM: modeled decode roofline 6.2% (linear) and 5.0% (full) on p150x4, 10.2% / 8.2% on p150x2.
  Decode is latency bound.
- Data movement: prefill dispatch bookkeeping (gathers, scatter, layout conversions) does not shrink with EP.
  It is the main p150x4 prefill target.
- SLOW rows still flagged by tt-perf-report (decode), after the geometry sweep:

| Op | Shape (M x K x N) | Now | Status |
|---|---|---:|---|
| GDN in-projection | 32 x 2048 x 3136 (x4) | 41.2 us | BF16 HiFi4; best measured geometry; BFP8 failed real-weight PCC in Stage 3. Stage 5 |
| qkvg | 32 x 2048 x 2560 (x4), 4608 (x2) | 20.9 / 34.2 us | best measured geometry (x2 DRAM-sharded rejected, see plan) |
| Router | 32 x 2048 x 288 | 8.0-11.1 us | replicated BF16 HiFi4; within 0.3 us of the best grid |
| Shared gate/up `minimal_matmul` | 32 x 2048 x 256 (x4), 512 (x2) | 11.8 / 13.8 us | best measured config |
| attention_out | 32 x 1024 x 2048 (x4), 32 x 2048 x 2048 (x2) | 9.5-16.3 us | best measured geometry |
| Shared down | 32 x 128 x 2048 (x4), 32 x 256 x 2048 (x2) | 4.2-4.3 us | best measured geometry |

- These ops read 0.3-12.8 MB each. They stay SLOW because one row per step cannot saturate DRAM. Fusing
  them is Stage 5 work.
- Interleaved FP32 GDN norms are not flagged but run on one core for 22-23 us each. Sharded norm failed
  real-weight PCC in Stage 3. Stage 5.

## Memory and capacity (`evidence7/inventory/`, `evidence7/capacity_model.json`, `evidence7/memory_handoff_recheck.json`)

| Profile | Allocator per layer (linear / full) | Construction peak per layer | 40 layers (weights) | 262144-token sequences (8 GDN slots) |
|---|---:|---:|---:|---:|
| Stage 3 single chip | 1577.0 / 1536.6 MiB | | about 60 GiB | does not fit |
| p150 | 761.0 / 720.6 MiB | 2601 / 2563 MiB | 24.3 GiB | 1.32 |
| p150x2 | 680.9 / 692.4 MiB | 1977 / 2001 MiB | 23.1 GiB | 3.89 |
| p150x4 | 365.4 / 403.7 MiB | 1683 / 1836 MiB | 11.6 GiB | 12.7 |

- p150 holds one expert copy: dispatch reads the stacked decode weights. p150x2 and p150x4 hold an expert-TP copy
  for decode and an EP copy for prefill, each 1/tp of the experts.
- Construction peak per layer: `evidence7/memory_handoff_recheck.json::construction_transient_peaks`
  (`evidence7/construction_peaks.py`).
  p150 is the setup transient before cleanup in `OptimizedDecoder.from_state_dict`; meshes are the replica
  build (`inventory/replica_peaks/`). Replicas are released after merge (`released_after_merge` true).
  A layer-by-layer full-model build adds at most this transient per layer.
- Host peak RSS during construction: 6.5 GiB (p150x2), 4.5 GiB (p150x4). Not recorded for p150 (no replica build).
- Handoff recheck (`memory_handoff_recheck.json`): 7 of 9 required items addressed. Open for the full model:
  share RoPE (64 MiB per full layer) once per mesh, and measure capacity on the full model.
- Capacity conditions: constants shared once per mesh in the full model; 2 GiB reserve is an estimate; GDN slots
  configurable.
- No capability reduction: 262144 tokens on every profile (`../context_contract.json::multichip_decoder_validation`).

## Code changes

| File | Change |
|---|---|
| `tt/multichip_decoder.py` | new: slicing, replica merge, EP dispatch (padding routes masked to the dummy slot), all-reduce wrappers (prefill: `all_reduce_async` with construction-time semaphores; decode: `ttnn.all_reduce`), mesh policies, per-kind decode projection configs |
| `tt/optimized_decoder.py` | `dispatch_weights` policy (`lists` default), `shared_decode_minimal_config`, `prefill_moe_pieces` and `_moe_pieces` (MoE group split, default 1 = Stage 3 padding), EP `valid_rows` hook (reset in `finally`), `moe_pieces_log`, `_StackedDispatch`, restore hook, width from shapes, fused-reduce axis attribute, GDN a/b column stride, device-0 replica readbacks |
| `tt/fused_decoder.py` | shared-expert packing and indexed-RoPE map from shapes; replica readbacks |
| `tt/functional_decoder.py` | `replicated_mesh` flag; layer shell without the discarded qwen36 GDN on meshes |
| `tt/functional_gdn.py` | head-count check relaxed to local head counts |
| `ttnn/.../moe_fused_swiglu` | stacked weights (`stacked_intermediate`, column offsets), separate down dtype (`evidence7/ttnn_patch.diff`) |
| `tests/` | `--mesh`/`--decoder multichip` in harnesses, `moe_pieces` in `run_layer.py` rows, `mesh_util.py`, `test_multichip_plan.py`, `test_multichip_decoder_device.py` |

- p150 outputs are bitwise equal to the Stage 3 baseline in 28 of 30 rows (`control_equivalence`, length, headline and
  B32 rows); the 2 exceptions are
  rows where the baseline itself does not repeat ("Nondeterminism").
- `moe_fused_swiglu` op tests: new `test_moe_fused_swiglu_stacked_mixed_weights` passes (3 cases).
- CPU tests: 1167 pass (`evidence7/cpu_pytest.xml`).
- Source identity: `evidence7/source_manifest_start.json` equals `source_manifest_end.json`;
  snapshot `evidence7/source_snapshot.tar.gz`.

## Anomaly ledger

| Anomaly | Evidence | Control | Resolution |
|---|---|---|---|
| Layer-0 positions row 17 token 4 PCC 0.994 | `real_inputs_m*/positions_0.json` | Stage 3 single chip 0.99393 | controlled (same on all profiles) |
| Prefill tokens below 0.995 vs HF (1-3 per 4096) | headline `prefill_token_pcc` | baseline; tokens 950, 637, 1720, 3407 | controlled: the four tokens tie exactly at the 8th/9th BF16 router logit; 913 of 4096 tokens (fraction 0.2229) have a gap below 0.01 (`evidence7/micro/router_margin_layer0.json`) |
| Prefill not bitwise repeatable: p150x4 B32 always; p150x4 B1 9 of 75 rows since chain 5 (0 of 64 before); p150x2 and baseline rarely | "Nondeterminism" | p150 B32 and baseline B32 repeat; op replays repeat (E2a, E2b) | controlled by bound (max abs 0.0156, repeat PCC >= 0.9999973 across 7 chains, gates pass); root cause unknown, owner Stage 5 |
| p150x4 watcher fails to load without NOINLINE | `evidence1/accept_watcher_m4/linear.log` | p150x2 watcher passes | fixed with `TT_METAL_WATCHER_NOINLINE=1` |
| B32 decode 29-31x B1 | `b32_*` | baseline has the same cost | controlled: per-row decode inherited from Stage 1; Stage 5/7 |
| p150x4 3-token prefill slower than p150 (5.1 ms vs 3.8 ms, full, repeated medians) | `len_m4_full_3` | decode-path steps pay 2 CCLs each | expected; Stage 5 |
| Mesh prefill at 2176 tokens no faster than 4096 (review 4) | chain rows 21.6-24.4 ms vs 19.7-20.9 ms; `evidence7/micro/padding_routes/` | p150 unaffected | fixed: cause is the MoE group padded to 4096 (EP cost follows padded rows; padding rows also overloaded rank 0). Split into 2048 + 128 on meshes: p150x4 14.5 / 15.2 ms (repeated medians) |
| Single chain prefill rows are up to 3.8 ms above repeated medians on meshes | chain rows vs repeated medians | repeated timing spread <= 2.1 ms | controlled: repeated table is the latency reference |
| L1 clash: `moe_fused_swiglu` static circular buffers vs a CCL semaphore at 1039104 (chain 6, p150x4 linear 2600) | `evidence7/micro/l1_clash/` | 1000 and 2400 pass in the same process; semaphores at 1.56 MB there | fixed: prefill all-reduce semaphores created at construction; chain 7 row passes |
| p150x4 linear 2600 decode PCC vs HF 0.99911 (p150 0.99971) | `evidence7/len_m4_linear_2600` | control vs baseline 0.99924; other mesh rows 0.99909-0.99984 | controlled: inside the mesh range (e.g. x2 linear 4095 decode 0.99920) |
| Reshape, typecast and slice: largest p150x4 linear decode family (112.5 of 522.5 us) | `evidence7/profiles/m4_linear_attention/decode_replay_summary.json` | inherited GDN decode path of Stage 3 (p150x2 131.1 us) | open, Stage 5: fuse or remove the GDN layout ops |
| Chip 0 card slower decode by 2.5-8.5 us | `placement_p150x2_01_*` (`TT_VISIBLE_DEVICES=0,1` set by the chain command; artifacts record logical IDs [1, 0]) | Stage 3 device A/B | controlled; p150x2 uses PCIe 2,3; p150x4 must use chip 0 |

## Limitations and Stage 5 work

- Batched decode (one layer pass for B rows) and the B32 expert choice (section "Batch dimension").
- Prefill dispatch bookkeeping (index-driven activation reads in `moe_fused_swiglu`, fused gather-combine, scatter).
- GDN in-projection and decode norms (accuracy constrained).
- Remaining SLOW rows above (fusion, not geometry).
- Short-prompt fixed cost: 3-128 token prefill takes 3.4-7.6 ms on every profile (inherited from Stage 3).
- Reshape, typecast and slice in the GDN decode path (112.5 us per p150x4 linear step).
- CCL path selection: prefill vs decode is chosen by `partial.shape[-2] == 1`. A batched decode (Stage 5) must select
  by mode, or it takes the prefill path (+9 us per traced all-reduce on p150x4).
- Decode all-reduce semaphores are created when the decode program is first built. The full model (Stage 6) must
  build decode programs before it allocates persistent L1 state, or give decode its own pre-built semaphores.
- Prefill semaphore ping-pong is per wrapper; back-to-back attention and MoE all-reduces reuse a set. Safe by the
  reduce-scatter and all-gather start barrier (source inspection; native-context rows pass).
- MoE prefill nondeterminism: AutoDebug E3, E1 (needs raw tensors of both outcomes), E5. E0 and E2 ran with summary
  statistics and one saved call per target.
- Fused all-gather + matmul consumer for a sharded residual: at most about 4 us per boundary on x4
  (`evidence7/micro/residual_contract_x4.json`).

## Reproduction

```bash
source /home/hous/dev/qwen3.5-35b-a3b/scripts/environment.sh
cd /home/hous/dev/qwen3.5-35b-a3b/tt-metal
bash build_metal.sh
/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain7.sh
/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/process_mesh_profiles7.sh 4
/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/process_mesh_profiles7.sh 2
```

- Micro benchmarks (`evidence7/micro/`) are standalone op probes, run separately from the chain. Scripts are
  copied next to their JSON.
