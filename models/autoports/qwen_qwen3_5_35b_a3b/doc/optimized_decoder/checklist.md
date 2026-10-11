# Stage 03 optimize checklist

Evidence paths are relative to this directory. "Final" means `final_evidence15/` on the
frozen final source.

- [x] **Decoder path fully traced, no host fallbacks.** Every acceptance run uses traced
  decode with `--audit` (device-only runtime audit) and trace allocation tracking with the
  program cache included. Prefill routing, dispatch maps, ranks and combine are on device.
- [x] **Decode activations sharded in L1 where it wins.** Measured: sharded residual
  through the whole layer (Codex `attention_real`, slower), sharded norms with L1/DRAM
  outputs (`norm`, `norm2`), L1-interleaved intermediates (`gdn_l1`, kept for GDN), L1
  attention activations vs DRAM (`prefill4k_full` disp_2048_attn_dram, L1 kept). L1
  interleaved norm outputs slow consumer matmuls by about 30 us. The selected path is
  DRAM-interleaved residual, L1 or sharded intermediates where measured faster.
- [x] **Prefill activations DRAM interleaved; large prefill matmuls with 2D/optimized
  configs.** Projections use auto multicast configs with `packer_l1_acc` (`inproj_*`);
  routed experts use dispatch plus native fused SwiGLU; full-attention prefill uses
  chunked SDPA.
- [x] **Operation-topology audit.** `work_log.md` table "Initial operation-topology
  audit" plus per-op traces in `work_log.md` and `profile_c2/`, `profile_diag_heads/`.
- [x] **Multi-device topology families.** Not applicable: single-device layer stage
  (no CCL in the measured path).
- [x] **Lower-movement residual candidates.** Not applicable for CCL. Sharded residual
  measured (Codex `attention_real`), rejected.
- [x] **Best-candidate comparison.** Every matrix has a matched control row in the same
  frozen matrix. Final vs Stage 2 and vs all candidates: README headline and
  `work_log.md` tables. Final decode beats every correct candidate that passes all gates.
- [x] **Final default reproduces the selected candidate.** `final_evidence15/perf_*`
  (no policy argument, device 3): full 434.60 us, linear 709.10 us; matched candidates
  `review1/p1/perf/out_bfp8_hifi2` 709.63 us, `review1/reruns` full 434.26 us. Prefill
  grid control in the same chain: `final_evidence15/control_grid_*` (selected 8x8 for 2048
  and 4096 groups beats 4x4 by 0.26-1.82 ms).
- [x] **Final dtype/fidelity verified in runtime rows.** `final_evidence15/profile/tracy/*/{decode,prefill}_perf_report.csv`:
  prefill `moe_fused_swiglu` BF16xBFP8, 64 cores; k-major tile combine from 2048 tokens;
  QKVG BF16xBFP8 LoFi; full out BF16xBFP8 LoFi; GDN in BF16xBF16 HiFi4; GDN out BF16xBFP8 HiFi2; router+gate
  BF16xBF16 HiFi4 N=288; sparse gate/up BF16xBFP4 LoFi; sparse down BF16xBFP8 LoFi;
  shared SwiGLU BF16xBFP4 LoFi; shared down BF16xBFP8 LoFi DRAM-sharded; SDPA K/V BFP8.
- [x] **SDPA and composite ops.** Paged SDPA decode, chunked SDPA prefill, indexed RoPE,
  fused paged cache update, fused SwiGLU, fused MoE reduce, native GDN chunk op, fused
  causal conv. Explicit SDPA program configs measured (`gate_full`, `c4_full`): no win.
- [x] **Same-input projections packed where beneficial.** QKVG packed (Stage 2); GDN
  QKV/A/B/Z packed (split variant measured, `c7_linear`); router + shared gate packed
  (new, `gate_*`); shared gate/up packed vs split (`shared_full`).
- [x] **Explicit memory/program/compute configs for important ops.** `DEFAULT_POLICIES`
  plus `_Projection`, `_PackedGateMoE`, `_DecodeNorm`, sparse expert program configs.
- [x] **Dominant matmul program-config sweeps.** Tables in `work_log.md` and README
  rejected-options table: attention in/out (1D grids 2/4/8 cores with k 8/16/32, 8x4,
  8x6, 8x8; DRAM-sharded 4/8/16/32 cores, k 2-16, readers 1/2/3), router, shared gate,
  shared down (readers 1/2), sparse gate/down (cores 8-64, k 8-64, subblocks),
  dispatch grids (4x2/4x4/8x2/8x4 under BFP4; 4x4/8x4/8x8 small and 8x4/8x8/11x8 large under
  the selected BFP8, `review2/bfp8_grid`), gate-pack grids (k 8/16/32, 8x1/8x2).
  In-projections stay on the auto config at `in0_block_w=2`: every larger-K 1D and
  DRAM-sharded candidate was slower or failed (`geom_*`, `attention_real`, `gdn_real`).
- [x] **Decode compute fidelity swept per dominant group.** attention in (BFP8 LoFi/HiFi2/HiFi4,
  BF16 LoFi/HiFi2/HiFi4), attention out (BFP8 and BF16 x LoFi/HiFi2/HiFi4), router
  (BF16 HiFi4, BFP8 HiFi2/LoFi, BFP4 LoFi), experts LoFi, SDPA LoFi.
- [x] **Attention projection dtype/fidelity separate from MLP.** BFP4 attention-in tried
  on real activations: full 0.9893, GDN 0.9638 (fail). BFP8 GDN attention-in and
  attention-out fail real-activation gates (README). Full attention-out BFP8 LoFi kept.
- [x] **BFP4/LoFi expert trials.** Gate/up BFP4 LoFi selected. Down BFP4 tried
  (`shared_full`, `c5v_full`): fails at token 127 (0.9946), rejected.
- [x] **Shard specs divide tiles cleanly.** Norm 16 cores x 4 tiles; out-proj 8x4 x 2
  tiles; gate pack 9 of 16 cores x 1 tile; DRAM-sharded shared down 8 banks x 8 tiles.
- [x] **DRAM-sharded decode matmuls with Blackhole readers 1/2/3.** Measured for
  attention in and out, router, shared gate, shared down (r3 illegal for 8 tiles/bank),
  GDN in/out: `attention_real`, `gdn_real`, `projection_extra`, `aobf16_linear`,
  `geom_*`. Selected: shared down 1 reader (r2 flat). Others: 1D or auto configs won.
- [x] **Collective topology minimized.** Not applicable (single device).
- [x] **Fused matmul-CCL.** Not applicable (single device).
- [x] **Persistent CCL buffers.** Not applicable (single device).
- [x] **MoE active-expert path.** Indexed `sparse_matmul` (8 active, no dense
  all-expert decode), separate gate/up and down tuning, nnz omitted because indexed
  mode takes explicit expert IDs, routing scores in fused weighted reduce, L1
  intermediates. Prefill: device dispatch, only routed rows, BFP8 weights (per-token
  prefill accuracy, `review2/dispatch_dtype`), one MoE call per group of up to 4096 tokens;
  sibling ops `unified_routed_expert_moe` / `hybrid_routed_expert_moe` measured slower or
  blocked (`review3/routed_ffn`).
- [x] **LM head / sampling.** Not present in a decoder-layer stage; owned by full-model
  stages.
- [x] **LM head DRAM-sharded.** Not present in this stage.
- [x] **Reduced precision with real weights and activations.** All precision decisions
  use the real checkpoint and recorded HF activations; random-activation gates are the
  inherited acceptance bar; disputed cases re-run with recorded activations (OPT-012).
- [x] **Performance accounting reconciled.** README roofline table: roofline, device
  and host decode from the same final source.
- [x] **Batch capability preserved.** B32 heterogeneous positions (`positions_*`), B2
  reuse; 32 configured GDN state slots (65.5 MiB per linear layer, recorded separately from
  active batch in `final_evidence15/memory_handoff_recheck.json`).

Additional stage requirements:

- [x] Non-aligned lengths: reuse (1, 31, 33, 127, 129, 511, 513, 1153, padded 35/36,
  cuts, page 128), positions (1..1006), context 262143, timed 2/3/4/5/100/1000/1100/1500/2176/2600/
  4000/4095/4097/4098/4100 and aligned 1024/2048/3072/4096 plus a 128-token continuation
  (`final_evidence15/prefill_*`), pytest 129 and 1000, CPU planner invariants
  (`tests/test_prefill_plan.py`, 1125 cases), reuse scratch-page write (2047 tokens, cut 1152).
- [x] Watcher-clean optimized run: `final_evidence15/watcher_*` (11 logs, 0 errors; 4096, 4095, 1000 and full 1500 with a scratch write).
- [x] Stress / repeated runs: B32 trace replays, determinism checks, 128-step decode.
- [x] tt-perf-report tables for prefill and decode, advice enabled:
  `final_evidence15/profile/tracy/*/{prefill,decode}_perf_report.{txt,csv}`.

Stage-review items: see `work_log.md` sections "Stage review 1 response" through "Stage review 5 response".
