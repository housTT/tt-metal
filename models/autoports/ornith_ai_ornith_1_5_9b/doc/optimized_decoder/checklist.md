# Optimize checklist

Each item below reproduces the selected skill checklist, with evidence relative to this directory. Final gate status is recorded in README and final_gates_summary.json.

1. [x] Decoder path fully traced with no host fallbacks

   Evidence: Final short/no-host guard, pair exact restored trace, long trace and batch resource gates; test_optimized_decoder binds OptimizedDecoder and forbids FunctionalDecoder._block.

2. [x] Decode activations generally width-sharded in L1 across norm, attention, residual, MLP, and output projection boundaries.

   Evidence: Runtime _norm/_linear/_residual_add retain width-sharded boundaries at batch 1; batch folding and explicit GDN/SDPA handoffs have measured movement controls.

3. [x] Prefill activations generally DRAM interleaved; use 2D matmul program configs for large prefill matmuls.

   Evidence: Explicit 11x10/K16 large-prefill with 1x7 subblock bounds and 8x8/K8 short-prefill 2D configurations. final_prefill_subblock_plan compares both orientations and per-role/all-role changes; cap8 supplies no measured benefit. final_prefill_mlp_grid_plan adapts the full-width35 allocation failure to legal M1 and measures its loss. Earlier 8x8 controls are preserved in final_combined_topology_plan.

4. [x] Operation-topology audit completed: current op sequence, repeated same-input matmuls, collectives, reshard/layout conversions, candidate fused/lower-movement replacements, dtype/fidelity constraints, and action taken are recorded.

   Evidence: work_log initial topology table plus topology, movement, packed-adaptation and final combined plans; decisions and timings in README.

5. [x] Multi-device topology candidates were measured as coherent families when applicable: residual layout, collective placement, fused CCL+matmul use, projection packing or separation, activation/CCL dtype, and persistent-buffer use. A rejection measured only under an incompatible residual/layout contract does not complete this item.

   Evidence: Not applicable: this stage executes a single chip and contains no device collectives.

6. [x] Lower-movement residual candidates were measured without an immediate old-contract restore when applicable. If a reduce-scatter or fused CCL+matmul path only lost after an immediate all-gather or full replication, a stack-compatible sharded/fractured residual path was also measured or a minimal repro proves the next op cannot consume that layout.

   Evidence: Output projection and residual remain sharded through the consuming norm/MLP. residual-grid and output-retention candidates measured together; multi-device reduce-scatter is absent.

7. [x] Best-candidate comparison completed: the final path is compared against the strongest available correct baseline, earlier optimized artifact when present, and material candidates from this stage. The final choice wins traced warmed decode or has an explicit target-specific reason for prioritizing another workload. A synthetic-only precision veto does not count as a correctness reason when real-weight evidence passes. A geometry sweep measured only under a different dtype/fidelity does not reject the final dtype/fidelity policy.

   Evidence: candidate_measurements.csv, geometry*.json, final_combined_topology_plan.results.json, cache_precision_summary.json; real-weight acceptance selects BFP4/LoFi projections and BFP8 KV.

8. [x] Final default performance reproduced the selected best candidate under the final code path. If the final default is slower, the report uses the final number and explains why the candidate was not preserved.

   Evidence: final_default_measurements.json comes from final production-default pair, compared to final separate control and selected candidate with declared 0.5% repeat noise tolerance.

9. [x] Final dtype/fidelity policy is verified in the measured runtime rows, not only in policy JSON or constructor defaults. For each dominant matmul, the `tt-perf-report` row or an equivalent profiler artifact must show the expected input/weight dtype and math fidelity. If the row shows BF16 or BFP8 where the selected policy claims BFP4, the policy did not reach the measured op and the stage is incomplete.

   Evidence: tracy/*/optimized_release_v4/decode_accounting.json asserts every projection weight is BFLOAT4_B, fidelity LoFi and readers equal best_geometry_config.json; original rows compressed alongside reports.

10. [x] Used SDPA and other optimized composite ttnn ops instead of hand-built attention primitives where the target model fits their contracts.

   Evidence: Native paged/chunked SDPA, RoPE, flat chunk GDN and KDA Conv1d+SiLU. Native padded chunk decode and broadcast outer-product alternatives passed but lost.

11. [x] Fused or packed repeated same-input projections where legal and beneficial, such as Q/K/V-style projections, paired gate/up projections in 3-matmul MLPs, or other model-specific projection groups. If kept separate, there is measured evidence or a specific unresolved TTNN/runtime blocker after adapting layout, rank, padding, weight packing, and output splitting. If kept packed, it wins against a well-tuned legal separate candidate after counting split, activation, binary elementwise, and layout overhead, or the evidence explains why the separate candidate is invalid.

   Evidence: Final packed MLP readers 1/2/3 all lose tuned separate; packed QKV/AB and full QKVG beat legal separate controls; final packed GDN including Z loses. Includes split and layout costs.

12. [x] Explicitly configured `memory_config`, `program_config`, and `compute_kernel_config` for important ops.

   Evidence: DecoderConfig and named projection policy explicitly configure all dominant projections, residual/norm, recurrent reads, SDPA and prefill.

13. [x] For any matmul or repeated matmul group that is one of the largest decode-time consumers: swept legal program configs separately for each dominant role, including core grid, larger legal `in0_block_w` values, output subblocks, output blocks, memory configs, and compute kernel config where applicable. The stage is incomplete without a before/after evidence table or an exact TTNN/runtime blocker.

   Evidence: Per-role geometry inventories include small/large grids, legal divisors, output blocking, L1/resource errors and adapted successful configurations; final_prefill_grid plans include K4/8/16/32/64/128 with legal M1/N1 adaptations. Reader reports prove final choices at BFP4/LoFi.

14. [x] Decode compute fidelity was swept as a real performance knob for each dominant projection group. Do not assume BFP8 implies HiFi2 is fastest; try legal LoFi and HiFi2 candidates with the same dtype and real traced decode evidence, then keep the fastest policy that passes correctness.

   Evidence: precision_residual_plan and recorded candidate ledger compare LoFi and HiFi2 independently for attention and MLP. Both selected LoFi.

15. [x] Attention projection weight dtype/fidelity was swept separately from MLP weight dtype/fidelity when QKV, Q/K/V, output projection, or fused attention matmul rows are material. If attention projections remain BFP8 or BF16, the report names the BFP4 attention candidate tried on real weights or recorded real activations, plus the precise correctness, latency, or op-contract blocker.

   Evidence: Separate attention policy sweep passed BFP4/LoFi on recorded target activations; actual final profiler rows confirm propagation.

16. [x] If dense MLP or expert matmuls are among the largest decode-time consumers: BFP4/LoFi trials for FF1/FF3 or equivalent gate/up projections were run before lower-priority prefill-only advice was pursued to completion. FF2/down BFP4 was also tried or rejected with PCC/runtime evidence.

   Evidence: All gate/up/down roles tested at BFP4/LoFi across material geometries before completing lower-priority prefill work; original checkpoint packing fixes double-rounding accuracy loss.

17. [x] Shard specs and core grids that divide tensor dimensions cleanly into tiles where possible, code grids as large as this and the model/hardware allows.

   Evidence: best_geometry_config.json and geometry inventories; final narrow-reader-3 comparison adapts padded N18 rather than discarding first legality failure.

18. [x] DRAM-sharded decode matmuls. On Blackhole builds with multi-reader support, every material role was measured with one reader and each legal two- or three-reader candidate. The final count is explicit in the model's per-role configuration; the generic default remains one.

   Evidence: reader_comparison_summary.md/csv/json and AUTOFIX_reader_profiler.md: all 66 profiled and 66 unprofiled alternating cases pass; selected counts win actual kernel and host trace time. Generic reader default remains 1.

19. [x] Collective topology minimized. Avoidable gather, reshard, all-reduce, reduce-scatter, and all-gather operations have been removed, moved to cheaper boundaries, or justified with before/after evidence.

   Evidence: No collectives on one chip. Measured reshard/movement controls retain sharded residual and shared gate/up input, flat GDN, direct DRAM rotary tails.

20. [x] Fused matmul-CCL ops used where possible, including fused all-gather-matmul or matmul-reduce-scatter patterns when a collective and matmul are adjacent or can be made adjacent. If rejected, the rejection includes an adapted attempt, not only the first API error.

   Evidence: Not applicable: no adjacent CCL or multi-chip path in this decoder.

21. [x] Repeated decode CCLs use persistent or preallocated intermediate/output buffers where the API supports it. If unavailable or slower, the reason and measurement are recorded.

   Evidence: Not applicable: no CCL operations or communication buffers in this decoder.

22. [x] For MoE models: optimized the routed active-expert path with `ttnn.sparse_matmul` where the model/hardware fits, correct `nnz` handling, separate gate/up and down tuning, correct sparse-input handling where applicable, routing-score weighting, expert reduction, no dense all-expert runtime path, and no avoidable DRAM round trips through decode intermediates.

   Evidence: Not applicable: the checkpoint has dense MLPs, no routing or expert weights.

23. [x] For models with an LM head and sampling: final norm, LM head, logits movement, sampling, and token feedback are included in the optimized token-out path; terminal costs are profiled separately in full-model or reduced non-serving evidence, not in vLLM serving stages; LM-head weights are padded when needed for legal/fast DRAM-sharded or vocab-sharded matmuls; padded vocab IDs are masked in local logits shards before force-argmax or TopK; split-sampling TopK input widths are padded to avoid the slow single-core TopK fallback where possible; avoidable `ArgMaxDeviceOperation`, full-vocab all-gather, generic `TopKDeviceOperation`, host argmax, and full-logits readback have been removed. If a TTNN/runtime limitation blocks removal, the stage remains incomplete until there is a minimal repro or a lower-level fix.

   Evidence: Not applicable: this decoder layer has no LM head, sampling, token feedback or generation output.

24. [x] LM Head is optimized for DRAM-sharded matmuls if present.

   Evidence: Not applicable: LM head is absent.

25. [x] Reduced precision/fidelity experiments appropriate to this module-level optimization stage have been carried out and documented using real weights and input activations. For complete full-model top-k tuning, final datatype frontier selection is deferred to `$datatype-sweep`.

   Evidence: Separate real-weight attention/MLP/fidelity/KV trials plus recorded input provenance. Synthetic diagnostics retain exact replay and finite output without vetoing real-weight precision.

26. [x] Performance accounting reconciled: roofline estimate, device-time decode, and end-to-end decode reported from the same run; avoidable gaps optimized away, and any remaining gap named as a ttnn/runtime/API limitation only after a targeted fix attempt; `perf_summary.json` written when optimizing a complete model or serving path. For vLLM serving stages, use same-harness primary single-user and CI serving-burst serving metrics and set device-time/profile fields to `null` with the no-profiler reason.

   Evidence: tracy/*/{baseline_recorded,optimized_release_v4}/decode_accounting.json reconciles physical bytes, kernel sum, device span and same-run host timing. README discusses remaining measured gaps and targeted trace-window/drain controls.

27. [x] Batch capability preserved: batch-1 is the primary optimized latency target, and larger-batch or concurrent-serving correctness was tested up to 32 where hardware and memory allow it.

   Evidence: Final batch resource gate covers 1–32 including primes, both layouts, exact restored outputs/state, per-user HF, input ownership and repeated trace stress; native context gate preserves batch-1 262144.

## Final implementation audit

- Traced optimized runtime uses no torch/from_torch/to_torch or host fallback; construction and external harness restoration are outside the measured path.
- Necessary format boundaries are documented in README. Native KDA convolution requires row-major input/history; grouped rotary and SDPA require legal explicit shard grids. No silent functional block fallback is permitted by delivered tests.
- Both kinds preserve real-weight PCC 0.995, repeated-run stress, arbitrary logical lengths, and separate Watcher evidence.
- Serving, asynchronous adapter, on-device sampling and qualitative generation checks are not applicable to a decoder-only stage.
