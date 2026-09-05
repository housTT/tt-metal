# AutoDebug: sharded decoder eager/trace disagreement

Source-only investigation, 2026-09-05. Applied AutoFix, AutoDebug, and the trace-debugging guidance. No implementation edits, TTNN imports, hardware commands, or experiments were performed by this investigator. **No root cause is verified yet.**

## Evidence and limits

- Failure: [mlp_dram_c8_b8.log](logs/mlp_dram_c8_b8.log), with [provenance](logs/mlp_dram_c8_b8.provenance.json) and its frozen source archive. The linear-attention test fails `test_optimization_experiments.py:91`, exact own-eager/trace equality. Example leading values are eager `[0.0649, 0.1504, 0.0437]` and trace `[0.0688, 0.1592, 0.0376]`, both BF16. Capture/replay completes; this is not a hang or capture-write exception.
- The assertion does **not** include `name`, so the saved traceback alone does not establish whether `fused` or `optimized` failed. Its location is after both complete. Add the decoder name and discrepancy statistics before attributing the failure to an optimized operation.
- The first checked replay is after seven windows of 32 replays, followed by restore and one replay (`tests/test_optimization_experiments.py:58–75`). There is no immediate first-replay check, no restored second-eager check, and no eager post-decode state snapshot. The log therefore cannot distinguish first-call behavior, immediate trace divergence, and accumulated corruption.
- Passing control: [bfp4_mlp_recorded.provenance.json](logs/bfp4_mlp_recorded.provenance.json). Same MLP BFP4/LoFi policy, with interleaved operations, passes both layers and exact own-eager/trace checks. Its linear HF PCCs are prefill **0.9984640467**, decode **0.9989911755**, and stress **0.9986211093**. The optimized source subsequently changed, so rerun a current-source `config={}` control.
- Both runs' log hashes, archive hashes, and all archived source hashes were verified using Python's standard library. Current `tt/optimized_decoder.py`, `tests/test_optimization_experiments.py`, `tt/fused_decoder.py`, and `tt/functional_decoder.py` match the failing archive. Optimized source SHA-256: `6f1a99df19152ade07288b7993391b32057d50fd29430206587a94f37a011016`.
- The coordinator reports a current-source interleaved attention-BFP4 screen passing exact own-eager/trace (separately failing its HF prefill accuracy gate), and successful standalone DRAM-sharded gate projections passing own-eager/trace in the ongoing `geometry_gate_bfp4.log`. These are useful additional controls, but do not test the complete sharded residual/MLP chain. They were not run by this investigator.

Recorded reproduction, in the existing project environment:

```bash
OMP_NUM_THREADS=8 ORNITH_WEIGHTS=real \
ORNITH_OPT_POLICY='{"mlp_gate_up":"bfloat4_b","mlp_down":"bfloat4_b","mlp_fidelity":"LoFi"}' \
ORNITH_OPT_CONFIG='{"residual_cores":8,"dram_roles":["gate_proj","up_proj","down_proj"],"cores":8,"block_w":8}' \
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimization_experiments.py -x -v -s
```

The provenance also records the existing `TORCHINDUCTOR_CACHE_DIR`, `TT_METAL_HOME`, `TT_METAL_CACHE`, and virtual environment. Preserve those when reproducing.

## Precision and layout ledger

| Boundary | Passing recorded control | Failing candidate |
| --- | --- | --- |
| Attention projections | BFP8 weights, HiFi2 | Same |
| MLP gate/up/down | BFP4 weights, LoFi | Same policy; additional DRAM-sharded weights materialized at setup |
| Projection compute | Approximation off, FP32 destination off, packer L1 accumulation on | Same |
| Residual/norm activation | BF16, interleaved | BF16, L1 width-sharded across 8 cores |
| MLP activation/output | Interleaved | Width-sharded MLP chain, DRAM output at block boundary |
| MLP DRAM program | Absent | `in0_block_w=8`, `per_core_M=1`, `per_core_N=ceil(N/256)`, one reader per DRAM bank |
| Linear state | FP32 recurrent tensor; BF16 convolution history | Same allocation/update/restore implementation |
| Test geometry | Real layer-0 input, batch 1, prompt 2048, token at 2048 | Same |

Linear attention does not consume paged KV cache, current-position, or RoPE inputs; cache-page boundary theories do not explain this failing case. The full-attention case was not reached with `-x`.

## Ranked hypotheses and smallest verify/refute experiments

### H1. Sharded residual/norm behavior differs between first eager and subsequent execution

**Leading boundary to test, mechanism unverified.** The changed path performs two sharded RMSNorms and residual additions (`tt/optimized_decoder.py:101–114,143–172`). The norms have identical tensor geometry but different gamma weights, so cached tensor binding and first-call versus subsequent behavior deserve a direct check. The harness currently compares the first eager decode with a much later replay.

Source does **not** support simply changing `inplace`: it is already explicitly `False` at line 111. `LayerNormDeviceOperation::create_output_tensors` allocates a distinct output unless `program_config.inplace` is true (`ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:482–496`). The sharded implementation supplies current input, weight, and output as run arguments; there is no visibly missing gamma binding (`sharded_layernorm_factory_helpers.cpp:1883–1902`). That does not prove runtime/cache correctness.

First run the reproduction with each config separately, holding the policy, activations, and test constant:

| Control | `ORNITH_OPT_CONFIG` | Interpretation |
| --- | --- | --- |
| C0 | `{}` | Current-source baseline |
| C1 | `{"residual_cores":8}` | Residual conversion, sharded norms/adds; interleaved MLP weights |
| C2 | `{"dram_roles":["gate_proj","up_proj","down_proj"],"cores":8,"block_w":8}` | DRAM-sharded MLP with interleaved residual/norm |
| C3 | Original failing config | Combination |

Use `-k linear_attention` for these initial controls. Do not run them concurrently with the coordinator's device work. C1 failure localizes away from the DRAM-sharded MLP; only C3 failure points to composition/aliasing/allocation pressure. C2 failure moves priority to H2.

For **each decoder name**, collect this short diagnostic sequence before timing:

1. Save pre-decode state `S`; execute eager `E1`; snapshot resulting state and output.
2. Restore `S`, read it back exactly, execute eager `E2`; compare output and post-state exactly with `E1`.
3. Restore `S`, capture, restore `S`, execute **one** replay `T1`; compare output and post-state with `E2`.
4. Restore `S` before each of three further one-replay checks. Then retain the original 32-replay stress and post-stress one-replay checks.

`E1 != E2` disproves a trace-only explanation. `E1 != E2 == T1` identifies first-call/cached-execution behavior; do not hide it by silently dropping the first correctness sample. `E1 == E2 != T1` identifies a capture/replay boundary. Immediate equality followed by later failure prioritizes H3.

If C1 fails, isolate `_norm` with the exact `[1,1,4096]` BF16 recorded token, then test **two consecutive same-shape norms with different gamma tensors** and each result converted to DRAM. Compare repeated eager and replay outputs and check the input remains unchanged. A runtime-only control can temporarily implement `_norm` via interleaved conversion, the inherited norm, and conversion back to the same width shards; keep the rest of C3 unchanged. Passing localizes the issue to the norm boundary, but does not alone prove its particular kernel or arithmetic cause.

### H2. DRAM-sharded MLP composition or tensor binding fails despite a standalone projection passing

**Plausible; standalone gate evidence lowers the likelihood of a universal matmul defect.** `_activate_mlp` executes same-shape gate and up projections with different weight buffers, fused SiLU/multiply, and down projection. Outputs remain sharded for batch 1 (`optimized_decoder.py:117–137,174–185`). C2 isolates this group.

Focused experiment: take the exact recorded post-attention normalized activation and compare the complete gate/up/SiLU-multiply/down chain eager and traced. Check gate, up, activation, and down boundaries. Include the complete gate/up pair: a one-weight matmul probe does not exercise switching two same-signature weight buffers in one trace. If only down diverges, test its exact `[1,1,12288] × [12288,4096]` geometry independently. Keep BFP4/LoFi throughout this initial localization.

Source audit found no missing matmul tensor binding: the DRAM factory represents activation and output as tensor-backed CBs 2/6 (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:565–635`), and weight runtime argument 1 as a tensor reference (`:907–916`). These bindings still require execution evidence, especially when the program is reused for gate and up.

Input mutation is not established. `_linear`'s batch-1 reshapes preserve shape; it does not explicitly deallocate either the folded view or original input. The DRAM reader can zero an input's final partial K tile, but that branch is disabled here: K is 4096 or 12288, divisible by 32 (`factory:1047`; `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp:100–118,161–179`). Do not blame that padding path for this shape without contrary lowered-shape evidence.

The prior block-width-16 run hit a real static-CB L1 capacity check (`mlp_dram_c8_b16_v3.log`, 1,602,560 bytes versus 1,572,864). The block-width-8 run completes. This makes exact L1 allocation/config logging useful, but **does not prove an overlap** in the passing-capacity run. A smaller legal block width is an allocation-pressure control, not an accepted fix by itself.

### H3. Replay corrupts persistent input/weights/state, or restore is incomplete

**Lower source support; directly testable.** Snapshot persistent token `d`, both norm weights, convolution taps, recurrent/convolution buffers, and selected projection weights before capture. Check immutable tensors after one replay and after stress; compare every state buffer exactly to `S` immediately after restore. Record buffer IDs/addresses before and after. Persistent buffers must retain both identity and value where appropriate.

For linear attention the snapshot covers all mutated model state: `_snapshot_state` captures recurrent and all convolution tensors (`test_functional_decoder.py:783–796`); restore constructs host-only tensors and copies into those same device allocations (`:799–811`). GDN convolution uses in-place history copies (`fused_decoder.py:588–600`), while the recurrent update writes `self.recurrent_state` in place (`:622–652`). No additional mutable optimized cache is evident.

If output divergence has identical eager/trace post-state, prioritize the MLP/residual tail. Divergent convolution history points at normalized input/projection/FIR or persistent corruption before the recurrence. Equal convolution history but different recurrent state localizes farther into GDN. Avoid changing recurrence precision before making these comparisons.

## Explanations not supported by current evidence

- **The previous cross-decoder live-trace allocation bug:** the current harness finishes replay/readback and releases each trace at line 80 before constructing the next decoder. No later decoder executes while an older trace is live. Host state snapshots do not allocate replacement device state. The prior coexisting-traces diagnosis is not directly applicable.
- **Capture executes an extra recurrent step:** restore occurs before every timed window and before the final one-replay check. Whether capture executes its body cannot explain the final comparison by itself.
- **Missing program warmup:** an eager call uses the same decode mode, shapes, config, and input values before capture, and capture completes without unsupported-write failure. For a precise cache diagnostic, forbid program-cache misses around capture and restore the setting afterward; do not substitute extra warmups for localization.
- **Intrinsic BFP4 numerical drift:** both sides of the failed check use the same policy and weights. The interleaved BFP4/LoFi control already passed this exact-equality gate. Higher precision can be a later A/B probe, not a source-supported fix.

## Instrumentation and disposition

Keep diagnostics in the optimized stage's tests/runtime/doc scope. For capture-boundary probes, preallocate **all** device probe destinations before capture and copy into them using captured device operations; perform host readback after replay. Merely retaining Python references cannot protect explicitly deallocated intermediate buffers. Extra probes change allocation/lifetime and may mask an allocation-sensitive failure, so confirm any repair with the uninstrumented original reproduction.

The next authorized work is C0/C1/C2/C3 plus named E1/E2/T1 output/state comparisons, followed by the implicated standalone chain. Do not batch speculative norm, memory-layout, and precision fixes. Preserve exact own-eager/trace equality and the existing real-HF output gates. No performance conclusion or implementation fix follows from this source-only report.
