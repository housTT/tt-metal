# AutoFix: sharded residual trace disagreement

## Starting evidence

Investigation followed [AUTODEBUG_sharded_trace.md](AUTODEBUG_sharded_trace.md), the failed `mlp_dram_c8_b8` provenance/source archive, and the AutoFix/device-usage/trace skills. Hardware commands were serialized under an exclusive lane. No prior-stage source was changed.

**Verified cause:** the first linear-attention residual addition receives BF16 residual and FP32 GDN output in width-sharded L1. The mixed-dtype `ttnn.add` produces different values on repeated eager calls and immediate trace replays despite exactly unchanged operands. The earliest divergence is the add output. This is not a trace-only failure, MLP precision drift, or state-restoration failure. The lower-level kernel mechanism remains uninvestigated; no C++ change was made.

## Hypothesis experiments

All model runs used real weights, recorded layer inputs, prompt 2048 / decode position 2048, attention BFP8/HiFi2 and MLP BFP4/LoFi. The added harness names each decoder and checks E1/E2, four restored immediate replays, seven windows of 32 replays followed by restored replay, exact recurrent/convolution or KV state, and immutable token/norm weights. State restores are read back and checked exactly.

| Experiment | Result and interpretation | Log |
| --- | --- | --- |
| C0: `{}` | Fused and optimized exact throughout. | [C0](logs/autofix_c0.log) |
| C1: `{"residual_cores":8}` | Optimized E1/E2 max error 0.072265625; E2/T1 0.07958984375. All recurrent/convolution state and immutable checks exact. Fused exact. | [C1](logs/autofix_c1.log) |
| C2: DRAM gate/up/down, cores 8, block width 8 | All checks exact. Refutes H2 as the cause of this failure. | [C2](logs/autofix_c2.log) |
| C3: C1+C2 | Optimized eager/replay failure persists, with exact state. | [C3](logs/autofix_c3.log) |
| Two same-shape sharded norms, different gamma weights | E1/E2/E3/replay and unchanged input all exact. Simple norm cache/gamma binding hypothesis refuted. | [norms](logs/autofix_two_norms.log) |
| Eager boundary readbacks | Attention norm, GDN Z projection, GDN output, and both first residual-add operands exact. Add output differs (max error 0.0682373046875). FF norm receives that already-corrupt input. | [boundaries](logs/autofix_eager_add.log) |
| Exact recorded residual-add operands, preserving BF16/FP32 | Mixed sharded add nondeterministic E1/E2/E3/T1–T4. Homogeneous BF16 and mixed interleaved controls deterministic. | [dtype controls](logs/autofix_residual_add_dtype.log) |
| Homogeneous FP32 sum followed by BF16 cast | Exact eager/replay and unchanged inputs. Chosen adaptation preserves FP32 update until addition. | [FP32 control](logs/autofix_residual_add_fp32.log) |
| Durable component repro, recorded BF16 token + seeded FP32 update | Raw primitive fails eager/replay (max error 0.1015625), unchanged inputs. Runtime adaptation is exact across all calls and exactly equals `(a.float()+b).bfloat16()`. | [raw](logs/autofix_residual_primitive_repro.log), [fixed](logs/autofix_residual_regression.log) |

The first standalone add probe accidentally converted the saved FP32 operand to BF16; it did not reproduce the bug. Its padded-logical-row control was illegal because reshape cannot change logical volume. These were rejected probes, not evidence for a fix; retained in `autofix_residual_add.log`. The corrected dtype-preserving experiment above establishes the cause. Original exploratory test source is preserved in `logs/autofix_boundary_experiments.py`; pre-fix runtime in `logs/autofix_optimized_before.py`.

H1 is verified at the residual-add boundary, with the norm-specific suspicion refuted. H2 is refuted for this case. H3's state-corruption explanation is refuted by exact restored/eager/replay state and exact add operands; the standalone stateless reproduction confirms it is unnecessary to explain the symptom.

## Fix and verification

`OptimizedDecoder._residual_add` promotes mismatched sharded operands to FP32, adds them in the existing shards, then casts the sum to the residual dtype. Both block residual additions use this helper. Matching-dtype and interleaved additions retain the previous path. No precision-policy, cache, recurrence, or HF threshold changes.

- [C3 fixed](logs/autofix_c3_fixed.log): both linear/full-attention cases pass every exact output/state/restore check before and after stress.
- [Original pair fixed](logs/autofix_original_fixed.log): original uninstrumented two-case command passes exact own-eager/replay equality and all HF gates. Linear prefill/decode/stress HF PCC: **0.9984640467 / 0.9998637616 / 0.9998063501**. Full-attention prefill/decode HF PCC: **0.9980597998 / 0.9997966152**.
- [Component regression](logs/autofix_residual_regression.log): exact Torch BF16-rounded sum, repeated eager, four trace replays, unchanged inputs.
- Watcher verification is recorded in `logs/autofix_watcher.log` and its separate watcher directory; see final status below.
- Python-only change: no build required. `python_env/bin/black tests/test_optimized_trace_regression.py` and `python_env/bin/python -m compileall -q` on the changed test/runtime passed. Runtime formatting was left for the coordinator's stage-wide formatting pass to keep this patch narrow.

Reproduction/verification commands from repository root (the existing virtual environment, `TT_METAL_HOME`, and `TT_METAL_CACHE` were preserved):

```bash
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export ORNITH_WEIGHTS=real
export ORNITH_OPT_POLICY='{"mlp_gate_up":"bfloat4_b","mlp_down":"bfloat4_b","mlp_fidelity":"LoFi"}'
export ORNITH_OPT_CONFIG='{"residual_cores":8,"dram_roles":["gate_proj","up_proj","down_proj"],"cores":8,"block_w":8}'
python_env/bin/pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimization_experiments.py -x -v -s
python_env/bin/pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_trace_regression.py -x -v -s
# Expected failure: raw mixed-dtype primitive, bypassing runtime adaptation.
ORNITH_REPRO_MIXED_ADD=1 python_env/bin/pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_trace_regression.py -k residual_add -x -v -s
# Separate watcher run, never combined with profiling:
env -u TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER=120 \
  TT_METAL_LOGS_PATH="$TT_METAL_HOME/models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder/logs/autofix_watcher" \
  python_env/bin/pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_trace_regression.py -x -v -s
```

C0/C1/C2/C3 used the same regression command with `-k linear_attention` and each config from the table. The test file was subsequently reduced to the durable restored-trace and residual-add tests; the exploratory-source snapshot retains the two-norm and boundary probes.

## Final status

The decoder failure is fixed with focused component and original-pair evidence. The raw TTNN mixed-dtype sharded-add primitive still fails and is preserved as an opt-in reproduction. No lower-level kernel repair, multichip/full-model/serving claim, or new performance conclusion is made here. Final watcher result and hardware release are appended after completion.

Final watcher run: **3 passed** in 34.06 seconds (linear/full restored-trace checks plus exact residual-add regression), no watcher errors/asserts, all devices detached cleanly. Watcher checks were enabled and recorded in `logs/autofix_watcher/generated/watcher/watcher.log`. No profiler was enabled. **Hardware lane released** after this run; no further device commands by this investigator.

Command/runtime provenance: [autofix_commands.provenance.json](logs/autofix_commands.provenance.json) records each exact command, environment, log hash, and the applicable runtime hash. Pre-fix runtime SHA-256: `6f1a99df19152ade07288b7993391b32057d50fd29430206587a94f37a011016`; verified post-fix runtime: `eb079e6202bfe82b0334f5c5d49d1dea1f848ac8b25bccdfe01f5bcc5e061e46`. [Final SHA ledger](logs/autofix_final_sha256.json) covers all logs and the final formatted regression harness. The evolving experimental harness was not hashed per execution; only its final exploratory snapshot and final durable source hashes are available. Those snapshots are not represented as exact per-run test provenance.
