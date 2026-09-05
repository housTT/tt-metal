# AutoFix: real-input BFP4 cache failure

## Starting evidence

Source diagnosis: [AUTODEBUG_cache4_batch.md](AUTODEBUG_cache4_batch.md).
Original failure: `final_release_short`, full-attention B32, prefill96 then
decode96, per-user PCC 0.9936757125704948 against the unchanged 0.995 gate.
This verifier changed only a new diagnostic/regression test and this report
before obtaining controlled results. Hardware commands ran serially; no device
recovery was needed. No functional/fused source or block fallback was changed.

## Hypothesis experiments

1. **Batch-specific cache/kernel behavior:** refuted. Actual recorded prefix
   rows `(39 + 137*u + arange(96)) % N` and token row `(131 + 137*u) % N`
   fail for users **8 and 26**, both in the original B32 and when all 32 users
   are individually extracted to B1. B32 minimum PCC is 0.9936757125704948;
   B1 minimum is 0.9937164783974881. A B>1-only precision policy would retain
   known-invalid B1 inputs.
2. **Cumulative attention/cache precision:** verified as a sensitivity.
   Cache-only K8/V8 passes all 32 original users and all 32 extracted B1
   users. Keeping K4/V4 while using BF16/HiFi4 attention projections and
   HiFi4/FP32 decode SDPA also passes. This is a localization control, not an
   unmeasured production fallback or a claim of an inherent BF4 kernel limit.
3. **Key vs value boundary:** K8/V4 passes; K4/V8 fails user8 in B32 and B1.
   Mixed caches use **two independent `paged_update_cache` calls**. They
   never invoke the unsupported mixed-dtype fused update. The source hazard
   documented in AutoDebug was respected throughout.
4. **Page addressing / update / rounded SDPA read window:** refuted for this
   exact failure. Actual model allocation is `[1024,4,64,256]`, with B32
   `[32,32]` identity and independently permuted disjoint tables. Position96
   is page1, offset32; fixed Kchunk256 reads a rounded window ending256,
   inside 2048 backed positions per user. Both probes prove fused cache
   contents equal independent unfused updates **bit-for-bit**, only mapped
   row96 changes, and all other physical entries remain exact. CPU SDPA on
   the captured query and dequantized actual cache agrees at PCC
   0.9998792721946472 for both table mappings. The decoder is thus reading
   the low-precision contents that were written, not a misaddressed user.

| Policy | Min B32 decode PCC | Min extracted-B1 decode PCC | Failed users |
| --- | ---: | ---: | --- |
| K4/V4, BF4/LoFi attention | 0.993675713 | 0.993716478 | 8,26 |
| K8/V8, BF4/LoFi attention | 0.996978225 | 0.997006229 | none |
| K4/V4, BF16/HiFi4 attention + precise SDPA control | 0.995715590 | 0.995701183 | none |
| K8/V4, BF4/LoFi attention, separate updates | 0.995823360 | 0.995843682 | none |
| K4/V8, BF4/LoFi attention, separate updates | 0.994855963 | 0.994926688 | 8 |

The remaining ledger is constant: BF16 activations; BF4/LoFi MLP; original
host-packed weights; DRAM tile caches, replicated mapper, page64, four local
KV heads, head256; selected sharded DRAM projection geometry; no CCL.
Actual policy, cache shape/dtype and per-user arrays are recorded in
[controls](logs/autofix_cache_precision_controls.log) and the compact
[summary](cache_precision_summary.json).

### Oracle correction, preserved failure evidence

The first exact-cache probe used host `from_torch(BF16_update, dtype=BF4)` as
its expected freshly quantized row. That does not match device cache-update
packing bit-for-bit (13,369 K and 13,574 V differing elements). This failed
oracle is preserved in `autofix_cache_exact_rows`; it was not silently
relabeled a pass. The adapted probe uses independently cloned original
caches and separate device `paged_update_cache` operations as the packing
oracle, then separately enforces the exact mapped-row/untouched-entry
contract. Its identity and permuted results both pass. No snapshot tolerance
was added: host readback/from-torch restoration of *already quantized* cache
contents remains exact, as proved by both policies' full-state trace checks.

### Correct-policy latency and deterministic replay

The unchanged T2048 recorded-input pair harness measures seven warmed
prefills (discard two) and seven groups of 32 traced decodes (discard two).
Snapshot restoration occurs outside timed windows. Both policies additionally
pass four restored B32 output/state replays, 32 repeated trace executions,
then another restored exact replay, with immutable public inputs.

| Correct policy | Warm prefill ms | Traced warmed decode ms | HF prefill/decode PCC |
| --- | ---: | ---: | --- |
| K8/V8 | 6.395060918 | **0.427813531** | 0.998705091 / 0.999059938 |
| K8/V4, separate updates | **6.297976011** | 0.432901998 | 0.998283644 / 0.998820596 |

The archived mixed-policy pair interceptor also changed its equal-dtype fused
baseline to separate update calls. The comparison above uses only each
optimized-policy timing; those paths and timings are unaffected. The delivered
interceptor now delegates equal-dtype calls to the original fused operation,
so future paired baselines preserve the original fused topology. The final
production pair is independently rerun by the main agent.

K8/V4 improves prefill but loses decode by 1.19%, so it is rejected under the
stage's strongest-correct-traced-decode criterion. K8/V8 is the minimal
verified repair. The old K4/V4 B1 timing is not an acceptable baseline winner
because it fails other real B1 inputs. No capability reduction is needed.

## Exact commands and artifacts

All commands use:

```bash
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8 ORNITH_WEIGHTS=real
D=models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder
T=models/autoports/ornith_ai_ornith_1_5_9b/tests/test_cache_precision_regression.py
python "$D/record_run.py" autofix_cache_precision_controls python -m pytest -q -s "$T" -x
python "$D/record_run.py" autofix_cache_exact_rows python -m pytest -q -s "$T" -k exact_cache -x
python "$D/record_run.py" autofix_cache_exact_rows_unfused_oracle python -m pytest -q -s "$T" -k exact_cache -x
python "$D/record_run.py" autofix_cache_correct_pair_trace python -m pytest -q -s "$T" -k 'precision_pair or restored_trace' -x
```

The first command's archived source contains the original five controls;
subsequent focused tests were added afterward. Results are respectively
**5 passed**, **1 failed** (host packing oracle), **2 passed**, **4 passed**.
Each run has an exact `.provenance.json`, source archive `.sources.json.gz`,
and `.log` under `logs/`; names are immutable and must not be overwritten.
The new test is `tests/test_cache_precision_regression.py`.

## Final status

**Cause localized; minimal verified fix integrated with main-agent agreement.**
The optimized allocator now defaults globally to K8/V8 while preserving
explicit dtype arguments. No runtime input- or
batch-dependent precision fallback. The main agent must rerun final default
stage gates, including native context, Watcher and final profiling, then
independent stage review. The failing original B32 workload, all extracted
B1 users, and exact restored state already pass the candidate allocator.
