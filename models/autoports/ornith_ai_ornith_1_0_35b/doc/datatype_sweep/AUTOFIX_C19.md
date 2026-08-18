# AutoFix Report — C19 (`expert_act_dtype = bfloat4_b`)

## Starting Evidence

No `AUTOTRIAGE.md`/`AUTODEBUG.md` existed for this failure. The starting report was the stage's own
blocker record plus the captured triage, which `$autotriage` had already produced from the live
stall:

* [`blocked/C19-expert-act-bfp4.json`](blocked/C19-expert-act-bfp4.json) and
  [`blocked/C19-expert-act-bfp4.config.json`](blocked/C19-expert-act-bfp4.config.json)
* [`logs/smoke/C19-expert-act-bfp4.txt`](logs/smoke/C19-expert-act-bfp4.txt) — the failing console
* [`triage/tt-triage-C19.txt.gz`](triage/), [`triage/triage-summary-C19.txt`](triage/)
* README §5.2, `work_log.md` §8

Original failing command (~50 s reduced two-layer build):

```bash
python models/autoports/ornith_ai_ornith_1_0_35b/doc/datatype_sweep/logs/smoke_policy.py \
  --config models/autoports/ornith_ai_ornith_1_0_35b/doc/datatype_sweep/blocked/C19-expert-act-bfp4.config.json \
  --output /tmp/c19_smoke.json
```

Observed: `TT_FATAL: Writes are not supported during trace capture. trace id: 0` on all four devices
inside `OrnithGenerator._ensure_decode_trace`, then a mesh stall.

The `$autofix` loop was run **serially** — forked subagents were not available in this environment —
and every experiment below was a standalone script, so the original stalling command was never
re-run blind.

## Hypothesis Experiments

### H1 — `ttnn.typecast` bfloat4_b → bfloat8_b is not trace-capturable

* **Hypothesis.** The widening added to `OptimizedMoE._routed_experts` (`_MOE_REDUCE_DTYPES` /
  `MOE_REDUCE_FALLBACK_DTYPE`, `tt/optimized_decoder.py:1718`) takes a path that issues a host
  write, which trace capture forbids. This is what the blocker record asserted.
* **Experiment.** [`logs/autofix_c19/probe_bfp4_widen.py`](logs/autofix_c19/probe_bfp4_widen.py) —
  open the 1x4 mesh, allocate the decode `down` tensor (`[1, 64, 32, 2048]`, TILE, bfloat4_b) in L1
  and in DRAM, and capture a trace containing only `ttnn.typecast(..., ttnn.bfloat8_b)`.
* **Result.**

  ```
  RESULT L1   typecast  eager=ok dtype=DataType.BFLOAT8_B | traced=CAPTURABLE
  RESULT DRAM typecast  eager=ok dtype=DataType.BFLOAT8_B | traced=CAPTURABLE
  ```

* **Verdict: refuted.** The widening typecast is capturable in both memory configs. The blocker
  record's attribution of the trace-capture write to the typecast was wrong.
* **Evidence.** [`logs/autofix_c19/probe_bfp4_widen.txt`](logs/autofix_c19/probe_bfp4_widen.txt)
* **Fix.** None. No code change was needed or made for this hypothesis.

### H2 — some *other* op in the bfloat4_b routed-expert chain issues the host write

* **Hypothesis.** `expert_act_dtype` reaches only three places (`grep` over
  `tt/optimized_decoder.py`: the `_expert_mem` width estimate at :1431 and the `dtype=` of the two
  `ttnn.sparse_matmul` calls at :1660 and :1700), so the blast radius is the chain between the two
  sparse matmuls. One of those ops has a host-side path for bfloat4_b operands.
* **Experiment.**
  [`logs/autofix_c19/probe_routed_experts_trace.py`](logs/autofix_c19/probe_routed_experts_trace.py)
  builds a real `MultichipMoE` on the 1x4 mesh with **synthetic** expert weights (no checkpoint, no
  generator, no trace of the whole model), calls the real `OptimizedMoE._routed_experts` twice
  eagerly and once inside `ttnn.begin_trace_capture`, and lets the Python traceback name the line.
  It runs the shipped `selected_precision_config.json` first as a control.
* **Result.**

  ```
  policy=C06-proj-bfp4-lofi   expert_act_dtype=BFLOAT8_B : CAPTURABLE
  policy=C19-expert-act-bfp4  expert_act_dtype=BFLOAT4_B : NOT-CAPTURABLE
    File ".../tt/optimized_decoder.py", line 1651, in _routed_experts
      packed = ttnn.sparse_matmul(
    RuntimeError: TT_FATAL @ tt_metal/distributed/fd_mesh_command_queue.cpp:760:
      !trace_id_.has_value()  info: Writes are not supported during trace capture. trace id: 1
  ```

* **Verdict: verified.** The write comes from `ttnn.sparse_matmul` itself — the *first* op of the
  chain, upstream of the SwiGLU, the score multiply, the widening typecast and the reduction. Note
  that the eager calls succeed and return the right dtype: the op works, it is only uncapturable.
* **Evidence.**
  [`logs/autofix_c19/probe_routed_experts_trace.txt`](logs/autofix_c19/probe_routed_experts_trace.txt)
* **Fix.** None available at the model level — see H3.

### H3 — the mechanism is `sparse_matmul`'s output zero-fill, and bfloat4_b has no device-side fill path

* **Hypothesis.** From source: `SparseMatmulDeviceOperation::create_output_tensors`
  (`ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:311-336`)
  zero-fills its output with `ttnn::zeros_like` on **every** call — in both the fresh-output and
  the `optional_output_tensor` branch — because the blocks of inactive experts are never written by
  the kernel. `full_like_impl`
  (`ttnn/cpp/ttnn/operations/creation/creation.cpp:239-244`) takes the on-device `ttnn::fill` fast
  path only for `BFLOAT8_B`, `BFLOAT16` and `FLOAT32`; `BFLOAT4_B` falls through to `full_impl`,
  which builds a host tensor and `copy_to_device`s it (`creation.cpp:52-73`). That is the host
  write.
* **Experiment.**
  [`logs/autofix_c19/probe_sparse_zero_fill.py`](logs/autofix_c19/probe_sparse_zero_fill.py) —
  each spelling run warm and then under trace capture, on a 1x4 mesh, plus the reduction's own
  dtype contract.
* **Result.**

  ```
  zeros_like bf16 -> new      traced=CAPTURABLE
  zeros_like bfp8 -> new      traced=CAPTURABLE
  zeros_like bfp4 -> new      traced=NOT-CAPTURABLE  (fd_mesh_command_queue.cpp:760)
  fill bfp8 in place          traced=CAPTURABLE
  fill bfp4 in place          traced=CAPTURABLE          <-- the device fill *does* support bfp4
  sparse_matmul dtype=bf16    traced=CAPTURABLE
  sparse_matmul dtype=bfp8    traced=CAPTURABLE
  sparse_matmul dtype=bfp4    traced=NOT-CAPTURABLE
  sparse_matmul bfp4 + out=   traced=NOT-CAPTURABLE      <-- preallocated output does not help
  REDUCE bfp8: ok
  REDUCE bfp4: FAIL  DeepseekMoEFastReduceNC input only supports specific data types.
                     [BFLOAT16, BFLOAT8_B]  (moreh_helper_functions.cpp:285)
  ```

* **Verdict: verified.** `ttnn.zeros_like` on a bfloat4_b device tensor is exactly as uncapturable
  as `ttnn.sparse_matmul(dtype=bfloat4_b)`, and `ttnn.fill` on the same tensor is capturable — so
  the gap is the dtype list in `full_like_impl`, not any hardware or kernel limitation. Passing a
  preallocated `optional_output_tensor` does not avoid it, because that branch zero-fills too.
* **Evidence.**
  [`logs/autofix_c19/probe_sparse_zero_fill.txt`](logs/autofix_c19/probe_sparse_zero_fill.txt)
* **Fix.** None applied. The one-line upstream change (`creation.cpp:241`, add
  `DataType::BFLOAT4_B` to the fast-path condition) is a **core TTNN behaviour change affecting
  every model in the repo** and is out of this stage's scope; it is recorded here as the follow-up,
  with the evidence that it would work.

### H4 — the zero-fill is the *only* blocker; the rest of the bfloat4_b chain is capturable

* **Hypothesis.** If the sparse matmul's bfloat4_b zero-fill were device-side, C19 would capture:
  nothing else in `_routed_experts` needs a host write.
* **Experiment.** `probe_routed_experts_trace.py shim:<C19 config>` — the same real
  `_routed_experts`, with `ttnn.sparse_matmul` shimmed to run at bfloat16 and `typecast` down to
  bfloat4_b, so every downstream op sees exactly the operands C19 asks for while the op's own
  bfloat4_b zero-fill is removed. (The shim is a **diagnostic, not a candidate fix**: it
  materialises the full-width bfloat16 intermediate C19 exists to avoid.)
* **Result.** `VERDICT policy=C19-expert-act-bfp4 shim=True: CAPTURABLE`.
* **Verdict: verified.** The two unpacking slices, the SwiGLU multiply, the reshape, the score
  multiply, the widening `ttnn.typecast` and `deepseek_moe_fast_reduce_nc` are all capturable on
  bfloat4_b operands. The whole of C19 reduces to one op's output zero-fill.
* **Evidence.**
  [`logs/autofix_c19/probe_routed_experts_shim.txt`](logs/autofix_c19/probe_routed_experts_shim.txt)
* **Fix.** None. The shim was reverted with the probe process; it is not in shipped code.

### The neighbouring question — is `expert_act_dtype = bfloat4_b` even a meaningful request?

Yes, and it is orthogonal to the weight dtype. `policy.expert_act_dtype` is the `dtype=` of the two
`ttnn.sparse_matmul` calls, i.e. the **output activation**; the expert weights come from
`policy.expert_gate_up_dtype` / `expert_down_dtype`. The shipped
[`selected_precision_config.json`](selected_precision_config.json) (C06) already runs
`routed_expert_gate_up`/`routed_expert_down` at `bfloat4_b` with `routed_expert_output` at
`bfloat8_b`, and the H2 control confirms that combination captures. So asking for a bfloat8_b
output while keeping bfloat4_b weights is not only possible, it is what already ships — and C19's
narrowing of the *activation* is a genuinely distinct arm. It is simply unsatisfiable: the
reduction accepts only `[BFLOAT16, BFLOAT8_B]`, and a bfloat4_b sparse-matmul output cannot be
captured at all, so the two dtypes the reduction accepts are already the only two the op can be
asked for in a traced decode.

## Final Status

**Blocked, with the blocker proven and re-attributed.** C19 remains not measurable, but the cause
recorded in `blocked/C19-expert-act-bfp4.json` was wrong and has been corrected: the trace-capture
write is **not** the widening typecast (H1, refuted) but `ttnn.sparse_matmul` with a `BFLOAT4_B`
output dtype.

Proven blocker:

* **Op** — `ttnn.sparse_matmul` (`tt/optimized_decoder.py:1651`, the packed gate/up projection;
  the down projection at :1690 has the same property).
* **Assertion** — `TT_FATAL: Writes are not supported during trace capture. trace id: N`,
  `tt_metal/distributed/fd_mesh_command_queue.cpp:760` (`FDMeshCommandQueue::write_shard_to_device`).
* **Mechanism** — `sparse_matmul_device_operation.cpp:311-336` zero-fills the output with
  `ttnn::zeros_like` on every call; `creation.cpp:239-244` omits `BFLOAT4_B` from the device-fill
  fast path, so the fill becomes a host tensor plus `copy_to_device` (`creation.cpp:52-73`).
* **Not avoidable from the model** — the zero-fill runs in both the fresh-output and the
  `optional_output_tensor` branch, and no `ttnn.sparse_matmul` argument disables it.
* **The pre-existing blocker 1 is confirmed verbatim** —
  `TT_FATAL: DeepseekMoEFastReduceNC input only supports specific data types. [BFLOAT16, BFLOAT8_B]`,
  `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.cpp:285` — the console the record said had
  been overwritten is now reproduced in `logs/autofix_c19/probe_sparse_zero_fill.txt`.

Commands that prove the final state:

```bash
python .../doc/datatype_sweep/logs/autofix_c19/probe_bfp4_widen.py typecast
python .../doc/datatype_sweep/logs/autofix_c19/probe_routed_experts_trace.py
python .../doc/datatype_sweep/logs/autofix_c19/probe_sparse_zero_fill.py
python .../doc/datatype_sweep/logs/autofix_c19/probe_routed_experts_trace.py \
  shim:.../doc/datatype_sweep/blocked/C19-expert-act-bfp4.config.json
```

All four exited 0 and closed the mesh cleanly; none stalled the hardware, so no reset was needed at
any point in this pass.

Shipped code: **unchanged.** The `_MOE_REDUCE_DTYPES` widening guard stays exactly as it was — H1
proves it is correct and capturable, and it dispatches nothing under any shipped policy. Only the
prose that misattributed the failure was corrected (this file, `blocked/C19-expert-act-bfp4.json`,
README §5.2, `work_log.md` §8, and the comment above the guard).

Remaining risk / follow-up: the upstream one-line fix at `creation.cpp:241` is untested — it needs a
tt-metal rebuild and a regression pass over every op that zero-fills a block-float output, which is
a TTNN change rather than a model change. If it lands, C19 becomes measurable with the existing
widening guard and nothing else; the expectation from C13 (−1.33 %) and C14 (−1.87 %) is still that
it would lose on measurement, since this decode step is launch-bound.
