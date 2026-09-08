# Ornith-1.5-9B TT serving

**Primary vLLM warm TTFT:35.25ms; decode:87.70 tokens/s/user**, measured with
128 input/128 output/1 request, concurrency1 and max-num-seqs1 at native
context262144. All32 layers run through the TT plugin on four Blackhole chips
on two P300c boards, TP4 mesh1x4 (`P150x4`). The same workload before serving
optimization measured49.96ms TTFT and87.88t/s/user.

[Serving report](doc/optimized_vllm/README.md) records complete before/after
metrics, secondary CI capacity results, commands, correctness and gate status.
The current standalone traced128-input/128-output/B1 control is87.96t/s/user;
serving decode is within0.3%. The selected datatype-sweep policy and native
context are preserved.

[Precision and accuracy evidence](doc/datatype_sweep/README.md),
[context contract](doc/context_contract.json),
[serving integration](doc/vllm_integration/README.md), and
[optimization work log](doc/optimized_vllm/work_log.md).
