# Ornith-1.5-9B TTNN autoport

**Full-model warm TTFT: 47.46 ms. Trace-verified batch-1 token-out decode:
81.56 tokens/s/user (12.26 ms/token).** Measured with all 32 layers,
prompt128/generate128 and a native262144-token cache on **four Blackhole chips
on two P300c boards**, using the optimized TP4 ring. `p150x4` is the software
profile name.

The separate logits-only trace measures85.61 tokens/s; traced AIME teacher
forcing measures80.85 tokens/s/user and explicitly supplies reference tokens
from the host. Neither replaces the on-device token-feedback headline above.
TTFT includes request reset/configuration, prefill and first-token sampling;
weight loading and initial compilation/capture are outside the warm result.

Prefill accuracy is96% top-1,100% top-5 and100% top-100. Teacher-forced decode is
94%,100% and100%, using the fresh HF chat-template AIME24 reference with100
continuation tokens. The final six-prompt HF/TT qualitative check passes its
128-token coherence/regression window; finished-answer quality is not claimed
for outputs truncated during reasoning.

[Full-model report](doc/full_model/README.md),
[measurements](doc/full_model/perf_final_v2.json),
[context contract](doc/context_contract.json), and
[work log](doc/full_model/work_log.md) contain policies, commands, exact evidence,
rejection decisions, and the final independent review/checkpoint status.
