# Sampler decision

Decision: select `models.common.sampling.generator.SamplingGenerator` and reject
`models.common.modules.sampling.sampling_1d.Sampling1D` for this full model.

`SamplingGenerator` is the existing state owner used by the shared
`tt_transformers` generator.  It captures sampling, writes the sampled token to
the persistent model input, and owns per-slot seeds, penalties, top-k/top-p,
log-prob state, and reset.  Temperature-zero greedy is normalized to top-k=1 and
top-p=0, so it has true argmax semantics without a full-vocabulary host gather.

`Sampling1D` is a stateless kernel wrapper.  It does not own the matching trace,
token-feedback, request-slot, seed, penalty, or log-prob lifecycle.  Selecting it
would require a second custom sampling runtime despite an established common
path, so it is rejected.  No custom sampler is needed.

The selected path was compared against host argmax for 16 teacher-forced steps
using two real decoder layers and the real LM head.  Every token matches.  The
device arm records zero host argmax calls and zero full-logit reads.

The multi-chip regular top-k path is retained instead of the optional
force-argmax mode.  Repository policy identifies the regular sharded top-k path
as the multi-chip fit, and profiler evidence confirms no bottleneck: a steady
capture measures 27.500 us for `SamplingDeviceOperation` versus 708.553 us for
the TP-sharded LM head.  Sampling is 3.88% of LM-head device time and does not
dominate token-out.
