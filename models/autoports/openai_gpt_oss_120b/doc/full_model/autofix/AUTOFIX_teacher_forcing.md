# AutoFix: traced teacher forcing

## Failure

The initial 36-layer `run_teacher_forcing` run scored top-1/top-5/top-100 of
0.47/0.57/1.00.  Prefill had already passed at 0.94/1.00/1.00, localizing the
failure to decode state rather than weights or prefill cache fill.

## Diagnosis and repair

Fresh-context AutoDebug (`AUTODEBUG_teacher_forcing.md`) showed that the shared
traced generator preserves the device-sampled token during async-ahead decode.
Without an explicit caller-authoritative marker, the teacher-forcing callback's
token could be replaced by the previous sampled token after the first
divergence.  A separate compatibility defect returned the host sampler's tuple
wrapper instead of its logits tensor.

The retained autoport-local fixes are:

- low-level `decode_forward(..., force_host_tokens=True)` marks fixed slots as
  freshly host supplied before delegation;
- high-level `generate` selects that mode only when `next_input` is present;
- the host sampler unwraps the tuple before logits indexing.

Free-running device generation is unchanged and still leaves Python token input
stale so `tt_out_tok` remains authoritative.

## Verification

The two-layer real-weight 16-step comparison then produced exact device and host
argmax predictions.  The device arm measured 103.7426 t/s/u and recorded zero
host argmax and full-logit reads; the explicit host arm measured 46.6153 t/s/u.
The original 36-layer gate was rerun and passed at top-1/top-5/top-100
0.95/1.00/1.00, with TTFT 3751.46 ms and decode 52.1145 t/s/u.
