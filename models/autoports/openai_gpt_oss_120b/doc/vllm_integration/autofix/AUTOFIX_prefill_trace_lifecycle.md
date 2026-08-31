# AutoFix: host compatibility and device trace lifecycle

## Failure

After the B1 trace-bucket optimization, the full sampling profile passed the
host-only cases and then hung during the first logprobs request's following
prefill. The server eventually raised a per-op timeout at a prefill transpose.
`tt-triage` showed device NoC activity while the host was blocked reserving the
next dispatch program.

The first hypothesis was that vLLM had selected an all-logits prefill path.
Source inspection refuted it: the plugin did not pass `return_all_logits`, and
the adapter deleted unknown kwargs and used last-token prefill logits. The
actual changed condition was eager host-logits decode allocating buffers while
B1/B8 device decode traces remained resident.

## Isolated candidates

1. **Capture a host-logits trace.** Reduced hardware captured B1, B8, and host
   traces, but repeated host requests hung after replay. Rejected.
2. **Release device traces before host compatibility sampling.** Reduced
   hardware crossed every host-only and first-logprobs case, then failed when
   returning to device sampling because a one-row prefill page table attempted
   to update the persistent B8 decode page-table buffer. Partially proven.
3. **Release/recapture plus separate prefill page-table routing.** Request-sized
   prefill tables remain active only for prefill; fixed B1/B8 decode tables are
   persistent trace inputs. Returning to device sampling recaptures both
   prepared trace signatures before the device-sampled prefill. Retained.

## Evidence

- Adapter unit suite: 20/20 pass.
- Reduced two-layer host->device transition: 7/7 pass.
- Reduced second lifecycle cycle: 2/2 pass.
- Final 36-layer full sampling profile: 72 passed, one expected skip, zero
  failures in 588.60 seconds.
- Final-code 36-layer lifecycle smoke: host compatibility followed by seeded
  device sampling, 2/2 pass; subsequent host snapshot flush 1/1 pass.
- Runtime capability snapshot: 190 device trace replays/token-out submissions,
  217 async output collections, four host trace releases, three device trace
  recaptures, and zero host argmax calls.
- Clean process audit after both full-model runs: no vLLM/EngineCore process and
  no device holder remained.

The retained change is outside steady device decode. File persistence occurs
only when the host/device trace lifecycle changes, so it does not add overhead
to the measured token-out loop.
