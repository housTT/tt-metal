# AutoFix: low-level scheduler sampling state

## Verified source-level contract gaps

1. No public method updated low-level sampling parameters. Private
   `_configure_sampling` released traces on a mode change, after which the live
   stream hit `ensure_traces`' prohibition on warming over live cache.
2. Slot-specific low-level prefill invoked the 32-lane common sampler, overwriting
   ongoing slots' next-token inputs, advancing their RNG seeds, and updating their
   output penalty histories even though those slots had no new prefill logits.
3. Low-level prefill did not populate per-slot prompt penalty masks or reset only
   the newly started slots' output histories. High-level generation had its own
   separate prompt initialization, which did not provide the low-level contract.

These findings come from generator/common-sampler source and were not diagnosed
as accelerator kernel failures. Root's high-level batch-32 page-permutation
investigation follows a different path: complete active-batch sampling, not this
partial low-level prefill helper.

## Implemented contract

`configure_sampling(sampling_params=None, *, reset_seed=False,
prompt_token_ids=None, generated_token_ids=None)` is now public. It changes mode
at an explicit scheduler boundary. For a live stream it snapshots sampling's
mutable tensors, releases old traces, warms only the terminal sampler, restores
tokens/logits/seeds/output penalty counts, and recaptures. It does not execute a
model forward, clear the cache, or refresh token/position/page tensors from host.
Seed state is preserved unless the caller explicitly requests `reset_seed=True`.

When enabling penalties on a live stream which had not tracked them, the caller
must supply prompt and generated histories for every fixed slot. This avoids
inventing missing historical counts. Optional explicit histories can replace
those sampler histories; otherwise an already active penalty mode preserves them.
An unbound initial trace now also invalidates when the first explicit sampling
configuration differs. The key includes `num_logprobs` as well as mode flags.

Low-level prefill prepares the common sampler's slots-aware prompt masks and
clears only newly started slots' output counts. Continuation prefill preserves
existing output counts and extends the prompt shadow. Sampling still uses the
common implementation, then merges only the selected lanes back into persistent
token, seed, and penalty buffers on device. No full-logits readback or custom
sampling algorithm is introduced. High-level generation delegates prompt-mask
preparation to that same low-level helper.

All added snapshot, masking, and recapture work occurs at prefill/configuration
boundaries. The unchanged decode replay loop adds no per-token host work.

## Verification and required hardware follow-through

The source-only harness executes the real generator methods with fake TT
boundaries. Eight tests pass in `generator_host_contract_tests.log`, including
restoration of all sampler warmup mutations, unchanged position/page inputs,
rejection of enabling untracked live penalties without explicit histories, and
preservation of other slots during partial sampling. Existing host sampling/EOS
tests also remain passing. The generator and hardware script were Black formatted.
No device command or TTNN import was run by this agent.

Required reduced real-hardware probe (written, **not run**):

```bash
PYTHONPATH=. python_env/bin/python \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/scheduler_sampling_contract.py \
  --batch 4 \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/scheduler_sampling_contract.json
```

The script loads real layers 0 and 3, starts two live fixed slots with penalties,
alternates sampled/greedy parameters, compares all-chip cache and persistent
input/sampler tensors exactly, forbids cache reset and per-token host writes,
then prefills another slot and checks ongoing prompt/output penalty histories,
feedback tokens and RNG seeds before continuing decode. Run under the parent's
normal serialized hardware environment/timeout and watcher policy.

Status: source contract repaired; hardware validation remains required. In
particular the new exact-shape UINT32 TILE `where`/row-major conversion and INT32
broadcast row masks must pass the written probe before this repair is considered
proven on device. The parent stage owner owns that final acceptance or follow-up.

## Follow-through after scheduler_sampling_v1 failure

Root ran the real probe under `TT_METAL_TRACE_ALLOC_ON`. Live mode changes and
full cache/input/sampler-state preservation checks passed, but slot-specific
prefill failed `ongoing token feedback changed`. Exact log:
`logs/scheduler_sampling_v1.log`. This was treated as required repair work.

Source localization found a concrete dispatch defect at
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp:529`:
`get_compute_defines(WHERE, dtype)` selects Float32 and Int32 explicitly, while
UInt32 takes the Float16_b branch. `ternary_program_factory.cpp` passes the
predicate dtype to this selection. Blackhole `ckernel_sfpu_where.h` selects
`LO16` loads/stores for Float16_b, versus `INT32` loads/stores for all 32-bit
formats. Thus a UINT32 0/1 predicate truncates UINT32 token and seed payloads
rather than preserving their upper bits. This is a causal source explanation,
not an assumption that tilize/untilize numerics are inherently unstable.

Minimal generator repair: upload the lane predicate as INT32. Its exact 0/1
values select the 32-bit WHERE LLK, while the true/false/persistent tensors remain
UINT32. INT32 WHERE moves the complete bit pattern, including seeds above 2^31;
no token/seed cast or precision reduction occurs. No C++ files were changed.

The hardware script now has `--merge-only` and always runs an isolated merge
first. Its 32 distinct old/new UINT32 values include nonzero upper 16 bits and
values above 2^31. It separately checks tilize identity, records unsigned-control
versus signed-predicate WHERE outputs, and checks untilize plus persistent copy
on every chip. The signed result must exactly match the Torch selection. Partial
prefill failures now print full before/after token and seed vectors for diagnosis.
The parent must run this probe and the original reduced scheduler test.

Coordinated related batch repair: `_write_positions` now uploads active masks
using each persistent tensor's dtype, rather than hard-coded FP32. The independent
batch agent proved the same WHERE dispatch issue with BF16 convolution state and
is changing model-owned mask allocation/transfer types; this generator edit keeps
its host upload compatible while leaving FP32 recurrent masks unchanged.

## Request-inclusive TTFT correction

Moved `generate`'s TTFT start immediately before request reset, sampling parameter
configuration, trace readiness and seed/penalty initialization. `ttft_s` now
includes that request work through first-token availability. It additionally
reports `request_setup_s` and `prefill_only_s` for the former isolated boundary.
Warmed repeated benchmark iterations naturally avoid first-call compilation;
older TTFT numbers need rerunning under this corrected boundary.

The source-only suite now has nine passing tests, including a deterministic
clock test proving reset occurs after the TTFT timer begins and that setup plus
prefill equals reported TTFT. No hardware was used by this repair agent.

## Hardware acceptance

Parent run `logs/scheduler_sampling_v2.provenance.json` exits0 with trace
allocation tracking and allocation tracebacks enabled. It reproduces the UINT32
predicate control failure and proves the INT32-predicate merge bit-exact for
large32-bit values on every rank. All four scheduler checks pass: live
sampling-mode reconfiguration preserves state, replay after reconfiguration
requires no host input copy, partial prefill preserves ongoing histories/token/
RNG state, and a new request joins live decode. See `scheduler_sampling_v2.json`.
