# AUTODEBUG: workspace reduced mixed-params sampling failure

## Scope

Source/log-only investigation of the workspace-local reduced two-layer GPT-OSS
120B vLLM smoke failure. I did not start vLLM, run hardware-facing commands,
reset/access TT devices, or edit implementation/test files. The AutoDebug nested
runner was not used because this task explicitly forbids spawning another
agent.

## Headline findings

1. The strongest current hypothesis is duplicate-seed salting in the GPT-OSS
   120B vLLM path. The common `SeedManager` defaults to
   `salt_duplicate_seeds=True`, while its own vLLM contract comment says
   independent concurrent requests carrying the same explicit seed must keep
   salt `0` so they stay bit-identical across batch order changes.
2. The failed test is specifically designed to catch that class of bug:
   `test_mixed_params_batch` submits several independent requests with
   `seed=42`, then shuffles request/config order for the second run and asserts
   each seeded logical request reproduces.
3. The failure is consistent with an order-dependent RNG stream, not with a
   server crash, trace lifecycle failure, or host-vs-device sampling mismatch.
   The server became healthy, performed on-device sampling, and shut down
   cleanly. Only the seeded temperature/presence-penalty mixed-batch request
   failed.

## Failure evidence

`models/autoports/openai_gpt_oss_120b/readiness_vllm/failed_attempts/20260831T1504Z_workspace_reduced_smoke_sampling.log`

- Pytest collected four smoke items.
- Only `vllm-tt-plugin/tests/tt/test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch`
  failed.
- `test_top1_is_greedy` passed, host-only `min_p` passed, and chat all-vocab
  logprobs was skipped.
- The deterministic request failed after the same leading fragment:
  - config: `temp=0.5,seed=42`
  - run 1: ` voclags Hamb Brook Maidr hiponza grab dominating`
  - run 2: ` voclags Courtney cv BU rechone drolots-sp`

`models/autoports/openai_gpt_oss_120b/readiness_vllm/failed_attempts/20260831T1504Z_workspace_reduced_smoke_server.log`

- Server arguments include `--max-model-len 131072`, `--max-num-seqs 32`,
  `--async-scheduling`, `sample_on_device_mode=all`, and B1/B32 decode trace
  capture.
- The server reported healthy on `/health`, initialized on-device sampling with
  vocab size `201088`, captured decode traces, served the sampling tests, and
  shut down cleanly.

## Source evidence

### The failed test depends on same-seed reproducibility across batch order

`vllm-tt-plugin/tests/tt/test_request_isolation.py`

- Lines 17-72 define a mixed batch with multiple independent requests carrying
  `seed=42`.
- Lines 74-83 run the batch once, shuffle configs and the corresponding first
  results together, then run again.
- Line 87 asserts every seeded logical request reproduces exactly.

This means same-seed requests are not "duplicate children" of one parent
request. They are independent requests that vLLM expects to be deterministic
per logical request regardless of neighboring same-seed requests or batch order.

### The common seed manager says vLLM duplicate seeds must not be salted

`models/common/sampling/generator.py`

- Lines 39-59 derive device RNG seeds from `(request_seed, token_counter,
  salt)`.
- Lines 800-813 document the exact policy distinction:
  - salting duplicate seeds is useful for multi-sample children when the backend
    receives true duplicates;
  - in the vLLM path, parent request children have already received `seed+i`;
  - therefore duplicate seeds reaching the backend are genuinely independent
    requests and must match.
- Lines 868-887 assign different salts to active same-seed slots when
  `salt_duplicate_seeds=True`, and salt `0` when it is false.
- Lines 888-910 preserve an existing salt for the same slot/seed, so once an
  order-dependent salt is assigned, later decode-state refreshes keep it.

### GPT-OSS 120B autoport currently appears to take the default salting policy

`models/common/sampling/generator.py`

- Lines 130-135 initialize `SeedManager` with
  `getattr(args, "salt_duplicate_seeds", True)`.

`models/autoports/openai_gpt_oss_120b/tt/model.py`

- `FullModelArgs` contains sampling controls such as sampling all-gather axis,
  sampling DP, top-k logprobs, and model config, but no `salt_duplicate_seeds`
  attribute.
- `Model.__init__` installs `self.args = args` before the base sampling
  generator is constructed, so this autoport reaches the default
  `salt_duplicate_seeds=True` path unless something else overrides it.

### Decode seed plumbing preserves slots/counters, but does not remove salting

`models/tt_transformers/tt/generator.py`

- Lines 1252-1268 register each sequential prefill request into its physical
  slot using `apply_prefill_state`.
- Lines 2194-2237 compute active seed slots, apply slot remap, deactivate only
  non-live slots, refresh seed rows if needed, align counters to decode
  positions, and then request per-slot device seed values.

`models/common/sampling/generator.py`

- Lines 960-981 re-register active decode seed rows with
  `keep_existing_salt=True`.
- Lines 983-1027 align counters to absolute decode positions.
- Lines 1031-1070 move seed/counter/salt state across slot remaps.

That plumbing is useful and likely necessary, but it preserves the salt that was
assigned when the request was admitted. If the same logical request receives a
different salt because same-seed neighbors were admitted in a different order,
counter alignment and remap preservation will make the divergence stable rather
than fix it.

### Fixed sampling-state trace reuse is probably not the cause

`models/autoports/openai_gpt_oss_120b/tt/generator_vllm.py`

- Lines 463-474 reject fixed sampling-state reuse when any request has an
  explicit seed or non-default presence/frequency/repetition penalty.
- Lines 620-645 therefore pass fresh sampling params into
  `generator.decode_forward` for this failing seeded/penalized request.

`models/autoports/openai_gpt_oss_120b/tt/generator.py`

- Lines 570-583 also reject sampling-state reuse when any active request has a
  seed.
- Lines 639-643 state that explicit request seeds run the sampler eagerly
  without internal sampling trace replay.

The failure is thus unlikely to be stale fixed sampling params from traced
sampling-state reuse.

### Host fallback is probably not the cause

`vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py`

- Lines 1851-1897 only force host fallback for host-only controls such as
  unsupported structured outputs/logprobs and unsupported sampling options.
- The failing request uses temperature plus presence penalty and explicit seed,
  which the common on-device sampling path supports.

The passing host-only `min_p` smoke is useful coverage, but this failure is on
the on-device supported path.

## Proposed causal chain

1. `test_mixed_params_batch` submits several independent same-seed requests
   (`seed=42`) in one batch.
2. Sequential prefill registers each request into a device slot.
3. Because GPT-OSS 120B autoport args do not define `salt_duplicate_seeds`, the
   common sampler likely uses the default `True`.
4. `SeedManager` assigns different salts to active same-seed slots based on
   current admission/slot order.
5. The test shuffles request order for run 2. The same logical request can now
   receive a different salt than it received in run 1.
6. Decode preserves and advances that per-slot salt/counter state correctly, so
   the device RNG stream is reproducible for the slot state it was given, but
   not reproducible for the logical request across shuffled batch order.
7. The failing text sharing the initial ` voclags` token and diverging on later
   tokens is consistent with this: the first visible sampled token collided, or
   the divergence only became visible after the first decode step, while the
   subsequent random stream differed.

## Lower-probability hypotheses to keep open

- B1/B32 trace transition or slot remap edge case. The server captured B1 and
  B32 decode traces. If same-seed salts prove identical across run 1/run 2, the
  next suspect is a mismatch when a request moves between full-width and
  singleton decode buckets or when slot remap interacts with active seed slots.
- Active-layout async drain bug. The plugin drains async decode on layout change
  before preparing new inputs. Current source evidence suggests this is handled,
  but if salts match and bucket forcing does not matter, inspect async finalize
  and layout-change sequencing next.

## Small focused verify/refute experiments

1. Host-only seed-manager reproduction: instantiate `SeedManager` with a fake
   sampling object whose `write_device_seed_values` is a no-op. Register the
   same logical seed/config set in original order and shuffled order. Compare
   `(slot, seed, salt, counter, derived_device_seed)` for the logical `List:`
   request with `salt_duplicate_seeds=True` and `False`.
   - Expected if this finding is correct: `True` gives different salts or
     derived device seeds for the same logical request when peer order changes;
     `False` pins duplicate same-seed salts to `0`.
2. Hardware smoke instrumentation when device access is allowed: log
   request/config label, physical slot, request seed, salt, counter, start
   position, and derived device seed from `_set_slot_seed` and
   `_next_device_seed_for_slot` for run 1 and run 2 of only
   `test_mixed_params_batch`.
   - Expected if this finding is correct: the failing logical request has the
     same explicit seed and positions but different salt or derived device seed
     between runs.
3. One-line policy experiment: set the GPT-OSS 120B vLLM model args to
   `salt_duplicate_seeds=False` before `SamplingGenerator` construction, then
   run only:
   - `vllm-tt-plugin/tests/tt/test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch`
   - `vllm-tt-plugin/tests/tt/test_seeding_and_variety.py::TestSeedingAndVariety::test_uniform_seed_deterministic`
   - `vllm-tt-plugin/tests/tt/test_seeding_and_variety.py::TestSeedingAndVariety::test_different_seeds_produce_different_outputs`
   - Expected if this finding is correct: mixed params and uniform same-seed
     determinism pass, while different-seed variety remains intact.
4. Bucket-transition control: force decode to use only the B32 path, or disable
   B1 decode trace selection, while leaving salting unchanged.
   - Expected if salting is the root cause: the mixed-params failure persists.
   - Expected if B1/B32 transition is the root cause: the failure disappears
     despite unchanged seed salts.

## Bottom line

The smallest source-level area to verify first is the GPT-OSS 120B model args
used by vLLM sampling. Current source evidence indicates the autoport is taking
the common `salt_duplicate_seeds=True` default, which conflicts with the
documented vLLM same-seed contract and directly matches the only failing smoke
test.
