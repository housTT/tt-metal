# AutoFix: request and fixed-slot sampling seeds

## Findings and causal evidence

Independent stage review identified two source-level defects:

1. `_configure_sampling` broadcast a scalar seed to every cache slot and mapped
   every omitted seed to `_hash_request_seed_to_device_seed(0, 0)`. This
   contradicted the selected common formatter, whose scalar seed belongs only
   to lane zero and whose omitted seeds require independent request/lane entropy.
2. The common 32-lane sampling trace advances every seed tensor lane. A newly
   admitted request in an inactive or reused slot inherited that lane's advanced
   value because low-level prefill did not reinitialize its request seed.

Neither finding requires weakening accuracy/quality gates or changing decoder
precision. Their intervention boundary is request initialization, not each token.

## Repair

- `_configure_sampling` now retains `formatted.seed` exactly. No scalar broadcast
  or implicit zero seed remains.
- `_request_seed_values` delegates request initialization and seed drawing to the
  selected common `SeedManager.reset_seed` and `_next_device_seed_for_slot`.
  Explicit seeds use its stable hash; omitted seeds get the manager's fresh
  entropy-seeded per-lane RNG. Full reset first clears prior active seed/salt state.
- Fixed cache slots represent independent requests, so the common manager's
  supported `salt_duplicate_seeds` option is disabled. Explicitly repeated seeds
  then start identical streams across fresh/joined/reused slots, independent of
  another request's lifecycle. Callers wanting distinct streams provide distinct
  per-lane seed values.
- `_reset_request_seeds` runs only for newly prefilling rows with `start_pos=0`
  (including omitted start positions). It generates new values for those rows
  and uses the already tested signed-predicate device merge to preserve every
  other lane's current device counter. Continuing prefill with a nonzero start
  keeps the stream and merely consumes the next sample.
- The steady-state model/sampling traces and device `plus_one` are unchanged.
  No per-token host RNG generation, seed upload, token reconstruction, or seed
  readback was added. The explicit public `reset_seed=True` still resets the
  configured full-slot seed tensor at a scheduler boundary.

## Source-only verification

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
python_env/bin/python -m pytest --noconftest \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/test_generator_host_contract.py -q
```

The suite loads actual common `SamplingParams`, `format_sampling_params`,
`SeedManager`, and hashing definitions through AST while excluding TT runtime
imports. Added cases prove explicit-list repetition, same-seed slot independence,
scalar lane-zero semantics through both the common formatter and generator
configuration, fresh per-request/per-lane entropy, new/reused slot initialization,
continuation preservation, and selected-device-seed merge preserving ongoing
counters. Earlier host sampling/EOS/request-TTFT/live-reconfiguration tests remain.

Exact final count/result is in `generator_host_contract_tests.log`. The generator,
probe, and tests were Black formatted. This repair agent ran no TTNN import,
accelerator operation, model weight load, or C++ build.

## Hardware acceptance still required

The existing `scheduler_sampling_contract.py` now also verifies actual device
seed tensors rather than requiring probabilistic token differences:

- repeated explicit lists initialize the same described lane seeds;
- a scalar seed repeats lane zero only, while other lanes receive new seeds;
- two omitted-seed requests have independently initialized lane states;
- the same explicitly seeded prompt has identical first token and seed after
  prefill when fresh, joining an already running fixed batch, or reusing a slot;
- a continuation prefill advances exactly once without reinitialization;
- ongoing requests' feedback, RNG, cache, and penalty histories survive admission.

The parent owns serialized hardware execution and final gate acceptance.
`full_stage_review` inspected the source patch and confirmed it addresses both
identified RNG gaps; its final verdict still depends on hardware evidence.

## Hardware acceptance: scheduler_sampling_final_v3

The parent executed the expanded real-layer hardware probe successfully; review
of `scheduler_sampling_final_v3.json` and its immutable provenance confirms the
required checks, rather than relying on a prose success claim. Exact command:

```bash
timeout 180 python_env/bin/python -m \
  models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.scheduler_sampling_contract \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/scheduler_sampling_final_v3.json
```

The recorded run started 2026-09-05T17:41:09 UTC, used batch 4 with real layers
0 and 3 on the four-chip mesh, and enabled trace-allocation tracking. It did not
combine watcher/profiler collection. Exact environment, source snapshots and
exit status are in `logs/scheduler_sampling_final_v3.provenance.json` and
`logs/scheduler_sampling_final_v3.sources.json.gz`.

Observed device evidence:

- Explicit seed list initializes the four described lanes identically on repeat:
  `[749808, 812499, 823144, 619512]` both times.
- Scalar seed repeats lane zero `218503`, while the other initialized lanes differ:
  `[495924, 943333, 203020]` versus `[880855, 538605, 469744]`.
- Omitted seeds initialize independent request states: the first four lanes are
  `[236163, 985877, 603320, 713092]`, then
  `[717169, 400007, 668699, 694021]`.
- Fresh, joined, and reused explicitly seeded requests all return token `[25]`
  and seed value `275415` after prefill. Continuation advances without reset.
- Live sampling changes preserve cache/input/sampler state; unchanged replay
  continues correctly; partial prefill preserves ongoing sampler rows; a new
  slot joins live decode. Every corresponding JSON check is true.
- The isolated large-UINT32 control reproduces the old unsigned-predicate failure,
  while the signed-predicate repair preserves all bits exactly.

Status: these two RNG review findings now have passing source tests and the
required focused on-device acceptance evidence. Full-stage completion remains
subject to the parent's other gates and independent review.
