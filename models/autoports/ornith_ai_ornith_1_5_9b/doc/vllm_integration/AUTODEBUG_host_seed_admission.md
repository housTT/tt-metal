# AutoDebug: seed ownership for requests admitted in host mode

2026-09-08, before the seed fix. The adjacent transition audit reproduced a
separate seed-admission defect using actual source methods and the common
CPU seed manager, without TTNN imports or hardware.

A host-mode fresh prefill omits sampling parameters. The generator still
prepares fresh-slot sampling state and initializes that slot using its old
`_configured_seeds`. At the request's first device step, configuration changes
to the requested seed while preserving the existing persistent seed tensor,
so the new request can retain the previous occupant's device seed.

[`host_seed_admission_before.json`](host_seed_admission_before.json) records
the source-executed proof: old request seed 11 initializes 32229; a new
request specifying seed 99 reaches its first device step with 32229 still
present, although seed 99 should initialize 43506. `_configured_seeds[0]`
becomes 99 but the actual common seed manager still identifies request 11.
The proof executes adapter prefill/decode, generator prompt preparation and
configuration, `_request_seed_values`, and the real common `SeedManager`.

The proposed fix tracks fresh host-prefilled rows until their first device
sampling use, including slot remaps and continuation prefill. The adapter
delegates those rows to the generator's canonical request-seed initializer
after their correct parameters are configured. Other rows keep their device
streams. This does not claim that different host and device stochastic
algorithms produce the same sequence, and adds no per-step seed upload or
scan to steady device decode.

The implemented pending-row flag and generator `fresh_seed_slots` contract
pass CPU source-execution regressions and the exact real-adapter hardware
probe, with and without slot remap:
[`host_resume_seed_device_exact.json`](host_resume_seed_device_exact.json).
The first device sample uses 43507 and the next 43508, while all other lanes
retain their streams. See
[`AUTOFIX_host_mode_transitions.md`](AUTOFIX_host_mode_transitions.md)
for scope, tests, and remaining final serving gates.
