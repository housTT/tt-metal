# AutoFix: host-to-device request state

2026-09-08. Two source-verified transition defects were fixed without changing
the canonical sampling algorithm or steady device decode path. Both history
and seed repairs pass the real adapter hardware comparisons; final serving
regressions remain with the supervising lane.

## Verified causes

[`AUTODEBUG_host_resume_history.md`](AUTODEBUG_host_resume_history.md):
unchanged sampling parameters caused an early return before caller histories
could restore counts accumulated during host sampling. The actual-method CPU
probe initially failed three checks and passed eight controls. This included
the host-sampled first token after a fresh host prefill, as well as ongoing
host continuations.

[`AUTODEBUG_host_seed_admission.md`](AUTODEBUG_host_seed_admission.md):
fresh host prefill did not receive the new request's parameters, yet prepared
the generator seed using the previous configuration. The first later device
step changed configuration but preserved that stale seed tensor. Actual common
CPU seed helpers reproduced old seed 11 yielding device seed 32229 instead of
new request seed 99 yielding 43506.

## Minimal changes

The adapter computes active host-owned resume slots only inside its existing
sampling boundary branch. If any such row requires presence, frequency or
repetition history, it delegates restoration to `configure_sampling` even when
the parameter key is unchanged. Missing caller histories fail before decode
replay. Neutral and freshly device-prefilled rows do not trigger restoration.

The adapter also tracks `_pending_device_seeds` for fresh host-prefilled rows.
The flag follows remaps and survives host continuation. At the first device
decode or continuation prefill, the adapter supplies `fresh_seed_slots` to
`generator.configure_sampling`, which calls the existing canonical
`_reset_request_seeds` after loading the correct parameters. Only those rows
are initialized. A new device prefill uses its existing seed-admission path
and clears any old pending flag. The adapter implements no RNG arithmetic.

Both selection scans are inside the existing scheduler/mode boundary branch.
There are no added steady token/position conversions, seed uploads, or history
updates. Ongoing rows retain their device RNG streams. This does not promise
equal stochastic sequences between different host and device algorithms.

## Transition coverage and evidence

`tests/test_sampling_mode_history.py` executes actual adapter and generator
methods with CPU TTNN boundaries and actual common seed-manager helpers.
Together with the neighboring admission, serving contract and penalty-order
tests, **53 checks pass**:
[`host_resume_seed_after.log`](host_resume_seed_after.log).

| Transition | Required behavior covered |
| --- | --- |
| Steady device decode | Keep device token/history/RNG authority; skip configuration and refresh |
| Device to host | Skip device sampler; retain its histories until explicit restoration |
| Penalized host to device, unchanged or changed key | Restore caller prompt/generated histories |
| Penalized host to device without history | Reject before replay |
| Neutral host to device | Avoid needless history configuration |
| Fresh device prefill after older host decode | Preserve the newly counted device first token |
| Fresh host prefill to device | Restore host first-token history and initialize the requested device seed |
| Neutral host row plus fresh penalized device row | Preserve the fresh row; no unnecessary restoration |
| Inactive penalized host slot | No history work until active |
| Remap before host resume | Reindex ownership/pending-seed flags and restore new slot order |
| Host-first continuation prefill on device | Initialize its request seed before device sampling |
| Reused host slot gets a new device prefill | Use normal admission once and clear old pending state |

The supervising real adapter probe passed before the seed extension:
[`host_resume_device_exact.json`](host_resume_device_exact.json) and
[`host_resume_device_exact.log`](host_resume_device_exact.log), exit 0.
It uses layers 0 and 3, B3, native logical context and 4108 shared physical
blocks, with greedy presence 2, frequency 0.5 and repetition 2. The host-step
detour then device resume exactly matches uninterrupted device tokens,
positions, prompt/output masks and counts across vocabulary shards and
gathered replicas. Host steps leave device counts unchanged; resume restores
them with one configuration update. Three model replays and two sampler
replays are recorded for the detour, versus three of each in the reference.
This is a contract proof, not full-model quality or performance evidence.

The extended `tests/host_resume_device_probe.py` also requires exact canonical
seed initialization for host-first requests, with and without slot remap,
every replica, preservation of the other 31 lanes, and one ordinary counter
advance on the following steady step. This extended hardware probe passed
(exit 0):
[`host_resume_seed_device_exact.json`](host_resume_seed_device_exact.json)
and [`host_resume_seed_device_exact.log`](host_resume_seed_device_exact.log).
The new request's first device seed is exactly **43507** (canonical seed-99
initial value 43506 plus one sampled step), then **43508** on the next step.
Both unmapped row 0 and remapped row 2 pass; every other lane advances without
reset. The history-detour comparison also passes again on this final source.

Runtime files changed: `tt/generator_vllm.py` and `tt/generator.py` only.
The CPU test fixture and new mode-transition tests/probe are the remaining
Python changes. Black and compilation pass. The supervising complete stage
Python and plugin pre-commit runs also pass, recorded in
`stage_python_precommit_release.log` and `plugin_precommit_release.log`.
This Python-only repair needs
no C++ build. Final all-layer serving and performance gates belong to the
supervising lane and are not inferred from these reduced controls.
