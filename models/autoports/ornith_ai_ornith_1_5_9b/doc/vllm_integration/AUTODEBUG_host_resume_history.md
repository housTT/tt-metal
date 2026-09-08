# AutoDebug: unchanged-parameter host-to-device penalty history

2026-09-08. Initial source diagnosis before implementation changes.

The adapter deliberately skips device sampling while a batch requires explicit
host compatibility. The generator's generated-token penalty counts therefore
remain at their last device step. On return to device sampling, decode calls
`_sampling` because the previous mode was host, but `_sampling` returns early
when the parameter key is unchanged. Supplied caller histories then never
reach `generator.configure_sampling`. An ongoing penalized request can resume
using stale counts despite receiving the correct host-sampled input token.

Hypothesis: force history restoration through the existing configure method
only for active host-owned rows that need penalties, even when parameters
are unchanged. Device-prefilled rows already own their first sampled token's
history and must not trigger this restoration merely because an older decode
used host mode. Host-prefilled rows do require the host first-token history.
Inactive slots and neutral resumed rows need no forced configuration.

The focused CPU experiment will execute actual adapter decode/configuration
and generator configuration methods at stubbed TTNN boundaries, covering
unchanged-key resume, missing history, neutral resume, steady device decode,
fresh device/host prefill, mixed-row lifetime, changed parameters and remap.
An exact reduced real-weight adapter probe must then compare an unbroken
device continuation with a host step followed by device resume, including
penalty histories and tokens. The supervising lane owns all hardware.

Before implementation changes, the real-method CPU regression produced
**3 failed, 8 passed** in `host_resume_history_before.log`. Unchanged-key
penalized host continuation retained stale counts, missing histories were
silently accepted, and a fresh host-prefilled row lost its host-sampled first
token's history. Neutral resumes, continuing device rows, fresh device
prefills, inactive slots, changed-key configuration, remap, and device-to-host
sampler bypass controls passed. This verifies the history-restoration gate
as the defect without requiring a speculative sampler change.

After the adapter boundary fix, the actual reduced real-weight host-step
detour matches uninterrupted device decoding, including exact histories on
all shards and replicated gathered counts. Both the original
`host_resume_device_exact.json` and extended final
`host_resume_seed_device_exact.json` hardware runs pass. The adjacent seed
repair and final transition table are documented in
[`AUTOFIX_host_mode_transitions.md`](AUTOFIX_host_mode_transitions.md).
