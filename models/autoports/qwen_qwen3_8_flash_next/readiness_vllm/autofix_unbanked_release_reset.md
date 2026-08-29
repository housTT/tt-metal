# AutoFix Report: unbanked virtual-slot release reset

## Starting Evidence

- `readiness_vllm/final_virtual_b2_upsample/server.log` recorded three sequential
  physical-B1 requests with `commits=0`, `restores=0`, and `resets=3`.
- The same metrics report `logical_bytes_per_slot=260718612`, so each finished
  request enqueued a 260,718,612-logical-byte zero-copy tree for a snapshot that
  had never been materialized.
- The source audit in `autofix_explicit_upsample_stall_recurrence.md` identified
  the unconditional `release_virtual_slot()` reset. No new AutoDebug pass was
  needed because this experiment targets that already localized lifecycle bug.

## Hypothesis Experiments

- Hypothesis: a slot with `_virtual_slot_banked[slot] == False` is not
  restore-eligible, so clearing its device-bank storage is unnecessary.
- Safety invariant: admission of a second live user calls
  `_commit_resident_if_needed()` and fully overwrites the newly reused slot
  before setting `_virtual_slot_banked=True`; `activate_virtual_slot()` rejects
  every non-resident unbanked slot.
- Before-fix experiment:
  `python_env/bin/python -m pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_virtual_decode_state_bank.py::test_unbanked_release_skips_device_reset_but_cleans_request_lifecycle`
- Result: failed with `assert 1 == 0` for the reset metric, verifying the
  unnecessary reset.
- Fix: gate bank zeroing in `reset_virtual_slot()` and
  `release_virtual_slot()` on `_virtual_slot_banked[slot]`. PLE cancellation,
  history deletion, host metadata clearing, validity clearing, generation
  guards, ownership release, and resident invalidation remain unconditional.
- Isolation control: a fake device bank retains an old A snapshot across an
  unbanked release, reuses the slot for C, and proves C is fully committed
  before its first restore. Banked release of B still performs a zero reset.

## Final Status

- Fixed. Sequential direct physical-B1 releases now report `bank.resets=0` in
  the fake-TT lifecycle metric while banked releases still reset.
- Verification:
  `python_env/bin/python -m pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_generator_vllm.py models/autoports/qwen_qwen3_8_flash_next/tests/test_virtual_decode_state_bank.py --junitxml=models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/autofix_unbanked_release_reset_host.xml`
- Result: `31 passed`; artifact:
  `readiness_vllm/autofix_unbanked_release_reset_host.xml`.
- Remaining evidence: the next real serving run should confirm the all-48
  sequential metric changes from `resets == releases` to `resets=0` while
  `commits=restores=0`. This host-only task did not touch TT hardware or server
  processes.
