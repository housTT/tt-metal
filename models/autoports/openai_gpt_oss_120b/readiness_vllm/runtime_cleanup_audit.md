# Runtime cleanup audit

- Final sampling/qualitative server: API PID 646299, EngineCore PID 646915.
- Clean primary-benchmark server: API PID 771121, EngineCore PID 771442.
- Both servers were stopped with SIGINT after their requests completed.
- The API process reports `EngineDeadError` while its output-handler task is
  cancelled during the zero-timeout abort shutdown. The marker occurs only
  after the explicit `[shutdown]` records; no request or benchmark failed.
- Post-shutdown process search found no vLLM API server, `vllm serve`, or
  `VLLM::EngineCore` process.
- Post-shutdown `tt-smi -s` reported four p300c boards, `dram_status=true` on
  every board, zero corrected/uncorrected GDDR errors, and no device holder
  other than the `tt-smi` probe itself.
- A second final audit after documentation/build checks saved the empty
  process search to `readiness_vllm/final_process_audit.txt` and the healthy
  JSON device status to `readiness_vllm/final_tt_smi_status.txt`.
- No device reset was needed. No runtime fallback was enabled for the measured
  benchmark: it used `sample_on_device_mode=all`, traced decode, asynchronous
  token output collection, and temperature-zero canonical TT sampling.
