# AUTOTRIAGE

## Diagnosis

- The first serving prefill hangs in `IndexedFillDeviceOperation` because the pinned TTNN runtime has mismatched indexed-fill program-factory and reader compile-argument/CB contracts. Selector placement is not the root cause: a retry with the selector in L1 hangs at the identical operation. The stage-local repair replaces both model serving uses with device-only slice/concat/copy row splices.

## Triage Evidence

- Live triage identifies op 1396, `IndexedFillDeviceOperation`, as the only running op on all four devices.
- The first capture's selector (`Tensor[0]`) is `UINT32`, row-major, shape `[1, 1, 1, 1]`, and in DRAM. A second capture proves the same stop with `buffer_type: L1`.
- The destination is the fixed serving recurrent state, shape `[32, 1, 4, 2560]`; the one-row replacement is shape `[1, 1, 4, 2560]`. These shapes and the single slot index agree.
- ARC, Ethernet, NoC-location, inactive-CB, and firmware health checks pass. The widespread BRISC/ERISC stop sites are downstream fanout from the four-device op, not independent hardware failures.
- Full evidence is in `models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/triage/tt-triage.txt` and the L1 retry in `tt-triage-l1-retry.txt`.

## Source Evidence

- In `/home/ttuser/dev/tt-metal@9b415f8`, `indexed_fill_program_factory.cpp` emits three fixed compile arguments and begins `TensorAccessorArgs` at index 3; `indexed_fill_reader.cpp` interprets argument 3 as `mode` and begins accessor metadata at index 4. The factory's legacy two-page CB/writer also disagrees with the reader's newer native/shard-local/generic paths.
- For this tiled DRAM destination, the shifted argument is interpreted as native mode, explaining why selector DRAM versus L1 does not change the hang.
- The later serving logits slot placement used the same broken operation and had to be removed as well.

## Downstream Effects

- vLLM waits for prefill completion and the HTTP request remains open.
- Decode, sampling, and trace capture never begin. The process and devices otherwise remain healthy enough for live triage.

## Proposed Fix

- Replace indexed fill with a device-only splice: slice maximal unchanged destination runs and selected source rows, concatenate them in destination order, then copy into the persistent destination buffer.
- Preserve destination identities, arbitrary stable slots, and the no-readback state/token paths.

## Uncertainty

- The structural fix must be verified with the same 65-token serving request after process cleanup and restart. The captures prove the runtime indexed-fill mismatch but do not yet prove later traced decode gates.
