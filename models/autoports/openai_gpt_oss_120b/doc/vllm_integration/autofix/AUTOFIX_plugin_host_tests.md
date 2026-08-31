# AutoFix: stale plugin host-unit API expectations

## Failure

The final broad host-only plugin command collected 153 tests and initially
failed six:

- one Gemma-4 streaming parser test called dictionary methods on vLLM's typed
  `DeltaFunctionCall` Pydantic object;
- five KV-capacity tests called `get_num_available_blocks_tt(cfg)` after its
  established source contract required the runtime-discovered physical
  `num_devices` argument.

Git history and source call sites showed both failures were stale baseline
tests. No GPT-OSS production diff touched the parser or block-count source
contracts.

## Minimal fixes

- `test_gemma4_tool_parser.py` now guards `function is not None` and reads the
  typed `.name` and `.arguments` fields.
- `test_num_available_blocks.py` now passes
  `num_devices=cfg.device_config.num_devices` in its five calls, matching the
  sole production caller.

No implementation source changed.

## Evidence

- Focused parser and block-count files: 14 passed.
- Exact original host-only plugin suite: 153 passed, 16 warnings, in 4.97 s.
- Black and `git diff --check`: pass.
