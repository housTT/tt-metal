# Coherence Spec AutoFix

## Starting evidence

- The no-Docker GPT-OSS-120B/P150x4 spec run failed
  `test_coherence_verbatim_echo` with `finish_reason=length` and
  `message.content=null` under a 32-token completion budget. The test's strict
  final-content helper correctly rejected the reasoning-only response.
- Report:
  `../tti_cache/workflow_logs/reports_output/spec_tests/data/report_data_id_openai-gpt-oss-120b-autoport_p150x4_2026-09-02_20-13-46.json`
- The workspace vLLM v0.26.0 request schema supports
  `max_completion_tokens` and `reasoning_effort`; its GPT-OSS Harmony renderer
  supports `low`, `medium`, and `high` effort.
- Existing optimized-vLLM qualitative evidence for this exact model uses low
  reasoning effort and a 256-token completion budget so requests reach
  user-visible final content.

## Hypothesis experiment

- Hypothesis: the generic 32-token budget is consumed by GPT-OSS reasoning
  before the final channel, so the result cannot validly test coherence.
- Focused API A/B: the same prompt with low reasoning effort and a 128-token
  budget returned HTTP 200, `finish_reason=stop`, 29 completion tokens, and a
  nonempty exact final answer. No response text was printed or retained.
- Verdict: verified.

## Fix

- The coherence request now uses low reasoning effort and the evidence-backed
  256-token `max_completion_tokens` budget.
- Budget exhaustion remains a failure, reasoning-only output remains a
  failure, and final content must equal the requested sentence after trimming
  surrounding whitespace. Extra words are not accepted.

## Host-only verification

```text
.workflow_venvs/.venv_workflow_run_script/bin/python -m pytest -q tests/llm_module/test_chat_completion_generated_text.py
13 passed in 0.25s

git diff --check -- llm_module/test_vllm_chat_completions.py tests/llm_module/test_chat_completion_generated_text.py
pass

.workflow_venvs/.venv_workflow_run_script/bin/python -m py_compile llm_module/test_vllm_chat_completions.py tests/llm_module/test_chat_completion_generated_text.py
pass
```

The full API conformance workflow still must be rerun against the existing
autoport server by the coordinating release task.
