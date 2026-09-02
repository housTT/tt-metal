# AutoFix: Harmony stop-string response semantics

The no-Docker TTI stop test exposed a real API integration defect: vLLM's v1
detokenizer removed the requested stop string from decoded text but retained
the raw token IDs for accounting, and the Harmony parser reconstructed the
stop suffix in `message.reasoning` from those IDs.

The fix is locally committed in the official workspace vLLM checkout at
`54dea57d98ccfaef072908f085d9296d544ba1fe`. Non-streaming parsed responses now
truncate the terminal matching reasoning/content/tool field according to
`include_stop_str_in_output` without changing token IDs, logprobs, or usage.
The TTI assertion was not weakened and still checks every generated channel.

Verification:

- host regression: 5 passed;
- compile and diff checks: passed;
- focused live endpoint request: HTTP 200, matching stop reason, and stop text
  absent from every generated channel;
- focused coherence request: HTTP 200 and exact requested final content;
- full no-Docker TTI spec result: recorded in the final release report.

No response text or token IDs are copied into this handoff. The affected TTI
gate is non-streaming; this change does not make a broader claim about Harmony
stream synchronization.
