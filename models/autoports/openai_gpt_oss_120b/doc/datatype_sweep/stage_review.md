# Independent Stage Review

Verdict: **clean-pass**

Required work: none.

The fresh-context stage reviewer independently inspected the retained artifacts,
runtime code, tests, source provenance, blocker logs, and scoped git diff.  The
review recomputed `ds00_baseline` as the fastest passing traced teacher-forcing
candidate at top-1/top-5/top-100 `0.95/1.00/1.00` and
`61.03824482016593` decode tokens/s/user.  The nearest passing alternative was
`ds11_lm_head_lofi` at `61.02997420162983` tokens/s/user.

The reviewer confirmed:

- all 13 candidate rows and their semantic config hashes agree across the JSON,
  CSV, and checked-in configs;
- every ranked measurement has 99 actual model and sampling trace submissions,
  zero unclassified submissions, and pre-existing trace handles before timing;
- `selected_precision_config.json` is the default runtime config and its actual
  tensor, collective, KV-cache, activation, logits/sampling, and compute-fidelity
  consumption is validated and retained;
- the BFP4 expert and attention matmul groups each have matched LoFi/HiFi2
  candidates;
- P150/P150x2/P150x4 capacity, non-aligned prompts, selected token-out, and
  qualitative evidence satisfy the stage contract;
- both Pareto plots contain all measured points, the frontier, the accuracy
  threshold, and the selected red marker;
- the cold performance anomaly and the two rejected-policy blockers are
  honestly excluded or classified; and
- no vLLM or serving work is present in the scoped changes.

Non-blocking residual risk: most candidates have one warmed timed repetition,
while the selected row and closest fidelity alternatives have two.  Full model
execution is physically feasible only on P150x4; P150 and P150x2 are retained as
exact capacity-accounting targets with zero feasible resident context.
