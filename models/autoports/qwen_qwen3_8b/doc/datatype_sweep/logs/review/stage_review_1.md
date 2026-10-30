# Stage review 1 (12:39 to 12:52 ET, fresh read-only subagent, prompt logs/stage_review_prompt.md)

Verdict: more-work-needed

Required work:
- P1: committed sweep_results/README recorded c8 and c10 as pass (no robustness rows) while c9 was selected; the resolving rows (c8 aime_p1/book32/natural, c10 natural) were produced during the review and were uncommitted and undocumented.
- P2: resumed-prefill hit window L=1500 s=1472 fails top-1 (27 of 28) under c9 with no margin probe on c9; README wording "unchanged" applies to the PCC minima only.
- P2: context_contract.json note says the budget matches the no-mesh formula, but Qwen3Generator.max_tokens_all_users without a mesh uses the stage-6 resident constant (112320); sweep_results.csv and logs are gitignored and not committed.

Other concerns: natural-text reference covers the prefill LM-head path (decode LM head covered by book32 TF and batch-32 only); the per-group fidelity field governs decode only (prefill fidelity is fixed by dtype), which is why c10 equals c5; c3 capacity trade not weighed (carry to the vLLM stage); book32 reference untracked; README section 4 "default path" wording for the 36-layer propagation check; write_sidecar.py records top1=1.0 for the resume run; wording of the c8 sentence and the natural-text label; shell helpers carry comment lines.

Hard-check gaps: top-5 chart draws no frontier (single frontier point); three natural-text JSONs still carry the first-revision fields; check_propagation does not compare fp32_acc, packer_l1_acc, in0_block_w or prefill fidelity; review placeholders in README section 8 and work log section 9.

The reviewer's full verdict text (anomaly ledger, scope, residual risk) is reproduced in stage_review_1_full.md.
