# Responses to review R4 (release)

Review: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/convaiinnovations_laya/doc/review/review_R4_release.md`
(verdict more-work-needed, five Required Work items at P2, all text). Dispositions by the orchestrator, 2026 Oct 6 03:30 UTC.

| review item | disposition |
|---|---|
| REPORT section 4 shows `bf8w_hifi3` and `bf8w_hifi3_head_bf16` as pass with an empty invariance cell | fixed: 0.0150 and 0.0154, "fails invariance", sentence under the table |
| Not-reproducible list attributes MASSIVE 0.783 and XNLI 0.860 to the multilingual checkpoint and a missing protocol | fixed in REPORT section 7 and in the summarizer template (`/home/hous/dev/laya/evals/summarize.py`); plan amendment A15; the other English card claims (SST-5, act AUROC, Khmer) listed with a scope statement |
| REPORT section 9 lacks the sibling commands and gives a `stop` form the CLI cannot resolve | fixed: sibling serve, push and publish commands added; stop by manifest path with the docker fallback documented |
| Sibling card "all within margin 0.01" against a flip at 0.0145 | fixed at commit 578c6dbdb5 ("under 0.015"); sibling build 4 carries it |
| Demo script cites build 4 and a tile value matching neither screenshot | fixed: tiles 26.0 / 21.6 / 19.6 ms, throughput 201 to 249; evidence references point at the final English build |
| Sibling manifest resolved packages at build time | fixed at 578c6dbdb5 (`runtime.lock: requirements.lock`) |
| Scorecard quotes two throughput ranges | fixed: one range, the final English build's |
| Investigation method sentence overstated the 800-item run | fixed at 578c6dbdb5 (scoped) |
| Hard-check gaps (poll granularity of "ready after 20 s", `stop` by bare name never exercised, no script ties REPORT to SUMMARY, sibling single-row corpus) | accepted as recorded; "ready after 20 s" is the chain's 5-second poll (health uptime 13.1 s at that poll), stated in REPORT section 2 |
