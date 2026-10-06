# Responses to review R3 (staged package, release candidate build 4)

Review: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/convaiinnovations_laya/doc/review/review_R3_staged_package.md`
(verdict more-work-needed, one Required Work item at P2). Dispositions by the orchestrator, 2026 Oct 6 01:33 UTC.

## Orchestrator dispositions

| review item | disposition | owner |
|---|---|---|
| Required Work P2: served DAIR Emotion deviates up to 0.27 with 2 of 386 confident flips at the 1x128 bucket; the card quotes the parity corpus bound (0.12, no confident flip) without disclosing it | fix: device investigation (per-option logit deltas on the flipped texts; bucket control at 5x256 and 1x256; policy control bf16_hifi4; fp32 eager versus SDPA noise), a single-row parity corpus of the 800 suite calls (or a seeded subset) with the stage 6 gates in process, the package E1 rerun on it in the build 5 chain, the card's `risks` and `limitations` restated per evidence set, RUN_NOTES and REPORT.md updated, build 5 card-only (same code sha) | Track T3, orchestrator |
| Demo page counts 114 of 300 against 113 from the Python clients (browser JSON writes 1625.0 as 1625) | fix: integral floats written as integers in `demo/feed_cases.json`, page and client serialization aligned, one sentence in the serving README | Track S |
| RUN_NOTES "within 0.7 ms on every latency cell" holds for p150 only | fixed by the orchestrator (sentence scoped to the p150 profile; p150x4 differences stated) | orchestrator |
| p150x4 SUMMARY places a 100-case partial typed-decisions row in the published-format table without n | fix in `summarize.py` (n and "partial" in the label) | Track E |
| REPORT.md "52 host tests" unsupported | fixed by the orchestrator from a fresh run of the six host test files before build 5 | orchestrator |
| E3 confident-agreement block missing from SUMMARY.md | fix in `summarize.py` | Track E |
| E1 tensor path never run against the package (`RAW_FORWARD=1` not set by the chain) | fixed by the orchestrator: `bin/final-build.sh` exports `RAW_FORWARD=1` for the evaluation step from build 5 on | orchestrator |
| `health.shapes.calls` is a startup snapshot | fix: live per-bucket counters | Track S |
| Uncommitted sibling edits in the main checkout touch shipped paths | accepted: the sibling work is committed as its own checkpoint; the English bundle's build 5 is built from the commit the chain detaches and re-verified by the chain | orchestrator |
| UMD board warnings, allocator warning at startup, p150x4 1x5 cell noise | accepted as recorded (controlled) | orchestrator |

## Track T3 responses

Track T3 wrote `tests/single_row_investigation.py` (01:57 UTC) and did not report further; the orchestrator ran it.

## Orchestrator investigation (Track T3's script, run by the orchestrator), 2026 Oct 6 02:18 to 02:35 UTC

Evidence: `doc/full_model/single_row_investigation.md`, `doc/full_model/single_row/` (device_*.json, gates_single_row_*.json,
served_single_row_gates_build4.json), corpus `/home/hous/dev/laya/reference/parity_corpus_single.npz` (800 items).
Findings: the two confident Emotion flips reproduce bit for bit alone at 1x128 and inside 5x256 and 5x128 batches (not a
bucket or placement effect); `bf16_hifi4` keeps the same texts within 0.108 with no flip (a policy effect on near-tied
short inputs). On the 800 served single-row decisions 777 of 779 confident decisions agree, median max abs
dp 0.0012: the decision gates pass; the placement spread on 16 single-row questions is 0.019 (shipped) and
0.011 (`bf16_hifi4`) against the 0.01 gate that holds on the gate corpus, with the same answers. Disposition: the
shipped policy stays by the plan's rule; the card states agreement and spread per evidence set and names
`LAYA_PRECISION=bf16_hifi4` as the slower alternative; build 5 carries the card. The E1 tensor path runs in the chain
from build 5 (`RAW_FORWARD=1`), together with E1 on the single-row corpus.
