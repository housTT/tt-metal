# Vendored `trex` package (the T-Rex harness)

This directory is a vendored copy of the T-Rex demo harness from the upstream Contrastive Language Models
repository. It drives the live demo served at `/demo` (`../demo.py`, `../demo_runner.py`, `../demo_brain.py`).

| Field | Value |
| --- | --- |
| Upstream repository | https://github.com/Contrastive-LM/CLM |
| Upstream path | `examples/t_rex/trex/` |
| Upstream commit | `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` (2026-09-24 23:09:47 UTC) |
| Vendored on | 2026 Oct 2 |
| Upstream license | Apache-2.0; the upstream `LICENSE` file is at `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/clm/LICENSE`. The game rules and constants come from Chromium's offline dinosaur game (BSD-3-Clause), via virajbhartiya/laya-vs-jev (Apache-2.0), as the package docstring says. |

## Files

| File | Role | Status |
| --- | --- | --- |
| `__init__.py` | package docstring | unchanged |
| `engine.py` | deterministic Chromium dino clone, 600 x 150, 60 FPS | unchanged in content; the repository's black hook (line length 120) reformats it on commit |
| `planner.py` | exact-physics safe or unsafe labels per action for the frame an answer will land | unchanged in content; reformatted by the hook |
| `safety.py` | the shield applied when an answer lands and the emergency check | unchanged |
| `backends.py` | builds the Choice request and posts it with httpx; env `CLM_BASE_URL`, `CLM_API_KEY` | unchanged in content; reformatted by the hook |
| `brain.py` | the player process (`serve`, `RemoteBrain`, `think`, `warm`) | unchanged in content; reformatted by the hook |
| `pilot.py` | `Pilot` (in-flight decision loop), `Arena` (frame clock), `Pacer`, `Stats` | unchanged in content; reformatted by the hook |
| `course.py` | the upstream staged course (not used by the demo) | unchanged |
| `match.py` | round scoring imported by `Arena` (not used by the demo) | unchanged |

Not vendored: `examples/t_rex/run.py` (it imports `examples/common.py`, which needs the `clm` client and
`requests`); its `warm_up()` and the per-course result row are re-expressed in `../demo_runner.py` with the same
field names, so the demo's rows compare one to one with `examples/t_rex/results/clm_realtime.json` (shipped as
`clm/demo/reference_rtx4090.json`) and with the evaluation runs in `/home/hous/dev/clm-v0.1-8B/evals/trex/results/`.

## How the demo hooks in without editing these files

`../demo_brain.py` spawns the upstream `serve()` through a subclass of `RemoteBrain` whose child first replaces three
module globals of `trex.brain` (`create`, `build_question`, `think`) with wrappers that record the server's
`X-CLM-Latency-Ms` header, the state text and the three option texts for each decision. `../demo_runner.py` subclasses
`Pilot` to collect every `note()` call (accepted answers, discarded late answers, shield events) and streams game state
from the public `Game` attributes.

## Refreshing

```
git clone https://github.com/Contrastive-LM/CLM /tmp/CLM && git -C /tmp/CLM checkout <commit>
for f in __init__ engine planner safety backends brain pilot course match; do diff -u /tmp/CLM/examples/t_rex/trex/$f.py $f.py; done
```

Expect only formatting differences for the five files the hook touches; apply `black --line-length 120` to the upstream
copy first for a content-only diff.
