# Qwen3.6-27B bring-up — paused 2026-08-11

Branch `agentic-research/hous/qwen3.6-27b-v2`, paused mid stage 3 to work on a different model.

## Where the pipeline is

| stage | status |
|---|---|
| 1 · functional-decoder | **complete** |
| 2 · fused-decoder | **complete** (via `resume-1`; 21 review rounds) |
| 3 · optimized-decoder | **in progress** — blocked/resumed 4×, 8 commits so far |
| 4–11 | not started |

Run directory — every stage log, the manifest, and the `stage_N_session_id` values that
`--resume-stage` needs. **Durable copy (63 MB), made at pause:**

```
/home/ttuser/dev/qwen/multigoal-runs/20260809T184609Z/
```

The original is at `/tmp/claude-multigoal-runs/20260809T184609Z/` and will not survive a reboot.
If it is gone when you return, copy the durable one back before resuming — `--resume-stage` reads
`manifest.txt` from `--log-dir`, so either path works as long as it exists:

```bash
mkdir -p /tmp/claude-multigoal-runs
cp -a /home/ttuser/dev/qwen/multigoal-runs/20260809T184609Z /tmp/claude-multigoal-runs/
```

Last commit at pause: `21db0b0a7f2`. Working tree had **93 uncommitted files** — stage 3's
in-flight evidence. Do not `git clean` or `checkout .` here; that is the stage's unsaved work.

## Resume command

```bash
cd /home/ttuser/dev/qwen/tt-metal && source python_env/bin/activate && \
python .agents/scripts/multigoal-claude \
  --claude-bin /home/ttuser/.local/bin/claude \
  --resume-stage 3 \
  --log-dir /tmp/claude-multigoal-runs/20260809T184609Z \
  --replace HF_MODEL=Qwen/Qwen3.6-27B \
  --replace MODEL_DIR=models/autoports/qwen3.6-27b \
  .agents/prompts/model_bringup_multigoal/*.txt \
  >> /tmp/claude-multigoal-runs/20260809T184609Z/runner.log 2>&1 &
```

`--resume-stage` reattaches to the recorded `stage_3_session_id`, so the stage keeps its context
instead of restarting. Once stage 3 records `complete`, the *same* invocation continues into
stages 4–11 automatically.

## What stage 3 still owes (its own sequence)

1. **Evidence campaign** — `probes/run_campaign.sh probes realprobes tracy suite`, ~2.5 h.
   Needed because every tracy/probe artifact predates the final precision policy, so the
   before/after tables, the isolation 2×2 and the OPT-013 fidelity proof are stale.
   **It was stopped mid-run at the pause (SIGTERM, clean — device released, ARC responsive), so
   it must be re-run from the start.** Two things to know about the partial run:
   its first phase had already failed — `10:00:56 FAILED probes (exit 2)` — and the script
   records failures and continues rather than stopping. Nothing in the 13 probe logs from that
   window shows a traceback or assertion, and each ends with a clean device teardown, so it
   looks like a probe exiting non-zero on its own criteria rather than a crash.
   **Diagnose that before trusting a re-run.** Also: probe logs written between 09:41 and 10:07
   are from this aborted campaign and are neither complete nor consistent with each other.
2. `probes/finalize_evidence.sh` — perf_summary.json → pcc_evidence.json → contract block →
   generated doc tables → docs gate. Three gate checks currently fail *only* because artifacts
   predate the policy, including `test_prefill_fidelity_override_reached_the_measured_ops`,
   which is designed to fail until tracy re-runs.
3. Local commit, then `$stage-review` to `clean-pass`.

## Decision waiting for you: the history rewrite

Stage 3 intends to run, at commit time:

```
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --msg-filter 'grep -v "^Co-Authored-By: Claude" || true' a6a91a89da5..HEAD
```

to strip `Co-Authored-By: Claude` trailers from six earlier stage commits (it picked up the
housTT identity rule). Backup ref **`backup/pre-trailer-strip`** exists and it verified the
command on a scratch clone. This rewrites six commits' history and **will fire on the next
resume unless you intervene.** Harmless on a local branch nobody has pulled; decide deliberately.

## Environment — the part that matters most

The previous attempt was thrown away because **source and runtime came from different
lineages**: the build lived at `~/.local/lib/model-bringup/tt-metal` on branch
`deepseek-v4-flash-bringup` @ 2026-07-03, while the repo branch was based on `main` @
2026-06-22 and carried eight C++ edits that were never compiled. Ten hours of stage-3 evidence
was measured against code that was not under test.

The fix, and the thing to preserve:

* This checkout has its **own venv** at `python_env` (created by `./create_venv.sh`).
  `import ttnn` must resolve **inside this checkout**:

  ```bash
  source python_env/bin/activate
  python -c "import ttnn; print(ttnn.__file__)"   # must print .../dev/qwen/tt-metal/ttnn/...
  ```

* `PYTHONPATH` alone is **not** enough. `~/.tenstorrent-venv` has an editable ttnn install plus a
  `ttnn-custom.pth` that pin `import ttnn` to the old tree regardless of `PYTHONPATH`. Plain
  `python3` on this host still resolves to the old tree.
* `create_venv.sh` uses `uv`, so the venv ships **without pip**. A `pip` symlink → `pip3` was
  added by hand; re-add it if the venv is ever regenerated, or bare `pip install` silently
  installs into `~/.tenstorrent-venv` instead.
* Stage 1's prompt carries a guard requiring the activation check before any device job, and
  stage 1 wrote the activation into `doc/functional_decoder/ttenv.sh`, which later stages source.

## Technical results worth not losing

**The SDPA decode fix is upstreamable.** Stage 1 patched
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp`
(committed inside `9d18c856aaa`, tangled with ~344 lines of autoport docs). Flash-decode kept its
running max, softmax denominator and output accumulator in `Float16_b`; once a chunk's
contribution falls below bfloat16's half-ULP the denominator stops growing and the output comes
out uniformly too large — a **pure scale error that PCC cannot see**. Measured, at the full
262144 context: device/float32-golden scale **1.290 → 1.006** at position 262143, and the layer
**0.9779 → 0.9992** against the 0.995 bar. The fix promotes only the *core-local* accumulators,
gated on `num_cores_per_head == 1`, so the cross-core packet format is untouched.

Upstream context: [PR #48753](https://github.com/tenstorrent/tt-metal/pull/48753) (merged
2026-07-16) closed [issue #13364](https://github.com/tenstorrent/tt-metal/issues/13364) for
ring-joint, and the pattern was later ported to the standard prefill factory behind
`fp32_dest_intermediate_dataformat()`. **The decode path was never done** — that is the gap this
fills. Extracting it onto a branch off `main` would make a clean PR; it gets harder the longer it
sits under later stages.

**A second, independent decode bug is still open.** `paged_scaled_dot_product_attention_decode`
needs an `SDPAProgramConfig` whose validity condition — `num_k_chunks == 1` or
`num_k_chunks % (2 * cores_per_head) == 0` — depends on `ceil((cur_pos+1)/k_chunk)`, a *runtime*
quantity, while the config is compile-time. No single setting is correct at every position:
12287 gives a 780440× scale error, 261887 gives NaN.

**GDN.** `gated_delta_attn_seq` was measured and **rejected on accuracy** in the previous run:
real-weight output PCC 0.989740, state 0.982896, versus 0.99998 for the composed implementation,
and no Python-side precision knob moved it (the loss is inside the C++ scan). Suspected but never
verified cause: `gated_delta_attn/device/gated_delta_attn_program_factory.cpp:170` hard-codes
`MathFidelity::HiFi2` and **discards the caller's `compute_kernel_config`**. Chunk-size
equivalence (kernel needs 128, model uses 64) is proven exact on CPU — PCC 1.000000000, max abs
diff 3.7e-8.

This tree additionally has **`ttnn.transformer.chunk_gated_delta_rule`** (with device kernels and
a phased program factory) and `models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py`, neither of
which the old build had. Both came from
[PR #48861](https://github.com/tenstorrent/tt-metal/pull/48861) "Optimizations for Qwen3.6-27B"
and [PR #48380](https://github.com/tenstorrent/tt-metal/pull/48380) "batch=8 support" — i.e.
**someone else is optimising this exact model** in `models/demos/blackhole/qwen36/`, almost
certainly the CS team referenced in
[issue #50475](https://github.com/tenstorrent/tt-metal/issues/50475). Read #48861 before redoing
GDN work.

**Precision policy.** Stage 3's own full-context sweep, against a (0.98, 1.02) scale bar on real
weights (`doc/optimized_decoder/logs/realweight_longcontext.log`):

| policy | tail_scale | decode_scale |
|---|---|---|
| + HiFi4 on every projection | 0.990969 ✓ | 1.021896 ✗ |
| + float32 dest acc everywhere | 0.947509 ✗ | — |
| + bfloat16 MLP weights | 0.926727 ✗ | — |
| fused-stage policy (control) | 0.985581 ✓ | 1.000867 ✓ |

It ultimately shipped a shared-role `mlp_down` prefill fidelity that passed the four-case
full-context gate (`4 passed`, exit 0, 2026-08-11 08:39).

## Operational notes

* **The runner stops whenever device work outlives the agent's turn.** Stage 3 blocked four times
  this way; none were real failures. The recovery is always `--resume-stage`. Resume only once the
  device work it was waiting on has actually finished — resuming early just produces another
  ~14-minute block. An auto-resume wrapper was discussed but never written.
* **OAuth expiry kills long runs.** Stage 2 died at `turnFailed` after 10.5 h with
  "OAuth session expired and could not be refreshed" (168 × HTTP 401). Nothing in the runner
  detects or recovers from this. `/login`, then `--resume-stage`.
* **Never `kill -9` a process mid-device-op** — it hangs PCIe and needs a driver reload plus
  `warm_reset`.
* Log filter for the stage JSONL streams: `~/Downloads/tail_filter_color.jq`
  (`tail -f <stage>.jsonl | jq -r --unbuffered -f ~/Downloads/tail_filter_color.jq`).
* Hardware: two p300c boards. Devices 2 and 3 healthy; **device 0's ARC is wedged** (no `tt_*`
  sysfs attributes) and its partner is device 1. Stages run on device 2 via
  `doc/functional_decoder/ttenv.sh`. Exposing a single chip of a 2-chip board makes metal classify
  the cluster as `CUSTOM` and require `TT_MESH_GRAPH_DESC_PATH`.
