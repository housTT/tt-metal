# AutoFix Report

## Starting Evidence

- Source diagnosis: `AUTODEBUG.md` in this directory.
- Original failure: the real-weight TP=4 full-attention layer-3 test passed
  non-aligned sequence-33 prefill at PCC 0.9984013856, then the first decode
  failed while launching MLP `up`.  The live gate allocation began at
  1,330,752 while the static-CB region ended at 1,339,904.

## Hypothesis Experiments

- Hypothesis: the live gate L1 result causes the up projection's static-CB
  overlap, and the common MLP gate-to-DRAM spill contract is sufficient for
  the complete TP-local MLP.
- Experiment: only for DRAM-sharded `m <= 32` decode, copy gate to interleaved
  DRAM and deallocate gate L1 before `up`; emit SiLU-multiply to DRAM; reshard
  the product to the down projection's eight-core width-sharded L1 input; run
  down and the existing ring all-reduce.  Prefill and all other paths remained
  delegated to `OptimizedDecoder` unchanged.
- Command (from `/tmp`):

  ```bash
  env TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
      TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
      PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
      python /tmp/qwen36_h1_full_decode.py
  ```

- Result: prefill still passed at PCC 0.9984013855574635.  The `up` launch
  passed, directly verifying that releasing the live gate allocation removes
  the original collision.  The complete path then failed at `down`: the
  resharded activated L1 input began at 1,324,608 while the down program's
  static-CB region ended at 1,333,760, another 9,152-byte overlap.
- Verdict: **verified** for the original gate/up cause; **refuted** as a
  sufficient complete fix because the prescribed down-input reshard exposes
  the same lifetime/CB conflict at the next projection.
- Evidence artifact: terminal traceback from program 204 at
  `MultichipDecoder._mlp`'s down `ttnn.linear`; the device closed cleanly.
- Fix: none retained.  The H1-only implementation change was reverted as
  required.  `multichip_decoder.py` is back to the pre-experiment path.
- Verification: `python -m py_compile` and `git diff --check` pass after the
  revert.

## Final Status

- H1 alone is not a viable functional repair.  It proves the source diagnosis
  but does not complete decode.
- The next independent experiment should tune the static-CB geometry (H2) or
  select a down-input placement/core geometry that is valid for the DRAM-
  sharded down program; no H1 spill code should be retained by itself.

## H2: DRAM-sharded `in0_block_w` scan

- Hypothesis: reducing the TP-local eight-core gate/up program's legal
  `in0_block_w` below the automatically selected 20 reduces its static-CB
  footprint enough to keep both MLP projection outputs in L1.  Down was also
  assigned each legal block (17 in the largest-control run, 1 in the minimum
  runs), without spill or padding.
- Experiment: a temporary `MultichipDecoder` subclass changed only
  `dram_sharded_mlp_blocks` after `from_torch`.  Each legal gate/up candidate
  `{10, 5, 4, 2, 1}` ran the exact real-weight full-attention layer-3,
  non-aligned sequence-33 prefill/decode check.  Candidate 10 used down=17;
  candidates 5, 4, 2, and 1 used down=1 so the smallest legal down geometry
  was already selected if execution reached it.
- Command template (from `/tmp`, repeated for every value above):

  ```bash
  env TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
      TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
      PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
      QWEN_H2_BLOCK=<10|5|4|2|1> QWEN_H2_DOWN_BLOCK=<17|1> \
      python /tmp/qwen36_h2_full_decode.py
  ```

- Result: every candidate preserved the passing prefill PCC
  0.9984013855574635, then failed on the up projection with exactly the same
  allocator evidence as the default block 20: live L1 allocation 1,330,752,
  static-CB end 1,339,904, overlap 9,152 bytes, program 196, core range
  `[0-0 - 7-9]`.  No candidate reached down, so its block cannot repair the
  earlier gate/up collision.
- Verdict: **refuted**.  For this Blackhole DRAM-sharded program factory,
  `in0_block_w` does not change the static-CB high-water mark responsible for
  this collision.  There is no passing candidate and therefore no latency
  comparison or selected geometry.
- Fix: none.  The production `multichip_decoder.py` was not edited; its zero
  values continue to request the inherited automatic block selection.
- Remaining uncertainty: this experiment rules out every legal block width at
  the current no-padding eight-core geometry.  It does not rule out the wider
  16-core padded geometry (H3) or a different complete spill/down-placement
  strategy.

## H3: TP-local intermediate padding and 16-core geometry

- Hypothesis: padding each full-attention TP-local MLP intermediate shard from
  4,352 (136 tiles) to 4,608 (144 tiles) permits 16 compute cores and reduces
  each live decode output shard from 34,816 bytes to 18,432 bytes, avoiding the
  gate/up and activated/down L1/static-CB collisions.
- Experiment: only for full-attention layers, split gate/up output rows into
  four device-local 4,352-row chunks and append 256 zero rows to each; split
  down input columns into four device-local 4,352-column chunks and append 256
  zero columns to each.  The runtime-local intermediate is 4,608 and all three
  MLP projections use 16 cores.  The linear-attention 4,352/8-core path and
  public 5,120-wide residual/output contract are unchanged.
- Exact command (from `/tmp`):

  ```bash
  env TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
      TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
      PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
      python /tmp/qwen36_h1_full_decode.py
  ```

- Result: the real-weight full-attention layer-3 sequence-33 prefill and decode
  completed without an L1/CB exception.  Prefill PCC was 0.9983840627447128;
  decode PCC was 0.9991158445371842.  The mesh devices and ring fabric closed
  cleanly.
- Reuse experiment: `/tmp/qwen36_h3_full_decode.py` wrapped the same exact test
  and executed `decode_forward` twice consecutively against the same cache and
  position.  Both calls completed with output shape `[1, 1, 32, 5120]`; the
  returned second iteration retained decode PCC 0.9991158445371842.  The
  multichip counters recorded six ring all-reduces (prefill plus two decodes).
- Verdict: **verified**.
- Fix: retained the per-device zero padding and full-attention-only 16-core
  geometry in `tt/multichip_decoder.py`.
- Verification: `python -m py_compile` and `git diff --check` pass.  A final
  `tt-smi -ls --local` found all four local P300c devices available after the
  run.

## Final Status

- **Fixed**: the original real-weight full-attention first-decode L1/static-CB
  collision is resolved by H3 with PCC above the stage's 0.995 threshold and
  consecutive decode reuse coverage.
- Follow-up risk: stage-level trace, watcher, and comparative warmed-latency
  gates remain separate from this focused AutoFix experiment.
