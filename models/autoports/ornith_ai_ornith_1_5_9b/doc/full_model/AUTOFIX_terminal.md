# AutoFix: terminal LM head and sampler

## Starting evidence

`AUTODEBUG_terminal.md` H1 explains the original static circular-buffer end of
7,992,320 bytes versus 1,572,864 bytes L1. The original command failed in the
BF16/HiFi4 K4096 × local-vocabulary65536 LM head before `PREFILL_OK`.

## H1 experiment and verified fix

Hypothesis: native DRAM reader buffers depend on each worker's full N range;
rank-preserving output-column splitting can fit L1 without changing precision.

Created `probe_terminal.py`: no decoder layers or cache allocation, real pinned
checkpoint final norm and LM head, TP4, K4096, global padded vocabulary262144,
and logical row counts1 and32. Every chunk contains the corresponding local
vocabulary slice for each TP rank. Concatenation happens locally into the same
`[1,1,32,65536]` BF16 TILE/interleaved DRAM sampler contract.

Exact commands, from repository root (each exited0):

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_terminal --columns 16384 --in0-block-w 1 --readers 1
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_terminal --columns 8192 --in0-block-w 4 --readers 1
```

| Candidate | Row1 terminal trace | Row32 terminal trace | Linear-oracle top5/top100 | Full-terminal top5/top100 |
| --- | ---: | ---: | --- | --- |
| A: four16384, block1, readers1 | 1.957 ms | 1.949 ms | 100%/100% | 100%/100% |
| B: eight8192, block4, readers1 | 1.859 ms | 1.851 ms | 100%/100% | 100%/100% |

Times include RMSNorm, input sharding, every projection and local concat, measured
with32 nonblocking trace replays followed by synchronization, without logits
readback. Both candidates are finite, repeat exactly in eager execution, and
match eager exactly under trace. Their normalized-hidden linear-oracle PCC
rounds to1.0. CPU full-terminal comparisons independently include zero-centered
RMSNorm. JSON records all top1/top5/top100 metrics, full shapes/configuration,
absolute errors, and block/TP boundary checks. Top1 for32 random hidden rows is
90.625% versus the FP32 CPU oracle; top5/top100 are100%. This is a terminal
packing/kernel probe, not full-model accuracy evidence.

Verdict: verified. Keep B as the faster measured candidate. Decoder dtypes,
fidelities, cache policy, and residual layout are unchanged. Removed inert
`topk_num_groups`, which neither selected common sampler consumes.

Original reduced probe was rerun once with A and once after selecting B:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe
```

Both exited0 with `PREFILL_OK`, `SAMPLING_OK`, `DECODE_OK`, and `PROBE_OK`.
Selected B completed in9.633 seconds including setup. This is the existing real
layers0 and3 reduced path, not an all-layer result.

Artifacts:
- `logs/terminal_16384_block1.log`, `terminal_16384_block1_readers1.json`
- `logs/terminal_8192_block4.log`, `terminal_8192_block4_readers1.json`
- `logs/probe_terminal_repaired.log` (A)
- `logs/probe_terminal_selected.log` (B)

## H3 sampler experiment

Hypothesis: force-argmax's missing CCL accessors prevent comparison. Added the
two thin aliases onto the same existing cycling TP4 semaphore manager; no new
collective state or fallback. Tested both greedy strategies against exact CPU
argmax of identical real-head BF16 logits for32 distinct rows. Both eager and
common-sampler internal trace write exactly the expected tokens into persistent
`tt_out_tok`.

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_sampler
```

Exit0, `SAMPLER_PROBE_OK`. Both use semantically greedy k1/p0/temperature1.
Canonical sampling retains physical max_top_k32 for tile-shaped candidates.
Over32 warmed internal trace calls, canonical sampling measured0.574 ms versus
force-argmax2.748 ms. Keep canonical sampling; reject force-argmax as4.79× slower
in this exact comparison. The logged common force-argmax gather selects a Linear
topology for the1D mesh. This comparison includes the common sampler's replay
orchestration; it does not measure the later generator-owned seeded wrapper.

Artifacts: `probe_sampler.py`, `sampler_greedy_comparison.json`, and
`logs/sampler_greedy_comparison.log`.

## Final status

H1 fixed and original reduced check passes. H3 compatibility fixed and tested;
canonical greedy sampling selected by exact-token and timing evidence. No new
sampler failure emerged. No device reset or triage was needed during this
experiment; every process closed its mesh before the next command. Hardware
ownership was returned to the stage owner after the final process exited.
Full-stack accuracy, trace feedback/state ownership, reduced profiling, and
stage review remain the stage owner's gates. No full-model completion claim or
commit is made by this experiment.

## Precision-locked full-model follow-up

Real all32-layer `teacher_head4_lofi.json` passed top1 92%, top5/top100100%
with a BFP4/LoFi terminal (decoder unchanged), compared with94%/100%/100%
for the BF16/HiFi4 terminal. This justifies examining the faster terminal policy
with the same full-model gates rather than rejecting it from random hidden PCC.
The geometry probe reserved114688L1 bytes/core on the32-core input grid to
represent all24 resident hybrid states. Immutable commands and snapshots are
`logs/terminal_bf4_*.provenance.json` and `.sources.json.gz`.

- Four16384 columns/rank, block4, reader1: about0.594ms.
- Two32768 columns/rank, block4, readers2: about0.476ms, selected prospectively.
- One65536 columns/rank, block1, readers2: about0.807ms, slower.

All passing candidates have top5/top100100% against the real-weight terminal
oracle and exact repeat/eager/trace agreement. The16k and32k candidates have
identical logits hashes. A single-part concat returns its input as an alias;
the wrapper originally deallocated that returned tensor and caused a host-read
segfault. The one-part path now returns the part directly. The exact64k candidate
then passed after bounded reset/list/mesh recovery. This is an ownership fix,
not a rejection of the64k shape. See `terminal_bf4_64k_b1r2_fixed`.

Block16 on the32-core input shard is invalid:4096/32/32=4 K tiles per core,
so the block must divide4. The32-core layout retains the decoder residual grid.
The selected head is not the dominant layer-stack cost. Full-model final gates
must confirm this candidate before the stage may pass.
