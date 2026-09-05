# AutoDebug: terminal LM head

Inspection-only investigation, 2026-09-05. No devices accessed, no implementation changes, and no new runtime or performance claims. Existing optimized decoder precision must remain unchanged.

## Starting evidence

- Failure: `doc/full_model/logs/probe_cache_env.log`, terminal `ttnn.linear` before `PREFILL_OK`.
- Original command: `TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe`.
- Error: static circular-buffer region end `7992320 B` on `[0-0 - 7-9]`, beyond `1572864 B` L1.
- Current head: TP4, K=4096, global padded vocabulary 262144, local vocabulary 65536, BF16 weights/output, HiFi4, FP32 destination accumulation, packer L1 accumulation. Input is one 32-row tile width-sharded across 32 cores with shard `[32,128]`. Program uses `in0_block_w=4`, `per_core_M=1`, `per_core_N=64`, default one reader per DRAM bank.

## H1: head width exceeds native reader-buffer capacity

**High-confidence source diagnosis; proposed repairs remain unverified.**

The apparent 32-core partition is an output-storage partition. Native `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:160` derives actual reader width as `ceil(N_tiles / num_workers)`. Its argument named `per_core_N_storage` is the Python `per_core_N`; reducing this argument does not shrink reader/compute buffers. `num_workers` is DRAM-bank count times `num_workers_per_dram_bank`.

For 8 banks and one reader per bank, N=2048 tiles gives 256 tiles per reader. The factory triples the weight buffer (`:230-235`) and allocates distinct BF16 output and FP32 intermediate buffers (`:237-240`, `:581-605`). Its CB ranges are the bounding rectangle of senders/readers (`:299-304`, `:543-598`), explaining the 8x10 failure range despite the activation's 8x4 storage grid.

Exact payload calculation, 32x32 BF16 tiles = 2048 B:

| Buffer | Calculation | Bytes per core |
| --- | --- | ---: |
| Activation CB | 2 × 4 × 2048 | 16384 |
| Weight CB | 3 × 4 × 256 × 2048 | 6291456 |
| Output CB | 256 × 2048 | 524288 |
| FP32 intermediate CB | 256 × 4096 | 1048576 |
| Static CB payload total | | 7880704 |

The observed region end minus this payload is 111616 B, consistent with the reserved L1 starting region. The error reports an address/region end, not solely the CB payload (`tt_metal/impl/program/program.cpp:1868-1875`). Input and resharded-output CB descriptors refer to existing tensor buffers and are not additional copies of this payload (`factory:567-578`, `:624-635`).

Rejected fixes by source arithmetic:

- `in0_block_w=1` alone still requires 3149824 B payload.
- Full local width, block 1, two readers per bank still requires 1576960 B payload, already beyond total L1 before reserved space or resident tensor buffers.
- Four readers are unsupported: `matmul_utilities.cpp:341-345` permits only 1..3.
- Three readers reject the current 256-tile bank shard because it is not divisible by 3 (`factory:144-168`). More padding/repacking could make that legal, but it is a separate experiment and changes the sampler's convenient 65536-wide shard.
- Smaller `per_core_N`, freeing unrelated tensors, cache/context reductions, or decoder precision changes do not address this static weight-buffer requirement.

## Smallest useful real-shape experiment

Use an isolated terminal-only script with `OrnithModel(None, mesh, layer_indices=[], max_context=2048)`, current pinned real head/norm weights, and the existing `open_ornith_mesh`. Do not allocate decoder/cache state. Run one case per process using the hardware owner's existing serialized runner and watchdog.

Proposed script invocation after the repair agent creates it:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_terminal --columns 16384 --in0-block-w 1 --readers 1
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_terminal --columns 8192 --in0-block-w 4 --readers 1
```

These commands are proposals, not executed commands; `probe_terminal` did not exist at investigation time.

| Candidate | Per-device blocks | Weight DRAM shard | Program per_core_N | Static payload | Predicted region end using observed base |
| --- | ---: | --- | ---: | ---: | ---: |
| A: block 1 | 4 × 16384 columns | `[4096,2048]` | 16 | 790528 B | 902144 B |
| B: block 4 | 8 × 8192 columns | `[4096,1024]` | 8 | 999424 B | 1111040 B |

Both retain current input sharding, output BF16, BF16 weights, HiFi4, FP32 accumulation, packer L1 accumulation, and final padded vocabulary. These estimates are necessary capacity checks; runtime must still verify collisions with tensor buffers and kernel correctness.

Required experiment details:

1. Read the same checkpoint head and zero-pad to `[4096,262144]`. For split `s` of width `C`, upload `torch.cat([head[:,d*65536+s*C:d*65536+(s+1)*C] for d in range(4)], dim=1)` with the existing `ShardTensorToMesh(dim=1)` and per-split DRAM shard configuration. Splitting the global head into ordinary consecutive chunks would assign wrong vocabulary ranges to devices.
2. Use deterministic BF16 hidden states at logical row counts 1 and 32. Include distinct, nonzero active rows. Normalize exactly as current terminal, and read normalized rows once for a linear-only CPU oracle. Compute CPU FP32 matmul from these same BF16 normalized inputs and checkpoint BF16 weights. Also compare the complete terminal to CPU zero-centered RMSNorm plus head, reporting both boundaries separately.
3. Run all head blocks, convert each output to interleaved DRAM, then concatenate locally along vocabulary in split order. Final per-device output must remain `[1,1,32,65536]`. Compose TP4 along the last dimension and restrict accuracy to the 248320 valid tokens.
4. Record every input/weight/output logical and padded shape, dtype, memory config, program config, reader count, DRAM-bank count, finite-value check, PCC, max/mean absolute error, and top-1/top-5 comparison against the CPU oracle. Explicitly check columns around every block boundary and TP boundary. Use the stage's existing accuracy gates; do not invent relaxed thresholds.
5. A deterministic random-hidden probe establishes packing/kernel behavior. Before keeping a fix, rerun the original reduced `[0,3]`-layer probe and its prefill/decode sampling path; full-model accuracy remains a separate gate. Check repeated execution and trace replay after eager correctness. Measure warmed head latency only if choosing between passing candidates; do not infer a speedup from buffer size.

Candidate A changes blocking and packing; candidate B changes packing while preserving the current K block. Test separately and retain only a verified candidate. The old 35B model already illustrates rank-preserving split packing (`references/.../tt/model.py:718-732`) and local concatenation (`:1424-1466`); its old measurements and decoder policy are not evidence for 9B.

## H2: sampler configuration does not supply historical grouping

Both current common samplers accept full vocabulary-sharded logits and perform local top-k followed by candidate all-gather. Neither reads `topk_num_groups`; the argument in the new model's `build_sampler` is inert. Neither public sampling interface accepts arbitrary pre-reduced `(values, global_indices)` candidates. Copying the old model's `topk_num_groups` option therefore does not install the old grouped implementation.

However, **retain the current `[1,1,32,65536]` BF16 TILE/interleaved-DRAM sampler shape for the first repair**. Current `models/common/sampling/tt_sampling.py:994-1009` queries `topk_would_route_to_large_indices` and disables `stable=True` when the native Blackhole composite is eligible. `ttnn/.../reduction/topk/topk.cpp:272-363` admits this exact width, dtype, layout, last-dimension reduction, unrestricted core grid, and k=32. It routes the wide-row reduction to `topk_large_indices` instead of the stock slow single-core factory. `models/common/modules/sampling/sampling_1d.py:610-615` also calls native top-k without requesting stable ordering, so its exact shape is eligible too. These are source predictions, not measured 9B timing.

The next sampler-only probe should log the route query on the actual masked terminal logits, compare local candidates against CPU top-k, then time warmed current common sampling before porting historical grouped top-k. Changing `topk_num_groups` from 1 to 32 should currently leave the graph unchanged.

If routing or measured latency still warrants grouping, an exact candidate hierarchy for current width is 32 groups × 2048 columns, group top-32 → 1024 candidates per user → local top-32 → TP4 gather of 128 candidates. Carry original global token IDs through every reduction. This requires a real common-sampler implementation change (group reductions, candidate-index recovery, tests), not a head reshape alone. Preserve penalties before candidate reduction, padding masks, tie behavior, sampling controls, and any full-logit logprob contract. There is no evidence yet that this beats the current native composite.

## H3: force-argmax is both wider and incompatible with current CCL wrapper

Keep the default top-k sampling route initially. In `TTSampling.forward:793-889`, force-argmax gathers the full vocabulary before untilize/argmax. It directly calls `get_and_cycle_ag_semaphore_handles` and `get_and_cycle_barrier_semaphore_handle`, which `SamplingCCL` does not provide. Its `getattr` default also evaluates the latter attribute eagerly. The alternate `Sampling1D` force-argmax path similarly bypasses `line_all_gather`, then untilizes the full gathered logits (`sampling_1d.py:313-325`, `:347-373`). It lacks the current TTSampling wide-untilize chunk workaround.

Thus `force_argmax=True` is not a demonstrated fast fallback. Standard top-k with greedy parameters keeps candidate communication small and uses the existing compatible `line_all_gather` hook. The latent semaphore mismatch is distinct from the present head compile failure.

## Verdict

H1 explains the exact reported allocation numerically. Proceed with the two terminal-only, precision-preserving chunk probes above; retain native 65536-wide common sampling and verify its route. No repair or benchmark was performed by this investigation.
