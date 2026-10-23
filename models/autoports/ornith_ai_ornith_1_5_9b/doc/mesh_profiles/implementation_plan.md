# One-, two-, and four-chip profile bringup

Preparation only: TP1/TP2 device accuracy, capacity, and performance are not yet
validated. The isolated branch starts at stage 9 commit 539734fbb6. Integrate
after the autonomous stage's source freeze and rerun affected gates.

The source baseline is the stage 5 optimized TP4 decoder and stage 8 precision
policy. Stage 3's one-chip decoder provides a measured local-compute baseline;
its layer measurements do not predict full-model serving speed. The previous
35B port provides examples of topology validation and explicit KV ownership;
this 9B model has four KV heads and needs no duplicated KV heads at TP4.

## Geometry before implementation

| Tensor or configuration | TP1 | TP2 | TP4 | Rule |
|---|---:|---:|---:|---|
| Residual hidden width | 4096 | 4096 | 4096 | Replicated selected baseline |
| Optional sharded residual width | 4096 | 2048 | 1024 | hidden / TP |
| Query / KV heads | 16 / 4 | 8 / 2 | 4 / 1 | Complete local GQA groups |
| GDN key / value heads | 16 / 32 | 8 / 16 | 4 / 8 | Head-local recurrence |
| GDN QKV channels | 8192 | 4096 | 2048 | Split Q, K, V separately |
| GDN A and B physical columns each | 32 | 32 | 32 | Independently tile-padded |
| Dense MLP intermediate | 12288 | 6144 | 3072 | gate/up columns, down rows |
| Embedding local hidden width | 4096 | 2048 | 1024 | Gather once before decoder |
| Padded vocabulary per chip | 262144 | 131072 | 65536 | Contiguous output shards |
| LM-head 32768-column chunks | 8 | 4 | 2 | Keep tested chunk geometry |
| B1 recurrent matrix bytes/layer/chip | 2097152 | 1048576 | 524288 | FP32; excludes padding |
| Native KV bytes/chip, all 8 full layers | 4563402752 | 2281701376 | 1140850688 | BFP8 tiles 1088B; excludes null block |

DRAM width-shard widths must be recomputed from each local projection's N and
the selected number of readers/banks. Decode K-block divisibility remains a
device-test gate; a TP4 program configuration is not assumed optimal on TP1/2.
The model keeps non-aligned logical prompts and paged64 cache semantics.

## Collective candidates

| Path | Residual before/after | Next consumer | Communication | Storage and decision |
|---|---|---|---|---|
| Native all-reduce | replicated partial/full | residual add and norm | Two reductions per layer | TP4 measured baseline; TP2 to measure; TP1 bypass |
| Reduce-scatter with delayed gather | partial/sharded | distributed norm | Reduced hidden shard, then norm/activation gather | Existing sharded path retained; compare whole layer |
| Fused all-gather matmul | sharded/full or partial | attention/MLP projection | Fuses input gather | Requires adapting existing packed projection programs; device comparison pending |
| Fused matmul reduce-scatter | replicated/sharded | residual add and distributed norm | Fuses output reduction | Requires compatible DRAM reader and persistent output contracts; pending |
| Single chip | full/full | ordinary residual add and norm | None | No CCL manager or semaphores; clone gathered outputs to preserve ownership |

Activation/collective precision follows the selected BF16 residual policy. The
initial TP2 candidate uses the same ring fabric family as the local GPT-OSS
one-dimensional mesh fixture, with one link. Topology and link choices require
on-device verification before selection. TP1 disables fabric explicitly.

## Required verification

Host checks reconstruct every sharded projection and GDN channel group, prove
Q/gate head ownership, validate mesh rejection, and verify one-chip gather
returns independently owned storage. Hardware checks then cover both layer
kinds, HF full-model accuracy, native/non-aligned context and cache continuity,
traced sampling, batches/concurrency/cancellation, and served API behavior.
Update context_contract.json only from actual profile allocation/validation
evidence; estimated memory above does not establish maximum supported context.

## Host verification, 2026-09-08

Seven tests passed using the tested serving environment, with this isolated
worktree first on PYTHONPATH and as the working directory:

```bash
python models/autoports/ornith_ai_ornith_1_5_9b/tests/test_mesh_profiles_host.py
```

The tests import the actual candidate modules and use CPU tensors or mocked
device entry points; they never open hardware. They cover all three partition
sizes and both attention kinds. Repository-pinned Black and isort hooks and
`git diff --check` passed. Root evidence is
`evidence/mesh-profile-host-tests-final.log`. No C++ changed, so no build was
required for this candidate. This is not a completed profile release gate.

## Prepared full-model gates

The selected datatype-sweep `doc/datatype_sweep/run_candidate.py` and complete
native serving `tests/serving_native_capacity_probe.py` now accept
`--num-devices {1,2,4}`. Their existing accuracy thresholds, native262143/262144
windows, final-position decode, cache identity, device sampling and trace teardown
assertions remain intact. These CLI changes have only been checked with `--help`
and repository hooks; they do not establish device results.

Run each command independently inside the task container after the multigoal
worker releases the hardware lane. Use `flock -n` on the shared
`/home/hous/dev/ornith-1.5-9b/state/hardware.lock` before launching Python, with
the candidate first on PYTHONPATH and the validated serving interpreter. The
lock must be held inside the container by the test process, not just by a Docker
client on the host. Output paths must be unique per mesh and attempt.

```bash
python doc/datatype_sweep/run_candidate.py --num-devices 1 --output tp1_accuracy.json
python tests/serving_native_capacity_probe.py --num-devices 1 --output tp1_native.json
```

The commands above are relative to the autoport and illustrate the test argv;
the supervising invocation must also hold the lock as described. Repeat for2
and4 chips. The datatype-sweep command uses the pinned existing HF reference;
its smoke mode uses reduced layers and must never be reported as a full-model
accuracy pass. Serving concurrency, prefix caching, API and final package checks
are separate required gates.
