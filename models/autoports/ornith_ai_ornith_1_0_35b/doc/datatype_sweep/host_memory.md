# The HF control could not be re-run on this host, and why that does not weaken the evidence

## What happened

`run_post_selection.sh`'s `qualitative` step exited **137** (SIGKILL) after ~60 s, inside
`Loading ornith-ai/Ornith-1.0-35B as Qwen3_5MoeForConditionalGeneration`. `dmesg` names the killer:

```
oom-kill:constraint=CONSTRAINT_NONE, ... task=python,pid=329722,uid=1000
Out of memory: Killed process 329722 (python) total-vm:270116540kB, anon-rss:53270728kB
```

The 35B bfloat16 CPU reference needs ~70 GiB resident. `/proc/meminfo` at the time:

| | |
|---|---|
| MemTotal | 261,425,304 kB (249 GiB) |
| MemAvailable | **52,989,704 kB (50 GiB)** |
| Cached / Shmem / Slab | 1.3 GiB / 0.3 MB / 1.0 GiB |
| HugePages_Total | 0 |
| sum of every visible process's RSS | ~1 GiB |

198 GiB is *used* and essentially none of it is attributable to a process this run can see. Five
long-running `tt_studio_*` docker containers (up 2 days, healthy, not this run's) and the TT driver's
pinned host DMA memory are the known holders. Nothing here belongs to this stage, and
`$tt-device-usage` says to preserve rather than reclaim other people's state.

This is the **same persistent host condition** the previous stage recorded in
[`../optimized_full_model/logs/host_memory_event.txt`](../optimized_full_model/logs/host_memory_event.txt),
where a standalone retry on an otherwise idle host with 55 GiB available died identically. It is an
infrastructure fault, not a model result.

## What was run instead, and why it answers the question

The shared qualitative suite ran **twice on device**, `--skip-hf` on both, through the ordinary
`build_generator` path:

| arm | policy | artifact |
|---|---|---|
| selected | the selected precision config, taken by **default** | [`readiness_qualitative.json`](readiness_qualitative.json) |
| pre-sweep | `ORNITH_PRECISION_POLICY=optimized` | [`readiness_qualitative_baseline.json`](readiness_qualitative_baseline.json) |

and [`logs/compare_qualitative.py`](logs/compare_qualitative.py) joins them to the HF column from the
previous stage's [`../optimized_full_model/readiness_qualitative.json`](../optimized_full_model/readiness_qualitative.json).

Three things make that join sound rather than convenient:

1. **The HF reference does not depend on the precision policy.** It is a torch model on the CPU. Its
   completions are the same control for the pre-sweep arm and for the selected arm; re-running it
   would reproduce the archived text, not test anything this stage changed.
2. **All three arms are asserted to have seen byte-identical rendered prompts.**
   `compare_qualitative.py` compares the `token_ids` of all three artifacts and raises if they differ,
   so the comparison cannot silently drift on prompt formatting - which is the failure
   `$qualitative-check` exists to prevent.
3. **The question this stage has to answer is a TT-against-TT one.** "Did narrowing the dense
   projections and the LM head to bfloat4_b change the visible text?" is answered by the two device
   arms, which differ *only* in the precision policy. The HF column is context for both.

## What is therefore not claimed

No fresh HF-against-TT qualitative comparison was produced on this stage's tree. The archived one
stands for the pre-sweep policy, and the selected config is compared against that pre-sweep policy
directly. If a later stage runs on a host with ~80 GiB free, re-running
`run_readiness.py --check qualitative` (without `--skip-hf`) restores the direct comparison in one
command.
