# AutoFix: generator serving contracts

## Starting evidence

`AUTODEBUG_serving_contract.md` records the initial source/AST diagnosis.
The first focused CPU run failed all five tests on the predicted missing
methods/keywords. The supervising agent owns all device execution; this
subagent ran no TTNN imports, device listings, device tests, or servers.

## Hypothesis experiments

1. **Explicit physical pool absent — verified.** `allocate_cache(3,
   num_blocks=80)` originally raised `TypeError`. The allocator now records and
   allocates exactly 80 physical blocks while retaining context 262144 in that
   allocation-only CPU experiment. This does not prove that a 80-block pool can
   execute a native-context decode (see the device constraint below).
   Standalone allocation keeps the real decoder helper's 32-block page-table
   rounding. A shared pool cannot silently receive an out-of-range standalone
   page table. Focused `-k explicit_pool`: passed.
2. **Host logits step still samples — verified.** `_replay` originally always
   ran sampling in device mode. `sample_on_device=False` now executes only the
   model trace and returns logits; captured sampler identity survives. Output
   history cannot be collected without sampling. Focused `-k host_step`: passed.
3. **Async output split absent — verified.** Copies now enqueue on CQ0 before
   returning an event. Replicated tokens read only the first shard; explicit
   host-logits copies retain all vocabulary shards. `tokens_from` accepts the
   live replicated tensor and completed first-shard host tensor. CPU queued
   snapshots and logits formatting tests pass.
4. **Slot changes lose request association — verified at orchestration level.**
   A full permutation now moves recurrence, convolution, active masks,
   token/current/RoPE inputs, device seed counters, prompt/output penalty state,
   and host seed/prompt metadata in place. Source rows are saved before writes;
   exact large integers and NaNs remain isolated across a cyclic move. The
   scheduler still owns KV page tables and parameter ordering. Focused
   `-k 'slot_cycle or boundary_refresh'`: passed.
5. **Empty shared-pool trace warmup doubles KV memory — verified by source and
   host experiment.** `ensure_traces(preserve_cache=False)` avoids snapshots
   only for explicitly empty caller pools; the default preserves existing
   caller buffers. Both paths pass the focused host test.

## Hardware follow-up: full-width penalty repeat

The supervising agent's real reduced-layer server accepted one request, then
failed concurrent serving in `remap_serving_slots` on
`repeat(saved_row, [32,1])`. `reduced_v2.server.log:129` through `:139`
records `repeat_upper_dims_rm` allocating 2209024 bytes of CB storage with only
1572864 bytes of L1 available. This is a new **verified device limitation in
the first remap implementation**, not a passing concurrency gate.

`penalty_remap_device_probe.py` allocates the actual common sampler through
`OrnithModel.build_sampler`, with all four INT32 penalty buffers at TP4 local
and gathered vocabulary widths. It checks a four-row cyclic permutation on
every replica and persistent addresses. Directly broadcasting
`WHERE([32,1], [1,V], [32,V])` avoids repeat's full-width row CB. The supervising
agent ran that candidate successfully: `penalty_remap_broadcast_retry1.json`
and `.log` show exact values on all four replicas for `prompt_mask`,
`output_mask`, and `output_counts` at `[32,65536]`, and
`output_counts_gathered` at `[32,262144]`. The probe also asserts stable
addresses. The minimal implementation now uses this device-proven broadcast
only for 2D penalty buffers; recurrent/conv state retains its prior mechanism.
The focused full-width host regression fails before this correction and
passes after it.

The actual implementation also passed `--method current`: supervising-lane
`penalty_remap_current.json` and `.log` confirm exact values across all four
penalty buffers and replicas, with stable addresses.

The original concurrency failure now passes in the supervising agent's
`reduced_v3` server (real layers 0 and 3, TP4, batch 4, logical context 262144).
`reduced_v3_requests.json` records one 131-token request followed by four
concurrent prompts of lengths 131, 129, 65, and 131. All five responses are
HTTP 200 with nonempty `choices`, eight completion tokens in usage, and no
error bodies. This verifies recovery of the specific serving failure; these
reduced-layer structural responses are not qualitative model evidence.

The first supervising-lane candidate launch exited 134 during mesh open with
Ethernet core `29-25` active timeout, before the candidate ran. The supervising
agent is preserving that log and performing serialized list/reset/list and
mesh-smoke recovery. This launch neither verifies nor refutes broadcasting.

## Reduced real-weight probe setup correction

The first `serving_contract_device_probe` run (`serving_contract_device_v1.log`)
failed during initial trace warmup, before its assertions: the paged fused
cache-update primitive requires page-table width (4096) to be no larger than
the physical KV pool (the probe initially supplied 96). The runtime server
already supplies a larger pool. The probe now allocates `width + 96 = 4192`
blocks shared across three native-context slots, still below standalone
allocation `3 * width = 12288`. It retains the exact requested physical count
and supplies unique pages for the short test prompts. This was a test geometry
error, not a successful trace or serving result; the corrected probe awaits
the supervising lane.

Its next launch (`serving_contract_device_v2.log`) failed at mesh open with
the same Ethernet active timeout after the reduced server shut down; it did
not execute the corrected probe. The supervising agent is recovering the
devices and investigating the serving shutdown path separately.

After recovery, the corrected broader probe **passed** as
`serving_contract_device_v3.json` and `.log`. It used real layers 0 and 3, batch
3, native logical context 262144, and exactly 4192 shared physical blocks. State
remapping matched on all replicas, including NaNs and exact large seed values;
buffer addresses stayed fixed. Two deferred decode reads matched synchronous
baseline tokens `[[12,12,220],[220,220,220]]`. Explicit host compatibility
returned finite `[3,248320]` logits without advancing sampler state; counters
record six model replays and five sampler replays. These reduced checks prove
the generator contracts, not full-model accuracy.

## Commands and status

Host verification (actual source methods with torch-backed TT boundaries):

```bash
USER=hous python_env/bin/python -m pytest -q \
  models/autoports/ornith_ai_ornith_1_5_9b/tests/test_generator_serving_contract.py \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/test_generator_host_contract.py \
  --confcutdir=models/autoports/ornith_ai_ornith_1_5_9b/tests
```

Initial five tests failed; each passed after its corresponding change. The
first combined run passed 21 tests. The final combined run passes **25 tests**,
including async logits, both cache snapshot policies, and full-width penalty
remapping. Black checks and byte compilation pass for both implementation
files and all three focused test/probe files; `git diff --check` passes.

Prepared but not executed by this subagent:

```bash
USER=hous ../state/serving-env/bin/python -m \
  models.autoports.ornith_ai_ornith_1_5_9b.tests.penalty_remap_device_probe \
  --model-path /home/hous/dev/ornith-1.5-9b/upstream --method broadcast \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/penalty_remap_broadcast.json

USER=hous ../state/serving-env/bin/python -m \
  models.autoports.ornith_ai_ornith_1_5_9b.tests.serving_contract_device_probe \
  --model-path /home/hous/dev/ornith-1.5-9b/upstream --layers 0,3 \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/serving_contract_device.json
```

The second probe uses real reduced-layer weights and actual cache allocation,
preserves native logical context, verifies state remaps and exact async replay
tokens, and exercises explicit host compatibility separately. It does not
replace full-model accuracy, all-profile serving, or release validation.
Python-only changes require no C++ build. No full-model accuracy or performance
improvement is claimed from these contract tests.

## Final status of this delegated change

The missing Python serving contracts are implemented, 25 host tests pass,
the actual penalty-remap implementation and broader generator probe pass
exact hardware checks, and the original reduced concurrent-serving failure
passes. The separate adapter async checks, full-model checks, and all-profile
serving/release gates remain at the parent stage. A subsequent source-verified
fresh-prefill penalty activation repair is documented separately in
`AUTOFIX_prefill_penalty_admission.md`.
