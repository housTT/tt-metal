# Integrated H2 serving-path review

2026-09-08, source-only review of the supervising checkout plus existing evidence.
No main-checkout edits, TTNN imports, device commands or profiling. No correctness
blocker found. Final full-model serving performance remains the supervisor's gate.

**Canonical device sampling.** [Adapter `_device_sampling`](../../tt/generator_vllm.py#L142)
rejects missing device parameters unless explicit host compatibility is enabled;
recorded B1/B32 server commands set `ORNITH_VLLM_ALLOW_HOST_SAMPLING=0`.
[Device decode](../../tt/generator_vllm.py#L245) requires tracing and passes
`tokens=None`, `start_pos=None`, `sample_on_device=True` into the generator.
[`_replay`](../../tt/generator.py#L726) submits the separate model and sampler
traces nonblocking. [`_sample_device`](../../tt/generator.py#L334) binds
`tt_out_tok=self._inputs[0]` and advances the persistent UINT32 seed on device;
the model consumes that same feedback allocation and advances current/RoPE
positions inside [its captured decode](../../tt/model.py#L600). Penalty counting
is inside the common sampler's captured `_run_sampling` operation sequence.

**First-token sampling.** [`_sample_prefill_device`](../../tt/generator.py#L561)
stages transient eager logits, retires them before recapture/replay, and copies
back into the current canonical logits after recapture. A traced B1 prefill
already populates that canonical destination. The final call sets
`require_trace=True`; a late cache miss raises instead of falling back to eager
sampling. The standalone helper's default eager branch is outside this serving
call path. B32/unsupported prefill bodies remain eager model work; their sampling
still replays the canonical trace. See [B1](h2_sampling_b1.json) and
[B32](h2_sampling_b32.json): eight exact all-rank sampler cases each, four actual
model-prefill calls, twelve prefill sampler replays, native allocation tracking
including programs, and zero trace bytes after teardown.

**Persistent inputs and boundary refresh.**
[Sampler backup/mask/logits buffers](../../tt/generator.py#L525) are allocated
and copy/merge programs warmed before initial capture. Sorted admission membership
controls changed-only mask upload. Recapture preserves the masks and membership
cache. INT32 predicates preserve exact token/UINT32-seed and integer history
values in untouched physical lanes. B1 must preserve all31 other sampler lanes.
Backup copies and row-major/tiled integer restoration remain admission work;
they are absent from steady decode. A separate follow-up now skips
mask refresh, backups and restoration when the sorted admission rows are exactly
all32 unique physical lanes, after canonical logits staging/recapture. B1 and
partial admissions retain their exact preservation. This follow-up passes CPU
contracts and the supervisor's B32 native-tracker rerun (h2_sampling_b32_final.json).

Adapter sampling reconfiguration, fresh token/current/RoPE refresh and hybrid
activity-mask writes occur at admission/remap/mode boundaries. Page-table device
copies occur only on changed contents in
[`_refresh_table`](../../tt/generator.py#L197). The
[final worker contract](after_async_worker_contract_v2.json) predates only the
three-line all32 admission shortcut, which is unreachable in its B4 cases. It
passes synchronous/deferred-stale/changed-page/remap
comparisons. Its two-step steady window records two model and two sampler replays,
zero token/current/RoPE/page refreshes, zero sampling updates and zero host decodes.
Changed-page and remap windows each record exactly one page-table refresh;
persistent addresses remain unchanged. Async output copies precede reuse. This
focused adapter probe does not exercise the scheduler; its historical
`supports_async_decode_promoted=false` metadata is not a capability verdict.
The actual adapter advertises `supports_async_decode=True`; recorded real servers
use `--async-scheduling`, and the supervising real-server counters establish that
the plugin's `non_dp_async_scheduling=True` path was exercised.

**Actual measured page-table shape.** The pinned plugin `model_runner.py` hash
`d4fc721d8eb965c5c1e15cdb3ebf28e43f78c63bc2ce792c7c565e40c864e303`
matches [B1](before_b1.command.json), [B32](before_b32.command.json) and
[H1 B1](h1_b1.command.json) manifests. Source chain under
`/home/hous/dev/ornith-1.5-9b/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/`:

- `input_batch.py:235` constructs persistent CPU block tables for max context;
  `model_runner.py:323` uses262144 and block64. Width at `:357` is
  `min(ceil(262144/64), physical_blocks)=4096`; the observed B1 pool has4098
  blocks and the pool budget is independent of batch capacity.
- `model_runner.py:1027` obtains width4096 snapshots; the shorter-width branch
  at `:1035` requires DP>1. These P150x4 measurements use TP4/DP1. At `:1138`
  decode rows are padded to fixed capacity1 or32, preserving width4096.
- `async_decode.py:573` passes `model_input.block_tables` directly as
  `page_table`. [Adapter `_page_table`](../../tt/generator_vllm.py#L127)
  therefore sees INT32 `[1,4096]` or `[32,4096]` and returns the same tensor.

An AST execution of the actual adapter method confirmed object/address identity
for both shapes with `torch.zeros` patched to fail if padding were attempted.
Consequently the conditional shorter-table adapter allocation is not exercised
by measured steady decode. No adapter page-cache patch is warranted. The upstream
scheduler still makes CPU row snapshots (`input_batch.py:660`) and pads inactive
rows; generator host equality checks also remain. These are distinct from device
input refresh, and no claim of zero host scheduler work is made.

**Conversions and inherited sampler selection.** Current BF16 logits bypass the
common sampler's no-op BF16 typecast (`models/common/sampling/tt_sampling.py:904`).
Final norm/head layout operations are inside the captured model graph, not eager
per-step dispatch. The selected head directly accepts the final-norm layout.
The outer sampler wrapper uses the common split strategy with physical top32
per65536-column vocabulary shard and128 gathered candidates; semantic greedy is
k1/p0/temperature1. The [inherited exact greedy comparison](../optimized_full_model/sampler_greedy_comparison.json)
measured split **0.572463ms** versus force-argmax **2.744207ms**, both matching CPU
choices. [Selection context](../optimized_full_model/README.md#L89) rejects
full-vocabulary gather/host argmax for this path. These are inherited reduced
sampler measurements, not a new H2 server-speed claim.

Pre-shortcut model SHA256 receipts (also in the final worker-contract artifact);
the final B32 probe and serving manifests record the three-line successor:

```text
generator.py       158268dbf2669afe7f0cc44c7742b8ecf9ef1fc5738c9db00637ee2cbbfc2059
generator_vllm.py  6743d2f82e7d3fa91a92019aae2a75eab6837a19559c12e7fb96d250240dd3f6
model.py           f9da26100bdf48dd1a2e487a3687895e75b2b385f0a32cf21d10c2786907e615
```
