# Full-output stall AutoFix triage

Date: 2026-08-28 (America/New_York)

## Failing workload

The full 48-layer vLLM server was running on P300 devices 0 and 1 with
`max_model_len=262144`, `max_num_seqs=2`, async scheduling, decode-only
tracing, and the final on-device sampling profile. The direct non-aligned
prompt check sent two-token greedy requests in this order:

```text
logical prompt lengths: 1, 1, 63, 63, 64, 64, 65, 65, 67, 67, 127, 127, 129, 129
```

The first three requests completed. The repeated length-63 request entered
prefill at 20:18:55 and did not complete before the client's 180-second
timeout. The server log stopped advancing at 20:19:15. The latest model
marker reported `completed_requests=3` with event `prefill_start`; vLLM's
Prometheus counters reported exactly three successful requests,
`prompt_tokens=65`, and `generation_tokens=6`. No fourth HTTP response was
written.

Server log:
`readiness_vllm/final_virtual_b2/server.log`

Client:
`readiness_vllm/run_non_aligned_prompt_check.py`

## Host process evidence

The client was blocked polling its established HTTP socket until its own
timeout. The API server remained responsive to `/metrics`. Engine statistics
printed `Running: 0, Waiting: 0`, but that was an async-scheduling accounting
artifact: the synchronous worker call was still live.

Repeated `py-spy dump --pid 896761 --nonblocking` snapshots placed the
EngineCore main thread here:

```text
ttnn.to_torch
  _read_compact_route_ids (multichip_decoder.py:1434)
  _routed_experts (multichip_decoder.py:1700)
  _moe
  prefill_forward_fractured
  Qwen38FullModel.prefill_forward
  Qwen38Generator._prefill_forward_virtual
  generator_vllm.prefill_forward
  vllm_tt_plugin.model_runner.submit_prefill
```

This rules out a lost final async decode output, release-hook failure, and a
client-only wait. The host route-ID read was a synchronization victim queued
behind an earlier device operation.

## Live TT triage evidence

The workload was kept alive while the read-only capture ran:

```bash
timeout 120 tools/tt-triage.py --llm-output \
  --run=dump_callstacks \
  --run=dump_running_operations \
  --run=check_eth_status \
  --run=check_arc
```

`dump_running_operations` identified the first non-completing operation on
both devices:

```text
Op Id:       91449
Op Name:     ReshapeViewDeviceOperation
Input:       logical [1, 63, 1280], tiled BF16, interleaved DRAM
Previous:    91448 TilizeWithValPaddingDeviceOperation
Devices:     0, 1
Core count:  46
```

Every listed reshape reader NCRISC remained in
`reader_reshape_tiled.cpp:68`, blocked in `cb_input.reserve_back(1)`. The
corresponding `writer_reshape_tiled` BRISCs showed `GO`, completion waypoint
`X`, and PCs outside the running kernel ELF: the consumers had exited while
the producers still attempted to reserve the one-tile input circular buffer.
CQ dispatch and the host `to_torch` were downstream waiters. ARC heartbeats
were healthy on both devices. Devices 0 and 1 remained active at 59 W and
56 W, respectively.

The Ethernet checker observed pre-existing retrain counts of two on three
links, but the stopped operation was a local tiled reshape on both ranks, not
a CCL/fabric operation. No ARC or DRAM-health failure was reported.

## Source contract and stop site

The only endpoint operation consuming tiled `[1, logical_tokens, 1280]` is
`Qwen38FullModel.embed_tokens`. The prior implementation expanded the TP2
embedding shard as follows:

```python
expanded = ttnn.reshape(embedded, (1, rows, 1, 1280))
expanded = ttnn.repeat(expanded, (1, 1, 4, 1))
residual = ttnn.reshape(expanded, (1, 1, rows * 4, 1280))
```

For `rows=63`, the first reshape is transformed internally from logical
`[1,63,1280]` to rank-three `[63,1,1280]`; it therefore uses
`ReshapeViewTiledProgramFactory`, exactly matching op 91449. The immediately
preceding tilize also matches the embedding endpoint. No other full-model
operation accepts this rank-three embedding shape.

The tiled reshape factory assigns identical output-page ranges to the reader
and writer and gives their input CB one tile. The reader pushes one input tile
for each distinct mapped input page; the writer must pop the same sequence.
The capture proves that invariant failed on an identical-shape cache reuse:
the first length-63 request completed, while the next length-63 invocation
reused the cached workload and left all readers reserving after its writers
had exited. This tt-metal checkout's tiled reshape was migrated to a
`ProgramDescriptor` workload/cache implementation in commit `b2af0cd67b4`;
the model does not own or need that mapping mechanism.

## Smallest model-local fix

For prefill shapes (`rows > 1`), the endpoint now constructs the same
token-major four-stream ordering with:

```python
expanded = ttnn.repeat_interleave(embedded, repeats=4, dim=-2,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
residual = ttnn.unsqueeze_to_4D(expanded)
```

`repeat_interleave` performs the row interleave through its row-major data
movement path, and the final rank extension is a view. It never launches the
failing tiled reshape map. The ABI remains `[1,1,4*rows,1280]`; no layer,
sampler, cache, or lifecycle behavior changes. The `rows == 1` decode ingress
retains the original reshape/repeat/reshape source: at that shape both
reshapes collapse to the identical rank-three `[1,1,1280]` view, so its trace
still captures only the original repeat operation. The fix therefore adds
zero traced decode operations and zero decode replay latency by construction.

Focused regression:

```text
test_repeated_nonaligned_embedding_expansion_has_no_tiled_reshape_stall
lengths = 1,63,1,63,1,63
checks = completion plus exact four-stream copies on both TP ranks
```

The CPU source-contract gate passed immediately. After the owning root task
cleaned up the timed-out server, no vLLM, EngineCore, client, or pytest process
held `/dev/tenstorrent/{0,1}`. Both devices reported healthy DRAM and
heartbeats, so no reset was required. The focused TT regression then passed:

```text
artifact log: readiness_vllm/autofix_repeated_nonaligned_embedding.log
artifact JUnit: readiness_vllm/autofix_repeated_nonaligned_embedding.xml
result: 1 passed, 0 failed, 0 errors
elapsed: 6.166 s suite, 6.073 s test
source digest: 50f510d7fe23214242c751880d292d4ec97535d0538cf6a53c5169ea4c2a4030
model.py sha256: 68b831ebc406754f4567b5eede4ea98ed98059318c554b52af7381254a98cdb5
```

This test executes the old and new decode ingress identically: for every
physical-B1 decode row, Python selects the `rows == 1` branch and invokes the
same two reshape views around one `ttnn.repeat`. `repeat_interleave` is only
reachable from multi-token prefill. Consequently the decode comparison is:

```text
traced device operations before: 1 repeat
traced device operations after:  1 repeat
added traced decode operations:  0
added trace replay calls:        0
decode replay latency delta:     0 by identical executed operation sequence
```

## Adapter-level reused-program regression

The closer adapter regression
`test_reused_prefill_programs_alternating_one_and_sixty_three_tokens` creates
the reduced real `(0,1,3)` stack, allocates its vLLM-owned attention cache,
alternates logical prompt lengths `1,63,1,63` through
`Qwen4ExpForConditionalGeneration`, decodes one traced token after every
prefill, and requires identical outputs for each repeated prompt shape.

The first invocation failed before cache allocation or model execution because
the test had not declared the adapter's supported reduced-stack environment;
the production guard therefore correctly rejected `num_layers=1` instead of
the full model's 12 QSA layers. The gate now uses the same explicit
`QWEN38_VLLM_LAYER_INDICES=0,1,3` control as the neighboring virtual-B2 TT
gate. The failure and corrected rerun share these evidence paths:

```text
readiness_vllm/autofix_reused_program_virtual_adapter.log
readiness_vllm/autofix_reused_program_virtual_adapter.xml
```

Corrected command (the `monkeypatch` in the test supplies the reduced-layer
adapter environment):

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
TT_VISIBLE_DEVICES=0,1 \
TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto" \
RUN_QWEN38_VLLM_REUSED_PROGRAM_TT=1 \
pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_vllm_virtual_slots_tt.py::test_reused_prefill_programs_alternating_one_and_sixty_three_tokens \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/autofix_reused_program_virtual_adapter.xml
```

The corrected regression passed in 17.641 seconds (17.570 seconds JUnit test
time). Both same-shape comparisons were exact across prefill and the following
traced decode token:

```text
logical length order: 1,63,1,63
length 1 tokens:      [[248044,6021], [248044,6021]]
length 63 tokens:     [[152489,27210], [152489,27210]]
program cache entries after each request: 700,761,761,775
trace mode: token_out
prefill admissions while trace live: 3
virtual assignments/releases: 4/4
active/valid slots on exit: 0/0
sequential-B1 bank commits/restores: 0/0
stale rejections: 0
```

The changing program-cache total is expected because later adapter/decode
paths encounter operations not reached by earlier lengths. The decisive
result is that the second length-63 request completed with its first and
decode tokens exactly matching the first length-63 request; no cached tiled
reshape program is present in the fixed prefill endpoint.

On exit, no vLLM, EngineCore, pytest, or client process remained and neither
`/dev/tenstorrent/0` nor `/dev/tenstorrent/1` had a process holder. The root
task's post-run `tt-smi` audit reported healthy DRAM and heartbeats; no reset
was required.

## Final verdict

Fixed. The fourth-request stall was a reused, unaligned tiled reshape program
in embedding expansion, not async output delivery or vLLM lifecycle cleanup.
The prefill-only `repeat_interleave` replacement removes that program while
preserving exact stream order and the existing one-repeat decode trace. Both
the direct repeated endpoint gate and the adapter-level prefill/decode gate
pass, and the process/device audit is clean.
