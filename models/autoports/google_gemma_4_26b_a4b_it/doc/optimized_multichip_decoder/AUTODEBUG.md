# AutoDebug Report: google_gemma_4_26b_a4b_it Optimized Multichip Decoder

Focus path: `models/autoports/google_gemma_4_26b_a4b_it`

Scope: inspection only. I read source, tests, retained profiler/candidate/watcher
artifacts, and the TTNN sparse-matmul contract. I did not edit implementation
files, run hardware tests, launch profiler/watcher, reset devices, or invoke the
AutoDebug launcher.

## Headline Findings

### 1. Verified: decode uses fixed `nnz=8` without proving the exact post-conversion sparse count

The optimized multichip packed expert decode path passes `nnz=TOP_K_EXPERTS`
to both sparse matmuls (`tt/multichip_decoder.py:1547-1576`). The `sparsity`
operand is the routed score tensor itself, either preserved as row-major metadata
or converted to row-major (`tt/multichip_decoder.py:1521-1528`).

That tensor is produced by the optimized router after FP32 router input, FP32
linear, BF16 logits, `topk`, softmax, BF16 scatter into zeros, and optional
per-expert scaling/typecast (`tt/optimized_decoder.py:2020-2054`). The code does
not construct an independent binary eight-entry mask. It relies on all eight
selected score values remaining nonzero in the exact tensor scanned by
`ttnn.sparse_matmul`.

The TTNN contract is exact, not an upper bound: the nanobind doc says `nnz` must
equal `count_nonzero(sparsity)` and warns that mismatch can deadlock
(`ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1077-1139`). The device op
comments repeat that the sender multicasts once per nonzero while receivers loop
`nnz` times (`ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp:223-229`),
and the dataflow sender asserts the count only after issuing multicasts
(`reader_bmm_tile_layout_in0_sender_padding.cpp:436-444`).

I found no retained test/artifact that observes the actual post-conversion
`sparsity` tensor and asserts `count_nonzero == 8`. The stacked trace artifact
uses `torch.topk(..., k=8)` on captured routing values
(`tests/test_multichip_decoder.py:3719-3724`), which always reports eight indices
and does not prove eight nonzero sparse entries.

Impact: this is a real source-level safety gap. I did not reproduce a hang, but
the current code does not prove the precondition required by the callee. BF16
softmax/scatter/lowering, adversarial large logit gaps, or disabled expert-scale
folding with tiny/zero scales are enough reasons not to infer exact nonzero count
from `topk(k=8)` alone.

Remediation experiments:

- Safest: remove fixed `nnz` for decode and measure the runtime-inference path.
- Faster alternative: use indexed sparse mode from `top_indices` and keep score
  weighting separate, so the loop count comes from the compact index list rather
  than score nonzeros.
- Add a hardware assertion run that captures the exact `sparsity` operand after
  TTNN conversion for TP1/TP2/TP4, B1/B32, real recorded activations, and
  adversarial logit/scale cases.

### 2. Verified: TP2 packed expert geometry was not retuned for the packed 704-wide output

TP2 derives `expert_gate_per_core_n` from the unpacked local MoE width:
`profile.local_moe_intermediate_size // TILE_SIZE` (`tt/multichip_decoder.py:423-425`).
For TP2 this is 352 / 32 = 11 tiles, so the default becomes `per_core_N=1`.
The packed path then uses `packed_width = 2 * local_width` (`tt/multichip_decoder.py:1530-1536`),
and setup records `decoder.packed_expert_width = 2 * profile.local_moe_intermediate_size`
(`tt/multichip_decoder.py:668`), i.e. 704 / 32 = 22 tiles.

The helper would accept the more natural packed geometry: `_optimized_sparse_decode_config`
requires `n_tiles % per_core_n == 0` and chooses subblock width 2 when
`per_core_n=2` (`tt/optimized_decoder.py:803-829`). For `n=704`, that is 11
projection cores and output subblock 1x2. The prefill helper has the same
legalization shape for the actual 32-token chunked path where groups=1
(`tt/optimized_decoder.py:760-799`).

The retained TP2 profiler rows confirm the selected path still uses 22 cores,
`in0_block_w=44`, `Output Subblock W=1`, and advice to try area >= 2:
`profiler/p150x2_sliding/decode_ops.csv:76`,
`profiler/p150x2_full/decode_ops.csv:58`, and repeated prefill rows such as
`profiler/p150x2_sliding/prefill_ops.csv:501`. The packed candidate artifacts
only set packing/coherent flags, not expert geometry overrides
(`candidates/packed_expert_both/p150x2/layer0_sliding_attention_seq1024_batch1_host_timings.json:22-30`).

Impact: the stage's dominant prefill family and a material decode row were left
on geometry selected for a different N dimension. This is an optimization gap,
not a proven correctness bug.

Remediation experiments:

- Run a precision-locked TP2 packed expert sweep with `expert_gate_per_core_n=2`
  and `expert_gate_out_subblock_w=2` for decode, plus a corresponding prefill
  hook/kwargs path for `prefill_expert_per_core_n=2`.
- Retain row CSVs, whole-layer latency, B1/B32 PCC, and trace replay evidence for
  both sliding and full profiles.

### 3. Verified: TP2 sliding QKV DRAM advice was neither tried nor blocked

The TP2 sliding decode profiler advises a DRAM-sharded program for the QKV row:
`MatmulDeviceOperation 32 x 2816 x 4096`, 64 cores, DRAM-bound
(`profiler/p150x2_sliding/decode_ops.csv:47`, repeated at lines 128 and 193).

The source accepts `qkv` as a multichip DRAM-sharded role
(`tt/multichip_decoder.py:723-730`) and only hard-bypasses full-attention QKV
(`tt/multichip_decoder.py:1064-1069`). The retained TP2 DRAM candidate sets only
`GEMMA4_MULTICHIP_DRAM_SHARDED_ROLES=o_proj`
(`candidates/tp2_dram_o/p150x2/layer0_sliding_attention_seq1024_batch1_host_timings.json:22-28`).

Impact: a current, concrete profiler recommendation remains open for TP2 sliding
decode. I found no capacity, divisibility, L1, or op-contract failure artifact
that would close it.

Remediation experiment:

- Try TP2 sliding QKV DRAM with `GEMMA4_MULTICHIP_DRAM_SHARDED_ROLES=qkv` or
  `qkv,o_proj` under the selected precision/residual contract. If it fails,
  retain the exact validation/capacity/op-contract blocker.

### 4. Verified: activation/CCL precision coverage is TP4-global and incomplete for TP2

The activation-BFP8 candidate directory contains only one summary JSON. It is a
P150x4 global activation switch with `GEMMA4_MULTICHIP_ACTIVATION_DTYPE=bfp8`
and no JUnit, timing JSON, raw log, source hash, or role-isolated matrix
(`candidates/activation_bfp8/pcc_result.json:1-17`).

The CCL-BFP8 evidence is P150x4-only. Its summary reports P150x4 B32 PCC misses
and decode regressions (`candidates/ccl_bfp8/p150x4/pcc_result.json:1-15`), and
the timing artifacts show four device IDs plus `GEMMA4_MULTICHIP_CCL_DTYPE=bfp8`.
No TP2 CCL-BFP8 artifact exists under `candidates/ccl_bfp8/`.

TP2 and TP4 are not interchangeable here. The runtime chooses Linear/one-link
for TP2 and Ring/two-link for TP4 (`tt/multichip_decoder.py:955-990`), with
different rank count and numerical reduction structure. The selected TP2
coherent R22 evidence is a BF16/default control, not crossed with BFP8 activation
or CCL. Existing "real-weight" harnesses still seed random activations
(`tests/test_functional_decoder.py:326-329` and `:1614-1616`), so they are not
recorded target-model activation evidence.

Impact: the global statement that activation/CCL precision is closed is not
supported. This is an evidence and optimization-coverage gap, not proof that
BFP8 would pass on TP2.

Remediation experiments:

- Run TP2 coherent-R22 BF16 versus BFP8 CCL for layer0/layer5 with B32 PCC and
  B1 traced latency, retaining JUnit, timing JSON, exact wrapper command, source
  hashes, hardware IDs, and raw logs.
- Isolate BFP8 activation by role: attention/CCL-facing tensors, dense MLP, MoE,
  residual/norm. Cross the useful subsets with selected TP2 and TP4 topologies.
- Use recorded target-model activations, or relabel existing checks as
  real-weight synthetic-activation tests.

### 5. Verified: profiler/accounting/provenance artifacts are not reproducible as written

All retained `profiler/*/{prefill,decode}_report.txt` files are CSV-mode console
chatter, not human-readable `tt-perf-report` tables. For example,
`profiler/p150x2_sliding/decode_report.txt:1-15` prints detected CSV format,
"Writing CSV output", warnings, and summary generation, but no rendered rows.
This matches the exact artifact mistake called out in `optimize/SKILL.md:515-536`.

`perf_summary.json` mixes final 30-replay host timings with independently
captured 3-replay profiler device totals (`perf_summary.json:4-13` and
`work_log.md:151-188`). It reports modeled DRAM percentages but not theoretical
bytes, aggregate bandwidth, or roofline ms/token from the same run, which the
optimize instructions require (`optimize/SKILL.md:97-107`).

The final timing JSONs also embed the delegated functional-test command, not the
actual multichip wrapper command. Example:
`final_post_format/p150x2/layer0_sliding_attention_seq1024_batch1_host_timings.json:17-42`
has measured decoder `tt/multichip_decoder.py` and device IDs `[1,0]`, but
`exact_command` points at
`test_functional_decoder.py::test_functional_decoder_perf_profile`. The wrapper
actually delegates through `test_required_profile_perf_profile` or
`test_multichip_perf_profile` (`tests/test_multichip_decoder.py:3154-3180`).
The code cause is clear: `_evidence_provenance` uses `GEMMA4_EVIDENCE_COMMAND`
when present (`tests/test_functional_decoder.py:51-94`), but the multichip
harness installation does not set it (`tests/test_multichip_decoder.py:3078-3098`).

Impact: the measurements may be real, but the evidence package does not provide
the promised reproduction commands or same-run accounting.

Remediation experiments:

- Regenerate no-CSV human `tt-perf-report` tables from the retained raw CSVs,
  keeping CSV-mode stdout under console-log filenames.
- Set `GEMMA4_EVIDENCE_COMMAND` in the multichip wrapper before delegating to the
  functional perf harness, including fallback policy and wrapper nodeid.
- Rerun final default timing/profiling or assign explicit run IDs and report
  host/device/roofline limitations without mixing them as one same-run result.

### 6. Verified: watcher evidence is scoped because Ethernet watcher was disabled

The only final watcher command sets `TT_METAL_WATCHER_DISABLE_ETH=1`
(`doc/optimized_multichip_decoder/work_log.md:221-232`). The host API turns that
runtime option into `FORCE_WATCHER_OFF` for Ethernet kernels
(`tt_metal/impl/host_api/tt_metal.cpp:1459-1463`).

The final artifacts do include useful positive evidence: ten tests pass in
`final/watcher.xml`, `final/watcher.log` is clean for worker/core status, and
`final/tt_smi_post.json` reports healthy devices with zero GDDR errors. However,
`optimize/SKILL.md:508-513` permits disabled-Ethernet retry only after retaining
the ACTIVE_ETH config-buffer overflow signature and recording a scoped
limitation. I found no enabled-Ethernet watcher attempt, overflow log, or scoped
limitation under `doc/optimized_multichip_decoder/`; README still calls the run
unqualified "watcher-clean" (`README.md:161-162`).

Impact: this does not invalidate the worker-watcher evidence, but it does not
cover the Ethernet kernels carrying persistent async CCL.

Remediation experiment:

- Run a bounded selected-profile watcher control with Ethernet watching enabled.
  If ACTIVE_ETH buffer overflow reproduces, retain that failure log, rerun with
  `TT_METAL_WATCHER_DISABLE_ETH=1`, and explicitly scope the limitation.

## Other Potential Issues

- Packed expert construction is not transient-capacity-neutral. Setup uploads
  separate expert gate/up tensors (`tt/multichip_decoder.py:593-601`), uploads
  the packed tensor (`:659-678`), then deallocates the separate tensors
  (`:679-685`). Persistent accounting can still be correct, but TP1 full-stack
  headroom is only 1,037,824 bytes, so full-model bringup should account for
  construction order before large KV allocation.
- Prefill profiler rows were generated with `--active-experts 8`, but routed
  prefill uses a per-32-token union and `nnz=None`
  (`tt/optimized_decoder.py:3013-3026`). Device durations remain useful; the
  `active=8/128` labels and modeled FLOPs/bytes are not proven to describe the
  actual group union.
- TP4 full decode is reported slightly slower than baseline
  (`perf_summary.json:31-32` speedup 0.998812). The report discloses it, but the
  "measurement noise" classification lacks retained repeated-run variance or
  alternating-order stability evidence.

## Refuted Or Controlled Hypotheses

- Not a replicated single-chip fallback: final timing JSONs record
  `measured_decoder_path` as `tt/multichip_decoder.py` and TP2/TP4 hardware
  device IDs. The command field is wrong, but the measured module path supports
  that the multichip implementation ran.
- Not a proof that BFP8 activation/CCL is invalid everywhere: the retained
  failures are TP4/global or TP4 CCL-specific. TP2 and role-isolated policies
  remain unmeasured.
- Not a proof of an observed sparse deadlock: the issue is a missing exact-count
  precondition for a kernel whose contract requires it. Hardware reproduction was
  intentionally out of scope.

## Commands And Checks Run

Read-only inspection commands only: `sed`, `nl`, `rg`, `find`, `head`, `tail`,
`wc`, `stat`, small Python JSON/CSV summaries, `git status --short`, and
subagent sidecar audits. No TT hardware, profiler, watcher, reset, server, or
implementation-edit command was run.
