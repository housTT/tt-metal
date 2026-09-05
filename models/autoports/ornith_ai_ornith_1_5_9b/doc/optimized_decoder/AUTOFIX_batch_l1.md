# AutoFix: batched L1 resources

Starting evidence: `AUTODEBUG_batch_l1.md`, and the exact command/environment/source
archives for `sharded_real_batch_contract`, `kda_real_batch32_contract`, and
`combined_decode_batch_contract`. All three original optimized-decoder sources
match the preserved starting file. This repair uses original-host BFP4 weights,
LoFi projections, original FP32 GDN projection output, and residual32 geometry.
No functional/fused decoder, dtype policy, public batch or context limit changed.

## Hypothesis experiments

| Hypothesis/control | Result | Verdict / retained change |
| --- | --- | --- |
| H1: skip only redundant FP32 cast | `autofix_batch_h1_skip_cast_v2`: B32 prefill PCC .999026, then identical 16,777,216-byte allocation failure moves to FP32 add; live 1,310,720 B/bank | Verified padding pressure; cast removal alone refuted as a repair |
| H1: compact norm and residual helpers | `autofix_batch_h1_compact`: original FP32 base B32 passes; prefill .999026, decode .998976 | Keep physical `[1,B,H]` residual/norm working tensors, one tile row per shard; preserve public/mixer `[B,1,H]` |
| H2: only state=DRAM, original residual | `autofix_batch_h2_dram_state`: prefill .999026 passes, later decode state-read CB collision | Verified persistent state caused first prefill norm failure |
| H2 transient: compact residual, DRAM state, L1 intermediates | `autofix_batch_h2_compact_dram_state`: outer requests 67,108,864 B / 610,304 B per bank, with 901,120 B allocated and 535,552 B free | Verified state-sized transient independently exceeds capacity |
| H2 transient: change intermediates to DRAM | `autofix_batch_h2_compact_dram_both`: B32 passes; prefill .999026, decode .998986 | Keep setup-selected placement for persistent state and intermediates |
| Upper L1 budget | `autofix_batch_h2_l1_boundaries`: B8/B13/B16 pass HF prefill/decode. `autofix_batch_h2_l1_upper`: B24 decode CB collision, B31 prefill norm collision | Persistent-state control supports a 300 KiB/bank bound; sharded public-input tests refine the transient bound below |

The first exact matrix (`autofix_batch_exact_resources`) passed 14/16 cases,
including all DRAM-input cases and B32 sharded inputs, but found B13/B16 linear
outer-matmul CB collisions with borrowed sharded inputs still live. B13's lowest
L1 address was 272384 versus static CB end 275456 (3072-byte overlap); B16's was
159744. `autofix_batch_l1_public_bound` proves B8/B12 exact restored/stressed replay
with L1 state and intermediates and sharded public inputs.
`autofix_batch_l1_state_dram_outer` proves B13/B16 with only intermediates moved
to DRAM; persistent state remains L1. The final setup policies therefore allow
300 KiB/bank for persistent state (B<=16) and 224 KiB/bank for intermediates
(B<=12). Both requests fall back to DRAM independently above these bounds.

These are validated working-set budgets, not a claim that every larger
allocation necessarily fails. FP32 state uses 4096-byte tiles, allocated evenly
across compute/storage banks; `ceil(state_tiles/banks)*4096` selects placement in
`allocate_state`, before capture. There is no state relocation in forward/replay.
The original B1 requested L1 policy remains selected.

Residual addition still promotes BF16 to FP32, adds homogeneous FP32 inputs, and
rounds once to BF16. The redundant same-dtype cast is skipped. Compact tensors
remove per-user padded rows while `_linear` still sees `[B,1,H]` and selects decode
projections normally. Norm output and final helper output expand at their public
boundaries. No borrowed input is explicitly deallocated or mutated.

## Verification in progress

- `autofix_batch_integrated_contract`: combined original `batched or traced_decode`
  selection without maxfail passes all 12 cases, both layer kinds and B1/4/32
  traced decode, B4/32 prefill/decode, ragged full attention B4/B13.
- Exact batched trace, all-batch component oracle, watcher, KDA and B1 timing
  validation are recorded below once complete.

Every meaningful hardware experiment uses `record_run.py`; its adjacent
`logs/<name>.provenance.json` contains exact argv, environment, return code and log
hash, and `logs/<name>.sources.json.gz` preserves the actual Python source. The
initial `autofix_batch_h1_skip_cast` launcher failed before importing pytest
because a resolved Python symlink escaped the virtual environment; `_v2` corrected
that launcher and is the actual H1 experiment. No hardware reset or hang occurred.

## Broader boundary findings

`autofix_batch_final_resources` retains the corrected independent state budgets:
34/38 tests pass. All linear HF and exact trace cases pass; both kinds pass exact
trace B4/8/13/16/32 for DRAM and sharded public input. It also exposes two separate
full-attention defects: B17/B31 allocation failures at grouped RoPE concat, and
B12 restored-output nondeterminism (max absolute .0625; KV state remains exact).
These are retained failures, not waived or hidden by selecting a narrower batch
range. The coordinator authorized a narrow optimized RoPE lifetime override;
focused follow-up artifacts are pending.


## Prime-batch RoPE repair

The broader B17/B31 failures occur at grouped native-RoPE concat, whose per-core
output is `B*32*256*2` bytes when a prime batch occupies one core. The
`autofix_batch_prime_lifetimes` control explicitly freed rotated-input and
partial/tail buffers and immediately interleaved each completed output: B17/19/23
passed, B29/B31 still failed while allocating a single joined shard. For B29,
475136 B were requested with 997376 B allocated and only 439296 B free.
`autofix_batch_prime_dram_concat` interleaves rotated and tail partials before
concatenating in DRAM; all B17/19/23/29/31 HF checks pass. Merely requesting DRAM
output on sharded-input concat does not remove the allocation: the wrapper
(`concat_device_operation.cpp:388`) first builds a sharded output. The retained
optimized override handles only grouped users, preserves native rotary arithmetic
and rectangles, and leaves the B1 inherited path unchanged. Original q/k are
borrowed and never explicitly freed.

## Independent SDPA query mapping repair

Fresh source-only report: `AUTODEBUG_batch12_attention.md`. B12 RoPE places Q in
a 6x2 rectangle; the SDPA8x8 sharded reader instead fetches Q from its own first
12 cores, including two with no Q shard and four with another user's Q. Exact KV
restoration therefore did not protect attention output. Both isolated controls
pass the two public memory variants with exact restored eager/trace/stress output
and state plus per-user HF gates:

- `autofix_batch12_q_dram_control`: change only B12 Q to DRAM at the handoff.
- `autofix_batch12_q_canonical`: retain sharded Q but move it into SDPA's exact
  first-B row-major core ordering. Minimum per-user decode PCC .9962795316 in
  both controls, with original BFP4/LoFi projection and default HiFi2 decode SDPA.

Keep the generic `_sdpa_query` handoff in base, CacheSDPA and ShardedHeadNorm
candidate call sites; derive its expected grid from the same coordinate passed
to SDPA's program config, not the device width. Compatible shards are returned
unchanged; grouped-user interleaved Q remains interleaved. This fixes the address
contract without changing precision or SDPA arithmetic configuration.

## Additional completed verification

- `autofix_batch_residual_all`: 64/64, every B1–32 with DRAM/sharded BF16+FP32
  public operands; exact CPU `(a.float()+b).bfloat16()` oracle, repeated eager,
  four trace replays, unchanged operands. Instrumented sum inputs prove physical
  `[1,32,4096]` and 32-core width shards for every B.
- `autofix_batch_kda32`: original KDA B32 real contract passes prefill .999025,
  decode .998989.
- `autofix_batch_base_contract`: original default `batched or traced_decode`
  selection, all 12 pass without maxfail, after grouped-RoPE resource repair.
- `autofix_batch_b1_pair`: combined original-policy B1 median warmed decode
  .536405812 ms linear and .427920906 ms full, versus prior .536840/.427923 ms;
  no measured regression. This run predates the final Q-handoff helper; final
  paired verification follows because that helper also checks the common path.

## Final verification and status

- `autofix_batch_all_full_exact`:32/32 full-attention batches, per-user HF
  prefill/decode PCC plus exact eager, four restored traces, post32-replay
  stress output/state and immutable inputs.
- `autofix_batch_query_geometry`:32/32 canonical query geometry checks.
- `autofix_batch_watcher_final`:8/9 targeted cases pass at Watcher10, including
  linear B12 with borrowed sharded input. B31 full fails allocating its
  380928-byte passthrough tail on the grouped RoPE core.
- `autofix_batch_prime_tail_dram`: construct passthrough directly in DRAM and
  interleave/free the rotated partial first. B31 borrowed-sharded-input
  Watcher10 HF/exact trace passes; native rotary arithmetic is unchanged.
- `autofix_batch_combined_final`:12/12 original combined batch/traced cases.
- `autofix_batch_b1_pair_final`:2/2 paired cases; linear0.536725875ms and
  full0.427756782ms, with all HF/stress and exact replay gates passing.

All commands, source snapshots and exit codes are in matching provenance.
The independent verifier explicitly released the hardware lane after every
command exited and devices closed. Fixed: compact residual allocation,
batch-dependent persistent/transient placement, grouped rotary lifetimes and
SDPA query coordinates. No physical capability reduction, precision fallback,
assertion relaxation, hardware hang or reset. Final stage-default integration
and full native-context/profile/review evidence remain with the coordinator.
