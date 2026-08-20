# AUTOTRIAGE

## Diagnosis

- The adapted DRAM-interleaved fused all-gather-matmul candidate stops making
  host-visible progress after the 1x4 mesh is opened.  The live process ended
  before `tt-triage` could attach to an Inspector runtime, so the evidence does
  not support a kernel-level semaphore, CB, or NoC root-cause claim.

## Triage Evidence

- `fused_agmm_linear_attempt3_dram.log` reaches mesh creation and then emits no
  model, kernel, assertion, or teardown output for more than 25 seconds.
- The first LLM-readable capture reports that `/tmp/tt-metal/inspector` was not
  present and that the Metal runtime was unavailable to triage.  A focused
  `--dev=all` retry reports the same missing-Inspector dependency.
- PID 1958892 no longer existed by the time the second capture completed.
  Therefore there is no reliable call stack, running-op, CB, semaphore, NoC,
  Ethernet, or ARC stop-site to interpret.  Broad CCL-deadlock conclusions
  would be speculation.

## Source Evidence

- Earlier attempts prove the fused AGMM API reaches validation and kernels
  after adapting weights from rank 2 to rank 4.
- Its four-core gathered-L1 form has a concrete capacity conflict: static CB
  allocation ends at 968000 while the gathered buffer begins at 953664.
- The hanging retry removes that conflict by using a DRAM-interleaved gathered
  output and `persistent_output_buffer=None`.  Because the log stops before an
  AGMM kernel marker, the first observable stuck boundary is host-side graph or
  resource construction, not a proven device producer/consumer imbalance.

## Downstream Effects

- The pytest process does not reach correctness, timing, or normal device
  teardown.  Any retained UMD lock is a consequence of terminating the hung
  process, not evidence for the original stop-site.

## Proposed Fix

- Do not select this DRAM-interleaved AGMM layout.  Preserve the exact L1
  overlap and the two adapted DRAM retry logs as the blocker evidence.
- Keep the independently correct fused-RS coherent family measurement and the
  selected persistent replicated all-reduce path.  If the kernel gains a
  lower-L1 gathered-buffer contract, rerun the same exact-shape candidate with
  Inspector enabled from process start so a future hang leaves device stacks.

## Uncertainty

- No live Inspector data was available, so the precise host/runtime location
  of the DRAM retry hang is unresolved.  This report intentionally does not
  attribute it to a particular kernel semaphore or fabric route.

