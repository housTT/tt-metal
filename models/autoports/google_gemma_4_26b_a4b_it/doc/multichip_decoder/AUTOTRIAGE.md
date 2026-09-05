# AUTOTRIAGE

## Final resolution

The original Ring-topology proposal below was refuted. The physical QB2 must
be opened as a 2x2 `FABRIC_2D` parent, but its 1x2 compute submesh correctly
uses `Topology.Linear` with one link. A standalone collective passed; live
markers later showed that the first decoder all-reduce also passed and the
second back-to-back reduction waited. Three rotating persistent asynchronous
all-reduce buffers/semaphores repaired that resource-reuse boundary. Both TP2
representative layers subsequently passed prefill and decode PCC >= 0.995, so
the selected implementation keeps Linear rather than the proposed Ring. See
`AUTODEBUG_TP2_DECODE.md` and `artifacts/pcc_tp2_final.xml`.

## Diagnosis

- The TP2 proxy stalls at its first collective because the decoder selected
  `Topology.Linear` while its containing QB2 control plane was initialized as
  `FABRIC_1D_RING`. The TP1 submesh, which performs no collective, passes the
  same test and the TP4 ring path passes.

## Triage Evidence

- The TP2 run completed weight loading and emitted the all-gather/all-reduce
  fabric warnings immediately before making no further progress.
- TP1 passed after its profile-specific matmul geometry was corrected, which
  separates model math and submesh creation from the TP2-only collective wait.
- `tools/tt-triage.py` could not capture device stacks because its `ttexalens`
  dependency is absent. `AGENTS.md` forbids installing dependencies on this
  runner; the failed capture command is therefore retained as evidence rather
  than mutating the environment.

## Source Evidence

- `MultichipDecoder._all_reduce_hidden` passes `self.topology` to every
  contraction collective.
- The TP profile refactor selected `Topology.Linear` for TP2, while the proxy
  test must open the complete four-device QB2 ring before carving a two-device
  compute submesh. Opening only the subset cannot initialize this board's
  physical ring because absent neighboring routers never handshake.
- The producer ledger is one rank-local row-parallel partial per active rank;
  the consumer is one all-reduce result on each active rank. Counts match. The
  first-hop topology/control-plane mismatch is the earliest differing contract.

## Downstream Effects

- The host pytest wait and eventual teardown errors are downstream of the
  incomplete collective. They are not cache, MoE routing, or decoder-shape
  failures.

## Original proposed fix (refuted)

- Use `Topology.Ring` for TP2 when running as a submesh of the QB2 ring. A
  two-rank ring preserves the same one-partial-per-rank reduction contract and
  matches the initialized control plane. Keep TP1 collective-free.

## Uncertainty

- Device call stacks are unavailable for this run because the triage dependency
  is absent. The topology repair must be accepted only if an unchanged TP2
  correctness run completes and clears PCC; otherwise capture must be repeated
  in an image that includes the triage requirements.
