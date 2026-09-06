# Autofix closure

The initial stage review and fresh `AUTODEBUG.md` report were handled through
three independent `$autofix` investigations. No speculative subagent edit was
accepted directly; each recommendation was implemented and measured on the
hardware path.

| Investigation | Finding | Action and evidence |
| --- | --- | --- |
| sparse routing | fixed `nnz=8` is unsafe after BF16 score conversion | removed the hint; measured runtime scan-all; selected exact TopK indexed sparse gate/up and down after scan-all regressed decode about 17%; B1/B32/trace gates pass |
| TP2 geometry | packed expert width 704 has a legal `per_core_N=2`, 1x2 subblock candidate; QKV DRAM sharding needed a real retry | both geometry variants and QKV block-width 11/1 ran; each decode candidate lost to the final default and QKV cost another 550 MiB/device |
| evidence and precision | CCL dtype must be resolved before persistent buffer allocation; activation roles and profiler/watcher provenance were incomplete | added role controls and matching CCL buffers; ran isolated/combined TP2/TP4 precision; regenerated six profiler captures/reports; attempted active-ETH watcher and retained its physical-size failure plus 12-case clean worker/idle-ETH run |
| final watcher B32 regression | TP4 full scored a deterministic 0.994989 against the prior stage's BFP8 dense oracle | refuted watcher drift and nondeterminism; a prompt-derived single-chip oracle with the same BF16 dense policy passes the unchanged LoFi TP4 path at 0.999833, so the temporary HiFi2 expert candidate was rejected and watcher was rerun with matched references |
| constructor capacity peak | packed-both setup briefly allocated separate gate/up device tensors before their shared replacement | upload one host-packed tensor directly; fresh final-source 50,624-token P150 probes pass both layer kinds |
| final TP4 QKV advice | TP4 sliding QKV DRAM sharding showed a possible 1.49% decode win | two alternating controls confirmed a 1.50-1.65% win, so QKV is selected only for TP4 sliding; full-attention allocation is excluded and the 288,358,400-byte 25-layer copy is charged to capacity |

The resulting default remains BF16 for activations and CCL, keeps persistent
buffers, keeps the existing residual contract, and uses runtime-indexed active
top-8 MoE. All material recommendations were tried; rejected candidates have
completed before/after artifacts under `candidates/`.

The final accepted implementation is SHA-256
`9c6735c56ff48309215845e3f73aa640c54fdb7e5ca0a5ff3b36155b409473ba`
with multichip test SHA-256
`54a87ce28b179efc740bd5ea1078340348a2fd7dee92f01379e3c814916f8071`.
All six canonical timing/PCC and Tracy captures, the capacity boundary, and
the watcher provenance were regenerated from this frozen pair.

The final precision accounting was also corrected after isolation showed that
the shared full-attention BF16 dense override covers gate, up, and down. The
five-layer gate/up/down deltas are included in `capacity_projection.json`; every
profile remains within its conservative capacity limit.
