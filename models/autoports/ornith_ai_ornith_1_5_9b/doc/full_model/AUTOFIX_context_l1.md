# AutoFix: full-stack large-prefill L1 working set

Starting report: [AUTODEBUG_context_l1.md](AUTODEBUG_context_l1.md). The native
context command allocated all weights, caches, and traces, then failed at
GDN's FP32 output projection. Context capacity and the previous precision,
state layout, and decode policy are preserved.

Experiment prepared in `probe_context_l1.py`: capture the real layer-0 gated
FP32 [1,2048,1024] boundary, retain its original-program output, reserve exactly
221952 L1 bytes/bank, reproduce the 1422336 static end versus 1318144 allocation
frontier collision, and compare role-specific N6/K16 against N12/K8. Both retain
the original 11×10 grid, M7, FP32 inputs, BFP4 weights, LoFi computation and BF16
outputs. Each rank is checked against the original output and a CPU oracle
using device-quantized weights. A complete reduced-stack 2048-token prefill
under that full-stack resident footprint follows the selected candidate.

The optional `DecoderConfig.large_prefill_role_configs` defaults empty, keeping
all standalone optimized programs unchanged. The full-model constructor now sets only GDN out's N cap to 6 through
`MeshConfig.local`; standalone defaults and every other role remain unchanged.

## Hardware evidence

`logs/context_l1_probe_v1` reproduced the exact original collision, then verified
N6/K16 bit-for-bit on all four ranks. The initial comparison stopped when K8
exceeded the probe's predeclared 1% baseline relative-L2 screening bound (not a pre-existing full-model gate). `context_l1_probe_v2`
retains that rejection criterion, records all rejected-candidate ranks and
latency, and selects only among accepted candidates. No probe threshold was relaxed. This screen does not prove K8 fails the actual full-model top-k contract; its full-model numerical behavior remains unvalidated.
Both immutable provenance/source/log archives are preserved.

| Candidate | Warm projection ms | Exact original result on all ranks | Accepted |
| --- | ---: | --- | --- |
| N6/K16 | 0.134915 | Yes | Yes |
| N12/K8 | 0.115746 | No; rank-0 relative L2 1.1345% | Faster; outside probe screen, full-model unvalidated |

N6/K16 quantized-weight CPU-oracle PCC is at least 0.99987477. Input remains
FP32 [1,2048,1024], weight BFP4 [1024,4096], output BF16. Actual L1 resident
allocation is exactly 221952 bytes/bank. The original program's host allocation
check reproduces static end 1422336 against frontier 1318144. The selected
program's derived static end 1225728 fits with 92416 bytes of margin.
`context_l1_probe_v2.json` also passes an entire 2048-token real reduced layer
0/3 prefill under the same full-stack L1 residency. All devices closed normally;
no hardware reset or recovery was necessary.

Exact commands, environment, source hashes and exit status are in
`logs/context_l1_probe_v2.provenance.json`. Python compilation, probe Black
formatting, and `git diff --check` pass. This change is Python-only and needs no
C++ build. The parent owns the all-32-layer native context, readiness, and final
performance gates; the reduced experiment does not claim those completed.

N6 was chosen to preserve the prior optimized K16 accumulation exactly. K8 is
a legal, faster projection candidate, not a physical or proven full-model
accuracy rejection; validating a changed accumulation policy is separate work.

Final status: isolated L1 allocation bug verified and repaired with an exact
numerical geometry change. Hardware returned to the parent for full-stack
validation.
