# AutoFix: LM-head geometry closure

Select **64 input cores (8x8), K1, two DRAM readers**, with input shard
`[32,64]`, `per_core_M=1`, and output-storage `per_core_N=16`. Preserve the
existing two32768-column weights per rank, BF16/HiFi4, FP32 destination and
packer L1 accumulation. Keep final normalization and padding at **8x4**,
`[32,128]`, then explicitly reshard the normalized tensor to8x8 before the
common `LMHead1D`. No precision, decoder, residual, cache, or CCL change.

The paired real-hidden terminal median improves **1.371676 → 1.137086 ms**
(17.10%); normalized-input head median improves **1.358203 → 1.124876 ms**
(17.18%). All248320 logical logits are bit-identical to the selected baseline,
including eager repeats and trace replay. These are terminal/head measurements
on four Blackhole chips on physical P300c boards, not full-model speedup claims.

## Starting evidence and experiment

`AUTODEBUG_head_geometry.md` was written before the probe. Native factory
arithmetic showed input-core changes do not shrink static weight CBs but can
change dynamic allocations and DRAM-reader placement. The previous report only
tested32 input cores, so the64-core family required an actual experiment.

`probe_head_geometry.py` uses the existing common `LMHead1D` and materialized
selected weights, with `full_model/french_head_v1/hidden.pt` dictionary
`hidden[0]`. All four saved replicas are asserted equal. Both sides use the
same8x4 norm; geometry changes occur after normalization. Every device run
reserves exactly **221952 L1 bytes/bank**, covering25344 constants and196608
for24 recurrent states. It tests normalized-input head plus full terminal,
saves complete logical logits, and times three alternating trials of64 replays
per geometry. Each replay batch is checked against its saved eager logits.

## Results

All rows use two logical32768-column chunks. Three readers pad each weight chunk
to33024 physical columns, with129 tiles/bank =3x43 reader tiles; each chunk is
trimmed independently before restoring local vocabulary65536.

| Input cores / K / readers | Head baseline → candidate ms | Terminal baseline → candidate ms | Logical logits versus baseline | Verdict |
| --- | --- | --- | --- | --- |
| 16 /1 /2 | 1.358086 →1.782081 | 1.371790 →1.794770 | Bit-identical | Slower |
| **64 /1 /2** | **1.358203 →1.124876** | **1.371676 →1.137086** | **Bit-identical** | **Selected** |
| 64 /2 /2 | — | — | Allocation rejected before candidate execution | K2 fit hypothesis refuted |
| 32 /1 /3 | 1.358411 →1.343536 | 1.371723 →1.355583 | Bit-identical | Smaller gain than selected |
| 16 /2 /3 | 1.358509 →1.579915 | 1.371867 →1.592960 | Changed; top1 equal | Slower |
| 32 /2 /3 | 1.358492 →1.345954 | 1.371828 →1.358946 | Changed; top1 equal | Smaller gain than selected |
| 64 /2 /3 | 1.358534 →1.298999 | 1.371702 →1.312336 | Changed; top1 equal | Smaller gain than selected |

The64/K2/two-reader failure confirms static CB end **1299456**, exactly the
source prediction, versus measured dynamic frontier **1268992**: collision
**30464 bytes**, even without terminal normalization temporaries. Larger input
geometry improves the frontier but does not rescue K2. The device closes
normally after this host allocation error; no reset is needed.

K2/three-reader candidates fit with predicted static end918528. Their max
logical score difference is0.125 and top1 agrees on this hidden row. None beats
the bit-identical64/K1/two-reader winner, so no numerical or French qualitative
policy exception is needed. One-reader32768, R2/K4, and R3/K4 were source-pruned
at the existing full-stack residency; precise arithmetic is in AutoDebug.

`head_geometry_summary.json` retains exact medians, checks, configurations,
commands, exit statuses, and per-run artifact/provenance paths. The immutable
`logs/head_geometry_*.provenance.json` and `.sources.json.gz` include selected
model/probe/common-head source snapshots and binary hashes.
`head_geometry_source_hashes.json` adds the inspected native factory, validation,
and utility hashes. The model hash there is the frozen baseline before parent
integration; the probe hash includes the watcher lifecycle repair and explicit
baseline configuration described below.

## Watcher lifecycle repair and final verification

`head_geometry_c64_k1_r2_watcher_v1` deliberately retained both timing traces.
The allocation tracker rejected replay of the earlier baseline after candidate
capture introduced four live buffers, including candidate reshard/matmul cache
allocations and its output. It raised before replay; both initial capture/replay
checks had passed. This is a probe lifecycle failure, not a kernel failure or
model fix. The failed log/JSON are preserved.

The focused repair adds `--serial-traces`: eager/capture/replay/release the
baseline completely, then the candidate. It uses no corruptible-allocation
scope, tracker suppression, or model change. The corrected
**`head_geometry_c64_k1_r2_watcher_v2` passes, exit0**, with:

```text
TT_METAL_WATCHER=10
TT_METAL_WATCHER_DISABLE_ETH=1
TT_METAL_TRACE_ALLOC_TRACKING=1
--cores 64 --in0-block-w 1 --readers 2 --repeats 2 --rounds 1 --serial-traces
```

Both head and terminal pass bitwise baseline comparison, eager repeat and trace
replay. Watcher memory reports221952 allocated bytes/bank,1206528 free and
1206528 largest contiguous free before terminal. The reserve starts at1318144
and uses the whole11x10 grid, shard `[1,110592]` BF16, after existing768 bytes
of constants in the reduced model. Frozen normalized-input head adds8192
bytes/bank, yielding230144 allocated and1190144 largest contiguous free. The
same memory readings hold before each serial geometry. No profiler was enabled,
no reset/recovery was needed, and all meshes closed normally.

Host checks: `python -m py_compile` for the probe, hardware-free `--plan`, and
`python_env/bin/python -m black --target-version py310` passed. This agent made
stage-doc/probe changes only; no C++ build, runtime implementation edit, commit,
or push was performed.

After parent integration changed the model's default head to64 cores, the stage
probe was made independent of those defaults: it explicitly constructs the
C32/K1/R2 baseline head and8x4 normalization config. Archived run sources remain
immutable. This final maintenance edit passed host compile, Black, and `--plan`;
it was not rerun on hardware after ownership returned to the parent.

## Handoff

Hardware was explicitly returned to the parent after corrected watcher exit0.
The parent owns integration, all32-layer and mixed-batch32 checks, full French
qualitative review, native-context validation, and final full-model performance
gates. This experiment selects a component configuration; it does not waive any
of those gates. The recorded hidden's next token remains39102 (`inform`), as in
the baseline; local exactness is not a claim about the prior full French branch.


## Review addendum: fixed C64/K1 reader control

The stage reviewer identified a missing comparison: C64/K2/R3 changed both K
and readers relative to the winner. The probe now accepts `--baseline-cores 64`
(default remains32 for prior reproduction), independently of the unchanged
8x4 normalization. The new direct comparison fixes both input core count64 and
K1, and changes only the adapted reader family: R2/N32768/storage-N16 versus
R3/N33024/storage-N17, with each physical chunk trimmed to logical32768.

`head_geometry_c64_k1_r3_vs_c64_k1_r2_v1` passes, exit0, at the same221952-byte
persistent L1 reservation. Three paired64-replay trials give:

| Fixed C64/K1 | Two readers, selected | Three readers, adapted |
| --- | ---: | ---: |
| Head median ms | 1.125082 | 1.296014 |
| Terminal median ms | 1.137138 | 1.309113 |
| Logical logits versus two readers | Baseline | Bit-identical |

Three readers are15.12% slower in the terminal comparison; all248320 logical
logits, eager repeats, and trace outputs are exact. The selected C64/K1/R2
therefore remains unchanged, and its existing corrected watcher pass remains
applicable. No additional watcher, model edit, reset, or precision change was
needed. All device jobs closed normally and hardware was promptly handed back.

The exact command and source snapshots are in the matching `logs/*.provenance.json`
and `.sources.json.gz`; `head_geometry_summary.json` now contains eight geometry
runs and explicitly labels each baseline core count. Host compile, Black check,
and hardware-free `--plan` passed after the baseline option was added.
