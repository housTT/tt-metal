# AUTODEBUG — full-stack prefill L1 geometry

## Diagnosis

The native-context run fails because the single-decoder large-prefill program
was reused with all 24 recurrent states resident in L1. The first 2048-token
GDN output projection requests **1,310,720 bytes of static CB payload per
core**, ending at **1,422,336**, but the full-stack persistent allocation starts
at **1,318,144**. The overlap is exactly **104,192 bytes**. This is a
host-detected allocation contract violation, not a DRAM context-capacity limit,
firmware assert, device hang, or precision failure.

The smallest targeted candidate is to keep the tuned 11×10 grid, K-block 16,
out-block M=7, FP32 GDN activation, BFP4/LoFi weights, and BF16 result, while
splitting the per-core N=12 output into two N=6 blocks. It predicts a static end
of **1,225,728**, below the observed allocation frontier by **92,416 bytes**.
Compare it with K-block 8 / N-block 12 using real GDN input and the full
**221,952 bytes/bank** resident footprint; select the fastest validated legal
geometry. No change to context, TP4 strategy, residual layout, cache dtype,
recurrent-state placement, or decode program is required.

This isolated investigation on 2026-09-05 was source-only. No implementation
edits, hardware access, experiment, build, or performance measurement was done
by this investigator. The parent owns the repair experiment.

## Direct evidence

- `logs/native_context_v1.log` names program 444 and core range `[0-0 - 10-9]`,
  with static CB end 1422336 and L1 allocation frontier 1318144.
- The Python stack identifies `_gdn_prefill` → `_gdn_out_head_major` →
  `_linear(gated, "gdn_out")` → `_prefill_linear` → `ttnn.linear` in the first
  public full-stack prefill. The host then closes devices normally; there is no
  live hang to capture or reason for a speculative reset.
- `native_context_v1.json` records native context 262144, TP4, batch 1, and zero
  completed long-prompt windows. All model weights, native caches, and traces
  allocated successfully before prefill failed.
- DRAM after trace/sampler construction is 5,047,560,704 allocated bytes per
  device, against 8 × 4,259,840,384 = 34,078,723,072 available bytes. The largest
  contiguous free allocation per bank is 3,622,562,176 bytes. Reducing native
  context is unsupported by this evidence.
- Persistent L1 is 221952 bytes/bank: 25344 weights/constants plus 196608 for 24
  recurrent states. Each TP-local batch-1 state has 128 FP32 tiles; interleaving
  over 110 banks rounds each allocation to two 4096-byte tiles/bank. Therefore
  24 × 8192 = 196608. Decode trace capture already succeeded with this state.
- The 300 KiB/bank state policy in `OptimizedDecoder.allocate_state()` tests
  one decoder's allocation. It does not budget the combined full-stack states
  against every large-prefill matmul working set.

## Source contract and exact resource equation

Model `tt/` paths below are relative to
`models/autoports/ornith_ai_ornith_1_5_9b/`; native `ttnn/` paths are repository-relative.

- `tt/optimized_decoder.py:580` requests FP32 from the fused gated RMSNorm.
  Multiplication by Z preserves that FP32 GDN projection input. `_linear()` at
  line 255 explicitly requests BF16 output for `gdn_out`. Casting the activation
  to BF16 would change the selected policy and is not the proposed fix.
- `_prefill_linear()` at line 673 uses the large-prefill grid `(11,10)` and
  K block 16 for sequences at least 2048. For 2048 token rows it computes
  `per_core_M=ceil(64/10)=7`, `per_core_N=ceil(128/11)=12`, and chooses output
  blocks M=7, N=12. GDN local K is 1024 elements = 32 tiles.
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:152`
  allocates input CBs with buffering depth two, as declared in
  `device/utilities/matmul_utilities.hpp:23`. Its output and intermediate CBs
  alias because both are BF16 and the output is interleaved DRAM; see factory
  lines 1031–1090. FP32 destination accumulation is false for this projection;
  do not add an invented separate FP32 intermediate buffer to the calculation.

For block M=`m`, N=`n`, K=`k`, the relevant payload is:

```text
CB_input_0 = 2 * m * k * 4096  # FP32 activation tiles
CB_input_1 = 2 * n * k * 576   # BFP4_B weight tiles, DRAM-aligned
CB_output_and_intermediate = m * n * 2048  # aliased BF16
static_end = 111616 + sum(CBs)
```

The fixed 111616-byte base is confirmed by subtracting the factory-derived
payload from the observed static end. It must be checked again if the device
configuration or firmware reservation changes.

| Geometry (m,n,k) | Input 0 | Input 1 | Output/intermediate | Static end | Margin below 1318144 | Change in block loops |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 7,12,16 — failing | 917504 | 221184 | 172032 | 1422336 | -104192 | Original |
| 7,6,16 — first candidate | 917504 | 110592 | 86016 | 1225728 | 92416 | Two N blocks instead of one |
| 7,12,8 — comparison | 458752 | 110592 | 172032 | 852992 | 465152 | Four K blocks instead of two |
| 1,12,16 — deprioritized | 131072 | 221184 | 24576 | 488448 | 829696 | Seven M blocks instead of one |

N=6 divides per-core N=12 and permits the existing selected subblock width 6;
M=7 stays unchanged. K=16 remains a divisor of 32, keeping the original K-block
accumulation grouping. M=7 is prime, so the next smaller legal output-M block
under this planner is 1. K=12 does not divide 32 and must be pruned before any
hardware experiment. K=32 increases the dominant FP32 input working set.

For comparison, the same m=7,n=12,k=16 program with BF16 input ends at 963584,
which fits the observed frontier. Thus a global K-block reduction on every
large-prefill projection is unnecessarily broad; the exceptional FP32
`gdn_out` input explains both the exact failure and the earlier successful
projections. The original small-prompt AIME gates do not select the ≥2048
large-prefill program, so their success cannot validate this configuration.

## Minimal repair boundary

Preserve the standalone optimized decoder's existing defaults. Add an optional
role-specific large-prefill output-N cap (or equivalent explicit per-role
program override) to `DecoderConfig`, absent by default, and honor it in
`_prefill_linear()` when choosing a divisor of `per_core_N`. Set only the
full-model `gdn_out` cap to 6 through a `MeshConfig.local` replacement when
constructing its layers. `MultichipDecoder.from_state_dict()` already forwards
`MeshConfig.local` to `OptimizedDecoder` as `config`; reuse that configuration
boundary instead of runtime monkey-patching or adding a per-token policy check.

To isolate the comparison, expose K=8 only for that same full-model prefill
role, then benchmark both candidates. Keep the same original compute kernel
config and precision policy. Choose the measured winner and record the
rejected legal alternative. Do not claim either candidate is fastest from
resource arithmetic alone. No native kernel/C++ change is necessary for this
configuration repair; Python formatting and meaningful hardware checks are the
appropriate validation scope.

## Verification experiment

1. Use real GDN-prefill input from the optimized layer, or a captured equivalent
   FP32 boundary tensor. Reserve/retain the actual full-stack **221952
   bytes/bank** L1 footprint, including constants. The older 114688-byte head
   probe reservation does not represent this run. Prefer retaining all actual
   states when measuring the full-stack reproduction.
2. Reproduce the 2048-token gdn_out failure with m7/n12/k16. Then run the two
   candidates with the same tensor and compare outputs to the original program
   executed in a legal isolated memory setup. Require prior decoder numerical
   thresholds, exact logical shapes, unchanged dtype/layout, and no allocation
   collision. Measure warmed program latency; no sampler or host output work
   should contaminate that projection comparison.
3. Run one full-stack 2048-token prefill with all 32 real layers and all 24 L1
   recurrent states, followed by traced decode. This detects later operators
   whose working sets might also require an explicit full-stack budget.
4. Rerun the public native-context command below, preserving both requested
   non-aligned and aligned windows and the final valid decode position. Leave
   context 262144 advertised only once actual execution is validated; the
   existing allocation-only JSON is incomplete evidence.

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \
timeout 1200 python_env/bin/python -m \
models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.context_capacity \
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/native_context_v2.json
```

The parent must serialize hardware, wrap commands in the existing provenance
recorder, and retain the failure artifacts. Recheck full-model top-k and trace
contracts after selecting the final geometry; short-prompt gates supplement
rather than replace the long-prefill experiment.

## Uncertainty and rejected interventions

The exact CB equation identifies this collision confidently. It does not prove
that every later full-stack operator fits, that either candidate preserves
long-context accuracy, or that the selected candidate meets the prior latency
expectation. Those remain device gates. A failure at a later operator is new
work, not grounds for reducing native context without physical DRAM evidence.

Rejected first interventions: shrinking public/internal context or chunk size,
casting GDN activations to BF16, moving all recurrent states to DRAM, changing
the TP4/cache/CCL policy, disabling trace or allocator validation, and weakening
watcher. These either miss the exact resource contract or violate the stage's
preservation requirements. Keep inactive-slot and cache-ownership semantics
unchanged while testing the targeted prefill geometry.
