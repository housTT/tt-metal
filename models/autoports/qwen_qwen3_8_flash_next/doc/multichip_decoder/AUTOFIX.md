# AutoFix: repository-local mesh runtime

Date: 2026-08-27

## Failure

The first 1x2 mesh smoke used an unsourced Python environment.  It imported
TTNN from `/home/ttuser/.local/lib/model-bringup/tt-metal` while the JIT used
this checkout's dispatch sources.  Compilation consequently failed on
revision-skewed dispatch contracts before model code ran.

## Hypothesis and prediction

The failure was runtime provenance skew, not a decoder or hardware failure.
Sourcing the repo-local `ttenv.sh` should make Python and native TTNN resolve
only beneath this checkout; with the P300 descriptor and devices 0,1 selected,
the 1x2 mesh and `FABRIC_1D` should then initialize.

## Verification

The no-hardware provenance probe reported:

```text
python_ttnn_origin /home/ttuser/dev/qwen3.8-flash-next/tt-metal/ttnn/ttnn/__init__.py
stale_native_mappings 0
runtime_provenance_clean
```

The corrected bounded hardware probe used:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/dev/qwen3.8-flash-next/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
timeout 60 python <1x2 FABRIC_1D mesh probe>
```

It opened both Blackhole dies and reported:

```text
mesh_shape (1, 2)
num_devices 2
device_ids [1, 0]
compute_grid 11-10
dram_grid 8-1
dram_channels 8
dram_bytes_per_channel_from_bh_mem_map 4278190080
nominal_dram_per_device_bytes 34225520640
nominal_dram_per_device_gib 31.875
mesh_smoke_pass
```

Device and fabric shutdown were clean.  The earlier intermediate probe also
opened the fabric successfully but used the obsolete `get_devices()` Python
accessor; replacing it with `get_num_devices()`/`get_device_ids()` fixed the
probe without an implementation change.

## Fix kept

All multichip commands source the repo-local `ttenv.sh`, then override its
single-chip defaults with devices `0,1` and the P300 mesh descriptor.  No
global environment, installed TTNN package, shared cache, or hardware state was
modified.

## Decoder repair loop

The first TP2 implementation also split the 48-head GDN recurrence into two
24-head kernels.  Prefill passed, but final decode PCC was about 0.992.  State
copy, q/z inputs, rank patching, output-projection precision, and collective
precision were isolated in turn.  FP32 row partials and FP32 all-reduce did not
repair the error; the local recurrence core itself was already about 0.9969
because the target kernel geometry changed at 24 heads.

The kept fix replicates GDN with the exact optimized-decoder shape and program
configuration.  QSA heads and both routed/shared expert intermediate widths
remain TP2.  Final real-weight decode PCC is 0.99999958 for layer 0 and
0.99999976 for PLE layer 1.  A candidate L1 all-reduce result was also
refuted: it improved an isolated timing probe but produced rank-divergent layer
0 output under the combined trace/correctness test.  The accepted reduction
returns BF16 in DRAM and has bit-identical ranks.

QSA uses the optimized stage's validated BFP8 cache candidate.  Against the
BF16 optimized control, prefill K/V/index cache PCC is
0.99997681/0.99997419/0.99997586 and decode cache PCC is
0.99997675/0.99997461/0.99997586.  It saves 1.7578125 GiB/die at maximum
context compared with the BF16 cache control.

## Full-stack capacity autofix

The working layer graph exposed a separate physical blocker.  Across 48
layers, TP2 expert weights contain 58,982,400 tiles/die.  Standard BFP4 needs
33,973,862,400 bytes (31.640625 GiB) per die.  After the maximum-context cache
(2,340,421,632 bytes), conservative non-expert weights including replicated
GDN (3,921,895,424 bytes), and a 1 GiB runtime/trace reserve, experts may use at
most 26,889,461,760 bytes.  Even optimistic mixed packing permits at most
53.081868% BFP4 tiles; at least 46.918132% must be BFP2/zero.  Bank padding in
the available compressed expert kernel raises the physical routed-expert count
to 66,846,720 tiles/die: uniform BFP2 is then 21,390,950,400 bytes and the
usable BFP4 fraction is at most 32.131060%, so at least 67.868940% must be
BFP2/zero.

The following isolated candidates were tested and rejected:

| Candidate | Result | Verdict |
| --- | ---: | --- |
| uniform/heavy BFP2 routed experts | about 0.73 PCC | does not preserve decoder output |
| real layer-0 selected top-10 down projection in BFP2 | 0.98258221 final PCC | below gate |
| low-error gate/up BFP2 mix, down kept higher precision, 60% overall BFP4 | 0.98320884 final PCC | below gate |
| BFP2 plus second residual quantizer | about 0.72--0.77 PCC | refuted |
| BFP2 plus rank-96 BFP4 low-rank residual | 0.96831983 final PCC | refuted |

Raw stdout, timestamps, command-execution provenance, selected expert IDs, and
error metrics are retained in `capacity_candidate_probes.log`.

Expert parallelism, layer pipeline placement, and TP2 all divide the same
global expert bytes across two dies and therefore do not solve capacity.
Host weight streaming violates the resident full-stack and runtime-fallback
contract.  The repository's `CompressedTensor`/DeepSeek expert path supports
variable-size mixed precision for a different geometry, but this model's local
640 packed gate/up width pads to 768 across eight banks, and no compatible
validated prefill plus decode sparse kernel replaces the current Qwen path.

This repair loop therefore ends with a hard **autofix-failed** verdict for
full-stack residency on the available two-die hardware.  The per-layer decoder
is real and correct, but it cannot truthfully be signed off as the resident
48-layer baseline until a more accurate sub-BFP4 expert representation and a
compatible active-expert prefill/decode kernel exist.

## Watcher/fabric recovery

Watcher with Ethernet-core instrumentation made Blackhole fabric ERISC
`29-25` fail to return to base firmware during process teardown, after the
stacked decoder workload and both-device watcher polling had passed.  The
board was reset with `tt-smi -r 0 1`; DRAM/firmware health and a bounded 1x2
fabric open/close then passed.  The supported
`TT_METAL_WATCHER_DISABLE_ETH=1` mode is used for the retained watcher run so
Tensix, dispatch, NoC, CB, assert, stack, and waypoint checks remain enabled
without instrumenting fabric ERISC.  Fabric itself is covered by the separate
correctness, trace, and profiler runs.  Both failed teardown artifacts are
retained rather than hidden.
