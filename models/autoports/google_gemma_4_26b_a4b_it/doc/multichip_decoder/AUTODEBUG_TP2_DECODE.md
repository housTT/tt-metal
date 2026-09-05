# AutoDebug: TP2 sliding decode wait

## Scope and evidence

This is a source/host investigation of the P150x2 proxy on a four-P300C QB2.
The hardware test opens the complete `(2, 2)` parent with `FABRIC_2D`, then
carves a `(1, 2)` compute submesh.  This investigator did not open a device.

Current live markers establish:

1. `MultichipDecoder.from_state_dict` returns (`decoder_ready`).
2. `prefill_forward` returns to Python (`prefill_ready`).
3. Decode enters and exits the attention output all-reduce.
4. Decode enters dense MLP and then enters, but does not exit, the dense-down
   all-reduce.  Router/MoE are not reached.

A standalone TP2 BF16 `ttnn.all_reduce` with `Topology.Linear` and
`num_links=1` passes.  That proves the selected submesh/control plane can run
one collective, but it does not prove repeated decoder collectives are safe.
The reported `[1, 1, 32, 4352]` probe is not the decoder's row-parallel
reduction shape: attention O, dense down, and expert down all produce hidden
width 2816.  The useful follow-up must use `[1, 1, 32, 2816]` and repeat the
collective at least three times.

## Call graph and first observed wait

The raw-weight constructor selects `residual_shard_cores=0` and disables all
four graph folds.  The effective decode dispatch is:

```text
MultichipDecoder.decode_forward              phase wrapper
  OptimizedDecoder.decode_forward            selects R0 path
    FunctionalDecoder.decode_forward
      MultichipDecoder._attention_decode
        QKV -> split/norm/rope -> cache update -> paged SDPA -> O
        MultichipDecoder._all_reduce_hidden   completes
      MultichipDecoder._dense_mlp
        OptimizedDecoder._dense_mlp
          packed gate/up -> GeGLU -> local down
          MultichipDecoder._all_reduce_hidden waits
      OptimizedDecoder._router_weights        not reached
      MultichipDecoder._moe_decode             not reached
```

For TP2 sliding attention, persistent collectives default off.  Both decode
reductions therefore call ordinary `ttnn.all_reduce` using BF16,
`cluster_axis=1`, `Topology.Linear`, `num_links=1`, and DRAM output.  Because
the first returns and the next waits, while a standalone call passes, the
leading hypothesis is a back-to-back non-persistent CCL completion/resource
reuse hazard on the submesh rather than an invalid topology or a generic
all-reduce failure.

`prefill_ready` alone is not device-completion evidence under asynchronous
dispatch.  A synchronized prefill boundary remains useful to rule out a
deferred prefill fault, but the finer decode markers already identify the
dense-down reduction as the first observed missing exit.

## Static contract audit

### Paged cache and attention: refuted for this wait

- TP2 sliding QKV is split into 8 local Q heads and 4 local KV heads.
- The test allocates `[blocks=4, heads=4, block=64, dim=256]` caches.
- Sliding cache update and SDPA intentionally need no full-attention geometry
  override.
- Decode completes cache update, SDPA, O projection, and the following
  reduction before the wait.

Thus a cache-head or paged-view mismatch cannot explain the later dense CCL
wait.  Full-attention TP2 still needs separate hardware coverage, but is not
the failing layer in this evidence.

### Decode DRAM configs: refuted for TP2 defaults

`default_dram_roles` is empty unless `tp_size == 4`.  With no explicit env
override, TP2 builds no decode-only DRAM-sharded weights/configs and cannot be
waiting in those programs.  Dense decode consumes the normal interleaved
packed gate/up tensor and normal local down tensor.

### Dense matmul geometry: legal and completes before the wait

TP2 uses a padded/local dense width of 2112/1056.  The relevant tile counts are:

- residual K: `2816 / 32 = 88`;
- packed gate/up N: `2112 / 32 = 66`;
- each gate/up half: `1056 / 32 = 33`;
- local down K: `1056 / 32 = 33`;
- down output N: `2816 / 32 = 88`.

All are tile integral, there is no TP2 DRAM program config, and instrumentation
places the wait after local down production at its reduction boundary.

### Expert decode geometry: second deterministic TP2 bug

The profile gives TP1/TP2/TP4 local expert widths of 704/352/192, or 22/11/6
tiles.  Multichip setup previously defaulted `expert_gate_per_core_n=2` for all
profiles.  `_optimized_sparse_decode_config` rejects TP2 because 2 does not
divide 11.  TP1 and TP4 are legal.

This does not cause the current earlier wait because MoE is not reached, but it
would fail immediately after CCL is repaired.  Prefill can still pass because
its different `_optimized_sparse_prefill_config` searches upward for a legal
divisor; decode uses the requested value strictly.

Host verification:

```bash
python_env/bin/python - <<'PY'
from types import SimpleNamespace
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import _profile_for_tp
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import _optimized_sparse_decode_config

class FakeDevice:
    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=11, y=10)

for tp in (1, 2, 4):
    profile = _profile_for_tp(tp)
    try:
        config = _optimized_sparse_decode_config(
            FakeDevice(), n=profile.local_moe_intermediate_size,
            per_core_n=2, in0_block_w=44, out_subblock_w=None,
        )
    except ValueError as error:
        print(tp, error)
    else:
        print(tp, config.per_core_N)
PY
```

Observed: TP1 passes with 2, TP2 raises
`per_core_n=2 must divide n_tiles=11`, and TP4 passes with 2.  TP2 passes when
`per_core_n=1`.

The implemented minimal repair derives the default from the profile's expert
tile count: use 2 when even and 1 when odd.  Explicit caller/env overrides are
still honored and validated by the inherited builder.

## Focused experiments and predictions

### 1. Persistent rotating CCL resources (leading hypothesis)

```bash
GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE=1 \
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python_env/bin/python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k 'p150x2_proxy_matches_optimized_single_chip and sliding_attention' \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/optimized_pcc_tp2_sliding_persistent.xml
```

Prediction: rotating buffers/semaphores allow the attention, dense, and expert
reductions to complete.  If the run then raises a sparse expert configuration
error without the profile fix, that independently confirms the second bug.  A
completed passing-PCC run after both changes verifies the repair combination.

### 2. Serialize the ordinary collective (mechanism control)

In the TP2 test only, wrap `_all_reduce_hidden` after construction so it calls
the original method and then `ttnn.synchronize_device(decoder.mesh_device)`.
Do not add this synchronization to production unless the experiment proves it
is required and its cost is accepted.

Prediction: if explicit completion makes all reductions finish, the problem is
ordinary CCL reuse/ordering rather than tensor shape.  If it still waits on the
second reduction, rotating persistent resources are the stronger repair.

### 3. Repeated exact-payload smoke

Extend the passing smoke to issue three consecutive BF16 Linear/one-link
all-reduces of `[1, 1, 32, 2816]`, first synchronizing only after the sequence,
then synchronizing after each call.

Prediction: sequence-only failure plus per-call-sync success reproduces the
ordering hazard without model math.  Both variants passing would imply the
dense producer's memory configuration or lifetime is part of the bug; record
the actual `partial.memory_config()`, dtype, logical shape, and padded shape at
the live boundary and reproduce those exactly.

### 4. Synchronized prefill boundary

Add a test-only `ttnn.synchronize_device(proxy_mesh)` immediately after
`prefill_forward`.

Prediction: it completes.  A wait here would revise the diagnosis to a deferred
prefill failure, because the existing `prefill_ready` print only proves host
dispatch returned.

## Repair decision

- Keep the profile-aware expert gate `per_core_n` fix: the host contract is
  deterministically verified and it prevents the next TP2 decode failure.
- Do not change cache geometry, DRAM roles, dense tensor dimensions, topology,
  or `num_links`; source evidence and live markers refute them as the current
  first wait.
- Accept a persistent-CCL default change only after the unchanged TP2 decode
  completes and clears PCC with the env-gated A/B above.  Full TP2 should then
  be run separately because its persistent default and cache geometry differ.

## Remaining uncertainty

There is no watcher/tt-triage device stack for this latest run, and this
investigator did not reproduce on hardware.  The exact low-level reason that
the second ordinary reduction waits is therefore not proven.  The current
evidence localizes the boundary and distinguishes it from the independently
verified expert geometry bug; it does not yet establish whether rotating
resources, explicit synchronization, or a producer memory/lifetime difference
is the final CCL repair.

## Hardware verification and accepted repair

The env-gated persistent-resource experiment completed the previously waiting
sliding layer, and the same policy then passed both TP2 representative layers.
The implementation now defaults TP2 and TP4 decode to three rotating explicit
buffers/semaphores and retains Linear/one-link topology for TP2. It also keeps
the profile-derived `per_core_n=1` correction for TP2's 11-tile local expert
width.

Final real-weight evidence in `artifacts/pcc_tp2_final.xml`:

| Layer | Prefill PCC | Decode PCC |
| --- | ---: | ---: |
| sliding attention, layer 0 | 0.9977247196 | 0.9952320721 |
| full attention, layer 5 | 0.9990006995 | 0.9998525194 |

The earlier uncertainty is therefore closed at the software contract level:
rotating resources are necessary for this path and pass the unchanged
end-to-end layer gate. The underlying firmware-level reason ordinary
back-to-back CCL state is not safely reusable was not independently proven and
does not justify changing the selected topology.
