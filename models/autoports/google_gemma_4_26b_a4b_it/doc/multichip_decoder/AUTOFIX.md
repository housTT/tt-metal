# AutoFix Report

## Starting Evidence

- Source: `AUTODEBUG.md` in this directory.
- Original failure: both cases of
  `test_multichip_matches_optimized_single_chip` raise in
  `OptimizedDecoder.__init__` because the multichip factory explicitly selects
  R0 while inherited row-major routing defaults to enabled.
- Constraint: keep `optimized_decoder.py` unchanged and do not open TT devices.

## Hypothesis Experiments

- Hypothesis: The manual multichip loader inherited newer single-chip routing
  and graph-fold defaults that are incompatible with its R0 residual and raw
  weights.
- Experiment:

  ```bash
  env -u GEMMA4_OPT_ROUTING_ROW_MAJOR python_env/bin/python -c '<inspect MultichipDecoder.from_state_dict, OptimizedDecoder signature, and _bool_from_env>'
  ```

- Result: The routing default resolved to true, the multichip source explicitly
  supplied `residual_shard_cores=0`, and all four inherited graph-fold defaults
  were true.
- Verdict: verified.
- Fix: Route multichip construction through `_construct_raw_weight_decoder`.
  It forces `folded_router_projection`, `shared_ffn_norm`,
  `folded_expert_scale`, and `fused_final_scalar` false.  When the routing env
  variable is absent, an exception-safe context temporarily supplies `0` only
  for the inherited constructor and then removes it.  Explicit caller values
  are preserved, including `1`, which intentionally reaches the inherited
  incompatibility check.
- Regression: Added a host-only test using a capture constructor to prove the
  four effective flags, the absent-variable default inside construction, caller
  environment restoration, explicit-value preservation, and production-factory
  wiring.
- Verification:

  ```bash
  python_env/bin/python -m py_compile \
    models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py
  # passed

  python_env/bin/python -m pytest -q \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    -k 'multichip_shape_contract or multichip_raw_weight_policy_restores_routing_environment or decode_only_packed_source_preserves_per_rank_gate_up_pairing or decode_only_packed_dtype_uses_independent_host_source or decode_weight_selection_uses_phase_not_ambiguous_tile_count or multichip_inherits_optimized_baseline_and_has_no_host_hot_path or multichip_preserves_active_expert_execution'
  # 7 passed, 31 deselected

  python_env/bin/pre-commit run --files \
    models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/AUTODEBUG.md
  # passed
  ```

### Phase-owned attention compute configuration

- Hypothesis: The multichip attention overrides retained the former shared
  `attention_compute_config` attribute after the optimized baseline split it
  into prefill- and decode-owned configurations.
- Evidence artifact:
  `artifacts/optimized_pcc_v2.xml` records both layer cases constructing
  successfully after the preceding fix and then failing at the first prefill
  QKV projection with `AttributeError: attention_compute_config`.
- Experiment:

  ```bash
  python_env/bin/python -c 'import ast, inspect, textwrap; from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import MultichipDecoder; from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder; methods = ("_attention_prefill", "_attention_decode"); found = {}; [(found.__setitem__(name, [node.attr for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(getattr(MultichipDecoder, name))))) if isinstance(node, ast.Attribute) and "attention_compute_config" in node.attr])) for name in methods]; base = inspect.getsource(OptimizedDecoder.__init__); print(found); print({"base_prefill_owned": "self.prefill_attention_compute_config =" in base, "base_decode_owned": "self.decode_attention_compute_config =" in base, "base_legacy_owned": "self.attention_compute_config =" in base}); assert found == {"_attention_prefill": ["attention_compute_config", "attention_compute_config"], "_attention_decode": ["attention_compute_config", "attention_compute_config"]}; assert "self.prefill_attention_compute_config =" in base and "self.decode_attention_compute_config =" in base and "self.attention_compute_config =" not in base'
  ```

- Result: The multichip prefill and decode methods each had two legacy
  references.  The current base constructor owns both phase-specific fields and
  never creates the legacy field.
- Verdict: verified.
- Fix: Changed only the four multichip call sites: prefill QKV/O now use
  `prefill_attention_compute_config`; decode QKV/O now use
  `decode_attention_compute_config`.
- Verification:

  ```bash
  python_env/bin/python -c 'import ast, inspect, textwrap; from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import MultichipDecoder; found = {}; [(found.__setitem__(name, [node.attr for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(getattr(MultichipDecoder, name))))) if isinstance(node, ast.Attribute) and "attention_compute_config" in node.attr])) for name in ("_attention_prefill", "_attention_decode")]; print(found); assert found == {"_attention_prefill": ["prefill_attention_compute_config", "prefill_attention_compute_config"], "_attention_decode": ["decode_attention_compute_config", "decode_attention_compute_config"]}'
  # passed

  python_env/bin/python -m py_compile \
    models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py
  # passed

  python_env/bin/python -m pytest -q \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    -k 'multichip_shape_contract or multichip_raw_weight_policy_restores_routing_environment or decode_only_packed_source_preserves_per_rank_gate_up_pairing or decode_only_packed_dtype_uses_independent_host_source or decode_weight_selection_uses_phase_not_ambiguous_tile_count or multichip_inherits_optimized_baseline_and_has_no_host_hot_path or multichip_preserves_active_expert_execution'
  # 7 passed, 31 deselected
  ```

### Paged-cache update/SDPA keyword split

- Hypothesis: The full-attention decode failure in
  `artifacts/optimized_pcc_v3.xml` comes from forwarding the update API's loose
  `block_size` and `num_kv_heads` keywords to paged SDPA, whose current API
  requires those values grouped in `PagedCacheGeometryOverride`.
- Source/API evidence: The v3 binding error lists
  `paged_cache_geometry`/`cache_position_modulo` but not `block_size` or
  `num_kv_heads`.  `FunctionalDecoder` and `OptimizedDecoder` use
  `_cache_view_kwargs` for `paged_update_cache` and
  `_sdpa_cache_view_kwargs` for
  `paged_scaled_dot_product_attention_decode`.  The multichip view must retain
  its TP-local full-attention count of one KV head.
- Experiment:

  ```bash
  python_env/bin/python -c 'from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FULL_KIND, SLIDING_KIND; from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import MultichipDecoder; d=object.__new__(MultichipDecoder); d.layer_kind=FULL_KIND; update=d._cache_view_kwargs(prefill=False, cache_position_modulo=257); sdpa=d._sdpa_cache_view_kwargs(cache_position_modulo=257); geometry=sdpa["paged_cache_geometry"]; print(update); print(sdpa); print(geometry.block_size, geometry.num_kv_heads); assert update == {"block_size": FULL_KIND.block_size, "num_kv_heads": 1, "cache_position_modulo": 257}; assert set(sdpa) == {"paged_cache_geometry", "cache_position_modulo"}; assert geometry.block_size == FULL_KIND.block_size and geometry.num_kv_heads == 1; d.layer_kind=SLIDING_KIND; assert d._cache_view_kwargs(prefill=False) == {} and d._sdpa_cache_view_kwargs() == {}'
  ```

- Result: The root patch produces the exact update kwargs, an SDPA geometry
  object with `(block_size=128, num_kv_heads=1)`, and no override for sliding
  attention.  `_attention_decode` sends the two forms to their correct
  consumers.
- Verdict: verified; keep the root patch.
- Regression: Added
  `test_multichip_cache_view_separates_update_and_sdpa_geometry`, including
  production-call wiring assertions.
- Verification:

  ```bash
  python_env/bin/python -m py_compile \
    models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py
  # passed

  python_env/bin/python -m pytest -q \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_multichip_cache_view_separates_update_and_sdpa_geometry
  # 1 passed

  python_env/bin/python -m pytest -q \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    -k 'multichip_shape_contract or multichip_raw_weight_policy_restores_routing_environment or multichip_cache_view_separates_update_and_sdpa_geometry or decode_only_packed_source_preserves_per_rank_gate_up_pairing or decode_only_packed_dtype_uses_independent_host_source or decode_weight_selection_uses_phase_not_ambiguous_tile_count or multichip_inherits_optimized_baseline_and_has_no_host_hot_path or multichip_preserves_active_expert_execution'
  # 8 passed, 31 deselected
  ```

### Factory-state drift through v7 and numerical experiment plan

#### Attribute failures (v4-v6)

The three successive failures are one class of integration drift: the
multichip loader constructs `OptimizedDecoder` directly and therefore does not
run the setup tail of `OptimizedDecoder.from_state_dict`.

| Artifact | First missing state | Reachable consumer | Current resolution |
| --- | --- | --- | --- |
| `artifacts/optimized_pcc_v4.xml` | `decode_dram_padded_input_widths` | inherited packed dense decode | initialize the map in the multichip factory |
| `artifacts/optimized_pcc_v5.xml` | `decode_dram_logical_output_widths` | inherited packed dense decode | initialize the map in the multichip factory |
| `artifacts/optimized_pcc_v6.xml` | `decode_packed_expert_gate_up` | polymorphic optimized `_moe_decode_single_user` | disable the unsupported packed-expert policy for raw TP weights |

A source audit found no additional reachable uninitialized factory-only state.
The base constructor creates ordinary runtime fields.  The multichip factory
creates `expert_weights`, `packed_mlp_gate_up`, all six DRAM decode maps, and
the routing base.  `decode_attention_weights_batch32` is not consumed because
multichip overrides attention; the packed expert and batch-32 expert tensor
fields are not consumed under the explicit unpacked R0 policy.  The host
regression now checks both packed-expert flags as well as the graph folds, and
checks the reachable factory state assignments.

#### v7 numerical evidence

`artifacts/optimized_pcc_v7.xml` is the first run in this sequence that reaches
the numerical assertions.  Both prefill assertions pass.  Decode is not
uniformly about 0.9949: sliding attention is `0.9892006673654702`, while full
attention is `0.9949203490627379`, against the `0.995` gate.  This layer split
is evidence for at least one sliding-specific mismatch plus a smaller shared
decode mismatch.

The strongest source match for the sliding-only loss is attention storage:
the captured `OptimizedDecoder` policy defaults sliding QKV/O to BF16 and full
QKV/O to BFP8, while v7 multichip defaults both kinds to BFP8.  A kind-aware
multichip default is therefore the first experiment.  It should materially
improve sliding decode and leave full decode nearly unchanged.

Two completed controls refine the remaining search:

- `artifacts/optimized_pcc_v8.xml` changed only the expert uploads to match the
  optimized gate/up BFP4 plus down BFP8 policy.  Decode worsened to
  `0.9885124312118698` sliding and `0.9930889096865416` full, so that expert
  storage policy is refuted and was reverted.
- `artifacts/optimized_pcc_hifi4.xml` set all four multichip fidelity controls
  to HiFi4.  Full passed, but sliding worsened to `0.9848282846715828`.
  Higher fidelity can recover the full layer, but the combined run does not
  identify which component did so and is not a general repair.

#### Ranked isolated hardware experiments

Each command below changes one factor relative to v7, except the explicitly
staged DRAM/packing comparisons which hold the preceding BFP8 control fixed.
No device command was run during this investigation.

1. Verify the kind-aware attention default after the focused source change:

   ```bash
   GEMMA4_RANGE_DOWNLOAD=1 \
   TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
   python_env/bin/python -m pytest -q \
     models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
     -k multichip_matches_optimized_single_chip \
     --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/optimized_pcc_attention_kind_default.xml
   ```

   Prediction: sliding rises substantially because it now matches baseline
   BF16 QKV/O; full is stable because both implementations remain BFP8.

2. Isolate the decode-only packed dense BFP4 copy while preserving its DRAM
   role and program:

   ```bash
   GEMMA4_MULTICHIP_DECODE_MLP_GATE_UP_WEIGHT_DTYPE=bfp8 \
   GEMMA4_RANGE_DOWNLOAD=1 \
   TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
   python_env/bin/python -m pytest -q \
     models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
     -k multichip_matches_optimized_single_chip \
     --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/optimized_pcc_decode_packed_bfp8.xml
   ```

   Prediction: prefill is bitwise/PCC-stable because the independent copy is
   decode-only; improvement in both decode cases identifies BFP4 as the shared
   loss.  If BFP8 is still marginal, repeat with `bf16` before changing a
   program or topology.

3. Starting from the BFP8 packed control, remove one DRAM-sharded role per run.
   This keeps packed precision fixed while distinguishing layout/program roles:

   ```bash
   for spec in \
     'no_o:packed_mlp_gate_up,mlp_down' \
     'no_down:o_proj,packed_mlp_gate_up' \
     'no_packed:o_proj,mlp_down'; do
     name=${spec%%:*}
     roles=${spec#*:}
     GEMMA4_MULTICHIP_DECODE_MLP_GATE_UP_WEIGHT_DTYPE=bfp8 \
     GEMMA4_MULTICHIP_DRAM_SHARDED_ROLES="$roles" \
     GEMMA4_RANGE_DOWNLOAD=1 \
     TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
     python_env/bin/python -m pytest -q \
       models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
       -k multichip_matches_optimized_single_chip \
       --junitxml="models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/optimized_pcc_dram_${name}.xml"
   done
   ```

   The full baseline does not auto-select O projection as a DRAM role, making
   `no_o` the highest-value role check for layer 5.  A change isolated to one
   omission implicates that role's DRAM-sharded program/conversion rather than
   CCL or the complete decoder.

4. Hold packed weight precision at BFP8 and split the packed dense projection
   into separate gate/up matmuls:

   ```bash
   GEMMA4_MULTICHIP_DECODE_MLP_GATE_UP_WEIGHT_DTYPE=bfp8 \
   GEMMA4_MULTICHIP_PACKED_DENSE_GATE_UP=0 \
   GEMMA4_RANGE_DOWNLOAD=1 \
   TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
   python_env/bin/python -m pytest -q \
     models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
     -k multichip_matches_optimized_single_chip \
     --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/optimized_pcc_dense_unpacked_bfp8.xml
   ```

   Improvement relative to experiment 2 implicates packing/program geometry,
   not BFP4 storage.

5. Deconfound the successful all-HiFi4 run with one fidelity at a time:

   ```bash
   for spec in \
     'attention:GEMMA4_MULTICHIP_ATTENTION_FIDELITY' \
     'mlp:GEMMA4_MULTICHIP_MLP_FIDELITY' \
     'expert_gate:GEMMA4_MULTICHIP_EXPERT_GATE_FIDELITY' \
     'expert:GEMMA4_MULTICHIP_EXPERT_FIDELITY'; do
     name=${spec%%:*}
     variable=${spec#*:}
     env "$variable=hifi4" \
       GEMMA4_RANGE_DOWNLOAD=1 \
       TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
       python_env/bin/python -m pytest -q \
         models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
         -k multichip_matches_optimized_single_chip \
         --junitxml="models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/optimized_pcc_hifi4_${name}.xml"
   done
   ```

   MLP and expert controls affect both layer kinds.  The current attention env
   changes sliding attention; full R0 decode resolves the separate inherited
   `full_attention_math_fidelity` default, so an unchanged full result in the
   attention-only run is expected.  Once a component is identified, retry it
   at HiFi2 to select the smallest fidelity increase.

6. CCL is already BF16 in v7, the highest precision exposed by the multichip
   control.  BFP8 is therefore a negative sensitivity control, not a proposed
   fix.  Separately, disabling the full layer's default persistent async path
   distinguishes CCL algorithm/buffer behavior without changing dtype:

   ```bash
   GEMMA4_MULTICHIP_CCL_DTYPE=bfp8 GEMMA4_RANGE_DOWNLOAD=1 \
   TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
   python_env/bin/python -m pytest -q \
     models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
     -k multichip_matches_optimized_single_chip \
     --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/optimized_pcc_ccl_bfp8.xml

   GEMMA4_MULTICHIP_PERSISTENT_ALL_REDUCE=0 GEMMA4_RANGE_DOWNLOAD=1 \
   TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
   python_env/bin/python -m pytest -q \
     models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
     -k 'multichip_matches_optimized_single_chip and full_attention' \
     --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/optimized_pcc_full_nonpersistent_ccl.xml
   ```

   BFP8 should worsen PCC if collective quantization is material.  An
   improvement only from disabling persistence implicates the async collective
   path rather than projection math.

7. Padding is lowest-ranked.  Dense 2112 is extended to 2176 and expert 704 to
   768 so every physical TP shard is tile-width legal.  Host checks prove the
   logical prefix is unchanged and only zeros are appended.  There is no
   existing env that removes padding without also making the local matmuls
   illegal, so `PACKED_DENSE_GATE_UP=0` must not be described as a padding A/B.
   If the preceding experiments all fail, add a diagnostic that redistributes
   the same 64 zero columns/rows as 16 zero values per rank (rather than all on
   the final global shard), then compare only that layout.  This is lower risk
   than attempting an unpadded 528/176-wide local tile program.

Host-only padding/state verification command:

```bash
python_env/bin/python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k 'multichip_raw_weight_policy_restores_routing_environment or multichip_tp_padding_preserves_logical_values_and_zero_fills_tail or multichip_initializes_reachable_inherited_factory_state'
```

### TP1/TP2/TP4 profile coverage

The generalized loader now obtains a frozen profile from
`_profile_for_tp(mesh_width)` instead of relying on TP4-only shape constants.
Host-only coverage locks the supported profile matrix:

| TP | padded/local dense | padded/local expert | local Q | local sliding KV | local full KV |
| --- | --- | --- | --- | --- | --- |
| 1 | 2112 / 2112 | 704 / 704 | 16 | 8 | 2 |
| 2 | 2112 / 1056 | 704 / 352 | 8 | 4 | 1 |
| 4 | 2176 / 544 | 768 / 192 | 4 | 2 | 1 |

The tests also prove:

- the supported sizes are exactly `(1, 2, 4)` and 0, 3, and 8 are rejected
  with a useful error;
- every padded width is aligned to `tp_size * 32` and every profile is frozen;
- the factory stores the profile and its local head counts on the decoder;
- full attention shards normally for TP1/TP2, while TP4 duplicates each of its
  two KV heads across the corresponding pair of Q ranks;
- full-attention paged-cache geometry uses the profile's local KV count; and
- packed dense gate/up host preparation preserves per-rank pairing for all
  three supported TP sizes.

Verification:

```bash
python_env/bin/python -m py_compile \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py

python_env/bin/python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k 'multichip_shape_contract or multichip_profile or multichip_cache_view_separates_update_and_sdpa_geometry or decode_only_packed_source_preserves_per_rank_gate_up_pairing or decode_only_packed_dtype_uses_independent_host_source'
# 14 passed, 41 deselected
```

## Final Status

- All constructor/API/attribute failures through v6 are fixed and reach the
  numerical gate in v7.
- `optimized_decoder.py` was not modified.
- Existing hardware artifacts confirm prefill now passes for both layers, but
  v7 decode remains below 0.995.  The primary sliding-specific mismatch is the
  attention dtype default; the leading shared decode hypothesis is the
  independent BFP4 packed dense copy.
- Remaining implementation caveat: the base API exposes routing selection only
  through a process environment variable, so the scoped default necessarily
  mutates that variable during construction.  It restores the caller's state in
  `finally` and never overrides an explicit caller value; a future base
  constructor parameter would eliminate even the temporary process-global
  state.

## Capacity policy repair after stage review

### Starting Evidence

- The stage review found that the former projection stopped at decoder weights,
  BF16 KV, and CCL: TP1 left 2,744,465,408 B, but separate BF16 embedding and
  LM-head tensors need 2,952,790,016 B before final norm, trace, activations, or
  allocator slack.
- The existing TP1 decoder policy was already accuracy-gated in
  `artifacts/pcc_tp1_capacity_selected.xml`: sliding prefill/decode is
  0.9981487298/0.9951300642 and full prefill/decode is
  0.9970008655/0.9985550317.
- This is an arithmetic placement defect rather than a hang, so the relevant
  stage review and existing AutoDebug/AutoFix reports were used directly; no
  fresh AutoTriage or broad source-only AutoDebug pass was needed.

### Hypothesis Experiments

- Hypothesis: a broader decoder-side BFP4 expert-down policy could create the
  missing full-stack reserve without changing terminal precision.
- Experiment:

  ```bash
  GEMMA4_RANGE_DOWNLOAD=1 \
  GEMMA4_MULTICHIP_EXPERT_DOWN_WEIGHT_DTYPE=bfp4 \
  GEMMA4_MULTICHIP_ARTIFACT_SUFFIX=_capacity_down_bfp4 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  python_env/bin/python -m pytest -q -s \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    -k 'test_p150_proxy_matches_optimized_single_chip' \
    --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/pcc_tp1_capacity_down_bfp4.xml
  ```

- Result: one pass, one failure on one P300C. Full attention passes at prefill
  0.9962860301/decode 0.9977604982, but sliding decode is 0.9941133959 against
  the 0.995 gate. The candidate artifact suffix kept this evidence separate
  from the selected policy.
- Verdict: refuted.
- Follow-up hypothesis: HiFi2 expert-down math could recover the BFP4 storage
  error.
- Experiment: reran the exact command with
  `GEMMA4_MULTICHIP_EXPERT_FIDELITY=hifi2`, suffix
  `_capacity_down_bfp4_hifi2`, and JUnit
  `artifacts/pcc_tp1_capacity_down_bfp4_hifi2.xml`.
- Result: one pass, one failure. Sliding decode is still 0.9941319551; full
  attention is unchanged at 0.9962860301/0.9977604982.
- Verdict: refuted. No expert-down dtype or fidelity production change was
  kept.

- Hypothesis: retaining the proven decoder policy while requiring downstream
  BFP8_B terminal storage and an explicit operating reserve produces a valid
  full-stack placement contract.
- Experiment: reconstruct every byte in `capacity_projection.json` from model
  constants. Each 262144x2816 terminal matrix has 720,896 tiles; at 1,088 B per
  BFP8 tile, embedding plus LM head use 1,568,669,696 B before TP sharding.
  Final norm uses 180,224 B. The largest live chunk estimate is one replicated
  BF16 `[30720,2816]` residual plus one TP-local BF16 sliding QKV projection,
  where 30,720 is the source `PREFILL_SLIDING_CHUNK_SIZE`. Together with the
  64 MiB trace region, the profile-specific allocator remainder makes the
  operating reserve exactly 1 GiB/device.
- Result: TP1 full-stack total is 34,257,864,704 B of 34,359,738,368 B, leaving
  101,873,664 B before considering any packed-expert retained copy. The BF16
  terminal control exceeds TP1 capacity.
- Verdict: verified as an analytical capacity contract.
- Fix: expanded `capacity_projection.json`, `context_contract.json`,
  `README.md`, and `mesh_plan.md`; strengthened the host regression to derive
  terminal tiling, chunk activation, reserve, total, and headroom identities.
  The downstream BFP8_B terminal contract is deliberately not implemented by
  this decoder stage and still requires terminal/logit correctness evidence in
  the full-model stage.

- Follow-up hypothesis: the newly selected packed-expert decode tensor can be
  retained on all TP profiles within the repaired full-stack contract.
- Experiment: reconstruct its persistent physical size independently of
  hardware. Each layer retains one BFP8 tensor with per-device shape
  `[1,128,2816,2*E_local]`. At 1,088 B per physical BFP8 tile, the TP2
  `E_local=352` tensor has 247,808 tiles and uses 269,615,104 B/layer; the TP4
  `E_local=192` tensor has 135,168 tiles and uses 147,062,784 B/layer. Over 30
  layers those copies use 8,088,453,120 B and 4,411,883,520 B respectively.
  The host regression derives those values from the profile widths and keeps
  them distinct from TP4's prior 642,611,200 B of O/dense retained copies.
- Result: with packed expert enabled, TP2 projects to 26,433,070,080 B with
  7,926,668,288 B headroom, and TP4 projects to 16,717,143,040 B with
  17,642,595,328 B headroom. TP1's analogous all-BFP8 copy would use
  16,176,906,240 B. Even the unselected all-BFP4 physical-tile lower bound,
  495,616 tiles/layer x 30 layers x 576 B/tile = 8,564,244,480 B, exceeds its
  2,744,465,408 B decoder-stage headroom.
- Verdict: refuted for TP1 and verified for TP2/TP4. The selected policy is
  packed expert disabled on TP1 and enabled on TP2/TP4. This follow-up changed
  only the capacity ledger, host regression, and documentation; it did not use
  hardware or edit production code.

Verification:

```bash
python -m json.tool \
  models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/capacity_projection.json
python -m json.tool \
  models/autoports/google_gemma_4_26b_a4b_it/doc/context_contract.json
python_env/bin/python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k multichip_capacity_projection_is_internally_consistent
# 1 passed, 69 deselected
```

### Final Status

- Fixed: the capacity ledger no longer treats decoder/KV headroom as sufficient
  for the full stack, and every mandatory profile fits the 32 GiB analytical
  basis with explicit terminal, final-norm, trace, activation/transient, and
  allocator accounting.
- Kept policy: the previously proven decoder precision policy is unchanged;
  both broader expert-down experiments were refuted and discarded.
- Packed-expert policy: disabled on TP1 by a hard capacity limit, and included
  as an exact 30-layer retained-copy charge on TP2/TP4. The ledger separately
  identifies pre-existing O/dense retained copies.
- Remaining risk: BFP8_B terminal accuracy and measured full-model peak
  allocation are downstream gates. This stage records a mandatory placement
  contract, not a full-model correctness claim.

## Optimized-baseline contract preservation

### Verified source hypothesis

The original multichip factory instantiated `OptimizedDecoder` with raw TP
weights, R0 residuals, and every optimized graph/packing flag disabled. The
optimized baseline instead relies on an R22 L1-sharded local residual, folded
router and scale algebra, shared FFN norm, fused final scalar, row-major
routing, and packed rank-local expert decode. Source and host-only algebra
checks established the required adaptation boundaries:

- graph folds must run on the immutable host state before transpose, TP
  fracture, padding, and upload; folded cache paths must be policy-qualified;
- R22 remains local to a device. QKV and dense inputs cross from the inherited
  sharded residual to DRAM before TP-local projections, while reduced hidden
  outputs return to the inherited residual memory config;
- expert gate/up packing must concatenate each rank's local `[up, gate]` pair,
  not concatenate global tensors before mesh sharding;
- rank-local packed expert decode is a retained copy. TP1 cannot afford it,
  while the independently audited TP2/TP4 full-stack projections can;
- row-major routing is valid only when the prepared expert scale fold and R22
  residual are both selected. The base constructor intentionally rejects an
  explicit row-major request with R0;
- optimized multi-reader DRAM padding can deallocate its input. A candidate
  must clone an aliased prefill tensor first, and its K/N padding and logical
  output slice must remain end-to-end visible to `_linear`.

Host regressions cover fold-before-fracture algebra, rank-local packing for
TP1/TP2/TP4, R22 projection boundaries, inherited batch-32 expert attributes,
temporary routing-env restoration, the row-major/R0 base contract, and the
DRAM reader maps. `optimized_decoder.py` was not changed.

### Isolated hardware experiments

On the four-chip P300C QB2, each TP4 direct candidate cleared the 0.995
prefill/decode gate for both representative layer kinds. The cumulative R22 +
four folds + packed expert + row-major candidate produced sliding
0.997090/0.999223 and full 0.998581/0.999715. At logical S=33 its trace replay
was 0.651225/0.728323 ms for sliding/full, versus optimized single-chip
0.765691/0.812069 ms. These results verified the adapters and the B1 execution
benefit, but did not by themselves settle batch-32 correctness.

Batch 32 exposed two independent approximation effects. The cumulative TP4
full candidate scored 0.994389, below the unchanged 0.995 gate; higher
attention/expert/MLP fidelity and isolated BF16 attention, expert gate/up,
expert down, dense gate/up, and dense down candidates did not recover it. R0
raw passed at 0.995053. Starting there, the maximal passing optimized subset
was packed expert + fused final scalar + folded expert scale: PCC 0.995025 and
9.173047 ms versus the optimized single-chip 12.199845 ms. Adding folded
router projection scored 0.994890, and adding shared FFN norm scored 0.994940,
so neither is selected for TP4 full layers. The TP4 sliding cumulative policy
passes at 0.995238 and 8.818332 ms versus 12.199192 ms.

The selected default is therefore profile and layer specific:

| Profile/layer | local residual | router fold | shared FFN norm | expert-scale fold | fused scalar | row-major routing | packed expert decode |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| TP1 sliding/full | R22 | yes | yes | yes | yes | yes | no |
| TP2 sliding/full | R22 | yes | yes | yes | yes | yes | yes |
| TP4 sliding | R22 | yes | yes | yes | yes | yes | yes |
| TP4 full | R0 | no | no | yes | yes | no | yes |

TP4 keeps O, packed dense gate/up, and dense down as decode DRAM-sharded
roles. A later HF-oracle AutoFix retained the same BF16 O tensor but selected
`in0_block_w=2` for logical B1 and `4` for B32; see
`AUTOFIX_HF_ORACLE.md`. The selected value is one reader per bank. Workers 2 and 3 were not
timed: both deterministically reached the inherited primitive's unit-mesh
fatal (`mesh->num_devices() == 1`). Construction now raises `ValueError` for
workers greater than one on TP2/TP4, preventing an environment override from
reaching that fatal.

### Canonical reruns

The final policy was rerun without candidate environment overrides:

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python_env/bin/python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k 'test_multichip_matches_optimized_single_chip or test_p150_proxy_matches_optimized_single_chip or test_p150x2_proxy_matches_optimized_single_chip' \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/direct_pcc_selected.xml
```

All six cases passed. Canonical `pcc_tp{1,2,4}_layer{0,5}.json` values are:

| Profile | sliding prefill/decode | full prefill/decode |
| --- | --- | --- |
| TP1 | 0.998077/0.999262 | 0.997276/0.998433 |
| TP2 | 0.997943/0.999121 | 0.998801/0.999741 |
| TP4 | 0.997090/0.999319 | 0.998620/0.999875 |

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python_env/bin/python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k test_multichip_non_aligned_prefill_and_decode_trace \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/trace_selected.xml
```

Both TP4 S=33 cases passed 30 bit-exact replays and replica equality. Canonical
After the HF-oracle repair, `trace_sliding_attention_batch1.json` is
0.652139 ms (1.1741x versus optimized) and
`trace_full_attention_batch1.json` is 0.952428 ms (0.8526x).

The post-repair combined batch-32 gate is `batch32_selected_final.xml` plus
`multichip_batch32_layer{0,5}.json`. Both
repeat bit-exactly and pass PCC as reported above. The full candidate ladder is
preserved under `batch32_*.xml` and the corresponding suffixed JSON artifacts.

The selected TP2 stacked resource/layout stress is
`stacked_tp2_selected.xml` and `stacked_tp2_mixed_trace.json`. Twenty trace
replays, eager-versus-trace output, and device replicas are bit exact. The
stage-review repair additionally runs both layer kinds on identical inputs and
fresh caches: Optimized-versus-TP2 decode is 0.991223 sliding and 0.995297
full, both above 0.99. Independent real-weight TP2-versus-HF decode is
0.999566/0.999711, above 0.995. Chaining the approximate layer-0 MoE into layer
5 changes one of eight routes and yields 0.983471, so only that final divergent
output uses its explicit 0.98 discontinuity threshold; see
`AUTOFIX_STACKED_CCL.md` and `tp2_hf_same_input_stacked_final.xml`.

Final watcher coverage was run independently of profiling:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python_env/bin/python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k 'test_multichip_matches_optimized_single_chip or test_multichip_non_aligned_prefill_and_decode_trace' \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/watcher_selected.xml
```

All four cases passed without Tensix or dispatch errors. Active-ETH watcher
instrumentation remains a scoped firmware-size limit and is preserved in
`watcher_active_eth_overflow.xml`; correctness, trace, and performance gates
use the normal active-ETH fabric path.

### Verdict and remaining uncertainty

Verified and kept: profile-aware R22 boundaries, host graph folds, stable
row-major routing storage, rank-local packed expert decode for TP2/TP4,
explicit inherited fidelity defaults, complete inherited factory state, and
the TP4 selected DRAM roles. The TP4/full B32 exception is the largest measured
subset that clears the original gate. No PCC threshold was reduced except for
the separately identified chained-MoE stress.

The remaining uncertainty is downstream rather than hidden by this repair:
the terminal BFP8_B full-stack placement remains analytical until full-model
measurement; TP4 full S=33 is slower than its one-chip optimized reference;
and the TP2 chained MoE route discontinuity remains visible in its dedicated
artifact. Workers greater than one are unsupported, not a performance result.

## Stage-review P1: full-length prefill lifetime

### Reproduced defect

The prior full-stack ledger treated `PREFILL_SLIDING_CHUNK_SIZE=30720` as the
largest activation lifetime. Source inspection disproves that assumption:
`_attention_prefill` creates the complete QKV tensor before selecting the
chunked SDPA path, both attention helpers retain every output chunk until
`ttnn.concat`, and `_prefill_forward_single_user` keeps full residual and
branch tensors live across attention and FFN. `_moe_prefill` additionally
holds those outer tensors while it executes 1,024-token expert chunks.

### Isolated hypotheses

- Projection-level chunking could reduce QKV, but it would not make TP1 fit
  while the inherited caller retains a replicated S=262144 residual/output.
  No partial chunking change was kept.
- Explicit deallocation was not credited because the inherited ownership and
  concat lifetime have not been independently proven safe.
- Broader TP1 BFP4 expert-down storage was already isolated with real-weight
  prefill/decode. HiFi4 and HiFi2 controls both missed PCC 0.995, so the
  candidate remains refuted.
- Removing TP2's BFP8 packed-expert decode copy reclaims exactly
  8,088,453,120 B/device without changing the prefill weight placement. This
  capacity-mandated policy change is kept; fresh TP2 PCC, stacked, and timing
  evidence is required.

### Conservative accounting and fix

At advertised S=262144, with `H=S*2816*2`, profile-local BF16 QKV/Q, full
BF16 RoPE, and caller padding copies, the maximum source-live tensors are:

| Profile | attention concat | attention O/reduce | dense MLP | MoE | selected peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP1 | 22,817,013,760 B | 21,474,836,480 B | 14,931,722,240 B | 15,927,869,440 B | 22,817,013,760 B |
| TP2 | 13,153,337,344 B | 15,435,038,720 B | 13,639,876,608 B | 15,651,045,376 B | 15,651,045,376 B |
| TP4 | 8,858,370,048 B | 12,213,813,248 B | 12,297,699,328 B | 17,001,611,264 B | 17,001,611,264 B |

With terminals, norm, 64 MiB trace, and preserved allocator slack, TP2 and TP4
fit at S=262144 with 788,749,312 B and 939,828,224 B headroom. TP1 would
need 56,398,546,944 B, 22,038,808,576 B beyond 32 GiB. It therefore receives
the explicit hard-physical-limit exception required by the user.

For TP1, the supported-length projection is
`28,212,824,064 + ceil128(S)*20,480 + max(attention(S), moe(S))`, where
`attention(S)=ceil32(S)*87,040 + padding(S)`,
`moe(S)=2,036,334,592 + ceil32(S)*52,992 + padding(S)`, and
`padding(S)=S*7,680` for non-tile-aligned caller-owned hidden/cos/sin inputs.
Every length through 50,624 fits: S=50,623 is the tightest at 1,037,824 B
headroom, aligned S=50,624 leaves 389,822,464 B, and S=50,625 is the first
failure at -673,280 B. `MultichipDecoder.prefill_forward` now rejects an
over-limit TP1 input before device allocation; TP2/TP4 retain 262,144.

The machine ledger and context contract encode every intermediate and a host
regression independently recomputes KV block rounding, all four candidate
activation peaks, retained-copy policy, totals, and boundaries. The terminal
BFP8_B policy remains a mandatory downstream placement contract, not a
full-model implementation or logit-correctness claim.

### Verification status

- Focused host capacity/profile command: six passed.
- S=50,625 safe preflight rejection: covered by the host policy test.
- The earlier real-weight TP1 S=53,343/S=53,344 layer probes passed but are
  superseded as boundary evidence: a follow-up audit found that the MoE bound
  must include both accumulated chunks and the newly allocated concat output.
- Corrected real-weight TP1 S=50,623 and S=50,624 layer probes: both
  representative layer kinds passed at both lengths (four tests), with finite
  last-token output and fallback throwing. Authoritative evidence is
  `artifacts/p150_prefill_capacity_{50623,50624}.xml` and the corresponding
  four `prefill_capacity_{sliding,full}_attention_{50623,50624}.json` files.
- TP2 no-packed direct PCC, same-input stacked gate, and performance rerun:
  pending serialized hardware execution.
