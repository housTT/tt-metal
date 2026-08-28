# AutoDebug: full-model reduced endpoint all-gather crash

Date: 2026-08-27

Command that failed:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
python -m pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_reduced_real_weight_full_model_endpoint_smoke -s
```

Observed symptom: native exit 139 inside `ttnn.all_gather` from
`MultichipDecoder.gather_residual`, called by `Qwen38FullModel.final_hidden`
after a real layer-0 prefill. The same all-gather primitive is covered by
existing multichip decoder tests, so the first investigation focused on the
new full-model tensor lifetime and endpoint path.

## Ranked hypotheses

1. **Confirmed: double-free corrupts the gathered input lifetime.**
   `Qwen38FullModel.prefill_forward` sliced the final logical token, called
   `_functional_decoder._free(residual, last)`, and then immediately called
   `ttnn.deallocate(residual)` again. `_free` already deallocates non-aliased
   live TTNN tensors, so the second deallocate can poison later collective
   execution. The redundant deallocate was removed.

2. **Possible if the crash persists: embedding-created fractured residual has
   a layout/ownership difference from `_fractured_upload`.** Existing tests
   construct stack residuals with `ShardTensorToMesh(dim=3)` from a grouped
   tensor. The full model constructs residuals via sharded embedding plus
   reshape/repeat. If the double-free fix does not clear the failure, compare
   per-rank shape/layout/address metadata immediately before final gather.

3. **Less likely: final endpoint reshapes alias input unexpectedly.** The
   final hyperconnection path reshapes gathered/normed/mix tensors heavily; no
   pre-gather mutation was found here, but if hypothesis 1 is insufficient,
   capture one all-logits path with `return_all_logits=True` and compare.

## Focused verification plan

- Rerun the same reduced real-weight smoke after the double-free fix.
- If it still crashes, add a temporary single-use diagnostic around
  `final_hidden` to print `shape`, `layout`, `buffer_address`, and
  `buffer_unique_id` for the residual shards before `gather_residual`.
- If gather succeeds, extend the reduced gate to a layer `(0, 1, 3)` stack and
  then the token-out trace split before attempting full 48-layer validation.
