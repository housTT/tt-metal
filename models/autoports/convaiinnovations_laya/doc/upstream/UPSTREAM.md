# Upstream source

Copied from tenstorrent/tt-metal commit dfb35061a56010be87c86fc903fd42ad890fa4b5 (2026-09-15, PR #51450, "[modernbert] ModernBERT-base bring-up on Wormhole N300"), path models/experimental/modernbert/. Module paths were rewritten to models.autoports.convaiinnovations_laya with sed; no other change at copy time. Per-file blob ids: PROVENANCE.txt. Files under src/ are kept for reference only and are not shipped.

```
dfb35061a56010be87c86fc903fd42ad890fa4b5 2026-09-15 Soma Sreeshanth [modernbert] ModernBERT-base bring-up on Wormhole N3

 models/README.md                                   |   1 +
 models/experimental/modernbert/README.md           | 517 +++++++++++++++++++++
 models/experimental/modernbert/common.py           | 136 ++++++
 models/experimental/modernbert/demo/__init__.py    |   0
 models/experimental/modernbert/demo/demo.py        | 162 +++++++
 .../experimental/modernbert/reference/__init__.py  |   3 +
 .../modernbert/reference/modernbert.py             | 277 +++++++++++
 models/experimental/modernbert/runner/__init__.py  |   3 +
 .../modernbert/runner/performant_runner.py         | 159 +++++++
 .../modernbert/runner/performant_runner_infra.py   |  92 ++++
 models/experimental/modernbert/tests/__init__.py   |   0
 models/experimental/modernbert/tests/pcc_utils.py  |  43 ++
 .../modernbert/tests/test_model_config.py          | 192 ++++++++
 .../tests/test_modernbert_device_perf.py           |  54 +++
 .../modernbert/tests/test_modernbert_perf.py       |  78 ++++
 .../modernbert/tests/test_modernbert_performant.py | 170 +++++++
 .../modernbert/tests/test_modernbert_profile.py    |  39 ++
 .../modernbert/tests/test_reference_coverage.py    | 147 ++++++
 .../modernbert/tests/test_reference_parity.py      | 115 +++++
 .../modernbert/tests/test_ttnn_attention.py        | 164 +++++++
 .../modernbert/tests/test_ttnn_embeddings.py       |  83 ++++
 .../modernbert/tests/test_ttnn_layer.py            |  94 ++++
 .../experimental/modernbert/tests/test_ttnn_mlm.py | 135 ++++++
 .../experimental/modernbert/tests/test_ttnn_mlp.py |  99 ++++
 .../modernbert/tests/test_ttnn_model.py            | 141 ++++++
 .../modernbert/tests/test_ttnn_rope.py             | 114 +++++
 models/experimental/modernbert/tt/__init__.py      |   3 +
 models/experimental/modernbert/tt/model_config.py  | 311 +++++++++++++
 .../modernbert/tt/modernbert_attention.py          | 114 +++++
 .../modernbert/tt/modernbert_embeddings.py         |  31 ++
 .../experimental/modernbert/tt/modernbert_head.py  |  59 +++
 .../experimental/modernbert/tt/modernbert_layer.py | 162 +++++++
 .../experimental/modernbert/tt/modernbert_masks.py |  86 ++++
 .../experimental/modernbert/tt/modernbert_mlp.py   | 127 +++++
 .../experimental/modernbert/tt/modernbert_model.py | 113 +++++
 .../experimental/modernbert/tt/modernbert_rope.py  | 139 ++++++
 models/experimental/modernbert/tt/weights.py       | 182 ++++++++
 models/model_ci_tiers.md                           |   1 +
 tests/pipeline_reorg/models_e2e_tests.yaml         |  15 +
 tests/pipeline_reorg/models_unit_tests.yaml        |  18 +
 40 files changed, 4379 insertions(+)
```
