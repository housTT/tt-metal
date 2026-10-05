# Policy probe files (stage 3)

Per-layer encoder traces (`layer_pcc_*.json`, same layout as `../../functional_decoder/layer_pcc_bf8w_hifi3_stage1port.json`)
and the PCC rows (`pcc_*.json`) of the policy probes of 2026 Oct 5 21:50 to 22:01 UTC, all on the C1 port
(`minimal_matmul` Wqkv and Wo, interleaved 2816 GeGLU on 11x8, SDPA 8x8, rotary unsharded, L1 chain 2048 rows).

| file | policy | source log |
|---|---|---|
| `layer_pcc_candidate.json`, `pcc_candidate.json` | `bf8w_hifi3` | `/home/hous/dev/laya/logs/p3_s3_cand_correctness_20261005T215000Z.log` |
| `layer_pcc_fp32res.json`, `pcc_fp32res.json` | `bf8w_hifi3_fp32res` | `/home/hous/dev/laya/logs/p3_s3_fp32res_tests_20261005T215303Z.log` |
| `layer_pcc_bf16_hifi4.json`, `pcc_policy_bf16_hifi4.json`; `layer_pcc_bf8w_hifi2.json`, `pcc_policy_bf8w_hifi2.json` | `bf16_hifi4`, `bf8w_hifi2` | `/home/hous/dev/laya/logs/p3_s3_policy_probe_20261005T215512Z.log` |
| `layer_pcc_bf8w_hifi4.json`, `layer_pcc_bf16w_hifi3.json`, `layer_pcc_bf8w_hifi3_head_bf16.json` and their `pcc_policy_*.json` | `bf8w_hifi4`, `bf16w_hifi3`, `bf8w_hifi3_head_bf16` | `/home/hous/dev/laya/logs/p3_s3_policy_probe2_20261005T215659Z.log` |
| `layer_pcc_bf8w_hifi3_erf.json`, `pcc_policy_bf8w_hifi3_erf.json` | `bf8w_hifi3_erf` | `/home/hous/dev/laya/logs/p3_s3_policy_probe3_20261005T215933Z.log` |

The policy table of `../README.md` is generated from these PCC rows by `/home/hous/dev/laya/scratch/stage3_tables.py`.
