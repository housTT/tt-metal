# Work log: stage 3 (optimize), Track T1

All times UTC, 2026 Oct 5. Same conventions as `../functional_decoder/work_log.md`.

- 21:35: wrote `tests/bench_buckets.py` (one process per `PortConfig` variant, eager and traced p50 per bucket with the
  load average per cell) and the quiet-host queue `/home/hous/dev/laya/scratch/t1_quiet_queue.sh`: wait for a 1-minute
  load under 8, then under devlock in order: Tracy profile of layer 1 at B 8 (stage 1 gate), untracked trace timing
  (stage 2), Tracy at B 1 and B 64, then the 15 A/B variants of the README matrix on buckets 1x512, 8x512, 64x512.
- 21:36: queue started (`/home/hous/dev/laya/logs/p3_quiet_queue_20261005T213653Z.log`); the load fell under 8 at
  21:36 (7.09) and step 1 began at once.
- 21:38: Tracy profiles of layer 1 at B 8, 1 and 64 done (loads 7.1, 8.1, 8.5 at start); untracked trace timing done;
  `bench_default` done: traced p50 13.92 / 85.52 / 646.0 ms at 1x512 / 8x512 / 64x512 (load 8.5 to 8.7, slightly above
  the rule's 8; a `default_rerun` is queued at the end of queue 2). Per-op reading of the B 64 profile from
  `tt-perf-report`: Wqkv (my 8x8 config) 73 percent FLOPs on 64 cores with the advice "increase grid size"; mlp Wo (auto,
  `core_grid` 8x8) 72 percent; attn Wo (auto) 35 percent and the two Wi matmuls (auto, 88 cores) 39 percent, all three
  flagged SLOW with "in0_block_w=1 is small" and "place input 0 in L1"; the GELU runs as a separate
  `UnaryDeviceOperation` (4.2 percent) because `activation=` on `ttnn.linear` is not fused on the automatic path.
- 21:40: the three `minimal_matmul` variants failed with a `TypeError` (a positional `None` bias is rejected); fixed to
  pass `bias_tensor=` only when present. Added `PortConfig.interleaved_pad` (padded interleaved GeGLU with an explicit
  mcast config and the fused GELU), `mlp_grid` (8x8 or 11x8: 2816 is 88 tiles, divisible by 11) and
  `wo_program_config` (explicit 8x8 mcast configs for attn Wo and mlp Wo with `in0_block_w` 8). Config test 46 passed.
- 21:42: queue 2 launched behind queue 1 (`/home/hous/dev/laya/scratch/t1_quiet_queue2.sh`,
  `/home/hous/dev/laya/logs/p3_quiet_queue2_20261005T214223Z.log`): minimal variants again, il2816_8x8, il2816_11x8,
  il3072_8x8, wo_mcast, the combination, the combination with SDPA 128, and `default_rerun`.
- 21:44: queue 1 facts so far: `geglu_shard_to_4096` and `chain_l1_to_4096_outblock4` both fail at B 8 with the
  circular-buffer clash (`program.cpp:1932`); the clash address is the same with a halved Wqkv out block, so the large
  CB owner at 4096 rows is SDPA with 256-token chunks. `chain_dram_always` costs 6.3 percent at B 1 (14.79 ms). SDPA on
  8x8 is 8.8 percent faster at B 1 (12.70 ms). A follow-up variant (L1 chain to 4096 rows with 128-token SDPA chunks)
  goes into the final queue.
- 21:47: queue 2 first results (loads 4 to 7): the `minimal_matmul` variants run (bfp8 weights with bf16 activations
  are accepted by this build). As wired at the time, `qkv_minimal_11x10` switched Wqkv, attn Wo and mlp Wo to
  `minimal_matmul` (traced 565.8 ms at B 64, -12.4 percent; 81.3 ms at B 8, -4.9 percent), `down_minimal_11x10`
  switched only the two Wo (593.4 ms, -8.2 percent; 83.5 ms, -2.4 percent) and `both_minimal_11x10` equals the first.
  Fixed the wiring so `qkv_mode` and `down_grid` are independent (`BucketPlan.qkv_minimal`, `wo_minimal`); the
  measured variants are documented with the ops they actually ran. `il2816_8x8` (padded interleaved GeGLU with the
  fused-GELU 8x8 mcast config): 627.5 ms at B 64 (-2.9 percent), 85.3 ms at B 8 (-0.2 percent).
- 21:50: candidate correctness run `s3_cand_correctness` (overrides: Wqkv and both Wo through `minimal_matmul` 11x10,
  SDPA on 8x8, rotary never sharded, padded interleaved GeGLU 2816 with the fused-GELU mcast config;
  `/home/hous/dev/laya/logs/p3_s3_cand_correctness_20261005T215000Z.log`): encoder and head tests 17 passed (encoder
  0.9930 to 0.9974 by shape, end to end markers 0.9929 at B 1), tracked replay check pass at 1x512, 8x512 and 64x512
  (bit identical, no tracker error). `il2816_11x8` (88-core fused GeGLU config): 75.45 ms at B 8 (-11.8 percent),
  587.6 ms at B 64 (-9.0 percent), 12.55 ms at B 1 (-9.8 percent).
- 21:53: orchestrator relay of Track R's findings: fp32 residual outliers reach 33109 after layer 19 (channels 379 and
  382, bf16 quantum 256) and the aggregate per-layer PCC of the B 2 and B 4 batches falls under 0.99 before the final
  norm (0.9805 and 0.9610 at layers 19 to 26; tanh values from the stage 1 run, `/home/hous/dev/laya/logs/p3_s1_encoder_head_20261005T212633Z.log` and the regenerated `../functional_decoder/layer_pcc_bf8w_hifi3_stage1port.json`; the erf run of the shipped port is `layer_pcc_shipped.json` here), so the plan's section 12 item 4
  fallback is implemented and measured: `PrecisionPolicy.residual_dtype` (policies `bf8w_hifi3_fp32res` and
  `bf16_hifi4_fp32res`): the residual stream is carried in fp32 on the interleaved path, each LayerNorm reads fp32 and
  its output is typecast to bf16 before the matmuls, each branch output is typecast to fp32 before the residual add,
  the embedding output is typecast to fp32 and the final norm's output back to bf16 for the head; the sharded resident
  path is disabled under this policy. Correctness job `s3_fp32res_tests` launched (encoder per-layer PCC and head);
  its cost is measured by `fp32res_on_C1` in queue 3.
- 21:55: `s3_fp32res_tests` (`/home/hous/dev/laya/logs/p3_s3_fp32res_tests_20261005T215303Z.log`): the fp32 residual
  does not recover the content-dependent loss: fill row 0.9929 (bf16 residual 0.9911), B 2 0.9950 (0.9950), the fill
  row after layer 19 0.9778 (0.9751), B 1 0.9974 (0.9974). Encoder B 4 and B 8 failed under this policy with the
  circular-buffer clash (8x8 interleaved GeGLU config without the sharded plan); not pursued because the policy buys
  no accuracy. Conclusion: the loss on some sequences is not the bf16 storage of the residual stream. Probing the
  weight and math side next: `s3_policy_probe` runs the encoder and head tests under `bf16_hifi4` and `bf8w_hifi2`
  (B 1, 2 and 1024 shapes).
- 21:57: `s3_policy_probe` (`/home/hous/dev/laya/logs/p3_s3_policy_probe_20261005T215512Z.log`, candidate port):
  `bf16_hifi4` lifts the encoder to 0.99958 (B 1), 0.99797 (fill row, 0.9868 after layer 19), 0.99652 (B 2),
  0.99957 (S 1024) and the end-to-end marker logits to PCC 0.99974 with a max abs error of 0.119 (bf8w_hifi3: 0.9929
  and 0.343). `bf8w_hifi2` sits with `bf8w_hifi3` (0.99746, fill row 0.99093, markers 0.99598 / 0.306). So the
  content-dependent loss comes from the weight format or the HiFi4 math, not from the activation storage or the
  fidelity step from 3 to 2; `s3_policy_probe2` runs `bf8w_hifi4` and `bf16w_hifi3` to attribute it, plus
  `bf8w_hifi3_head_bf16`; queue 3b measures the cost of every policy on the candidate port at B 1, 8, 64.
- 21:59: `s3_policy_probe2` (`/home/hous/dev/laya/logs/p3_s3_policy_probe2_20261005T215659Z.log`): `bf8w_hifi4`
  (bfp8 weights, HiFi4, tanh GELU) 0.99738 at B 1, fill row 0.99214, markers 0.99433 / 0.329; `bf16w_hifi3` (bf16
  weights, HiFi3, tanh GELU) 0.99737, fill row 0.99237. Neither recovers what `bf16_hifi4` gains, so the third
  difference of that policy, the exact erf GELU instead of the tanh approximation, is the suspect: the massive
  activations come out of `gelu(a) * gate` with a gate of thousands, which multiplies any error of the activation.
  Added `bf8w_hifi3_erf` (bfp8, HiFi3, erf GELU fused in the matmul) and launched `s3_policy_probe3`; its cost bench
  is queued (queue 3c).
- 22:01: `s3_policy_probe3` (`/home/hous/dev/laya/logs/p3_s3_policy_probe3_20261005T215933Z.log`): `bf8w_hifi3_erf`
  (bfp8 weights, HiFi3, exact erf GELU fused in the Wi_act matmul) 17 passed; encoder 0.99936 (B 1), 0.99894 (fill
  row; 0.9976 after layer 19), 0.99550 (B 2 sharded), 0.99741 (B 2 interleaved), 0.99848 (B 4), 0.99785 (B 8),
  0.99939 (S 1024); end to end markers 0.99986 with max abs 0.090 (B 1) and 0.99934 with 0.153 (B 8), CLS 0.99999.
  The tanh GELU approximation was the dominant error of the upstream default policy on -large; the exact erf costs
  nothing in accuracy anywhere and its time cost is measured by queue 3c. Decision pending that number: make
  `bf8w_hifi3_erf` the shipped default policy (PLAN.md A.7 names `bf8w_hifi3` as the start; the sweep of stage 8 keeps
  both).
- 22:09: queue 2 done (`/home/hous/dev/laya/logs/p3_quiet_queue2_20261005T214223Z.log`): `il2816_11x8` 12.55 / 75.45 /
  587.6 ms (-9.8 / -11.8 / -9.0 percent), `il3072_8x8` -2.9 / -5.2 / -1.9, `wo_mcast` -0.8 / -0.3 / -3.7,
  `il2816_11x8_wo_mcast` 12.87 / 74.76 / 536.6 (-7.5 / -12.6 / -16.9), the same with SDPA 128 chunks 12.79 / 76.54 /
  545.8 (128 loses to 256 above 768 rows), `default_rerun` at load 7.0: 13.79 / 85.69 / 646.0 (baseline reproducible
  within 1 percent).
- 22:10: queue 2b done (B 2 and B 4, loads 9.3 to 10.0, no load gate in that script, compared with each other):
  sharded 2816 18.55 / 32.15 ms, interleaved 2816 on 11x8 19.53 / 34.06, interleaved 2816 on 8x8 with Wo configs
  19.89 / 35.02, interleaved automatic 22.14 / 39.15; sharded 3072 fails at trace capture with the circular-buffer
  clash (eager passes). The sharded 2816 plan stays for 1024 to 2048 rows; the shipped B 2 and B 4 cells are
  re-measured under a quiet host in the final queue.
- 22:16: queue 3 done (`/home/hous/dev/laya/logs/p3_quiet_queue3_20261005T215409Z.log`, loads 7.0 to 8.2):
  C1 (minimal Wqkv and both Wo, 11x8 GeGLU, SDPA 8x8, rotary unsharded) 12.66 / 67.60 / 482.3 ms (-9.0 / -21.0 /
  -25.3 percent); C2 (Wo through mcast configs instead) 12.31 / 69.22 / 507.6; C3 (minimal Wqkv only, Wo on the 8x8
  core grid) 11.86 / 69.77 / 556.7; C1 at B 2 and 4 18.94 / 32.12 against 18.55 / 32.15 for the core-grid Wo; C1 with
  the interleaved GeGLU at B 2 and 4 19.54 / 32.55; `fp32res_on_C1` 13.94 / 80.38 / 581.8 (+10 / +19 / +21 percent for
  no accuracy gain, rejected); `qkv_minimal_only` 12.28 / 67.53 / 481.5: its process started after the defaults below were set, so it measured the shipped port (its JSON records the port); the B 1 cell against C3's identical configuration (11.86) puts the B 1 run-to-run scatter at about 3.5 percent.
  Decision (shipped `PortConfig` defaults): Wqkv through `minimal_matmul` 11x10 at every bucket; Wo through
  `minimal_matmul` from 4096 rows (`wo_minimal_min_rows`), on the 8x8 core grid below; interleaved GeGLU padded to
  2816 with the fused-GELU 11x8 mcast config above 2048 rows, the sharded 2816 plan for 1024 to 2048 rows, ttnn's
  automatic path only at 512 rows when sharding is off; SDPA on 8x8 with chunks 128 below 768 rows and 256 above;
  rotary never sharded; L1 chain to 2048 rows. The stage 1 baseline is kept as `STAGE1_PORT`. Config test 52 passed.
- 22:21: policy costs on the C1 port (queues 3b and 3c, loads 4.8 to 7.6): `bf16_hifi4` 14.56 / 81.47 / 582.1 ms,
  `bf8w_hifi4` 13.24 / 73.29 / 528.8, `bf16w_hifi3` 12.97 / 70.31 / 493.4, `bf8w_hifi2` 11.83 / 62.15 / 442.8,
  `bf8w_hifi3_head_bf16` 11.81 / 67.29 / 482.6, `bf8w_lofi_mlp` 11.79 / 61.23 / 435.5 (LoFi did not hang on p150),
  `bf8w_hifi3_erf` 12.97 / 73.04 / 527.3. Decision: `DEFAULT_POLICY_NAME = "bf8w_hifi3_erf"` (+2.4 / +8.0 / +9.3
  percent for the accuracy recorded at 22:01); `bf8w_hifi3` stays in the table. Config and weight tests 60 passed.
- 22:22: final queue launched (`/home/hous/dev/laya/scratch/t1_final_queue.sh`,
  `/home/hous/dev/laya/logs/p3_final_queue_20261005T222233Z.log`): correctness tests on the shipped defaults 42 passed
  (MLP 0.99999 with the erf GELU, encoder 0.99940 / 0.99805 fill / 0.99547 B 2 / 0.99845 B 4 / 0.99785 B 8 / 0.99939
  S 1024, end to end markers 0.99998 with max abs 0.036 at B 1 and 0.99920 with 0.163 at B 8;
  `pcc_rows_shipped.json`); tracked replay pass at 1x512, 8x512, 64x512 (bit identical, no tracker error). The
  shipped bench, the B 2 / 4 / 16 / 32 cells, the L1-chain follow-up and the shipped-layer Tracy profiles wait for a
  quiet host.
- 22:26: `bench_shipped` (12.42 / 68.12 / 481.4 ms) was taken with the tanh policy: `bench_buckets.py`,
  `profile_layer.py` and `replay_trace_check.py` defaulted `--policy` to the literal `bf8w_hifi3` instead of
  `DEFAULT_POLICY_NAME`. Fixed (default None resolves to the shipped policy). The running final queue still produces
  valid shipped-port numbers under the tanh policy (recorded as such: `bench_shipped*.json`,
  `tracy/shipped_layer1_*`); final queue 2 (`/home/hous/dev/laya/scratch/t1_final_queue2.sh`,
  `/home/hous/dev/laya/logs/p3_final_queue2_20261005T222647Z.log`) repeats the shipped bench at 1 / 8 / 64 and
  2 / 4 / 16 / 32 rows and the layer-1 profiles at B 64 and B 1 with the erf policy (`bench_shipped_erf*.json`,
  `tracy/shipped_erf_layer1_*`).
- 22:28: final queue 1 done: `chain_l1_4096_sdpa128` at B 8 (erf policy, started after the default fix) 65.80 ms
  against 73.04 for the erf C1 configuration (-10 percent): the L1 attention chain fits at 4096 rows when SDPA uses
  128-token chunks (the clash was SDPA's 256-token circular buffers). Decision: `l1_attention_max_rows` 4096 with
  `l1_chain_small_chunk_rows` 4096 (128-token chunks only in that band; 256 stays at 1024 to 2048 rows and in DRAM
  above 4096). `STAGE1_PORT` keeps 2048. Config test 52 passed. Final queue 4 re-validates the new default (encoder
  and head tests, tracked replay at 1 / 8 / 64, bench at 1 / 8 / 64 and 2 / 4 / 16 / 32, Tracy at B 8).
- 22:30: final queue 2 done (erf policy, L1 chain 2048 at the time): `bench_shipped_erf` 12.93 / 73.24 / 527.5 ms
  (77.3 / 109.2 / 121.3 rows per second, loads 5.6 to 5.9), B 2 / 4 / 16 / 32: 21.32 / 36.75 / 137.96 / 269.05 ms
  (93.8 / 108.8 / 116.0 / 118.9 rows per second). Tracy of layer 1 at B 64 and B 1 under the erf policy in
  `tracy/shipped_erf_layer1_b{64,1}s512`. Final queue 3: `gelu_separate_erf` (exact GELU as a separate unary op instead
  of the fused activation, L1 chain 4096) 12.92 / 21.33 / 37.14 / 66.23 / 529.7 ms at B 1 / 2 / 4 / 8 / 64, no gain
  over the fused erf, so the erf cost is the SFPU erf itself; the fused form stays.
- 22:35: final queue 4 done (`/home/hous/dev/laya/logs/p3_final_queue4_20261005T222955Z.log`): encoder and head tests
  17 passed on the final defaults (end to end markers 0.99992 / 0.055 at B 1, 0.99926 / 0.123 at B 8), tracked replay
  pass at 1 / 8 / 64, `bench_shipped_final` 12.96 / 65.81 / 528.2 ms (77.1 / 121.6 / 121.2 rows per second, load 7.5),
  `bench_shipped_final_b2b4b16b32` 21.01 ms at B 2 and the B 4 / 16 / 32 cells in the JSON, Tracy of layer 1 at B 8 in
  `tracy/shipped_final_layer1_b8s512`. README tables regenerated from every `bench_*.json` by
  `/home/hous/dev/laya/scratch/stage3_tables.py` and `/home/hous/dev/laya/scratch/stage3_shipped.py`. Devlock free, no
  T1 job running.
- 23:01 (review R1 responses): the stage 1 per-layer file had been overwritten by the shipped erf run of
  `s3_final_tests2` (test wrote a fixed name). Copied that run to `layer_pcc_shipped.json` here; `tests/test_ttnn_encoder.py`
  now writes `layer_pcc_<policy>_<port label>.json` (`LAYA_PORT=stage1` selects `STAGE1_PORT`); the stage 1 file is
  regenerated by job `s1_regen_layer_pcc` (queued under devlock behind Track T2). Scratch per-layer and PCC files of the
  policy probes copied to `policy_probe/`. README statements on the load range, the B 1 scatter and the erf cost at
  B 2 and B 4 corrected; stage 2 README re-validation section re-pointed at the 22:32 UTC JSON with the allocator
  warning explained.
