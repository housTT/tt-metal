# Work log: stage 1 (functional decoder), Track T1

All times UTC, 2026 Oct 5. Host qb2-120-p11t01, chip 0 (`TT_METAL_VISIBLE_DEVICES=0`). Every device command ran
through `/home/hous/dev/laya/bin/devlock`; `source /home/hous/dev/laya/bin/ttenv.sh` provides the environment.
Background jobs: `/home/hous/dev/laya/scratch/t1_launch.sh <job> <command>` (log `/home/hous/dev/laya/logs/p3_<job>_<stamp>.log`,
PID in `/home/hous/dev/laya/state/jobs/<job>.pid`, last line `DONE <job>` or `FAIL <job> <reason>`).

- 20:58: P0 mesh smoke had failed on every chip ("Device 0: Timed out while waiting for active ethernet core 29-25 to
  become active again", `/home/hous/dev/laya/logs/p0_mesh_smoke_diag.log`). I queued a reset job under devlock, then
  cancelled it by PID (it never held the lock) because the orchestrator was already running the reset
  (`/home/hous/dev/laya/logs/p0_reset_smoke_20261005T205841Z.log`). The board reset fixed it; chips 0 to 3 and the 1x4
  mesh opened (`mesh_smoke.done` at 21:00).
- 20:59 to 21:15: CPU work. Read the upstream port and README, inspected `model.safetensors` (206 tensors, float16 on
  disk, not bfloat16 as PLAN.md section 1 says; 170 `encoder.*`, 24 `head.*`, 6 `scorer.*`, 4 `act_head.*`,
  `type_emb.weight`, `temperature`). Wrote `tt/model_config.py` (PrecisionPolicy table of Appendix A.7, PortConfig,
  Blackhole program-config rules, bucket helpers), `tt/weights.py` (key map, Q-scale fold for `Wqkv` and the head
  `in_proj` weight and bias, padded GeGLU blocks, head and scorer upload), the runtime mask builder, rope with the
  per-core shard bound, attention, MLP, layer, encoder, head (`tt/laya_head.py`) and the device model
  (`tt/laya_model.py`). `tests/test_model_config.py` and `tests/test_weights.py` (no device): 54 passed in 20 s.
- 21:16: first device job `s1_components` (`/home/hous/dev/laya/logs/p3_s1_components_20261005T211629Z.log`): rope
  (8 cases) and masks passed; embeddings, MLP, attention and layer errored at fixture setup (tt-metal's `device`
  fixture is function scoped and my module-scoped weight fixtures requested it: `ScopeMismatch`). Fixed with a
  module-scoped alias of `_device_module_impl` in `tests/conftest.py`.
- 21:18: second device job `s1_devtests` over all eight device test files
  (`/home/hous/dev/laya/logs/p3_s1_devtests_20261005T211822Z.log`, PCC rows in `/home/hous/dev/laya/scratch/pcc_stage1_run2.json`).
- 21:22: `s1_devtests` finished: 54 passed, 5 failed in 221 s. Passed: embeddings (B 1 and 8), rope (8 cases, PCC
  0.99999), masks (3 shapes plus the pad_row rewrite check), MLP (layers 0 and 16, B 1 interleaved and B 2 sharded,
  padded widths), attention (full and sliding on a padded B 2 batch) with five controls, layers 0, 1, 16, 27 at B 1 and
  B 2, encoder at (1,512), (2,512), (4,512), (1,1024), head layers, scorer, end to end B 1. Failed: encoder and end to
  end at B 8 (`program.cpp:1932` circular buffers clash with L1 buffers: the L1 attention chain does not fit at 4096
  rows next to the 8x8 matmul CBs), encoder vs bf16 reference 0.9838 (the bf16 CPU model is a poor yardstick with
  27000-magnitude outliers; made informational), and the two head controls ReLU->GELU and head pad mask dropped (PCC
  0.99999 and 0.99989 on the hidden state: outlier channels hide them).
- 21:24: `l1_attention_max_rows` 8192 -> 2048. Encoder test gained an interleaved B 2 variant, a "fill row only" B 1
  variant and per-row PCC; head controls moved to scorer logits; torch limited to 6 threads (orchestrator notice: host
  load 47, no performance numbers until the 1-minute load is under 8).
- 21:26: `s1_encoder_head` (`/home/hous/dev/laya/logs/p3_s1_encoder_head_20261005T212633Z.log`): 15 passed, 2 failed
  (the two head controls, now 0.99993 and 0.99971 on scorer logits). Per-row data: the 165-token row scores 0.9973 in
  every batch; the synthetic 512-token fill row scores 0.9920 alone and drags B 2 and B 4 to 0.9949 and 0.9920;
  interleaved B 2 equals sharded B 2 (0.9950 vs 0.9949); B 8 with the DRAM chain 0.9928; one real 127-token row at B 8
  is at 0.908 after layer 19 and 0.9896 after the final norm. Loss is content dependent (massive-activation channels
  379, 382, 963, 195 from layer 19), not placement dependent.
- 21:27: `s1_watcher` (`TT_METAL_WATCHER=10`, `/home/hous/dev/laya/logs/p3_s1_watcher_20261005T212635Z.log`): 10 passed in 37 s;
  watcher log copied to `watcher_layer_run.log`.
- 21:30: CPU sensitivity of the plan's two head controls on the fp32 reference itself: ReLU->GELU moves the scorer
  logits by at most 0.0136 (PCC 0.999943), dropping the head pad mask by 0.0091 (PCC 0.999844); Q-scale fold removed
  0.999085; head biases dropped 0.9973; Q/K swapped 0.9919. Detectable: head layers skipped 0.9048, wrong question type
  0.7606, type_emb dropped 0.6188. The two weak controls are kept as recorded measurements; "head layers skipped" and
  "wrong question type" are the gated head controls. Head test rerun `s1_head_rerun` launched; Tracy profile of layer 1
  at B 8 queued behind it with an in-job wait for a 1-minute load under 8.
- 21:32: `s1_head_rerun` (`/home/hous/dev/laya/logs/p3_s1_head_rerun_20261005T213121Z.log`): 9 passed in 38 s; head
  layers skipped 0.9048 (reference 0.9048), wrong question type 0.7602 (reference 0.7606); end to end marker PCC 0.9942
  (B 1) and 0.9921 (B 8), max abs 0.344 and 0.379 logits.
- 21:33: the Tracy job had been waiting for the load drop while holding devlock; killed it by PID (flock 1045309 and its
  children) and relaunched with the load wait outside the lock (`/home/hous/dev/laya/scratch/t1_tracy_layer.sh`,
  waiter pid in `state/jobs/s1_tracy_layer.pid`, log `p3_s1_tracy_layer_20261005T213312Z.log`). Rows merged into
  `pcc_rows.json` (59 rows with their source logs); README results written from it by
  `/home/hous/dev/laya/scratch/stage1_tables.py`.
- 21:34: Stage 1 README complete except the Tracy row, which waits for a 1-minute load under 8 (orchestrator rule).
  Decision: start the stage 2 correctness runs now (trace safety does not need a quiet host; its timing columns will be
  labelled with the load and re-taken under a quiet host) rather than leave the device idle. `s2_replay_check` queued
  with `TT_METAL_TRACE_ALLOC_TRACKING=1` on buckets 1x512 and 8x512.
