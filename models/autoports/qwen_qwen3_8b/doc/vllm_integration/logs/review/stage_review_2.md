# Stage Review 2 (fresh read-only subagent, HEAD bc74e93d1b, 16:07 to 16:30 ET Oct 30)

Verdict: clean-pass

Required Work: none. Every review-1 item verified against artifacts:
1. P1 context-end windows: tt/model.py prefill_hidden moves phys_base down; arithmetic checked; context_end_probe_L36_chip2_v4.json PCC min 0.9975 to 0.9997, program cache 461 flat; live server: the 40956-token prompt served as hits at 40000, 40928 and 36864 returns identical greedy tokens; tracked_served_final program cache 444 flat, no tracker error.
2. P2 served greedy vs control: generator controls on chips 0 and 1 token-identical to the stage-8 control; served greedy 128/128 on six prompts, three runs, chat included (final_server/served_determinism.log); async-off identical; cause (plugin start_pos lagging the device by one under overlap) and fix (host mirror of the device position, tt/generator.py) verified against the plugin's drain rules.
3. P2 seeded mixed-batch and logit reproducibility: logit_repro_check.json identical across runs and buckets; admission windows (3 + 29 all host passes; 1 + 31 with a device-sampled lone prefill fails) support the classification; README counts match the rerun logs.
4. P2 contract notes: regenerated; gate checker passes.

Other Concerns (closed after the review, see work log 10): work-log 8.7 drew an invalid conclusion from the probe's reset-per-case served-entry tokens; README carried numbers from the earlier chain (CI burst, prefix cache, ITL p99) and a stale sentence about the logit comparison and a host-capture count of three (five in the final chain); contract notes carried 461, the old tracked directory, probe v1 and "pending the device reset"; the primary benchmark ran while chip 0 ran tracked tests on the same host (note added); device and host samplers disagree at temperature 0.01 for the same prompt and seed (stage-10 probe suggested); Qwen3Generator.generate does not advance the position mirror (bucket now invalidated at the end of generate); the appendix edit was uncommitted.

Hard-Check Gaps: no in-log proof that tracking was active in the final tracked server (evidence is the launch line, the raises on earlier code with the same wrapper and the positive control); host-mode deferred captures for buckets 24 and 32 observed only untracked; shifted windows with whole lead tiles plus an unaligned remainder untested at the logit level (unreachable under this plugin configuration); the 27-token decode to position 40959 has coherence only; stale-token coverage through the adapter remains indirect; vllm_run_summary.json has server_command null (attached stages).

Residual Risk: async overlap correctness rests on the plugin draining before every reload; device sampler behaviour at very low temperature unexamined; supports_device_penalties False (stage 10); deprecated-fork serving stack; headline TTFT measured with chip 0 busy on the same host.
