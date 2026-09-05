# AutoFix: French qualitative branch and terminal policy

The final BF4/LoFi full-model run labels Bonjour informal at generated token 34.
Pinned CPU HF conditioned on the exact same TT prefix instead continues with
formal/greeting; the TT token `inform` (39102) has conditional HF rank 7.
See `qualitative_final_v1/french_branch_control.json` and its CPU source/control.
The previous BF16/HiFi4 full-model qualitative completion had no such error.

Hypothesis: final-head quantization or fidelity changes the greedy branch.
The experiment `probe_french_head.py` first reproduces the original token using
all 32 optimized layers and teacher-forced exact TT prefix through traced decode,
then saves the final hidden tensor from every rank. Reduced terminal candidates
reuse that identical hidden input with the actual full-stack 221952-byte/bank L1
residency: BF4 LoFi/HiFi2/HiFi4; BF8 LoFi/HiFi2; and the prior BF16/HiFi4 control
as needed. No decoder policy, state, cache or collective is changed.

The terminal probe records top ten scores/token text, rank of the known wrong
branch, exact eager/trace agreement, and warmed terminal trace latency. A winning
candidate must then produce a reviewed complete French-prompt generation from
the original chat prompt; a local score change alone is not a qualitative pass.
## Localization result: local terminal defect refuted

`french_head_v1/capture_bf4_lofi.json` reproduces token 39102 at generated
index 34 through all 32 layers with the exact TT prefix forced. The saved
`hidden.pt` is [1,1,4096] on every rank, with all four replicas equal.
Every terminal candidate on this same hidden still chooses `inform`. The
CPU FP32 norm plus unquantized FP32 head also chooses `inform` (23.0049)
over `form` (22.5874); see `cpu_fp32_oracle.json` and
`french_hidden_cpu_oracle.py`. Thus neither head quantization nor the terminal
kernel alone explains the wrong branch at this already-reached TT hidden.
The exact-prefix HF control exposes accumulated hidden-state divergence, and
higher precision cannot retroactively change earlier greedy choices.

## Full-prompt controls and selection

Each full French completion below uses the original 24-token chat prompt,
128-token greedy budget, and all unchanged optimized decoder layers. The
completions were read directly; process success is not a qualitative pass.
`french_head_v1/qualitative_review.json` records the explicit verdicts. Original
immutable reports containing `pass: true` mean probe execution only; subsequent
reports name this field `probe_execution_pass` and leave `qualitative_correct`
unset until review.

| Head policy | Terminal trace ms on exact hidden | Full French qualitative verdict |
| --- | ---: | --- |
| BFP4/LoFi | 0.482021 | Fails: Bonjour labeled informal |
| BFP4/HiFi2 | 0.560522 | Same failure |
| BFP8/LoFi | 0.811986 | Same failure |
| BFP8/HiFi2 | 0.817781 | Same failure |
| BFP4/HiFi4 | 0.948137 | Same failure |
| BF16/HiFi4 | 1.858692 | Pass: earlier branch says Bonjour formal/greeting, Salut informal |

BF16/HiFi4 reproduces the earlier correct full-model qualitative completion.
Its formal/informal full translations are correct, reasoning remains coherent,
and no repetition or wrong-language drift appears. The 128-token budget ends
mid-reasoning, as in the HF control; this is not a completed-answer claim.
Selection is based on the entire free-running branch, **not** a claim that this
policy changes the already-wrong hidden tensor's next-token choice.

## Geometry at selected BF16/HiFi4 precision

The reader packetization repair makes new two-reader geometries available.
`french_head_geometry_v1` reuses the exact saved hidden and retains full-stack
221952 L1 bytes/bank. It saves all logical logits for exact comparison to the
previous 8192-column, K4, one-reader BF16 baseline.

| Local columns / K block / readers | Terminal trace ms | Result |
| --- | ---: | --- |
| 8192 / 4 / 1 | 1.858810 | Qualifying precision baseline |
| 16384 / 4 / 2 | 1.453912 | All logical logits bit-identical to baseline; selected |
| 32768 / 1 / 2 | 1.403517 | Fits but changes logits; full French behavior unvalidated |
| 32768 / 2 / 2 | — | Static end 1299456 overlaps live allocation frontier 1244416 by 55040 bytes |
| 32768 / 4 / 2 | — | Static end 2094080 exceeds physical L1 limit 1572864 bytes |

The selected geometry preserves the qualifying K4 accumulation and removes
half of its terminal matmul chunks. It is about 22% faster in this isolated
terminal trace measurement; parent final full-model performance gates remain
required. K1 is a legal faster but numerically different candidate, not a
physical rejection or proof of full-model correctness failure.

Required default configuration for parent integration: `lm_head_dtype=bfloat16`,
`lm_head_fidelity=HiFi4`, `lm_head_columns=16384`, `lm_head_block_w=4`,
`lm_head_readers=2`. This agent has not edited runtime defaults.

Immutable commands, environment, source hashes, binary hashes and exit status
are recorded in `logs/french_head_capture_v1.provenance.json`, the
`logs/french_terminal_*` archives, `logs/french_full_*` archives, and
`logs/french_geom_*` archives. The two rejected allocation probes exit 1 as
expected and close devices normally. No reset or device recovery was needed.
No profiler and watcher were combined; terminal timings use plain device trace.

Final status: local terminal fault hypothesis refuted; qualifying full-prompt
head policy and an exact-preserving faster geometry identified. Hardware has
been returned to the parent for default integration and final all-model gates.


## Selected geometry follow-up

The parent tested `bf16_hifi4_n32k_k1_r2` through full French generation
(`logs/french_bf16_n32k_k1_final.provenance.json`, exit0). The actual output
correctly identifies Bonjour as formal/greeting and Salut as informal, then
gives appropriate formal/informal sentence translations. This qualifies the
fastest legal measured BF16 geometry,1.4035 ms terminal-only. It is promoted
to the model default pending final all-layer gates. This selection changes an
earlier free-running branch; the frozen-hidden local ranking remains the
refuted terminal-defect control described above. K2/K4 do not fit the measured
full-stack L1 reservation.
