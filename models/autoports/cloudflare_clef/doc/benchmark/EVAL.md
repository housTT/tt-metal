# Clef on Tenstorrent p150x2: evaluation (stage 5)

Generated 2026-10-05 23:38:43 UTC by `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/scripts/summarize_eval.py` from the outputs of `scripts/run_eval.sh` under `/home/hous/dev/clef/reports/eval`. Machine-readable copy: `/home/hous/dev/clef/reports/final_numbers.json`. The serving performance table of stage 3 is in `README.md` next to this file. Times in this file are UTC (the host clock); ET is UTC-4.

Acronyms: TT (Tenstorrent), TP (tensor parallel), CPU (host processor), dp (absolute difference of an option probability against the CPU reference), pp (percentage points), ECE (expected calibration error), p50 and p95 (percentiles), macro-F1 (unweighted mean of the per-class F1 scores).

## Method

1. Public benchmarks from Clef's model card, rendered once to SystemOne requests by `scripts/render_public_evals.py` (`/home/hous/dev/clef/evals/*.jsonl`, revisions in `/home/hous/dev/clef/evals/MANIFEST.json`; no prompt tuning, no option dropping), sent to the TT server (`POST /v1/systemone`) by `scripts/eval_remote.py` and scored by `scripts/eval_metrics.py` (accuracy; macro-F1 over the 77 intents for BANKING77). The card numbers come from the non-public Decision Index requests, so they are indicative only; the control that separates the rendering from the device is the CPU bf16 run of the author's own code on a stratified 100-item sample of each benchmark (`/home/hous/dev/clef/evals/*_sample100.ref_bf16.jsonl`, `/home/hous/dev/clef/reports/eval_cpu/`). The TT server runs the same 100 items; `scripts/parity_compare.py` reports max dp, mean dp and argmax flips at reference margin >= 0.05 between the two, and a TT-versus-CPU gap above 1.0 pp on any benchmark is a finding.
2. Kev suites (`/home/hous/dev/kev/kev/evals/{hard-v1,devtools-v1,documents-v1}/test.jsonl`) through `kev.benchmark --remote` unchanged (`clean` block: accuracy, Brier, ECE over the clean knowable rows), which gives a second labelled corpus and the Clef-versus-Kev-on-the-same-silicon row.
3. Parity against the author's reference on the stage 1 and 2 sets over HTTP (`run_eval.sh --steps parity`): the 16 text and 8 image reference records, dev64 text and dev16 image, against the CPU bf16 rows (and fp32 where it exists).

## Model-card table

| Benchmark | Metric | TT, full test set | scored / records | TT, 100-item sample | CPU bf16, same 100 items | TT minus CPU (pp) | Model card | TT vs CPU on the sample: max dp / mean dp / flips at margin 0.05 | TT latency_ms p50 / p95 (full set) | Wall time (full set) |
|---|---|---|---|---|---|---|---|---|---|---|
| ARC-Challenge test | accuracy | 97.8 | 1172/1172 | 96.0 | 95.0 | +1.00 | 97.7 | 0.3112 / 0.0064 / 1 (0 near-tie) | 232 / 240 | 5 min |
| BANKING77 test | macro-F1 | 94.4 | 3080/3080 | 91.1 | 91.1 | +0.00 | 94.2 | 0.1101 / 0.0055 / 0 (2 near-tie) | 839 / 870 | 43 min |
| New Yorker caption matching test (images) | accuracy | 58.3 | 528/528 | 60.0 | 60.0 | +0.00 | 69.5 | 0.2124 / 0.0312 / 3 (1 near-tie) | 575 / 962 | 5 min |

Card values: ARC-Challenge 97.7, BANKING77 94.2, New Yorker 69.5 (Clef model card, non-public requests). The 100-item samples carry a binomial 95 percent interval of about plus or minus 4 points on ARC and plus or minus 10 on New Yorker, so the sample columns compare the TT server to the CPU control, not to the card. `latency_ms` is the server's model time per request (prefix-cache miss or hit, schema continuation, head on the host); the wall time is the client run at the concurrency of the summary file and includes the HTTP and encode time.

## Kev suites (test splits, clean knowable rows)

| Suite / test | records | questions (clean knowable) | Clef TT acc | Brier | ECE | coverage | client latency p50 / p95 ms | Kev-9B on one P150 (stage 6): acc / ECE |
|---|---|---|---|---|---|---|---|---|
| hard-v1 | 700 | 1088 | 0.774 | 0.321 | 0.044 | 700 evaluated, 0 rejected, 0 truncated | 1608 / 4243 | 0.826 / 0.053 |
| devtools-v1 | 900 | 1073 | 0.741 | 0.399 | 0.143 | 900 evaluated, 0 rejected, 0 truncated | 1464 / 2126 | 0.787 / 0.101 (n=1,071) |
| documents-v1 | 574 | 936 | 0.893 | 0.169 | 0.044 | 574 evaluated, 0 rejected, 0 truncated | 2309 / 3164 | 0.896 / 0.015 |

The Kev column is `jaredpalmer/kev-9b` served by its own TT engine on this box (`/home/hous/dev/kev/reports/final_numbers.json`, `eval` block, test splits). Where the Kev cell carries an `n`, Kev's clean question count differs from Clef's: Kev's devtools-v1 test number is over n=1,071 questions after two audited drops (`drop_ids` in that file: `codereviewer/cls-test/13657` and `19245`), Clef's over all 1,073; the difference moves accuracy by at most 0.2 pp. Clef and Kev are different models with different training data; the row shows two decision models on the same silicon and the same labelled requests, not a ranking of the ports.

## Parity against the author's reference over HTTP

| Set | CPU reference | questions | max dp | mean dp | argmax flips | flips at margin 0.05 | near-tie flips | accuracy delta pp | ECE shift | stage 1 bars (max dp <= 0.10, 0 flips at margin) |
|---|---|---|---|---|---|---|---|---|---|---|
| reference_text | bf16 | 27 | 0.0722 | 0.0109 | 0 | 0 | 0 | +0.00 | +0.0075 | met |
| reference_text | fp32 | 27 | 0.0759 | 0.0104 | 0 | 0 | 0 | +0.00 | +0.0089 | met |
| reference_image | bf16 | 8 | 0.0463 | 0.0200 | 0 | 0 | 0 | +0.00 | +0.0073 | met |
| reference_image | fp32 | 8 | 0.0383 | 0.0189 | 0 | 0 | 0 | +0.00 | -0.0017 | met |
| dev64_text | bf16 | 95 | 0.1923 | 0.0124 | 1 | 1 | 0 | -1.15 | -0.0105 | not met |
| dev16_image | bf16 | 16 | 0.1754 | 0.0318 | 0 | 0 | 0 | +0.00 | -0.0571 | not met |

## Coverage and findings

- Coverage complete (0 rejected, 0 truncated, 0 missing, 0 unanswered on every file): yes.
- Finding: parity dev64_text: outside the stage 1 bars (max dp 0.1923, flips at margin 1); flips at margin: `hard-v1/temporal_numeric/development/00008` `refund` b to c (reference margin 0.1679, dp 0.1028, label b); largest dp: `hard-v1/tradeoff/development/00013` `choice` dp 0.1923 (reference wildfern_agency, TT wildfern_agency, reference margin 0.0833, label paper_kite)
- Finding: parity dev16_image: outside the stage 1 bars (max dp 0.1754, flips at margin 0); largest dp: `7b7ef38338fffa28b9106027c882c1b4` `caption` dp 0.1754 (reference A, TT A, reference margin 0.3550, label A)
- Note: ARC-Challenge test: TT sample accuracy differs from the CPU control by exactly +1.00 pp, which is not above the 1.0 pp finding threshold; flips at margin: `Mercury_7177398` `answer` A to D (reference margin 0.3652, dp 0.3112, label D)

## Findings investigated (device agent, 2026 Oct 05, 19:08 to 19:25 ET; per-layer probe added by the stage 5 remediation, 19:29 to 19:35 ET)

This section is `EVAL_findings.md` next to this file; `summarize_eval.py` appends it to `EVAL.md` on every `--write`, so a regeneration keeps it. Rows: `/home/hous/dev/clef/reports/eval/metrics/parity_*.json` (`flips`, `per_question`); the CPU fp32 control of this section: `/home/hous/dev/clef/reports/eval/findings_records.jsonl` (3 records) through `scripts/cpu_reference.py --dtype float32 --threads 8` to `/home/hous/dev/clef/reports/eval/findings_records.ref_fp32.jsonl` (log `/home/hous/dev/clef/logs/stage5_findings_fp32_control.log`, 1 min 35 s, 160.8 GB peak RSS). Stage 4 ledger: `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/doc/datatype_sweep/README.md`, "Anomaly ledger entry 1", "Anomaly ledger entry 2" (the per-layer probe of the three records of this section, run after the stage 5 review) and "The four baseline sweep200 flips at margin (for stage 5)".

Which bars apply where. The stage 1 gate as amended (plan, Stage 1, gate (d)) puts the 0-flips-at-margin rule on the 16 reference records, the 100 BANKING77 sample rows and the 8 long records, and the max dp <= 0.10 rule on the 16 reference records; max dp and near-tie flips on every other set are reported, not gated. The last column of the parity table applies both rules to every set as one uniform check. Three of the four sets the stage 1 gate names pass on the shipped server over HTTP in this stage: reference_text max dp 0.0722, 0 flips against bf16 (0.0759, 0 against fp32), reference_image 0.0463, 0 (0.0383, 0), BANKING77 sample 0 flips at margin (2 near-tie flips at reference margins 0.0437 and 0.0209). The fourth, the 8 long records (2,000 to 16,384 tokens), did not go over HTTP in this stage: `run_eval.sh --steps parity` sends records_text, records_image, dev64 and dev16 only. Their evidence is the stage 1 engine-level run of all 8 (`/home/hous/dev/clef/logs/stage1r_parity_long_records_*.log`, 0 flips at margin) and the stage 3 remediation's HTTP run of the longest one, `long/max` at 16,384 tokens, on the shipped server (`/home/hous/dev/clef/logs/stage3r_parity_shipped_long_max.log`, two passes, max dp 0.0181, 0 flips; `README.md` next to this file, "Long state over HTTP"). dev64 and dev16 are the stage 5 parity sets; the stage 5 gate text says "all parity rows within the stage 1 bars", so their two rows marked `not met` are findings and are investigated below, one record at a time.

### 1. ARC-Challenge sample, +1.00 pp: exactly at the threshold, one record, device-side

The TT sample is 96 of 100 against the CPU control's 95 of 100. The rule is a gap above 1.0 pp; this gap is exactly 1.0 pp (one question), so it is not a finding under the rule (the first summary printed it as one because `100 * (0.96 - 0.95)` evaluates to `1.0000000000000009`; `summarize_eval.py` now compares the rounded gap and prints an exact-threshold gap as a note). It is still the one argmax flip of the sample and the largest dp of the ARC sample (0.3112; the other 99 questions have max dp 0.0961 and the sample mean dp is 0.0064), so it is investigated.

`Mercury_7177398` (`answer`, label D). State: "The particles within a substance are in constant motion. The particles in which of these substances have the lowest amount of kinetic energy?"; options A wax of a candle at room temperature, B water in a glass of ice water, C steam from a cup of coffee, D ice cube in a glass of tea. CPU bf16: A 0.6505, D 0.2853 (margin 0.3652). CPU fp32 (this control): A 0.6804, D 0.2581 (bf16 against fp32 dp 0.0299, same argmax). TT: D 0.5957, A 0.3393, identical in the full-set run and the sample run (bit-equal rows). Both CPU precisions agree with each other and disagree with the device, so the disagreement is device-side, and the device happens to land on the label. This is the signature of the stage 4 ledger entry 1 (a confident reference, a large device dp, both CPU precisions in agreement): the two entry 1 records are `hard-v1` records, so this record extends the class to a public benchmark. The per-layer probe (`scripts/sweep_anomaly_probe.py`, run after the stage 5 review; ledger entry 2, report `/home/hous/dev/clef/reports/stage5r_anomaly_probe_selected.json`) confirms the classification on this record: the TT head reproduces the served row (D 0.5957, A 0.3393, dp 0.0) and the HF bf16 head reproduces the CPU reference (dp 0.0004); every layer is inside the stage 1 band (teacher-forced delta PCC mean of means GDN 0.999787 and attention 0.999767 against the band's 0.999785 and 0.999761, worst layer 12 at 0.999507, no layer below 0.999, worst row the layer-12 row-0 `<|im_start|>` artifact at 0.952 that stage 1 measured); and the splice curve is not monotonic (A is 0.339 with every layer on the device, 0.534 with HF layer 0 alone, 0.401 with HF layers 0 to 1, back to D at the prefixes 7 to 10, 17 and 19, converging to 0.65 from about 47 HF layers). Under `QWEN36_MATMUL_FIDELITY=HiFi4` the TT argmax is A (0.5082 against D 0.4254) with the per-layer means moved by at most 1e-5, the entry 1 behaviour of a near-boundary record. Rate on this benchmark: 1 flip in 100 questions; the full-set TT accuracy (97.78, 1,146 of 1,172) sits at the card number (97.7). No CPU run of the full 1,172 exists (about 74 min of host CPU at 3.8 s per record); the control is the 100-item sample.

### 2. dev64_text: one flip at margin and one max dp above 0.10, both new instances of the ledger class

The first 16 records of dev64 are the 16 reference records and reproduce stage 1 (27 questions, max dp 0.0722, 0 flips). The 48 new records (68 questions, first run on the device in this stage; dev64 and sweep200 are disjoint by construction, so the stage 4 ledger could not contain them) carry the two questions above 0.10:

| Record, question | CPU bf16 | CPU fp32 (this control) | TT (server) | dp TT vs bf16 / vs fp32 | bf16 vs fp32 | label |
|---|---|---|---|---|---|---|
| `hard-v1/temporal_numeric/development/00008`, `refund` (314 tokens) | b 0.4639, c 0.2960 (margin 0.1679) | b 0.4594, c 0.3039 | c 0.3988, b 0.3770 | 0.1028 / 0.0949 | 0.0079 | b (TT wrong) |
| `hard-v1/tradeoff/development/00013`, `choice` (530 tokens) | wildfern_agency 0.5389, paper_kite 0.4556 (margin 0.0833) | wildfern_agency 0.5725, paper_kite 0.4217 | wildfern_agency 0.7302, paper_kite 0.2633 | 0.1923 / 0.1584 | 0.0339 | paper_kite (all three wrong) |

The `meets` question of `00013` matches to 0.0005. On both records the fp32 and bf16 references agree with each other (dp 0.008 and 0.034) and the device differs from both by 0.09 to 0.19, so the disagreement is device-side, as for the two ledger records (`hard-v1/multi_hop/development/00017 paged` dp 0.33, `hard-v1/probability/development/00084 value` dp 0.47). The family pattern of the ledger repeats: on the 48 new dev64 records the mean dp per source family is `hard-v1/multi_hop` 0.047, `hard-v1/temporal_numeric` 0.043, `hard-v1/tradeoff` 0.030, `hard-v1/ambiguous` 0.026, against 0.001 to 0.013 for `cfpb` (23 questions), `commitpackft`, `when2call`, `aegis`, `flakeflagger`, `codereviewer`, `prompt_injection`, `hard-v1/judge`, `hard-v1/long_policy`, `hard-v1/probability`; the ledger measured 0.039 to 0.051 for the `hard-v1` reasoning families against 0.002 to 0.029 for the others on sweep200. Argmax agreement with the CPU bf16 reference: 94 of 95 on dev64 (98.9 percent) against 274 of 278 on sweep200 (98.6 percent); label accuracy 63 of 87 (72.4 percent) against the reference's 64 of 87 (73.6 percent), the one flip. ECE shift -0.0105. Classification: two more instances of the ledger entry 1 class (precision sensitivity of near-decision `hard-v1` reasoning records, device-side, no knob in the stage 4 sweep brought the class into band), established on these two records by the per-layer probe of the stage 5 remediation (ledger entry 2 in `doc/datatype_sweep/README.md`; reports `/home/hous/dev/clef/reports/stage5r_anomaly_probe_{selected,hifi4}.json`). Anchors: the TT head equals the served rows (dp 0.0 on every question) and the HF bf16 head equals the CPU reference within 0.0018 and 0.0002. Every layer inside the stage 1 band on both records (GDN means 0.999781 and 0.999777, attention 0.999766 and 0.999754, worst layer 16 at 0.999520 and 0.999524, no layer below 0.999). Splice curves not monotonic: on `00008` HF layer 0 alone restores the reference argmax b (0.4015 against c 0.3613) and the curve then stays between 0.41 and 0.47; on `00013` the probability of wildfern_agency wanders between 0.49 and 0.75 over the prefixes (paper_kite at the prefixes 24 and 26 only) and converges to the reference's 0.54 from about 46 HF layers. Under HiFi4 both records land on the reference argmax (b 0.4318; wildfern_agency 0.6074) with the per-layer means unchanged to 1e-5.

### 3. dev16_image: max dp 0.1754 on `7b7ef383`, the stage 2 and 4 known record, unchanged

`7b7ef38338fffa28b9106027c882c1b4` `caption`: reference A, TT A, label A, reference margin 0.3550, dp 0.1754, no flip; the other 15 dev16 questions have max dp 0.0517. The server's 16 dev16 rows equal the stage 4 sweep's dev16 rows (`/home/hous/dev/clef/reports/sweep/baseline_stage1/dev16_image.jsonl`) within the 4-decimal API rounding (max dp 5e-5), and the stage 4 vision control table reports the same 0.1754 for this record under the shipped tower (0.1024 under the upstream tower precision, with 0 flips either way and dev16 accuracy 10 of 16 on both sides). It is the stage 2 review carry-over 4 (the vision block-0 per-channel offset), listed as open in `doc/datatype_sweep/README.md`; stage 5 adds nothing new to it and does not close it.

### 4. New Yorker: 58.3 on the full test set against the card's 69.5 is the rendering, not the device

On the 100-item sample the TT server and the CPU bf16 control both score 60.0 (gap 0.00): the four argmax flips split two each way (`ee0b64ab` and `7e59b11c` TT right with dp 0.1178 and 0.2124 at margins 0.0798 and 0.2858; `3444cff0` TT wrong, dp 0.0895 at margin 0.0763; `7152b437` a near-tie flip at margin 0.0146, TT wrong), mean dp 0.0312, ECE 0.127 against 0.159. The author's own code on the host gives 60 of 100 under this rendering, so the 9 to 11 point gap between the full-set TT number (308 of 528) and the card is the difference between this rendering (the cartoon image and the five captions only, no dataset description or entity fields, one fixed prompt, per the one-fixed-rendering rule; `/home/hous/dev/clef/reports/reference/README.md`, "Image ablation") and the non-public Decision Index requests behind the card number. The 100-item sample's binomial 95 percent interval is about plus or minus 10 points, the dev16 set is 10 of 16 on both sides, and the 8-record reference set (2 of 8) was a low draw. The `7e59b11c6f8b5f519c89d9ca3ad02ec3` flip (dp 0.2124 at a 0.2858 margin, TT right) is a fourth large-dp record of the stage 5 run and the only one without a control: image records have no fp32 reference apart from the 8 reference images, and `scripts/sweep_anomaly_probe.py` has no path for it (it encodes text only, computes the HF hidden states with the text model alone and opens the engine with `vision=False`; `tt/encode.py` `encode` without a processor raises on a record with images). "Device-side" is therefore not established for this record, only that the TT server and the CPU bf16 run disagree. It stays open (item 2 below).

### 5. Kev suites against Kev-9B on the same silicon

Clef is 5.2 pp below Kev-9B on hard-v1 (0.774 against 0.826), 4.6 pp on devtools-v1 (0.741 against 0.787) and 0.3 pp on documents-v1 (0.893 against 0.896); ECE 0.044 against 0.053, 0.143 against 0.101, 0.044 against 0.015. The suites are Kev's own evaluation sets and Clef is a different model with different training data, so this is not a device finding and there is no CPU control on the test splits; the nearest control is stage 4's sweep200 development subset of the same three suites, where the device is within 0.36 pp of the CPU reference (274 of 278 argmaxes). Coverage is complete (0 rejected, 0 truncated, no over-length rows).

### Gate statement

Coverage complete on every file: yes (0 rejected, 0 truncated, 0 missing, 0 unanswered, 0 non-200 responses in the server log). Parity: the 16 text reference records, the 8 image reference records and the BANKING77 sample are within their stage 1 bars on the shipped server over HTTP; the 8 long records have their stage 1 engine-level result and the stage 3 remediation's HTTP run of `long/max` (above) and were not re-sent in this stage. The two stage 5 sets outside the uniform bars are explained record by record above; the text records are confirmed members of the stage 4 ledger class by the per-layer probe (ledger entry 2) and the image record is the stage 2 carry-over record. Benchmarks recorded with the CPU sample controls: yes; one TT-versus-CPU gap at exactly the 1.0 pp trigger (ARC, one record, device-side, TT on the label, probed), none above it.

### Stated limitation (for stage 6)

Confident decisions move on the device at a rate of about 1 in 100: 1 of 100 questions on the ARC-Challenge sample (dp 0.31, TT on the label), 1 of 95 on dev64 (dp 0.10, TT off the label), 4 of 278 on the stage 4 sweep200 development set (dp up to 0.47). On every probed record (two in stage 4, three here) every layer is inside the stage 1 band and the splice curve is non-monotonic, so this is accumulated bf16 and bfp8 precision noise read by the head on near-boundary records, not an op fault; the lever that remains is the fused `chunk_gated_delta_rule` kernel's internal precision (stage 1 open item 1, a ttnn hand-off). MLP HiFi4 moves the probed records toward the reference (all three stage 5 records and one of the two stage 4 records land on the reference argmax under it) without changing any layer's standing against the band; the stage 4 selection rule excluded it (one ref16 flip at margin; also 17 percent slower per record) and this stage does not reopen the selection.

### Open items from this stage

1. Closed by the stage 5 remediation: `hard-v1/temporal_numeric/development/00008 refund`, `hard-v1/tradeoff/development/00013 choice` (dev64) and `Mercury_7177398 answer` (ARC sample) went through `scripts/sweep_anomaly_probe.py` at the selected precision and HiFi4 (device 4 min 49 s, `/home/hous/dev/clef/logs/stage5r_anomaly_batch.log`); every layer in band on all three, classification confirmed (ledger entry 2).
2. Open: New Yorker sample `7e59b11c6f8b5f519c89d9ca3ad02ec3 caption` flips E (CPU bf16, margin 0.2858) to B (TT, on the label) with dp 0.2124. No fp32 control and no probe path exist for image records (section 4). The accuracy-level control holds (TT and CPU both 60 of 100, flips 2 each way). A probe for it needs the vision tower in the probe and HF hidden states from the multimodal forward; not done here.
3. `7b7ef383` (dev16) stays the stage 2 carry-over 4; unchanged at dp 0.1754.
4. No CPU run of a full public test set exists; the controls are the three 100-item samples (ARC plus or minus 4 points, New Yorker plus or minus 10).
5. The parity table's last column applies the 0.10 max-dp rule to every set, which is stricter than stage 1 gate (d); the text above states which sets the gate names.
6. The 8 long records were not sent over HTTP in this stage (`run_eval.sh --steps parity` does not include them); their HTTP evidence is the stage 3 remediation's `long/max` run.

## Wall-time estimate (made before the run, from the stage 1 and 3 latencies)

| Step | records | s per request | wall | basis |
|---|---|---|---|---|
| ARC-Challenge test | 1172 | 0.20 to 0.35 | 4 min to 7 min | 167 to 274 tokens: bucket 256, stage 3 server 395 ms median on the 16 text records |
| BANKING77 test | 3080 | 0.70 to 1.00 | 36 min to 51 min | 1,824 to 1,871 tokens (77-option schema): 1024 + 512 + 256 buckets; stage 1 uncontended 1,774-token record 0.69 s device |
| New Yorker matching test | 528 | 0.45 to 0.70 | 4 min to 6 min | 307 to 791 tokens with one image: stage 3 server 530 ms median on the 8 image records |
| three 100-item samples | 300 | 0.45 to 0.70 | 2 min to 4 min | one third each of the three rows above |
| Kev suites test (hard-v1, devtools-v1, documents-v1) | 2174 | 0.25 to 0.55 | 9 min to 20 min | 145 to 2,672 tokens: sweep200 development records of the same suites, stage 1 0.19 to 0.69 s device |
| server start and warmup |  |  | 5 min | stage 3 server: engine load 103 to 126 s, traces, warmup request |
| total | 7254 |  | 60 min to 93 min | one TP=2 worker serves requests in series; client concurrency hides only the HTTP and encode time, so the wall time is the sum of the per-request model times plus the fp32 head on the host. Per-request figures are the stage 1 eager device times and the stage 3 traced server latencies (doc/optimized/perf_summary.json, logs/stage3_parity_remote.log). |

Measured client wall time of the eval_remote.py files present: 56 min in total (6 files; the Kev suites are not included, see their report.json latency blocks).

## Commands

```
cd /home/hous/dev/clef/tt-metal
SNAP=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c
nohup /home/hous/dev/clef/bin/devrun timeout 18000 env OMP_NUM_THREADS=8 CLEF_MODEL=$SNAP HF_MODEL=$SNAP HF_HUB_OFFLINE=1 CLEF_MESH_SHAPE=1x2 MESH_DEVICE=P150x2 CLEF_TRACED=0 CLEF_PLANNER=1 python -m uvicorn models.autoports.cloudflare_clef.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/clef/logs/stage5_server.log 2>&1 &
bash models/autoports/cloudflare_clef/scripts/run_eval.sh --base-url http://127.0.0.1:8008 --concurrency 4 --steps "arc banking77 newyorker samples parity kev summarize"
/home/hous/dev/clef/bin/hostrun python models/autoports/cloudflare_clef/scripts/summarize_eval.py --write
```

`run_eval.sh` writes one log per step under `/home/hous/dev/clef/logs/stage5_*.log`, the TT rows and `*.summary.json` under the out root, the Kev reports under `<out root>/kev/<suite>/test/` (`report.json`, `rows.json`, `predictions.jsonl`), and the metrics under `<out root>/metrics/`. `eval_remote.py` resumes a file (ids already answered are skipped), so an interrupted step is rerun with the same command.
