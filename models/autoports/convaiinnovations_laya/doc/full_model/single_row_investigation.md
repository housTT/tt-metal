# Single-row deviation investigation (review R3, Required Work P2)

Question: the served DAIR Emotion suite (400 single-row six-option decisions at the 1x128 bucket) deviates from the
CPU fp32 reference by up to 0.273 in probability with two confident flips (suite cases 209 and 381), while the 488-item
parity corpus (5-row calls) stays within 0.1225 with no confident flip. Is the cause the bucket, the precision policy,
or the inputs?

Method (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/convaiinnovations_laya/tests/single_row_investigation.py`,
written by Track T3, run by the orchestrator on chip 0 under devlock, 2026 Oct 6 02:18 to 02:30 UTC): (1) `corpus` builds
`/home/hous/dev/laya/reference/parity_corpus_single.npz` with the 800 single-row suite calls (400 AG News, 400 Emotion),
fp32 logits and probabilities, same format as the parity corpus; (2) `device` runs the seven Emotion decisions with
|dp| above 0.12 (cases 398, 381, 335, 209, 244, 286, 140) through the device model in four placements: alone at 1x128
exactly as served, alone at 1x256, inside a 5x256 batch and inside a 5x128 batch (filled with cases 0 to 3), for the
shipped policy and for `bf16_hifi4`, and compares the gathered scorer logits and probabilities with fp32; (3) the stage 6
decision gates on the whole 800-item single-row corpus for both policies (`gates_single_row_<policy>.json`).

## Placement and policy controls (seven flagged Emotion decisions)

| policy | placement | max abs dp | median max abs dp | max abs logit delta | argmax flips | confident flips |
|---|---|---|---|---|---|---|
| bf8w_hifi3_erf (shipped) | alone_1x128_as_served | 0.273 | 0.160 | 0.68 | [381, 209] | [381, 209] |
| bf8w_hifi3_erf (shipped) | alone_1x256 | 0.336 | 0.164 | 0.83 | [398, 381, 209] | [398, 381, 209] |
| bf8w_hifi3_erf (shipped) | in_5x256 | 0.273 | 0.160 | 0.68 | [381, 209] | [381, 209] |
| bf8w_hifi3_erf (shipped) | in_5x128 | 0.273 | 0.160 | 0.68 | [381, 209] | [381, 209] |
| bf16_hifi4 | alone_1x128_as_served | 0.108 | 0.076 | 0.27 | [] | [] |
| bf16_hifi4 | alone_1x256 | 0.133 | 0.068 | 0.51 | [] | [] |
| bf16_hifi4 | in_5x256 | 0.108 | 0.076 | 0.27 | [] | [] |
| bf16_hifi4 | in_5x128 | 0.108 | 0.076 | 0.27 | [] | [] |

Reading: with the shipped policy the result is bit-for-bit the same whether the text runs alone at 128 tokens or inside
a 5-row batch at 128 or 256 tokens (the bucket and the placement are not the cause); the 256-token single-row placement
adds a third flip (case 398). With `bf16_hifi4` the same texts stay within 0.108 (0.133 at 1x256) and no decision
flips. The deviation is a property of the shipped bfp8-weight HiFi3 numerics on these inputs: short texts whose six
options are nearly tied, where a 0.7 change of a scorer logit moves the top option. It is not a bucket or trace defect.

## Stage 6 gates on the single-row corpus (800 decisions)

Agreement of the served package (build 4) with the CPU fp32 reference on the 800 single-row suite decisions
(`served_single_row_gates_build4.json`): 777 of 779 confident decisions agree
(99.7 percent; gate 98), 792 of 800 argmax agree, median max abs dp
0.0012 (gate 0.02), p95 0.037, max 0.273, 7 decisions above 0.12.
Per suite: AG News 393 of 393 confident, max 0.056; Emotion 384 of 386 confident,
median 0.0016, max 0.273. The decision gates pass on this corpus; the two Emotion flips are the
only confident disagreements.

Placement invariance (`tests/decision_agreement.py` on 16 single-row questions of this corpus, alone versus 2, 4, a
mixed 8 and the largest bucket; gate max abs dp 0.01, which holds on the gate corpus): shipped `bf8w_hifi3_erf`
max 0.0192 with the same argmax in all placements (16 of 16); `bf16_hifi4`
max 0.0111, same argmax 16 of 16. On short single-row inputs the placement spread exceeds the
0.01 gate for both policies without changing any answer; the card states the spread per corpus.

## Disposition

The shipped policy stays (plan rule: fastest policy passing the gates; the gates are evaluated on the parity corpus and
now also reported on the single-row corpus). The card states the agreement per evidence set and names
`LAYA_PRECISION=bf16_hifi4` as the slower alternative that removes the observed flips (about 11 percent more latency on
the published cells; it passed every accuracy gate in the stage 8 sweep; its placement invariance was not separately
measured because it was slower than the shipped policy).
