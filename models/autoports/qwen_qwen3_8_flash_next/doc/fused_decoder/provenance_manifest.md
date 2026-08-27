# Final-source provenance

All final gates below ran on P300c Blackhole chip 0 after the delivered source
was formatted. Parent checkpoint is `85b1099e34bb89307dd77b913f4b74f5fdc71283`
on `hous/qwen3.8-flash-next`.

## Inputs

| Path (relative to model root) | SHA-256 |
| --- | --- |
| `tt/fused_decoder.py` | `3f033b75dd81ee615bc8309e8e24dbc7aa43f3a6699dfc3bba596879fb7b9d3b` |
| `tests/test_fused_decoder.py` | `a06d7e0adfb2323021e7ece8d36f4847183d9ba485816a5a936807712b7c8b24` |
| `tests/test_fused_decoder_perf.py` | `f6d263a3bd1b888da6bd551b907e631634bb6f72cbe11ff5933f084ab32ae884` |
| `doc/context_contract.json` | `1ccec0bf032a139ab325ddf17f2b427f0b11d2a74d26be09577f2417de923ddc` |
| `tt/functional_decoder.py` | `361daec4bfd3014f3fe959f576c0efa9bd17e27fd2262a3604a06ffd51146fdc` |
| `tests/test_functional_decoder.py` | `9e014bb63c248619f6384d1c5d84900399f07eab128ef89b63ff94b5f64b60c6` |
| `tests/test_functional_decoder_perf.py` | `3cf4f046af6336af2ec689c8d4df08535a59c96339d5e14eaf5feba3fd84a9f1` |

## Primary gates

| Artifact | SHA-256 | Result |
| --- | --- | --- |
| `final_correctness_trace_alloc.xml` | `6a4c5f3cfbcc28c87d45237839746626f440e83e2b1cd36b09169280497eff73` | 35 passed, zero errors/failures, trace-allocation tracking. |
| `final_long_context.xml` | `8da35be968f50c77cfcba3b0d29eb9b3d851dc184ccd34ccc3161af69c0b34f5` | 7 passed, exact/non-aligned 262144 context and max decode. |
| `final_stress.xml` | `66935bbf0d4516bc2fd5622746f6affedf538ca29ba5c64fd7a6591fd0270822` | 9 repeated deterministic trace cases passed. |
| `final_watcher.xml` | `9032bf23cc34e89e7a5feeb4d6823491115d57197c222c4ae66d09262159fbe8` | 35 watcher-enabled tests passed. |
| `watcher_final_v5/generated/watcher/watcher.log` | `ad8251c8066fab1461533b0c4d5220b53a9427e8eb4807784555e6d1c9d2474a` | Clean dump and detach; no audited error signature. |
| `final_perf_count3.xml` | `e702be81db454e6db93959b847aa647c13d37f9b5999832c6766940d36c7f4e0` | Exact-source 9-test host timing pass. |
| `functional_perf_10replay_count7.raw.log` | `388cd8235067c47f36f8c687002825febfb43968bc3a331d982a376b03f02777` | Functional seven raw samples/layer at ten decode replays; 21 passed. |
| `functional_perf_10replay_count7.log` | `3a68a99ebbdb96f3479f23bc16b2cb1000c8b1b4c0fe5cd4d035582332d44c17` | Command/source-linked summary of the unedited functional transcript. |
| `fused_perf_final_count7.log` | `67bf57ca90a0aa72e901d0f48afd0461b4a35d97d60d7ae164b8ddc1c49686ba` | Seven raw samples/layer and medians; 21 passed. |
| `final_tt_smi.log` | `412c65f1d3290c1eea08ccd55d8f3ee8d00b5124c01a11e0f64edbcfb957768f` | DRAM healthy, zero corrected/uncorrected errors. |

The concise sibling `.log` files record exact commands, source/test hashes,
exit status, and the corresponding primary XML hash. JUnit XML is the
machine-readable source of selected nodes, durations, and pass/fail state.

## Final Tracy and tt-perf-report

| Window | Raw/stable ops SHA-256 | Filtered report SHA-256 | Device sum us | Ops |
| --- | --- | --- | ---: | ---: |
| L0 prefill | `a562f4a03e0edbfc4dc922b13242840939c883cd9b10c8cc053c818943f5f4cf` | `932518a076c78f08bee23e0fa48e1dd7046e05e79ba5f1106c1f99cca070acc5` | 16736.260 | 118 |
| L0 decode | `4785243a0f03d69479e7c71b93f976731696340383cf009f8be360644abcc8e0` | `788b80ef2383a25d4b27069c428de2b68d6374e50f3941337602faa059af5981` | 3848.370 | 125 |
| L1 prefill | `da468f6341478d606ac9aefa4aa7c44bcf39dd2e9964643899e1143df052f6f3` | `27e0033777c363c7b090592e146d8a25966eeb743c3e2264c27ddca5d17cd59d` | 20161.470 | 181 |
| L1 decode | `25c9d4c4d9c148642cd45bc34cee13fb7002768547b2270933b05a25fdda767f` | `628717c5c7223b90bb3ddc4d642fd233d42d931fa74c839a2621e47717d771ab` | 4269.910 | 170 |
| L3 prefill | `c8c6e483f5260a829ae96d8baed42f1c4835289579fb9e5e8196e32f6a8b8f7f` | `fbf72aa5477dcb2596fccf6d37456ca999b7ad6e699a321e0b4d5dc6f554e55e` | 43401.980 | 220 |
| L3 decode | `5bf78eaad460acee54d394501118144b27f363f47ad9cff7692b417f7dc84d57` | `298fa406a21f9bfce62747ed21a7a38240e23d0eb741927a68103b64816f60a6` | 5338.730 | 240 |

For each row, `<mode>_tracy.log` records the exact capture command, selected
node, raw path/hash, pass, and exit 0. `<mode>_tt_perf_report.log` independently
records the exact signpost transform, input/output hashes, sum, op count, and
exit 0. Thus capture and offline-transform provenance are not conflated.

## Candidate provenance

`candidates/README.md` indexes every executable candidate artifact. Historical
terminal transcripts include source diffs/hashes, exact commands, PCC/timing,
validator failures, and exit status. The final bounded KDA/indexer entries link
recoverable source patches or journal diffs, exact source/test hashes, commands,
and raw device artifacts; promoted paths are also proven by all final-source
gates above. Non-executable dedicated operations have exact binding/semantic
blockers in `graph_inventory.md`.

### Bounded-review candidate artifacts

| Candidate artifact | SHA-256 | Source/test identity and result |
| --- | --- | --- |
| `candidates/gdn_kda_qkv_causal_conv1d_silu.journal.jsonl` | `ae21bf2147fd27385aee47f5a3ae9499ecdfc7f27fd5017005364413f0e7c71d` | Byte-exact original patch/command/output records for chunks 320/640/1280; reconstructed sources `6d82dfcf...`/`952290bf...`/`82d1b89e...`, test `1d72572c...`. |
| `candidates/gdn_kda_sigmoid_gated_rms_norm.patch` | `c61dcb930126d53e365f1f0a4fb7713ce03cea5c47f71e5c2931f5035c07fb61` | Valid patch from final `3f033b75...` to perf candidate `e2918eba...`; correctness source `2989758f...` differs only by comments; test `a06d7e0a...`. |
| `candidates/gdn_kda_sigmoid_gated_rms_norm_candidate.xml` | `767d03a77e20986b6216bbb344860107e3c2949db9bb790a57f5e58a1e7201b9` | Real L0/L3 pass; L1 retained-input traced decode PCC 0.86216938 fails. |
| `candidates/gdn_kda_sigmoid_gated_rms_norm_state.xml` | `1d28958df2c5f34cad9e7229709b2c7778ef42377e19ae27c9549ba09e0f3c42` | Saved recurrent/conv/PLE state equality probe passes. |
| `candidates/gdn_kda_sigmoid_gated_rms_norm_eager_live_output_same_inputs.xml` | `320a3f02639faf8c0d4d5454f6899fa33d40e575742005c10432261e4446cf5d` | Valid caller-retained inputs: first eager L1 decode PCC 0.86030912. |
| `candidates/gdn_kda_sigmoid_gated_rms_norm_perf_count7.raw.log` | `b2ad46fedca71261ee9c4a33b7c4a76f106a3fafd2c2a803df3c4df5adcb2b5c` | Seven raw samples/layer; 21 passed. Faster GDN prefill but not deliverably correct. |
| `candidates/qsa_indexer_score_dsa.patch` | `621d6c383d17687837efe34450c0114c58ad35b4a063247bc6c98c0aa404b07f` | Exact forward diff from base `52fac21e...` to candidate `aad66b6c...`; correctness/perf tests `1d72572c...`/`f6d263a3...`. |
| `candidates/qsa_indexer_score_dsa.journal.jsonl` | `1cfcf06439cb43489b8d4c0c1c692c47ec09baccdb6aa93b67fd8d4b4fb438a0` | Byte-exact forward/reverse patches, commands, PCC, three timings and exit-0 output. |

The KDA epilogue AutoFix investigation and exact original journal timestamps
are in `candidates/gdn_kda_sigmoid_gated_rms_norm.log`; `AUTODEBUG.md` records
the fresh-context diagnosis and corrected conclusion. The causal-convolution
chunk sweep and indexer rows are completed by their candidate logs and linked
primary patches/transcripts listed in `candidates/README.md`.
