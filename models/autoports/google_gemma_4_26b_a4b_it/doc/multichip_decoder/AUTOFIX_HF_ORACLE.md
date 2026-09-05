# AutoFix: TP4 sliding HF-oracle decode

## Failure

The final selected TP4 policy passed the direct optimized-baseline gate, but
the independent real-weight Hugging Face oracle exposed a sliding-only decode
miss:

- selected sliding prefill: `0.9984599503624263` (pass);
- selected sliding decode: `0.9947442319102862` (fail, threshold `0.995`);
- selected full decode control: `0.9997050635593928` (pass).

The one-chip optimized decoder's real-weight sliding decode is `0.9965878663`,
while TP4 versus that optimized decoder was `0.9992228587`. Their errors
compound through the nonlinear decoder/MoE path; lowering the gate was not
considered acceptable.

## Isolated candidates

Every run used the same real layer-0 weights, four-chip P300C ring, watcher
disabled, and
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'`. JUnit and JSON
evidence is under `artifacts/hf_candidates/`.

| Candidate | HF decode PCC | Verdict |
| --- | ---: | --- |
| disable packed expert decode | 0.994744232 | refuted; bit-identical |
| omit decode-only packed-dense DRAM copy | 0.994653929 | refuted |
| expert math HiFi2 | 0.994662112 | refuted |
| disable fused final scalar | 0.994728974 | refuted |
| disable shared FFN norm | 0.994611072 | refuted |
| disable folded router projection | 0.994661112 | refuted |
| disable folded expert scale and row-major routing | 0.994786338 | refuted |
| one all-reduce link | 0.994744232 | refuted; bit-identical |
| non-persistent all-reduce | 0.994624397 | refuted |
| omit O from decode DRAM roles | 0.999532151 | passes; localizes failure |
| O DRAM `in0_block_w=1` | 0.994737875 | refuted |
| O DRAM `in0_block_w=2` | 0.999532151 | passes |

The O weight is BF16 in both cases. Omitting the DRAM role and retaining the
role with `in0_block_w=2` produce the same HF PCC, so the fault is numerical
grouping in the TP4 sliding O DRAM-sharded matmul program, not weight storage,
CCL, graph folding, or active-expert selection.

The single `in0_block_w=2` program did not satisfy all workloads: the TP4 B32
optimized-baseline PCC became `0.9949080351`, whereas the existing
`in0_block_w=4` program passes at `0.9952382122`. A single program geometry is
therefore not selected.

## Minimal repair

The production path retains one BF16 DRAM-sharded O weight and two lightweight
program configurations:

- logical B1 sliding decode: `in0_block_w=2`;
- logical multi-user sliding decode, including B32: `in0_block_w=4`;
- TP4 full attention remains on its existing `in0_block_w=4` program.

No weight is duplicated and the capacity projection is unchanged. Dispatch
uses the logical batch already known by `_attention_decode`; physical tensor
row count is not used because both B1 and B32 occupy a 32-row tile. Explicit
`GEMMA4_MULTICHIP_DRAM_BLOCK_W_O_PROJ` overrides still select one caller-owned
program and suppress the automatic dual policy.

## Final evidence

Real-weight HF command:

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
python_env/bin/python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k 'multichip_real_weights_prefill_decode' \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/hf_oracle_selected_final.xml
```

Result: two passed. Sliding prefill/decode is
`0.9984599504/0.9995321511`; full is `0.9985002091/0.9997050636`.

The direct TP4 optimized-baseline rerun in
`artifacts/direct_pcc_tp4_after_hf_fix.xml` passed both kinds. Sliding is
`0.9970896450/0.9993185418`; full is `0.9986196042/0.9998745871`.

The combined canonical B32 rerun in `artifacts/batch32_selected_final.xml`
passed both kinds. Sliding remains `0.9952382122` at `8.8175026 ms`
(`1.3835201x`); full is `0.9950253440` at `9.1740186 ms` (`1.3298256x`).

The S=33 warmed trace rerun in `artifacts/trace_selected_final.xml` passed both
kinds with bit-exact replay and replicas. Sliding is `0.6521392 ms`
(`1.1741220x`), a `0.001051 ms` increase from the old program; full is
`0.9524278 ms` (`0.8526302x`).

Post-run `tt-smi -s` at `2026-09-05T18:32:11Z` reported all four P300C devices,
healthy DRAM, zero corrected/uncorrected GDDR errors, and temperatures from
34.1 to 37.7 C. The final watcher sweep is deliberately serialized after this
source freeze and is recorded by the parent work log.
