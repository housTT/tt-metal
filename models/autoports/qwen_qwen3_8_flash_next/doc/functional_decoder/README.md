# Qwen3.8-Flash-Next functional decoder

This directory records the single-chip functional TTNN decoder stage for
`Qwen/Qwen3.8-Flash-Next`, checkpoint revision
`f5d08274bafd880402bd16f5e3e6c514136ec06c`. The implementation is
`../../tt/functional_decoder.py`; it subclasses `LightweightModule`, consumes
real HF layer state dictionaries, and implements all three meaningful target
layer forms: Gated DeltaNet, Gated DeltaNet with PLE, and Qwen Sparse Attention
(QSA), each followed by the real 512-expert sparse MoE path.

## Runtime contract

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    *,
    hf_config,
    layer_idx,
    mesh_device,
    max_batch=1,
    max_seq_len=262144,
    block_size=64,
)

prefill_forward(
    hidden_states,                 # [1, 1, seq_len, 10240], TTNN device tensor
    *,
    user_id=0,
    page_table=None,               # QSA int32 virtual-to-physical pages
    page_tables_per_chunk=None,    # QSA chunk-local page slices
    rot_mats=None,                 # QSA full (cos, sin) TTNN tables
    ple_embeddings=None,           # PLE [1, 1, seq_len, 2560]
)

prepare_decode_state()

decode_forward(
    hidden_states,                 # [1, 1, max_batch, 10240]
    *,
    current_pos,                   # device int32 [max_batch]
    page_table=None,
    rot_mats=None,
    ple_embeddings=None,           # PLE [1, 1, max_batch, 2560]
)
```

Public prefill accepts every logical length from 1 through `max_seq_len` and
owns 128-token physical padding/chunking and output slicing. QSA prefill and
decode fill/update paged K/V and raw-index caches using the caller's page table.
Decode mutates stable device buffers and is captured and replayed entirely by
TTNN trace execution. Call `prepare_decode_state()` after prefilling all users
and before trace capture.

The full PLE n-gram table is `320001536 x 160` BF16, or 102400491520 bytes.
It does not fit on one P300. The decoder boundary therefore takes the
input-specific, caller-prepared PLE embedding tensor. PLE projection, gating,
convolution, and recurrent state remain inside the measured TTNN pass; this is
a preprocessing boundary, not a context reduction.

## Correctness

Acceptance bar: PCC >= 0.995. Values below are from real checkpoint tensors;
decode PCC is read from trace replay output, not an eager substitute.

| Representative layer | Kind | Prefill PCC | Traced decode PCC |
| ---: | --- | ---: | ---: |
| 0 | Gated DeltaNet | 0.99871051 | 0.99997765 |
| 1 | Gated DeltaNet + PLE | 0.99910986 | 0.99991739 |
| 3 | QSA | 0.99679226 | 0.99988198 |

The real tests execute the true 10240-wide four-stream hyperconnections,
target attention/GDN/PLE path, real router and selected expert weights, shared
expert, and residual order. `ForbidHostFallback` replaces
`ttnn.from_torch`, `ttnn.as_tensor`, and `ttnn.to_torch` with immediate
failures during each measured prefill/decode call. A source gate separately
checks the public runtime entry points. Setup and final PCC conversion remain
explicit test boundaries.

## Capability evidence

| Claim | Evidence | Remaining risk |
| --- | --- | --- |
| HF advertises 262144 tokens; the functional contract remains 262144 | Exact target config assertion and `../context_contract.json` | None observed |
| GDN public prefill at 262144 | `long_context_linear0_full.log` | None observed |
| GDN+PLE public prefill at 262144 | `long_context_linear1_full.log` | Caller supplies prepared PLE rows |
| QSA public prefill at 262144 | `long_context_qsa_public_full.log`: all 2048 public chunks, the 10240-wide hyperconnection/MoE boundary, all 65536 compressed blocks, shuffled pages and cache progression (`1 passed` in 17m52s) | None observed |
| Public non-aligned prefill at 262143 | `near_max_gdn_public_final.log`, `near_max_ple_public_final.log`, and `near_max_qsa_public.log` | None observed |
| Logical boundaries | Public device tests at 31/32/33, 63/64/65 and 127/128/129 for every kind; QSA public prefill at 2049; planner checks at 2047/2048/2049; public near-max execution at 262143 | None observed |
| QSA underfilled top-k semantics | `topk_multiset.log` proves every valid token from positions 0 through 2049 appears exactly once after the fixed-shape 512-block top-k | None observed |
| Paged multi-user semantics | shuffled-page invariance, exact batch-2 cache-slot updates at positions 63 and 128, two-user prefill-to-decode for every kind, and 32-user decode with distinct QSA page rows/positions (`batch32_decode.log`) | None observed through batch 32 |
| Advertised-context traced decode | `qsa_max_context_trace_decode.log` captures and replays QSA decode at current position 262143 with full cache/page/RoPE geometry; full-context eager decode also runs for every kind | None observed |
| Deterministic trace replay | two identical replays are bitwise equal for layers 0, 1, and 3 | None observed |

There is no advertised capability reduction. The precise machine-readable
statement is in `../context_contract.json`.

## Performance

One 128-token prefill was warmed before timing. Decode was compiled, captured,
warmed, then timed over ten trace replays. Host timing includes dispatch and
one final synchronization. The profiler rows are separately signpost-filtered
device-op sums; each human-readable report says `0 host ops`.

| Layer | Warmed prefill host ms | Traced decode host ms/token | Prefill device-op sum us | Decode device-op sum us |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 24.126221 | 5.729530 | 23790 | 5396 |
| 1 | 27.602558 | 6.700191 | 27269 | 5922 |
| 3 | 408.469450 | 11.338447 | 408094 | 11081 |

`perf_host_timing.log` is the ten-replay host provenance. For every layer and
mode, `tracy/<layer>/<mode>_ops.csv` is the original Tracy CSV,
`<mode>_perf_report.csv` is the signpost-filtered machine table,
`<mode>_perf_report.txt` is the human-readable table, and
`<mode>_provenance.log` is the exact pytest/Tracy output. QSA artifacts were
recollected after both the row-major embedding-index and underfilled top-k
repairs. The profiler provenance's one-replay host number includes profiler
flush/readback overhead and is not used as latency; only the filtered device-op
sum is taken from that run.

## Watcher and determinism

The final aggregate correctness command ran separately from profiling with
`TT_METAL_WATCHER=10`.
`watcher_final_20260826_1419/pytest.log` records `34 passed` and seven
intentional long-context skips; its `generated/watcher/watcher.log` is the
device watcher log. The audit found no fatal watcher exception,
stack/L1/NOC/CB sanitizer message, or selection of the unsafe
`embedding_ind_tilized` kernel.

An earlier watcher run exposed a real tiled-index embedding scratch-buffer
overflow. `AUTOTRIAGE.md` contains the source/CB ledger; the preserved failure
is under `watcher_embedding_failure_20260826_1244/`. All QSA embedding indices
now convert on device to row-major before embedding, and the focused repaired
watcher evidence is under `watcher_qsa_fix_20260826_1251/`.

## Reproduction

Source the pinned environment before every command:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
```

Correctness and trace replay:

```bash
pytest -q -s models/autoports/qwen_qwen3_8_flash_next/tests/test_functional_decoder.py \
  --tb=short --durations=25
```

Advertised-context probes:

```bash
pytest -q -s models/autoports/qwen_qwen3_8_flash_next/tests/test_functional_decoder.py \
  --long-context -k 'full_advertised_context or near_max_non_aligned_context or qsa_traced_decode_at_advertised_context' \
  --tb=short
```

Warmed timing:

```bash
pytest -q -s models/autoports/qwen_qwen3_8_flash_next/tests/test_functional_decoder_perf.py
```

Watcher, kept separate from Tracy:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/watcher_final_20260826_1419 \
pytest -q -s models/autoports/qwen_qwen3_8_flash_next/tests/test_functional_decoder.py \
  --tb=short --durations=25
```

One profiler mode, with layer 3 decode as the example:

```bash
TT_METAL_PROFILER_MID_RUN_DUMP=1 QWEN38_PERF_FLUSH_PROFILER=1 \
QWEN38_PERF_MODE=decode QWEN38_PERF_DECODE_REPLAYS=1 \
python -m tracy -r -p -v --check-exit-code -m pytest -q \
  'models/autoports/qwen_qwen3_8_flash_next/tests/test_functional_decoder_perf.py::test_warmed_prefill_and_traced_decode[3]' -s
```

The exact `tt-perf-report` commands and artifact copies are logged in
`work_log.md` and the per-mode provenance files.

## Known limitations

- This is intentionally only the functional decoder layer stage. It contains
  no optimized decoder, multichip, full-model, generator, or vLLM work.
- The 95.37 GiB full PLE n-gram table is a caller-side preprocessing boundary;
  the decoder consumes the input-specific prepared rows. This does not reduce
  the 262144-token decoder capability.
- The TTNN tiled-index embedding implementation still has the source-level
  scratch-size mismatch documented in `AUTOTRIAGE.md`; this decoder avoids the
  affected program factory.
