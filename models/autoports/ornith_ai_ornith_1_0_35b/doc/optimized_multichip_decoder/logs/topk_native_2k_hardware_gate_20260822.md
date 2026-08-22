# Top-k-native 2K post-install hardware gate

The opt-in `ORNITH_MOE_TOPK_NATIVE_SUB_CHUNK=2048` candidate passed the production-ring,
focused-correctness, full 40-layer B1/B2/B4, and timing gates on a four-chip Blackhole mesh after
the shared-runtime-geometry sources were installed. Every authoritative model run used
`ORNITH_MOE_TOPK_NATIVE=1`, `ORNITH_MOE_TOPK_NATIVE_SUB_CHUNK=2048`, and
`ORNITH_MOE_GATHER=1`, and every captured command exited zero.

## Gate summary

| gate | result | decisive evidence |
|---|---|---|
| Production 1x4 ring | pass | Balanced 1K/2K, device-0-skew 2K, and captured-layer-0 2K composites all returned and their immediately following all-reduces finished. |
| Focused correctness | 3 passed | Layer 0 PCC `0.999999868`; layer 3 PCC `0.999999665`; one 2K composite and zero fallback per tested layer. Real-weight reduced-stack layers 0, 1, and 3 completed their MoE all-reduces. |
| Full 40-layer B1/B2/B4 | pass | `calls=280`, `subchunks=280`, `layer_calls=120`, `fallbacks=0`, and `observed_composites=280`; every batch ended at `layer=39:moe-all-reduce:finish`. |
| Real layer-0 timing | pass | Gathered median `25.074 ms`; native-2K median `9.995 ms`; ratio `0.3986`. |

The full-stack counters are exact, not merely nonzero. Each layer receives one composite at B1,
two at B2, and four at B4, giving seven calls/subchunks and three public layer calls per layer.
Across 40 layers that is 280 calls/subchunks and 120 layer calls, with no fallback.

## Timing

Both timing arms used the same real 2,048-token linear-attention layer. After two untimed warmups
per branch, five gathered/native samples were interleaved:

| path | median layer time | delta vs gathered |
|---|---:|---:|
| gathered | 25.074 ms | — |
| native, one 2,048-token subchunk | 9.995 ms | -60.14% |

The measured native/gathered ratio is `0.398634472661395`. This clears both promotion bars:
native median `<=22.91 ms` and ratio `<=0.90`. Exact samples and machine-readable thresholds are in
[`topk_native_2k_hardware_gate_20260822.json`](topk_native_2k_hardware_gate_20260822.json).

## Authoritative artifacts

The raw logs are retained as external workspace artifacts rather than committed files. Their
basenames, capture intervals, and content hashes make the recorded evidence unambiguous without
creating links that would be broken in a normal clone.

| artifact | UTC interval | SHA-256 |
|---|---|---|
| `topk-native-production-ring-post-install-20260822T1900Z.log` | 19:00:53–19:00:59 | `dc746a7e9f29e0b957301bd0e9492921018a467da50f13aee409f790c950d9b9` |
| `topk-native-2k-focused-post-install-20260822T1901Z.log` | 19:01:18–19:02:07 | `d0493024da88eb6e9fe32090935c67c6bf3dd23f2652dcc7fb5e3cac2d1f035b` |
| `topk-native-2k-full40-b1-b2-b4-post-install-20260822T1902Z.log` | 19:02:20–19:07:29 | `276472e6a1e6e3be841dd6af91f5a8b77536d094dfff763c79770c82f8b3f325` |
| `topk-native-2k-timing-post-install-20260822T1907Z.log` | 19:08:01–19:08:16 | `039b49cd0a4583ab4c509dabac67b2ad342c439d56e47907c41e99e0844354e8` |

## Source and binary provenance

The candidate was tested from dirty, uncommitted changes on base commit
`5672083478cf7440b029ee5e074581d9d5697451`. The SHA-256 of the binary Git diff over the model
runtime, both relevant tests, and the native composite/unified-FFN operation trees is
`2e6eb4419bb8b14646d7c3737ed574b9be3d8bff071c81a29841394650d6633c`. The narrower six-file
shared-runtime-geometry diff is
`d2e5b98aa817f068c93740dcb2d05018f41aa210a5cebd326c901efec2fb5769`.

The installed binaries used by all four authoritative runs were:

| binary | installed mtime UTC | SHA-256 |
|---|---|---|
| `build_Release/lib/_ttnncpp.so` | 19:00:18 | `41a8d8815e83e529f1930298b5ed3af81cf08eecfb198dd3385db2c7145b6f00` |
| `build_Release/lib/_ttnn.so` | 19:00:31 | `9b09d8f7285f544078b74dfa4ada86703afc4cc18fbe6185b6c271fec13b7b98` |

The latest relevant source mtime was `18:58:21.486393756Z`; both binaries and all scoped sources
therefore predate the first authoritative post-install gate at `19:00:53Z`. The JSON records the
exact diff pathspecs and individual hashes for the promotion/ring tests and shared-geometry source
files.

These gates establish the native composite, immediate CCL boundary, representative-layer
correctness, all-layer exact invocation counts, and isolated layer performance. End-to-end serving
and the vLLM latency sweep remain separate acceptance evidence.
