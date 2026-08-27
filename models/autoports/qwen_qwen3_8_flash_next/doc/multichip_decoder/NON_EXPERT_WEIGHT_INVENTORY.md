# Exact non-expert weight inventory

This is the metadata-only physical-storage audit for the fixed P300 `1x2`
Qwen/Qwen3.8-Flash-Next multichip decoder.  It follows the final tensor objects
that remain allocated after `FunctionalDecoder` loading, `FusedDecoder`
packing, `OptimizedDecoder` typecasting, and `MultichipDecoder` TP2 slicing.
It does not load tensor contents and does not open TT hardware.

The result is:

```text
48 decoder layers                         3,479,858,176 B/die
future full-text entry/final/head         1,279,016,960 B/die
capacity-plan non-expert weights          4,758,875,136 B/die
```

Routed-expert slots, the host-backed PLE table, KV/index caches, recurrent
state, and runtime constants are separate capacity categories.  The landed
within-stream fractured residual saves `706,805,760 B/die` from the earlier
replicated-residual decoder inventory.  The original `3,921,895,424 B/die`
allowance now exceeds the decoder alone, but it still omits the future
full-text endpoints and undercounts that required stack by `836,979,712`
bytes.  There is no double count.

## Metadata and physical-byte method

The audited snapshot is
`f5d08274bafd880402bd16f5e3e6c514136ec06c`.  Its index reports
`359,999,963,128` checkpoint bytes.  Raw `config.json` has 48 text layers:
36 `linear_attention` layers, 12 `full_attention` layers at zero-based indices
`3,7,...,47`, and PLE one-based id `2` (zero-based layer 1).  Transformers
canonicalizes `full_attention` to the autoport's `qwen_sparse_attention`.

All persistent weights use TT TILE layout.  For logical TT shape `S`, the
per-device allocation used below is:

```text
tiles(S) = product(S[:-2]) * ceil(S[-2]/32) * ceil(S[-1]/32)
bytes(S, dtype) = tiles(S) * tile_bytes(dtype)

BF16 tile  = 2,048 bytes
BFP8 tile  = 1,088 bytes
FP32 tile  = 4,096 bytes
```

BFP8 is not one byte per logical element: every 1,024-element tile has block
format metadata and occupies 1,088 bytes.  The physical-shape columns expose
all 32-by-32 padding.  `R` means one identical copy per die, `TP` means a
distinct rank-local shard, and `mixed` means a fused tensor contains both.

The test uses `safetensors.safe_open(...).get_slice(key).get_shape()` and
`get_dtype()` for checkpoint validation; it never calls `get_tensor` for this
inventory.

## Common graph in all 48 layers

Checkpoint shapes are `[output,input]`; TT matmul weights are transposed.
The stacked decoder carries the four residual streams persistently as
`[1,1,4*M,1280]` per die.  Setup reshapes each HC tensor by stream and
mesh-partitions its 2,560-wide hidden axis.  The checkpoint-global shape is
still shown so the rank-local storage remains mechanically traceable.

| Final tensor role | Checkpoint constituents | Global TT logical | Per-die TT logical | Physical per die | Dtype | Placement | Stack count | Stack bytes/die |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| attention/MLP HC norm | `[10240]` | `[4,2560]` | `[4,1280]` | `[32,1280]` | BF16 | TP hidden | 96 | 7,864,320 |
| attention/MLP HC down+inject | `[320,10240]` + `[4,10240]` | `[10240,324]` | `[5120,324]` | `[5120,352]` | BFP8 | TP hidden input | 96 | 183,828,480 |
| attention/MLP HC up | `[10240,320]` | `[320,10240]` | `[320,5120]` | same | BFP8 | TP hidden output | 96 | 167,116,800 |
| fused MoE input | router `[512,2560]`; shared gate/up each `[640,2560]`; scalar gate `[1,2560]` | `[2560,1793]` | `[2560,1153]` | `[2560,1184]` | BFP8 | router/scalar R, gate/up TP | 48 | 154,583,040 |
| shared down | `[2560,640]` | `[640,2560]` | `[320,2560]` | same | BFP8 | TP input | 48 | 41,779,200 |

The common subtotal is `11,566,080 B/layer`, or `555,171,840 B/die` for all
48 layers.  Fracturing these six HC tensors saves `405,995,520 B/die` versus
their earlier replicated storage.  The temporary rank-1 optimized graph is
released after its shard is copied, so it does not create a second persistent
allocation.

## GDN graph in 36 layers

The numerically sensitive GDN qkv/z recurrence remains replicated.  Its output
projection is now column-parallel over the persistent residual width, so each
die stores 1,280 of the global 2,560 output columns and produces its residual
shard directly.

| Final tensor role | Checkpoint constituents | Per-die TT logical | Physical per die | Dtype | Count | Stack bytes/die |
| --- | --- | --- | --- | --- | ---: | ---: |
| packed qkv+b+a, ordinary | `[10240,2560]` + 2 x `[48,2560]` | `[2560,10336]` | same | BFP8 | 35 | 983,987,200 |
| packed qkv+b+a, layer 0 | same | `[2560,10560]` | same | BFP8 | 1 | 28,723,200 |
| qkv+b+a bias, ordinary | `dt_bias [48]` plus zeros | `[1,10336]` | `[32,10336]` | FP32 | 35 | 46,305,280 |
| qkv+b+a bias, layer 0 | same | `[1,10560]` | `[32,10560]` | FP32 | 1 | 1,351,680 |
| z | `[6144,2560]` | `[2560,6144]` | same | BFP8 | 36 | 601,620,480 |
| decode convolution taps | `[10240,1,4]` split four ways | `[1,10240]` | `[32,10240]` | FP32 | 144 | 188,743,680 |
| retained prefill convolution taps | same checkpoint tensor | `[1,10240]` | `[32,10240]` | BF16 | 144 | 94,371,840 |
| negative exponent A | `A_log [48]` | `[1,48]` | `[32,64]` | FP32 | 36 | 294,912 |
| GDN norm | `[128]` | `[1,128]` | `[32,128]` | BF16 | 36 | 294,912 |
| output | `[2560,6144]` | `[6144,1280]` | same | BFP8 | 36 | 300,810,240 |
| q/k normalization constants | setup-derived | `[1,128]` | `[32,128]` | FP32 | 72 | 1,179,648 |

The two packed-qkv rows together are `1,012,710,400 B/die`.  Layer 0 alone
uses its configured 55-core decode program.  Its output width is padded from
10,336 to `ceil(10336/(32*55))*(32*55) = 10,560`; padding the BFP8 matrix and
FP32 bias costs `609,280 + 28,672 = 637,952` bytes.

The GDN subtotal is:

```text
36 * 62,417,920 + 637,952 = 2,247,683,072 B/die
```

Column-fracturing the output saves `8,355,840 B/layer`, or
`300,810,240 B/die` across the 36 GDN layers, without changing the replicated
recurrence geometry whose TP2 alternative failed PCC.

The BF16 prefill taps and FP32 decode taps are both real persistent tensors:
`FusedDecoder` retains references to the original taps before widening the
decode copies.  Omitting either copy undercounts the delivered graph.

## PLE addition in layer 1

| Final tensor role | Checkpoint constituents | Per-die TT logical | Physical per die | Dtype | Placement | Count | Bytes/die |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| key+value | `[10240,2560]` + `[2560,2560]` | `[2560,12800]` | same | BFP8 | R | 1 | 34,816,000 |
| key/query/conv norms | 3 x `[10240]` | `[1,10240]` | `[32,10240]` | BF16 | R | 3 | 1,966,080 |
| convolution taps | `[10240,1,4]` split four ways | `[1,10240]` | `[32,10240]` | BF16 | R | 4 | 2,621,440 |

PLE projection weight subtotal: `39,403,520 B/die`.  The 128 mmap embedding
shards are deliberately excluded.  The exact selected-row TT staging is
charged separately by the host-weight contract.

## TP2 QSA graph in 12 layers

| Final tensor role | Checkpoint constituents | Global TT logical | Per-die TT logical | Physical per die | Dtype | Placement | Count | Stack bytes/die |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| fused QSA input | q+gate `[12288,2560]`, k/v each `[512,2560]`, index q/k widths 512/128 | `[2560,13952]` | `[2560,7296]` | same | BF16 | heads TP, index R | 12 | 448,266,240 |
| attention output | `[2560,6144]` | `[6144,2560]` | `[3072,2560]` | same | BF16 | TP input | 12 | 188,743,680 |
| q/k norms | 2 x `[256]` | `[1,256]` | same | `[32,256]` | BF16 | R | 24 | 393,216 |
| index q/k norms | 2 x `[128]` | `[1,128]` | same | `[32,128]` | BF16 | R | 24 | 196,608 |

QSA subtotal: `53,133,312 B/layer`, or `637,599,744 B/die` for 12 layers.

## Decoder and full-text totals

The exact decoder-stack formula is:

```text
48 * 11,566,080             common fractured HC
+ 36 * 62,417,920           replicated GDN core + fractured output
+ 637,952                   layer-0 GDN explicit padding
+ 39,403,520                PLE projection graph
+ 12 * 53,133,312           TP2 QSA graph
= 3,479,858,176 B/die
```

The future full model also needs both untied vocabulary tensors and the final
hyperconnection mixer.  The selected capacity strategy shards the input
embedding along hidden width and performs one gather at stack entry, keeps the
final mixer replicated, and shards the LM head along vocabulary columns.

| Full-text tensor | Checkpoint shape | Per-die TT logical | Physical per die | Dtype | Placement | Bytes/die |
| --- | --- | --- | --- | --- | --- | ---: |
| token embedding | `[248320,2560]` | `[248320,1280]` | same | BF16 | hidden TP | 635,699,200 |
| final HC norm | `[10240]` | `[1,10240]` | `[32,10240]` | BF16 | R | 655,360 |
| final HC down | `[320,10240]` | `[10240,320]` | same | BFP8 | R | 3,481,600 |
| final HC up | `[10240,320]` | `[320,10240]` | same | BFP8 | R | 3,481,600 |
| LM head | `[248320,2560]` | `[2560,124160]` | same | BF16 | vocabulary TP | 635,699,200 |

The endpoint subtotal is `1,279,016,960 B/die`.  The checkpoint declares
`tie_word_embeddings=false`, so the input embedding and LM head cannot be
counted as one shared tensor.  Replicating the input embedding would add
`635,699,200 B/die`; BF16 is retained because no lower-precision full-model
accuracy result exists yet.

Thus the full-text capacity charge is:

```text
3,479,858,176 + 1,279,016,960 = 4,758,875,136 B/die
```

## Previous allowance provenance and correction

The previous source comment decomposed the allowance into a pre-code TP2
estimate plus 1.125 GiB for the second half of replicated GDN:

```text
2,713,935,872 + 1,207,959,552 = 3,921,895,424 B/die
```

Applying the delivered fusion, dtype, duplicate-tap, and tile-padding rules to
the rejected GDN-TP2 geometry produces `2,914,749,440 B/die`, so the pre-code
base was short by `200,813,568` bytes.  The exact additional cost of replacing
that TP GDN with replicated GDN is `1,271,914,496` bytes, so 1.125 GiB was
short by another `63,954,944` bytes.  Before the residual-fracturing change,
their sum was the decoder-only shortage:

```text
200,813,568 + 63,954,944 = 264,768,512
4,186,663,936 - 3,921,895,424 = 264,768,512
```

That historical comparison is retained because the rejected GDN-head TP plan
has not changed.  The landed residual path then removes:

```text
405,995,520 B/die           TP2 HC norm/down/up storage
+ 300,810,240 B/die         GDN output-column TP storage
= 706,805,760 B/die

4,186,663,936 - 706,805,760 = 3,479,858,176 B/die
```

Equivalently, the source-visible provenance is
`2,914,749,440 + 1,271,914,496 - 706,805,760 = 3,479,858,176`.
There is no double count.  The old `3,921,895,424` allowance now has
`442,037,248` bytes of decoder-only margin, but it omitted `1,279,016,960`
bytes of full-text endpoints and is therefore still `836,979,712` bytes below
the required full-text charge.

With the corrected `4,758,875,136` weight charge, the host-backed stack plan is
`9,373,395,968 B/die` and leaves `24,852,124,672 B/die` of nominal DRAM
headroom.  The resident-expert budget used by the compression comparison is
`25,582,031,872` bytes.  The optimistic standard-layout and bank-padded maximum
BFP4 fractions become `0.4442310248480903` and `0.2449097278071385`,
respectively.  Neither changes the selected exact host-backed expert design.

## Persistent non-weight constants

These are not included in `4,758,875,136`, but they must not disappear inside
an unexplained reserve:

| Constant family | Formula | Bytes/die |
| --- | --- | ---: |
| GDN fused kernel tiles | `36 * 6 FP32 tiles` | 884,736 |
| QSA max-context static index constants | `12 * 33,699,840` | 404,398,080 |
| warmed QSA compressed-block RoPE | `12 * 2 * bytes([65536,64], BF16)` | 201,326,592 |
| total |  | 606,609,408 |

The current 1 GiB runtime/activation/allocator/trace reserve covers this known
constant floor, leaving `467,132,416` bytes for its other stated purposes.

## Reproduction

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
pytest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py::test_non_expert_weight_inventory_from_checkpoint_metadata
```

The test defines the row-level physical inventory as pure Python, validates
representative common/GDN/PLE/QSA/full-text shapes and BF16 source dtypes from
the safetensors index, walks all 48 layer kinds, and asserts the decoder,
endpoint, and combined totals above.
