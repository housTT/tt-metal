# AutoDebug: initial Qwen3.5-35B-A3B functional decoder

Investigation date: 2026-10-10 UTC. Inspection only: no TT device access, target execution, source edits, launcher invocation, or additional agents. This report is the only file written by this investigation.

**Revision scope:** the detailed causal findings and adapter line numbers below describe the initial adapter with SHA256 `f33df4ec57ed994d0bae568371063966cb3d379ae5e539249382ff44a699e3a5`. Final verification detected concurrent edits to that untracked file by another workflow. The later adapter addresses several initial findings structurally; see the concurrent-update assessment below. Do not apply the initial adapter call chains to the later file as though they were the same revision.

## Findings at a glance — initial adapter

The adapter does not yet implement the requested functional contract. There are several independent problems; removing the reported FIR host upload alone will not make the decoder correct.

| Priority | Finding | Scope and confidence |
|---|---|---|
| 1 | Every GDN prefill constructs and uploads a Torch FIR selector, including aligned lengths. Padded scans construct another host mask. | Direct call-chain proof; violates device-only forward and prevents capture of that prefill path. |
| 1 | Rank-3 single-device layer activations reach rank-4 MoE code without conversion. Hidden width 2048 is interpreted as the token/batch count. | Both layer kinds, prefill and decode; direct incompatible shape calculations. |
| 1 | Default GDN chunk preprocessing uses diagonal regularization `alpha=0.25`, changing the reference recurrence. | GDN prefill with default environment, including aligned inputs. Mathematical discrepancy, independent of quantization. |
| 1 | Paged-cache binding does not ensure paged prefill. Fill offsets, logical lengths, and request rows are not fully represented by the adapter. | Full attention: missing chunk table, non-page-aligned continuation, partial pages, and nonzero request slots. |
| 1 | Batch support is contradicted by fixed B=1 GDN state and the full-attention paged decode axis conversion. | B>1; independently broken even after repairing MoE rank. |
| 2 | The boundary rejects unpadded lengths, returns physical-length outputs, and does not bound whole-call GDN work. | Arbitrary lengths and native 262144-token context are not implemented by the initial wrapper. |
| 2 | Reuse silently selects BFP4/BFP8 weights, BF16 persistent GDN state, and LoFi projections. | Confirmed policy mismatch with an evidence-backed functional baseline; no measured PCC failure is asserted. |

Runtime manifestation was not measured by this investigation. At initial inspection, no supplied runtime trace established which error happened first on hardware. In particular, attention/state work runs before the malformed MoE call; the MoE failure can hide earlier numerical or cache defects and leave already-advanced state behind. Runtime artifacts appeared alongside the concurrent update and are distinguished below.

## Evidence boundaries and source identity

Direct observations:

- Repository HEAD: `98dc4ae62ccfc32836ddcfad50f9fd97863612d4`.
- Initial `git status --short`: `?? models/autoports/`. Thus the adapter is untracked working-tree content, not part of that HEAD.
- Read the local checkpoint configuration and selected safetensors **headers**, without materializing weights. They confirm H=2048; full attention Q=16, KV=2, D=256; GDN key heads=16, value heads=32, Dk=Dv=128, convolution width=4; 256 experts, top-8, expert/shared I=512; 40 layers and context=262144.
- Example BF16 checkpoint shapes: GDN QKV `[8192,2048]`, convolution `[8192,1,4]`, experts gate/up `[256,1024,2048]`, experts down `[256,2048,512]`, full-attention Q/gate `[8192,2048]`, K `[512,2048]`.
- Read the installed reference at `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`. Conclusions about HF continuation refer to this installed file, not an assumed historical Transformers release.
- Parsed all 62 Python files beneath the three focus paths with `ast.parse`; no target modules were imported. Also evaluated small integer shape/address calculations and a scalar recurrence witness. These are inspection checks, not model correctness tests.

Requester context and `../source-audit.json` guided the questions but were not treated as correctness evidence. There is no hardware PCC, latency, watcher, or trace-replay result from this investigation. Existing CPU test success reported by another workflow is not re-certified here.

Content hashes for revision consistency:

| File | SHA256 |
|---|---|
| `models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py` | `f33df4ec57ed994d0bae568371063966cb3d379ae5e539249382ff44a699e3a5` |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py` | `2771ed59f13a374d5d64fd75b7eccfe88714f7517ee779a4566b239b8cadfde5` |
| Installed HF `modeling_qwen3_5_moe.py` above | `f738a9eb46d1351a751f36a1cae5f7435d589e8c3218e7315182419c4c0230f1` |

For compact source references below:

- **adapter** = `models/autoports/qwen_qwen3_5_35b_a3b/tt/functional_decoder.py`.
- **q36/** = `models/demos/blackhole/qwen36/tt/`.
- **experimental/** = `models/experimental/gated_attention_gated_deltanet/tt/`.
- **seq.py** = `experimental/ttnn_delta_rule_seq.py` using that directory prefix.
- **ops/** = `ttnn/cpp/ttnn/operations/`.
- **HF** = the installed reference file identified above.

### Concurrent-update assessment

The final checksum check deliberately failed for the adapter. The observed replacement has SHA256 `b15c48d7083607187804576c597a505bfea2e7e5eb9c4ce718a2553b1a8d15d2`, with file mtime `2026-10-10 00:00:49.239664556 UTC`. Shared seq code and the installed HF reference retain the hashes above. I read the replacement's construction, state preparation, prefill/decode, and rank bridge to assess which original findings still apply.

| Initial finding | Later adapter source status |
|---|---|
| Forced GDN FIR/mask uploads | `prefill_forward:298–318` now runs GDN in full 128-token chunks with no `valid_len` argument, and real tails in T=1 decode. The identified forced-host path is structurally bypassed. The placement caveat in finding 1 still applies to those full chunks. |
| MoE rank mismatch | Constructor installs `_RankThreeMoE` at line 189; its forward at 366–371 reshapes to `[1,1,B*T,H]` and restores rank 3. The original mismatch is addressed. The inherited shared-expert axis-1 compute/memory heuristic remains a review item. |
| GDN diagonal damping | Still constructs `Qwen36DecoderLayer` and calls its chunk prefill for aligned GDN blocks. No explicit local exact-recurrence policy appears in the replacement. The default-alpha finding remains applicable unless execution explicitly overrides the environment. Tail decode uses the undamped path. |
| Paged dispatch and unaligned writes | Full-attention prefill now requires binding plus a full page table, derives a selected row/chunk table, passes absolute start for aligned fills, and uses ordered single-token decode otherwise (`286–318`). This addresses the identified fallback and intra-page-start paths. Binding still performs no local cache dtype/geometry validation. |
| B>1 state and Q-axis defects | It allocates independent B=1 state pairs per slot and loops through rows (`194–207,225–235,298–322,341–350`). Thus those shared implementations are called with B=1 and their batched defects are bypassed. This is source support for a serial batch strategy, not proof of multirow correctness or replay behavior. |
| Arbitrary lengths and unbounded GDN call | New prefill accepts logical T, processes bounded chunks/real tails, and returns concatenated real rows. The old padding rejection and whole-T GDN call no longer describe it. It still needs native-context capacity evidence, including accumulated output pieces, and accepts only scalar `valid_len`/`start_pos` for the call. |
| Inherited precision | Still uses the same Qwen36 constructors/loaders; default BFP4/BFP8 weights, BF16 state, and associated compute policy remain. |

The replacement builds a device RoPE table at setup and indexes it from positions. Its legacy `cos`, `sin`, `chunk_page_table`, and `chunk_start_idx_tensor` arguments are retained but not used by the new prefill body; `chunk_start_idx` overrides scalar `start_pos`. Clarify that compatibility contract if callers depend on the old explicit/tensor-position arguments. This is a newly observed interface discrepancy, not evidence that the original adapter behaved that way.

New `tests/run_layer.py`, `tests/hf_reference.py`, `tests/runtime_audit.py`, and documentation/artifacts also appeared. I read the host-operation guard and README/work-log summaries, but did not execute or independently audit their numerical/replay evidence. Those summaries claim small real-weight smoke/trace checks and explicitly leave the stage in progress. Their presence means the initial statement “only CPU contract tests exist” is historical. They do not retroactively validate the initial malformed adapter, nor does this investigation certify the later implementation. The report's repair and validation sections remain guidance to compare against the later implementation, not an instruction to duplicate changes already made.

The file continued changing during final verification. A subsequent sample, SHA256 `2dbb1ab0cf70a4820df23ff000731273cf89b5f1611d907b21bebc7d410bfc69`, mtime `2026-10-10 00:07:18.817189207 UTC`, adds `expert_precision="inherited"` with a `"bf16"` option that reloads routed expert gate/up and down weights from checkpoint tensors (lines 190–201 in that sample). That explicitly addresses part of the precision recommendation; it does not change the default, shared expert, attention/GDN weights, or recurrence state policy. The assessment table above belongs to the earlier `b15c...` sample, and its line numbers must not be applied to this next sample. This is an investigation of the initial handoff with a bounded update assessment, not a continuous review or approval of the concurrently evolving implementation.

## 1. Forced host work in every GDN prefill

### Causal chain

1. `adapter:219–242` computes `logical_len = physical_len` when the caller omits `valid_len`, then **always** forwards `valid_len=logical_len`.
2. `q36/layer.py:291–295` selects GDN `mode="chunk"`; `q36/gdn/decode.py:28–79` forwards the length, persistent recurrent/conv state, cached taps, and mega-fused projection weight.
3. The weight loader constructs that mega weight and taps (`q36/gdn/weights.py:215–247`). Therefore the ordinary prefill path is `experimental/ttnn_gated_deltanet.py:415–459`: mega projection, then `_causal_conv1d_fir` with a non-None length.
4. FIR lines 181–203 allocate `torch.zeros(B,3,T+3)`, fill one-hot indices on the host, and call `ttnn.from_torch(..., device=device)` on **every invocation**. The device matmul selects the state tail; the activation itself is not downloaded to Torch.
5. For actual padding, `experimental/ttnn_delta_rule_seq.py:439–457` separately constructs and uploads a `[B*32,T,1]` Torch mask. This second path is skipped for scalar `valid_len == T`, unlike FIR.

Concrete cases at B=1:

| Logical length | Physical length | FIR selector upload | Scan mask upload |
|---:|---:|---|---|
| 1 | 128 | yes | yes |
| 127 | 128 | yes | yes |
| 128 | 128 | yes | no |
| 129 | 256 | yes | yes |
| 2200 | 2304 | yes | yes |
| 262144 | 262144 | yes | no |

This is runtime host tensor preparation/upload, not CPU execution of the GDN recurrence. That distinction does not make it satisfy the requested device-only forward. The mesh command queue explicitly rejects writes during capture (`tt_metal/distributed/fd_mesh_command_queue.cpp:616,655,804`). The sibling TP implementation documents the same failure and deliberately preserves `valid_len=None` on full chunks (`q36/gdn/tp.py:558–564`). This contrast supports the diagnosis without relying on its comments as proof.

### Smallest local repair

- Preserve `None` for a wholly valid internal chunk. FIR's static tail slice at lines 176–180 selects exactly the same three rows as the one-hot for `valid_len=T`.
- Decouple memory placement from this normalization: `ttnn_gated_deltanet.py:410–411` uses `mc=None` when a length is supplied, but `_seq_memory_config` selects L1 for T<=512 when it is absent (lines 20–25). Keep an explicit DRAM placement policy for the functional prefill baseline while removing host selection. Equal selected rows do not establish equal resource behavior after this change.
- Do **not** stop there: padded lengths still require a device-only mask and correct state-tail selection. A local FIR helper can use device slices for eager scalar lengths, or a preallocated position vector plus device length comparisons/gather/selection for captured fixed buckets. Cache only length-independent constants at construction.
- For a simple functional baseline, run aligned prefixes and process each real tail token through the on-device recurrent step. This avoids advancing state through padding and avoids both host masks; it is a correctness-first option, not a performance claim.
- If using padded scans, set padded `beta=0` and log-decay `g=0` (and mask q/k/v consistently). Zero hidden states alone are insufficient: convolution history and biased gates can still generate state-changing values.

**Scope limit:** this finding concerns prefill. `decode_forward` passes no `valid_len`, and the T=1 GDN path uses precomputed taps plus in-place state copies. It is incorrect to claim that this particular FIR upload automatically breaks ordinary decode capture.

## 2. The single-device layer and MoE disagree about tensor rank

The adapter and both single-device attention implementations interpret input as `[B,T,2048]`: `adapter:219,253`, `experimental/ttnn_gated_deltanet.py:408–409`, and `experimental/ttnn_gated_attention.py:238–239`. The layer sends the post-attention normalized tensor directly to `feed_forward.forward` (`q36/layer.py:298–306`). There is no MoE-boundary reshape. RMSNorm does not supply one implicitly: its C++ call forwards the input to layer norm, whose output specification starts from the input logical shape (`ops/normalization/layernorm/device/layernorm_device_operation.cpp:484–487`).

The routed expert implementation explicitly expects `[1,1,S,H]`, with S=sequence tokens in prefill or batch users in decode (`q36/moe/experts.py:25–40`). Its prefill and decode bodies read `hidden_states.shape[2]` (`q36/moe/prefill.py:47,128`; `q36/moe/decode.py:66`). On a rank-3 input that value is **2048**, irrespective of T.

For prefill `[1,128,2048]`:

- Expert prefill infers S=2048, passes its multiple-of-32 assertion, and splits dimension 2 into 512-wide pieces. This splits hidden channels, not tokens (`prefill.py:137–147`).
- The first hidden piece is `[1,128,512]`, containing 65,536 elements.
- `_process_prefill_chunk` infers `chunk_len=512` and requests `[1,16,32,2048]`, containing 1,048,576 elements (`prefill.py:47–53`). These volumes cannot agree. Routing has its own real token extent, so its split sequence is also inconsistent.

The integer `ttnn.split` argument is a split **size**, not a number of chunks; its C++ overload builds sizes from the requested dimension without moving axes (`ops/data_movement/split/split.cpp:436–461`). Thus the channel-split calculation above follows the actual wrapper contract.

For decode `[1,1,2048]`, the code chooses `batch_size=2048` and later reshapes a one-token sparse output as though it contained 2048 token rows (`decode.py:66,97–121`). An earlier kernel/resource validation may fail first; no successful numerical result can satisfy these contradictory extents.

This affects **both** layer kinds. I explicitly checked an apparent escape hatch: `ttnn.transformer.concatenate_heads` does **not** leave full attention rank 4. `ops/transformer/concatenate_heads/concatenate_heads.cpp:44–46` calls the primitive and then squeezes axis 1, yielding `[B,T,Q*D]`. Full attention therefore returns rank 3 before MoE as well.

Changing the public input to `[1,1,T,H]` is not a solution: the adapter reads axis 1 as T, and the attention implementations also read the first two dimensions as B,T.

### Smallest local repair

Add an autoport-local MoE boundary that flattens independent token rows to `[1,1,B*T,H]` (decode: `[1,1,B,H]`), calls the routed MoE with the explicit mode, and restores `[B,T,H]` before the residual add. Preserve actual logical token counts, including trimming any tile padding used for expert kernels. Do not flatten B into time through GDN or attention; only the tokenwise MoE permits this.

Audit the shared expert at the same boundary: it reuses single-device `Qwen36MLP`, whose T heuristic is `x.shape[1]` (`q36/mlp.py:204–206`). With a rank-4 MoE input it always sees T=1, choosing decode compute/L1 policy even during prefill. That heuristic is not itself a proof of wrong mathematics, but a naive rank fix leaves it wrong for resource planning. Keep the local shared-expert call in its intended rank-3 shape or fix its local mode/sequence interpretation, and reshape the sigmoid-gated result explicitly.

## 3. Default GDN prefill changes the model's recurrence

`experimental/ttnn_delta_rule_seq.py:565–589` defaults `QWEN_GDN_DIAG_ALPHA` to `0.25`. Let

`A = (k_beta @ k^T) * causal_decay_mask`.

HF constructs the inverse of `I + strictly_lower(A)`: HF lines 283–294 mask the diagonal and upper triangle before its triangular solve. The inherited TT path constructs

`L = I + strictly_lower(A) + alpha * diag(A)`.

Its subsequent row normalization is **not** a cancellation: both RHS terms are divided by `diag(L)` (`seq.py:593–615`), so solving the normalized system still solves this modified L. `QWEN_GDN_INV_DOUBLING != 0` instead keeps the entire diagonal term, which also changes the model.

### Lowered operation confirms that the change survives

- Python passes the normalized matrix, normalized RHS tensors, diagonal block inverses, decay tensors, and initial state to `ttnn.transformer.gated_delta_attn_seq` (`seq.py:683–734`). They are float32, tiled, DRAM tensors.
- `ops/transformer/gated_delta_attn/gated_delta_attn.cpp:11–35` forwards those tensors; it does not remove regularization.
- Validation requires exactly C=Dk=Dv=128 and validates all supplied shapes/dtypes (`device/gated_delta_attn_device_operation.cpp:15–72`). These dimensions **match this checkpoint**.
- The program maps one flattened (batch, value-head) row per core and passes Ct=Kt=Vt=4 (`device/gated_delta_attn_program_factory.cpp:32–63,134,192–208`). The compute descriptor is explicitly HiFi2 with FP32 destination accumulation at lines 174–183, regardless of Python preprocessing's HiFi4 config.
- The selected compute kernel performs forward substitution for both RHS tensors, then `v_new=v_cor-k_cum@S`, output formation, and `S=S*decay+k_decay_t@v_new`. There is no compensating diagonal correction (`device/kernels/compute/gated_delta_attn.cpp:7–18,244–272,280–435`).

For a normalized key, beta=0.5, and initial state zero, the first state-write coefficient is:

- HF / recurrent decode: `0.5`.
- Default chunk prefill: `0.5 / (1 + 0.25*0.5) = 0.444444...`.

This is an approximately 11.1% reduction in that **core state update**, not a claimed 11.1% decoder-output error. Gated RMSNorm can partially hide scalar output differences. It cannot restore the different stored recurrent state or make subsequent updates generally equivalent. The T=1 decode implementation uses the direct undamped delta rule (`experimental/ttnn_delta_rule_ops.py:451–479`), so prefill and decode also differ in recurrence policy.

### Smallest local repair

Make the autoport's exact recurrence explicit: unit diagonal, strictly lower interaction matrix, and no environment-dependent diagonal damping. An eager diagnostic control is `QWEN_GDN_DIAG_ALPHA=0` with `QWEN_GDN_INV_DOUBLING=0`; that is not a substitute for a local explicit implementation policy. Keep existing scan kernels if using the exact preprocessing, or establish a baseline with the existing on-device recurrent formulation.

The newer `q36/gdn/fused_chunk.py` is a potential reuse candidate, not the path this single-device adapter reaches. Replacing the class with TP/fused machinery wholesale is not the smallest repair. Its own padded-length wrapper still uploads Torch masks at lines 155–169, and its constants must be built outside capture. Hardware parity of any replacement remains to be demonstrated.

## 4. Paged cache: dispatch, position, length, and ownership drift

### 4a. Binding a cache does not ensure prefill populates it

`bind_paged_kv_cache` merely stores external cache references and sets `use_paged_attention` (`adapter:196–199`; `q36/attention/gated_attention.py:120–124`). That class selects paged prefill only if `T>1` **and** `chunk_page_table is not None` (lines 77–95). Otherwise it takes concat prefill, omitting the bound cache and page table from the helper call (lines 96–113).

Consequently, an aligned `prefill_forward(x, page_table=pt)` with a bound cache and no chunk table populates `past_key/past_value`, not the external cache. The next T=1 decode with a page table uses the external cache and cannot read the prefilled prefix from the private concat cache. The adapter's check only requires a chunk table when logical and physical lengths differ, so this aligned case is admitted.

There is a second missing precondition: supplying a chunk table without `page_table` still reaches the helper, whose paged branch requires both (`experimental/ttnn_gated_attention.py:283`); it can fall back again. The default `chunk_start_idx=None` is also forwarded rather than normalized to zero, even though actual chunked SDPA requires a scalar or tensor start (`ops/transformer/sdpa/device/sdpa_device_operation.cpp:186–218`).

**Repair:** local dispatch must validate a complete cache/table/start contract before any state mutation, default first-chunk start to zero, and consistently use the bound cache. For an aligned first chunk, the full selected request table can be its fill table; derive later chunk tables on device or require explicit correctly scoped metadata. Do not silently change cache ownership based on an optional argument.

### 4b. Fill has no intra-page start offset; SDPA uses the absolute start

The helper calls `paged_fill_cache(..., chunk_page_table, batch_idx=0)` at lines 291–298, then calls chunked SDPA with `chunk_start_idx` or its device tensor at lines 302–325. The start is **not passed to the fill operation**.

The lowered writer computes

`virtual_block = seq_tile_idx / block_size_t`

`block_row_offset = seq_tile_idx % block_size_t`

and writes to the selected physical block at that offset (`ops/experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp:20–37,202–220,269–287`). Local input token zero starts at row zero of the first listed page. There is no hidden adjustment for the attention's absolute chunk start.

Concrete example: page size 128, continuation start=160, T=128, and a chunk table beginning with the page containing absolute token 160. Fill writes local token 0 at absolute position 128; SDPA reads it at 160. The previous 32 tokens in that page can be overwritten. Start=128 is the valid contrast: a properly rebased table then starts on the required page boundary. `q36/attention/tp.py:918–965` independently contains an exact-position update path for this mismatch; the single-device adapter never exposes it.

Start=129 additionally causes the scalar program-config helper to select q/k chunks of size 1 (`experimental/ttnn_gated_attention.py:82–91`), which C++ rejects because chunk sizes must be positive multiples of 32 (`sdpa_device_operation.cpp:340–370`). A runtime start tensor alone does not repair cache writes.

**Repair:** fill only page-aligned prefix ranges with whole-tile ownership. Handle nonaligned real tokens using exact-position device cache updates and an attention path that supports those positions, such as ordered per-token paged decode for a functional baseline. A device-side read/modify/write tail path is another option. Adjacent tokens of one request must not be treated as independent parallel cache-update users: sibling exact-position code documents the tile-level write conflict and serializes them.

### 4c. A page table cannot express arbitrary real-token length

`valid_len` is not passed to single-device full attention (`q36/layer.py:280–290`). The helper derives `page_len = chunk_page_table.shape[1] * cache_block_size` and slices only at that page extent (`experimental/ttnn_gated_attention.py:291–295`).

For L=129, physical T=256, block size=128:

- One page represents 128 rows and omits real token 128 from the cache.
- Two pages represent 256 rows and include 127 padding rows.

There is no page-table width representing 129 rows. The fill writer copies complete 32-row tiles. The optional newer `valid_seq_len` facility is for bounded circular fills and rounds to tile/block boundaries; it is not an arbitrary-row write mask (`paged_fill_cache_device_operation.cpp:157–179`; writer lines 150–159).

Writing padding into unused future slots of an exclusively owned final page is not automatically wrong: causal reads bounded to real positions can ignore it, and later decode can overwrite it. Thus this report does **not** claim that every padded initial full-attention prefix has incorrect real-token outputs. The definite gaps are the promised real-token-only write contract, returned padding, missing length ownership, and incorrect continuation/overlapping tail handling. Trimming outputs alone does not repair those.

### 4d. Request rows and cache dtype require an explicit contract

All fill calls use `batch_idx=0`; the public adapter has no user/slot selector. Nonzero **physical page IDs** work through table indirection. Nonzero **request rows** in a multirow table are a separate coordinate and are not selected by the adapter. A caller can preselect a one-row table for B=1, but that is an additional ownership convention, not general batched support.

The fill factory rejects B>1 without `batch_idx_tensor` (`ops/experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.cpp:163–190`). Therefore this path does not silently fill every row just because K/V have a batch dimension.

Fill is also a raw tile copy and requires source/cache dtypes to match (`paged_fill_cache_device_operation.cpp:32–45`). The reused attention path assumes BF16 K/V and performs no cache-dtype conversion. Binding a BFP8 cache does not make fill convert it. Prefer explicit BF16 cache validation for the functional baseline or an explicit local conversion with separate accuracy evidence.

`reset_request_state` calls `reset_cache`, which clears only private concat references and leaves external buffers/bindings intact (`adapter:201–205`; attention class lines 115–124). That is consistent with external cache ownership and is **not** evidence that reset must zero the entire externally owned allocation. Define reset as clearing only the selected request's recurrent state and logical progress; do not erase other requests or replace trace-bound buffers.

## 5. B>1 requires changes beyond the MoE reshape

### GDN has fixed B=1 state

`q36/gdn/gated_deltanet.py:50–65` allocates recurrent state `[1,32,128,128]` and fused conv history `[1,3,8192]` plus same-shaped reset buffers. No adapter argument changes those allocations.

At B=2, FIR concatenation needs `[2,3,8192]` history alongside `[2,T,8192]` input; concat does not broadcast the batch dimension. Even bypassing that, the scan adapter reshapes initial state to `[B*H,128,128]` (`seq.py:171–174`), doubling its required volume. Finally, state copyback tries to reshape new state back to the fixed old shape (`q36/gdn/decode.py:82–90`). These are independent concrete contradictions.

Per-row lengths also cannot enter through the adapter: `int(valid_len)` requires a scalar (`adapter:220`), despite lower helpers accepting lists. Fixing only that conversion still leaves host masks and state geometry broken.

Allocate persistent state for the admitted batch/slots, and support slot-scoped reset and continuation. A local implementation can prefill users independently and copy results into stable slot buffers. Do not adopt the old seq kernel as an unlimited batched primitive: it maps `BH=B*32` rows to individual cores and rejects `BH > grid.x*grid.y` (`gated_delta_attn_program_factory.cpp:48–59`). Grouped calls provide a local fallback without changing the kernel.

### Full-attention paged decode transposes the wrong axes

After head formation, Q is `[B,16,1,256]`. The code transposes axes 1 and 2, producing `[B,1,16,256]`, whereas decode expects `[1,B,16,256]` (`experimental/ttnn_gated_attention.py:355–371`). B=1 conceals the error.

The actual C++ path makes this more concrete than a documentation mismatch:

- The paged wrapper forwards Q without canonicalizing those axes (`ops/transformer/sdpa_decode/sdpa_decode.cpp:150–172`).
- The factory obtains B from the page-table rows for interleaved paged attention (`device/sdpa_decode_program_factory.cpp:122–133`).
- Output specification starts from Q's logical shape and replaces only axis 1 with that B (`device/sdpa_decode_device_operation.cpp:588–634`). Thus B=2 produces **`[2,2,16,256]`**, not `[1,2,16,256]`. The kernel's batch work is derived from B=2, not from this extra leading dimension.
- The subsequent transpose and head concatenation turn the malformed shape into `[2,2,4096]`, while the gate is `[2,1,4096]`. Broadcasting cannot restore a one-token-per-user output.

Do not cite the non-paged `Q.shape[0]==1` validation as proof of an early paged rejection; it is in a different branch. The output-spec contradiction applies to this actual paged path regardless of whether a different validation fails first.

**Repair:** explicitly reshape/permute Q to `[1,B,QH,D]`, then transform the output back to `[B,QH,1,D]` before head concatenation and gating. Build cache-update shard grids across available rows/columns; the current one-row grid ending at `(B-1,0)` (`gated_attention.py:342–344`) also fails once B exceeds device width. Validate slot/page-table and position-vector correspondence.

## 6. Public length handling and bounded context processing are missing

`physical_prefill_length` is a planner helper, not padding. `prefill_forward` rejects any physical T not divisible by 128 and returns the entire layer output (`adapter:190–194,219–242`). It neither pads hidden states/RoPE nor slices back to logical length. The current interface therefore does not accept raw `[B,129,H]`, and `[B,256,H], valid_len=129` returns physical-length activations after the other defects are fixed.

The forwarded `chunk_size=128` is the GDN scan's internal mathematical tile/chunk size. It does **not** bound the length handled by the Python projection, FIR, layout transforms, or preprocessing tensors in one call.

For T=262144, B=1, Nv=32:

- The scan has NC=2048; each `[BH,T,Dv]` float32 output is `32*262144*128*4 = 4,294,967,296` bytes (4 GiB).
- `experimental/ttnn_delta_rule_seq.py:737–747` explicitly requests that output in L1 during reshape; lines 131–158 and 194–197 also request whole-tensor L1 layout conversions.
- Several `[BH,NC,128,128]` preprocessing matrices are likewise 4 GiB each, allocated before scan dispatch.
- Blackhole's per-core L1 size is 1536 KiB (`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:33`). A device-wide 4 GiB L1 activation is not a viable native-context execution plan. The exact earliest allocation failure was not measured.

The reshape wrapper does not make an explicit DRAM-to-L1 request disappear as a metadata-only view: its view eligibility checks matching L1 placement (`ops/data_movement/reshape_view/reshape.cpp:713–714`), and its memory-change paths materialize the requested placement. The earlier whole-tensor untilize into L1 is independently unbounded.

**Repair:** the autoport owns bounded outer chunks independently of native context capacity. Keep global positions and caller-owned paged K/V at the full supported context, advance GDN state through each real chunk, and stage only bounded activations in L1. Do not cap context to 128/2048 merely because a kernel uses that chunk size. For full attention, align bulk chunks with page boundaries and handle arbitrary continuation/tails explicitly as described above.

## 7. Inherited precision is a policy choice, not established accuracy

| Component | Reached policy | Evidence |
|---|---|---|
| GDN projections, including mega projection and output | BFP8; LoFi, FP32 destination accumulation | `q36/gdn/weights.py:77–88,115–122,208–228`; `gated_deltanet.py:37–46` |
| Full-attention Q/K/V/O weights | BFP8; LoFi projection compute | `q36/attention/weights.py:18–27`; `gated_attention.py:26–35` |
| Routed expert gate/up | BFP4 | `q36/moe/weights.py:52–58,108–125` |
| Routed expert down | BFP8 | Same loader, lines 128–136 |
| Shared expert gate/up and down | BFP4 and BFP8; LoFi | `q36/moe/shared.py:35–37`; `q36/mlp.py:143–158,187–195` |
| Persistent GDN state | BF16 at construction/copyback; ordinary decode `high_precision=False` | `q36/gdn/gated_deltanet.py:50–65`; `experimental/ttnn_delta_rule_ops.py:484–508` |
| Chunk preprocessing / scan | Float32 tensors; preprocessing HiFi4, scan descriptor HiFi2 | `seq.py:413–419`; scan factory lines 174–183 |
| Router | BF16 weight; HiFi2/FP32 accumulation, but no requested float32 probability output | `q36/moe/router.py:20–55` |

For BF16 hidden input, router matmul defaults its **output** to the activation dtype, and softmax's output spec retains that dtype (`ops/matmul/device/matmul_device_operation.cpp:2980–2983`; `ops/normalization/softmax/device/softmax_device_operation.cpp:350–363`). FP32 accumulation is not FP32 stored probabilities. HF computes softmax/top-k normalization in float32 before converting the selected weights back (HF:784–790). BF16 probabilities can introduce ties near the eighth/ninth expert boundary; actual route changes on these weights were not measured. GDN prefill can have different intermediate dtypes, so this is not a claim that every router invocation is necessarily BF16.

Setting `QWEN_GDN_FP32_STATE=1` alone is insufficient for this single-device class: it changes the temporary state returned by the seq adapter (`seq.py:199–203`), while persistent storage remains BF16 and copyback targets that original storage. High precision must be an end-to-end local policy, including state allocation and update computation.

**Repair:** expose an autoport-local functional precision policy. Load a BF16 weight baseline directly from checkpoint tensors, use appropriate high-fidelity/FP32 accumulation, preserve FP32 recurrence and decay where needed to match HF, and use float32 routing probabilities through selection/normalization. Cast at explicit module boundaries. Do not upcast already quantized BFP4/BFP8 weights and call that a BF16 baseline. Namespace cached tensors by model identity, precision, and layout when changing policy. Measure the whole real decoder and component deltas before retaining any lower precision.

No evidence here proves that BFP4 alone violates a particular PCC threshold. The definite issue is inheriting it without a baseline or exposed policy; the diagonal-damped recurrence is a stronger, independently proven semantic discrepancy.

## State, trace, and completion ledger

| Boundary | Current behavior | Consequence / required invariant |
|---|---|---|
| GDN construction | Allocates zero recurrent and three-token conv buffers once | Good B=1 identity/address ownership; wrong capacity for B>1. |
| GDN prefill | Reads previous conv/recurrent buffers, derives real tail, copies final state into original buffers | Continuation exists; do not replace it with zero state on every chunk. Fix masking, recurrence policy, and batching while preserving this behavior. |
| GDN T=1 decode | Conv and recurrent helpers copy into persistent buffers | Suitable source structure for trace-stable state; hardware trace support still needs a complete replay test. |
| GDN reset | Copies zero templates into the same allocations | Correct address-preserving pattern; generalize to selected slots. |
| Full-attention binding | Stores external K/V objects | Callers own allocation and pages; trace captures their addresses. Rebinding requires a deliberate recapture/rebind protocol. |
| Full-attention concat decode | Replaces Python `past_key/past_value` with growing tensors | A captured decode trace cannot rerun these Python assignments and dynamically grow caches. Use persistent paged caches for replay. |
| Full-attention paged decode | Reads device position/table and updates bound cache | B=1 source path avoids runtime Torch conversion; ensure persistent inputs/tables/positions and correct B axes. |
| Layer failure after attention | MoE runs after attention cache/state mutation | A failed call is not transactional. Reset/reseed before retrying a rejected functional test. |

No adapter forward/reset explicitly synchronizes the device. TTNN calls enqueue work; a host-side return or assertion is not a completed correctness check. Cache writers use NoC barriers before releasing consumed tiles, and the GDN scan communicates through explicit buffers, but those local barriers do not replace host-visible completion when validating results. Future tests must read outputs/state or synchronize after replay before claiming success.

The selected seq kernel's dimensions do not suggest a C/K/V tile-count mismatch for this model. Instantiated at B=1,T=128, it uses BH=32, NC=1, Ct=Kt=Vt=4: six matrix input streams each contain 16 tiles, decay one tile, inverse four one-tile blocks, and initial/final state 16 tiles per head. T=256 changes NC to 2, not these per-chunk counts. Reader counts (`reader_gated_delta_attn.cpp:103–200`), forward-substitution consumers, and final-chunk state selection (`compute/gated_delta_attn.cpp:244–277,419–435`) agree at this level. This was a focused count/ownership check, not a proof that all low-level arithmetic or scheduling is bug-free. No supplied hang warrants promoting a speculative queue bug over the concrete caller/preprocessing violations.

## Smallest coherent autoport-local repair plan

1. **Own composition and contracts locally.** Keep correct checkpoint remapping and reusable kernels. Add explicit `[B,T,H]` validation, the MoE-only row/rank bridge, logical output slicing, and fail-fast metadata checks before state mutation. Avoid process-global monkeypatches to shared Qwen36 modules.
2. **Own a functional precision/state policy.** Construct local weights/state with explicit dtypes. Reuse mathematical building blocks, not unproven low-precision constructor defaults. Preserve real Q/K/V dimensions, all 256 experts, top-8 routing, and the gated shared expert.
3. **Make GDN forward entirely device-resident.** Use exact recurrence, device FIR/tail extraction, and persistent per-slot buffers. Full aligned chunks may pass `valid_len=None`; handle real tails with device recurrent steps or a device mask. Precompute constants before capture. Bounded groups can support batches beyond the seq kernel's one-head-per-core limit.
4. **Give full attention one cache owner.** Use caller-owned persistent paged storage through prefill and decode. Derive/validate request table rows, logical positions, page capacity and dtype. Use page-aligned fills for bulk work and ordered exact-position updates/decode for arbitrary tails and starts. Never batch adjacent same-request token writes as independent tile owners.
5. **Separate kernel chunk size from request length.** Accept logical lengths at the public interface, orchestrate bounded work, and keep native context metadata. Initially a slower per-token tail/unaligned path is sufficient if every operation stays on device and state/output semantics match HF.
6. **Capture only after preparation.** Allocate stable input/output/state/cache and device position/table buffers, warm the selected paths, then capture. Replay with changed tensor contents, not replacement Python objects or shape growth. GDN state and full-attention positions must advance exactly once per token; request reset must preserve captured addresses.

These are proposals, not implemented fixes. A two-line `valid_len=None` change would address only aligned FIR uploads and leave the other findings intact.

## Validation needed after repair

The initially present autoport tests (`tests/test_functional_contract.py`) check config, padding arithmetic, checkpoint index keys, and shard selection; they do not construct or run a TT decoder. The new harness that appeared during investigation is noted in the concurrent-update assessment. Existing Qwen36 single-device layer tests (`models/demos/blackhole/qwen36/tests/unit/test_layer.py`) are shape/finite-value sanity checks. The standalone MoE test supplies rank-4 input directly (`models/demos/blackhole/qwen36/tests/unit/test_moe.py:45–50`), which bypasses the faulty integration boundary.

Use the actual pinned layer 0 and layer 3 checkpoint tensors and the installed HF layer with matching caches/RoPE. Validate output, selected routes where relevant, and post-call state/cache contents. Suggested matrix:

| Dimension | Required contrasts |
|---|---|
| Length | 1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 2200; longer bounded runs through 262144 when physical capacity is established |
| Continuation | Monolithic vs splits at 1/3/127/128/129 and page boundaries; prefill→prefill→decode; decode→continuation; fresh/reset equivalence |
| Batch | B=1 and B=2 minimum; later admitted larger B; compare each row to an independent run; unequal lengths and selected-slot reset |
| Paging | Permuted nonzero physical pages; nonzero request rows; page sizes 32/128; starts 0/128/160/129; guard values in other users' pages and unowned tails |
| Dispatch | Bound cache with omitted chunk table; missing required full table/position; valid first-chunk default; reject incomplete metadata before mutation |
| Trace | Multiple decode replays with changed hidden input, RoPE, positions, and table contents; compare to eager; reset/reseed in place; verify output and state after synchronization |
| Host boundary | Instrument runtime forward helpers to reject Torch/from_torch/to_torch while allowing construction and test I/O; include padded and unpadded calls |
| Precision | Baseline first; lower each policy independently; whole-decoder output and recurrent/cache comparisons, not merely finite values |

No hardware acceptance, accuracy threshold pass, or performance claim can be inferred from this report's static checks.

## Other Potential Issues and ruled-out suspects

- **Convolution split constants:** `q36/weight_mapping.py:29–30,95–103` hardcodes Q=K=2048. Those values match this checkpoint; this is not a 9B-to-35B bug here. `FULL_ATTENTION_LAYERS` is not used by the remapping dispatch; actual layer kind comes from configuration.
- **K−1 versus K conv history:** TT stores three previous tokens for a width-4 FIR; HF stores a width-4 window and prepends it before slicing continuation output. Three prior inputs are sufficient for the next causal output. The FIR real-tail window starts at `valid_len` in `[old_history(3),x]`, so even lengths 1 and 2 retain the required old history. Do not replace it with `x[max(0,L-3):L]` without the missing prefix.
- **GDN continuation is not missing wholesale:** `q36/gdn/decode.py:54,70,82–90` passes and updates both persistent states. The faults are policy, lengths, batch capacity, and runtime host preparation.
- **Repeated GDN heads:** q/k are repeated twice from 16 to 32 in the reached single-device path, matching HF. The seq adapter then sees Hq=H=32 and does not repeat them again.
- **Norm conventions:** decoder and full-attention Q/K norms apply `1+weight`; GDN output norm uses the raw standard RMSNorm weight. Source ordering matches HF; numerical precision remains separate.
- **Expert gate/up order:** checkpoint `[gate|up]` is deliberately repacked as `[up|gate]` for TTNN SwiGLU (`q36/moe/weights.py:139–142`), whose helper uses that order. This is not evidence of swapped activation/gate math. Top-k ordering is also harmless if expert IDs and weights remain paired; precision-induced selection changes are a different question.
- **Decay clipping and output clipping:** seq preprocessing clamps exponent arguments to `[-20,0]` (`seq.py:539–554,638–653`); GDN clips SiLU gate activations and its normalized gated output to ±1e4 (`ttnn_gated_deltanet.py:28–34,757`). HF does not apply those exact clamps. They are additional semantic/numerical deviations, but no inspected real activation proves they trigger materially here. Keep secondary to the always-active default diagonal change.
- **Memory and precision promotion:** device casts, aliasing/deallocation, and BFP cache reload behavior need checking when implementing the local baseline. This report does not infer a use-after-free solely from an explicit `deallocate` after a TTNN operation.
- **RoPE ownership:** the adapter accepts cos/sin but does not build them. The actual checkpoint rotates 64 of 256 dimensions with theta=10,000,000 and its configured multimodal sections; the generic helper derives rotary width from cos/sin. Do not reuse an incorrectly shaped default table. There is no supplied adapter caller proving a wrong table was used.
- **TP sibling evidence is not this path:** the adapter explicitly admits one device and takes single-device constructors. TP fixes, batched state helpers, exact-KV updates, and newer fused GDN kernels do not automatically apply to it.

## Report review

Completed a second inspection after writing the first draft:

| Claim challenged | Recheck and verdict |
|---|---|
| FIR host work might be a dead fallback | Mega projection and cached taps are created by the selected constructor; T>=128 enters FIR with the adapter's non-None length. **Confirmed reachable.** |
| Full attention or normalization might repair rank before MoE | Public head concat squeezes axis 1; RMSNorm preserves logical shape. Integer split retains the specified channel axis. **Confirmed integration mismatch for both layer kinds.** |
| Diagonal damping might cancel during normalization | Both RHS terms carry the reciprocal diagonal into the actual forward-substitution kernel, and its state update has no inverse compensation. **Confirmed recurrence change.** |
| Paginated fill might secretly consume absolute start | Writer derives offsets only from local sequence tile and page table; no start is passed. **Confirmed nonaligned write/read mismatch.** |
| Every padded fill might be wrong | Future slots can be harmless if exclusively owned and excluded by causal bounds. **Narrowed to exact-write/length contract and unsafe continuation, not universal initial-prefix corruption.** |
| Paged decode might normalize Q axes | Wrapper forwards Q; output spec copies axis 0 and inserts page-table B on axis 1. **Confirmed B>1 shape inflation.** |
| GDN reset/continuation might drop history | Constructor allocates once; forward reads and copies both states in place. **Ruled out as a general B=1 fault.** |
| Removing `valid_len` is a behavior-neutral workaround | Tail rows agree, but short-sequence placement changes to L1. **Repair recommendation amended to preserve explicit placement.** |
| Large reshape might ignore the requested L1 config | C++ view eligibility requires matching L1/DRAM placement. **Confirmed unbounded resource request; earliest failure remains unmeasured.** |
| Low precision proves a PCC failure | Only policy and dtype contracts are available. **Kept as a baseline/evidence gap; no numeric failure claimed.** |

The final checksum check found concurrent adapter changes, so the initial and later call paths are separated above. The seq and HF hashes are unchanged; no tracked source diff was introduced by this investigation. Hardware-dependent follow-ups remain explicitly unverified.
