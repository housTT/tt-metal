# Runtime fallback audit

Status: clean for the completed multichip decoder scope.

Audited files:

- `tt/multichip_decoder.py`;
- `tt/optimized_decoder.py` as the required TP1 baseline;
- `tt/fused_decoder.py` as the TP2/TP4 layer shell;
- the canonical GPT-OSS attention, CCL, router, sparse expert, and RMSNorm
  modules instantiated by the multichip path.

Findings:

- TP1 construction delegates directly to `OptimizedDecoder.from_state_dict`.
  There is no functional-decoder or PyTorch forward fallback.
- TP2/TP4 forward paths contain TTNN device operations only. PyTorch is used
  during construction to repack checkpoint tensors, assign rank-zero-only
  biases, and create immutable zero sparsity metadata. It is not called during
  prefill, decode, or trace replay.
- The custom TP4 attention changes only the decode output-collective tail. QKV,
  RoPE, paged-cache writes, page-table/current-position handling, SDPA, head
  concatenation, O projection, and bias remain TTNN operations.
- Decode routing remains `TopKRouter` with model top-k 4. Expert projections use
  indexed `ttnn.sparse_matmul` and never instantiate throughput or dense
  all-expert decode. The canonical dense expert weight tensors are deallocated
  after the compact rank-local indexed representation is loaded.
- Prefill deliberately executes the repository packed 128-expert formulation;
  this is not a decode fallback and is covered by real-weight PCC/performance.
- The static multi-user decode loop splits a fixed captured device tensor and
  invokes the same device-only top-4 graph per logical user. It does not copy
  activations or routing results to the host.
- All row-parallel reductions use TTNN CCL on mesh axis 1. The CCL manager's
  selected topology and link count flow into the TP4 physical-hidden helper.
- Public outputs remain replicated logical tensors. No `to_torch`, `from_torch`,
  checkpoint reload, Python reference computation, or silent single-device
  reroute occurs in forward methods.

Automated guard: `test_runtime_fallback_and_active_expert_audit` checks the TP1
constructor seam, the absence of a functional fallback, sparse active-expert
markers, policy flow, and the TP4 physical-hidden attention specialization.
The profiled final TP4 graph independently reports 76 device operations and
zero host operations in decode.
