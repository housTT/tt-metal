# Decoder fused-collective audit

The installed fused kernels execute correctly on this four-chip ring, but they
are not drop-in replacements for the selected decoder programs.

- Fused matmul/reduce-scatter is algebraically legal for all row projections
  (`M=32`, local K 1,536/4,352/4,608, global N 5,120). It requires a
  `MinimalMatmulConfig`, three global semaphores, dedicated reduce-scatter
  cores, a barrier, and persistent output ownership. The selected path uses L1
  width-sharded activations, DRAM width-sharded weights, and a DRAM-sharded
  matmul program; substituting the fused op changes the local winner. Its
  ternary residual add also requires the residual format to match BFP4/BFP8
  weights, which rejects the BF16 residual.
- Fused all-gather/minimal-matmul is algebraically legal for fractured
  `[32,1280]` inputs with K-block 8. It requires two global semaphores, the
  two-link/four-worker Blackhole contract, and a trace-stable persistent
  `[32,5120]` gather buffer. Gate/up must share that buffer to avoid performing
  the gather twice. None of those resources is owned by the current TTNN
  all-gather or optimized DRAM-sharded projection programs.

The smallest future decoder-faithful A/B would use an 8x2 fused-RS grid with
`Mblock=1,Kblock=8,Nblock=4` and an 8x8 fused-AGMM grid with the same blocks,
two links, and four workers/link. That is a new graph-fusing implementation,
not a safe multichip-decoder substitution. The generic installed-op probe
already passes PCC >=0.9999869 (fused RS) and 0.9999915 (fused AGMM), so the
rejection is the exact program/layout/semaphore/persistent-buffer contract,
not unsupported hardware or dimensions.
