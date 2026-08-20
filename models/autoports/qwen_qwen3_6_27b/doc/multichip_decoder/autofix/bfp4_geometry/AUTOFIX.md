# Linear BFP4 geometry autofix

The Blackhole report classified all three exact TP-local linear-MLP rows as
`SLOW`, so every legal precision-locked DRAM-sharded geometry was tested.

- Exact gate/up tiles are `K=160,N=136`; exact down tiles are `K=136,N=160`.
  Their greatest common divisor is eight, so eight is the only supported
  unpadded DRAM-worker count. Six gate/up K-block divisors
  `{20,10,5,4,2,1}` and both down divisors `{17,1}` were swept.
- All four devices passed against the FP32 reference at PCC 0.9930 or better.
  The repeat sweep in `sweep.log` selected gate/up block 10 (57.787 us) over
  20/5/4/2/1 (94.687/59.041/61.638/73.956/125.220 us), and down block 17
  (60.470 us) over block 1 (114.866 us).
- The only higher-worker alternative pads the local intermediate 4,352 to
  4,608 and uses 16 workers. It passed real-layer PCC, but its authoritative
  profile regressed gate/up/down from about 43 us to 45.7/45.8/46.3 us. The
  artifacts are `selected_padded16_{ops,report}_blackhole.csv`; it is rejected.
- Padding to 5,120 enables 32 workers but adds 17.6% inert intermediate work,
  more than the already slower 4,608 candidate. A 64-worker geometry needs at
  least 6,144 in both dimensions and about 69% weight/compute padding, so it is
  dominated without another hardware run.

The selected exact geometry is 8 workers with blocks 10/10/17. The selected
Blackhole report remains labeled `SLOW` because the report cannot derive an
output-subblock recommendation for this DRAM program family, but there is no
untested legal core/block/padding alternative. `selected_bw10_*` contains the
winning whole-layer profiler artifacts.
