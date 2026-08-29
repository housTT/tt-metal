# Rejected PLE decode-row dedup candidate

This exact candidate specialized the decode lookup for 16 one-token PLE row
IDs and avoided the general `torch.unique` path. The retained prefill path,
mmap shard lookup, row cache, BF16 assembly/staging, EOS history, cancellation,
and request isolation were unchanged.

CPU microbenchmarks over identical row IDs showed a 6.89x cold-lookup gain and
a 1.14x hot-cache gain. The 18-test host-weight suite passed, including the
candidate-specific exact-row test (`ple_decode_dedup_cpu.xml`).

It was rejected after the real vLLM workload regressed. On P300 mesh 1x2,
`max_num_seqs=2`, `max_model_len=262144`, full sampling profile,
`sample_on_device_mode=all`, greedy temperature 0, and the standard single-user
128-input/128-output/1-request workload, mean TPOT was `270.037506 ms`
(`3.703189 t/s/u`), TTFT was `4145.433 ms`, ITL p50/p99 was
`271.026/304.695 ms`, and aggregate output throughput was `3.329819 token/s`.
That is slower than both the baseline TPOT (`268.980406 ms`) and the retained
direct-slot/async candidate repeat (`267.016417 ms`).

The source and its extra test were reverted. The measured result remains here
to prevent a CPU-only lookup win from being reintroduced without an
end-to-end serving gain.
