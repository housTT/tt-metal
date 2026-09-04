# Ornith bringup contract

This autoport implements text and agentic inference for `ornith-ai/Ornith-1.5-9B`.
The supervising user's full release requirements below apply in addition to the
current multigoal stage. Finish only the current stage; do not skip ahead.

- Base tt-metal main: `e7638d2859b6a1ef30eb984781cbddf9872a8d62`.
- Workflow `.agents/` only, imported from tt-metal branch
  `agentic-research/fast-models-fast`, commit
  `70a596f92229ada922fba743cd0cd9d2658a5c1c`.
- Branch `hous/ornith-1.5-9b`. Local commits only. Never open a tt-metal PR or
  push to main. The supervisor handles eventual HF publication after all gates.
- Pin all HF loads to revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.
  A pinned local snapshot is `/home/hous/dev/ornith-1.5-9b/upstream`.
  The supervisor records file hashes in `../state/provenance.json` relative to
  the tt-metal checkout root. Never silently resolve HF main again.
- This checkpoint is Qwen3.5-based with 24 linear-attention and 8 full-attention
  layers, 4096 hidden width, and a 248320-token vocabulary. Inspect the exact HF
  implementation and config; do not substitute an existing Qwen model.
- Implement both layer kinds and their recurrent/convolution and paged KV cache
  semantics, including continuity across prefill/decode. Vision/video is excluded.
- Hardware is four Blackhole chips on physical P300c boards. Profiles `p150`,
  `p150x2`, and `p150x4` name 1/2/4-chip meshes, not measured P150 boards.
  Disclose P300c hardware on all performance/release reports.
- Follow `$tt-device-usage`. Only one hardware workload at a time, including
  device listing, opening, tests, benchmarks, profiling, and servers. Independent
  review agents are read-only and must not access devices. Watchdog owns the
  task's hardware lock for this entire runner; never start competing lanes.
- Native context target is 262144 tokens independently for each profile.
  Any reduction requires physical device-memory evidence and the largest
  validated context in `doc/context_contract.json`; testing cost is no reason.
  Record per-profile limits and add opt-in YaRN targeting 1000000 tokens where
  capacity permits; disclose unresolved capacity/implementation limits.
- Preserve the upstream chat template verbatim, streaming, structured tool
  calls with `qwen3_xml`, separate `reasoning_content` with `qwen3`, prefix
  caching, and the model card's sampling controls. Resolve parser field/version
  differences with API tests; pin tested vLLM, Transformers, and TT-plugin.
- Validate each mesh independently: real-weight HF comparisons at workflow
  thresholds, cache continuity, non-aligned prompts, long context, concurrency,
  cancellation, repeated server use, streaming/non-streaming reasoning, tool
  names and JSON arguments, tool-result follow-ups, sampling, and prefix cache.
- Measure cold startup, warm prefill, TTFT, decode latency, throughput, and
  memory with commands and hardware provenance. Tune interactive latency and
  concurrent throughput separately, recording batch/block/context/precision/
  memory settings. Default to single-chip interactive serving.
- Preserve every verification gate. Do not weaken checks, waive accuracy/API
  failures without evidence, or call a stage done merely because a process exits
  zero. Stage 11 must validate this autoport. Label any permitted nightly subset
  accurately; never claim reproduction of every upstream benchmark.
- Keep stage logs, README, work log, checks, independent stage-review results,
  commits, and machine-readable measurements. Never delete runner manifests,
  Codex session state, or upstream pinned artifacts.
- Final package is one v5.1 container with all three profiles using
  `/home/hous/dev/tt-model-manager`, full runtime import/data closure and parser
  declarations. Weights are referenced, never embedded. Package build, serving
  on every profile, public HF upload to
  `tt-hous/ornith-1.5-9b-p150-p150x2-p150x4`, clean pull, and repeated API checks
  are separate required release gates managed by the supervisor.

Do not expose credentials in logs or process command arguments. Stop task-owned
servers after tests. Do not reboot the host without first handing the diagnosis
to the supervisor, which must preserve monitoring and resume state.
