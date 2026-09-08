# AutoFix: vLLM server evidence provenance

Verdict: **fixed.** The coordinating agent executed the prescribed clean
evidence reruns after this focused investigation. No serving implementation
defect was established.

## Starting evidence and experiment

Inspected `AUTODEBUG.md`, the stage README/work log, shared runner and live
check sources, local readiness artifacts, and Git at
`c718255087bc62907ac2882fdfe3fc7eda613c42` on 2026-09-08. The initial worktree
already contained three determinism JSON reference corrections and the
qualitative-control wording correction. Later coordinating-agent changes are
outside this inspection; the measurements below describe the initial files.
Only this report was written. No hardware, server, reset, or live test was run.

Hypothesis: the generic retained logs describe the determinism-only lifetimes,
not the earlier final gates, and the final client logs are ignored/uncommitted.

Executed from the model directory (the actual calls used equivalent full paths):

```bash
for profile in P150 P150x2 P150x4; do
  sha256sum readiness_vllm/$profile/server.log.gz \
    readiness_vllm/$profile/server_logit_determinism_final_20260908.log.gz
  gzip -cd readiness_vllm/$profile/server.log.gz |
    awk 'NR==1 {print} /POST/ {n++} /09-08/ {last=$0} END {print last; print n}'
  for name in sampling_tests.log vllm_benchmark.log vllm_ci_serving_benchmark.log; do
    stat --printf='%n | %s bytes | %y\n' readiness_vllm/$profile/$name
    sha256sum readiness_vllm/$profile/$name
    git check-ignore -v readiness_vllm/$profile/$name
    git ls-files --error-unmatch readiness_vllm/$profile/$name
  done
done
sha256sum readiness_vllm/P150x2/server_full_sampling_20260908.log.xz
xz -cd readiness_vllm/P150x2/server_full_sampling_20260908.log.xz |
  awk 'NR==1 {print} /POST/ {n++} /09-08/ {last=$0} END {print last; print n}'
git ls-tree -r --name-only HEAD readiness_vllm
```

The nine `git ls-files --error-unmatch` failures are the expected experimental
result: all nine paths are untracked and ignored by `.gitignore:7:*.log`.
`git ls-tree` confirms that the server archives, but none of those client logs,
are in HEAD.

## Results

Each generic archive and its named final-determinism archive have the same
SHA256, including the compressed bytes:

| Profile | SHA256 of both archives |
| --- | --- |
| P150 | `30a75b86d3eea9b10edf998ea9f25231c19e1180cbc0c4bf622c89ebd5c5c8c9` |
| P150x2 | `edb0755858bf4786a374f5e886e4ee53a154054d8ac2f523e4e4fa1733038e9a` |
| P150x4 | `bcd5e2e40510fb6c93fad6b0d6f8a0893add40b1be8efe9dd38f83fe3d8e0e12` |

| Profile | First log timestamp, Sep 8 UTC | Last timestamp, Sep 8 UTC | API PID | POSTs |
| --- | --- | --- | ---: | ---: |
| P150 | 17:05:44 | 17:08:05.756 | 1681900 | 26 |
| P150x2 | 17:09:09 | 17:14:19.401 | 1682493 | 26 |
| P150x4 | 17:18:07 | 17:23:10.125 | 1683746 | 26 |

Each has 16 completion POSTs and 10 chat POSTs. This matches the determinism
source exactly: two sequential plus eight concurrent chat requests, and two
sequential requests for each of four direct prompts plus eight concurrent
direct requests. All three logs contain application shutdown and device close.

The earlier gate evidence cannot belong to those lifetimes:

| Profile | Sampling log mtime, UTC | Primary result embedded date | CI result embedded date | Feature / overlap JSON mtimes, UTC |
| --- | --- | --- | --- | --- |
| P150 | 12:36:54.643900193 | `20260908-124126` | `20260908-124205` | 12:27:14.403550252 / 12:38:21.632253637 |
| P150x2 | 13:08:08.670672430 | `20260908-144336` | `20260908-144410` | 14:41:13.570371085 / 14:42:34.032359383 |
| P150x4 | 15:01:15.915160136 | `20260908-154807` | `20260908-154839` | 14:49:31.751557540 / 15:47:05.749932290 |

Sampling results are 72 passed, one skipped in 577.19/629.87/701.04 seconds.
The qualitative JSON mtimes are about 15:55:14, and verdict mtimes about
16:17:43. These files lack embedded execution timestamps, so those mtimes
must not be presented as request timestamps. They still predate the retained
determinism lifetimes. Benchmark JSONs were also rewritten at about 15:55;
their embedded dates and client log timestamps preserve the earlier runs.

P150x2's separate sampling archive has SHA256
`9bb1f29ff4bf5046bd7dbfe89eceaaac7cb1425f2f8db050986eba8ef1a3634a`.
It spans 12:43:29–13:08:26 UTC, API PID 1642442, with 3,073 POSTs: 2,994
completion and 79 chat, including two HTTP 400 responses. Its startup records
block size 64, 32 sequences, context 262144, async scheduling, trace mode all,
device sampling all, and the documented FABRIC_2D parent/submesh config.
The archive also covers an earlier suite attempt whose local client log ends
at 12:55:37 with one failed/71 passed/one skipped; therefore 3,073 is the whole
lifetime count, not the final successful suite's request count. The archive
ends with an idle engine message and contains no shutdown/destructor record.
It preserves final sampling execution/configuration but not that lifetime's
cleanup, nor the later 14:41–14:44 gates.

All other retained server archives were inspected. The Sep 6 archives have
46/46/1555 POSTs; the earlier P150 Sep 8 determinism archive has 26; the TP4
heartbeat-failure archive has zero. None supplies the missing main lifetimes.
No archive search outside this model's readiness directory was performed.

The overwrite mechanism is explicit:
`models/common/readiness_check/run_vllm_server.py:253` opens the fixed
`server.log` in `wb` mode. Reusing an output directory for another serve stage
replaces its previous lifetime. A runner implementation change is unnecessary
for this evidence repair if every launch gets a fresh directory.

## Smallest definitive repair

First recover an exact original archive from other run records if one exists;
only an archive with matching timestamps/API PID and check mapping avoids a
rerun. Preserve the nine existing client logs immediately: each is only about
10 KB, so force-adding the plain files is sufficient and keeps current README
paths valid. Their absence from Git alone requires no live test.

When no original main lifetime is recoverable, run one new server lifetime per
profile. Within it, run feature checks, full sampling, qualitative prompts,
overlap, and both benchmarks, then shut down and preserve the complete log.
P150/P150x4 require all of these. P150x2 can omit sampling only if the retained
sampling archive is explicitly mapped separately and its missing historical
shutdown is documented. To close runtime/cleanup evidence entirely with one
lifetime, include P150x2 sampling too. Existing determinism and standalone
mutable-state/oracle evidence need no rerun for this hypothesis.

Use these exact fresh output subdirectories, subject to an absent-directory
check before launch: `P150/evidence_closure_20260908_01`,
`P150x2/evidence_closure_20260908_01`,
`P150x4/evidence_closure_20260908_01`. If already used, increment the suffix.
Keep the generated server filename `server.log` while running; after shutdown,
retain it losslessly as `server.log.xz` using `xz -T1 -9 -k server.log`. Capture
the coordinating commands/output in `runner.log`. Do not launch again into any
of these directories. These are proposed commands, not experiments executed
by this investigation.

From the repository root, set `profile`, `port`, `mesh`, `context`, and `config`
from this table and `subdir=$profile/evidence_closure_20260908_01`:

| profile | port | mesh | context | config |
| --- | ---: | --- | ---: | --- |
| P150 | 8000 | N150 | 50624 | `{"trace_region_size":220000000}` |
| P150x2 | 8001 | N300 | 262144 | `{"trace_region_size":220000000,"fabric_config":"FABRIC_2D","parent_mesh_shape":[2,2],"submesh_offset":[0,0]}` |
| P150x4 | 8002 | P300x2 | 262144 | `{"trace_region_size":220000000,"fabric_config":"FABRIC_1D_RING"}` |

Start and hold the server using the shared runner, retaining its PID:

```bash
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TT_METAL_HOME=/home/hous/dev/tt-metal \
  TT_GEMMA4_TEXT_VER=google_gemma_4_26b_a4b_it_autoport \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it --mesh-device "$mesh" \
  --max-num-seqs 32 --block-size 64 --max-model-len "$context" \
  --port "$port" --output-subdir "$subdir" --tt-config "$config" \
  --additional-server-args \
    '--async-scheduling --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4'
```

After `/health` succeeds, run these checks sequentially against that same PID;
retain each command's start/end UTC and exit code in the run record:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/tests/run_vllm_feature_checks.py \
  --server-url "http://127.0.0.1:$port" --model google/gemma-4-26B-A4B-it \
  --expected-max-model-len "$context" \
  --output "models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/$subdir/openai_feature_checks.json"

python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages sampling,qualitative --server-url "http://127.0.0.1:$port" \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it --max-num-seqs 32 \
  --sampling-profile full --output-subdir "$subdir"

python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/tests/run_vllm_async_overlap.py \
  --server-url "http://127.0.0.1:$port" --model google/gemma-4-26B-A4B-it \
  --output "models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/$subdir/async_overlap_state_test.json"

python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages benchmark --server-url "http://127.0.0.1:$port" \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it --max-num-seqs 32 \
  --output-subdir "$subdir" --benchmark-temperature 0 \
  --benchmark-prompt-len 128 --benchmark-output-len 128 \
  --benchmark-num-requests 1 --benchmark-concurrency 1 \
  --benchmark-ci-serving --ci-benchmark-prompt-len 100 \
  --ci-benchmark-output-len 100 --ci-benchmark-num-requests 32
```

Qualitative defaults are six prompts, 12 completions, max 256 tokens,
greedy `top_k=1`, sampled `top_k=32`/temperature 0.7/top_p 0.9. Do not set
`--ci-benchmark-concurrency`; its unset value is the required unbounded burst.
Feature/qualitative/overlap checks contribute 3/12/4 chat POSTs. The benchmarks
contribute 35 completion POSTs: 1+32 measured requests and one initial test
request per benchmark. Thus the non-sampling addition is 54 POSTs, absent
retries or extra requests. Record the observed sampling count independently;
do not assume the historical mixed-lifetime count of 3,073.

After the checks, signal the retained readiness-runner PID with SIGTERM, wait
for its child/device cleanup, and then compress. Record actual API/engine PIDs,
startup/shutdown UTC, request counts by endpoint/status, runtime fallback/fatal
audit, repository SHAs, exact launch/check commands, and every artifact SHA256
in `evidence_manifest.json`. A new cleanup audit must describe this new run.
Read all new qualitative outputs, regenerate their verdict and control
comparison references, and update the README/work log to the new metrics and
exact lifetime-to-artifact mappings before independent stage review.

## Exact artifact preservation

The following command, from the model directory, names precisely the nine
currently ignored final files to preserve; it was not executed here:

```bash
git add -f -- \
  readiness_vllm/P150/sampling_tests.log \
  readiness_vllm/P150/vllm_benchmark.log \
  readiness_vllm/P150/vllm_ci_serving_benchmark.log \
  readiness_vllm/P150x2/sampling_tests.log \
  readiness_vllm/P150x2/vllm_benchmark.log \
  readiness_vllm/P150x2/vllm_ci_serving_benchmark.log \
  readiness_vllm/P150x4/sampling_tests.log \
  readiness_vllm/P150x4/vllm_benchmark.log \
  readiness_vllm/P150x4/vllm_ci_serving_benchmark.log
```

| Profile / file | SHA256 before repair |
| --- | --- |
| P150 / sampling_tests.log | `20a29470719c9e48aa009d8b8ba39c1f84faa04f93178680d1177bbb93e0e46d` |
| P150 / vllm_benchmark.log | `c607ca9ddc401f617cc458c2711cfca077c43bb8159019f7fe6b5363d6b5a7cd` |
| P150 / vllm_ci_serving_benchmark.log | `21de271838041ac635d5131502e15bcaf4910344198160e20373bde4d7f57247` |
| P150x2 / sampling_tests.log | `bdb7d7e6a02b16039e1030fef23357fe706cd025b059ab0a6292ca477ef747fd` |
| P150x2 / vllm_benchmark.log | `a7ef8fa70a7ff26ae5aba582bde12295167e926de6f47cf34893a05697ef6037` |
| P150x2 / vllm_ci_serving_benchmark.log | `b2b7303e2d2a4a202aa5070fbf7333e019239d0ae8bed3d96d5f339ee4cb33b2` |
| P150x4 / sampling_tests.log | `5d8c46f1b7ff1a118a5d11494ea2ee0de3088e3736f68817bf841d66b340b551` |
| P150x4 / vllm_benchmark.log | `cd7133e2e7b00ac6b05b7d0c5d9c5e508c52a369d7dfc1b2f340a327b8868718` |
| P150x4 / vllm_ci_serving_benchmark.log | `768fc5e38f297c9f7df3f08392132eb76a23cad1eb4ae6ed1b5059b776675c07` |

For each new `readiness_vllm/$subdir`, force-add exactly `runner.log`,
`sampling_tests.log`, `vllm_benchmark.log`, and
`vllm_ci_serving_benchmark.log`. Add normally: `server.log.xz`,
`evidence_manifest.json`, `openai_feature_checks.json`,
`async_overlap_state_test.json`, `vllm_qualitative_outputs.json`,
`qualitative_degeneracy_check.json`, `qualitative_verdict.json`,
`vllm_result.json`, `vllm_benchmark.json`, `vllm_ci_serving_result.json`,
`vllm_ci_serving_benchmark.json`, and any per-run cleanup audit. Also retain
the updated cross-profile comparison, runtime cleanup audit, README/work log,
and independent review verdict. Check `git diff --cached --name-only` and
`git ls-files --error-unmatch` against the exact intended files before commit.

## Coordinating-agent verification and final status

The hypothesis was verified and the evidence defect was repaired on the
unchanged implementation. After a bounded device reset and successful 2x2 mesh
open/close smoke, the coordinating agent launched one fresh server lifetime per
profile. Against each single API PID it ran, in order, OpenAI feature checks,
the full sampling suite, device-compatible qualitative and degeneracy checks,
async overlap, numeric logit determinism, the primary benchmark, and the CI
serving-burst benchmark. It then stopped the readiness runner, observed device
close, and retained the complete server log as `server.log.xz`.

| Profile | UTC lifetime on 2026-09-08 | API PID | Sampling | POST/status audit |
| --- | --- | ---: | --- | --- |
| P150 | 17:49:09–18:04:15 | 1696137 | 72 passed, 1 skipped in 592.78s | 1,581 POST; 1,580 HTTP 200; 1 expected HTTP 400; 0 HTTP 500 |
| P150x2 | 18:06:03–18:21:27 | 1699866 | 72 passed, 1 skipped in 626.31s | 1,581 POST; 1,580 HTTP 200; 1 expected HTTP 400; 0 HTTP 500 |
| P150x4 | 18:23:11–18:39:39 | 1703317 | 72 passed, 1 skipped in 693.68s | 1,581 POST; 1,580 HTTP 200; 1 expected HTTP 400; 0 HTTP 500 |

The final process audit at 18:44:06 UTC found zero API-server, EngineCore, and
readiness-runner processes. All four local P300C devices were visible on
firmware 19.13.1. `readiness_vllm/unified_gate_manifest.json` maps the exact
configuration and SHA256 of every gate artifact to these lifetimes;
`runtime_cleanup_audit.json` and `hardware_recovery_20260908.json` retain the
cleanup and reset evidence. The nine ignored client logs named above are
force-added to the stage commit. The evidence repair is complete; independent
stage review remains the separate release gate.

The ancillary shared-runner finding was reproduced by inspection and fixed as
well: a tokenizer whose `chat_template` is absent now selects raw completions
before local template rendering. The focused host regression test
`models/common/readiness_check/test_run_vllm_server.py` passes and verifies the
raw endpoint, prompt format, rendered prompt, token IDs, and device-compatible
top-k settings.
