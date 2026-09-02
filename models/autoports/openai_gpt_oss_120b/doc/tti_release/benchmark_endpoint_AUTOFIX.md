# Benchmark endpoint AutoFix

Date: 2026-09-02

## Failure

The original 21-row TTI sweep used the random dataset with the
`openai-chat` backend. vLLM forces `ignore_eos=true` for random datasets, so
GPT-OSS generation continued beyond Harmony's terminal return token. The
stream parser then returned HTTP 200 while logging Harmony errors and the
client recorded truncated, empty, or non-exact output lengths as completed.

The archived release contained 351 non-exact output lengths and 351 Harmony
errors (350 unexpected return-token errors and one call-token error). This
made its benchmark rows invalid even though the client reported zero failed
requests.

## Repair

- Added a validated `metadata.benchmark_endpoint` model-spec option.
- The default remains `chat_completions` for existing models.
- This generated autoport selects `completions`, producing
  `--backend vllm --endpoint /v1/completions` for all 21 sweep points.
- Added explicit `--temperature 0` to preserve legacy greedy benchmark
  semantics.
- Random completions retain the requested logical ISL/OSL and vLLM's intended
  exact-output `ignore_eos=true` behavior without routing raw tokens through
  the Harmony chat parser.

## Live discrimination

Both cases used the same warmed autoport server, eight random requests,
ISL=128, OSL=128, concurrency 1, and temperature 0.

| Endpoint | Completed/failed | Exact lengths | Harmony errors | TTFT | Output throughput |
| --- | ---: | --- | ---: | ---: | ---: |
| `/v1/chat/completions` | 8/0 | no; inputs 128/195 and outputs 33/38/47/128 | 3 | 4739.26 ms | 13.50 tok/s |
| `/v1/completions` | 8/0 | yes; every input/output is 128/128 | 0 | 497.58 ms | 45.06 tok/s |

Server-log evidence is bounded to lines 9214-9335 of the retained local
server log. Lines 9214-9293 contain the chat case and three Harmony errors;
lines 9302-9330 contain eight successful raw-completions requests and no
parser error. The large server log and raw generated texts are not release
copy-back artifacts.

## Host verification

- Driver/config focused suite: 46 passed.
- Full `tests/llm_module` plus API fixture coverage: 303 passed.
- Independent read-only review verified metadata propagation to all 21 sweep
  points, exact-output CLI semantics, backward compatibility, and alignment
  with the optimized-vLLM reference path; no blocking defect found.

The full corrected 21-row sweep must still complete before this repair is
accepted for release.
