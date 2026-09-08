# AutoFix: reject unsuccessful benchmark warmups

## Evidence and hypothesis

[The source audit](AUTODEBUG_benchmark_warmup.md) established that explicit warmups were missing and that vLLM discarded warmup result objects. A failed request returned with `success=False` could therefore print `Warmup run completed.` and enter the measured workload.

Before editing, a standard-library-only probe extracted the actual warmup block and following main-run marker from `vllm/benchmarks/serve.py`. A fake asynchronous request returned `success=False` with a synthetic error. The original code printed both the completion banner and `Starting main benchmark run...` without raising. The hypothesis was verified.

No `AGENTS.md` or `CLAUDE.md` was present in the sibling module's ancestor directories. The sibling checkout's contribution entry point and Ruff pre-commit configuration were inspected. Existing plugin worker/test changes were left untouched.

## Fix

Only [the benchmark module](/home/hous/dev/vllm/vllm/benchmarks/serve.py:723) changed: 13 insertions and one deletion. It now retains `asyncio.gather` results, prints `Successful warmup requests: <successful>/<requested>`, and raises `ValueError` with the count and indexed request errors if any warmup failed. The normal completion banner and main benchmark are reached only after every warmup reports success.

The request inputs, number of warmups, concurrency limit, sampling mode, endpoint ready-check behavior, and location of the measurement timer are unchanged. No retry or extra inference was introduced. Request exceptions still propagate. With zero warmups, behavior is unchanged.

Module SHA-256 for **both** forthcoming warmed baseline and candidate runs:

```text
4df3cbab368d59cc45ad3b9c55ccef01e9f103009ffbd1b06806406d541090b8
```

## Verification

[Host evidence JSON](benchmark_warmup_validation.json) records the original failing control, patched source hash, and all seven passing cases:

- One successful warmup reports `1/1` and proceeds.
- Four successes with concurrency two preserve the cap and close the progress bar.
- Four successes without a concurrency cap proceed.
- Two failures among four requests report both indexed errors, close the progress bar, and prevent the main run.
- All failures prevent the main run.
- Zero warmups issue no requests and preserve the original main-run behavior.
- A raised request exception propagates and prevents the main run.

The probe compiled only the actual AST warmup block and next marker, using fake request/session/progress objects. It asserted request-input identity, request counts, concurrency bounds, success/failure output, and that the benchmark timer remains after warmup validation. It imported standard-library modules only; neither the benchmark module nor TTNN was imported. No pytest or device operation was run.

From `/home/hous/dev/vllm`, both formatting checks passed using the already-installed pre-commit Ruff executable:

```sh
/home/hous/.cache/pre-commit/repo_hon4j4j/py_env-python3/bin/ruff check vllm/benchmarks/serve.py
/home/hous/.cache/pre-commit/repo_hon4j4j/py_env-python3/bin/ruff format --check vllm/benchmarks/serve.py
git diff --check -- vllm/benchmarks/serve.py
```

## Remaining serving evidence

The host validation proves failed result objects cannot silently admit the main benchmark. The live warmed baseline/candidate runs remain the parent's task. Their primary and CI logs must each show `Successful warmup requests: 1/1` before measurement, and their manifests must record the same benchmark-module hash. Continue using `--additional-benchmark-args='--num-warmups 1'` on both sides.

Success follows the endpoint request function's existing `RequestFuncOutput.success` contract; this patch does not redefine it or add a token-length gate. One CI warmup still covers a single request, not every burst batch shape. No new performance claim is made.
