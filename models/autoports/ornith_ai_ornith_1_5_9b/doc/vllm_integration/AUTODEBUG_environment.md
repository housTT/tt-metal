# AutoDebug: serving environment prerequisites

Date: 2026-09-05 UTC. Inspection-only investigation of stage 9, `ornith-ai/Ornith-1.5-9B`. Starting branch `hous/ornith-1.5-9b`, HEAD `85710be49fbcce5579aeb6d9571cd3bc1829d953`; initial `git status --short` was empty. This report does not establish serving readiness or model correctness. No installation, server launch, TTNN import, device open, reset, or hardware command was performed.

## Finding

The active interpreter cannot launch vLLM because neither vLLM nor its TT plugin is installed. The runner also needs `openai`, which is absent. No existing usable serving checkout or interpreter was found in the mounted locations inspected. Restoring the missing shared runner is a separate, locally fixable source issue; it does not supply these runtime dependencies.

The root user instructions explicitly prohibit installing compilers or dependencies. A pre-provisioned compatible serving environment, or explicit user authorization for dependency provisioning, is required to resolve that runtime gap. Do not substitute a fake registry or call adapter-only tests proof of serving.

## Exact experiments and evidence

1. Package-only probes, avoiding TTNN imports:

   ```bash
   python_env/bin/python - <<'PY'
   import importlib.util, sys
   print(sys.executable)
   for name in ['vllm', 'vllm_tt_plugin']:
       print(name, importlib.util.find_spec(name))
   PY
   ```

   Active interpreter `/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python`: both `None`. The same snippet under `/opt/venv/bin/python` and `../state/device-env/bin/python` also returned both `None`.

2. Exact failing launch-entrypoint probe, with no server or device reached:

   ```bash
   python_env/bin/python -m vllm.entrypoints.openai.api_server --help
   ```

   Exit 1: `Error while finding module specification for 'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')`.

3. Distribution metadata in active interpreter:

   ```bash
   python_env/bin/python - <<'PY'
   from importlib import metadata
   for name in ['torch', 'transformers', 'vllm', 'vllm-tt-plugin', 'openai', 'requests']:
       try:
           print(name, metadata.version(name))
       except metadata.PackageNotFoundError:
           print(name, 'NOT_INSTALLED')
   PY
   ```

   Result: torch `2.11.0+cpu`, transformers `5.12.1`, requests `2.34.2`; vllm, vllm-tt-plugin, openai `NOT_INSTALLED`.

4. Mounted runtime/source search:

   ```bash
   rg --files --hidden /home/hous /opt /usr/local/lib /root 2>/dev/null | rg '(run_vllm_server\.py$|vllm_tt_plugin/platform\.py$|vllm/__init__\.py$|pyvenv\.cfg$|vllm[^/]*\.whl$)'
   rg --files --hidden /work /opt/venv /opt/tenstorrent /home/hous/dev/tt-model-manager 2>/dev/null | rg '(vllm(/|[-_])|pyvenv.cfg|readiness_check/run_vllm)'
   ```

   No serving checkout, plugin `platform.py`, vLLM wheel, or importable runtime discovered. Search includes ignored files and hidden files; unreadable locations were not evidence of absence. The one `vllm/__init__.py` at `/home/hous/.local/lib/tt-inference-server/workflows/helm_generator/vllm/__init__.py` contains license comments only and belongs to a Helm generator, not the vLLM engine. `/home/hous/.tenstorrent-venv/bin/python` is a dangling symlink to absent `/usr/bin/python3.12`; invoking it returns shell exit 127. `/work` and this tt-metal root have identical inode `138312301` and are the same mounted checkout.

5. Container fallback inspection:

   ```bash
   ls -l /usr/bin/docker /usr/bin/sudo /var/run/docker.sock /opt/venv/bin/python* 2>/dev/null
   cat ../bin/docker
   ```

   `/usr/bin/docker` and `/var/run/docker.sock` do not exist. `../bin/docker` delegates to `sudo ... /usr/bin/docker`; it is not a usable Docker client here. No Docker command, credentials, or container mutation was attempted.

6. Missing shared source is locally recoverable:

   ```bash
   rg --files models/common/readiness_check | rg 'run_vllm|contract_vllm'
   git cat-file -e 70a596f92229ada922fba743cd0cd9d2658a5c1c:models/common/readiness_check/run_vllm_server.py
   git cat-file -e f7662055fe4ae3d66509335d96a7c74acd53911b:models/common/readiness_check/run_vllm_server.py
   git show 70a596f92229ada922fba743cd0cd9d2658a5c1c:models/common/readiness_check/run_vllm_server.py
   ```

   Initial working tree had no runner; both object probes exit 0. The skill-source commit supplies runner, corresponding test file, and degenerate-output checker. The runner has full/smoke sampling, primary `128/128/1`, secondary `100/100/32`, and TT config support. It imports `openai`, `requests`, and `transformers`; launches `[sys.executable, '-m', 'vllm.entrypoints.openai.api_server', ...]`; its sampling suite locator requires an importable `vllm_tt_plugin` or `vllm` plus adjacent TT pytest suite. Restoring source alone cannot fix launch. Its mesh choices are only N150, N300, T3K, TG; the four-chip Blackhole profile will need explicit integration rather than claiming those names prove physical hardware. During this investigation the parent began restoring runner/test source; those untracked files are parent-owned, not changes made by this report author.

7. A separate environment issue was observed when probing nested `find_spec('models.common.readiness_check.run_vllm_server')`: importing the readiness parent package reaches Transformers/PyTorch, then `getpass.getuser()` raises `KeyError: 'getpwuid(): uid not found: 1002'`. Avoid nested `find_spec` for existence checks. Provision `USER=hous`/`LOGNAME=hous` or a task-specific `TORCHINDUCTOR_CACHE_DIR` in the launch environment, consistent with previous-stage runtime setup. This is independent of missing vLLM and does not install anything. The stage owner independently reproduced the UID failure in `runner_host_tests.log`; its `USER=hous` control in `runner_host_tests_user.log` then reached `ModuleNotFoundError: openai`. These parent-owned logs live beside this report.

## Hypothesis ledger

| Hypothesis | Prediction / focused experiment | Result | Verdict |
| --- | --- | --- | --- |
| vLLM is installed but inactive venv selected | Probe all three runnable discovered Python environments for top-level package specs | All return `None` for vLLM and plugin | Refuted for inspected environments |
| An existing checkout/wheel can be reused without installing | Search mounted runtime/source paths for package roots, platform.py, and wheels | No usable source/wheel discovered | Refuted within inspected paths |
| Docker wrapper can enter a prepared serving image | Inspect executable and socket targets | Client and socket absent | Refuted in this container |
| Runner source must be newly invented/fetched | Probe already-fetched skill and 35B reference commits | Both contain shared runner | Refuted: restore known source and test it |
| Missing runtime is an adapter bug | Run vLLM API-server help before adapter import | Fails finding vLLM itself | Refuted: prerequisite boundary precedes adapter |

## Resolution / next checks

Restore and host-test shared runner source within the authorized code scope. Obtain an existing compatible vLLM environment or explicit dependency provisioning authorization before installation. After provision, record exact vLLM/TT-plugin/Transformers pins, verify package imports and canonical TT pytest paths, and rerun `python -m vllm.entrypoints.openai.api_server --help` before any reduced-layer hardware serving test. The 35B reference records `tenstorrent/vllm@bf98d556bb46a5cda25fac540629251e7f474200`, installed with `VLLM_TARGET_DEVICE=empty`, plus plugin changes; that is historical evidence only, not a validated version for this 9B checkpoint.

AutoFix conclusion for this prerequisite: no local code patch can supply the absent runtime under the current no-install constraint. This is an unresolved external dependency until an environment or changed authorization is supplied. Adapter implementation and host checks remain useful independent work. No serving performance/quality results exist from this investigation, and no processes were left holding devices.
