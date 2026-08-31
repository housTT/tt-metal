# AutoDebug Report: GPT-OSS 120B vLLM TT Device Inference Regression

Date: 2026-08-31

Scope: source/log/import diagnosis only. No implementation files were edited by this investigation. I did not start a vLLM server, access TT hardware, or reset devices.

## Headline Finding

The resumed startup failure was caused by an environment dependency disappearing, not by a vLLM or TT plugin source regression.

The exact failing command still set `LD_LIBRARY_PATH=/tmp/codex-gpt120b-libnsl`, but that `/tmp` support directory had vanished. As a result, `import ttnn` failed while the TT platform plugin was being probed:

```text
TT plugin platform is not available because: libnsl.so.2: cannot open shared object file: No such file or directory
```

After `libnsl.so.2` was re-extracted into `/tmp/codex-gpt120b-libnsl`, the same host-only platform probe reported:

```text
platform_plugin_result vllm_tt_plugin.platform.TTPlatform
current_platform TTPlatform tt
Platform plugin tt is activated
```

This exactly explains the observed shape: vLLM discovered the entry point `tt -> vllm_tt_plugin.entrypoints:platform_plugin`, but did not activate the TT platform, and then `DeviceConfig(device="auto")` failed with `Failed to infer device type` before any device access.

## Evidence

### Source Chain

1. `/home/ttuser/dev/ornith/vllm/plugins/vllm-tt-plugin/pyproject.toml:31-32` registers the TT platform plugin:

```toml
[project.entry-points."vllm.platform_plugins"]
tt = "vllm_tt_plugin.entrypoints:platform_plugin"
```

2. `/home/ttuser/dev/ornith/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/entrypoints.py:50-59` activates TT only if `import ttnn` succeeds. Any exception is swallowed as a debug-only "not available" result:

```python
def platform_plugin() -> str | None:
    try:
        import ttnn
    except Exception as exc:
        logger.debug("TT plugin platform is not available because: %s", exc)
        return None
    return "vllm_tt_plugin.platform.TTPlatform"
```

3. `/home/ttuser/dev/ornith/vllm/vllm/platforms/__init__.py:187-225` first logs discovered platform plugins, then calls each plugin function. It only logs `Platform plugin tt is activated` when a plugin returns a non-`None` platform class.

4. `/home/ttuser/dev/ornith/vllm/vllm/config/device.py:52-59` raises the observed `RuntimeError` when `current_platform.device_type` is empty.

### Failing Logs

The 2026-08-31 failed server attempts show discovery but no activation:

```text
Available plugins for group vllm.platform_plugins:
- tt -> vllm_tt_plugin.entrypoints:platform_plugin
RuntimeError: Failed to infer device type
```

Files:

- `/home/ttuser/dev/gpt-oss-20b/tt-metal/models/autoports/openai_gpt_oss_120b/readiness_vllm/failed_attempts/20260831T133108Z_bad_system_start_date.log`
- `/home/ttuser/dev/gpt-oss-20b/tt-metal/models/autoports/openai_gpt_oss_120b/readiness_vllm/failed_attempts/20260831T133143Z_device_inference_retry.log`

The debug host probe recorded in `/home/ttuser/dev/gpt-oss-20b/tt-metal/.agents/runs/gpt-oss-120b-p150-family-20260827T214421Z/09-09-vllm.resume-1.jsonl` shows the missing shared library directly:

```text
TT plugin platform is not available because: libnsl.so.2: cannot open shared object file: No such file or directory
```

### Binary Dependency Check

With the missing temp path:

```text
env LD_LIBRARY_PATH=/tmp/definitely-missing-gpt120b-libnsl ldd ttnn/ttnn/_ttnn.so
libnsl.so.2 => not found

env LD_LIBRARY_PATH=/tmp/definitely-missing-gpt120b-libnsl ldd ttnn/ttnn/_ttnncpp.so
libnsl.so.2 => not found
```

With the restored path:

```text
env LD_LIBRARY_PATH=/tmp/codex-gpt120b-libnsl ldd ttnn/ttnn/_ttnn.so
libnsl.so.2 => /tmp/codex-gpt120b-libnsl/libnsl.so.2

env LD_LIBRARY_PATH=/tmp/codex-gpt120b-libnsl ldd ttnn/ttnn/_ttnncpp.so
libnsl.so.2 => /tmp/codex-gpt120b-libnsl/libnsl.so.2
```

Current restored directory contents:

```text
/tmp/codex-gpt120b-libnsl/libnsl.so.2
/tmp/codex-gpt120b-libnsl/libnsl2_1.3.0-3build3_amd64.deb
```

### Host-Only A/B Probe

Failing control:

```text
LD_LIBRARY_PATH=/tmp/definitely-missing-gpt120b-libnsl
platform_plugin_result None
TT plugin platform is not available because: libnsl.so.2: cannot open shared object file
device_config_error RuntimeError Failed to infer device type
```

Passing control:

```text
LD_LIBRARY_PATH=/tmp/codex-gpt120b-libnsl
platform_eps [('tt', 'vllm_tt_plugin.entrypoints:platform_plugin')]
platform_plugin_result vllm_tt_plugin.platform.TTPlatform
current_platform TTPlatform tt
```

The post-repair server log also crosses the previous failure boundary:

```text
Platform plugin tt is activated
```

at `/home/ttuser/dev/gpt-oss-20b/tt-metal/models/autoports/openai_gpt_oss_120b/readiness_vllm/final_server.log:6` and again at line 36 in the engine process.

## Causal Chain

1. vLLM metadata discovery finds the TT entry point. This does not require `ttnn` to import.
2. Platform resolution calls `vllm_tt_plugin.entrypoints.platform_plugin()`.
3. That function imports `ttnn`.
4. `ttnn/ttnn/_ttnn.so` and `_ttnncpp.so` need `libnsl.so.2`.
5. The command points the loader at `/tmp/codex-gpt120b-libnsl`, but that temporary directory was missing on 2026-08-31.
6. The loader raises `libnsl.so.2: cannot open shared object file`.
7. The TT plugin catches the exception and returns `None`.
8. No built-in platform is available on this host, so vLLM falls back to `UnspecifiedPlatform`.
9. `DeviceConfig(device="auto")` sees an empty `device_type` and raises `Failed to infer device type`.

## Why It Worked On 2026-08-30

The 2026-08-30 command used the same `/tmp/codex-gpt120b-libnsl` loader dependency path and passed platform activation. A `/tmp` path is ephemeral across resumes/reboots/cleanup. The source trees can remain unchanged while the dynamic loader input disappears, which matches the reported "same source/command worked yesterday, failed today" behavior.

## Smallest Next Experiments

1. Keep this as a pre-server check before any hardware run:

```bash
env VLLM_LOGGING_LEVEL=DEBUG \
  LD_LIBRARY_PATH=/tmp/codex-gpt120b-libnsl \
  PYTHONPATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/ttnn:/home/ttuser/dev/gpt-oss-20b/tt-metal:/home/ttuser/dev/ornith/tt-metal:/home/ttuser/dev/ornith/vllm:/home/ttuser/dev/ornith/vllm/plugins/vllm-tt-plugin/src \
  VLLM_SYSTEM_START_DATE=2026-08-30 \
  MESH_DEVICE=P150x4 \
  /home/ttuser/dev/ornith/ornith-pyenv/bin/python - <<'PY'
from importlib.metadata import entry_points
print([(ep.name, ep.value) for ep in entry_points(group="vllm.platform_plugins")])
from vllm_tt_plugin.entrypoints import platform_plugin
print(platform_plugin())
from vllm.platforms import current_platform
print(type(current_platform).__name__, current_platform.device_type)
PY
```

Expected: `vllm_tt_plugin.platform.TTPlatform` and `TTPlatform tt`.

2. Add a cheap loader preflight to the run script or readiness workflow:

```bash
test -r /tmp/codex-gpt120b-libnsl/libnsl.so.2
env LD_LIBRARY_PATH=/tmp/codex-gpt120b-libnsl ldd /home/ttuser/dev/gpt-oss-20b/tt-metal/ttnn/ttnn/_ttnn.so | rg 'libnsl.so.2 => .*not found' && exit 1
```

3. Avoid recurrence by moving this dependency out of `/tmp` or rehydrating it every run before platform detection. A stable run-artifact path or pyenv-adjacent dependency path is safer than an overnight `/tmp` dependency.

4. Optional source hardening, not required for this fix: make TT platform plugin import failures easier to see at normal log level, or have `DeviceConfig` surface the last platform-plugin exception when all platforms resolve to unspecified.

## Not Headline Findings

- `VLLM_SYSTEM_START_DATE=2026-08-31` was present in one failed attempt, but the second failed attempt restored `VLLM_SYSTEM_START_DATE=2026-08-30` and still failed at the same device-inference boundary. That date mismatch was not the root cause.
- TT hardware health is not implicated by this failure. The error occurs while building CLI/config defaults, before device open.
- The vLLM TT plugin entry point exists and is discoverable. The failure is inside the entry point's `ttnn` import dependency, not entry-point metadata registration.
