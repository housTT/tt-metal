# Serving environment provisioning

Date: 2026-09-08 UTC. This continues the missing-runtime finding in
`AUTOFIX_environment.md` after the updated root/model AGENTS instructions and
`SUPERVISOR_RECOVERY.md` resolved the old permission interpretation. This pass
installs the runtime and runs host checks. It does not open devices or start a
server, and it does not establish a serving-stage pass.

## Hypothesis and intervention

The existing interpreter lacks `vllm`, `vllm_tt_plugin`, and `openai`. Providing
the pinned TT fork and its plugin should allow the original import/help probes
and restored runner host tests to advance. Source inspection confirms the empty
vLLM target reads `requirements/common.txt`, which requires Transformers >=5.5.3
and leaves Torch unpinned. Its isolated build requirements otherwise pin Torch
2.10.0; `--no-build-isolation` uses the existing 2.11.0+cpu build instead.

Initially created `/home/hous/dev/ornith-1.5-9b/state/serving-env`, with a `.pth` calling
`site.addsitedir` for the existing `tt-metal/python_env` site-packages. New
packages install only in serving-env, which takes precedence; TTNN and
Torchvision remain available from the existing environment. The resolver also
installed a separate copy of the same Torch 2.11.0+cpu into serving-env.
`provision_base_packages_before.json` records the original environment. This
initial broad overlay was replaced by the clean runtime described below.

Constrained NumPy to 1.26.4, Transformers to 5.12.1, Torch to 2.11.0+cpu,
Torchvision to 0.26.0+cpu, and OpenCV headless to 4.11.0.86. This OpenCV version
meets vLLM's >=4.11 requirement without requiring NumPy 2. Setuptools is pinned
to 80.10.2, within the source build requirement >=77,<81. Constraints are in
`provision_constraints.txt`.

## Commands

Run from `/home/hous/dev/ornith-1.5-9b/tt-metal`:

```bash
python_env/bin/uv venv --python python_env/bin/python ../state/serving-env
python_env/bin/python - <<'PY'
from pathlib import Path
root = Path.cwd()
pth = root.parent / 'state/serving-env/lib/python3.10/site-packages/zz_ornith_base.pth'
pth.write_text('import site; site.addsitedir(' + repr(str(root / 'python_env/lib/python3.10/site-packages')) + ')\n')
PY
USER=hous python_env/bin/uv pip install --python ../state/serving-env/bin/python \
    'setuptools-scm>=8' 'grpcio-tools==1.78.0'
USER=hous VLLM_TARGET_DEVICE=empty python_env/bin/uv pip install \
    --python ../state/serving-env/bin/python --no-build-isolation \
    --constraint models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/provision_constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match -e ../vllm
USER=hous python_env/bin/uv pip install --python ../state/serving-env/bin/python \
    --no-build-isolation -e ../vllm/plugins/vllm-tt-plugin
USER=hous python_env/bin/uv pip install --python ../state/serving-env/bin/python \
    'setuptools==80.10.2'
```

The parent agent cloned the sibling vLLM checkout and checked out
`bf98d556bb46a5cda25fac540629251e7f474200` on branch
`hous/ornith-1.5-9b-vllm`. This is the predecessor's source baseline, not borrowed
validation evidence. Editable vLLM version is
`0.1.dev14188+gbf98d556b.empty`; plugin distribution version is `0.0.0`.
The parent owns any plugin source changes after this baseline.

Installation logs: `provision_build_helpers.log`, `provision_vllm_install.log`,
`provision_plugin_install.log`, and `provision_setuptools.log`.

## Activation

```bash
cd /home/hous/dev/ornith-1.5-9b/tt-metal
source ../state/serving-env/bin/activate
export USER=hous
export TT_METAL_HOME=/home/hous/dev/ornith-1.5-9b/tt-metal
export PYTHONPATH=/home/hous/dev/ornith-1.5-9b/tt-metal
```

The stable alias `serving-env` now points to `serving-clean-candidate`. Its
activation and console entrypoints use that canonical directory. The original
overlay is retained as `serving-overlay-backup`. The existing runner additionally
supplies its established library/cache/device configuration. Do not replace the
original `python_env` activation or locks.

## Isolating the runtime dependency closure

The initial overlay passed imports/help and four runner tests, and `uv pip check`
passed for its 143 locally installed packages. A second check, `python -m pip
check`, also inspected inherited development packages and exposed four conflicts:
datasets 2.21.0 with newer dill/fsspec, fiftyone 0.25.2 with newer sse-starlette,
and myst-parser 3.0.0 with newer markdown-it-py. The original environment itself
passed pip check. Logs: `provision_effective_pip_check.log` and
`provision_base_pip_check.log`.

The focused repair was to stop inheriting the development environment as a
whole. Created `../state/serving-clean-candidate` with the same interpreter,
hard-linked the already downloaded serving packages, and removed
`zz_ornith_base.pth`. The replacement `zz_ornith_ttnn.pth` lists only the checkout
root, `ttnn`, and `tools`. Symlinks expose the existing `torchvision`,
`torchvision.libs`, Torchvision metadata, and TTNN metadata; the built TTNN
extension stays in the checkout. Installed TTNN's declared runtime dependencies
and host test requirements locally, pinned to the working base versions:

```bash
python_env/bin/uv venv --python python_env/bin/python ../state/serving-clean-candidate
cp -aln ../state/serving-env/lib/python3.10/site-packages/. \
    ../state/serving-clean-candidate/lib/python3.10/site-packages/
python_env/bin/python - <<'PY'
from pathlib import Path
root = Path.cwd()
site = root.parent / 'state/serving-clean-candidate/lib/python3.10/site-packages'
(site / 'zz_ornith_base.pth').unlink()
(site / 'zz_ornith_ttnn.pth').write_text(
    '\n'.join(str(root / s) for s in ['', 'ttnn', 'tools']) + '\n'
)
base = root / 'python_env/lib/python3.10/site-packages'
for name in ['torchvision', 'torchvision.libs', 'torchvision-0.26.0+cpu.dist-info',
             'ttnn-0.75.0rc10.dev1256.dist-info']:
    (site / name).symlink_to(base / name, target_is_directory=True)
PY
USER=hous python_env/bin/uv pip install \
    --python ../state/serving-clean-candidate/bin/python \
    --constraint models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/provision_constraints.txt \
    -r models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/provision_clean_requirements.txt
```

The first `cp -al` attempt reported that the venv's two `_virtualenv` files
already existed. Rerunning with `-n` completed the copy while preserving those
files. No package install or base-environment write depended on the failed copy.
The final console scripts were copied from the initial overlay, replacing their
interpreter prefix with the candidate path. Existing candidate scripts were
preserved. After the candidate passed every original probe plus full pip check,
the parent confirmed that the environment was idle. The initial overlay was
renamed to `serving-overlay-backup` and `serving-env` became a symlink to the
candidate. `vllm --help` through that stable alias also passed.

The exact console-copy and alias-switch operation was:

```python
from pathlib import Path
import shutil
root = Path.cwd().parent
old = root / 'state/serving-env'
new = root / 'state/serving-clean-candidate'
for entry in (old / 'bin').iterdir():
    target = new / 'bin' / entry.name
    if target.exists() or entry.is_symlink() or not entry.is_file():
        continue
    shutil.copy2(entry, target)
    data = target.read_bytes()
    if data.startswith(b'#!'):
        target.write_bytes(data.replace(str(old).encode(), str(new).encode()))
backup = root / 'state/serving-overlay-backup'
assert not backup.exists()
old.rename(backup)
old.symlink_to(new.name, target_is_directory=True)
```

Evidence: `provision_clean_install.log`, `provision_clean_host_probes.json`,
`provision_clean_*.log`, and `provision_cli_help.log`.

## Verification

| Check | Result |
| --- | --- |
| Import Torch, Torchvision, NumPy, Transformers, OpenAI, TTNN, vLLM, TT plugin | Pass; TTPlatform activated |
| `python -m models.common.readiness_check.run_vllm_server --help` | Exit 0 |
| `python -m vllm.entrypoints.openai.api_server --help` | Exit 0 |
| `python -m pytest -q models/common/readiness_check/test_run_vllm_server.py` | 4 passed |
| `python -m pip check` in the final clean environment | Exit 0, no broken requirements |
| `../state/serving-env/bin/vllm --help` after switching the alias | Exit 0 |
| Base package inventory before/after | All 373 name/version/metadata-path records unchanged |

Initial commands, statuses and logs: `provision_host_probes.json`. Final clean
runtime commands, statuses and logs: `provision_clean_host_probes.json`.
Base preservation evidence: `provision_base_preservation.json` and both package
inventory JSON files. No C++ or CMake source changed, so no TT build was needed.

Final pins: Python 3.10.19, Torch 2.11.0+cpu, Torchvision 0.26.0+cpu, Transformers
5.12.1, NumPy 1.26.4, OpenCV headless 4.11.0.86, OpenAI 3.8.0, TTNN
0.75.0rc10.dev1256, vLLM distribution 0.1.dev14188+gbf98d556b.empty, and plugin
0.0.0. The editable vLLM module reports 0.1.dev14188+gbf98d556b. Full final
distributions (`python -m pip freeze`) are in `serving_environment_freeze.txt` and
`serving_effective_packages.json`. `serving_runtime_requirements.lock` contains
the exact pip-installable runtime/test package versions, excluding vLLM/plugin
source and the shared TTNN/Torchvision artifacts.
There are 165 distinct effective distributions; TTNN has same-version editable
source metadata in addition to its preserved dist-info, recorded separately in
`serving_duplicate_metadata.json`.

`serving_source_pins.json` records source revisions, package versions, shared
extension SHA256 hashes, environment paths, and plugin source modifications
owned by the parent. Subsequent plugin edits are independent source work;
record their final commits/patches alongside serving validation. The environment
requires the preserved checkout and Torchvision artifacts at their recorded
paths and is not yet a standalone release container.

**Environment prerequisite fixed.** This verifies host/runtime readiness only.
No TT device was opened, no server was launched, and no accuracy, latency,
memory, parser API, or serving-stage gate was established by this pass. Those
remain the parent's serialized integration and release work.

## Bounded qualitative-control follow-through, 2026-09-08

Applied the qualitative-check skill to the twelve saved shared serving256
texts, exact pinned HF/selected-TT128 controls, and the longer prompt 0 output.
All six greedy serving prefixes match the previous selected TT128 exactly,
including tokenizer-reconstructed IDs. The original-template haiku error lies
beyond that previous coverage; the completed serving answer has 6/7/5
syllables. The prior HF256 control uses a correct different 5/7/5 draft.

Authored `hf_haiku_control.py` and ran it offline on CPU with the unchanged
base environment, explicit original BF16, Qwen3_5ForCausalLM, all 8,953,803,264
parameters, eight threads, and the common qualitative helper's original
generation kwargs. It inherited cache=False and EOS 248046; no fast-path,
dtype, or cache substitution was introduced. Local download metadata pins
all relevant snapshot files to revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53.
The bounded 512-token request exited 0 after 376 tokens/EOS, 546.115s generation,
and 0.325s in the loader call. The first 256 IDs exactly reproduce the older
HF256 control, and the first 128 text matches the saved HF128 control. Its
completed final haiku is correctly 5/7/5. No TTNN module was imported.

Authored `standalone_haiku_control.py` without importing or executing TTNN.
The supervising lane ran that source after its live server closed. Actual
32-layer, batch-one selected precision with cache 2048 returned 390 tokens and
closed cleanly. Independent saved-artifact comparison proves all 390 IDs, including
EOS, exactly equal the serving output. The current scripts' SHA256 hashes
match their execution metadata. AST and all applicable pre-commit checks
passed; no C++ or CMake changed and no build was needed.

The limited verdict is explicit: no vLLM-specific divergence was found on
this extended prompt, but the selected full-model output has a newly exposed
completed-task quality gap against matching HF. No checkpoint explanation,
specific precision cause, or overall quality pass is claimed. Selected
precision, production source, and both runtimes remain unchanged by this
control work. Further source changes or inference runs were not performed
after the supervising agent requested finalization.

Evidence: `qualitative_extended_control_report.md`,
`qualitative_extended_control_summary.json`,
`qualitative_token_prefix_comparison.json`,
`haiku_standalone_serving_exact_comparison.json`, `hf_haiku_512_v1/` plus its
log, and the supervising `standalone_haiku_512_v1/` plus its log. The report
records exact commands, prompt/template IDs, observed text, and limitations.
The preceding optional native allocation-tracker pass is also now recorded
in `adapter_probe_authoring_report.md` and `trace_allocation_audit.md`: all
four reduced adapter cases passed with program-cache tracking included and
clean shutdown. That guarded lifetime result does not claim physical-address
overlap measurement or full-server coverage.
