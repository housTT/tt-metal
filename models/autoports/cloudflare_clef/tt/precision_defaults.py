import json
import os
from pathlib import Path

SELECTED_PATH = Path(__file__).resolve().parents[1] / "doc" / "datatype_sweep" / "selected_precision_config.json"

STAGE1_PROFILE = {
    "QWEN36_MLP_GATE_UP_DTYPE": "bfp8",
    "QWEN36_MLP_DOWN_DTYPE": "bf16",
    "QWEN36_PROJ_DTYPE": "bfp8",
    "QWEN36_MATMUL_FIDELITY": "HiFi2",
    "QWEN36_GDN_GATE_FP32": "1",
    "CLEF_VISION_PRECISION": "accuracy",
    "CLEF_VISION_ACT_BF16": "1",
}
KNOBS = tuple(STAGE1_PROFILE)
EXTRA_PREFIXES = ("QWEN36_", "QWEN35_", "QWEN_GDN_", "CLEF_VISION_")
PROFILES = {"stage1": STAGE1_PROFILE}


def selected_profile(path=None):
    path = Path(path or SELECTED_PATH)
    if not path.exists():
        return None
    flags = json.loads(path.read_text()).get("runtime_flags") or {}
    values = {key: str(value) for key, value in flags.items() if key in KNOBS or key.startswith(EXTRA_PREFIXES)}
    return values or None


def profile_name():
    name = os.environ.get("CLEF_PRECISION")
    if name:
        return name
    return "selected" if selected_profile() else "stage1"


def profile(name=None):
    name = name or profile_name()
    if name == "selected":
        values = selected_profile()
        if values is None:
            raise FileNotFoundError(f"CLEF_PRECISION=selected but {SELECTED_PATH} has no runtime_flags")
        return dict(STAGE1_PROFILE, **values)
    if name in PROFILES:
        return dict(PROFILES[name])
    raise ValueError(f"CLEF_PRECISION={name!r}; expected 'selected' or one of {sorted(PROFILES)}")


def apply(values=None):
    for key, value in (values or profile()).items():
        os.environ.setdefault(key, value)
    return active()


def active():
    try:
        extra = sorted(key for key in profile() if key not in KNOBS)
    except (FileNotFoundError, ValueError):
        extra = []
    return {key: os.environ.get(key) for key in KNOBS + tuple(extra)}
