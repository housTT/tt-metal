import os

STAGE1_PROFILE = {
    "QWEN36_MLP_GATE_UP_DTYPE": "bfp8",
    "QWEN36_MLP_DOWN_DTYPE": "bf16",
    "QWEN36_PROJ_DTYPE": "bfp8",
    "QWEN36_MATMUL_FIDELITY": "HiFi2",
    "QWEN36_GDN_GATE_FP32": "1",
}
KNOBS = tuple(STAGE1_PROFILE)


def apply(profile=None):
    for key, value in (profile or STAGE1_PROFILE).items():
        os.environ.setdefault(key, value)
    return active()


def active():
    return {key: os.environ.get(key) for key in KNOBS}
