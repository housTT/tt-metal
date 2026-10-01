import os

SELECTED_CONFIG_ID = "mlp_bfp8"
PROFILES = {
    "selected": {
        "QWEN36_MLP_GATE_UP_DTYPE": "bfp8",
        "QWEN36_MLP_DOWN_DTYPE": "bfp8",
        "QWEN36_PROJ_DTYPE": "bfp8",
        "QWEN36_MATMUL_FIDELITY": "LoFi",
        "QWEN_GDN_FP32_STATE": "0",
        "QWEN_SDPA_BF8": "0",
        "KEV_MATMUL_POLICY": "1",
    },
    "baseline": {
        "QWEN36_MLP_GATE_UP_DTYPE": "bfp4",
        "QWEN36_MLP_DOWN_DTYPE": "bfp8",
        "QWEN36_PROJ_DTYPE": "bfp8",
        "QWEN36_MATMUL_FIDELITY": "LoFi",
        "QWEN_GDN_FP32_STATE": "0",
        "QWEN_SDPA_BF8": "0",
        "KEV_MATMUL_POLICY": "1",
    },
}
CACHE_TAG_KEYS = (("gu", "QWEN36_MLP_GATE_UP_DTYPE"), ("dn", "QWEN36_MLP_DOWN_DTYPE"), ("pj", "QWEN36_PROJ_DTYPE"))


def profile_name():
    name = os.environ.get("KEV_PRECISION", "selected")
    if name not in PROFILES:
        raise ValueError(f"KEV_PRECISION={name!r}; expected one of {sorted(PROFILES)}")
    return name


def apply():
    profile = PROFILES[profile_name()]
    for key, value in profile.items():
        os.environ.setdefault(key, value)
    return active()


def active():
    profile = PROFILES[profile_name()]
    return {key: os.environ.get(key, value) for key, value in profile.items()}


def cache_tag():
    env = active()
    return "_".join(f"{short}-{env[key]}" for short, key in CACHE_TAG_KEYS)


apply()
