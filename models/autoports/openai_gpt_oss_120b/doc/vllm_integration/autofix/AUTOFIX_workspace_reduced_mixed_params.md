# AutoFix Report: workspace reduced mixed-params sampling failure

## Hypothesis Experiment: vLLM duplicate-seed salting policy

- Hypothesis: GPT-OSS 120B vLLM inherits `SeedManager(salt_duplicate_seeds=True)`, so independent same-seed requests receive order-dependent salts and violate cross-batch-order reproducibility.
- Experiment: Ran the exact common `SeedManager` host logic with a fake `TTSampling` object and a recorder in place of `write_device_seed_values`, using logical requests from the failed mixed-params batch in original and reordered admission order.

```bash
python_env/bin/python - <<'PY'
from models.common.sampling.generator import SeedManager

class FakeSampling:
    _sampling_dp = 1
    _param_dims = None
    cluster_shape = None
    mesh_device = None
    seeds_tt_tensor = None

requests = [
    ("Count", None),
    ("Random", 42),
    ("Repetition", 42),
    ("List", 42),
    ("Word", 99),
    ("Letter", 42),
    ("Number", 42),
    ("Frequency", 42),
    ("All", 6),
    ("TopP", 7),
]
shuffled_indices = [3, 5, 1, 2, 7, 6, 4, 8, 9, 0]
orders = {
    "original": requests,
    "shuffled": [requests[i] for i in shuffled_indices],
}

def run_order(salt_duplicate_seeds, ordered):
    mgr = SeedManager(FakeSampling(), max_batch_size=len(ordered), salt_duplicate_seeds=salt_duplicate_seeds)
    writes = []
    mgr.write_device_seed_values = lambda seed_values: writes.append(list(seed_values))
    mgr.reset_seed([seed for _label, seed in ordered], list(range(len(ordered))))
    mgr.get_new_values(list(range(len(ordered))))
    return {
        label: {
            "slot": slot,
            "seed": seed,
            "salt": mgr.seed_salts[slot],
            "counter_after_first_seed": mgr.seed_counters[slot],
            "first_device_seed": writes[-1][slot],
        }
        for slot, (label, seed) in enumerate(ordered)
        if seed is not None
    }

for salt_policy in (True, False):
    print(f"salt_duplicate_seeds={salt_policy}")
    by_order = {name: run_order(salt_policy, ordered) for name, ordered in orders.items()}
    for label in ("Random", "Repetition", "List", "Letter", "Number", "Frequency"):
        original = by_order["original"][label]
        shuffled = by_order["shuffled"][label]
        print(
            f"  {label:10s} original(slot={original['slot']}, salt={original['salt']}, seed={original['first_device_seed']}) "
            f"shuffled(slot={shuffled['slot']}, salt={shuffled['salt']}, seed={shuffled['first_device_seed']})"
        )
    print()
PY
```

- Result:

```text
salt_duplicate_seeds=True
  Random     original(slot=1, salt=0, seed=275414) shuffled(slot=2, salt=2, seed=35571)
  Repetition original(slot=2, salt=1, seed=62798) shuffled(slot=3, salt=3, seed=292530)
  List       original(slot=3, salt=2, seed=35571) shuffled(slot=0, salt=0, seed=275414)
  Letter     original(slot=5, salt=3, seed=292530) shuffled(slot=1, salt=1, seed=62798)
  Number     original(slot=6, salt=4, seed=863760) shuffled(slot=5, salt=5, seed=76793)
  Frequency  original(slot=7, salt=5, seed=76793) shuffled(slot=4, salt=4, seed=863760)

salt_duplicate_seeds=False
  Random     original(slot=1, salt=0, seed=275414) shuffled(slot=2, salt=0, seed=275414)
  Repetition original(slot=2, salt=0, seed=275414) shuffled(slot=3, salt=0, seed=275414)
  List       original(slot=3, salt=0, seed=275414) shuffled(slot=0, salt=0, seed=275414)
  Letter     original(slot=5, salt=0, seed=275414) shuffled(slot=1, salt=0, seed=275414)
  Number     original(slot=6, salt=0, seed=275414) shuffled(slot=5, salt=0, seed=275414)
  Frequency  original(slot=7, salt=0, seed=275414) shuffled(slot=4, salt=0, seed=275414)
```

- Verdict: verified. With duplicate-seed salting enabled, reordered independent same-seed requests get different salts and first device seeds. With salting disabled, every independent duplicate uses salt `0` and the stream is stable across order.
- Fix: Added an explicit `salt_duplicate_seeds` policy on `FullModelArgs` and `Model.from_checkpoint`, defaulting to `True` to preserve standalone/demo behavior. `TTGptOssForCausalLM.initialize_vllm_model()` now passes `salt_duplicate_seeds=False` before model construction, so the common `SamplingGenerator` constructs its `SeedManager` with the vLLM-serving policy.
- Verification:
  - `python_env/bin/python -m pytest models/autoports/openai_gpt_oss_120b/tests/test_generator_vllm.py::test_vllm_initialization_disables_duplicate_seed_salting_before_sampling_construction -q`: passed, 1 test.
  - `python_env/bin/python -m pytest models/autoports/openai_gpt_oss_120b/tests/test_generator_vllm.py -q`: passed, 26 tests.
  - `python_env/bin/python -m black --check models/autoports/openai_gpt_oss_120b/tt/model.py models/autoports/openai_gpt_oss_120b/tt/generator_vllm.py models/autoports/openai_gpt_oss_120b/tests/test_generator_vllm.py`: passed, 3 files unchanged.
  - Exact original reduced two-layer P150x4 vLLM smoke rerun: passed all four smoke targets, including `TestBatchIsolation::test_mixed_params_batch`; the server became ready in about 60 seconds and shut down cleanly. Evidence: `readiness_vllm/reduced_workspace_seed_autofix/server.log` and `readiness_vllm/reduced_workspace_seed_autofix/sampling_tests.log`.
- Uncertainty: the focused reduced target proves the serving seed-policy fix. The complete 36-layer full sampling profile remains the final-model regression gate.
