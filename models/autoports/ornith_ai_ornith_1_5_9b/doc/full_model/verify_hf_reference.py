"""Verify the saved reference, local revision receipts, and HF control provenance."""

import hashlib
import json
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[2]
meta_path = root / "readiness_aime24_chat.meta.json"
meta = json.loads(meta_path.read_text())
checkpoint = Path(meta["snapshot_path"])
receipts = {}
for path in sorted((checkpoint / ".cache/huggingface/download").glob("*.metadata")):
    lines = path.read_text().splitlines()
    assert lines[0] == meta["revision"], (path, lines[0])
    receipts[path.name] = {"revision": lines[0], "etag": lines[1]}
meta["local_snapshot_revision_receipts"] = receipts
meta["snapshot_metadata_hashes"]["chat_template.jinja"] = hashlib.sha256(
    (checkpoint / "chat_template.jinja").read_bytes()
).hexdigest()
for name, value in list(meta["loading_diagnostics"].items()):
    if value == "set()":
        meta["loading_diagnostics"][name] = []
for name in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]:
    assert meta["loading_diagnostics"][name] == []
ref_path = root / "readiness_aime24_chat.refpt"
ref = torch.load(ref_path, weights_only=True, map_location="cpu")
assert hashlib.sha256(ref_path.read_bytes()).hexdigest() == meta["reference_sha256"]
assert ref["k"] == 100
entry = ref["entries"][0]
assert entry["generated_tokens"].shape == (1, 100)
assert entry["topk_tokens"].shape == (100, 100)
assert entry["prompt_tokens"][0].tolist() == meta["prompt_token_ids"]
meta["hf_cached_vs_full_prefill_top1_agreement"] = float(
    (entry["generated_tokens"][0] == entry["topk_tokens"][:, 0]).float().mean()
)
meta["qualitative_control_verdict"] = {
    "prompt_id": "aime24:0",
    "verdict": "coherent partial mathematical reasoning",
    "evidence": "Explains walking time as distance/speed and derives 9/s = 4 - t/60.",
    "repetition": "No mechanical token duplication or collapse in inspected 100-token control.",
    "language": "English, matching the user prompt.",
    "divergence": "No TT output exists in this HF-only control; early HF/TT divergence is not evaluated.",
    "limitation": "Ends mid-sentence at the requested 100-token limit; no final math answer or full solution accuracy claimed.",
    "hf_runtime_warning": "Optional flash-linear-attention/causal-conv1d fast path unavailable; installed HF torch CPU implementation used. This is the reference path, not a TT runtime fallback.",
    "shared_suite_status": "This file is only the main AIME control; the six-prompt HF/TT shared suite remains a separate full-model gate.",
}
meta["post_generation_verification_script"] = str(Path(__file__).resolve())
meta["post_generation_verification_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
meta_path.write_text(json.dumps(meta, indent=2) + "\n")
print("Verified", len(receipts), "snapshot revision receipts")
print(
    "Reference shapes: prompt",
    tuple(entry["prompt_tokens"].shape),
    "generated",
    tuple(entry["generated_tokens"].shape),
    "top-k",
    tuple(entry["topk_tokens"].shape),
)
print("HF cached/full-prefill top-1 agreement", meta["hf_cached_vs_full_prefill_top1_agreement"])
print("Qualitative control:", json.dumps(meta["qualitative_control_verdict"]))
