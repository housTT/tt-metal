"""Count fatal patterns in the watcher log of the reduced-model run and write watcher/summary.json."""
import json
import re
import sys
from pathlib import Path

root = Path("models/autoports/qwen_qwen3_8b/doc/optimized_full_model")
log = root / "watcher" / "generated" / "watcher" / "watcher.log"
text = log.read_text(errors="replace") if log.exists() else ""
patterns = [
    "fatal",
    "exception",
    "assert",
    "overflow",
    "bad noc",
    "hung",
    "error",
    "stack overflow",
    "invalid",
    "corrupt",
]
counts = {p: len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns}
dumps = len(re.findall(r"Dump #\d+", text))
pytest_log = (root / "logs" / "pytest_watcher_reduced.log").read_text(errors="replace")
result = re.findall(r"(\d+ passed.*?)\n", pytest_log)
summary = {
    "command": sys.argv[1] if len(sys.argv) > 1 else "see logs/gate_chain_chip2.sh",
    "model": "reduced full model: two real decoder layers, real embedding, final norm, interleaved prefill norms, masked LM head, split sampling, traces, chunk-piece warmup",
    "result": (result[-1] if result else "see logs/pytest_watcher_reduced.log"),
    "watcher_log": str(log.relative_to(root)) if log.exists() else None,
    "dumps_completed": dumps,
    "pattern_counts": counts,
    "fatal_pattern_matches": sum(
        counts[p] for p in ("fatal", "assert", "bad noc", "hung", "stack overflow", "corrupt")
    ),
    "evidence_json": sorted(p.name for p in root.glob("*.watcher.json")),
}
(root / "watcher" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
