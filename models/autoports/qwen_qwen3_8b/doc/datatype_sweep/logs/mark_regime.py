"""Mark the chip-2 sweep sidecar runs that overlapped with the chip-0/1 jobs (started at or after 14:54:33 UTC) as concurrent."""
import json
from pathlib import Path

D = Path("models/autoports/qwen_qwen3_8b/doc/datatype_sweep")
CUTOFF = "14:54:33"
for sidecar in sorted(D.glob("readiness_c*_chip2.json")):
    data = json.loads(sidecar.read_text())
    changed = False
    for name, run in data["runs"].items():
        if (
            run.get("ended_utc", "") > CUTOFF
            and run.get("chip", 2) == 2
            and "aime_p1" not in sidecar.name
            and "book32" not in sidecar.name
        ):
            run[
                "timing_regime"
            ] = "concurrent: chips 0 and 1 ran accuracy jobs on the same host during this run; host-wall TTFT and decode t/s/u are not same-regime with the serialized rows"
            changed = True
        elif "timing_regime" not in run:
            run["timing_regime"] = "serialized: the only device job on the host during this run"
            changed = True
    if changed:
        sidecar.write_text(json.dumps(data, indent=2) + "\n")
        print(sidecar.name, {k: v["timing_regime"][:12] for k, v in data["runs"].items()})
