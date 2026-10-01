import json
from pathlib import Path

import pytest
import torch

from models.autoports.jaredpalmer_kev_9b.tt.head import PointerHead, choice, noul, score

ADAPTER = (
    "/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"
)
REF = Path("/home/hous/dev/kev/reports/reference")


@pytest.mark.eager_host_side
def test_head_reproduces_reference_probs():
    if not (REF / "hidden_fp32.pt").exists():
        pytest.skip("reference hidden states not generated yet")
    head = PointerHead(ADAPTER)
    assert head.temperature == pytest.approx(2.193649959389252) and head.head_dim == 256
    hidden = torch.load(REF / "hidden_fp32.pt")
    ref = json.loads((REF / "probs_fp32.json").read_text())
    worst = 0.0
    for key, h in hidden.items():
        rec, qid = key.split(":", 1)
        q = ref["records"][rec]["questions"][qid]
        z = head.logits(h[-1], h[:-1])
        p = torch.softmax(z, -1)
        worst = max(worst, (p - torch.tensor(list(q["probabilities"].values()))).abs().max().item())
        assert torch.allclose(z, torch.tensor(list(q["logits"].values())), atol=1e-4), key
        answer = {
            "choice": lambda: choice(p, q["keys"]),
            "noul": lambda: noul(p),
            "score": lambda: score(p, ref["records"][rec]["questions"][qid]["answer"].get("legend")),
        }[q["type"]]()
        assert answer == q["answer"], (key, answer, q["answer"])
    print(f"{len(hidden)} rows; max |dp| head vs reference {worst:.3e}")
    assert worst < 1e-5
