import json
import os
import time
from pathlib import Path

import pytest
import torch

from models.autoports.cloudflare_clef.tt import encode as E
from models.autoports.cloudflare_clef.tt.head import load_head, probs_for_record
from models.autoports.cloudflare_clef.tt.loader import LmHeadRows

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
KEV_DEV = Path("/home/hous/dev/kev/kev/evals/documents-v1/development.jsonl")
REPORT = Path("/home/hous/dev/clef/reports/stage1_head_cpu.json")
TOLERANCE = 1e-4


def records():
    out = []
    with KEV_DEV.open() as f:
        for line in f:
            r = json.loads(line)
            out.append(
                {
                    "model": "clef",
                    "state": r["state"],
                    "questions": {
                        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
                        for qid, q in r["questions"].items()
                    },
                }
            )
            if len(out) == 2:
                break
    out.append(
        {
            "model": "clef",
            "state": {"ticket": {"title": "Checkout returns 500", "customers_affected": 1200, "region": "eu-west"}},
            "questions": {
                "outage": {"type": "noul", "instructions": "Is this an outage?"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team owns it?",
                    "criteria": {"payments": "Payments", "web": "Web frontend", "infra": "Infrastructure"},
                },
                "severity": {
                    "type": "score",
                    "instructions": "Severity",
                    "criteria": ["low", "medium", "high", "critical"],
                },
            },
        }
    )
    return out


@pytest.mark.slow
@pytest.mark.eager_host_side
@pytest.mark.timeout(5400)
def test_head_reproduces_release_probs():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    release = E.release_module()
    t0 = time.time()
    model, processor = release.load_release_model(SNAPSHOT, device="cpu", dtype=torch.bfloat16)
    load_s = time.time() - t0
    text_model = model.language_model.model.language_model
    captured = {}

    def hook(module, inputs, output):
        captured["hidden"] = output.last_hidden_state.detach()

    handle = text_model.register_forward_hook(hook)
    head32 = load_head(SNAPSHOT)
    release_head32 = release.JointSchemaHead(**json.loads((Path(SNAPSHOT) / "joint_head_config.json").read_text()))
    release_head32.load_state_dict(model.head.state_dict(), strict=True)
    release_head32 = release_head32.float().eval()
    assert all(p.dtype == torch.bfloat16 for p in model.head.parameters())
    lm_head_weight = model.language_model.get_output_embeddings().weight
    rows_fn = LmHeadRows(SNAPSHOT)
    assert torch.equal(rows_fn([0, 248044]), lm_head_weight[[0, 248044]])
    rows = []
    worst_vs_fp32, worst_vs_bf16 = 0.0, 0.0
    for request in records():
        encoded = E.encode(processor.tokenizer, request, processor=processor)
        batch = release.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cpu"))
        t1 = time.time()
        with torch.inference_mode():
            release_logits = model(batch)[0]
        forward_s = time.time() - t1
        hidden = captured.pop("hidden")
        assert hidden.shape == (1, len(encoded.input_ids), 5120) and hidden.dtype == torch.bfloat16
        with torch.inference_mode():
            fp32_logits = release_head32(
                hidden.float(), batch["input_ids"], batch["attention_mask"], [encoded], lm_head_weight.float()
            )[0]
        ours = probs_for_record(head32, hidden[0].float(), batch["input_ids"][0], encoded, rows_fn)
        answers = release.systemone(model, processor, request)["answers"]
        for question, rl, fl in zip(encoded.questions, release_logits, fp32_logits):
            p_bf16 = rl.float().softmax(-1)
            p_fp32 = fl.float().softmax(-1)
            p_ours = torch.tensor([ours[question.question_id][o] for o in question.option_ids])
            d32 = (p_ours - p_fp32).abs().max().item()
            d16 = (p_ours - p_bf16).abs().max().item()
            worst_vs_fp32, worst_vs_bf16 = max(worst_vs_fp32, d32), max(worst_vs_bf16, d16)
            rows.append(
                {
                    "question": question.question_id,
                    "T": len(encoded.input_ids),
                    "options": list(question.option_ids),
                    "ours": p_ours.tolist(),
                    "release_fp32_head": p_fp32.tolist(),
                    "release_bf16_head": p_bf16.tolist(),
                    "max_dp_vs_fp32": d32,
                    "max_dp_vs_bf16": d16,
                    "argmax_equal_fp32": int(p_ours.argmax()) == int(p_fp32.argmax()),
                    "argmax_equal_bf16": int(p_ours.argmax()) == int(p_bf16.argmax()),
                    "forward_s": forward_s,
                }
            )
            assert d32 < TOLERANCE, (question.question_id, d32)
            assert answers[question.question_id]["type"] == request["questions"][question.question_id]["type"]
    handle.remove()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "snapshot": SNAPSHOT,
                "load_s": load_s,
                "tolerance": TOLERANCE,
                "max_dp_vs_release_fp32_head": worst_vs_fp32,
                "max_dp_vs_release_bf16_head": worst_vs_bf16,
                "rows": rows,
            },
            indent=1,
        )
    )
    print(
        f"{len(rows)} questions; load {load_s:.0f} s; max |dp| vs fp32 release head {worst_vs_fp32:.3e}; "
        f"vs bf16 release head {worst_vs_bf16:.3e}"
    )
