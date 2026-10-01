import math
from pathlib import Path

import torch

from models.autoports.jaredpalmer_kev_9b.tt.api import choice_confidence, round_prob, score_confidence


class PointerHead:
    def __init__(self, adapter_dir):
        meta = torch.load(Path(adapter_dir) / "head.pt", map_location="cpu", weights_only=False)
        head = meta["head"]
        self.q_weight, self.q_bias = head["q.weight"].float(), head["q.bias"].float()
        self.k_weight, self.k_bias = head["k.weight"].float(), head["k.bias"].float()
        self.head_dim = meta.get("head_dim", self.q_weight.shape[0])
        self.scale = 1 / math.sqrt(self.head_dim)
        self.temperature = float(meta.get("temperature", 1.0))
        self.meta = meta

    def logits(self, h_decide, h_opts):
        q = h_decide.float() @ self.q_weight.T + self.q_bias
        k = h_opts.float() @ self.k_weight.T + self.k_bias
        return (k @ q) * self.scale / self.temperature

    def probs(self, h_decide, h_opts):
        return torch.softmax(self.logits(h_decide, h_opts), -1)


def choice(p, keys):
    p = [float(x) for x in p]
    return {
        "type": "choice",
        "choice": keys[max(range(len(p)), key=lambda i: p[i])],
        "confidence": round_prob(choice_confidence(p)),
        "probabilities": {k: round_prob(v) for k, v in zip(keys, p)},
    }


def noul(p):
    return {"type": "noul", "noul": round_prob(float(p[1]))}


def score(p, legend):
    p = [float(x) for x in p]
    return {
        "type": "score",
        "score": round_prob(sum(i * pi for i, pi in enumerate(p))),
        "legend": legend,
        "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)},
        "confidence": round_prob(score_confidence(p)),
    }
