import json, sys
import torch
D = "/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/calibration"
hf = torch.load(f"{D}/hf_aime24_logits.pt")["logits"]
out = {}
for m in sys.argv[1:]:
    tt = torch.load(f"{D}/tt_m{m}.pt")
    for path in ("prefill", "decode"):
        t = tt[path][:, : hf.shape[1]]
        lp_h, lp_t = hf.log_softmax(-1), t.log_softmax(-1)
        kl = (lp_h.exp() * (lp_h - lp_t)).sum(-1)
        top_h, top_t = hf.max(-1).values, t.gather(1, hf.argmax(-1, keepdim=True))[:, 0]
        p_h, p_t = lp_h.exp().max(-1).values, lp_t.exp().gather(1, hf.argmax(-1, keepdim=True))[:, 0]
        idx = hf.topk(100, -1).indices
        x, y = hf.gather(1, idx), t.gather(1, idx)
        xc, yc = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
        slope = (xc * yc).sum(1) / (xc * xc).sum(1)
        unsat = p_h < 0.99
        q = lambda v: [round(float(v.quantile(a)), 4) for a in (0.1, 0.5, 0.9, 1.0)]
        out[f"m{m}_{path}"] = dict(
            positions=len(kl), kl_quantiles_10_50_90_max=q(kl), kl_mean=round(float(kl.mean()), 4),
            top1_logit_tt_minus_hf_quantiles=q(top_t - top_h), top1_logit_diff_mean=round(float((top_t - top_h).mean()), 3),
            unsaturated_positions=int(unsat.sum()),
            unsat_hf_top1_prob_mean=round(float(p_h[unsat].mean()), 4), unsat_tt_prob_of_hf_top1_mean=round(float(p_t[unsat].mean()), 4),
            unsat_top1_logit_diff_mean=round(float((top_t - top_h)[unsat].mean()), 3),
            top100_slope_quantiles=q(slope), argmax_agree=int((t.argmax(-1) == hf.argmax(-1)).sum()))
print(json.dumps(out, indent=1))
