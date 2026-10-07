"""Phase 1 FP32 aux diagnostics（DIAG_PROTOCOL_VERSION=v1，precision 同 Level C）。

按 frozen Phase 1 协议 §5 secondary 的既有定义（v2.1 §9.4 同款口径）：
- entropy：x_t 状态全部 mask 位置的预测分布熵 H(p̂) 均值（per-sample）
- confidence：mask 位置 argmax 概率的均值（per-sample）
- reliability 曲线：per-position (confidence 分桶, 经验命中率)——10 桶
- score-scale summary：mask 位置 GT logp 的 mean/SD（与 local CE 同源）

score_fn = fp32 镜像 forward（复用 evaluation/eval_diag_fp32.py 的 fp32_forward，
不改任何 frozen 文件）；提取 fp32；聚合 fp64；单 forward/样本（x_t 状态）。

用法:
  .venv/bin/python evaluation/eval_diag_fp32_aux.py --model_path DIR \
      --ckpt checkpoint_2500.pth --weights ema --tag v21-H1-2500 \
      --out results/phase1_diag/aux/v21-H1-2500.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from compatibility.posterior import clean_log_probs
from evaluation.eval_diag_fp32 import fp32_forward, load_model, MANIFEST, MANIFEST_SHA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 50258
DIAG_PROTOCOL_VERSION = "v1"
PRECISION_MODE = "true_fp32_mirrored_forward"
N_BUCKETS = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--tag", default="diag-fp32-aux")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    digest_man = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    assert digest_man == MANIFEST_SHA, "manifest sha 变化，硬停"

    model, step, param_dtypes = load_model(args.model_path, args.ckpt, args.weights)
    records = [json.loads(line) for line in open(MANIFEST)]

    entropy, confidence, ce = [], [], []
    confs_all, hits_all = [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for start in range(0, len(records), args.chunk):
            ch = records[start:start + args.chunk]
            x_t = torch.stack([torch.tensor(r["initial_state"]) for r in ch]).to("cuda")
            x0 = torch.stack([torch.tensor(r["x0"]) for r in ch]).to("cuda")
            sigma = torch.tensor([r["sigma"] for r in ch], device="cuda")
            logp = clean_log_probs(fp32_forward(model, x_t, sigma), D).double()
            mask = x_t == D - 1
            for k in range(len(ch)):
                pos = mask[k].nonzero().squeeze(1)
                if pos.numel() == 0:
                    entropy.append(float("nan")); confidence.append(float("nan"))
                    ce.append(float("nan")); continue
                lp = logp[k, pos]
                p = lp.exp()
                entropy.append((-p * lp).sum(-1).mean().item())
                confidence.append(lp.max(-1).values.exp().mean().item())   # 概率口径（非 log）
                gold = x0[k, pos]
                ce.append((-lp[torch.arange(pos.numel()), gold]).mean().item())
                confs_all.extend(lp.max(-1).values.exp().tolist())
                hits_all.extend((lp.argmax(-1) == gold).tolist())
            del x_t, x0, logp
    peak = torch.cuda.max_memory_allocated() / 1e9

    confs_t = torch.tensor(confs_all)
    hits_t = torch.tensor(hits_all, dtype=torch.float64)
    order = confs_t.argsort()
    edges = torch.linspace(0, confs_t.numel(), N_BUCKETS + 1).long()
    reliability = []
    for k in range(N_BUCKETS):
        sel = order[edges[k]:edges[k + 1]]
        if sel.numel() == 0:
            continue
        reliability.append(dict(bucket=k, n=int(sel.numel()),
                                conf_lo=confs_t[sel].min().item(),
                                conf_hi=confs_t[sel].max().item(),
                                mean_conf=confs_t[sel].mean().item(),
                                hit_rate=hits_t[sel].mean().item()))

    e_t = torch.tensor([v for v in entropy if v == v])
    c_t = torch.tensor([v for v in confidence if v == v])
    ce_t = torch.tensor([v for v in ce if v == v])
    gt_logp_sd = ce_t.std(unbiased=True).item()

    payload = dict(
        tag=args.tag,
        diag_protocol_version=DIAG_PROTOCOL_VERSION,
        precision_mode=PRECISION_MODE,
        evaluator_file=os.path.relpath(os.path.abspath(__file__), ROOT),
        evaluator_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        model_path=args.model_path, ckpt=args.ckpt, weights=args.weights, step=step,
        model_param_dtypes=param_dtypes,
        compute_dtype="fp32（镜像 forward）", extraction_dtype="fp32", aggregation_dtype="fp64",
        manifest_file=os.path.relpath(MANIFEST, ROOT), manifest_sha256=MANIFEST_SHA,
        n_samples=len(records),
        summary=dict(
            entropy_mean=e_t.mean().item() if e_t.numel() else float("nan"),
            entropy_sem=e_t.std(unbiased=True).item() / e_t.numel() ** 0.5 if e_t.numel() else float("nan"),
            confidence_mean=c_t.mean().item() if c_t.numel() else float("nan"),
            confidence_sem=c_t.std(unbiased=True).item() / c_t.numel() ** 0.5 if c_t.numel() else float("nan"),
            gt_logp_mean=ce_t.mean().item() if ce_t.numel() else float("nan"),
            gt_logp_sd=gt_logp_sd,
            n_positions=len(confs_all)),
        reliability_curve=reliability,
        per_sample=dict(
            entropy=entropy, confidence=confidence, local_ce=ce),
        peak_vram_gb=peak)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    s = payload["summary"]
    print(f"FP32 aux（{args.tag}）: entropy={s['entropy_mean']:.4f} conf={s['confidence_mean']:.4f} "
          f"gt_logp_mean={s['gt_logp_mean']:.4f} | VRAM {peak:.2f}GB")
    print(f"结果: {args.out}")


if __name__ == "__main__":
    main()
