"""G0 §4.7: residual time-dependence 检查 (同一 context 在 t1/t2/t3 上的后验 JSD)。

RADD 说 absorbing 的条件后验 p_hat(v|C) 应与 sigma 无关 (时间标量在 softmax 中
消掉); 但 unconstrained score 网络可能违反 (M2S 已报告 score-vector violations)。
本检查在冻结 sigma 网格 {0.5, 1.5, 3.0} 上量化真实模型的残余时间依赖:
同一个 mask 状态 x_t, 三个 sigma 下各算一次 p_hat, 逐 mask 位置算两两 JSD。
另做 §4.2-④ 确定性检查 (同输入重复 forward 逐位一致)。

用法: .venv/bin/python scripts/residual_time_check.py [--n_blocks 2]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from load_model import load_model
from data import get_dataset
from model import utils as mutils
from compatibility.posterior import clean_log_probs, check_deterministic

MASK_TOKEN = 50257


def jsd(lp, lq):
    """Jensen-Shannon divergence between two log-prob vectors (最后维)。"""
    p, q = lp.exp(), lq.exp()
    m = 0.5 * (p + q)
    log_m = m.log()
    return 0.5 * (p * (lp - log_m)).sum(-1) + 0.5 * (q * (lq - log_m)).sum(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="louaaron/sedd-small")
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--sigma_grid", default="0.5,1.5,3.0")
    parser.add_argument("--out", default="results/pretrained/residual_time.json")
    args = parser.parse_args()
    sigmas = [float(x) for x in args.sigma_grid.split(",")]

    device = torch.device("cuda")
    model, graph, noise = load_model(args.model_path, device)
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)

    ds = get_dataset("wikitext103", "test", cache_dir="data", block_size=256, num_proc=4)
    torch.manual_seed(0)

    jsd_pairs = {f"{a},{b}": [] for i, a in enumerate(sigmas) for b in sigmas[i + 1:]}
    n_masked_total = 0
    with torch.no_grad():
        for bi in range(args.n_blocks):
            x0 = ds[bi]["input_ids"].to(device)[None]                 # [1, L]
            # 同一状态 C: 以 sigma_c=1.5 腐蚀
            x_t = x0.clone()
            move = torch.rand_like(x0.float()) < (1 - torch.exp(torch.tensor(-1.5)))
            x_t[move] = MASK_TOKEN
            mask = x_t == MASK_TOKEN
            n_masked_total += mask.sum().item()
            if mask.sum() == 0:
                continue

            logps = {}
            for s in sigmas:
                sigma_b = s * torch.ones(1, device=device)
                out = score_fn(x_t, sigma_b)
                logps[s] = clean_log_probs(out, graph.dim)

            for i, a in enumerate(sigmas):
                for b in sigmas[i + 1:]:
                    d = jsd(logps[a], logps[b])[mask]                 # 只统计 mask 位置
                    jsd_pairs[f"{a},{b}"].append(d.mean().item())

        # §4.2-④ 确定性
        identical, max_diff = check_deterministic(score_fn, x_t, sigmas[1] * torch.ones(1, device=device))

    summary = {k: dict(mean=sum(v) / len(v)) for k, v in jsd_pairs.items()}
    summary["deterministic"] = identical
    summary["max_abs_diff_repeat_eval"] = max_diff
    summary["n_masked_positions"] = n_masked_total
    summary["sigma_grid"] = sigmas

    print("=" * 60)
    print(f"G0 §4.7 residual time-dependence (同一状态 C 上的后验 JSD, {args.n_blocks} 块)")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  JSD(sigma={k.split(',')[0]}, {k.split(',')[1]}): {v['mean']:.5f}")
    print(f"  确定性检查: {identical} (max diff {max_diff:.2e})")
    print(f"  mask 位置总数: {n_masked_total}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
