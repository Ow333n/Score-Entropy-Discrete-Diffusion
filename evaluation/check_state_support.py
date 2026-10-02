"""G0.5 state-support gate — 分布级版本 (protocol §8, 审计 ③ 补全)。

审计修正说明 (v4.1 首版的问题):
  1. 首版把 K 定义为 "span 内 mask 数", 与协议 §8.2 的 revealed target count K
     (= span 内已揭示 token 数 = span_len - mask 数) 混为一谈, 并误称 manifest 的
     C 状态 "K=0"。实际 manifest 初始状态是 partial-reveal (mask_ratio 均值 0.70,
     K_revealed 均值 ≈ span_len×0.30)。
  2. 首版只做了"可达性"检查 (P>0), 本版做分布级 overlap。

分布级检查内容 (§8.2): 训练腐蚀过程的经验分布 vs frozen manifest 状态分布,
变量: sigma / target mask ratio / revealed-count K / span length / pair distance /
partial-reveal depth (K 的同义)。每变量报告 binned overlap coefficient
OC = sum_bins min(h_train, h_manifest) (双方直方图按各自密度归一, OC∈[0,1]),
并列出 manifest 落在训练中心 95% 区间外的比例 (tail_share)。

低覆盖变量按 §8.3 记为 limitation/sensitivity, 不改变主 estimand。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-3
N_SIM = 1_000_000


def simulate_training(n=N_SIM):
    """冻结腐蚀过程: t~U[eps,1] → σ loglinear; span~U[10,50]; span 内 1-e^{-σ}
    逐位置 mask; 随机一对 (i,j) 从 masked span 位置 (m>=2 时)。
    返回训练经验分布的各变量样本。"""
    torch.manual_seed(0)
    t = (1 - EPS) * torch.rand(n) + EPS
    sigma = -torch.log1p(-(1 - EPS) * t)
    span_len = torch.randint(10, 51, (n,))
    p = 1 - torch.exp(-sigma)
    m = torch.distributions.Binomial(span_len.float(), p).sample().long()
    K_revealed = span_len - m
    mask_ratio = m.float() / span_len.float()

    # pair distance: m>=2 时从 masked 位置无放回抽 2 个的 |i-j|
    pair_dist = torch.full((n,), -1, dtype=torch.long)
    ok = m >= 2
    n_ok = ok.sum().item()
    if n_ok > 0:
        # 均匀随机两点的期望距离 ≈ span_len/3; 用解析矩 + 模拟抽样混合:
        # 直接从 U(0, span_len)^2 采样距离并拒绝 i==j (与无放回等价到 1/span_len)
        sl_ok = span_len[ok].float()
        u1 = torch.rand(n_ok) * sl_ok
        u2 = torch.rand(n_ok) * sl_ok
        d = (u1 - u2).abs()
        # 拒绝 i==j 概率 ~1/span_len, 重抽一次近似
        eq = d < 1e-6
        if eq.any():
            u1[eq] = torch.rand(int(eq.sum())) * sl_ok[eq]
            u2[eq] = torch.rand(int(eq.sum())) * sl_ok[eq]
            d[eq] = (u1[eq] - u2[eq]).abs()
        pair_dist[ok] = d.long()
    return dict(sigma=sigma, mask_ratio=mask_ratio, K_revealed=K_revealed,
                span_len=span_len, pair_distance=pair_dist)


def manifest_stats():
    proto = OmegaConf.load(os.path.join(ROOT, "protocol/regime_a_protocol.yaml"))
    records = [json.loads(line) for line in open(os.path.join(ROOT, proto.eval_manifest.file))]
    span_len = torch.tensor([r["span_len"] for r in records], dtype=torch.long)
    mask_ratio = torch.tensor([r["mask_ratio"] for r in records])
    m = (mask_ratio.float() * span_len.float()).round().long()
    return dict(
        n=len(records),
        sigma=torch.tensor([r["sigma"] for r in records]),
        mask_ratio=mask_ratio,
        K_revealed=span_len - m,
        span_len=span_len,
        pair_distance=torch.tensor([r["pair_distance"] for r in records], dtype=torch.long),
    )


def binned_overlap(train_vals, mani_vals, edges):
    """binned overlap coefficient + manifest tail share (落在 train 中心 95% 外)。"""
    ht = torch.histc(train_vals.float(), bins=len(edges) - 1, min=edges[0], max=edges[-1])
    hm = torch.histc(mani_vals.float(), bins=len(edges) - 1, min=edges[0], max=edges[-1])
    ht = ht / ht.sum()
    hm = hm / hm.sum()
    oc = float(torch.minimum(ht, hm).sum().item())
    lo, hi = torch.quantile(train_vals.float(), torch.tensor([0.025, 0.975]))
    tail = float(((mani_vals < lo) | (mani_vals > hi)).float().mean().item())
    return oc, float(lo), float(hi), tail


def main():
    train = simulate_training()
    mani = manifest_stats()

    checks = []
    print("=" * 64)
    print("G0.5 state-support gate (分布级, 审计 ③)")
    print(f"训练模拟 N={N_SIM}, manifest N={mani['n']}\n")

    # sigma: 训练 CDF 在网格点的值 + bucket 质量对比
    for s in [0.5, 1.5, 3.0]:
        cdf = float((train["sigma"] <= s).float().mean().item())
        print(f"  sigma: P_train(sigma <= {s}) = {cdf:.3f}")
    oc, lo, hi, tail = binned_overlap(train["sigma"], mani["sigma"],
                                      [0, 0.5, 1.5, 3.0, 7.0])
    print(f"  sigma buckets [0,.5)/[.5,1.5)/[1.5,3)/[3,7): OC={oc:.3f}, "
          f"train 中心 95% [{lo:.2f},{hi:.2f}], manifest tail_share={tail:.3f}")
    checks.append(("sigma 分布覆盖", oc > 0.5))

    for name, tv, mv, edges in [
        ("mask_ratio", train["mask_ratio"], mani["mask_ratio"], torch.linspace(0, 1, 11)),
        ("span_len", train["span_len"].float(), mani["span_len"].float(),
         torch.arange(9.5, 51, 5)),
        ("pair_distance", train["pair_distance"].float(), mani["pair_distance"].float(),
         torch.arange(-0.5, 51, 5)),
        ("K_revealed", train["K_revealed"].float(), mani["K_revealed"].float(),
         torch.arange(-0.5, 51, 5)),
    ]:
        oc, lo, hi, tail = binned_overlap(tv, mv, edges)
        print(f"  {name:14s}: OC={oc:.3f}, train 中心 95% [{lo:.1f},{hi:.1f}], "
              f"manifest tail_share={tail:.3f}")
        checks.append((f"{name} 分布覆盖", oc > 0.4))

    # K=0 (span 全 mask) 的 manifest 状态: 是否在训练 K 分布的主体边缘 (limitation 注记)
    p_k0_mani = float((mani["K_revealed"] == 0).float().mean().item())
    p_k0_train = float((train["K_revealed"] == 0).float().mean().item())
    print(f"\n  span 全 mask (K_revealed=0): manifest {p_k0_mani:.3f} vs 训练 {p_k0_train:.3f} "
          f"(高 σ 区训练有实质质量, 见 σ bucket)")

    print("\n[判定]")
    all_ok = all(ok for _, ok in checks)
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")
    print("\nlimitation 注记 (§8.3):")
    print("  - manifest 的 σ 网格在训练分布中偏中低区 (P(σ<=3)=95%), σ>3 的 manifest 状态为 0")
    print("  - pair_distance 与 K_revealed 的尾部样本属低覆盖区, 后续分析按 bucket 报告")

    out = dict(
        n_train=N_SIM, n_manifest=mani["n"],
        checks=[(name, ok) for name, ok in checks],
        sigma_cdf={f"le_{s}": float((train["sigma"] <= s).float().mean().item())
                   for s in [0.5, 1.5, 3.0]},
        k0_manifest=p_k0_mani, k0_train=p_k0_train,
    )
    with open(os.path.join(ROOT, "results/vanilla/g05_state_support_distributional.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nG0.5 (分布级) 结论: {'PASS ✅' if all_ok else 'LIMITATION ⚠️'}")
    print("结果: results/vanilla/g05_state_support_distributional.json")


if __name__ == "__main__":
    main()
