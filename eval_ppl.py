"""严格按论文 v3 (arXiv:2310.16834) 协议评估 SEDD 的 wikitext103 perplexity。

协议要素 (Appendix C.1/C.6 + Theorem 3.6/Eq.9):
  1. 每个样本采 1000 个随机 timestep t ~ U[eps, 1] 做 MC 估计 DWDSE 积分
     (Eq.10: ∫ σ'(t) E_{x_t}[SE] dt)
  2. 加 prior KL 项: D_KL(p_{T|0}(·|x0) || p_base),
     p_base = MASK + 泄漏到随机非 MASK token (C.1), KL ≈ eps * log(d-1) 每位置
  3. invertible byte-level tokenizer (GPT-2) ✓ data.py 已用
  4. WikiText103 test set ✓
  5. unconditional、无 sliding window ✓ (非重叠 1024 分块)
  6. 输出 PPL = exp(nats/token), 与论文 Table 1 的 SEDD-small Absorb <= 43.14 对比

用法: .venv/bin/python eval_ppl.py [--n_blocks 16] [--n_t 1000] [--t_chunk 16]
"""
import argparse
import math
import warnings

import torch

warnings.filterwarnings("ignore")

from load_model import load_model
from data import get_dataset
from model import utils as mutils

EPS = 1e-3  # loglinear schedule 的 eps (C.1: 1e-3 或 1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="louaaron/sedd-small")
    parser.add_argument("--n_blocks", type=int, default=16, help="评估多少个 1024-token 块")
    parser.add_argument("--n_t", type=int, default=1000, help="每块采多少个随机 t (论文=1000)")
    parser.add_argument("--t_chunk", type=int, default=16, help="一次前向放多少个 t (显存受限)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, graph, noise = load_model(args.model_path, device)
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)

    ds = get_dataset("wikitext103", "test", cache_dir="data", block_size=1024, num_proc=4)
    print(f"wikitext103 test 分块数: {len(ds)}, 本次评估前 {args.n_blocks} 块")

    # prior KL 项: p_{T|0} = eps*e_{x0} + (1-eps)*e_MASK, p_base = (1-eps)*e_MASK + eps*U(非MASK)
    # KL = eps * log(eps / (eps/(d-1))) = eps * log(d-1)  每位置, 与 x0 无关
    prior_kl = EPS * math.log(graph.dim - 1)
    print(f"prior KL 项 (每位置): {prior_kl:.4f} nats (可忽略, 仅按 Eq.9 补齐)")

    torch.manual_seed(args.seed)

    block_nats = []
    with torch.no_grad():
        for bi in range(args.n_blocks):
            x0 = ds[bi]["input_ids"].cuda()  # [1024]
            x0 = x0[None]                    # [1, 1024]

            losses = []
            for start in range(0, args.n_t, args.t_chunk):
                n = min(args.t_chunk, args.n_t - start)
                t = (1 - EPS) * torch.rand(n, device=device) + EPS       # [n]
                sigma, dsigma = noise(t)
                x0_rep = x0.repeat(n, 1)
                x_t = graph.sample_transition(x0_rep, sigma[:, None])    # 每个 t 独立采样 x_t
                log_score = score_fn(x_t, sigma)
                se = graph.score_entropy(log_score, sigma[:, None], x_t, x0_rep)
                losses.append((dsigma[:, None] * se).sum(-1))            # [n] 每序列 DWDSE 积分项

            block_loss = torch.cat(losses).mean() / x0.numel()           # 1000-t MC, nats/token
            block_nats.append(block_loss.item())

    mean_nats = sum(block_nats) / len(block_nats)
    total_nats = mean_nats + prior_kl
    ppl = math.exp(total_nats)
    sd = (sum((x - mean_nats) ** 2 for x in block_nats) / len(block_nats)) ** 0.5

    print(f"\n{'='*60}")
    print(f"DWDSE 积分项 (1000-t MC, {args.n_blocks} 块): {mean_nats:.3f} nats/token (±{sd:.3f} 块间)")
    print(f"prior KL 项:                                      +{prior_kl:.4f}")
    print(f"总 bound (Eq.9):                                  {total_nats:.3f} nats/token")
    print(f"Perplexity:                                       {ppl:.2f}")
    print(f"论文 v3 Table 1: SEDD-small Absorb ≤ 43.14 (GPT-2 small 41.60)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
