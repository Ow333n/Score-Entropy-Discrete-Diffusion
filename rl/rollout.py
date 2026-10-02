"""RL rollout (plan v0.2 §17, §25, §43): analytic predictor, no_grad, K=1 预采样 J。

- 从 frozen manifest 样本的 (x_t0, σ0) 出发 reverse 到 eps (denoiser 收尾)
- 每轨迹 J 从 σ_J ≥ SIGMA_GATE_MIN 的步 uniform 采样 (§25 精神: 排除 staggered 近似
  失效的 σ→0 末段; 修订候选, 见 SIGMA_GATE_MIN 注释)
- 只缓存 J 步的 (x_J, x_next, action, old_logπ, sigma_J, dsigma_J) + final_x,
  不落盘完整轨迹 (debug subset 除外)
- validity gate 两级: σ ≥ SIGMA_GATE_MIN → 硬检查 (违反 abort); σ < 阈值 → 软统计
  (负权重直接进 gumbel 采样, 与 repo 原版 AnalyticPredictor 行为一致, 数值安全)
- 所有输出 detached CPU tensors; 采样用全局 CPU RNG (调用方 seed)
"""
import torch

from catsample import sample_categorical
from model import utils as mutils
from rl import transition as T

EPS = 1e-3
SIGMA_GATE_MIN = 0.05   # staggered 近似在 σ<0.05 段理论上可失效 (Σscore≈1/(e^σ−1) 逼近负值阈值 e^{dσ}/(e^{dσ}−1), 实测 step127/σ=0.0085 出现 1 元素负值)


def _sampling_score_fn(model):
    return mutils.get_score_fn(model, train=False, sampling=True)


def rollout_chunk(model, sampling_score_fn, graph, noise, x0, x_t0, sigma0, steps,
                  seed, mask_token=50257, eps=1e-3):
    """chunk 条并行 rollout (显存友好: one_hot [B,256,50258] 大, B≤8)。

    x0/x_t0/sigma0: [B,L]/[B,L]/[B]。返回 (list of dicts, gate_stats)。
    """
    torch.manual_seed(seed)
    device = x0.device
    B = x0.shape[0]
    t0 = ((1 - (-sigma0).exp()) / (1 - eps)).clamp(min=eps + 1e-6)
    # 预计算每轨迹 σ 轨迹 → J 的 valid 采样范围 (σ_J ≥ SIGMA_GATE_MIN)
    ts_mat = eps + (t0[:, None] - eps) * (1 - (torch.arange(1, steps + 1, device=device) / steps)[None, :])
    sig_mat = noise.total_noise(ts_mat)                      # [B, steps]
    valid = sig_mat >= SIGMA_GATE_MIN
    J = torch.empty(B, dtype=torch.long, device=device)
    for b in range(B):
        v = valid[b].nonzero().squeeze(-1)
        J[b] = v[torch.randint(0, len(v), (1,), device=device)[0]]
    x = x_t0.clone()
    t_cur = t0.clone()
    saved = [None] * B
    gate_stats = dict(soft_neg_elements=0, soft_steps=0)
    with torch.no_grad():
        for i in range(steps):
            t_next = eps + (t0 - eps) * (1 - (i + 1) / steps)
            sigma_cur = noise.total_noise(t_cur)
            sigma_next = noise.total_noise(t_next)
            dsigma = (sigma_cur - sigma_next)[:, None]
            s = sampling_score_fn(x, sigma_cur.squeeze(-1))
            stag = T.staggered_score_fn(s, dsigma)
            trans = graph.transp_transition(x, dsigma)
            w = stag * trans
            ok, stats = T.validity_check(w)
            if not ok:
                if (sigma_cur >= SIGMA_GATE_MIN).all():
                    sigma_vals = sigma_cur.squeeze(-1).tolist()
                    raise RuntimeError(
                        f"rollout validity gate 硬失败 step={i}: {stats} | "
                        f"σ range=[{min(sigma_vals):.4f},{max(sigma_vals):.4f}] "
                        f"dσ={float(dsigma[0, 0]):.4f}")
                else:
                    gate_stats["soft_neg_elements"] += stats["n_neg"]
                    gate_stats["soft_steps"] += 1
            x_next = sample_categorical(w)
            hit = J == i
            if hit.any():
                logpi = T.policy_log_probs(w)
                lp = T.log_probs_for_actions(logpi, x_next)
                for b in range(B):
                    if hit[b]:
                        saved[b] = dict(
                            x_J=x[b].cpu(), x_next=x_next[b].cpu(), action=x_next[b].cpu(),
                            old_logpi=lp[b].cpu(), mask_J=(x[b] == mask_token).cpu(),
                            sigma_J=float(sigma_cur[b].item()),
                            dsigma_J=float(dsigma[b, 0].item()), J=int(J[b].item()))
            x = x_next
            t_cur = t_next
        # denoiser 收尾 (与 sampling.py 一致)
        s = sampling_score_fn(x, sigma_next.squeeze(-1))
        stag = T.staggered_score_fn(s, sigma_next[:, None])
        probs = stag * graph.transp_transition(x, sigma_next[:, None])
        probs = probs[..., :-1]
        x = sample_categorical(probs)
    for b in range(B):
        assert saved[b] is not None
        saved[b]["final_x"] = x[b].cpu()
    return saved, gate_stats
