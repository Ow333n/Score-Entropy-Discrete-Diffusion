"""Swap-SFT 目标的重计算实现 (protocol §13.2, §3.6)。

真实 objective:
    L_swap = lam * (1/N_pair) * sum_n delta_n^2
其中 delta 本身依赖模型参数。8GB 显存约束下采用重计算 + sg(delta) 系数技巧:

    Pass A (no-grad): 3 次 forward (C, C+a, C+b) -> 标量 delta (detach)
    Pass B (grad):   以 2*lam*sg(delta)/N 为固定系数, 分别 backward 三个状态的
                     logp 线性组合 (逐个 backward, 每次只保留一个状态的计算图):
                         C:   +coef * (log p(a|C) - log p(b|C))
                         C+a: +coef * log p(b|C,a)
                         C+b: -coef * log p(a|C,b)

这得到的是 lam*delta^2 的 exact first-order gradient (protocol §3.6),
由 test_swap_gradient.py 在 tiny model 上与 full-autograd 对照验证 (§4.9,
epsilon_g ≲ 1e-5, 不通过禁止进入 Swap-SFT)。

显存: 峰值 = 单状态的 forward+backward 激活 (与同 batch 普通训练相当);
算力: 6F + 3B ≈ 3.5-4x Vanilla (protocol §13.3)。
调用方负责 zero_grad (与 losses.py 的 accum 约定一致: 多个 micro-batch 的梯度
天然累加, accum_iter==accum 时才 optimizer.step)。
"""
import torch

from compatibility.posterior import clean_log_probs


def swap_recompute_loss(score_fn, x_t, i, j, a, b, sigma, D, lam):
    """计算 L_swap = lam * mean_n(delta_n^2), 执行 backward (exact first-order)。

    Args:
        score_fn: 支持梯度的 score_fn(x_t, sigma) -> [B, L, D]
        x_t: [B, L], i,j 位置为 MASK; i,j,a,b: [B]; sigma: [B]
        lam: swap 权重 (scalar)

    Returns: loss_value (detached scalar, 供日志)。
    """
    B = x_t.shape[0]
    idx = torch.arange(B, device=x_t.device)

    # Pass A: no-grad, 3 次 forward 得到 sg(delta)
    with torch.no_grad():
        s_C = score_fn(x_t, sigma)
        lp_C = clean_log_probs(s_C, D)
        pa_C = lp_C[idx, i, a]
        pb_C = lp_C[idx, j, b]
        del s_C, lp_C

        x_Ca = x_t.clone()
        x_Ca[idx, i] = a
        s_Ca = score_fn(x_Ca, sigma)
        lp_Ca = clean_log_probs(s_Ca, D)
        pb_Ca = lp_Ca[idx, j, b]
        del s_Ca, lp_Ca

        x_Cb = x_t.clone()
        x_Cb[idx, j] = b
        s_Cb = score_fn(x_Cb, sigma)
        lp_Cb = clean_log_probs(s_Cb, D)
        pa_Cb = lp_Cb[idx, i, a]
        del s_Cb, lp_Cb

        delta = pa_C + pb_Ca - pb_C - pa_Cb                       # [B]
        coef = 2.0 * lam * delta / B                              # d/dtheta[lam*mean(delta^2)] 的系数
        loss_value = lam * (delta ** 2).mean().detach()

    # Pass B: 3 个状态逐个 forward+backward, 每次只保留一个图
    s_C = score_fn(x_t, sigma)
    lp_C = clean_log_probs(s_C, D)
    term_C = (coef * (lp_C[idx, i, a] - lp_C[idx, j, b])).sum()
    term_C.backward()
    del s_C, lp_C

    s_Ca = score_fn(x_Ca, sigma)
    lp_Ca = clean_log_probs(s_Ca, D)
    term_Ca = (coef * lp_Ca[idx, j, b]).sum()
    term_Ca.backward()
    del s_Ca, lp_Ca

    s_Cb = score_fn(x_Cb, sigma)
    lp_Cb = clean_log_probs(s_Cb, D)
    term_Cb = (-coef * lp_Cb[idx, i, a]).sum()
    term_Cb.backward()
    del s_Cb, lp_Cb

    return loss_value
