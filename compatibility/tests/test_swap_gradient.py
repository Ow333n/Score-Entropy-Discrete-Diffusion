"""§4.9 全 autograd vs 重计算梯度等价测试 (Swap-SFT 的进入门)。

规格: tiny model vocab=10, hidden=8, seq_len=4, batch=2, dropout=0。
    g_full       = grad of L = lam * mean(delta^2)  (三个状态图同时存活)
    g_recompute  = swap_recompute_loss 的 sg(delta) 重计算实现 (生产路径)
要求 epsilon_g = ||g_full - g_recompute|| / (||g_full|| + eps) ≲ 1e-5 (§4.9)。

🛑 不通过禁止进入 Swap-SFT。
另测: 梯度累积语义 (多次 backward 累加, 与 losses.py accum 约定一致)。"""
import math

import torch

from compatibility.posterior import clean_log_probs
from compatibility.tests.toy_models import TinyScoreModel, D, MASK
from training.swap import swap_recompute_loss


def _make_batch():
    torch.manual_seed(0)
    x_t = torch.full((2, 4), MASK, dtype=torch.long)
    x_t[0, 3] = 5            # 一些可见位置
    x_t[1, 0] = 2
    i = torch.tensor([0, 1])
    j = torch.tensor([1, 2])
    a = torch.tensor([3, 7])
    b = torch.tensor([7, 4])
    sigma = torch.tensor([1.0, 2.0])
    return x_t, i, j, a, b, sigma


def _full_autograd_grads(model, x_t, i, j, a, b, sigma, lam):
    """三个状态图全部保留的完整 autograd 路径 (仅测试对照用)。"""
    model.zero_grad()
    B = x_t.shape[0]
    idx = torch.arange(B)

    s_C = model(x_t, sigma)
    lp_C = clean_log_probs(s_C, D)
    x_Ca = x_t.clone(); x_Ca[idx, i] = a
    s_Ca = model(x_Ca, sigma)
    lp_Ca = clean_log_probs(s_Ca, D)
    x_Cb = x_t.clone(); x_Cb[idx, j] = b
    s_Cb = model(x_Cb, sigma)
    lp_Cb = clean_log_probs(s_Cb, D)

    delta = lp_C[idx, i, a] + lp_Ca[idx, j, b] - lp_C[idx, j, b] - lp_Cb[idx, i, a]
    loss = lam * (delta ** 2).mean()
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    return loss.item(), grads


def _grad_l2(grads):
    return math.sqrt(sum((g ** 2).sum().item() for g in grads))


def test_gradient_equivalence_fp32():
    """protocol §4.9 规格 (fp32) 下 epsilon_g ≲ 1e-5。"""
    lam = 0.7
    x_t, i, j, a, b, sigma = _make_batch()

    model = TinyScoreModel()
    loss_full, g_full = _full_autograd_grads(model, x_t, i, j, a, b, sigma, lam)

    model.zero_grad()
    loss_re = swap_recompute_loss(model, x_t, i, j, a, b, sigma, D, lam)
    g_re = [p.grad.clone() for p in model.parameters()]

    assert abs(loss_full - loss_re.item()) < 1e-5 * max(1.0, abs(loss_full))

    diff = _grad_l2([gf - gr for gf, gr in zip(g_full, g_re)])
    eps_g = diff / (_grad_l2(g_full) + 1e-8)
    assert eps_g < 1e-5, f"epsilon_g = {eps_g:.2e} 超过 1e-5 (fp32 目标)"


def test_gradient_equivalence_fp64():
    """fp64 下 epsilon_g ≲ 1e-9 (更强的数值检查)。"""
    lam = 0.7
    x_t, i, j, a, b, sigma = _make_batch()

    model = TinyScoreModel().double()
    x_t = x_t.clone()
    loss_full, g_full = _full_autograd_grads(model, x_t, i, j, a, b, sigma.double(), lam)

    model.zero_grad()
    loss_re = swap_recompute_loss(model, x_t, i, j, a, b, sigma.double(), D, lam)
    g_re = [p.grad.clone() for p in model.parameters()]

    assert abs(loss_full - loss_re.item()) < 1e-9

    diff = _grad_l2([gf - gr for gf, gr in zip(g_full, g_re)])
    eps_g = diff / (_grad_l2(g_full) + 1e-12)
    assert eps_g < 1e-9, f"epsilon_g = {eps_g:.2e} 超过 1e-9 (fp64 目标)"


def test_gradient_accumulation_semantics():
    """连续两次 backward 不 zero_grad → 梯度累加为 2 倍 (losses.py accum 约定)。"""
    lam = 0.5
    x_t, i, j, a, b, sigma = _make_batch()
    model = TinyScoreModel()

    swap_recompute_loss(model, x_t, i, j, a, b, sigma, D, lam)
    g_once = [p.grad.clone() for p in model.parameters()]
    swap_recompute_loss(model, x_t, i, j, a, b, sigma, D, lam)
    g_twice = [p.grad.clone() for p in model.parameters()]

    for g1, g2 in zip(g_once, g_twice):
        assert torch.allclose(g2, 2 * g1, atol=1e-6)
