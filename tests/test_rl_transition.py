"""RL correctness 单测 (plan v0.2 §34): transition/policy/logπ/advantage/PPO/timestep-MC。

覆盖 §34 的 1-13, 15-17, 20-22 (CPU 可跑部分); 14 (RL step0) 与 GPU 项在 RL-G0 阶段。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import graph_lib
from rl import transition as T

D = 8            # toy vocab 7 + MASK
MASK = D - 1
DSIGMA = 0.05


def _graph():
    return graph_lib.Absorbing(D - 1)


def _toy_score(B=2, L=5, seed=0):
    """正真 score [B, L, D] (exp 后), MASK 列 = 1.0。

    采样路径: forward 里 scatter 把 x_t 位置 logit 置 0 → get_score_fn(sampling=True)
    返回 exp(score) → MASK 位置 score[MASK] = exp(0) = 1 (实测 /tmp/stag_debug.py)。
    """
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(B, L, D, generator=g) + 0.1
    s[..., MASK] = 1.0
    return s


def _toy_w(B=2, L=5, seed=0):
    graph = _graph()
    x_t = torch.full((B, L), MASK, dtype=torch.long)
    dsigma = torch.full((B, 1), DSIGMA)   # staggered_score 需要 [B,1] (与采样器一致)
    return graph, x_t, T.transition_weights(graph, x_t, _toy_score(B, L, seed), dsigma)


# --- §34.1: 正规化 Σπ = 1 ---
def test_policy_normalized():
    _, _, w = _toy_w()
    logpi = T.policy_log_probs(w)
    pi = logpi.exp()
    assert torch.allclose(pi.sum(-1), torch.ones_like(pi.sum(-1)), atol=1e-6)

# --- §34.2: sampler 频率 == 正规化权重 ---
def test_sampler_frequency_matches_weights():
    graph, x_t, w = _toy_w(B=1, L=1, seed=1)
    pi = w / w.sum(-1, keepdim=True)
    n = 20000
    counts = torch.zeros(D)
    for i in range(n):
        torch.manual_seed(1000 + i)
        a = T.sample_actions(w)
        counts[a.item()] += 1
    freq = counts / n
    assert (freq - pi[0, 0]).abs().max().item() < 0.02, (freq, pi[0, 0])

# --- §34.3: 采样动作 logπ finite ---
def test_sampled_action_logpi_finite():
    _, _, w = _toy_w(seed=3)
    logpi = T.policy_log_probs(w)
    torch.manual_seed(7)
    a = T.sample_actions(w)
    lp = T.log_probs_for_actions(logpi, a)
    assert torch.isfinite(lp).all()

# --- §34.4: old logπ detached (rollout no_grad 路径) ---
def test_logpi_detached_in_nograd():
    _, _, w = _toy_w()
    with torch.no_grad():
        logpi = T.policy_log_probs(w)
        assert not logpi.requires_grad
    w2 = _toy_w()[2].clone().requires_grad_()
    logpi2 = T.policy_log_probs(w2)
    assert logpi2.requires_grad

# --- §34.5: 同一 w 重算逐位一致 ---
def test_recompute_bitwise_identical():
    _, _, w = _toy_w(seed=5)
    a = T.policy_log_probs(w)
    b = T.policy_log_probs(w.clone())
    assert (a == b).all()

# --- §34.6: PG 梯度有限非零 (真实公式链: θ → score → w → logπ → backward) ---
def test_pg_grad_finite():
    graph = _graph()
    x_t = torch.full((1, 1), MASK, dtype=torch.long)
    theta = torch.tensor([0.3, 0.7, 0.2, 0.4, 0.1, 0.6, 0.5, 0.0], requires_grad=True)
    score = theta.exp()[None, None, :].clone()
    score[..., MASK] = 1.0        # 采样路径 exp(0)=1
    dsigma = torch.tensor([0.05]) # 真实轨迹量级 (stag_MASK = e^{dσ} + (1-e^{dσ})Σscore > 0)
    w = T.transition_weights(graph, x_t, score, dsigma)
    ok, _ = T.validity_check(w)
    assert ok
    logpi = T.policy_log_probs(w)
    L = -1.0 * logpi[0, 0, 0]
    L.backward()
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().sum() > 0

# --- §34.7: reward 不进梯度路径 ---
def test_pg_loss_independent_of_reward():
    r = torch.tensor([0.5, 0.7, 0.9, 0.4]).requires_grad_()
    A = T.mean_centered_advantage(r.view(1, -1))
    _, _, w = _toy_w()
    w = w.clone().requires_grad_()
    logpi = T.policy_log_probs(w)
    L = (-A.detach().sum() * logpi[..., 0]).sum()
    L.backward()
    assert r.grad is None

# --- §34.8: advantage mean-centered ---
def test_advantage_mean_centered():
    r = torch.tensor([[0.2, 0.5, 0.8, 0.1], [1.0, 0.0, 0.5, 0.5]])
    A = T.mean_centered_advantage(r)
    assert torch.allclose(A.sum(dim=1), torch.zeros(2), atol=1e-6)

# --- §34.9: A=0 → 更新 ≈ 0 ---
def test_zero_advantage_zero_update():
    theta = torch.zeros(5, requires_grad=True)
    w = theta.exp()
    logpi = T.policy_log_probs(w)
    L = (-0.0 * logpi).sum()
    L.backward()
    assert theta.grad.abs().max() == 0.0

# --- §34.10/11: toy 正/负 advantage 增/减选中动作概率 ---
def test_positive_advantage_raises_prob():
    theta = torch.zeros(2, requires_grad=True)
    logpi = T.policy_log_probs(theta.exp())
    L = -1.0 * logpi[0]          # action 0, A=+1
    L.backward()
    with torch.no_grad():
        theta -= 0.5 * theta.grad
    pi = torch.softmax(theta, dim=-1)
    assert pi[0] > 0.5

def test_negative_advantage_lowers_prob():
    theta = torch.zeros(2, requires_grad=True)
    logpi = T.policy_log_probs(theta.exp())
    L = +1.0 * logpi[0]          # action 0, A=-1
    L.backward()
    with torch.no_grad():
        theta -= 0.5 * theta.grad
    pi = torch.softmax(theta, dim=-1)
    assert pi[0] < 0.5

# --- §34.13: seed 复现 ---
def test_seed_reproducible():
    _, _, w = _toy_w(seed=9)
    torch.manual_seed(123)
    a1 = T.sample_actions(w)
    torch.manual_seed(123)
    a2 = T.sample_actions(w)
    assert (a1 == a2).all()

# --- §34.15: analytic weight positivity + validity gate (真实 graph 公式) ---
def test_weights_positive_valid():
    _, _, w = _toy_w()
    ok, stats = T.validity_check(w)
    assert ok and stats["n_neg"] == 0 and stats["n_nonfinite"] == 0
    assert (w >= 0).all()

def test_validity_gate_negative_tolerance():
    w = torch.ones(1, 1, D)
    w[0, 0, 0] = -1e-9                  # numerical negative, rel 1e-9 ≤ 1e-6 → 放行 + 统计
    ok, stats = T.validity_check(w)
    assert ok and stats["n_neg"] == 1
    w[0, 0, 0] = -0.1                   # 真实负权重 → 拒绝
    ok, stats = T.validity_check(w)
    assert not ok

# --- §34.15b: staggered 负值条件 → gate 拒绝 (防御语义保留) ---
def test_validity_gate_rejects_negative_stag():
    graph = _graph()
    x_t = torch.full((1, 1), MASK, dtype=torch.long)
    dsigma = torch.tensor([2.0])                    # 大 dσ (128 步网格第一步量级)
    s = torch.full((1, 1, D), 10.0)                 # 大 Σscore (制造 Σscore > e^{dσ}/(e^{dσ}-1))
    s[..., MASK] = 1.0
    w = T.transition_weights(graph, x_t, s, dsigma)
    ok, stats = T.validity_check(w)
    assert not ok and stats["n_neg"] > 0            # gate 正确拒绝, 不 clamp

# --- §34.16: 非 MASK 位置 logπ = 0 (吸收语义, exclude from PG) ---
def test_nonmask_logpi_zero():
    graph = _graph()
    x_t = torch.tensor([[3, 1]])       # 非 MASK 位置
    dsigma = torch.tensor([DSIGMA, DSIGMA])
    w = T.transition_weights(graph, x_t, _toy_score(B=1, L=2, seed=2), dsigma)
    logpi = T.policy_log_probs(w)
    for pos in range(2):
        tok = x_t[0, pos].item()
        assert logpi[0, pos, tok].item() == 0.0
        other = logpi[0, pos, [j for j in range(D) if j != tok]]
        assert (other == -math.inf).all()

# --- §34.17: MASK→MASK stay 动作计入 policy 且对 θ 有梯度 ---
def test_stay_action_has_grad():
    graph = _graph()
    x_t = torch.full((1, 1), MASK, dtype=torch.long)
    theta = torch.tensor([0.3, 0.7, 0.2, 0.4, 0.1, 0.6, 0.5, 0.0], requires_grad=True)
    score = theta.exp()[None, None, :].clone()
    score[..., MASK] = 1.0
    dsigma = torch.tensor([0.05])
    w = T.transition_weights(graph, x_t, score, dsigma)
    ok, _ = T.validity_check(w)
    assert ok
    # stay 权重 = stag[MASK]·trans[MASK] > 0 (依赖 score 和 → 依赖 θ)
    assert w[0, 0, MASK].item() > 0
    logpi = T.policy_log_probs(w)
    L = -1.0 * logpi[0, 0, MASK]      # 对 stay 动作做 PG
    L.backward()
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().sum() > 0

# --- §34.20: PPO negative-advantage clipping (min 形式手算对拍) ---
def test_ppo_negative_advantage_clipping():
    # ρ=3, A=−1: min(ρA, clipA) = min(−3, −1.2) = −3 → L = +3
    L = T.ppo_loss_per_position(torch.tensor([math.log(3.0)]), torch.tensor([0.0]), torch.tensor([-1.0]))
    assert abs(L.item() - 3.0) < 1e-6
    # ρ=0.5, A=−1: min(−0.5, clip(0.5)=0.8×(−1)=−0.8) = −0.8 → L = +0.8 (惩罚以 clip 为限)
    L = T.ppo_loss_per_position(torch.tensor([math.log(0.5)]), torch.tensor([0.0]), torch.tensor([-1.0]))
    assert abs(L.item() - 0.8) < 1e-6
    # ρ=0.5, A=+1: min(0.5, 0.8) → L = −0.5
    L = T.ppo_loss_per_position(torch.tensor([math.log(0.5)]), torch.tensor([0.0]), torch.tensor([1.0]))
    assert abs(L.item() + 0.5) < 1e-6
    # ρ=3, A=+1: min(3, 1.2) → L = −1.2
    L = T.ppo_loss_per_position(torch.tensor([math.log(3.0)]), torch.tensor([0.0]), torch.tensor([1.0]))
    assert abs(L.item() + 1.2) < 1e-6

# --- §34.21: K=1 uniform timestep MC estimator (toy 解析对拍, 无 1/q 乘子) ---
# REINFORCE 口径: L = −A·logπ(a) (a detach); 梯度下降最小化 L ⇔ 最大化 E[R] (R 与 θ 无关)。
# 无偏性: E[∇L] = −∇_θ E[R] 当 A 与动作无关 (baseline 取动作无关常数, 这里 base=0)。
# 注: mean_G baseline 依赖组内采样 → 期望为 −(1−1/G)·∇E[R] (方向正确, 幅度缩放),
# 属 GRPO/PPO 实践的标准已知性质, 不在本 toy 断言的范围内。
# 均匀 timestep 采样 q=uniform → 无 1/q 乘子。
def test_k1_timestep_mc_estimator():
    theta = torch.tensor([0.2, -0.1], dtype=torch.float64)
    R = torch.tensor([[1.0, 0.3], [0.4, 0.8]], dtype=torch.float64)  # R[J, a], 不依赖 θ
    # 目标 G = 0.5·Σ_J Σ_a π_J(a)·R[J,a]
    def G(th):
        pi1 = torch.softmax(th, -1)
        pi2 = torch.softmax(0.5 * th, -1)
        return 0.5 * ((pi1 * R[0]).sum() + (pi2 * R[1]).sum())
    th = theta.clone().requires_grad_(True)
    G(th).backward()
    grad_true = th.grad.clone()
    # MC: 采样 J~Uniform(2), a~π_J (detached), 贡献 −R[J,a]·∇logπ_J(a) (base=0),
    # 均值 → −∇_θ E[R]
    N = 20000
    pi1 = torch.softmax(theta, -1)
    pi2 = torch.softmax(0.5 * theta, -1)
    g = torch.Generator().manual_seed(0)
    J = torch.randint(0, 2, (N,), generator=g)
    u = torch.rand(N, generator=g)
    a = torch.where(J == 0, (u >= pi1[0]).long(), (u >= pi2[0]).long())   # inverse-CDF: a=0 ⇔ u<π(0)
    A = torch.where(J == 0, R[0, a], R[1, a])   # base = 0 (动作无关)
    est = torch.zeros(2, dtype=torch.float64)
    for j in range(2):
        mask = J == j
        thj = (theta if j == 0 else 0.5 * theta).clone().requires_grad_(True)
        logpi = torch.log_softmax(thj, -1)
        L = (-A[mask] * logpi[a[mask]]).sum() / N
        L.backward()
        est += (0.5 ** j) * thj.grad    # 链式: th1 = θ, th2 = 0.5θ → ∇_θ = 0.5·∇_th2
    rel_err = ((est - (-grad_true)).abs() / grad_true.abs().clamp(min=1e-6)).max().item()
    assert rel_err < 0.15, f"MC est {est} vs -∇E[R] {-grad_true}, rel err {rel_err}"

# --- §34.22: sampler factorizes per stochastic position (无跨位置耦合) ---
def test_sampler_factorizes_per_position():
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    w1 = torch.rand(4, generator=g1) + 0.1
    w2 = torch.rand(4, generator=g2) + 0.1
    w = torch.stack([w1, w2])                        # [2, 4] 两位置独立权重
    pi1, pi2 = w1 / w1.sum(), w2 / w2.sum()
    N = 30000
    joint = torch.zeros(4, 4)
    for i in range(N):
        torch.manual_seed(20000 + i)
        a = T.sample_actions(w)
        joint[a[0].item(), a[1].item()] += 1
    joint /= N
    outer = pi1[:, None] * pi2[None, :]
    assert (joint - outer).abs().max().item() < 0.02
