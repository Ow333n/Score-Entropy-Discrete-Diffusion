"""SEDD RL transition / policy 模块 (RL plan v0.2 §14, §14.1, §26)。

policy 口径 (analytic predictor, v0.2 审查定案为 primary):
    w = staggered_score(score, dsigma) * transp_transition(x_t, dsigma)   # [B, L, D]
    π(v|s) = w_v / Σ_u w_u
    logπ(v|s) = log(w_v) − logsumexp(log w)

同源约束: rollout 采样用 sample_categorical(w) (gumbel 技巧按 positive weights
proportional sampling, 无需显式归一化), logπ 用同一 w 计算。禁止两套分布。

validity gate (§14.1): 每 rollout step 检查 w 全 finite / Σw>0 / 负权重相对质量 ≤ tol=1e-6;
不静默 clamp。超 tolerance → abort batch (由调用方处理返回值)。

注意: score 入参是 **exp 后的真 score** (get_score_fn(sampling=True) 路径),
与 sampling.py::AnalyticPredictor 一致。
"""
import torch

from catsample import sample_categorical

NEG_TOL = 1e-6   # §14.1: numerical negative 放行 tolerance (相对质量)


def staggered_score_fn(score, dsigma):
    """Absorbing staggered_score 的纯函数版 (backward-safe)。

    与 graph_lib.Absorbing.staggered_score 数学等价:
      s'_v   = e^{dσ}·score_v                       (v 干净)
      s'_MASK = e^{dσ}·score_MASK + (1−e^{dσ})·Σ_u score_u
    原版用 in-place (score *= e; score[...,-1] += extra) —— 对带 grad_fn 的 tensor
    做 in-place 会产生 nan 梯度 (RL 是第一个在 staggered_score 上 backward 的路径);
    采样路径 no_grad 从未暴露。
    """
    e = dsigma.exp()
    mask_col = score[..., -1:] * e[..., None] + (1 - e[..., None]) * score.sum(dim=-1, keepdim=True)
    rest = score[..., :-1] * e[..., None]
    return torch.cat([rest, mask_col], dim=-1)


def transition_weights(graph, x_t, score, dsigma):
    """analytic predictor 转移权重 w [B, L, D]。

    score: exp 后的真 score [B, L, D] (含 MASK 列, 其值由 forward scatter 置 0)
    dsigma: [B] 或 [B, 1]
    """
    stag = staggered_score_fn(score, dsigma)            # p_{σ−dσ}/p_σ 近似 (backward-safe)
    trans = _transp_transition_f32(graph, x_t, dsigma)  # exact forward transition 行 (float32 one_hot, 显存减半)
    return stag * trans


def _transp_transition_f32(graph, i, sigma):
    """graph.transp_transition 的 float32 等价 (数值一致, one_hot int64 → float32 显存减半)。

    Absorbing 语义: 非 MASK 行 = e^{−σ}·one_hot(i); MASK 行 = 干净列 (1−e^{−σ}), MASK 列 1。
    """
    import torch.nn.functional as F
    from graph_lib import unsqueeze_as
    sigma = unsqueeze_as(sigma, i[..., None])
    edge = (-sigma).exp() * F.one_hot(i, num_classes=graph.dim).float()
    edge += torch.where(i == graph.dim - 1, 1 - (-sigma).squeeze(-1).exp(), 0)[..., None]
    return edge


def policy_log_probs(w):
    """logπ [B, L, D] = log(w/Σw) (线性归一化, 协议 §14)。

    backward-safety 关键: 直接用 π = w/Σw 构造 (0 权重处 π=0, ∂π/∂w=1/Σw 有限),
    再 log —— π 处处可微。而 log(w)−logsumexp(log w) 写法中 w.log() 独立节点的反向
    含 1/w, 在 w=0 的列产生 inf, 与 trans=0 相乘 → nan 梯度 (实测定位);
    torch.log_softmax 则是 softmax 归一化 (log e^w/Σe^w), 数学不对。
    w=0 处 logπ = log(tiny) ≈ −87 (数值上的 −inf 代表; 0 权重永不被采样, 无实际影响)。
    """
    pi = w / w.sum(dim=-1, keepdim=True)
    return pi.clamp_min(torch.finfo(w.dtype).tiny).log()


def sample_actions(w):
    """按 w proportional sampling 每位置采样动作 [B, L] (与 rollout sampler 同源)。

    使用全局 CPU RNG (catsample); 调用方负责在 rollout 前 manual_seed 并记录。
    """
    return sample_categorical(w)


def log_probs_for_actions(logpi, actions):
    """gather logπ [B, L, D] 在 actions [B, L] 处的值 → [B, L]。"""
    return torch.gather(logpi, -1, actions[..., None]).squeeze(-1)


def validity_check(w, tol=NEG_TOL):
    """§14.1 gate: 返回 (ok, stats)。不修改 w (不 clamp)。"""
    finite = torch.isfinite(w)
    mass = w.sum(dim=-1)
    neg = w < 0
    rel_neg_mass = 0.0
    if neg.any():
        neg_mass = (w * neg).abs().sum(dim=-1)
        rel_neg_mass = (neg_mass / mass.clamp(min=1e-12)).max().item()
    ok = bool(finite.all()) and bool((mass > 0).all()) and rel_neg_mass <= tol
    stats = dict(
        n_nonfinite=int((~finite).sum().detach()) if w.requires_grad else int((~finite).sum()),
        n_neg=int(neg.sum().detach()) if w.requires_grad else int(neg.sum()),
        rel_neg_mass=float(rel_neg_mass),
        min_mass=float(mass.min().detach()) if w.requires_grad else float(mass.min()),
    )
    return ok, stats


def mean_centered_advantage(rewards):
    """§22: A = r − mean_G(r)。rewards [P, G] → A [P, G]。std 仅 logging, 不除。"""
    return rewards - rewards.mean(dim=1, keepdim=True)


def ppo_loss_per_position(logp_new, logp_old, advantage, eps=0.2):
    """§26: L_ij = −min(ρ·A, clip(ρ,1±ε)·A)。per-position。

    logp_new / logp_old: [N] (同一 transition 的 new/old logπ, 广播前)
    advantage: [N] (trajectory-level advantage 广播到位置)
    返回 [N] per-position loss (调用方求和)。
    """
    ratio = (logp_new - logp_old).exp()
    clipped = ratio.clamp(1 - eps, 1 + eps)
    return -torch.minimum(ratio * advantage, clipped * advantage)
