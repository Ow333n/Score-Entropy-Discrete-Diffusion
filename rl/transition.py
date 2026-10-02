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


def transition_weights(graph, x_t, score, dsigma):
    """analytic predictor 转移权重 w [B, L, D]。

    score: exp 后的真 score [B, L, D] (含 MASK 列, 其值已由 forward scatter 置 0)
    dsigma: [B] 或 [B, 1]
    """
    stag = graph.staggered_score(score, dsigma)          # p_{σ−dσ}/p_σ 近似
    trans = graph.transp_transition(x_t, dsigma)         # exact forward transition 行
    return stag * trans


def policy_log_probs(w):
    """logπ [B, L, D] = log(w) − logsumexp(log w)。不 clamp (w=0 → −inf, 由 gate 保证无真实负值)。"""
    logw = w.log()
    return logw - torch.logsumexp(logw, dim=-1, keepdim=True)


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
        n_nonfinite=int((~finite).sum()),
        n_neg=int(neg.sum()),
        rel_neg_mass=float(rel_neg_mass),
        min_mass=float(mass.min()),
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
