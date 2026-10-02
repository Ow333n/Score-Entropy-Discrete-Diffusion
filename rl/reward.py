"""RL reward (plan v0.2 §18): M0 口径的 token 重建准确率。纯函数, no_grad, 不接触模型。"""
import torch


def m0_reward(x0, x_gen, m0):
    """R = |{i ∈ M0 : x̂_i == x_i}| / |M0|。

    x0 / x_gen: [B, L] long; m0: [B, L] bool (rollout 初始时 span 内实际 MASK 的位置)
    返回 [B] ∈ [0, 1]。|M0|=0 的行返回 0.0 (调用方负责按 §18 重采样并记录)。
    """
    with torch.no_grad():
        correct = ((x_gen == x0) & m0).sum(dim=-1).float()
        return correct / m0.sum(dim=-1).clamp(min=1).float()
