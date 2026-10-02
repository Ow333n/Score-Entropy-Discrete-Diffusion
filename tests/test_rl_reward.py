"""RL reward / corruption 单测 (plan v0.2 §34: 12, 18, 19)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import noise_lib
from rl import reward as R
from task_data import corruption

MASK_TOKEN = 50257


# --- §34.12: reward 无梯度路径 ---
def test_reward_no_grad():
    x0 = torch.tensor([[3, 5, 7, 9]])
    x_gen = torch.tensor([[3, 4, 8, 9]])
    m0 = torch.tensor([[True, True, False, True]])
    x0f = x0.float().clone().requires_grad_(True)
    r = R.m0_reward(x0f, x_gen, m0)
    assert not r.requires_grad
    assert r.grad_fn is None
    assert x0f.grad is None

# --- §34.19: reward 只计算 M0 ---
def test_reward_m0_only():
    x0 = torch.tensor([[10, 20, 30, 40, 50]])
    x_gen = torch.tensor([[10, 99, 30, 40, 50]])   # 位置 1 错 (M0), 位置 2 错 (非 M0)
    m0 = torch.tensor([[True, True, False, True, False]])
    r = R.m0_reward(x0, x_gen, m0)
    assert abs(r.item() - 2.0 / 3.0) < 1e-6       # M0 = {0,1,3}, 只 0,3 对 (fp32)

def test_reward_m0_empty():
    x0 = torch.tensor([[10, 20]])
    x_gen = torch.tensor([[10, 99]])
    m0 = torch.zeros(1, 2, dtype=torch.bool)
    r = R.m0_reward(x0, x_gen, m0)
    assert r.item() == 0.0                        # |M0|=0 → 0.0 (调用方负责重采样)

# --- §34.18: G rollouts 共享初始腐蚀 (同 seed → 逐位一致; 仅采样 RNG 可变) ---
def test_corruption_shared_across_group():
    noise = noise_lib.LogLinearNoise()
    torch.manual_seed(42)
    x0 = torch.randint(0, MASK_TOKEN, (2, 32))
    def corrupt(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        x_t, span_mask, sigma, dsigma, t = corruption.corrupt_span_batch(
            x0, noise, 10, 50, MASK_TOKEN, generator=g)
        return x_t, span_mask, sigma, (span_mask & (x_t == MASK_TOKEN))
    x1, m1, s1, m01 = corrupt(1234)
    x2, m2, s2, m02 = corrupt(1234)
    assert (x1 == x2).all() and (m1 == m2).all() and (s1 == s2).all()
    assert (m01 == m02).all()                     # M0 一致 → G 共享初始腐蚀
