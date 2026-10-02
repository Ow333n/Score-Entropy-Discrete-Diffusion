"""G0 单元测试用的玩具 score 模型 (全部 CPU, 无真实权重)。

MASK 条目故意置 1e9 —— 若提取代码误把 MASK 纳入 softmax, 所有 p_hat 都会错,
对应测试必然失败 (对抗性验证 §4.2-②)。
"""
import math

import torch
import torch.nn as nn

D = 11          # 10 个干净 token + MASK = 10
MASK = D - 1
TOK = (3, 7)    # 玩具分布的两个 token


class ToyJointScoreFn:
    """从 2 位置联合分布 P(a,b) 精确导出的条件分布作为 score 输出。

    位置 0/1, token 集合 {3,7}。状态 C(双 mask) 返回边缘分布,
    状态 C+a / C+b 返回对应条件分布。
    """
    def __init__(self, P):
        self.P = torch.tensor(P, dtype=torch.float64)
        self.n_calls = 0

    def __call__(self, x_t, sigma):
        self.n_calls += 1
        B, L = x_t.shape
        out = torch.full((B, L, D), -1e9, dtype=torch.float64)
        out[..., MASK] = 1e9
        t0, t1 = TOK
        for b in range(B):
            v0, v1 = x_t[b, 0].item(), x_t[b, 1].item()
            m0, m1 = v0 == MASK, v1 == MASK
            P = self.P
            if m0:
                if m1:  # 状态 C: 两位置都是边缘分布
                    out[b, 0, t0], out[b, 0, t1] = (P.sum(1)[0].log(), P.sum(1)[1].log())
                    out[b, 1, t0], out[b, 1, t1] = (P.sum(0)[0].log(), P.sum(0)[1].log())
                else:   # 状态 C+b: i 的条件分布 p(i | j=v1)
                    col = P[:, TOK.index(v1)]
                    p = col / col.sum()
                    out[b, 0, t0], out[b, 0, t1] = p[0].log(), p[1].log()
            elif m1:    # 状态 C+a: j 的条件分布 p(j | i=v0)
                row = P[TOK.index(v0)]
                p = row / row.sum()
                out[b, 1, t0], out[b, 1, t1] = p[0].log(), p[1].log()
        return out


class IncompatibleScoreFn:
    """故意不兼容的条件族: p_i(3)=0.6, p_j(7)=0.5, p_j(7|i=3)=0.8, p_i(3|j=7)=0.3。

    对 (a,b)=(3,7): delta = ln(0.6*0.8/(0.5*0.3)) = ln(3.2) ≈ 1.1631508
    """
    def __init__(self):
        self.n_calls = 0

    def __call__(self, x_t, sigma):
        self.n_calls += 1
        B, L = x_t.shape
        out = torch.full((B, L, D), -1e9, dtype=torch.float64)
        out[..., MASK] = 1e9
        t0, t1 = TOK
        for b in range(B):
            v0, v1 = x_t[b, 0].item(), x_t[b, 1].item()
            m0, m1 = v0 == MASK, v1 == MASK
            if m0 and m1:                       # 状态 C
                out[b, 0, t0], out[b, 0, t1] = math.log(0.6), math.log(0.4)
                out[b, 1, t0], out[b, 1, t1] = math.log(0.5), math.log(0.5)
            elif m1 and v0 == t0:               # 状态 C+a (a=3)
                out[b, 1, t0], out[b, 1, t1] = math.log(0.2), math.log(0.8)
            elif m0 and v1 == t1:               # 状态 C+b (b=7)
                out[b, 0, t0], out[b, 0, t1] = math.log(0.3), math.log(0.7)
        return out


class UniformScoreFn:
    """干净词表上均匀分布 (贪心解码会选 token 0)。"""
    def __init__(self):
        self.n_calls = 0

    def __call__(self, x_t, sigma):
        self.n_calls += 1
        B, L = x_t.shape
        out = torch.zeros(B, L, D, dtype=torch.float64)
        out[..., MASK] = -1e9
        return out


class LeftPeakScoreFn:
    """p_max 随位置递减: 最左 mask 位置置信度最高 (测 confidence 顺序)。"""
    def __init__(self):
        self.n_calls = 0

    def __call__(self, x_t, sigma):
        self.n_calls += 1
        B, L = x_t.shape
        out = torch.zeros(B, L, D, dtype=torch.float64)
        out[..., MASK] = -1e9
        for k in range(L):
            out[:, k, 0] = L - k
        return out


class TieScoreFn:
    """所有 mask 位置 p_max 完全相同 (测 §3.8 tie-breaking 与 tie 计数)。"""
    def __init__(self):
        self.n_calls = 0

    def __call__(self, x_t, sigma):
        self.n_calls += 1
        B, L = x_t.shape
        out = torch.zeros(B, L, D, dtype=torch.float64)
        out[..., MASK] = -1e9
        return out


class TinyScoreModel(nn.Module):
    """protocol §4.9 规格的 tiny model: vocab=10, hidden=8, seq=4, batch=2, dropout=0。"""
    def __init__(self, D=11, hidden=8):
        super().__init__()
        self.embed = nn.Embedding(D, hidden)
        self.out = nn.Linear(hidden, D)

    def forward(self, x, sigma):
        return self.out(self.embed(x))
