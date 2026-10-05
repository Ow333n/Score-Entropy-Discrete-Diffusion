"""Mechanism pilot corruption policies（protocol v1.2 §3/§4）。

纯 CPU、无模型依赖。replicate 内四 policy 共享同一 schedule 五元组
(sample_id, span_start, span_len, σ, K)；policy 只改变 mask 位置选择。

- K_t ~ Binomial(m_t, q_t)，q_t = 1 - exp(-σ_t)（显式 torch.Generator 实现，禁全局 RNG）
- A：span 内均匀 K 子集（与原始 iid Bernoulli 掩码逐分布等价，corruption.py:33）
- B：固定全序列排列 π_L 的 span 内秩最小 K 个（nested prefix：K1<K2 ⇒ mask(K1)⊂mask(K2)）
- C：span 最右侧 K 个（≡ B 在 π_L=reverse 的特例）；D：span 最左侧 K 个（≡ B 在 π_L=identity）
- 本模块不触碰任何 frozen 代码路径（corruption.py / losses.py / graph_lib.py 零改动）。

流种子（§4.2）：data_order=1000+r, sigma=2000+r, span=3000+r, k=4000+r,
policy_a=5000+r, policy_b=6000+r, dropout=7000+r（cuda.manual_seed）。
"""
import hashlib

import torch

STREAM_SEEDS = dict(
    data_order=1000,
    sigma=2000,
    span=3000,
    k=4000,
    policy_a=5000,
    policy_b=6000,
    dropout=7000,
)

POLICIES = ("A", "B", "C", "D")


def stream_seed(name, replicate):
    return STREAM_SEEDS[name] + replicate


def make_generator(name, replicate, device="cpu"):
    return torch.Generator(device=device).manual_seed(stream_seed(name, replicate))


def draw_pi_L(generator, L):
    """B 的全序列固定排列（run 开头一次性抽取，全部样本共享，§3-B）。"""
    return torch.randperm(L, generator=generator)


def draw_sigma_t(n_items, generator, eps=1e-3, noise=None):
    """t = (1-eps)U[0,1]+eps → (σ, dσ)（LogLinearNoise 口径，noise_lib.py）。

    noise 传 LogLinearNoise() 实例时走其 total_noise/rate_noise，否则用等价的
    闭式实现（数值一致，ε=1e-3 下相对差 <1e-9）。
    """
    t = (1 - eps) * torch.rand(n_items, generator=generator) + eps
    if noise is not None:
        sigma, dsigma = noise(t)
    else:
        sigma = -torch.log1p(-(1 - eps) * t)
        dsigma = (1 - eps) / (1 - (1 - eps) * t)
    return sigma, dsigma, t


def draw_span(n_items, L, span_min, span_max, generator):
    """span_len ~ U[span_min, span_max]、span_start ~ U[0, L-span_len]（corruption.py:23-27 同口径）。"""
    span_len = torch.randint(span_min, span_max + 1, (n_items,), generator=generator).clamp(max=L)
    span_start = (torch.rand(n_items, generator=generator)
                  * (L - span_len + 1).float()).long()
    return span_len, span_start


def draw_K(span_len, sigma, generator, span_max):
    """K ~ Binomial(m, q)，q = 1 - exp(-σ)。

    显式 generator 的独立 Bernoulli 计数实现（torch.distributions 不接受 generator）；
    与原始 corruption.py:33 的 per-position iid Bernoulli 同分布。
    """
    n = span_len.shape[0]
    q = 1 - (-sigma).exp()
    moves = torch.rand(n, span_max, generator=generator) < q[:, None]
    valid = torch.arange(span_max)[None, :] < span_len[:, None]
    return (moves & valid).sum(dim=-1)


def select_mask(span_start, span_len, K, policy, L, pi_L=None, generator=None):
    """返回每 item 的绝对 mask 位置（list[tensor]）。

    A/B/C/D 语义见协议 v1.2 §3。policy 流消耗：A 每 item 消耗 randperm(m)；
    B 消耗 argsort（确定性，不消耗 RNG）；C/D 确定性。
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    if policy == "B" and pi_L is None:
        raise ValueError("policy B 需要 pi_L")
    masks = []
    for s0, m, k in zip(span_start.tolist(), span_len.tolist(), K.tolist()):
        k = int(k)
        pos = list(range(s0, s0 + m))
        if policy == "A":
            if k > 0:
                perm = torch.randperm(m, generator=generator).tolist()
                sel = [pos[i] for i in perm[:k]]
            else:
                sel = []
        elif policy == "B":
            ranks = torch.argsort(pi_L[s0:s0 + m])
            sel = [pos[i] for i in ranks.tolist()[:k]]
        elif policy == "C":
            sel = pos[m - k:]
        elif policy == "D":
            sel = pos[:k]
        masks.append(torch.tensor(sel, dtype=torch.long))
    return masks


class SharedSchedule:
    """replicate 内共享的 (σ, span, K) 流 + 增量 SHA-256 摘要（§4.1）。

    sample_id 流由 DataLoader 的 data_order generator 决定（vanilla.py build_dataloader
    同机制），dry-run 单独重放并汇总为完整五元组 digest。
    """

    def __init__(self, replicate, L, span_min, span_max, eps=1e-3, noise=None):
        self.r = replicate
        self.L = L
        self.span_min = span_min
        self.span_max = span_max
        self.eps = eps
        self.noise = noise
        self.g_sigma = make_generator("sigma", replicate)
        self.g_span = make_generator("span", replicate)
        self.g_k = make_generator("k", replicate)
        self._digest = hashlib.sha256()

    def draw(self, n_items):
        sigma, dsigma, _ = draw_sigma_t(n_items, self.g_sigma, self.eps, self.noise)
        span_len, span_start = draw_span(n_items, self.L, self.span_min,
                                         self.span_max, self.g_span)
        K = draw_K(span_len, sigma, self.g_k, self.span_max)
        for arr in (sigma, dsigma, span_len, span_start, K):
            self._digest.update(arr.numpy().tobytes())
        return sigma, dsigma, span_len, span_start, K

    def hexdigest(self):
        return self._digest.hexdigest()
