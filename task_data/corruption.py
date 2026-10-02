"""Regime A span-infilling 腐蚀 (protocol §7.2)。

- span 外: x_t^context = x_0^context 恒可见 (始终是干净 token)
- span 内: partial absorbing 腐蚀, 每个位置以 1-e^{-sigma} 概率变 [MASK]
  (absorbing 转移的精确分布, 与 graph.sample_transition 一致)
- sigma: t ~ U[eps, 1] → loglinear schedule (训练分布)

同时返回 dsigma (loss 加权用) 与 t, 与 losses.py 的 DWDSE 积分加权口径一致。
"""
import torch

EPS = 1e-3


def corrupt_span_batch(x0, noise, span_min, span_max, mask_token, generator=None, eps=EPS):
    """x0: [B, L] → (x_t [B,L], span_mask [B,L], sigma [B], dsigma [B], t [B])"""
    B, L = x0.shape
    device = x0.device

    t = (1 - eps) * torch.rand(B, generator=generator, device=device) + eps
    sigma, dsigma = noise(t)                                    # loglinear schedule

    span_len = torch.randint(span_min, span_max + 1, (B,), generator=generator, device=device)
    span_len = span_len.clamp(max=L)
    # span_start ~ U[0, L - span_len], 保证 span 完整落在序列内
    span_start = (torch.rand(B, generator=generator, device=device)
                  * (L - span_len + 1).float()).long()

    x_t = x0.clone()
    span_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b in range(B):
        s0, ln = span_start[b].item(), span_len[b].item()
        move = torch.rand(ln, generator=generator, device=device) < (1 - torch.exp(-sigma[b]))
        x_t[b, s0:s0 + ln][move] = mask_token
        span_mask[b, s0:s0 + ln] = True

    return x_t, span_mask, sigma, dsigma, t


def masked_span_positions(x_t, span_mask, mask_token):
    """span 内当前为 MASK 的位置 mask (task loss 的支撑集, §7.3)。"""
    return span_mask & (x_t == mask_token)
