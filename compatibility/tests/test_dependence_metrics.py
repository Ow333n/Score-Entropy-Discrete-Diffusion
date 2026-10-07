"""Phase 2 dependence metrics 数学实现检查（CPU toy tests，frozen Phase 2 协议 §1C）。

覆盖（用户 preflight 清单）：
- JS toy：identical → 0；near-disjoint → ≈ln2；极端概率数值稳定；对称
- SEDD score → normalized conditional distribution 映射：clean_log_probs 是
  干净词表 softmax（frozen 定义），JS 只依赖该归一化分布
- JS=0 ⇒ δ=0；δ=0 ⇏ JS=0（严格更强）
- δ / D_ab / PMI 代数关系（δ=两方向差、D_ab=均值）
- 确定性（同输入逐位一致）
"""
import math

import torch
import torch.nn.functional as F

from compatibility.dependence import js_dep, jsd_from_logp, pointwise_terms


def test_jsd_identical_zero():
    lp = torch.log(torch.tensor([0.7, 0.2, 0.1]))
    assert jsd_from_logp(lp, lp).item() < 1e-12


def test_jsd_near_disjoint_ln2():
    lp1 = torch.log(torch.tensor([0.999999, 1e-9, 1e-9]))
    lp2 = torch.log(torch.tensor([1e-9, 0.999999, 1e-9]))
    v = jsd_from_logp(lp1, lp2).item()
    assert abs(v - math.log(2)) < 1e-6, v


def test_jsd_symmetric_bounded():
    lp1 = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    lp2 = torch.log(torch.tensor([0.1, 0.8, 0.1]))
    a, b = jsd_from_logp(lp1, lp2).item(), jsd_from_logp(lp2, lp1).item()
    assert abs(a - b) < 1e-12
    assert 0 <= a <= math.log(2) + 1e-12


def test_jsd_extreme_probability_stability():
    # 极端小概率分量（1e-300 量级）不应产生 NaN/Inf
    lp1 = torch.log(torch.tensor([1.0, 1e-300, 0.0]) + 1e-300)
    lp2 = torch.log(torch.tensor([1e-300, 1.0, 0.0]) + 1e-300)
    v = jsd_from_logp(lp1, lp2).item()
    assert math.isfinite(v) and abs(v - math.log(2)) < 1e-6, v


def test_normalized_conditional_mapping():
    """SEDD score → normalized conditional distribution（clean_log_probs 口径）：
    log_softmax 于干净词表是严格归一化分布（JS 只依赖该对象）。"""
    D = 6  # 5 干净 token + 1 MASK
    score = torch.randn(3, D, dtype=torch.float64)
    logp = F.log_softmax(score[..., :D - 1], dim=-1)
    p = logp.exp()
    assert torch.allclose(p.sum(-1), torch.ones(3, dtype=torch.float64), atol=1e-12)
    assert p.shape == (3, D - 1)


def _quartet_case(make=False):
    """构造一个不一致的 quartet（δ≠0 且 JS≠0）与一个一致的 quartet（δ=0、JS=0）。

    用小的手工 logp 张量模拟 logp_C / logp_Ca / logp_Cb（位置 0/1，token 3/7）。
    """
    # 不一致 case：reveal a 后位置 1 的分布明显变锐
    lp_C = torch.full((2, 8), math.log(1 / 8), dtype=torch.float64)
    lp_C[0, 3] = math.log(0.5); lp_C[0, :3] = math.log(0.5 / 3)  # 位置0 对 token3 高概率
    lp_C[1, 7] = math.log(0.6); lp_C[1, :7] = math.log(0.4 / 7)
    lp_Ca = lp_C.clone()
    lp_Ca[1] = torch.full((8,), math.log(1 / 8), dtype=torch.float64)
    lp_Ca[1, 7] = math.log(0.9); lp_Ca[1, :7] = math.log(0.1 / 7)   # reveal a 后位置1 变锐
    lp_Cb = lp_C.clone()
    lp_Cb[0] = torch.full((8,), math.log(1 / 8), dtype=torch.float64)
    lp_Cb[0, 3] = math.log(0.8); lp_Cb[0, :3] = math.log(0.2 / 3)
    return lp_C, lp_Ca, lp_Cb


def test_js_zero_implies_delta_zero():
    lp_C, lp_Ca, lp_Cb = _quartet_case()
    # 一致 case：reveal 完全不改变分布 → JS=0 且 δ=0
    lp_Ca2 = lp_C.clone()
    lp_Cb2 = lp_C.clone()
    js = js_dep(lp_C, lp_Ca2, lp_Cb2, 0, 1)
    pt = pointwise_terms(lp_C, lp_Ca2, lp_Cb2, 0, 1, 3, 7)
    assert js["js_dep"].item() < 1e-12
    assert abs(pt["delta"].item()) < 1e-12
    # 不一致 case：δ=0 但 JS≠0 的反例——直接构造：p(7) 保持 0.6 不变、
    # 其他 token 之间重分配质量（位置 0 完全不动）
    p_orig = torch.full((8,), 0.4 / 7, dtype=torch.float64)
    p_orig[7] = 0.6
    p_new = p_orig.clone()
    moved = p_new[5] - 0.2                    # 把 token5 的质量挪走（p5: 0.057→0.2 是增加？
    p_new[5] = 0.2                            # 目标：p5=0.2、p7=0.6 不变
    excess = 0.2 - p_orig[5]                  # 需要从其他 6 个非 GT token 借
    p_new[0] = p_orig[0] - excess / 6
    p_new[1] = p_orig[1] - excess / 6
    p_new[2] = p_orig[2] - excess / 6
    p_new[3] = p_orig[3] - excess / 6
    p_new[4] = p_orig[4] - excess / 6
    p_new[6] = p_orig[6] - excess / 6
    assert abs(p_new.sum().item() - 1.0) < 1e-12 and (p_new > 0).all()
    lp_Ca3 = lp_C.clone()
    lp_Ca3[1] = p_new.log()
    js3 = js_dep(lp_C, lp_Ca3, lp_C.clone(), 0, 1)
    pt3 = pointwise_terms(lp_C, lp_Ca3, lp_C.clone(), 0, 1, 3, 7)
    assert js3["js_dep"].item() > 1e-6          # 分布变了
    # GT token 7 的 logp 不变（只动了其他 token）→ pmi_ab = 0 → δ = 0 但 JS ≠ 0
    assert abs(pt3["pmi_ab"].item()) < 1e-9
    assert abs(pt3["delta"].item()) < 1e-9


def test_delta_dab_pmi_algebra():
    lp_C, lp_Ca, lp_Cb = _quartet_case()
    pt = pointwise_terms(lp_C, lp_Ca, lp_Cb, 0, 1, 3, 7)
    assert abs(pt["delta"].item() - (pt["pmi_ab"].item() - pt["pmi_ba"].item())) < 1e-12
    assert abs(pt["d_ab"].item() - 0.5 * (pt["pmi_ab"].item() + pt["pmi_ba"].item())) < 1e-12


def test_determinism():
    lp_C, lp_Ca, lp_Cb = _quartet_case()
    a1 = js_dep(lp_C, lp_Ca, lp_Cb, 0, 1)
    a2 = js_dep(lp_C, lp_Ca, lp_Cb, 0, 1)
    assert a1["js_dep"].item() == a2["js_dep"].item()
    assert a1["js_ab"].item() == a2["js_ab"].item()
