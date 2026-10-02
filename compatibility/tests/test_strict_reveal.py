"""§4.8 严格逐 token 揭示的 G0 测试: teacher-forced 与 sampled 两种模式。

不变式 (§4.8): mask 数每步 -1、只有选中位置变化、选中前是 MASK (违反直接抛错)。
另测: 顺序策略、§3.8 tie-breaking、fixed path、温度、TF 路径概率手算值。"""
import math

import torch

import compatibility.order_gap as og
from compatibility.tests.toy_models import (ToyJointScoreFn, UniformScoreFn,
                                            LeftPeakScoreFn, TieScoreFn, D, MASK)


def test_tf_bookkeeping_and_path_logp():
    fn = UniformScoreFn()
    x = torch.full((1, 4), MASK, dtype=torch.long)
    x[0, 1] = 5                                   # 位置 1 已可见, 必须不被改动
    gold = torch.tensor([[3, 5, 7, 4]])
    x_out, info = og.strict_reveal(fn, x, gold, 1.0, D, order="l2r", mode="teacher_forced")
    assert info["steps"] == 3
    assert info["n_mask_history"].squeeze(1).tolist() == [3, 2, 1, 0]
    assert info["pos_history"].squeeze(1).tolist() == [0, 2, 3]
    # TF: 揭示 gold token, 不是采样 token
    assert info["tok_history"].squeeze(1).tolist() == [3, 7, 4]
    assert x_out[0].tolist() == [3, 5, 7, 4]
    # 均匀分布下每步 logp = log(1/10) → Q_pi = 3*log(1/10)
    assert abs(info["path_logp"].item() - 3 * math.log(1 / 10)) < 1e-9
    assert info["n_forwards"] == 3


def test_tf_invariant_violation_raises():
    fn = UniformScoreFn()
    x = torch.full((1, 4), MASK, dtype=torch.long)
    x[0, 1] = 5
    gold = torch.tensor([[3, 5, 7, 4]])
    bad_path = torch.tensor([[1], [0], [2]])     # 第一步选可见位置 1 → 违反不变式
    try:
        og.strict_reveal(fn, x, gold, 1.0, D, mode="teacher_forced", fixed_path=bad_path)
        raised = False
    except (AssertionError, RuntimeError):
        raised = True
    assert raised, "选中非 MASK 位置必须抛错 (§4.8)"


def test_sampled_bookkeeping():
    fn = UniformScoreFn()
    x = torch.full((1, 4), MASK, dtype=torch.long)
    x[0, 1] = 5
    x_out, info = og.strict_reveal(fn, x, None, 1.0, D, order="l2r", mode="sampled", sample="greedy")
    assert info["pos_history"].squeeze(1).tolist() == [0, 2, 3]
    # 均匀分布贪心 → token 0 (干净词表最小 index)
    assert info["tok_history"].squeeze(1).tolist() == [0, 0, 0]
    assert x_out[0, 1].item() == 5                 # 可见位置不动
    assert abs(info["path_logp"].item() - 3 * math.log(1 / 10)) < 1e-9


def test_orders():
    x = torch.full((1, 4), MASK, dtype=torch.long)
    gold = torch.randint(0, D - 1, (1, 4))

    _, i1 = og.strict_reveal(UniformScoreFn(), x, gold, 1.0, D, order="r2l", mode="teacher_forced")
    assert i1["pos_history"].squeeze(1).tolist() == [3, 2, 1, 0]

    _, i2 = og.strict_reveal(LeftPeakScoreFn(), x, gold, 1.0, D, order="confidence", mode="teacher_forced")
    assert i2["pos_history"].squeeze(1).tolist() == [0, 1, 2, 3]

    _, i3 = og.strict_reveal(LeftPeakScoreFn(), x, gold, 1.0, D, order="reverse-confidence", mode="teacher_forced")
    assert i3["pos_history"].squeeze(1).tolist() == [3, 2, 1, 0]


def test_tie_break_smallest_index():
    """§3.8: p_max 相同时选最小 index, 并记录 tie 计数。"""
    x = torch.full((1, 4), MASK, dtype=torch.long)
    gold = torch.randint(0, D - 1, (1, 4))
    _, info = og.strict_reveal(TieScoreFn(), x, gold, 1.0, D, order="confidence", mode="teacher_forced")
    # 全并列 → 每步选最小 index; ties = (当前 mask 数 - 1) 累计 = 3+2+1 = 6
    assert info["pos_history"].squeeze(1).tolist() == [0, 1, 2, 3]
    assert info["tie_count"] == 6
    assert info["tie_fraction"] == 6 / 4


def test_fixed_path():
    fn = UniformScoreFn()
    x = torch.full((1, 4), MASK, dtype=torch.long)
    gold = torch.tensor([[3, 5, 7, 4]])
    path = torch.tensor([[2], [0], [3], [1]])     # manifest 预生成路径 (§3.10)
    x_out, info = og.strict_reveal(fn, x, gold, 1.0, D, mode="teacher_forced", fixed_path=path)
    assert info["pos_history"].squeeze(1).tolist() == [2, 0, 3, 1]
    assert x_out[0].tolist() == [3, 5, 7, 4]


def test_temperature_categorical_in_clean_vocab():
    torch.manual_seed(42)
    x = torch.full((1, 5), MASK, dtype=torch.long)
    gold = torch.randint(0, D - 1, (1, 5))
    for T in (0.5, 1.0, 1.5):
        x_out, info = og.strict_reveal(UniformScoreFn(), x, gold, 1.0, D,
                                       order="random", mode="sampled",
                                       sample="categorical", temperature=T)
        assert info["steps"] == 5
        assert (x_out[0] < MASK).all()            # 采样 token 全在干净词表


def test_tf_path_logp_compatible_joint():
    """兼容玩具上两条路径 Q_pi 手算值相等 (OrderGap=0 的正确性基准)。"""
    P = [[0.2, 0.3], [0.4, 0.1]]
    x = torch.full((1, 2), MASK, dtype=torch.long)
    gold = torch.tensor([[3, 7]])

    _, info_ab = og.strict_reveal(ToyJointScoreFn(P), x, gold, 1.0, D,
                                  mode="teacher_forced", fixed_path=torch.tensor([[0], [1]]))
    _, info_ba = og.strict_reveal(ToyJointScoreFn(P), x, gold, 1.0, D,
                                  mode="teacher_forced", fixed_path=torch.tensor([[1], [0]]))
    # 路径 [0,1]: log p_i(3|C)=log 0.5, log p_j(7|i=3)=log 0.6 → Q = log 0.3
    assert abs(info_ab["path_logp"].item() - math.log(0.3)) < 1e-9
    # 路径 [1,0]: log p_j(7|C)=log 0.4, log p_i(3|j=7)=log 0.75 → Q = log 0.3
    assert abs(info_ba["path_logp"].item() - math.log(0.3)) < 1e-9
