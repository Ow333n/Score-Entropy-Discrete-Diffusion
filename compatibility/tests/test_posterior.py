"""§4.2 归一化条件后验的 G0 测试: ① normalization ② MASK exclusion
③ time scalar cancellation ④ repeated deterministic evaluation consistency。"""
import torch

from compatibility.posterior import clean_log_probs, check_deterministic
from compatibility.tests.toy_models import D, MASK, TinyScoreModel


def test_normalization_sums_to_one():
    torch.manual_seed(0)
    score = torch.randn(2, 4, D, dtype=torch.float64)
    lp = clean_log_probs(score, D)
    assert lp.shape == (2, 4, D - 1)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, 4, dtype=torch.float64), atol=1e-9)


def test_mask_exclusion_adversarial():
    score = torch.zeros(1, 3, D, dtype=torch.float64)
    score[..., MASK] = 1e9                      # 对抗性 MASK 条目
    score[0, 1, 5] = 2.0                        # 位置 1 的 token 5 最高
    lp = clean_log_probs(score, D)
    assert lp[0, 1, 5] == lp[0, 1].max()        # softmax 保序
    assert lp.exp().sum(-1).allclose(torch.ones(1, 3, dtype=torch.float64), atol=1e-9)
    # 若 MASK 被误纳入 softmax, 干净词表概率全 ≈ 0, 上一条就过不了


def test_time_scalar_cancellation():
    """RADD 时间标量 log r(t) 是加性常数 → 对 p_hat 无影响 (§4.2-③)。"""
    torch.manual_seed(0)
    raw = torch.randn(2, 4, D, dtype=torch.float64)
    c = torch.tensor([3.7, -1.2])[:, None, None]   # 每序列一个常数 (如 log r(t))
    lp1 = clean_log_probs(raw, D)
    lp2 = clean_log_probs(raw - c, D)
    assert torch.allclose(lp1, lp2, atol=1e-12)


def test_repeated_deterministic_evaluation():
    """§4.2-④: eval 模式下同输入重复评估必须逐位一致。"""
    model = TinyScoreModel().eval()
    x = torch.randint(0, D, (2, 4))
    sigma = torch.tensor([1.0, 1.0])
    identical, max_abs_diff = check_deterministic(model, x, sigma)
    assert identical
    assert max_abs_diff == 0.0
