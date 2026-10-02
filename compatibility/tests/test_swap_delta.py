"""§4.5/§4.6 swap defect 的 G0 测试: 兼容玩具 → δ=0; 不兼容玩具 → 手算值;
batch 化 = 逐样本; forward 次数 = 3; 批量评估器与单样本 delta_swap 一致。"""
import math

import torch

import compatibility.cpi as cpi
from compatibility.tests.toy_models import ToyJointScoreFn, IncompatibleScoreFn, D, MASK

P = [[0.2, 0.3], [0.4, 0.1]]   # 行=pos0 的 token, 列=pos1 的 token; 边缘 p0=(0.5,0.5), p1=(0.6,0.4)


def test_compatible_joint_delta_zero():
    fn = ToyJointScoreFn(P)
    x_t = torch.full((1, 2), MASK, dtype=torch.long)
    sigma = torch.tensor([1.0])
    for a, b in [(3, 3), (3, 7), (7, 3), (7, 7)]:
        r = cpi.delta_swap(fn, x_t, torch.tensor([0]), torch.tensor([1]),
                           torch.tensor([a]), torch.tensor([b]), sigma, D)
        assert r["n_forwards"] == 3
        assert abs(r["delta"].item()) < 1e-9, f"(a,b)=({a},{b}) delta={r['delta'].item()}"


def test_incompatible_hand_value():
    fn = IncompatibleScoreFn()
    x_t = torch.full((1, 2), MASK, dtype=torch.long)
    r = cpi.delta_swap(fn, x_t, torch.tensor([0]), torch.tensor([1]),
                       torch.tensor([3]), torch.tensor([7]), torch.tensor([1.0]), D)
    expected = math.log(3.2)
    assert abs(r["delta"].item() - expected) < 1e-9
    assert abs(r["delta"].abs().item() - expected) < 1e-9   # CPI = |delta|


def test_batched_matches_single():
    fn = ToyJointScoreFn(P)
    x_t = torch.full((2, 2), MASK, dtype=torch.long)
    sigma = torch.tensor([1.0, 2.0])
    a = torch.tensor([3, 7]); b = torch.tensor([7, 3])
    r = cpi.delta_swap(fn, x_t, torch.tensor([0, 0]), torch.tensor([1, 1]), a, b, sigma, D)
    for k in range(2):
        r1 = cpi.delta_swap(fn, x_t[k:k + 1], torch.tensor([0]), torch.tensor([1]),
                            a[k:k + 1], b[k:k + 1], sigma[k:k + 1], D)
        assert abs(r["delta"][k].item() - r1["delta"].item()) < 1e-12


def test_evaluate_batch_consistent():
    fn = ToyJointScoreFn(P)
    samples = []
    for a, b in [(3, 3), (3, 7)]:
        x_t = torch.full((2,), MASK, dtype=torch.long)
        x0 = torch.tensor([a, b])
        samples.append(dict(x_t=x_t, i=0, j=1, a=a, b=b, x0=x0, sigma=1.0))
    res = cpi.evaluate_delta_swap_batch(fn, samples, D, chunk=1)
    assert res["delta"].numel() == 2
    assert res["delta"].abs().max().item() < 1e-9
    # local CE: 兼容玩具上每个 mask 位置的 CE 手算对照
    lp_C = fn(samples[0]["x_t"][None], torch.tensor([1.0]))
    expected_ce = -(lp_C[0, 0, 3] + lp_C[0, 1, 3]) / 2
    assert abs(res["local_ce"][0].item() - expected_ce.item()) < 1e-9
    # token acc: 兼容玩具上 argmax = 正确 token
    assert res["token_acc"][0].item() == 1.0


def test_summarize_rms_quantiles():
    delta = torch.tensor([0.1, -0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
    n = 5
    results = dict(delta=delta,
                   local_ce=torch.ones(n),
                   token_acc=torch.ones(n),
                   n_masked=torch.ones(n, dtype=torch.long),
                   sigma=torch.ones(n))
    s = cpi.summarize(results, name="toy")
    assert abs(s["cpi_abs"] - delta.abs().mean().item()) < 1e-6
    assert abs(s["cpi_rms"] - (delta ** 2).mean().sqrt().item()) < 1e-6
    assert abs(s["delta_median"] - delta.median().item()) < 1e-6
    assert abs(s["delta_abs_p90"] - torch.quantile(delta.abs().float(), 0.90).item()) < 1e-5
    assert abs(s["delta_abs_p99"] - torch.quantile(delta.abs().float(), 0.99).item()) < 1e-5
