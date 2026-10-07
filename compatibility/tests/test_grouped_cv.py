"""Phase 2 grouped CV 泄漏检查（CPU，frozen Phase 2 协议 §5 修订 ③）。

- 结构性：每个 group 的全部行必须同 fold（check_no_leakage）
- 行为性：group-memorizing 模型（仅用 group 均值作特征）在正确 grouped CV 下
  必须表现差（held-out 组无法记忆）；若泄漏则表现异常好——证明 split 防泄漏有效
- 确定性：同 seed 两次 split 逐位一致
"""
import numpy as np

from scripts.phase2_grouped_cv import (check_no_leakage, grouped_cv, grouped_folds)


def test_no_leakage_structural():
    rows = [{"group_id": g, "x1": 0.0, "y": 0.0}
            for g in ["s%03d" % i for i in range(20)]
            for _ in range(5)]   # 20 组 × 5 行
    gids = [r["group_id"] for r in rows]
    folds, _ = grouped_folds(gids, n_folds=5, seed=0)
    ok, msg = check_no_leakage(folds, gids)
    assert ok, msg
    # 每 fold 覆盖的组数 > 0 且总和 = 20
    n_groups = len({g for g, f in zip(gids, folds)})
    assert n_groups == 20
    assert sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]


def test_no_leakage_behavioral():
    """group-mean 记忆模型：y 只由 group 决定（无共享特征信号）。
    正确 grouped CV 下 held-out 组无训练行 → 只能退化为全局均值 → MSE ≈ 组间方差（大）；
    若 split 泄漏（同组跨 fold），模型背下组均值 → MSE ≈ 0（小）。
    两个方向都断言，证明 split 防泄漏有效。"""
    rng = np.random.default_rng(7)
    n_groups, rows_per = 40, 4
    group_means = rng.normal(0, 5, n_groups)
    rows = []
    for gi in range(n_groups):
        for _ in range(rows_per):
            rows.append(dict(group_id=f"g{gi}", y=float(group_means[gi] + rng.normal(0, 0.1))))
    gids = np.array([r["group_id"] for r in rows])
    y = np.array([r["y"] for r in rows])

    def group_mean_memorizer(folds):
        se = []
        for f in range(5):
            tr, te = folds != f, folds == f
            means = {}
            for g, v in zip(gids[tr], y[tr]):
                means.setdefault(g, []).append(v)
            gmean = {g: float(np.mean(v)) for g, v in means.items()}
            global_mean = float(y[tr].mean())
            pred = np.array([gmean.get(g, global_mean) for g in gids[te]])
            se.append(float(((y[te] - pred) ** 2).mean()))
        return float(np.mean(se))

    folds_good, _ = grouped_folds(gids, 5, seed=0)
    mse_good = group_mean_memorizer(folds_good)
    assert mse_good > 1.0, f"正确 grouped CV 应防泄漏（MSE={mse_good:.3f}，应 ≈ 组间方差）"
    # 泄漏对照：故意随机按行 split（组跨 fold）→ 记忆模型应得近零 MSE
    rng2 = np.random.default_rng(3)
    folds_bad = rng2.integers(0, 5, len(rows))
    mse_bad = group_mean_memorizer(folds_bad)
    assert mse_bad < 0.1, f"泄漏 split 应被测试检出（MSE={mse_bad:.3f}，应 ≈ 0）"
    assert mse_good > 10 * mse_bad, f"grouped 与泄漏 split 应显著区分（{mse_good:.3f} vs {mse_bad:.3f}）"


def test_no_leakage_deterministic():
    gids = ["s%03d" % i for i in range(15) for _ in range(3)]
    f1, _ = grouped_folds(gids, 5, seed=0)
    f2, _ = grouped_folds(gids, 5, seed=0)
    assert np.array_equal(f1, f2)


def test_binary_metrics():
    from scripts.phase2_grouped_cv import binary_metrics
    y = np.array([0, 0, 1, 1])
    phat = np.array([0.1, 0.2, 0.9, 0.8])
    m = binary_metrics(y, phat)
    assert 0 < m["brier"] <= 1 and m["log_loss"] > 0
