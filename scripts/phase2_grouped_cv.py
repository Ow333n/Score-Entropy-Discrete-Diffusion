"""Phase 2 grouped 5-fold CV（frozen Phase 2 协议 §5，FROZEN 修订 ③）。

- group key = 底层 evaluation sample / masked example ID；同一原始 sample 派生的
  全部 rows（tokens / block sizes / paths / scheduler observations）必须同 fold；
  禁止跨 train/test fold 泄漏
- 分 endpoint 评价：
  token recovery（回归）：held-out MSE / R²；增量 = ΔMSE / ΔR²
  exact-span（二元）：held-out log-loss / Brier；增量 = Δlog-loss / ΔBrier；
  AUROC 仅 secondary；**binary 禁用普通 R² 作主指标**
- 模型：OLS（lstsq，回归）+ 逻辑回归（Newton 迭代，二元）——纯 numpy 实现，无外部依赖
- 全部确定性（seed 冻结）
"""
import numpy as np


def grouped_folds(group_ids, n_folds=5, seed=0):
    """按 group 聚类分 fold：组打乱后连续切分；返回每行的 fold id（0..n_folds-1）。"""
    rng = np.random.default_rng(seed)
    groups = np.array(list(dict.fromkeys(group_ids)))   # 保持首次出现顺序
    rng.shuffle(groups)
    folds = np.empty(len(group_ids), dtype=int)
    for f, g in enumerate(np.array_split(groups, n_folds)):
        folds[np.isin(group_ids, g)] = f
    return folds, groups


def check_no_leakage(folds, group_ids):
    """结构性泄漏检查：每个 group 的全部行必须落在同一个 fold。"""
    seen = {}
    for g, f in zip(group_ids, folds):
        if g in seen and seen[g] != f:
            return False, f"group {g} 跨 fold（{seen[g]} vs {f}）"
        seen[g] = f
    return True, None


def ols_fit_predict(X_tr, y_tr, X_te):
    """OLS（含截距）→ held-out 预测。"""
    A = np.hstack([np.ones((len(X_tr), 1)), X_tr])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return np.hstack([np.ones((len(X_te), 1)), X_te]) @ coef


def logistic_fit_predict(X_tr, y_tr, X_te, iters=200, lr=0.1):
    """逻辑回归（Newton 式 IRLS 简化：梯度下降 + 冻结 iter/lr）→ held-out 概率。"""
    A_tr = np.hstack([np.ones((len(X_tr), 1)), X_tr])
    w = np.zeros(A_tr.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-A_tr @ w))
        grad = A_tr.T @ (p - y_tr) / len(y_tr)
        w -= lr * grad
    A_te = np.hstack([np.ones((len(X_te), 1)), X_te])
    return 1.0 / (1.0 + np.exp(-A_te @ w))


def regression_metrics(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(mse=ss_res / len(y), r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def binary_metrics(y, phat, eps=1e-12):
    p = np.clip(phat, eps, 1 - eps)
    ll = -float((y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    return dict(log_loss=ll, brier=brier)


def grouped_cv(rows, X_names, y_name, binary=False, n_folds=5, seed=0):
    """grouped 5-fold CV：返回 per-fold 指标 + 汇总。

    rows: list[dict]，含 group_id / X_names 列 / y_name 列。
    """
    gids = np.array([r["group_id"] for r in rows])
    X = np.array([[r[n] for n in X_names] for r in rows], dtype=float)
    y = np.array([r[y_name] for r in rows], dtype=float)
    folds, _ = grouped_folds(gids, n_folds, seed)
    ok, msg = check_no_leakage(folds, gids)
    assert ok, msg
    out = []
    for f in range(n_folds):
        tr = folds != f
        te = folds == f
        if binary:
            phat = logistic_fit_predict(X[tr], y[tr], X[te])
            m = binary_metrics(y[te], phat)
            m["auroc"] = auroc(y[te], phat)
        else:
            yhat = ols_fit_predict(X[tr], y[tr], X[te])
            m = regression_metrics(y[te], yhat)
        out.append(dict(fold=f, n_te=int(te.sum()), **m))
    return dict(folds=folds.tolist(), per_fold=out,
                pooled={k: float(np.mean([o[k] for o in out])) for k in out[0] if k != "fold"})


def auroc(y, phat):
    """AUROC（secondary 用）。"""
    order = np.argsort(phat)[::-1]
    y_sorted = y[order]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    return float((tp - y_sorted).sum() / (n_pos * n_neg))
