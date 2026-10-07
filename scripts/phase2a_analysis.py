"""Phase 2A trajectory pair diagnostics 分析（只读；用户 2026-10-07 执行纪律）。

派生量（per-sample，post-hoc，不改任何 evaluator）：
- PMI_forward = D_ab + δ/2；PMI_reverse = D_ab − δ/2（代数恒等：δ = PMI_fwd − PMI_rev，
  故 "PMI 收敛" ⟺ |δ| 下降；PMI_fwd/reverse 的个体轨迹提供新信息）
- A_sym = |δ| / (|PMI_fwd| + |PMI_rev| + 1e-9)——**exploratory only**（相对方向
  不对称程度），不进入 frozen protocol、不包装成正式 metric

核心判别（Case A/B/C，用户 Phase 2A 逻辑）：
- A：|δ| 基本跟 CE 同步下降，JS/D_ab/PMI 无独立 temporal signature
- B：D_ab 保持/上升 + JS_dep 保持/上升 + |δ|↓ + PMI gap↓，主变在 500→1020 段、
  1020 后 plateau → symmetrization candidate（最高优先级）
- C：无结构信号 → mechanism-screening stop

输出：
- per-checkpoint 汇总（mean/median/bootstrap CI：|δ|、signed δ、JS_dep、D_ab、
  PMI_fwd/reverse、CE、A_sym）
- per-run trajectory + 窗口分段（v1.2/v2.1: 500→1020、1020→2500；formal:
  1020→5100、5100→10200；RL: 50→250、250→500；p1: 仅 2500）
- 跨 run 方向一致性（每 metric 每窗口同号比例）
- checkpoint-level correlation + 控制 CE 的 rank-partial 关联
- grouped CV（secondary；group=underlying sample；M0 CE / M1 CE+JS / M2 CE+D_ab /
  M3 CE+JS+D_ab → |δ|；CV 增益≠因果，明示）
- Case A/B/C 判定 + 是否值得 Phase 2B

用法: .venv/bin/python scripts/phase2a_analysis.py
输出: results/phase2/phase2a_analysis.json
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(ROOT, "results", "phase1_diag")
PAIRS = os.path.join(ROOT, "results", "phase2", "pairs")
BM_SHA = "3b970159ecc6818beeefd2250d141ae54c7ea309a0dbb85e84415abe9c7bcc6c"
EPS = 1e-9
N_BOOT = 10000
BOOT_SEED = 0


def j(path):
    with open(path) as f:
        return json.load(f)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def boot_ci(x, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rank_partial(a, b, c):
    """控制 c 后的 rank 偏相关（rank 回归残差的 Spearman）。"""
    def resid(x, z):
        r = np.argsort(np.argsort(x)).astype(float)
        r -= r.mean()
        rz = np.argsort(np.argsort(z)).astype(float)
        rz -= rz.mean()
        beta = (rz * r).sum() / (rz ** 2).sum()
        return r - beta * rz
    return spearman(resid(a, c), resid(b, c))


def main():
    man = j(os.path.join(DIAG, "bulk_manifest.json"))
    digest = hashlib.sha256(open(os.path.join(DIAG, "bulk_manifest.json"), "rb").read()).hexdigest()
    assert digest == BM_SHA
    cks = man["checkpoints"]
    ids = [c["id"] for c in cks]
    fam = {c["id"]: c["family"] for c in cks}
    run = {c["id"]: c["run"] for c in cks}
    step = {c["id"]: c["step"] for c in cks}

    pairs = {cid: j(os.path.join(PAIRS, f"{cid}.json")) for cid in ids}

    # ---- per-checkpoint 汇总（+派生量）----
    table = []
    for cid in ids:
        ps = pairs[cid]["per_sample"]
        d = np.array(ps["delta"])
        dab = np.array(ps["d_ab"])
        pmi_f = dab + d / 2
        pmi_r = dab - d / 2
        asym = np.abs(d) / (np.abs(pmi_f) + np.abs(pmi_r) + EPS)
        ce = np.array(ps["local_ce"])
        js = np.array(ps["js_dep"])
        ad = np.abs(d)
        table.append(dict(
            id=cid, family=fam[cid], run=run[cid], step=step[cid],
            delta_abs=dict(mean=ad.mean(), median=np.median(ad), ci=list(boot_ci(ad))),
            delta_mean=dict(mean=d.mean(), ci=list(boot_ci(d))),
            js_dep=dict(mean=js.mean(), median=np.median(js), ci=list(boot_ci(js))),
            d_ab=dict(mean=dab.mean(), median=np.median(dab), ci=list(boot_ci(dab))),
            pmi_forward=dict(mean=pmi_f.mean(), ci=list(boot_ci(pmi_f))),
            pmi_reverse=dict(mean=pmi_r.mean(), ci=list(boot_ci(pmi_r))),
            ce=dict(mean=ce.mean(), ci=list(boot_ci(ce))),
            a_sym=dict(mean=asym.mean(), median=np.median(asym)),
            a_sym_exploratory=True,
        ))

    # ---- per-run trajectory + 窗口分段 ----
    # 分段映射：v1.2/v2.1 (500→1020, 1020→2500)；formal (1020→5100, 5100→10200)；
    # RL (50→250, 250→500)；p1 仅 2500（无段）
    seg_map = {(500, 1020): "500→1020", (1020, 2500): "1020→2500",
               (1020, 5100): "1020→5100", (5100, 10200): "5100→10200",
               (50, 250): "50→250", (250, 500): "250→500"}
    nxt_of = {("v1.2-AB", 500): 1020, ("v1.2-CD", 500): 1020, ("v2.1-HUL", 500): 1020,
              ("v1.2-AB", 1020): 2500, ("v1.2-CD", 1020): 2500, ("v2.1-HUL", 1020): 2500,
              ("formal", 1020): 5100, ("formal", 5100): 10200,
              ("RL", 50): 250, ("RL", 250): 500}
    segs = []
    for cid in ids:
        st = step[cid]
        nxt = nxt_of.get((fam[cid], st))
        if nxt is None:
            continue
        cand = [t for t in table if t["run"] == run[cid] and t["step"] == nxt]
        if not cand:
            continue
        t2 = cand[0]
        row = dict(run=run[cid], family=fam[cid], seg=seg_map[(st, nxt)],
                   from_step=st, to_step=nxt)
        for k in ("delta_abs", "js_dep", "d_ab", "pmi_forward", "pmi_reverse",
                  "ce", "a_sym"):
            row[f"d_{k}"] = t2[k]["mean"] - \
                next(t[k]["mean"] for t in table if t["id"] == cid)
        segs.append(row)

    # 跨 run 方向一致性（每 seg × metric 同号比例）
    consistency = {}
    for seg_name in sorted({s["seg"] for s in segs}):
        rows = [s for s in segs if s["seg"] == seg_name]
        consistency[seg_name] = dict(
            n_runs=len(rows),
            delta_abs=float(np.mean([np.sign(r["d_delta_abs"]) < 0 for r in rows])),
            js_dep=float(np.mean([np.sign(r["d_js_dep"]) > 0 for r in rows])),
            d_ab=float(np.mean([np.sign(r["d_d_ab"]) > 0 for r in rows])),
            ce=float(np.mean([np.sign(r["d_ce"]) < 0 for r in rows])),
            pmi_forward_pos=float(np.mean([np.sign(r["d_pmi_forward"]) > 0 for r in rows])),
            pmi_reverse_pos=float(np.mean([np.sign(r["d_pmi_reverse"]) > 0 for r in rows])),
        )
    # 注：PMI 收敛（gap 缩小）⟺ |δ| 下降（代数恒等），故以 delta_abs 方向为准

    # ---- checkpoint-level correlation / partial association（55 点）----
    ck = {k: np.array([t[k]["mean"] for t in table]) for k in
          ("delta_abs", "ce", "js_dep", "d_ab", "a_sym")}
    pmi_f = np.array([t["pmi_forward"]["mean"] for t in table])
    pmi_r = np.array([t["pmi_reverse"]["mean"] for t in table])
    corr = dict(
        delta_ce=spearman(ck["delta_abs"], ck["ce"]),
        delta_js=spearman(ck["delta_abs"], ck["js_dep"]),
        delta_dab=spearman(ck["delta_abs"], ck["d_ab"]),
        delta_asym=spearman(ck["delta_abs"], ck["a_sym"]),
        ce_js=spearman(ck["ce"], ck["js_dep"]),
        ce_dab=spearman(ck["ce"], ck["d_ab"]),
        partial=dict(
            delta_js_given_ce=rank_partial(ck["delta_abs"], ck["js_dep"], ck["ce"]),
            delta_dab_given_ce=rank_partial(ck["delta_abs"], ck["d_ab"], ck["ce"]),
            delta_ce_given_js=rank_partial(ck["delta_abs"], ck["ce"], ck["js_dep"]),
        ))

    # ---- grouped CV（secondary；group=sample；M0-M3 → |δ|）----
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from phase2_grouped_cv import grouped_cv
    rows = []
    for cid in ids:
        ps = pairs[cid]["per_sample"]
        for k in range(len(ps["delta"])):
            rows.append(dict(
                group_id=ps["manifest_index"][k],
                ce=ps["local_ce"][k],
                js_dep=ps["js_dep"][k],
                d_ab=ps["d_ab"][k],
                y=abs(ps["delta"][k])))
    cv_results = {}
    for name, feats in (("M0_CE", ["ce"]), ("M1_CE_JS", ["ce", "js_dep"]),
                        ("M2_CE_Dab", ["ce", "d_ab"]),
                        ("M3_CE_JS_Dab", ["ce", "js_dep", "d_ab"])):
        cv_results[name] = grouped_cv(rows, feats, "y", binary=False, n_folds=5, seed=0)
    cv_note = ("secondary evidence only：CV 增益不是因果证据，不能单独作为 Phase 2 "
               "成功标准（用户执行纪律）；rows=(checkpoint×sample)，group=underlying "
               "frozen sample；同样本跨 checkpoint 的 rows 同 fold")

    # ---- Case 判定（用户 Phase 2A 判别逻辑）----
    w1 = "500→1020"   # 250–1020 窗口的可及代理（幸存 checkpoint 分辨率）
    w2 = "1020→2500"
    def seg_agg(name):
        rows = [s for s in segs if s["seg"] == name]
        return {k: float(np.mean([r[f"d_{k}"] for r in rows])) for k in
                ("delta_abs", "js_dep", "d_ab", "ce", "pmi_forward", "pmi_reverse")}
    s1 = seg_agg(w1)
    s2 = seg_agg(w2)
    dab_up = s1["d_ab"] > 0
    js_up = s1["js_dep"] > 0
    delta_down = s1["delta_abs"] < 0 and s2["delta_abs"] < 0
    ce_down = s1["ce"] < 0
    plateau = abs(s2["delta_abs"]) < 0.6 * abs(s1["delta_abs"])
    b_signature = (dab_up and js_up and delta_down and
                   s1["pmi_forward"] > 0 and s1["pmi_reverse"] > 0)
    if b_signature and plateau:
        case = ("B（symmetrization / dependence-structure signal）：D_ab 与 JS_dep 在 "
                "500→1020 段保持/上升、PMI 双向为正、|δ| 下降且 1020 后趋缓——"
                "最高优先级 candidate mechanism；建议进入 Phase 2B targeted decoding")
    elif delta_down and ce_down and not b_signature:
        case = ("A（estimation-quality explanation）：|δ| 与 CE 同步下降，"
                "dependence 结构量无独立 temporal signature")
    else:
        case = ("C（no structural signal）：触发 mechanism-screening stop，"
                "项目降级到 empirical/diagnostic framing")

    out = dict(
        bulk_manifest_sha256=digest,
        n_checkpoints=len(ids),
        table=table,
        segments=segs,
        cross_run_consistency=consistency,
        checkpoint_correlation=corr,
        grouped_cv=dict(models=cv_results, note=cv_note),
        window_means=dict(**{w1: s1}, **{w2: s2}),
        case=case,
        caveats=[
            "幸存 checkpoint 分辨率：v1.2/v2.1 仅 500/1020/2500——'250–1020 窗口'由 "
            "500→1020 段代理；dense 中间权重已删（旧 JSON 无 JS/D_ab，仅 |δ|/CE 可作 "
            "secondary），250 前的窗口不可及",
            "PMI_forward − PMI_reverse = δ 为代数恒等：'PMI 收敛'与 |δ| 下降同义；"
            "个体 PMI 轨迹（是否保持正、是否接近）提供独立信息",
            "A_sym 为 exploratory diagnostic（不进入 frozen protocol、非正式 metric）",
        ])
    with open(os.path.join(ROOT, "results", "phase2", "phase2a_analysis.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("=" * 78)
    print("Phase 2A trajectory pair diagnostics")
    print(f"  window {w1}: { {k: round(v,4) for k,v in s1.items()} }")
    print(f"  window {w2}: { {k: round(v,4) for k,v in s2.items()} }")
    print(f"  cross-run consistency: { {k: {kk: round(vv,2) for kk,vv in v.items() if kk!='n_runs'} for k,v in consistency.items()} }")
    print(f"  checkpoint corr: { {k: round(v,3) for k,v in corr.items() if k!='partial'} }")
    print(f"  partial (control CE): JS={corr['partial']['delta_js_given_ce']:+.3f} "
          f"D_ab={corr['partial']['delta_dab_given_ce']:+.3f} | "
          f"CE given JS={corr['partial']['delta_ce_given_js']:+.3f}")
    print(f"  grouped CV (pooled MSE): "
          f"{ {k: round(v['pooled']['mse'],5) for k,v in cv_results.items()} }")
    print(f"\n  CASE: {case}")
    print(f"\nsaved: results/phase2/phase2a_analysis.json")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
