"""Phase 1 dynamics 分析（frozen Phase 1 协议 §5/§8/§9/§10/§11 口径，只读、无 GPU）。

研究问题：SFT 过程中 conditional estimation improvement 是否与 CPI attenuation 稳定对齐？
（observational dynamics analysis——不声称因果）

预注册分析：
- 每 checkpoint 核心指标表（CE/CPI_abs/RMS/signed δ/δ_SD/|δ| quantiles/acc）
- 每 run trajectory（baseline=pretrained step0；RL 的 baseline=其 init formal-s1-10200）
- PRIMARY：within-run centered ΔCE vs ΔCPI（baseline-relative Δ 对，Spearman/Pearson +
  slope + bootstrap CI；checkpoints 不是独立 training replicates）
- pooled centered regression（exploratory）+ family-stratified
- leave-one-family-out（formal / v1.2-AB / v1.2-CD / v2.1-HUL / RL / p1-dense）
- δ 分布：per-quantile step0→final 相对变化 → 整体收缩 vs 尾部抑制
- finite-path OrderGap dynamics（frozen 64 子集）
- exploratory lead-lag（3 点轨迹 within-run lag-1；禁称 Granger；p1-dense 旧 JSON 仅
  secondary historical）
- sample-level vs run-level uncertainty 双口径
- SESOI 敏感性 0.01/0.02/0.03
- Verdict P1-A / P1-B / P1-C

用法: .venv/bin/python scripts/phase1_dynamics_analysis.py
输出: results/phase1_diag/phase1_dynamics_analysis.json
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(ROOT, "results", "phase1_diag")
BM_SHA = "3b970159ecc6818beeefd2250d141ae54c7ea309a0dbb85e84415abe9c7bcc6c"
FAMILIES = ["formal", "v1.2-AB", "v1.2-CD", "v2.1-HUL", "RL"]
SESOI_GRID = (0.01, 0.02, 0.03)
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


def pearson(a, b):
    a = np.asarray(a) - np.asarray(a).mean()
    b = np.asarray(b) - np.asarray(b).mean()
    return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))


def bootstrap_ci(x, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cluster_bootstrap(diffs, n_boot=N_BOOT, seed=BOOT_SEED):
    """run 聚类 bootstrap（checkpoints within run 非独立，按 run 重采样）。"""
    rng = np.random.default_rng(seed)
    k = len(diffs)
    sizes = [len(d) for d in diffs]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        draws = [diffs[i][rng.integers(0, sizes[i], sizes[i])] for i in idx]
        boots[b] = np.concatenate(draws).mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    man = j(os.path.join(DIAG, "bulk_manifest.json"))
    digest = hashlib.sha256(open(os.path.join(DIAG, "bulk_manifest.json"), "rb").read()).hexdigest()
    assert digest == BM_SHA, "bulk manifest 被改动"

    cks = man["checkpoints"]
    ids = [c["id"] for c in cks]
    fam = {c["id"]: c["family"] for c in cks}
    run = {c["id"]: c["run"] for c in cks}
    step = {c["id"]: c["step"] for c in cks}

    core = {cid: j(os.path.join(DIAG, "core", f"{cid}.json")) for cid in ids}
    og = {cid: j(os.path.join(DIAG, "ordergap", f"{cid}.json")) for cid in ids}
    aux = {cid: j(os.path.join(DIAG, "aux", f"{cid}.json")) for cid in ids}
    for cid in ids:
        assert core[cid]["manifest_sha256"] == \
            "1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3"

    step0 = core["pretrained-step0"]["summary"]

    # ---- 每 checkpoint 表 ----
    table = []
    for cid in ids:
        s = core[cid]["summary"]
        table.append(dict(id=cid, family=fam[cid], run=run[cid], step=step[cid],
                          ce=s["local_ce"], cpi=s["cpi_abs"], rms=s["cpi_rms"],
                          dm=s["delta_mean"], dsd=s["delta_sd"],
                          acc=s["token_acc"],
                          q50=s["delta_abs_quantiles"]["p50"],
                          q75=s["delta_abs_quantiles"]["p75"],
                          q90=s["delta_abs_quantiles"]["p90"],
                          q95=s["delta_abs_quantiles"]["p95"],
                          og=og[cid]["summary"]["order_gap_raw"],
                          pathvar=og[cid]["summary"]["path_score_variance_mean"],
                          entropy=aux[cid]["summary"]["entropy_mean"],
                          conf=aux[cid]["summary"]["confidence_mean"]))

    # ---- per-run trajectory（baseline-relative Δ）----
    def baseline_for(cid):
        if fam[cid] == "RL":
            return "formal-s1-10200"      # RL init（EMA 同权重）
        return "pretrained-step0"

    traj = {}
    for cid in ids:
        if fam[cid] == "pretrained":
            continue
        b = baseline_for(cid)
        traj.setdefault(run[cid], []).append(dict(
            id=cid, family=fam[cid], step=step[cid],
            dce=core[cid]["summary"]["local_ce"] - core[b]["summary"]["local_ce"],
            dcpi=core[cid]["summary"]["cpi_abs"] - core[b]["summary"]["cpi_abs"],
            drms=core[cid]["summary"]["cpi_rms"] - core[b]["summary"]["cpi_rms"],
            dog=og[cid]["summary"]["order_gap_raw"] - og[b]["summary"]["order_gap_raw"]))
    for k in traj:
        traj[k].sort(key=lambda x: x["step"])

    # ---- PRIMARY：within-run centered ΔCE vs ΔCPI ----
    runs = sorted(traj)
    per_run_pairs = {k: [(p["dce"], p["dcpi"]) for p in traj[k]] for k in runs}
    all_ce = np.array([x for k in runs for x, _ in per_run_pairs[k]])
    all_cpi = np.array([y for k in runs for _, y in per_run_pairs[k]])
    run_diffs_ce = [np.array([x for x, _ in per_run_pairs[k]]) for k in runs]
    run_diffs_cpi = [np.array([y for _, y in per_run_pairs[k]]) for k in runs]

    # 斜率（within-run pooled；普通 OLS + run-clustered bootstrap CI）
    slope = float(np.polyfit(all_ce, all_cpi, 1)[0])
    slope_cis = []
    for k in runs:
        ce = np.array([x for x, _ in per_run_pairs[k]])
        cpi = np.array([y for _, y in per_run_pairs[k]])
        if len(ce) >= 2:
            slope_cis.append(float(np.polyfit(ce, cpi, 1)[0]))
    slope_cis = np.array(slope_cis)
    slope_boot = cluster_bootstrap([[slope_cis]])
    sp_ce_cpi = spearman(all_ce, all_cpi)
    pe_ce_cpi = pearson(all_ce, all_cpi)

    # run-clustered bootstrap of the (ΔCE, ΔCPI) relationship：per-run 斜率汇总 + CI
    per_run_slopes = {}
    for k in runs:
        ce = np.array([x for x, _ in per_run_pairs[k]])
        cpi = np.array([y for _, y in per_run_pairs[k]])
        per_run_slopes[k] = (float(np.polyfit(ce, cpi, 1)[0]) if len(ce) >= 2 else None,
                             len(ce), fam.get(k, fam.get(
                                 next(c["id"] for c in cks if c["run"] == k), "?")))
    slope_arr = np.array([v[0] for v in per_run_slopes.values() if v[0] is not None])
    slope_ci = bootstrap_ci(slope_arr)
    slope_sign_agree = float((slope_arr < 0).mean())

    within_run = dict(
        n_runs=len(runs), n_pairs=len(all_ce),
        pooled_spearman=sp_ce_cpi, pooled_pearson=pe_ce_cpi,
        pooled_slope=slope,
        per_run_slopes={k: dict(slope=v[0], n_pairs=v[1], family=v[2])
                        for k, v in per_run_slopes.items()},
        slope_bootstrap_ci=list(slope_ci),
        slope_negative_fraction=slope_sign_agree)

    # ---- family-stratified ----
    fam_analysis = {}
    for f in FAMILIES:
        ce = np.array([x for k in runs for x, _ in per_run_pairs[k]
                       if fam.get(next(c["id"] for c in cks if c["run"] == k)) == f])
        cpi = np.array([y for k in runs for _, y in per_run_pairs[k]
                        if fam.get(next(c["id"] for c in cks if c["run"] == k)) == f])
        if len(ce) >= 3:
            fam_analysis[f] = dict(n_pairs=len(ce), spearman=spearman(ce, cpi),
                                   slope=float(np.polyfit(ce, cpi, 1)[0]),
                                   ce_mean=float(ce.mean()), cpi_mean=float(cpi.mean()))
        else:
            fam_analysis[f] = dict(n_pairs=len(ce), spearman=None, slope=None,
                                   ce_mean=float(ce.mean()) if len(ce) else None,
                                   cpi_mean=float(cpi.mean()) if len(cpi) else None)

    # ---- leave-one-family-out ----
    loo = {}
    for f_drop in FAMILIES + ["none"]:
        keep = [k for k in runs
                if (f_drop == "none" or fam.get(next(c["id"] for c in cks if c["run"] == k)) != f_drop)]
        ce = np.array([x for k in keep for x, _ in per_run_pairs[k]])
        cpi = np.array([y for k in keep for _, y in per_run_pairs[k]])
        loo[f_drop] = dict(n_runs=len(keep), n_pairs=len(ce),
                           spearman=spearman(ce, cpi),
                           slope=float(np.polyfit(ce, cpi, 1)[0]) if len(ce) >= 2 else None)

    # ---- δ 分布：整体收缩 vs 尾部抑制 ----
    def quant_rel(baseline, target, keys):
        out = {}
        for k in keys:
            b, t = baseline[k], target[k]
            out[k] = dict(baseline=b, target=t, rel_change=(t - b) / b if b else None)
        return out
    sft_finals = [r for r in table if r["family"] in ("formal", "v1.2-AB", "v1.2-CD",
                                                      "v2.1-HUL", "p1-dense")
                  and r["id"].endswith(("2500", "10200"))]
    fin = dict(mean=float(np.mean([r["cpi"] for r in sft_finals])),
               rms=float(np.mean([r["rms"] for r in sft_finals])),
               q50=float(np.mean([r["q50"] for r in sft_finals])),
               q75=float(np.mean([r["q75"] for r in sft_finals])),
               q90=float(np.mean([r["q90"] for r in sft_finals])),
               q95=float(np.mean([r["q95"] for r in sft_finals])))
    base = dict(mean=step0["cpi_abs"], rms=step0["cpi_rms"],
                q50=step0["delta_abs_quantiles"]["p50"],
                q75=step0["delta_abs_quantiles"]["p75"],
                q90=step0["delta_abs_quantiles"]["p90"],
                q95=step0["delta_abs_quantiles"]["p95"])
    delta_shape = quant_rel(base, fin, ("mean", "rms", "q50", "q75", "q90", "q95"))

    # ---- OrderGap dynamics ----
    og_fin = {f: float(np.mean([r["og"] for r in sft_finals if r["family"] == f]))
              for f in ("formal", "v1.2-AB", "v1.2-CD", "v2.1-HUL")}
    og_base = og["pretrained-step0"]["summary"]["order_gap_raw"]
    rl_og = [r["og"] for r in table if r["family"] == "RL"]
    ordergap = dict(base=og_base,
                    sft_finals=og_fin,
                    sft_finals_delta={f: v - og_base for f, v in og_fin.items()},
                    rl_trajectory=rl_og,
                    pathvar_base=og["pretrained-step0"]["summary"]["path_score_variance_mean"],
                    pathvar_finals={f: float(np.mean(
                        [og[next(c["id"] for c in cks if c["run"] == r)]["summary"]
                         ["path_score_variance_mean"] for r in runs if fam.get(
                             next(c["id"] for c in cks if c["run"] == r)) == f]))
                        for f in ("formal", "v1.2-AB", "v1.2-CD", "v2.1-HUL")})

    # ---- exploratory lead-lag（within-run 相邻段，lag-1；禁称 Granger）----
    # 相邻段定义：segment t = (checkpoint_t → checkpoint_{t+1}) 的 ΔCE/ΔCPI 增量；
    # ce_leads: corr(ΔCE(seg t), ΔCPI(seg t+1))；cpi_leads: corr(ΔCPI(seg t), ΔCE(seg t+1))
    segs = {}
    for k in runs:
        t = traj[k]
        for a, b in zip(t[:-1], t[1:]):
            segs.setdefault(k, []).append((b["dce"] - a["dce"], b["dcpi"] - a["dcpi"]))
    lag_ce_cpi, lag_cpi_ce = [], []
    for k in runs:
        s = segs[k]
        for i in range(len(s) - 1):
            lag_ce_cpi.append((s[i][0], s[i + 1][1]))
            lag_cpi_ce.append((s[i][1], s[i + 1][0]))
    leadlag = dict(n_lag_pairs=len(lag_ce_cpi),
                   n_segments=sum(len(s) for s in segs.values()),
                   ce_leads=dict(
                       spearman=spearman(*zip(*lag_ce_cpi)) if lag_ce_cpi else None,
                       pearson=pearson(*zip(*lag_ce_cpi)) if lag_ce_cpi else None),
                   cpi_leads=dict(
                       spearman=spearman(*zip(*lag_cpi_ce)) if lag_cpi_ce else None,
                       pearson=pearson(*zip(*lag_cpi_ce)) if lag_cpi_ce else None),
                   note="exploratory temporal-ordering diagnostic；不声称 Granger causality")

    # ---- sample-level vs run-level uncertainty ----
    sample_se = step0["cpi_abs_sem"]
    run_level_sd = float(np.std([r["cpi"] for r in table if r["family"] != "pretrained"]))
    uncertainty = dict(
        sample_level_se_step0=sample_se,
        run_level_sd_14finals=run_level_sd,
        note="sample-level SE（~0.02）是固定模型估计精度；run-level SD 是训练层面噪声；"
             "两者不可互换；独立 training replicates 数少（≤2/family）")

    # ---- Verdict（P1-A/B/C，用户 §18 判据；operationalization 与输出一并披露）----
    ce_improved = float(np.mean([p["dce"] for k in runs for p in traj[k]])) < 0
    rel_stable = (within_run["pooled_spearman"] >= 0.5 and slope_ci[1] < 0)
    loo_stable = all(loo[f]["spearman"] is not None and loo[f]["spearman"] >= 0.3
                     for f in FAMILIES if loo[f]["n_pairs"] >= 10)
    if ce_improved and rel_stable and loo_stable:
        verdict = ("P1-A：conditional estimation improvement 与 SFT-induced compatibility "
                   "attenuation 存在稳定共变关系（仍非因果；可进入 Phase 2）")
    elif ce_improved and not rel_stable and not loo_stable:
        verdict = ("P1-B：单纯 conditional estimation improvement 不足以解释 SFT-induced "
                   "compatibility attenuation")
    else:
        verdict = ("P1-C：当前证据不足以判断 CE–CPI dynamics（方向可能但 CI/一致性不足；"
                   "停下 review，不立即加 seeds）")

    out = dict(
        bulk_manifest_sha256=digest,
        n_checkpoints=len(ids),
        table=table,
        trajectories={k: traj[k] for k in runs},
        within_run=within_run,
        family_stratified=fam_analysis,
        leave_one_family_out=loo,
        delta_distribution=delta_shape,
        ordergap=ordergap,
        lead_lag=leadlag,
        uncertainty=uncertainty,
        sesoi_grid=list(SESOI_GRID),
        verdict=verdict,
        verdict_operationalization=dict(
            ce_improved="mean(ΔCE over all trajectory points) < 0",
            rel_stable="pooled Spearman ≥ 0.5 且 run-clustered slope bootstrap CI 上限 < 0",
            loo_stable="所有 n_pairs≥10 的 leave-one-out Spearman ≥ 0.3"),
        verdict_components=dict(
            ce_improved=bool(ce_improved), rel_stable=bool(rel_stable),
            loo_stable=bool(loo_stable),
            pooled_spearman=within_run["pooled_spearman"],
            slope_ci=slope_ci))
    with open(os.path.join(DIAG, "phase1_dynamics_analysis.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("=" * 78)
    print("Phase 1 dynamics 分析")
    print(f"  within-run: n_runs={len(runs)} n_pairs={len(all_ce)} "
          f"Spearman={sp_ce_cpi:+.3f} Pearson={pe_ce_cpi:+.3f} slope={slope:+.4f} "
          f"slope CI={[round(x,3) for x in slope_ci]} 负斜率占比={slope_sign_agree:.0%}")
    print(f"  family: { {k: (round(v['spearman'],3) if v['spearman'] is not None else None) for k, v in fam_analysis.items()} }")
    print(f"  leave-one-out Spearman: { {k: round(v['spearman'],3) for k, v in loo.items()} }")
    print(f"  δ 分布 rel change: mean={delta_shape['mean']['rel_change']:+.1%} "
          f"q50={delta_shape['q50']['rel_change']:+.1%} q90={delta_shape['q90']['rel_change']:+.1%} "
          f"q95={delta_shape['q95']['rel_change']:+.1%}")
    print(f"  OrderGap: base={og_base:.2f} finals={ {k: round(v,2) for k, v in og_fin.items()} }")
    print(f"  lead-lag: ce_leads S={leadlag['ce_leads']['spearman']:+.3f} / "
          f"cpi_leads S={leadlag['cpi_leads']['spearman']:+.3f}（exploratory）")
    print(f"\n  VERDICT: {verdict}")
    print(f"\nsaved: {os.path.join(DIAG, 'phase1_dynamics_analysis.json')}")


if __name__ == "__main__":
    main()
