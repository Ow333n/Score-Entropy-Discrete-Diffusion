"""Mechanism pilot Stage P2 G2 分析（protocol v1.2 §5.2，只读）。

检验（G2 — directional auxiliary gate）：
- mean signed δ 的 C vs D 差异，paired bootstrap 95% CI（10k，按 replicate 聚类）排除 0；
- 且 sign(δ̄_C) = −sign(δ̄_D) 在两 replicate 一致。
- 三值判定同 G1a：
    PASS        : 聚类 CI 排除 0 且两 replicate 均为相反符号
    FAIL        : 聚类 CI 排除 0 且两 replicate 均为同符号（反驳"相反"预注册）
    INCONCLUSIVE: CI 含 0，或 replicate 间符号模式不一致，或任一点估计为 0
- primary 口径 step 2500 EMA；500/1020 作 temporal 参考（预注册未定义其 gate 地位）。

产出：
- temporal 表：step × replicate × {NLL, token acc, CPI_abs, CPI_RMS, signed δ mean}
- per-replicate paired sample bootstrap 95% CI（seed=0, 10k，同 p1_mech_analysis 约定）
- 按 replicate 聚类的 pooled bootstrap CI（gate 口径）
- G2 verdict + 两 replicate directional sign 一致性
- schedule SHA-256 / runtime / VRAM / 异常

用法: .venv/bin/python scripts/p2_g2_analysis.py
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "mechanism_pilot")

RUNS = ["C1", "D1", "C2", "D2"]
STEPS = [0, 500, 1020, 2500]
PRIMARY_STEP = 2500
N_BOOT = 10000
BOOT_SEED = 0
MANIFEST_SHA = "1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3"


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def paired_bootstrap(diff, n_boot=N_BOOT, seed=BOOT_SEED):
    """per-sample paired resampling，同 p1_mech_analysis 约定。"""
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def clustered_bootstrap(diffs, n_boot=N_BOOT, seed=BOOT_SEED):
    """按 replicate 聚类：先有放回重采样 cluster，再在每个 cluster 内有放回重采样样本。"""
    rng = np.random.default_rng(seed)
    k = len(diffs)
    sizes = [len(d) for d in diffs]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)  # 聚类重采样
        draws = [diffs[j][rng.integers(0, sizes[j], sizes[j])] for j in idx]
        boots[b] = np.concatenate(draws).mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sign(x, tol=1e-12):
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def verdict(ci_lo, ci_hi, opp_flags):
    excludes = ci_lo > 0 or ci_hi < 0
    if not excludes:
        return "INCONCLUSIVE"
    if all(opp_flags):
        return "PASS"
    if not any(opp_flags) and all(f is False for f in opp_flags):
        return "FAIL"
    return "INCONCLUSIVE"


def main():
    # 校验：manifest 一致 + 500 样本 + 逐样本 delta 存在
    for run in RUNS:
        for step in (500, 1020, 2500):
            d = load_json(f"cpi_{run}_{step}.json")
            assert d["manifest_sha256"] == MANIFEST_SHA, f"{run}@{step} manifest hash 不一致"
            assert len(d["per_sample"]["delta"]) == 500, f"{run}@{step} 样本数异常"
    d0 = np.array(load_json("cpi_step0.json")["per_sample"]["delta"])
    assert load_json("cpi_step0.json")["manifest_sha256"] == MANIFEST_SHA

    out = {"steps": [], "gate": {}}
    print("== G2 temporal 表（C vs D, mean signed δ） ==")
    for step in STEPS:
        rep_rows = []
        diffs = []
        opp_flags = []
        for r, (c_run, d_run) in zip((1, 2), (("C1", "D1"), ("C2", "D2"))):
            if step == 0:
                cc = cd = load_json("cpi_step0.json")["summary"]
                tc = td = load_json("task_step0.json")["summary"] \
                    if "summary" in load_json("task_step0.json") else load_json("task_step0.json")
                dc = dd = d0
            else:
                cc = load_json(f"cpi_{c_run}_{step}.json")["summary"]
                cd = load_json(f"cpi_{d_run}_{step}.json")["summary"]
                tj = load_json(f"task_{c_run}_{step}.json")
                tc = tj.get("summary", tj)
                tj = load_json(f"task_{d_run}_{step}.json")
                td = tj.get("summary", tj)
                dc = np.array(load_json(f"cpi_{c_run}_{step}.json")["per_sample"]["delta"])
                dd = np.array(load_json(f"cpi_{d_run}_{step}.json")["per_sample"]["delta"])
            diff = dc - dd  # 逐样本配对（同一 frozen manifest 顺序）
            lo, hi = paired_bootstrap(diff)
            opp = sign(cc["delta_mean"]) == -sign(cd["delta_mean"]) != 0
            opp_flags.append(bool(opp))
            diffs.append(diff)
            row = dict(
                step=step, replicate=r,
                nll_c=tc["masked_nll"], nll_d=td["masked_nll"],
                acc_c=tc["token_acc"], acc_d=td["token_acc"],
                cpi_c=cc["cpi_abs"], cpi_d=cd["cpi_abs"],
                rms_c=cc["cpi_rms"], rms_d=cd["cpi_rms"],
                delta_c=cc["delta_mean"], delta_d=cd["delta_mean"],
                d_cd=float(diff.mean()),
                boot_ci_lo=lo, boot_ci_hi=hi,
                ci_excludes_zero=(lo > 0 or hi < 0),
                signs_opposite=bool(opp),
                s_c=sign(cc["delta_mean"]), s_d=sign(cd["delta_mean"]),
            )
            rep_rows.append(row)
            print(f"step {step:>4} rep{r}: δ̄_C={row['delta_c']:+.4f} δ̄_D={row['delta_d']:+.4f} "
                  f"dCD={row['d_cd']:+.4f} [{lo:+.4f},{hi:+.4f}] 符号相反={bool(opp)}")
        clo, chi = clustered_bootstrap(diffs)
        v = verdict(clo, chi, opp_flags)
        out["steps"].append(dict(step=step, replicate1=rep_rows[0], replicate2=rep_rows[1],
                                 clustered_ci=[clo, chi],
                                 clustered_ci_excludes_zero=(clo > 0 or chi < 0),
                                 gate_verdict=v))
        print(f"  → 聚类 CI [{clo:+.4f},{chi:+.4f}] 排除0={(clo>0 or chi<0)} | G2@{step}: {v}")
        out["gate"][f"step{step}"] = v
    out["gate"]["primary"] = out["gate"][f"step{PRIMARY_STEP}"]
    out["gate"]["primary_step"] = PRIMARY_STEP

    # 训练元数据 + 异常扫描
    meta_rows = []
    anomalies = []
    for run in RUNS:
        m = json.load(open(glob.glob(os.path.join(
            ROOT, "exp_local", "regime_a", f"mechpilot-{run}-*", "run_metadata.json"))[0]))
        log = glob.glob(os.path.join(
            ROOT, "exp_local", "regime_a", f"mechpilot-{run}-*", "train.log"))[0]
        losses = []
        for line in open(log):
            mm = re.search(r"step\s+(\d+): train_loss=([\d.]+)", line)
            if mm:
                losses.append([int(mm.group(1)), float(mm.group(2))])
        loss_arr = np.array([x[1] for x in losses])
        bad = (~np.isfinite(loss_arr)).sum()
        runtime = m["n_iters"] / m["steps_per_second"]
        if bad:
            anomalies.append(f"{run}: {bad} 个非有限 train_loss")
        if len(losses) != 26:
            anomalies.append(f"{run}: train.log 解析 {len(losses)} 个 loss 点（预期 26）")
        meta_rows.append(dict(run=run, policy=m["policy"], replicate=m["replicate"],
                              schedule_sha256=m["schedule_sha256"],
                              git_commit=m["git_commit"],
                              steps_per_s=m["steps_per_second"],
                              runtime_s=round(runtime, 1),
                              vram_gb=m["peak_vram_allocated_gb"],
                              loss_first=float(loss_arr[0]), loss_last=float(loss_arr[-1]),
                              loss_nan=int(bad)))
        print(f"{run}: sha={m['schedule_sha256'][:16]}… {m['steps_per_second']:.2f} steps/s "
              f"vram={m['peak_vram_allocated_gb']:.2f}GB runtime={runtime:.0f}s "
              f"loss[{loss_arr[0]:.2f}→{loss_arr[-1]:.2f}] nan={bad}")
    out["training"] = meta_rows
    out["anomalies"] = anomalies
    out["n_boot"] = N_BOOT
    out["bootstrap_seed"] = BOOT_SEED

    path = os.path.join(RES, "p2_g2_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nG2 primary @{PRIMARY_STEP}: {out['gate']['primary']}")
    if anomalies:
        print("anomalies:", anomalies)
    print(f"analysis: {path}")


if __name__ == "__main__":
    main()
