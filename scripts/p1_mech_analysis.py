"""Mechanism pilot Stage P1 第一阶段分析（只读，等评估完成后运行）。

产出（protocol v1.2 §9 staged 交付口径）：
- temporal 表：step {0,500,1020,2500} × replicate {1,2} × {NLL, token acc,
  CPI_abs, CPI_RMS, signed δ mean, ΔCPI(A−B)}
- per-replicate paired sample bootstrap 95% CI（|δ|_A − |δ|_B 逐样本配对，
  10k 重采样，不 pool 两 replicate）
- 两 replicate 点估计是否同方向
- training loss curves（train.log 解析）
- schedule SHA-256 / peak VRAM / runtime / 异常（run_metadata.json）

用法: .venv/bin/python scripts/p1_mech_analysis.py
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "mechanism_pilot")

RUNS = ["A1", "B1", "A2", "B2"]
STEPS = [0, 500, 1020, 2500]
N_BOOT = 10000


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def cpi_summary(run, step):
    name = f"cpi_step0.json" if step == 0 else f"cpi_{run}_{step}.json"
    return load_json(name)["summary"]


def task_summary(run, step):
    name = f"task_step0.json" if step == 0 else f"task_{run}_{step}.json"
    d = load_json(name)
    s = d.get("summary", d)
    return s


def paired_bootstrap(diff, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    out = {"steps": []}
    # per-sample |δ| 载入（step0 一份；其余 per run/step）
    step0 = load_json("cpi_step0.json")
    d0 = np.array(step0["per_sample"]["delta_abs"])

    rows = []
    for step in STEPS:
        for r, (a_run, b_run) in zip((1, 2), (("A1", "B1"), ("A2", "B2"))):
            if step == 0:
                ca = cb = cpi_summary(None, 0)
                ta = tb = task_summary(None, 0)
                da = db = d0
            else:
                ca, cb = cpi_summary(a_run, step), cpi_summary(b_run, step)
                ta, tb = task_summary(a_run, step), task_summary(b_run, step)
                da = np.array(load_json(f"cpi_{a_run}_{step}.json")["per_sample"]["delta_abs"])
                db = np.array(load_json(f"cpi_{b_run}_{step}.json")["per_sample"]["delta_abs"])
            d_cpi = ca["cpi_abs"] - cb["cpi_abs"]
            diff = da - db
            lo, hi = paired_bootstrap(diff)
            rows.append(dict(
                step=step, replicate=r,
                nll_a=ta["masked_nll"], nll_b=tb["masked_nll"],
                acc_a=ta["token_acc"], acc_b=tb["token_acc"],
                cpi_a=ca["cpi_abs"], cpi_b=cb["cpi_abs"],
                cpi_rms_a=ca["cpi_rms"], cpi_rms_b=cb["cpi_rms"],
                delta_mean_a=ca["delta_mean"], delta_mean_b=cb["delta_mean"],
                d_cpi=d_cpi,
                boot_ci_lo=lo, boot_ci_hi=hi,
                ci_excludes_zero=(lo > 0 or hi < 0),
                direction="negative" if d_cpi < 0 else ("positive" if d_cpi > 0 else "zero"),
            ))
        rep1 = next(x for x in rows if x["step"] == step and x["replicate"] == 1)
        rep2 = next(x for x in rows if x["step"] == step and x["replicate"] == 2)
        same_dir = rep1["direction"] == rep2["direction"] and rep1["direction"] != "zero"
        print(f"step {step:>4}: rep1 ΔCPI={rep1['d_cpi']:+.4f} [{rep1['boot_ci_lo']:+.4f},{rep1['boot_ci_hi']:+.4f}] "
              f"rep2 ΔCPI={rep2['d_cpi']:+.4f} [{rep2['boot_ci_lo']:+.4f},{rep2['boot_ci_hi']:+.4f}] "
              f"同方向={same_dir}")
        out["steps"].append(dict(step=step, replicate1=rep1, replicate2=rep2,
                                 same_direction=same_dir))

    # 训练元数据
    meta_rows = []
    for run in RUNS:
        m = glob.glob(os.path.join(ROOT, "exp_local", "regime_a",
                                   f"mechpilot-{run}-*", "run_metadata.json"))
        m = json.load(open(m[0]))
        log = glob.glob(os.path.join(ROOT, "exp_local", "regime_a",
                                     f"mechpilot-{run}-*", "train.log"))[0]
        losses = []
        for line in open(log):
            mm = re.search(r"step\s+(\d+): train_loss=([\d.]+)", line)
            if mm:
                losses.append([int(mm.group(1)), float(mm.group(2))])
        meta_rows.append(dict(run=run, sha256=m["schedule_sha256"],
                              steps_per_s=m["steps_per_second"],
                              vram_gb=m["peak_vram_allocated_gb"],
                              losses=losses))
        print(f"{run}: sha={m['schedule_sha256'][:16]}… {m['steps_per_second']:.1f} steps/s "
              f"vram={m['peak_vram_allocated_gb']:.2f}GB loss[{losses[0][1]:.3f}→{losses[-1][1]:.3f}]")
    out["training"] = meta_rows

    path = os.path.join(RES, "p1_stage1_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nanalysis: {path}")


if __name__ == "__main__":
    main()
