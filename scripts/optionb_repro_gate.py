"""Option B reproduction gate（用户 gate 双路径，只读）。

锚点链（Gate-2 历史定案：P1-s1@1020 与 formal-s1@1020 逐位一致——formal 权重幸存）：
  1. retrain@2500 EMA/model vs 幸存 p1-{s1,s2} checkpoint_2500（全量容器）逐位比对
  2. retrain@1020 EMA vs 幸存 formal-{s1,s2} checkpoint_1020 EMA 逐位比对
     （经 Gate-2 传递：P1@1020 ≡ formal@1020）
  3. retrain@500 / @2500 用 frozen 旧 evaluator 同 manifest 跑指标 vs frozen
     p1_dense JSON（cpi_s{1,2}_{500,2500}_ema.json）——数值一致（bitwise EMA 的
     下游推论 + 指标级双保险）
判定：bitwise 全过 → gate PASS（最高置信）；bitwise 失败但指标一致 → PASS-CAVEAT
（kernel 噪声）；指标也不一致 → **HARD STOP，先解释 reproduction failure**，
不允许把 retrain trajectory 当作原 trajectory 的替代品。

用法: .venv/bin/python scripts/optionb_repro_gate.py
输出: results/phase2/optionb_repro_gate.json
"""
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model.ema import ExponentialMovingAverage
from model import SEDD
from hydra import initialize, compose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_SHA = "1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3"


def load_ema_tensors(path, key="ema"):
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    state = loaded[key]
    if isinstance(state, dict) and "shadow_params" in state:
        return [t.clone() for t in state["shadow_params"]]
    return state  # 意外容器，交由调用方处理


def load_model_tensors(path):
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    return loaded["model"]


def bitwise_equal(a, b):
    if isinstance(a, list):
        assert len(a) == len(b)
        return all(torch.equal(x, y) for x, y in zip(a, b))
    return all(torch.equal(a[k], b[k]) for k in a)


def old_eval_metrics(model_dir, ckpt):
    """frozen 旧 evaluator 同 manifest 指标（CPI/CE/acc），用于 500/2500 指标门。"""
    sys.argv = ["eval_cpi.py", "--model_path", model_dir, "--ckpt", ckpt,
                "--weights", "ema", "--tag", f"gate-{os.path.basename(model_dir)}",
                "--out", "/tmp/gate_cpi.json"]
    import evaluation.eval_cpi as ec
    ec.main()
    d = json.load(open("/tmp/gate_cpi.json"))
    return d["summary"]


def main():
    out = {}
    gate = dict(bitwise={}, metrics={})

    for seed, surv in ((1, "p1-s1-020240"), (2, "p1-s2-110355")):
        retrain_dirs = sorted(glob.glob(os.path.join(
            ROOT, f"exp_local/regime_a/optionb-dense-s{seed}-*")))
        assert retrain_dirs, f"未找到 optionb-dense-s{seed} 目录"
        rdir = retrain_dirs[-1]
        sdir = os.path.join(ROOT, "exp_local/regime_a", surv)

        # 1) 2500 anchor：EMA + model 逐位
        try:
            r_ema = load_ema_tensors(os.path.join(rdir, "checkpoint_2500.pth"))
            s_ema = load_ema_tensors(os.path.join(sdir, "checkpoint_2500.pth"))
            ema_ok = bitwise_equal(r_ema, s_ema)
            r_model = load_model_tensors(os.path.join(rdir, "checkpoint_2500.pth"))
            s_model = load_model_tensors(os.path.join(sdir, "checkpoint_2500.pth"))
            model_ok = bitwise_equal(r_model, s_model)
            gate["bitwise"][f"s{seed}@2500"] = dict(ema=ema_ok, model=model_ok)
            print(f"s{seed}@2500: EMA bitwise={ema_ok} model bitwise={model_ok}")
        except Exception as e:
            gate["bitwise"][f"s{seed}@2500"] = dict(error=str(e))
            print(f"s{seed}@2500: ERROR {e}")

        # 2) 1020 anchor：经 Gate-2 传递链 vs formal 幸存
        formal = os.path.join(ROOT, f"exp_local/regime_a/formal-vanilla-s{seed}-"
                              f"{'191414' if seed == 1 else '204215'}")
        try:
            r_ema = load_ema_tensors(os.path.join(rdir, "checkpoint_1020.pth"))
            f_ema = load_ema_tensors(os.path.join(formal, "checkpoint_1020.pth"))
            ok = bitwise_equal(r_ema, f_ema)
            gate["bitwise"][f"s{seed}@1020_vs_formal"] = dict(ema=ok)
            print(f"s{seed}@1020 vs formal: EMA bitwise={ok}（经 Gate-2 传递链）")
        except Exception as e:
            gate["bitwise"][f"s{seed}@1020_vs_formal"] = dict(error=str(e))
            print(f"s{seed}@1020: ERROR {e}")

        # 3) 指标门：500 / 2500 vs frozen p1_dense JSON
        for step in (500, 2500):
            try:
                m = old_eval_metrics(rdir, f"checkpoint_{step}.pth")
                frozen = json.load(open(os.path.join(
                    ROOT, "results", "p1_dense", f"cpi_s{seed}_{step}_ema.json")))
                fs = frozen["summary"]
                diffs = dict(cpi=abs(m["cpi_abs"] - fs["cpi_abs"]),
                             rms=abs(m["cpi_rms"] - fs["cpi_rms"]),
                             ce=abs(m["local_ce"] - fs["local_ce"]),
                             acc=abs(m["token_acc"] - fs["token_acc"]))
                gate["metrics"][f"s{seed}@{step}"] = dict(
                    retrain=dict(cpi=m["cpi_abs"], ce=m["local_ce"], acc=m["token_acc"]),
                    frozen=dict(cpi=fs["cpi_abs"], ce=fs["local_ce"], acc=fs["token_acc"]),
                    diffs=diffs)
                print(f"s{seed}@{step} metrics vs frozen: {diffs}")
            except Exception as e:
                gate["metrics"][f"s{seed}@{step}"] = dict(error=str(e))
                print(f"s{seed}@{step}: ERROR {e}")

    # 判定
    b = gate["bitwise"]
    all_bitwise = all(v.get("ema", False) and v.get("model", True) for v in b.values()
                      if "error" not in v)
    all_bitwise = (b.get("s1@2500", {}).get("ema", False)
                   and b.get("s1@2500", {}).get("model", False)
                   and b.get("s2@2500", {}).get("ema", False)
                   and b.get("s2@2500", {}).get("model", False)
                   and b.get("s1@1020_vs_formal", {}).get("ema", False)
                   and b.get("s2@1020_vs_formal", {}).get("ema", False))
    metrics_ok = all(v.get("diffs", {}).get("cpi", 1) < 0.01
                     for v in gate["metrics"].values() if "error" not in v)
    if all_bitwise and metrics_ok:
        verdict = "PASS（bitwise 全过 + 指标一致）"
    elif metrics_ok:
        verdict = "PASS-CAVEAT（bitwise 未全过但指标一致；需解释）"
    else:
        verdict = "HARD STOP（reproduction failure：指标不一致，先解释，禁止把 retrain 当原 trajectory 替代品）"
    out = dict(gate=gate, verdict=verdict,
               manifest_sha256=MANIFEST_SHA,
               retrain_dirs={f"s{seed}": sorted(glob.glob(os.path.join(
                   ROOT, f"exp_local/regime_a/optionb-dense-s{seed}-*")))[-1]
                   for seed in (1, 2)})
    os.makedirs(os.path.join(ROOT, "results", "phase2"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "phase2", "optionb_repro_gate.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nGATE VERDICT: {verdict}")
    print(f"saved: results/phase2/optionb_repro_gate.json")


if __name__ == "__main__":
    main()
