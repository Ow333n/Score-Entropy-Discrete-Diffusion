"""s2 deterministic reproduction gate（用户 2026-10-07 优先级修正版；只读）。

比较对象：新 seed2 run vs 旧 optionb-dense-s2-215621。

优先级（用户定义，SHA256 不作最高权威）：
  1. tensor-by-tensor torch.equal：model state_dict / EMA shadow_params /
     step / EMA num_updates / decay
  2. train_loss / eval_loss 逐位一致（learning_curve.csv + train.log 重叠区间）
  3. checkpoint 文件 SHA256（仅附加证据）

判定：
  PASS         所有对应 tensor torch.equal == True 且 loss 逐位一致
  PASS-CAVEAT  仅极小数值差异 → 报告 max_abs_diff / max_rel_diff /
               affected_tensor_count，硬停等待用户判断（不自动继续 Phase 2）
  HARD STOP    model/EMA trajectory 实质差异（max_rel > 1e-5）

用法: .venv/bin/python scripts/optionb_s2_repro_gate.py
输出: results/phase2/optionb_s2_repro_gate.json
"""
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_DIR = os.path.join(ROOT, "exp_local", "regime_a", "optionb-dense-s2-215621")
STEPS = [50, 100, 150, 200, 250, 300, 350, 400, 500, 750]  # 旧 s2 完整覆盖的步骤
REL_THRESH = 1e-5  # 与 compare_checkpoints.py 同阈值：kernel 噪声 ≤1e-6，实质分歧 ≥1e-3
OUT = os.path.join(ROOT, "results", "phase2", "optionb_s2_repro_gate.json")


def resolve_new_dir():
    dirs = sorted(glob.glob(os.path.join(ROOT, "exp_local/regime_a/optionb-dense-s2-*")))
    new = dirs[-1]
    assert os.path.basename(new) != os.path.basename(OLD_DIR), "新 run 目录与旧 run 目录相同"
    return new


def tensor_pair_stats(name, old, new):
    """torch.equal 逐个比较；不等时累计 max_abs / max_rel / affected。"""
    stats = dict(name=name, n=0, n_equal=0, max_abs=0.0, max_rel=0.0, affected=[])
    assert len(old) == len(new), f"{name}: 张量数不一致 {len(old)} vs {len(new)}"
    for i, (a, b) in enumerate(zip(old, new)):
        stats["n"] += 1
        if torch.equal(a, b):
            stats["n_equal"] += 1
        else:
            d = (a.float() - b.float()).abs()
            denom = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-12)
            mabs = d.max().item()
            mrel = (d / denom).max().item()
            stats["max_abs"] = max(stats["max_abs"], mabs)
            stats["max_rel"] = max(stats["max_rel"], mrel)
            stats["affected"].append(i)
    return stats


def compare_ckpt(new_dir, step):
    np_ = os.path.join(new_dir, f"checkpoint_{step}.pth")
    op = os.path.join(OLD_DIR, f"checkpoint_{step}.pth")
    res = dict(step=step, sha256={}, meta={}, tensors={})
    res["sha256"]["old"] = hashlib.sha256(open(op, "rb").read()).hexdigest()
    res["sha256"]["new"] = hashlib.sha256(open(np_, "rb").read()).hexdigest()
    a = torch.load(op, map_location="cpu", weights_only=False)
    b = torch.load(np_, map_location="cpu", weights_only=False)
    res["meta"] = dict(step_old=a["step"], step_new=b["step"],
                       decay_old=a["ema"]["decay"], decay_new=b["ema"]["decay"],
                       num_updates_old=a["ema"]["num_updates"],
                       num_updates_new=b["ema"]["num_updates"])
    res["tensors"]["ema"] = tensor_pair_stats(
        "ema_shadow", a["ema"]["shadow_params"], b["ema"]["shadow_params"])
    if "model" in a:
        assert "model" in b, f"step {step}: 新文件缺 model key"
        assert set(a["model"].keys()) == set(b["model"].keys()), "model key 集合不一致"
        res["tensors"]["model"] = tensor_pair_stats(
            "model", list(a["model"].values()), list(b["model"].values()))
    del a, b
    return res


def compare_losses(new_dir):
    res = dict(curve={}, log={})
    old_curve = open(os.path.join(OLD_DIR, "learning_curve.csv")).read().splitlines()
    new_curve = open(os.path.join(new_dir, "learning_curve.csv")).read().splitlines()
    for s in (500, 1000):
        ol = [l for l in old_curve if l.startswith(f"{s},")]
        nl = [l for l in new_curve if l.startswith(f"{s},")]
        res["curve"][s] = dict(old=ol, new=nl, equal=(ol == nl))

    # train.log 数值字段逐位比对（不含 steps/s / vram 等墙钟相关字段）
    def parse(path):
        out = {}
        for line in open(path):
            m = re.match(r"^step\s+(\d+):\s+train_loss=(\S+)\s+lr=(\S+)", line)
            if m:
                out[("train", int(m.group(1)))] = (m.group(2), m.group(3))
            m = re.match(r"^step\s+(\d+):\s+eval_loss=(\S+)", line)
            if m:
                out[("eval", int(m.group(1)))] = m.group(2)
        return out

    po = parse(os.path.join(OLD_DIR, "train.log"))
    pn = parse(os.path.join(new_dir, "train.log"))
    shared = sorted(set(po) & set(pn))
    mismatches = [[k, po[k], pn[k]] for k in shared if po[k] != pn[k]]
    res["log"] = dict(n_shared=len(shared), mismatches=mismatches)
    return res


def main():
    new_dir = resolve_new_dir()
    print(f"OLD: {OLD_DIR}")
    print(f"NEW: {new_dir}")
    out = dict(old=OLD_DIR, new=new_dir, checkpoints={}, losses=None, verdict=None)
    worst_rel = 0.0
    all_tensors_equal = True
    for step in STEPS:
        c = compare_ckpt(new_dir, step)
        out["checkpoints"][str(step)] = c
        sha_same = c["sha256"]["old"] == c["sha256"]["new"]
        print(f"step {step:4d}: sha256_equal={sha_same}")
        for name, st in c["tensors"].items():
            ok = st["n_equal"] == st["n"]
            all_tensors_equal = all_tensors_equal and ok
            worst_rel = max(worst_rel, st["max_rel"])
            print(f"  {name}: {st['n_equal']}/{st['n']} equal "
                  f"(max_abs={st['max_abs']:.3e} max_rel={st['max_rel']:.3e})"
                  + ("" if ok else f"  affected={len(st['affected'])}"))
        if not c["meta"]["step_old"] == c["meta"]["step_new"]:
            all_tensors_equal = False
    losses = compare_losses(new_dir)
    out["losses"] = losses
    curve_ok = all(v["equal"] for v in losses["curve"].values())
    log_ok = len(losses["log"]["mismatches"]) == 0
    print(f"losses: curve 500/1000 equal={curve_ok} log shared={losses['log']['n_shared']} "
          f"mismatches={len(losses['log']['mismatches'])}")
    for mm in losses["log"]["mismatches"]:
        print(f"  MISMATCH {mm}")
    losses_equal = curve_ok and log_ok

    if all_tensors_equal and losses_equal:
        verdict = "PASS"
    elif worst_rel <= REL_THRESH:
        verdict = "PASS-CAVEAT"
    else:
        verdict = "HARD STOP"
    out["verdict"] = verdict
    out["worst_max_rel"] = worst_rel
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nGATE VERDICT: {verdict} (worst_max_rel={worst_rel:.3e})")
    print(f"saved: {OUT}")
    if verdict == "PASS-CAVEAT":
        print("PASS-CAVEAT：硬停，等用户判断后再决定是否继续 Phase 2。")
    if verdict == "HARD STOP":
        print("HARD STOP：trajectory 实质分歧，禁止把新 run 当作原 trajectory 替代品。")


if __name__ == "__main__":
    main()
