"""E 步: RL-G0 (plan v0.2 §33) — RL init 严格复现 SFT checkpoint。

1. rl/loader.py 加载 formal-s1 checkpoint_10200 EMA → 校验 shadow/model 逐 tensor 一致
2. 复跑 CPI / G1 / OrderGap (frozen manifest, 确定性评估) 与 v4.1 formal s1-10200
   结果逐字段对拍 (全部确定性 → 应逐位一致)
输出: results/rl_gates/rl_g0.json
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch

RUN = "exp_local/regime_a/formal-vanilla-s1-191414"
CKPT = "checkpoint_10200.pth"


def run_eval(script, out):
    subprocess.run([sys.executable, script, "--model_path", RUN, "--ckpt", CKPT,
                    "--weights", "ema", "--out", out], check=True, cwd=ROOT)


def compare_summary(new_path, ref_path, fields):
    new = json.load(open(new_path))
    ref = json.load(open(ref_path))
    ns, rs = new["summary"], ref["summary"]
    diffs = {}
    for f in fields:
        if f in ns and f in rs:
            diffs[f] = abs(ns[f] - rs[f])
        else:
            diffs[f] = "MISSING"
    return diffs, ns, rs


def main():
    device = torch.device("cuda")
    from rl import loader
    model, info = loader.load_rl_init(RUN, CKPT, device)
    print("loader info:", info)
    assert info["step"] == 10200
    assert info["n_shadow"] == info["n_model_params"]
    assert info["shadow_vs_model_mismatches"] == 0, "shadow/model 不一致!"
    del model
    torch.cuda.empty_cache()

    os.makedirs(os.path.join(ROOT, "results/rl_gates"), exist_ok=True)
    results = {"loader": info}

    for script, out, ref, fields, flat in [
        ("evaluation/eval_cpi.py", "/tmp/g0_cpi.json", "results/vanilla/cpi_s1_10200.json",
         ["cpi_abs", "cpi_abs_sem", "cpi_rms", "delta_mean", "delta_sem", "delta_median",
          "delta_sd", "local_ce", "token_acc", "n_samples"], False),
        ("evaluation/eval_task.py", "/tmp/g0_g1.json", "results/vanilla/g1_s1_10200.json",
         ["masked_nll", "token_acc", "greedy_per_pos_hit_rate", "greedy_span_exact_match"], True),
        ("evaluation/eval_order_gap.py", "/tmp/g0_og.json", "results/vanilla/og_s1_10200.json",
         ["order_gap_raw", "order_gap_raw_sem", "order_gap_per_token", "var_pi_mean", "mean_m"], True),
    ]:
        run_eval(script, out)
        new = json.load(open(out)); ref = json.load(open(ref))
        if flat:
            ns = new; rs = ref
        else:
            ns, rs = new["summary"], ref["summary"]
        diffs = {f: (abs(ns[f] - rs[f]) if f in ns and f in rs else "MISSING") for f in fields}
        maxd = max((d for d in diffs.values() if isinstance(d, float)), default=0.0)
        print(f"{os.path.basename(script)}: max|Δ|={maxd:.3e}  diffs={diffs}")
        results[os.path.basename(script)] = dict(max_abs_diff=maxd, diffs=diffs)
        assert maxd < 1e-9, f"RL-G0 FAIL: {script} 与 v4.1 formal s1-10200 不一致!"

    results["verdict"] = "RL-G0 PASS"
    json.dump(results, open(os.path.join(ROOT, "results/rl_gates/rl_g0.json"), "w"),
              indent=2, default=str)
    print("RL-G0 PASS → results/rl_gates/rl_g0.json")


if __name__ == "__main__":
    main()
