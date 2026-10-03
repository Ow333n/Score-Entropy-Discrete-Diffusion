"""LR review 收尾离线计算 (用户 D 口径, eval-only):

对三条 150-step run 的 final snapshot (3e-6 smoke / 1e-6 / 1e-5) 统一补算:
  - raw NLL (冻结 corruption realization: seed+1000 流) → 与在线 RAW 交叉验证
  - EMA NLL (同流) → 与 smoke logged EMA 7.0171 交叉验证 harness
  - greedy eval8 (raw PRIMARY + EMA SECONDARY, 同 seed 同 J, argmax 无采样噪声)
  - sampled eval8 (raw) → 与 logged 值交叉验证
  - model drift ‖θ150−θ0‖/‖θ0‖ (raw & EMA)

复用 scripts/rl1_post_smoke_audit.py 的 rollout_audit / run_nll_eval (逐行同 rollout_chunk)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

import graph_lib
import noise_lib
from model import utils as mutils
from model.ema import ExponentialMovingAverage
from rl import loader
from data import get_dataset
from rl1_post_smoke_audit import build_pools, fixed_eval8, run_nll_eval

SNAPSHOTS = {
    "3e-6": "rl1-smoke-022319/eval_snapshot.pth",
    "1e-6": "lrprobe-1e6-125140/eval_snapshot.pth",
    "1e-5": "lrprobe-1e5-135019/eval_snapshot.pth",
}
LOGGED = {  # 交叉验证锚点 (来自各 run train.log)
    "3e-6": dict(eval8_raw_150=0.1837, ema_nll_150=7.0171),
    "1e-6": dict(eval8_raw_150=0.1933, raw_nll_150=7.0424, ema_nll_150=7.0448),
    "1e-5": dict(eval8_raw_150=0.1709, raw_nll_150=7.0228, ema_nll_150=7.0290),
}


def rel_drift(a, b):
    num, den = 0.0, 0.0
    for pa, pb in zip(a, b):
        num += ((pa - pb) ** 2).sum().item()
        den += (pb ** 2).sum().item()
    return (num ** 0.5) / (den ** 0.5)


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    device = torch.device("cuda")
    torch.manual_seed(cfg.seeds.model_seed)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    valid_ds = get_dataset("wikitext103", "validation", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)

    m0_model, info = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    theta0 = [p.detach().float() for p in m0_model.parameters()]
    pools = build_pools(cfg, device)
    eval_idx = torch.tensor(list(range(8)), device=device)

    print(f"{'run':>5} | {'drift_raw':>10} {'drift_ema':>10} | "
          f"{'raw_nll':>8} {'ema_nll':>8} | {'greedy8_raw':>12} {'greedy8_ema':>12} | "
          f"{'sample8_raw':>11}")
    for name, rel in SNAPSHOTS.items():
        snap = torch.load(os.path.join(cfg.work_dir, rel), map_location="cpu",
                          weights_only=False)
        m, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
        m.load_state_dict(snap["model"], strict=False)
        m.eval()
        theta150_raw = [p.detach().float() for p in m.parameters()]
        ema = ExponentialMovingAverage(m.parameters(), decay=0.9999)
        ema.load_state_dict(snap["ema"])
        theta150_ema = [s.detach().float().to(device) for s in ema.shadow_params]

        # raw NLL (冻结 seed+1000 流) + EMA NLL
        raw_nll = run_nll_eval(m, noise, graph, valid_ds, cfg, device,
                               gen_seed=cfg.seeds.corruption_seed + 1000)
        ema.store(m.parameters())
        ema.copy_to(m.parameters())
        ema_nll = run_nll_eval(m, noise, graph, valid_ds, cfg, device,
                               gen_seed=cfg.seeds.corruption_seed + 1000)
        ema.restore(m.parameters())

        # greedy eval8: raw (primary) + ema (secondary), 同 seed 同 J
        sfn = mutils.get_score_fn(m, train=False, sampling=True)
        g_raw, _, _ = fixed_eval8(m, sfn, graph, noise, pools, eval_idx, cfg, greedy=True)
        ema.store(m.parameters())
        ema.copy_to(m.parameters())
        g_ema, _, _ = fixed_eval8(m, sfn, graph, noise, pools, eval_idx, cfg, greedy=True)
        ema.restore(m.parameters())

        # sampled eval8 raw 交叉验证
        s_raw, _, _ = fixed_eval8(m, sfn, graph, noise, pools, eval_idx, cfg)

        drift_r = rel_drift(theta150_raw, theta0)
        drift_e = rel_drift(theta150_ema, theta0)
        print(f"{name:>5} | {drift_r:10.2e} {drift_e:10.2e} | "
              f"{raw_nll:8.4f} {ema_nll:8.4f} | "
              f"{sum(g_raw) / len(g_raw):12.4f} {sum(g_ema) / len(g_ema):12.4f} | "
              f"{sum(s_raw) / len(s_raw):11.4f}")
        print(f"        greedy per-prompt raw: {[round(r, 4) for r in g_raw]}")
        print(f"        sampled per-prompt raw: {[round(r, 4) for r in s_raw]}")

        # 交叉验证
        L = LOGGED[name]
        checks = []
        if "raw_nll_150" in L:
            checks.append(("raw_nll", raw_nll, L["raw_nll_150"], 0.001))
        if "ema_nll_150" in L:
            checks.append(("ema_nll", ema_nll, L["ema_nll_150"], 0.001))
        checks.append(("eval8_raw", sum(s_raw) / len(s_raw), L["eval8_raw_150"], 0.001))
        for label, got, want, tol in checks:
            print(f"        cross-check {label}: offline={got:.4f} logged={want:.4f} "
                  f"{'OK' if abs(got - want) <= tol else 'MISMATCH!'}")
    print("\nOFFLINE_REVIEW_DONE")


if __name__ == "__main__":
    main()
