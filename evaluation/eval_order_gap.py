"""Regime A Teacher-Forced OrderGap 评估 (protocol §6 / §3.16)。

对冻结 manifest 的每个样本、每条固定路径 (l2r/r2l/3 random/confidence) 计算
Q_pi(x) = sum log p_theta(gold token | 已揭示前缀), 然后:
  OrderGap_raw  = max_pi Q - min_pi Q
  OrderGap_per_token = OrderGap_raw / m   (nats / revealed token, §3.16)
  Var_pi(Q)
所有路径来自 manifest (§3.10), 评估不重新生成路径。

用法:
  .venv/bin/python evaluation/eval_order_gap.py --out results/pretrained/order_gap.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from hydra import initialize, compose
from model import SEDD
from model.ema import ExponentialMovingAverage
from model import utils as mutils
import compatibility.order_gap as og

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_model_for_eval(model_path, device, ckpt=None, weights="ema"):
    """HF id → from_pretrained; 本地目录 → checkpoint 的 EMA/raw 权重 (同 eval_cpi)。"""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    model = SEDD(cfg).to(device).eval()
    ckpt_path = os.path.join(model_path, ckpt) if ckpt else os.path.join(
        model_path, "checkpoints-meta", "checkpoint.pth")
    if os.path.exists(ckpt_path):
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        if weights == "ema":
            ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
            ema.load_state_dict(loaded["ema"])
            ema.copy_to(model.parameters())
        elif weights == "raw":
            model.load_state_dict(loaded["model"], strict=False)
        else:
            raise ValueError(f"weights 必须是 ema|raw, 得到 {weights}")
        step = loaded["step"]
        source = f"local-{weights} ({model_path})"
    else:
        hf = SEDD.from_pretrained(model_path)
        model.load_state_dict(hf.state_dict(), strict=False)
        del hf
        step = None
        source = model_path
    return model, step, source


def verify_manifest(manifest_path):
    digest = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
    assert digest == open(manifest_path + ".sha256").read().strip(), \
        "manifest SHA-256 不匹配! (§5A 冻结规则)"
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=os.path.join(ROOT, "protocol/regime_a_protocol.yaml"))
    parser.add_argument("--model_path", default="louaaron/sedd-small")
    parser.add_argument("--ckpt", default=None,
                        help="analysis checkpoint 文件名, 如 checkpoint_10200.pth")
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--adaptive", action="store_true",
                        help="§6.4: 用本 checkpoint 自己的置信路径替换 manifest 的 pretrained 置信路径")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--tag", default="pretrained")
    parser.add_argument("--out", default=os.path.join(ROOT, "results/pretrained/order_gap.json"))
    args = parser.parse_args()

    proto = OmegaConf.load(args.protocol)
    manifest_path = os.path.join(ROOT, proto.eval_manifest.file)
    digest = verify_manifest(manifest_path)
    path_types = list(proto.eval_manifest.path_types)

    device = torch.device("cuda")
    model, step, source = _load_model_for_eval(args.model_path, device, args.ckpt, args.weights)
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)
    print(f"模型: {source}" + (f" (step {step})" if step else ""))

    records = [json.loads(line) for line in open(manifest_path)]
    print(f"manifest: {manifest_path} (n={len(records)}, sha={digest[:16]}...)")

    # §6.4 adaptive 路径: 本 checkpoint 自己的置信路径 (TF 揭示, §3.8 tie→最小 index)
    if args.adaptive:
        print("生成 adaptive confidence 路径 (§6.4)...")
        groups = {}
        for r in records:
            groups.setdefault(r["sigma"], []).append(r)
        tie_total, tie_steps = 0, 0
        done, total = 0, len(records)
        for sigma, rs in groups.items():
            for start in range(0, len(rs), 8):
                chunk = rs[start:start + 8]
                xs = torch.tensor([r["initial_state"] for r in chunk], device=device)
                gold = torch.tensor([r["x0"] for r in chunk], device=device)
                _, info = og.strict_reveal(score_fn, xs, gold, sigma, 50258,
                                           order="confidence", mode="teacher_forced")
                for local, r in enumerate(chunk):
                    r["paths"]["confidence_adaptive"] = info["pos_history"][:, local].tolist()
                tie_total += info["tie_count"]
                tie_steps += info["steps"]
                done += len(chunk)
                print(f"  adaptive 路径生成进度: {done}/{total}", flush=True)
        print(f"  adaptive 路径完成: tie_fraction={tie_total / max(tie_steps, 1):.4f} (§3.8)")
        path_types = path_types + ["confidence_adaptive"]

    torch.cuda.reset_peak_memory_stats()
    Q_by_path = {}
    for ptype in path_types:
        samples = [dict(x_t=torch.tensor(r["initial_state"], device=device),
                        x0=torch.tensor(r["x0"], device=device),
                        sigma=r["sigma"],
                        path=r["paths"][ptype])
                   for r in records]
        Q = og.evaluate_path_Qs(score_fn, samples, D=50258, chunk=args.chunk)
        Q_by_path[ptype] = Q.cpu()
        print(f"  {ptype:12s}: Q mean {Q.mean().item():8.3f} ± {Q.std().item() / Q.numel() ** 0.5:.3f} (nats)")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # OrderGap (§3.16): 主比较用固定 6 路径 (与 pretrained 基线同口径, manifest §3.10);
    # adaptive 模式额外算含 adaptive 路径的版本 (报告用, 不与 pretrained 直接比)
    fixed_types = [p for p in path_types if p != "confidence_adaptive"]
    Qs_fixed = torch.stack([Q_by_path[p] for p in fixed_types], dim=0)
    m = torch.tensor([len(r["paths"]["l2r"]) for r in records], dtype=torch.float32)
    order_gap = Qs_fixed.max(0).values - Qs_fixed.min(0).values       # [N]
    og_per_token = order_gap / m
    var_pi = Qs_fixed.var(0, unbiased=True)                           # [N]

    order_gap_incl_adaptive = None
    if args.adaptive:
        Qs_all = torch.stack([Q_by_path[p] for p in path_types], dim=0)
        order_gap_incl_adaptive = (Qs_all.max(0).values - Qs_all.min(0).values).cpu()

    def sem(v):
        return v.std(unbiased=True).item() / v.numel() ** 0.5

    summary = dict(
        tag=args.tag,
        n_samples=len(records),
        order_gap_raw=order_gap.mean().item(),
        order_gap_raw_sem=sem(order_gap),
        order_gap_per_token=og_per_token.mean().item(),
        order_gap_per_token_sem=sem(og_per_token),
        var_pi_mean=var_pi.mean().item(),
        var_pi_sem=sem(var_pi),
        mean_m=m.mean().item(),
        Q_by_path={p: dict(mean=Q_by_path[p].mean().item(), sem=sem(Q_by_path[p]))
                   for p in path_types},
        peak_vram_gb=peak_gb,
        protocol_version="v4.2",
        manifest_file=proto.eval_manifest.file,
        manifest_sha256=digest,
        adaptive=args.adaptive,
        ckpt=args.ckpt,
        # per-sample (Stage-4 paired bootstrap 用, §3.17)
        # order_gap_raw = 固定 6 路径 (与 pretrained 同口径); incl_adaptive 仅报告
        per_sample=dict(
            order_gap_raw=[float(x) for x in order_gap.tolist()],
            order_gap_per_token=[float(x) for x in og_per_token.tolist()],
            order_gap_raw_incl_adaptive=(
                [float(x) for x in order_gap_incl_adaptive.tolist()]
                if order_gap_incl_adaptive is not None else None),
            Q_by_path={p: [float(x) for x in Q_by_path[p].tolist()] for p in path_types},
        ),
    )

    print("\n" + "=" * 60)
    print(f"Regime A Teacher-Forced OrderGap ({args.tag})")
    print(f"  OrderGap_raw:       {summary['order_gap_raw']:.4f} ± {summary['order_gap_raw_sem']:.4f} nats")
    print(f"  OrderGap_per_token: {summary['order_gap_per_token']:.4f} ± {summary['order_gap_per_token_sem']:.4f} nats/revealed-token")
    print(f"  Var_pi(Q):          {summary['var_pi_mean']:.4f} ± {summary['var_pi_sem']:.4f}")
    print(f"  平均 m (|M_0|):     {summary['mean_m']:.1f}; 峰值显存 {peak_gb:.2f} GB")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
