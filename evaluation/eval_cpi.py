"""Regime A 正式 CPI 评估 (protocol §5): 读取冻结 manifest, 输出 §5.4 全指标。

所有 checkpoint 必须使用同一 manifest + 同一 pair 样本 (paired 对比, §3.17)。
本脚本只读 manifest, 不重新采样。

用法:
  .venv/bin/python evaluation/eval_cpi.py --out results/pretrained/cpi.json \
      [--tag pretrained]
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
import compatibility.cpi as cpi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_model_for_eval(model_path, device, ckpt=None, weights="ema"):
    """HF id → from_pretrained; 本地目录 → checkpoint 的 EMA/raw 权重。

    注意: EMA.state_dict() 是 {decay, num_updates, shadow_params:[tensor...]} 非按
    参数名索引, 必须走 ema.load_state_dict + copy_to (直接 model.load_state_dict 会
    静默空载 → NLL=ln(50257) 的均匀分布签名, Day 4 踩过)。
    ckpt: analysis checkpoint 文件名 (checkpoint_<step>.pth, 同一格式)。
    weights: "ema" | "raw" (v4.2 双权重评估)。
    """
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
    return model, None, None, step, source


def verify_manifest(manifest_path):
    digest = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
    sidecar = manifest_path + ".sha256"
    expected = open(sidecar).read().strip()
    assert digest == expected, f"manifest SHA-256 不匹配! 文件已被改动, 按 §5A 冻结规则拒绝评估"
    return digest


def load_samples(manifest_path):
    records = [json.loads(line) for line in open(manifest_path)]
    samples = []
    for r in records:
        samples.append(dict(
            x_t=torch.tensor(r["initial_state"]),
            i=r["i"], j=r["j"], a=r["a"], b=r["b"],
            x0=torch.tensor(r["x0"]),
            sigma=r["sigma"],
            # 分桶字段 (§5.4)
            mask_ratio=r["mask_ratio"],
            span_len=r["span_len"],
            pair_distance=r["pair_distance"],
            seq_len=r["seq_len"],
        ))
    return samples, records


def env_info():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return dict(
        git_commit=git.stdout.strip() or "unknown",
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        protocol_version="v4.2",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=os.path.join(ROOT, "protocol/regime_a_protocol.yaml"))
    parser.add_argument("--model_path", default="louaaron/sedd-small")
    parser.add_argument("--ckpt", default=None,
                        help="analysis checkpoint 文件名, 如 checkpoint_10200.pth")
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--tag", default="pretrained")
    parser.add_argument("--out", default=os.path.join(ROOT, "results/pretrained/cpi.json"))
    args = parser.parse_args()

    proto = OmegaConf.load(args.protocol)
    manifest_path = os.path.join(ROOT, proto.eval_manifest.file)
    digest = verify_manifest(manifest_path)

    device = torch.device("cuda")
    model, _, _, step, source = _load_model_for_eval(args.model_path, device, args.ckpt, args.weights)
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)
    print(f"模型: {source}" + (f" (step {step})" if step else ""))

    samples, records = load_samples(manifest_path)
    samples = [{**s, "x_t": s["x_t"].to(device), "x0": s["x0"].to(device)} for s in samples]
    print(f"manifest: {manifest_path} (n={len(samples)}, sha={digest[:16]}...)")

    torch.cuda.reset_peak_memory_stats()
    results = cpi.evaluate_delta_swap_batch(score_fn, samples, D=50258, chunk=args.chunk)
    summary = cpi.summarize(results, name=args.tag)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # §5.4 分桶: sigma / mask_ratio / span_len / pair_distance
    results["mask_ratio"] = torch.tensor([s["mask_ratio"] for s in samples])
    results["span_len"] = torch.tensor([s["span_len"] for s in samples], dtype=torch.float32)
    results["pair_distance"] = torch.tensor([s["pair_distance"] for s in samples], dtype=torch.float32)
    buckets = {by: cpi.summarize_buckets(results, n_buckets=4, by=by)
               for by in ("sigma", "mask_ratio", "span_len", "pair_distance")}

    print("\n" + "=" * 60)
    print(f"Regime A CPI ({args.tag}, protocol v4.1, {args.model_path})")
    print(f"  CPI_abs (E|delta|): {summary['cpi_abs']:.4f} ± {summary['cpi_abs_sem']:.4f} (SEM, N={summary['n_samples']})")
    print(f"  CPI_RMS:             {summary['cpi_rms']:.4f}")
    print(f"  delta 均值:          {summary['delta_mean']:+.4f} ± {summary['delta_sem']:.4f}; median {summary['delta_median']:+.4f}")
    print(f"  |delta| p90/p99:     {summary['delta_abs_p90']:.4f} / {summary['delta_abs_p99']:.4f}")
    print(f"  delta SD:            {summary['delta_sd']:.4f} (SD(CPI_pre) 效应量基线, §5.4)")
    print(f"  local CE:            {summary['local_ce']:.4f} ± {summary['local_ce_sem']:.4f}")
    print(f"  token acc (mask):    {summary['token_acc']:.4f} ± {summary['token_acc_sem']:.4f}")
    print(f"  峰值显存:            {peak_gb:.2f} GB")
    for by, rows in buckets.items():
        print(f"\n按 {by} 分桶 (4 桶):")
        for r in rows:
            print(f"  [{r['lo']:.3f}, {r['hi']:.3f}] n={r['n']:4d}: CPI_abs={r['cpi_abs']:.4f}, delta_mean={r['delta_mean']:+.4f}")

    payload = dict(
        tag=args.tag, model_path=args.model_path,
        ckpt=args.ckpt,
        manifest_file=proto.eval_manifest.file,
        manifest_sha256=digest,
        **env_info(),
        summary=summary,
        buckets=buckets,
        peak_vram_gb=peak_gb,
        # per-sample (Stage-4 paired bootstrap 用, §3.17)
        per_sample=dict(
            delta=[float(x) for x in results["delta"].tolist()],
            delta_abs=[float(x) for x in results["delta"].abs().tolist()],
            local_ce=[float(x) for x in results["local_ce"].tolist()],
            token_acc=[float(x) for x in results["token_acc"].tolist()],
        ),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n结果已保存: {args.out}")
    print("注意: 后续 checkpoint 的 CPI 对比必须 paired (同一 manifest), 见 protocol §3.17。")


if __name__ == "__main__":
    main()
