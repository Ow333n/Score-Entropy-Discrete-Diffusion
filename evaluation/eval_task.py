"""G1 任务可学性评估 (protocol §9.4): 在冻结 manifest 上比较 pretrained vs SFT checkpoint。

指标:
  1. masked-token NLL (= local CE on span-masked 位置, 训练目标本身)
  2. token accuracy (span-masked 位置上 argmax p_hat == gold)
  3. span 贪心解码: 从 manifest 初始状态出发, l2r 严格逐 token 贪心揭示,
     报告每位置命中率 + span 精确匹配率 (10-50 token span 的 exact match 预期极低,
     仅作辅助 learnability 信号, 不单独作 gate)

模型源: --model_path 支持
  - HF id (louaaron/sedd-small) → pretrained 基线
  - 本地训练目录 (含 checkpoints-meta/checkpoint.pth) → SFT checkpoint, 用 EMA 权重

用法:
  .venv/bin/python evaluation/eval_task.py --model_path exp_local/regime_a/pilot-XXXXXX \
      --tag pilot --out results/vanilla/g1_pilot.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose
from omegaconf import OmegaConf

from model import SEDD
from model.ema import ExponentialMovingAverage
from model import utils as mutils
import compatibility.cpi as cpi
import compatibility.order_gap as og
from training.vanilla import init_model_from_pretrained

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def verify_manifest(manifest_path):
    digest = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
    assert digest == open(manifest_path + ".sha256").read().strip(), \
        "manifest SHA-256 不匹配! (§5A 冻结规则)"
    return digest


def load_model_for_eval(model_path, device, ckpt=None, weights="ema"):
    """HF id → from_pretrained 权重; 本地目录 → 训练 checkpoint 的 EMA/raw 权重。

    ckpt: 指定 analysis checkpoint 文件名 (如 checkpoint_10200.pth),
          默认 checkpoints-meta/checkpoint.pth。
    weights: "ema" (shadow, primary) | "raw" (训练参数, v4.2 双权重评估)。
    """
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    model = SEDD(cfg).to(device).eval()

    ckpt_path = os.path.join(model_path, ckpt) if ckpt else os.path.join(
        model_path, "checkpoints-meta", "checkpoint.pth")
    if os.path.exists(ckpt_path):
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        if weights == "ema":
            # EMA.state_dict() 格式是 {decay, num_updates, shadow_params: [tensor...]},
            # 不是按参数名索引 —— 必须走 EMA.load_state_dict + copy_to, 不能 model.load_state_dict
            # (后者全 key 不匹配会静默空载 → 输出层零初始化 → 均匀分布, 实测 NLL=ln(50257))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=os.path.join(ROOT, "protocol/regime_a_protocol.yaml"))
    parser.add_argument("--model_path", default="louaaron/sedd-small")
    parser.add_argument("--ckpt", default=None,
                        help="analysis checkpoint 文件名, 如 checkpoint_10200.pth")
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--n_greedy", type=int, default=100,
                        help="span 贪心解码最多评估多少个 manifest 样本 (每个 m 步 forward)")
    parser.add_argument("--tag", default="pretrained")
    parser.add_argument("--out", default=os.path.join(ROOT, "results/pretrained/g1_task.json"))
    args = parser.parse_args()

    proto = OmegaConf.load(args.protocol)
    manifest_path = os.path.join(ROOT, proto.eval_manifest.file)
    digest = verify_manifest(manifest_path)
    records = [json.loads(line) for line in open(manifest_path)]

    device = torch.device("cuda")
    model, step, source = load_model_for_eval(args.model_path, device, args.ckpt, args.weights)
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)
    D = 50258
    print(f"模型: {source}" + (f" (step {step})" if step else ""))

    # 指标 1/2: masked NLL + token acc (与 eval_cpi 同协议, 复用 evaluate_delta_swap_batch)
    samples = [dict(x_t=torch.tensor(r["initial_state"], device=device),
                    i=r["i"], j=r["j"], a=r["a"], b=r["b"],
                    x0=torch.tensor(r["x0"], device=device), sigma=r["sigma"])
               for r in records]
    results = cpi.evaluate_delta_swap_batch(score_fn, samples, D, chunk=args.chunk)
    summary = cpi.summarize(results, name=args.tag)

    # 指标 3: span 贪心解码 (l2r 严格逐 token, 从 manifest 初始状态出发)
    n_greedy = min(args.n_greedy, len(records))
    hit_rate, exact_match, n = 0.0, 0, 0
    with torch.no_grad():
        for r in records[:n_greedy]:
            x = torch.tensor(r["initial_state"], device=device)[None]      # [1, L]
            gold = torch.tensor(r["x0"], device=device)[None]
            path = r["paths"]["l2r"]
            x_out, info = og.strict_reveal(score_fn, x, gold, r["sigma"], D,
                                           mode="sampled", sample="greedy", fixed_path=torch.tensor(path)[:, None])
            span = range(r["span_start"], r["span_end"])
            hits = (x_out[0, span] == gold[0, span]).sum().item()
            hit_rate += hits / len(span)
            exact_match += float(hits == len(span))
            n += 1
    hit_rate /= n
    exact_match /= n

    print("=" * 60)
    print(f"G1 任务可学性 ({args.tag}, {source})")
    print(f"  masked-token NLL:  {summary['local_ce']:.4f} ± {summary['local_ce_sem']:.4f}")
    print(f"  token accuracy:    {summary['token_acc']:.4f} ± {summary['token_acc_sem']:.4f}")
    print(f"  span 贪心 per-pos 命中率 (l2r, n={n}): {hit_rate:.4f}")
    print(f"  span 精确匹配率:   {exact_match:.4f} (长 span 预期极低, 仅辅助)")
    print("=" * 60)

    payload = dict(
        tag=args.tag, model_path=args.model_path, source=source, step=step,
        protocol_version="v4.2",
        manifest_file=proto.eval_manifest.file,
        manifest_sha256=digest,
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip() or "unknown",
        masked_nll=summary["local_ce"],
        masked_nll_sem=summary["local_ce_sem"],
        token_acc=summary["token_acc"],
        token_acc_sem=summary["token_acc_sem"],
        greedy_per_pos_hit_rate=hit_rate,
        greedy_span_exact_match=exact_match,
        n_greedy=n,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
