"""v2.1 treated/heldout CPI evaluator（protocol mechanism_complementary_exposure_v2_1
§9.2/§9.3/§9.4；修订 #3：训练前冻结、训练中/后禁止修改）。

对冻结 manifest 的每个样本、每个 treated/heldout pair 边逐对 delta-swap：
- treated（--map-type replicate --replicate r）：rep r 的 frozen training map 边
  （even m：matching n 对；odd m：bank 的 m 条 cycle 边，§6）
- heldout（--map-type heldout）：frozen heldout map 的 skip-2 边（m 条，§9.3）
- 状态 C = 样本的冻结 corruption realization（initial_state），pair 两端强制置 MASK；
  quartet 共享同一 σ（§4.3 同款口径）；3 forward/边；确定性 map、无 RNG（§9 修订 #3）
- per-sample = mean |δ| over 该样本的 treated/heldout 边（§9.2 主口径）；
  per-pair δ 全落盘（paired bootstrap / §9.4 scale summary 用）
- §9.4 附带：local CE（x_t 的 masked-position NLL）、scale summary（四项 logp 的
  均值/SD、δ_SD/local CE 比）、masked-token reliability 曲线（x_t per-position
  预测置信度分桶 vs 命中率）

冻结保障：本文件 + maps + manifest 三者 sha256 全校验；manifest 校验同 frozen
evaluator 口径；corruption 流沿用 manifest 冻结 realization（无新采样）。

用法:
  treated: .venv/bin/python evaluation/eval_cpi_pairs.py --map-type replicate \
      --replicate 1 --model_path DIR --ckpt checkpoint_2500.pth --weights ema \
      --tag v21-trt-H1-2500 --out results/mechanism_pilot_v21/cpi_treated_H1_2500.json
  heldout: 同上 --map-type heldout（--replicate 忽略）
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
from compatibility.posterior import clean_log_probs
from task_data.v21_policy import load_maps_verified

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 50258


def verify_manifest(manifest_path):
    digest = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
    assert digest == open(manifest_path + ".sha256").read().strip(), \
        "manifest SHA-256 不匹配! (§5A 冻结规则)"
    return digest


def load_model_for_eval(model_path, device, ckpt=None, weights="ema"):
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


def pair_edges_for_sample(m, map_type, replicate, maps):
    """该样本的 treated/heldout 边（local 位置 [0, m)）。"""
    if map_type == "replicate":
        entry = maps[f"rep{replicate}"]["even" if m % 2 == 0 else "odd"][str(m)]
        return entry["treated"]
    entry = maps["heldout"]["even" if m % 2 == 0 else "odd"][str(m)]
    return entry["pairs"]


def build_pair_entries(records, map_type, replicate, maps, device):
    """为每 (样本, 边) 建评估条目：状态 = initial_state 且 pair 两端强制 MASK。"""
    entries = []
    sample_ids = []
    for sid, r in enumerate(records):
        m = r["span_end"] - r["span_start"]
        s0 = r["span_start"]
        base = torch.tensor(r["initial_state"], device=device)
        edges = pair_edges_for_sample(m, map_type, replicate, maps)
        for (li, lj) in edges:
            pi, pj = s0 + int(li), s0 + int(lj)
            assert s0 <= pi < s0 + m and s0 <= pj < s0 + m, (sid, pi, pj, m)
            x_pair = base.clone()
            x_pair[pi] = D - 1
            x_pair[pj] = D - 1
            entries.append(dict(x_t=x_pair, i=pi, j=pj,
                                a=int(r["x0"][pi]), b=int(r["x0"][pj]),
                                sigma=float(r["sigma"])))
            sample_ids.append(sid)
    return entries, sample_ids


def run_pair_evals(score_fn, entries, chunk):
    per_pair = dict(delta=[], log_p_a_C=[], log_p_b_C=[], log_p_b_Ca=[], log_p_a_Cb=[])
    N = len(entries)
    for start in range(0, N, chunk):
        ch = entries[start:start + chunk]
        c = len(ch)
        x_t = torch.stack([e["x_t"] for e in ch])
        i = torch.tensor([e["i"] for e in ch], device=x_t.device)
        j = torch.tensor([e["j"] for e in ch], device=x_t.device)
        a = torch.tensor([e["a"] for e in ch], device=x_t.device)
        b = torch.tensor([e["b"] for e in ch], device=x_t.device)
        sigma = torch.tensor([e["sigma"] for e in ch], device=x_t.device)
        r = cpi.delta_swap(score_fn, x_t, i, j, a, b, sigma, D)
        for k in ("delta", "log_p_a_C", "log_p_b_C", "log_p_b_Ca", "log_p_a_Cb"):
            per_pair[k].extend(r[k].detach().cpu().tolist())
        del x_t
    return per_pair


def run_reliability_and_ce(score_fn, records, chunk):
    """x_t（冻结 realization）上的 masked-position local CE + reliability 曲线（§9.4）。"""
    ces, confs, hits = [], [], []
    with torch.no_grad():
        for start in range(0, len(records), chunk):
            ch = records[start:start + chunk]
            xs = torch.stack([torch.tensor(r["initial_state"]) for r in ch]).to(
                records[0]["_device"])
            sigmas = torch.tensor([r["sigma"] for r in ch], device=xs.device)
            logp = clean_log_probs(score_fn(xs, sigmas), D)
            mask = xs == D - 1
            for k, r in enumerate(ch):
                pos = mask[k].nonzero().squeeze(1)
                if pos.numel() == 0:
                    ces.append(float("nan"))
                    continue
                gold = torch.tensor(r["x0"], device=xs.device)[pos]
                lp = logp[k, pos]
                ces.append((-lp[torch.arange(pos.numel()), gold]).mean().item())
                pred = lp.argmax(-1)
                confs.extend(lp.max(-1).values.tolist())
                hits.extend((pred == gold).tolist())
            del xs, logp
    # reliability：10 桶
    confs_t = torch.tensor(confs)
    hits_t = torch.tensor(hits, dtype=torch.float32)
    order = confs_t.argsort()
    edges = torch.linspace(0, confs_t.numel(), 11).long()
    reliability = []
    for k in range(10):
        sel = order[edges[k]:edges[k + 1]]
        if sel.numel() == 0:
            continue
        reliability.append(dict(
            bucket=k, n=sel.numel(),
            conf_lo=confs_t[sel].min().item(), conf_hi=confs_t[sel].max().item(),
            mean_conf=confs_t[sel].mean().item(),
            hit_rate=hits_t[sel].mean().item()))
    return ces, reliability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=os.path.join(ROOT, "protocol/regime_a_protocol.yaml"))
    parser.add_argument("--maps_dir", default=os.path.join(ROOT, "exp_local/regime_a/mechpilot_v21_maps"))
    parser.add_argument("--map-type", required=True, choices=["replicate", "heldout"])
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--model_path", default="louaaron/sedd-small")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--tag", default="v21-pairs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    proto = OmegaConf.load(args.protocol)
    manifest_path = os.path.join(ROOT, proto.eval_manifest.file)
    digest = verify_manifest(manifest_path)
    maps = load_maps_verified(args.maps_dir)

    device = torch.device("cuda")
    model, step, source = load_model_for_eval(args.model_path, device, args.ckpt, args.weights)
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)
    print(f"模型: {source}" + (f" (step {step})" if step else ""))
    print(f"map-type={args.map_type} replicate={args.replicate} "
          f"maps sha={maps['file_sha256']['maps_manifest.json'][:16]}…")

    records = [json.loads(line) for line in open(manifest_path)]
    for r in records:
        r["_device"] = device
    print(f"manifest: n={len(records)} sha={digest[:16]}…")

    torch.cuda.reset_peak_memory_stats()
    entries, sample_ids = build_pair_entries(records, args.map_type, args.replicate, maps, device)
    per_pair = run_pair_evals(score_fn, entries, args.chunk)

    # per-sample 汇总（§9.2 主口径：per-sample = mean |δ| over pairs）
    n_samples = len(records)
    per_sample_delta = [[] for _ in range(n_samples)]
    per_sample_abs = [[] for _ in range(n_samples)]
    for sid, d in zip(sample_ids, per_pair["delta"]):
        per_sample_delta[sid].append(d)
        per_sample_abs[sid].append(abs(d))
    sm_delta = [sum(v) / len(v) for v in per_sample_delta]
    sm_abs = [sum(v) / len(v) for v in per_sample_abs]
    sm_delta_t = torch.tensor(sm_delta)
    sm_abs_t = torch.tensor(sm_abs)

    # §9.4 附带（reliability + local CE 用冻结 realization x_t，一次 forward/样本）
    ces, reliability = run_reliability_and_ce(score_fn, records, max(args.chunk, 8))
    ce_t = torch.tensor([c for c in ces if c == c])  # 去 NaN

    def sem(v):
        return v.std(unbiased=True).item() / v.numel() ** 0.5

    delta_all = torch.tensor(per_pair["delta"])
    scale = dict()
    for k in ("log_p_a_C", "log_p_b_C", "log_p_b_Ca", "log_p_a_Cb"):
        v = torch.tensor(per_pair[k])
        scale[k] = dict(mean=v.mean().item(), sd=v.std(unbiased=True).item())

    payload = dict(
        tag=args.tag,
        model_path=args.model_path,
        ckpt=args.ckpt,
        weights=args.weights,
        step=step,
        map_type=args.map_type,
        replicate=args.replicate if args.map_type == "replicate" else None,
        manifest_file=proto.eval_manifest.file,
        manifest_sha256=digest,
        maps_manifest_sha256=maps["file_sha256"]["maps_manifest.json"],
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip() or "unknown",
        protocol_version="v2.1",
        n_samples=n_samples,
        n_pairs=len(per_pair["delta"]),
        summary=dict(
            cpi_abs=sm_abs_t.mean().item(),        # §9.2 主口径（per-sample mean |δ| 的均值）
            cpi_abs_sem=sem(sm_abs_t),
            cpi_pooled_abs=delta_all.abs().mean().item(),
            cpi_rms=(delta_all ** 2).mean().sqrt().item(),
            delta_mean=sm_delta_t.mean().item(),
            delta_sem=sem(sm_delta_t),
            delta_median=sm_delta_t.median().item(),
            delta_sd=sm_delta_t.std(unbiased=True).item(),
            delta_abs_p90=torch.quantile(sm_abs_t.float(), 0.90).item(),
            delta_abs_p95=torch.quantile(sm_abs_t.float(), 0.95).item(),
            local_ce=ce_t.mean().item() if ce_t.numel() else float("nan"),
            local_ce_sem=sem(ce_t) if ce_t.numel() else float("nan"),
        ),
        scale_summary=dict(
            logp_terms=scale,
            delta_sd_over_local_ce=None,  # 在下方以 summary 值补填
        ),
        reliability_curve=reliability,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
        per_sample=dict(
            mean_delta=sm_delta,
            mean_abs_delta=sm_abs,
            n_pairs=[len(v) for v in per_sample_abs],
        ),
        per_pair=dict(
            delta=per_pair["delta"],
            log_p_a_C=per_pair["log_p_a_C"],
            log_p_b_C=per_pair["log_p_b_C"],
            log_p_b_Ca=per_pair["log_p_b_Ca"],
            log_p_a_Cb=per_pair["log_p_a_Cb"],
            sample_id=sample_ids,
        ),
    )
    # δ_SD / local CE 比（§9.4.1，在 summary 完成后补）
    payload["scale_summary"]["delta_sd_over_local_ce"] = (
        payload["summary"]["delta_sd"] / payload["summary"]["local_ce"]
        if payload["summary"]["local_ce"] == payload["summary"]["local_ce"]
        and payload["summary"]["local_ce"] > 0 else None)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    s = payload["summary"]
    print("=" * 60)
    print(f"v2.1 {args.map_type} CPI ({args.tag})")
    print(f"  CPI_abs (per-sample mean|δ|): {s['cpi_abs']:.4f} ± {s['cpi_abs_sem']:.4f} "
          f"(N={n_samples} samples, {payload['n_pairs']} pairs)")
    print(f"  CPI_RMS (pooled): {s['cpi_rms']:.4f}; pooled |δ| mean: {s['cpi_pooled_abs']:.4f}")
    print(f"  δ mean: {s['delta_mean']:+.4f} ± {s['delta_sem']:.4f}; δ_SD: {s['delta_sd']:.4f}")
    print(f"  local CE: {s['local_ce']:.4f}; δ_SD/localCE: "
          f"{payload['scale_summary']['delta_sd_over_local_ce']}")
    print(f"  峰值显存: {payload['peak_vram_gb']:.2f} GB")
    print(f"结果: {args.out}")


if __name__ == "__main__":
    main()
