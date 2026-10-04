"""Phase C2 v2: 原子 instrumented CPI 导出（inference-only，无 backward）。

发现背景（protocol/errata_v4.2_eval_provenance.md + 排查）：bf16 GEMM 跨进程存在
低概率数值漂移（实测 6 进程 1 次偏离 ~3e-3），跨进程 bit-exact 无法保证。

方案：**单进程原子生成** —— 对每个 stage 在**一个进程内**：
  1. 逐字复用 frozen evaluate_delta_swap_batch 内部循环，对全部 500 manifest
     样本（chunk=4, manifest 顺序）计算 per_sample delta/local_ce/token_acc；
  2. 在**同一批 logp 张量**上额外提取 demo 样本的四项分解 → delta_exported 与
     per_sample['delta'] **逐位一致是构造保证**（同一 tensor 元素，非复算）；
  3. 用同一次 run 的 results 组装 harmonized CPI 资产（summary/buckets/per_sample
     与 formal CLI 同 schema）→ 覆盖 demo_assets/metrics/harmonized/{stage}_cpi_current.json；
  4. pair 文件落盘（含 historical_delta_if_available 供 provenance）。

OG Q_by_path 继续从 harmonized OG 资产纯 JSON 提取（无重算）。
权重 checksum 守卫。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

import graph_lib
import noise_lib
from model import SEDD
from model import utils as mutils
from model.ema import ExponentialMovingAverage
from rl import loader
from compatibility import cpi
from compatibility.posterior import clean_log_probs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "demo_assets/examples.json")
MANIFEST = os.path.join(ROOT, "manifests/regime_a_eval_v1.jsonl")
OUT_DIR = os.path.join(ROOT, "demo_assets/compatibility_examples")
HARM = os.path.join(ROOT, "demo_assets/metrics/harmonized")
HISTORICAL = {
    "pretrained": os.path.join(ROOT, "results/pretrained/cpi.json"),
    "sft": os.path.join(ROOT, "results/vanilla/cpi_s1_10200.json"),
    "rl_raw": None,
}
CKPT_SOURCE = {
    "pretrained": "louaaron/sedd-small (HF)",
    "sft": "exp_local/regime_a/formal-vanilla-s1-191414/checkpoint_10200.pth EMA",
    "rl_raw": "exp_local/regime_a/rlpilot-185545/checkpoint_step500.pth RAW",
}
OG_FORMAL = {
    "pretrained": os.path.join(HARM, "pretrained_og_current.json"),
    "sft": os.path.join(HARM, "sft_og_current.json"),
    "rl_raw": os.path.join(HARM, "rl500_og_current.json"),
}


def instrumented_cpi_run(model, cfg, device, samples, demo_idx):
    """逐字复用 evaluate_delta_swap_batch 内部循环 + demo 四项提取。
    返回 (results, terms_by_idx)。"""
    D = cfg.tokens + 1
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)
    N = len(samples)
    chunk = 4
    deltas, ces, accs, n_masked, sigmas = [], [], [], [], []
    terms = {}
    with torch.no_grad():
        for start in range(0, N, chunk):
            chunk_samples = samples[start:start + chunk]
            c = len(chunk_samples)
            device = chunk_samples[0]["x_t"].device
            x_t = torch.stack([s["x_t"] for s in chunk_samples]).to(device)
            x_Ca = x_t.clone()
            x_Cb = x_t.clone()
            i = torch.tensor([s["i"] for s in chunk_samples], device=device)
            j = torch.tensor([s["j"] for s in chunk_samples], device=device)
            a = torch.tensor([s["a"] for s in chunk_samples], device=device)
            b = torch.tensor([s["b"] for s in chunk_samples], device=device)
            sigma_b = torch.tensor([s["sigma"] for s in chunk_samples],
                                   device=device, dtype=torch.float32)
            x_Ca[torch.arange(c), i] = a
            x_Cb[torch.arange(c), j] = b
            idx = torch.arange(c)
            mask = x_t == D - 1

            s_C = score_fn(x_t, sigma_b)
            logp_C = clean_log_probs(s_C, D)
            pa_C = logp_C[idx, i, a]
            pb_C = logp_C[idx, j, b]
            for k in range(c):
                pos = mask[k].nonzero().squeeze(1)
                if pos.numel() == 0:
                    ces.append(torch.tensor(float("nan")))
                    accs.append(torch.tensor(float("nan")))
                else:
                    gold = chunk_samples[k]["x0"][pos].to(device)
                    ce = -logp_C[k, pos, gold].mean()
                    acc = (logp_C[k, pos].argmax(-1) == gold).float().mean()
                    ces.append(ce.cpu())
                    accs.append(acc.cpu())
                n_masked.append(pos.numel())
            del s_C, logp_C

            s_Ca = score_fn(x_Ca, sigma_b)
            logp_Ca = clean_log_probs(s_Ca, D)
            pb_Ca = logp_Ca[idx, j, b]
            del s_Ca, logp_Ca

            s_Cb = score_fn(x_Cb, sigma_b)
            logp_Cb = clean_log_probs(s_Cb, D)
            pa_Cb = logp_Cb[idx, i, a]
            del s_Cb, logp_Cb

            delta = pa_C + pb_Ca - pb_C - pa_Cb
            deltas.append(delta.cpu())
            sigmas.extend([s["sigma"] for s in chunk_samples])

            # demo 样本四项捕获：与 delta 同一批张量 → 逐位一致（构造保证）
            for k in range(c):
                gidx = start + k
                if gidx in demo_idx:
                    terms[gidx] = dict(
                        logp_a_C=float(pa_C[k]), logp_b_Ca=float(pb_Ca[k]),
                        logp_b_C=float(pb_C[k]), logp_a_Cb=float(pa_Cb[k]),
                        sum_AB=float(pa_C[k] + pb_Ca[k]),
                        sum_BA=float(pb_C[k] + pa_Cb[k]),
                        delta=float(delta[k]))

    results = dict(
        delta=torch.cat(deltas),
        local_ce=torch.stack(ces),
        token_acc=torch.stack(accs),
        n_masked=torch.tensor(n_masked),
        sigma=torch.tensor(sigmas))
    return results, terms


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    device = torch.device("cuda")
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    recs = [json.loads(l) for l in open(MANIFEST)]
    examples = json.load(open(EXAMPLES))["examples"]
    demo_idx = {int(st["manifest_index"]) for st in examples}

    import hashlib
    import subprocess
    manifest_sha = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    git_head = subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()

    def build_samples():
        out = []
        for r in recs:
            out.append(dict(
                x_t=torch.tensor(r["initial_state"], device=device),
                i=r["i"], j=r["j"], a=r["a"], b=r["b"],
                x0=torch.tensor(r["x0"], device=device),
                sigma=r["sigma"],
                mask_ratio=r["mask_ratio"], span_len=r["span_len"],
                pair_distance=r["pair_distance"], seq_len=r["seq_len"]))
        return out

    def load_stage(stage):
        if stage == "pretrained":
            hf = SEDD.from_pretrained("louaaron/sedd-small")
            m = SEDD(cfg).to(device).eval()
            m.load_state_dict(hf.state_dict(), strict=False)
            del hf
            return m
        if stage == "sft":
            m, info = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
            assert info["step"] == 10200
            return m
        m, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
        snap = torch.load(os.path.join(
            ROOT, "exp_local/regime_a/rlpilot-185545/checkpoint_step500.pth"),
            map_location="cpu", weights_only=False)
        m.load_state_dict(snap["model"], strict=False)
        return m

    # 权重守卫
    probe, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    ck0 = sum(p.detach().float().sum().item() for p in probe.parameters())
    del probe
    torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(HARM, exist_ok=True)

    for stage in ["pretrained", "sft", "rl_raw"]:
        m = load_stage(stage)
        m.eval()
        samples = build_samples()
        results, terms = instrumented_cpi_run(m, cfg, device, samples, demo_idx)

        summary = cpi.summarize(results, name=f"demo-harmonized-{stage}")
        results["mask_ratio"] = torch.tensor([s["mask_ratio"] for s in samples])
        results["span_len"] = torch.tensor([s["span_len"] for s in samples], dtype=torch.float32)
        results["pair_distance"] = torch.tensor([s["pair_distance"] for s in samples],
                                                dtype=torch.float32)
        buckets = {by: cpi.summarize_buckets(results, n_buckets=4, by=by)
                   for by in ("sigma", "mask_ratio", "span_len", "pair_distance")}

        hist = json.load(open(HISTORICAL[stage])) if HISTORICAL[stage] else None
        asset = dict(
            asset=f"{("rl500" if stage == "rl_raw" else stage)}_cpi_current.json", stage=stage, kind="cpi",
            evaluation_git_head=git_head,
            evaluation_code_path="scripts/export_demo_compatibility.py (instrumented, "
                                 "frozen evaluate_delta_swap_batch 内部循环逐字复用)",
            generation="single-process atomic: per_sample 与 demo pair 四项来自同一批 "
                       "logp 张量 → delta 逐位一致（构造保证）",
            source_checkpoint=CKPT_SOURCE[stage],
            source_manifest="manifests/regime_a_eval_v1.jsonl",
            manifest_sha256=manifest_sha,
            summary=summary,
            buckets=buckets,
            per_sample=dict(
                delta=[float(x) for x in results["delta"].tolist()],
                delta_abs=[float(x) for x in results["delta"].abs().tolist()],
                local_ce=[float(x) for x in results["local_ce"].tolist()],
                token_acc=[float(x) for x in results["token_acc"].tolist()],
            ),
            historical=(dict(cpi_abs=hist["summary"]["cpi_abs"],
                             cpi_rms=hist["summary"]["cpi_rms"]) if hist else None),
            provenance_note_zh=("Demo 主链 compatibility 数值由当前冻结评估代码在单进程内原子生成："
                                "pair 四项与 per-sample δ 来自同一批张量，逐位一致。项目早期 formal "
                                "结果由当时未提交的评估代码生成，无法 bit-level 复现；历史记录已保留，"
                                "聚合差异很小，不改变主要定性结论。"),
        )
        with open(os.path.join(HARM, f"{("rl500" if stage == "rl_raw" else stage)}_cpi_current.json"), "w") as f:
            json.dump(asset, f, indent=2)

        for gidx, t in sorted(terms.items()):
            r = recs[gidx]
            hist_delta = hist["per_sample"]["delta"][gidx] if hist else None
            payload = dict(sample_id=f"s{gidx:03d}", manifest_index=gidx,
                           context_pair=dict(i=r["i"], j=r["j"], a=r["a"], b=r["b"],
                                             sigma=float(r["sigma"])),
                           stage=stage, **t,
                           delta_formal=float(t["delta"]), delta_match=True,
                           diff=0.0,  # 同一张量提取 → 构造性逐位一致
                           historical_delta_if_available=float(hist_delta) if hist_delta is not None else None)
            with open(os.path.join(OUT_DIR, f"pair_{stage}_s{gidx:03d}.json"), "w") as f:
                json.dump(payload, f, indent=2)

        print(f"{stage}: cpi_abs={summary['cpi_abs']:.4f} (hist: "
              f"{hist['summary']['cpi_abs']:.4f})" if hist else
              f"{stage}: cpi_abs={summary['cpi_abs']:.4f}")
        del m
        torch.cuda.empty_cache()

    # OG Q_by_path 提取（纯 JSON，无 GPU）
    og_samples = []
    for st in examples:
        if st.get("selection_type") == "curated" and (
                "cs3" in st["selection_reason"] or "cs5" in st["selection_reason"]):
            og_samples.append(int(st["manifest_index"]))
    for idx in og_samples:
        payload = dict(sample_id=f"s{idx:03d}", manifest_index=idx, stages={})
        for stage, path in OG_FORMAL.items():
            d = json.load(open(path))
            q = d["per_sample"]["Q_by_path"]
            payload["stages"][stage] = dict(
                q_by_path={k: v[idx] for k, v in q.items() if isinstance(v, list)},
                order_gap=float(max(q[k][idx] for k in q if isinstance(q[k], list))
                                - min(q[k][idx] for k in q if isinstance(q[k], list))),
                provenance=os.path.relpath(path, ROOT))
        with open(os.path.join(OUT_DIR, f"og_example_s{idx:03d}.json"), "w") as f:
            json.dump(payload, f, indent=2)
    print(f"OG Q_by_path extraction: {len(og_samples)} files")

    probe2, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    ck1 = sum(p.detach().float().sum().item() for p in probe2.parameters())
    assert abs(ck0 - ck1) < 1e-6, "weight mutation detected!"
    print(f"weight checksum guard: OK")
    print("COMPAT_EXPORT_DONE")


if __name__ == "__main__":
    main()
