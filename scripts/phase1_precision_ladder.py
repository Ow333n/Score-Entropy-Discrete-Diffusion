"""Phase 1.1 precision-ladder preflight（protocol phase1 §3，FROZEN 判定规则）。

32 frozen samples（均匀间隔索引，sha256 sidecar）× 2 models（step0 + v21pilot-H1@2500）
× 3 levels：

  A = frozen bf16 全链（cpi.evaluate_delta_swap_batch + summarize 原路径）
  B = frozen forward（bf16 score）+ score.float() 后 fp32 提取 + fp64 聚合
  C = 镜像 forward 全 fp32（调用相同 frozen 子模块、去 bf16 autocast）+ 同 B 提取

输出 per-level：NLL lattice（off-bf16-grid fraction）、CPI_abs/RMS/signed δ/δ_SD、
CE/NLL、VRAM、runtime；B-C engineering equivalence gate
（|ΔCPI|<0.003 且 |ΔCE|<0.01，pre-registered engineering tolerances，非 statistical
equivalence test）；B 的主要 NLL lattice 消除检查。
Level A 保真校验：与 frozen evaluator 在同子集上的 CPI_abs 一致。

用法: .venv/bin/python scripts/phase1_precision_ladder.py
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from hydra import initialize, compose

from model import SEDD
from model.ema import ExponentialMovingAverage
from model import utils as mutils
import compatibility.cpi as cpi
from compatibility.posterior import clean_log_probs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "phase1_preflight")
D = 50258
N_SUBSET = 32
CHUNK = 8

MODELS = [
    ("step0", os.path.join(ROOT, "exp_local/regime_a/mechpilot_shared"), "checkpoint_0000.pth"),
    ("H1-2500", os.path.join(ROOT, "exp_local/regime_a/v21pilot-H1-230242"), "checkpoint_2500.pth"),
]
MANIFEST = os.path.join(ROOT, "manifests/regime_a_eval_v1.jsonl")
MANIFEST_SHA = "1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3"


def fp32_forward(m, indices, sigma):
    """镜像 transformer.py forward（L275-305），去掉 blocks/output 段的 bf16 autocast。

    忠实性已由 preflight 探针验证：blocks 段包 bf16 时与 frozen forward 逐位一致。
    本函数不改任何 frozen 文件，仅以 fp32 调用相同 frozen 子模块。
    """
    x = m.vocab_embed(indices)
    c = F.silu(m.sigma_map(sigma))
    rotary_cos_sin = m.rotary_emb(x)
    for i in range(len(m.blocks)):
        x = m.blocks[i](x, rotary_cos_sin, c, seqlens=None)
    x = m.output_layer(x, c)
    if m.scale_by_sigma:
        esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
        x = x - esigm1_log - np.log(x.shape[-1] - 1)
    x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
    return x


def load_model(model_path, ckpt):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    model = SEDD(cfg).to("cuda").eval()
    loaded = torch.load(os.path.join(model_path, ckpt), map_location="cuda", weights_only=False)
    if "ema" in loaded:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
        ema.load_state_dict(loaded["ema"])
        ema.copy_to(model.parameters())
    else:
        model.load_state_dict(loaded["model"], strict=False)
    return model


def make_samples(indices):
    records = [json.loads(line) for line in open(MANIFEST)]
    samples = []
    for k in indices:
        r = records[k]
        samples.append(dict(
            x_t=torch.tensor(r["initial_state"]), i=r["i"], j=r["j"],
            a=r["a"], b=r["b"], x0=torch.tensor(r["x0"]), sigma=r["sigma"]))
    return samples


def level_A(model, samples):
    """frozen 全链：get_score_fn（bf16 内部）+ evaluate_delta_swap_batch + summarize。"""
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)
    res = cpi.evaluate_delta_swap_batch(score_fn, samples, D, chunk=CHUNK)
    summ = cpi.summarize(res, name="A")
    nlls = res["local_ce"][torch.isfinite(res["local_ce"])]
    return dict(cpi_abs=summ["cpi_abs"], cpi_rms=summ["cpi_rms"],
                delta_mean=summ["delta_mean"], delta_sd=summ["delta_sd"],
                ce=float(nlls.mean()), nll_values=nlls.float().numpy())


def level_B(model, samples):
    """frozen forward（bf16 score）→ .float() → fp32 log_softmax/δ/CE → fp64 聚合。"""
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)
    deltas, ces, nll_vals = [], [], []
    with torch.no_grad():
        for start in range(0, len(samples), CHUNK):
            ch = samples[start:start + CHUNK]
            c = len(ch)
            x_t = torch.stack([s["x_t"] for s in ch]).to("cuda")
            x0 = torch.stack([s["x0"] for s in ch]).to("cuda")
            i = torch.tensor([s["i"] for s in ch], device="cuda")
            j = torch.tensor([s["j"] for s in ch], device="cuda")
            a = torch.tensor([s["a"] for s in ch], device="cuda")
            b = torch.tensor([s["b"] for s in ch], device="cuda")
            sigma = torch.tensor([s["sigma"] for s in ch], device="cuda")
            idx = torch.arange(c)
            s_C = score_fn(x_t, sigma).float()          # bf16 score → fp32 提取
            x_Ca = x_t.clone(); x_Ca[idx, i] = a
            x_Cb = x_t.clone(); x_Cb[idx, j] = b
            s_Ca = score_fn(x_Ca, sigma).float()
            s_Cb = score_fn(x_Cb, sigma).float()
            logp_C = clean_log_probs(s_C, D)
            logp_Ca = clean_log_probs(s_Ca, D)
            logp_Cb = clean_log_probs(s_Cb, D)
            delta = (logp_C[idx, i, a] + logp_Ca[idx, j, b]
                     - logp_C[idx, j, b] - logp_Cb[idx, i, a]).cpu().double()
            deltas.append(delta)
            mask = x_t == D - 1
            for k in range(c):
                pos = mask[k].nonzero().squeeze(1)
                gold = x0[k, pos]
                ces.append((-logp_C[k, pos, gold]).mean().cpu().double().item())
                nll_vals.extend((-logp_C[k, pos, gold]).cpu().tolist())
            del s_C, s_Ca, s_Cb, logp_C, logp_Ca, logp_Cb
    delta = torch.cat(deltas)
    ce = torch.tensor(ces, dtype=torch.float64)
    return dict(cpi_abs=delta.abs().mean().item(), cpi_rms=(delta ** 2).mean().sqrt().item(),
                delta_mean=delta.mean().item(), delta_sd=delta.std(unbiased=True).item(),
                ce=ce.mean().item(), nll_values=np.array(nll_vals))


def level_C(model, samples):
    """镜像 fp32 forward → 全 fp32 提取 → fp64 聚合。"""
    deltas, ces = [], []
    nll_vals = []
    with torch.no_grad():
        for start in range(0, len(samples), CHUNK):
            ch = samples[start:start + CHUNK]
            c = len(ch)
            dev = "cuda"
            x_t = torch.stack([s["x_t"] for s in ch]).to(dev)
            x0 = torch.stack([s["x0"] for s in ch]).to(dev)
            i = torch.tensor([s["i"] for s in ch], device=dev)
            j = torch.tensor([s["j"] for s in ch], device=dev)
            a = torch.tensor([s["a"] for s in ch], device=dev)
            b = torch.tensor([s["b"] for s in ch], device=dev)
            sigma = torch.tensor([s["sigma"] for s in ch], device=dev)
            idx = torch.arange(c)
            s_C = fp32_forward(model, x_t, sigma)
            x_Ca = x_t.clone(); x_Ca[idx, i] = a
            x_Cb = x_t.clone(); x_Cb[idx, j] = b
            s_Ca = fp32_forward(model, x_Ca, sigma)
            s_Cb = fp32_forward(model, x_Cb, sigma)
            logp_C = clean_log_probs(s_C, D)
            logp_Ca = clean_log_probs(s_Ca, D)
            logp_Cb = clean_log_probs(s_Cb, D)
            delta = (logp_C[idx, i, a] + logp_Ca[idx, j, b]
                     - logp_C[idx, j, b] - logp_Cb[idx, i, a]).cpu().double()
            deltas.append(delta)
            mask = x_t == D - 1
            for k in range(c):
                pos = mask[k].nonzero().squeeze(1)
                gold = x0[k, pos]
                ces.append((-logp_C[k, pos, gold]).mean().cpu().double().item())
                nll_vals.extend((-logp_C[k, pos, gold]).cpu().tolist())
            del s_C, s_Ca, s_Cb, logp_C, logp_Ca, logp_Cb
    delta = torch.cat(deltas)
    ce = torch.tensor(ces, dtype=torch.float64)
    return dict(cpi_abs=delta.abs().mean().item(), cpi_rms=(delta ** 2).mean().sqrt().item(),
                delta_mean=delta.mean().item(), delta_sd=delta.std(unbiased=True).item(),
                ce=ce.mean().item(), nll_values=np.array(nll_vals))


def lattice_stats(nll_values):
    v = torch.tensor(nll_values, dtype=torch.float64)
    off_bf16 = float((v.to(torch.bfloat16).double() != v).float().mean())
    off_fp16 = float((v.half().double() != v).float().mean())
    return dict(n=len(nll_values), off_bf16_grid_fraction=off_bf16,
                off_fp16_grid_fraction=off_fp16)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # 冻结子集（均匀间隔；sha256 sidecar）
    indices = [round(k * 499 / (N_SUBSET - 1)) for k in range(N_SUBSET)]
    assert len(set(indices)) == N_SUBSET
    sub_path = os.path.join(OUT_DIR, "ladder_subset_indices.json")
    with open(sub_path, "w") as f:
        json.dump(dict(rule="evenly_spaced_round", n=N_SUBSET, manifest_n=500,
                       manifest_sha256=MANIFEST_SHA, indices=indices), f, indent=2)
    digest = hashlib.sha256(open(sub_path, "rb").read()).hexdigest()
    with open(sub_path + ".sha256", "w") as f:
        f.write(digest + "\n")

    digest_man = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    assert digest_man == MANIFEST_SHA, "manifest hash 变化，硬停"
    samples = make_samples(indices)
    samples = [{**s, "x_t": s["x_t"].to("cuda"), "x0": s["x0"].to("cuda")} for s in samples]

    report = dict(subset=dict(indices=indices, sha256=digest, manifest_sha256=MANIFEST_SHA),
                  levels={}, gate={})
    for name, path, ckpt in MODELS:
        print(f"\n=== model {name} ===")
        model = load_model(path, ckpt)
        rows = {}
        for lvl, fn in (("A", level_A), ("B", level_B), ("C", level_C)):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            r = fn(model, samples)
            r["runtime_s"] = time.time() - t0
            r["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
            r["lattice"] = lattice_stats(r.pop("nll_values"))
            rows[lvl] = r
            print(f"  {lvl}: CPI_abs={r['cpi_abs']:.4f} RMS={r['cpi_rms']:.4f} "
                  f"δ̄={r['delta_mean']:+.4f} δ_SD={r['delta_sd']:.4f} CE={r['ce']:.4f} "
                  f"off_bf16={r['lattice']['off_bf16_grid_fraction']:.1%} "
                  f"VRAM={r['vram_gb']:.2f}GB {r['runtime_s']:.1f}s")
        report["levels"][name] = rows
        # B-C engineering equivalence gate
        d_cpi = abs(rows["B"]["cpi_abs"] - rows["C"]["cpi_abs"])
        d_ce = abs(rows["B"]["ce"] - rows["C"]["ce"])
        b_lattice_ok = rows["B"]["lattice"]["off_bf16_grid_fraction"] > 0.9
        gate = dict(d_cpi_abs=d_cpi, d_ce_abs=d_ce,
                    cpi_tol_ok=d_cpi < 0.003, ce_tol_ok=d_ce < 0.01,
                    b_lattice_removed=b_lattice_ok,
                    b_gate_pass=(d_cpi < 0.003 and d_ce < 0.01 and b_lattice_ok))
        report["gate"][name] = gate
        print(f"  gate: |ΔCPI_B−C|={d_cpi:.5f} |ΔCE_B−C|={d_ce:.5f} "
              f"tols={(d_cpi<0.003, d_ce<0.01)} B-lattice-removed={b_lattice_ok} "
              f"→ {'B PASS' if gate['b_gate_pass'] else '→ C'}")
        del model
        torch.cuda.empty_cache()

    # Level A 保真校验（frozen 路径同子集）
    print("\n=== A 保真（frozen 路径 CPI on 子集）===")
    model = load_model(MODELS[0][1], MODELS[0][2])
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)
    res = cpi.evaluate_delta_swap_batch(score_fn, samples, D, chunk=CHUNK)
    frozen_cpi = cpi.summarize(res)["cpi_abs"]
    a_cpi = report["levels"]["step0"]["A"]["cpi_abs"]
    print(f"  frozen evaluator CPI={frozen_cpi:.4f} vs ladder-A CPI={a_cpi:.4f} "
          f"diff={abs(frozen_cpi - a_cpi):.2e}")
    report["a_fidelity"] = dict(frozen_cpi=frozen_cpi, ladder_a_cpi=a_cpi,
                                diff=abs(frozen_cpi - a_cpi))
    del model
    torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, "precision_ladder_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {os.path.join(OUT_DIR, 'precision_ladder_report.json')}")


if __name__ == "__main__":
    main()
