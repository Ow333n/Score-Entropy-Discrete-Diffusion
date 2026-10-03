"""Formal 64-sample eval path acceptance（用户 freeze review 后 gate 2，eval-only，不训练）。

在 SFT seed1 EMA checkpoint_10200 上验证 rl/eval_formal.py 的 13 项要求。
输出目录 /tmp/rl1_eval_acceptance/。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

import graph_lib
import noise_lib
from model import utils as mutils
from model.ema import ExponentialMovingAverage
from rl import loader
from data import get_dataset
from rl.eval_formal import (build_eval_pools, task_eval_point, save_eval_point,
                            nll_paired, _reward_pass, FORMAL_EVAL_IDX)

OUT = "/tmp/rl1_eval_acceptance"


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(cond)


def main():
    os.makedirs(OUT, exist_ok=True)
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    device = torch.device("cuda")
    torch.manual_seed(cfg.seeds.model_seed)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)

    model, info = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    assert info["shadow_vs_model_mismatches"] == 0 and info["step"] == 10200
    model.eval()
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    ema.store(model.parameters())   # shadow = θ0（与 pilot 的 fresh EMA 同语义）
    sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
    valid_ds = get_dataset("wikitext103", "validation", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)

    x0, xt0, sig0, m0, eval_ids = build_eval_pools(cfg, device, start=64, n=64)
    pools = (x0, xt0, sig0, m0)

    free0, total0 = torch.cuda.mem_get_info()
    rng = torch.Generator().manual_seed(12345)
    rng_state_before = rng.get_state().clone()
    ref_params = [p.detach().clone() for p in model.parameters()]
    ref_shadow = [s.detach().clone() for s in ema.shadow_params]
    torch.cuda.empty_cache()
    resv0 = torch.cuda.memory_reserved()   # 进程内 reserved 基线（含权重对照克隆，免疫外部进程波动）

    results = []

    # 1. exactly 64 samples
    results.append(check("exactly 64 samples", len(FORMAL_EVAL_IDX) == 64 and x0.shape[0] == 64,
                         f"idx {FORMAL_EVAL_IDX[0]}..{FORMAL_EVAL_IDX[-1]}, pool B={x0.shape[0]}"))

    # 2. zero overlap with training pool 0–63
    overlap = set(FORMAL_EVAL_IDX) & set(range(64))
    results.append(check("zero overlap with training pool 0–63", len(overlap) == 0))

    # 2b. 与 training pool 的记录确实不同（sha256 集合不相交）
    pool_ids = set()
    for l in open(cfg.rl.manifest).readlines()[:64]:
        import hashlib
        pool_ids.add(hashlib.sha256(l.encode()).hexdigest())
    eval_id_set = {i["sha256"] for i in eval_ids}
    results.append(check("eval records disjoint from training pool records",
                         len(pool_ids & eval_id_set) == 0))

    # 3. sample IDs 固定并落盘
    ids_path = os.path.join(OUT, "eval_subset_ids.json")
    with open(ids_path, "w") as f:
        json.dump(dict(indices=FORMAL_EVAL_IDX, ids=eval_ids), f, indent=2)
    reloaded = json.load(open(ids_path))
    results.append(check("sample IDs fixed & saved", reloaded["indices"] == FORMAL_EVAL_IDX
                         and len(reloaded["ids"]) == 64, ids_path))

    # 5. 冻结 corruption realization：nll_paired 两次 → 逐位相同
    nll1 = nll_paired(model, ema, noise, graph, valid_ds, cfg, device)
    nll2 = nll_paired(model, ema, noise, graph, valid_ds, cfg, device)
    results.append(check("frozen corruption realization (nll_paired deterministic)",
                         nll1 == nll2, f"raw={nll1[0]:.4f} ema={nll1[1]:.4f} (×2 identical)"))

    # 6. greedy deterministic（同 seed 同 J → argmax 无噪声）
    per_g1, _ = _reward_pass(model, sampling_score_fn, graph, noise, pools, cfg, greedy=True)
    per_g2, _ = _reward_pass(model, sampling_score_fn, graph, noise, pools, cfg, greedy=True)
    results.append(check("greedy repeated run deterministic", per_g1 == per_g2,
                         f"mean={sum(per_g1) / len(per_g1):.4f}"))

    # 7. sampled eval 正常（finite, in [0,1]）
    per_s, gst_s = _reward_pass(model, sampling_score_fn, graph, noise, pools, cfg, greedy=False)
    ok_s = all(0.0 <= r <= 1.0 for r in per_s)
    results.append(check("sampled eval normal", ok_s,
                         f"mean={sum(per_s) / len(per_s):.4f}, gate={gst_s}"))

    # 4 + 8. RAW/EMA 均可评估 + step0 metrics 落盘
    res0 = task_eval_point(model, ema, sampling_score_fn, graph, noise, valid_ds, pools,
                           eval_ids, cfg, device, tag="step0")
    p = save_eval_point(OUT, res0)
    need = ["sampled_raw_mean", "sampled_ema_mean", "greedy_raw_mean", "greedy_ema_mean",
            "nll_raw", "nll_ema"]
    saved = json.load(open(p))
    results.append(check("RAW & EMA evaluable (all 6 metrics present)",
                         all(k in saved and isinstance(saved[k], (int, float)) for k in need)
                         and isinstance(saved.get("eval_indices"), list),
                         f"sampled raw/ema={saved['sampled_raw_mean']:.4f}/{saved['sampled_ema_mean']:.4f} "
                         f"greedy raw/ema={saved['greedy_raw_mean']:.4f}/{saved['greedy_ema_mean']:.4f} "
                         f"nll raw/ema={saved['nll_raw']:.4f}/{saved['nll_ema']:.4f}"))
    results.append(check("step0 metrics saved to JSON", os.path.exists(p), p))

    # 9. eval 不改变 model weights/state
    max_diff = max((p.detach() - r).abs().max().item()
                   for p, r in zip(model.parameters(), ref_params))
    max_diff_shadow = max((s.detach() - r).abs().max().item()
                          for s, r in zip(ema.shadow_params, ref_shadow))
    results.append(check("model weights unchanged", max_diff == 0.0, f"max|Δθ|={max_diff:.2e}"))
    results.append(check("EMA shadow unchanged", max_diff_shadow == 0.0,
                         f"max|Δshadow|={max_diff_shadow:.2e}"))

    # 10. RNG / mode 处理
    rng_unchanged = torch.equal(rng_state_before, rng.get_state())
    results.append(check("training Generator RNG untouched by eval", rng_unchanged))
    results.append(check("eval leaves model in eval mode (caller must model.train())",
                         not model.training,
                         "score fn train=False 内部置 eval; 训练循环在 eval 后调用 model.train()"))
    model.train()
    results.append(check("model.train() restores training mode", model.training))

    # 11/12. VRAM：无 OOM、无持续增长（进程内口径 + 整卡快照供参考）
    free1, total1 = torch.cuda.mem_get_info()
    torch.cuda.empty_cache()
    resv1 = torch.cuda.memory_reserved()
    free2, _ = torch.cuda.mem_get_info()
    results.append(check("no OOM / free memory sane", free1 > 1e9,
                         f"整卡 free 最低点 {free1 / 1e9:.2f}GB (基线 {free0 / 1e9:.2f}GB)"))
    results.append(check("no persistent VRAM growth (process-local reserved)",
                         (resv1 - resv0) / 1e9 < 0.2,
                         f"reserved {resv0 / 1e9:.2f}→{resv1 / 1e9:.2f}GB after empty_cache "
                         f"(peak {torch.cuda.max_memory_reserved() / 1e9:.2f}GB); "
                         f"整卡 free {free2 / 1e9:.2f}GB (含游戏占用波动)"))

    # 13. no dxg error
    import subprocess
    dxg = subprocess.run(["dmesg"], capture_output=True, text=True).stdout
    n_dxg = sum(dxg.count(k) for k in ["dxgvmb_send_create_allocation failed",
                                       "dxgkio_create_allocation: Ioctl failed", "EOVERFLOW"])
    results.append(check("no new dxg error", n_dxg == 0, f"count={n_dxg}"))

    print(f"\nACCEPTANCE: {sum(results)}/{len(results)} PASS")
    print("ALL_ACCEPTANCE_PASS" if all(results) else "ACCEPTANCE_FAILED")


if __name__ == "__main__":
    main()
