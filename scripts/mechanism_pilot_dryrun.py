"""Mechanism pilot dry-run / preflight（CPU，无模型）。

protocol v1.2 §9.4 + pilot preflight（2026-10-06 批准）：
1. K sanity：经验 mean(K_t) vs 经验 mean(m_t·q_t)（z-score 判定）；
2. 跨 policy 哈希：同一 replicate 下经 A/B/C/D 四条实际 policy execution
   path 后 schedule_hash_A == B == C == D，且 replicate 1 != replicate 2。

用法: .venv/bin/python scripts/mechanism_pilot_dryrun.py
输出: results/mechanism_pilot_dryrun/dryrun_summary.json + stdout 摘要
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import initialize, compose

from data import get_dataset
from task_data.policy_corruption import (SharedSchedule, draw_pi_L, make_generator,
                                        select_mask)

POLICIES = ("A", "B", "C", "D")


def replay_with_policy(r, policy, n_steps, micro_batch, N_chunks, cfg):
    """重放 replicate r 的共享五元组流并执行指定 policy 的选择路径。

    返回 (schedule_sha256, K_all, sigma_all, span_len_all, n_masks_total)。
    """
    g_data = make_generator("data_order", r)
    perm = torch.randperm(N_chunks, generator=g_data)
    sample_ids = perm[: n_steps * micro_batch]

    sched = SharedSchedule(r, cfg.data.seq_len, cfg.data.span_min,
                           cfg.data.span_max, noise=None)
    g_a = make_generator("policy_a", r)
    g_b = make_generator("policy_b", r)
    pi_L = draw_pi_L(g_b, cfg.data.seq_len)

    digest = hashlib.sha256()
    K_all, sigma_all, span_len_all, n_masks = [], [], [], 0
    for i in range(0, sample_ids.numel(), micro_batch):
        sigma, dsigma, span_len, span_start, K = sched.draw(micro_batch)
        K_all.append(K); sigma_all.append(sigma); span_len_all.append(span_len)
        digest.update(sample_ids[i:i + micro_batch].numpy().tobytes())
        for arr in (sigma, dsigma, span_len, span_start, K):
            digest.update(arr.numpy().tobytes())
        # 实际 policy execution path（A 消耗自身流；B 用 π_L；C/D 确定性）
        for j in range(micro_batch):
            kwargs = (dict(generator=g_a) if policy == "A" else
                      dict(pi_L=pi_L) if policy == "B" else {})
            sel = select_mask(span_start[j:j + 1], span_len[j:j + 1], K[j:j + 1],
                              policy, cfg.data.seq_len, **kwargs)[0]
            n_masks += sel.numel()
            if sel.numel() != int(K[j].item()):       # |M|==K 不变量
                raise RuntimeError(f"{policy}: |mask|={sel.numel()} != K={int(K[j])}")
    return (digest.hexdigest(), torch.cat(K_all).float(), torch.cat(sigma_all),
            torch.cat(span_len_all).float(), n_masks)


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")

    n_steps = 2500
    micro_batch = cfg.training.batch_size // (cfg.ngpus * cfg.training.accum)
    n_items = n_steps * micro_batch

    train_ds = get_dataset("wikitext103", "train", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)
    N_chunks = len(train_ds)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "mechanism_pilot_dryrun")
    os.makedirs(out_dir, exist_ok=True)

    summary = dict(protocol_version="v1.2", preflight="2026-10-06",
                   n_steps=n_steps, micro_batch=micro_batch, n_items=n_items,
                   N_train_chunks=N_chunks, seq_len=cfg.data.seq_len,
                   span_min=cfg.data.span_min, span_max=cfg.data.span_max,
                   replicates={})

    print(f"重放规模：{n_steps} 步 × {micro_batch} items = {n_items} 五元组/run（×4 policy × 2 replicate）")
    for r in (1, 2):
        hashes, stats = {}, {}
        for policy in POLICIES:
            h, Kt, st, mt, n_masks = replay_with_policy(r, policy, n_steps,
                                                        micro_batch, N_chunks, cfg)
            hashes[policy] = h
            q_t = 1 - (-st).exp()
            # Preflight 1: 经验 mean(K_t) vs 经验 mean(m_t·q_t)
            z = (Kt.mean() - (mt * q_t).mean()) / (
                ((mt * q_t * (1 - q_t)).mean() / n_items) ** 0.5)
            stats[policy] = dict(K_mean=float(Kt.mean()),
                                 mq_mean=float((mt * q_t).mean()),
                                 z_score=float(z),
                                 n_masks_total=n_masks)
            print(f"  r={r} policy {policy}: hash={h[:16]}… "
                  f"mean(K)={Kt.mean():.4f} mean(m·q)={(mt*q_t).mean():.4f} "
                  f"z={z:+.3f} Σ|mask|={n_masks}")
        all_equal = len(set(hashes.values())) == 1
        print(f"  r={r}: schedule_hash_A==B==C==D: {all_equal}")
        summary["replicates"][str(r)] = dict(
            schedule_sha256=hashes, policy_hashes_equal=all_equal, stats=stats)

    h1 = summary["replicates"]["1"]["schedule_sha256"]["A"]
    h2 = summary["replicates"]["2"]["schedule_sha256"]["A"]
    print(f"\nreplicate 1 != replicate 2: {h1 != h2}")
    print(f"preflight 1 (K sanity)：全部 |z| < 3 判定通过："
          f"{all(abs(summary['replicates'][k]['stats'][p]['z_score']) < 3 for k in ('1','2') for p in POLICIES)}")
    print(f"preflight 2 (跨 policy 哈希一致)："
          f"{summary['replicates']['1']['policy_hashes_equal'] and summary['replicates']['2']['policy_hashes_equal']}")

    path = os.path.join(out_dir, "dryrun_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary: {path}")


if __name__ == "__main__":
    main()
