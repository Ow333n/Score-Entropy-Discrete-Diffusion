"""Regime A 评估 manifest 构建器 (protocol §5.3 / §5A / §21)。

构造 span-structured 样本: wikitext103 test 256-chunk, target span 10~50 token,
span 外上下文恒可见, span 内 partial-absorbing 腐蚀 (mask prob = 1-e^{-sigma},
sigma 走冻结网格 {0.5, 1.5, 3.0} round-robin)。m<2 的样本构建期跳过并记录
skip count (§21), 不在评估期临时处理。

每条记录字段 (§5.3 + 评估便利字段 x0/initial_state):
  sample_id, source_offset, context_tokens, target_span, span_start, span_end,
  initial_masked_positions, initial_state, x0, i, j, a, b, sigma, seq_len,
  span_len, mask_ratio, revealed_target_count, pair_distance,
  paths{l2r,r2l,random_0..2,confidence}, random_path_seeds

路径 (§3.10/§6.3): l2r/r2l/3 条 seeded random/confidence(pretrained TF 揭示生成)。
manifest 版本化 (§5A): 同一版本号内容不可变, 写完后生成 SHA-256 侧车文件。

用法:
  .venv/bin/python evaluation/build_manifest.py [--with-confidence-path]
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from data import get_dataset
from load_model import load_model
from model import utils as mutils
import compatibility.order_gap as og

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_TOKEN = 50257          # GPT-2 vocab 50257 + MASK (configs/config.yaml tokens=50257)


def sample_span(g, seq_len, span_min, span_max):
    span_len = int(torch.randint(span_min, span_max + 1, (1,), generator=g).item())
    span_len = min(span_len, seq_len)
    span_start = int(torch.randint(0, seq_len - span_len + 1, (1,), generator=g).item())
    return span_start, span_len


def build_records(ds, proto, g):
    seq_len = proto.data.seq_len
    grid = list(proto.data.sigma_grid)
    n_samples = proto.eval_manifest.n_samples
    seed = proto.eval_manifest.eval_manifest_seed
    n_random = proto.eval_manifest.random_paths_per_sample

    records, n_skipped = [], 0
    for sample_id in range(n_samples):
        x0 = ds[sample_id]["input_ids"]                          # [L]
        span_start, span_len = sample_span(g, seq_len, proto.data.span_len_min, proto.data.span_len_max)
        span = list(range(span_start, span_start + span_len))
        sigma = float(grid[sample_id % len(grid)])

        # span 内 partial-absorbing: mask prob = 1-e^{-sigma}; span 外恒可见
        x_t = x0.clone()
        move = torch.rand(span_len, generator=g) < (1 - torch.exp(torch.tensor(-sigma)))
        for p, mv in zip(span, move.tolist()):
            if mv:
                x_t[p] = MASK_TOKEN

        masked_span_pos = [p for p in span if x_t[p].item() == MASK_TOKEN]
        m = len(masked_span_pos)
        if m < proto.eval_manifest.min_masked_target:            # §21: 构建期跳过
            n_skipped += 1
            continue

        # 随机一对 (i, j), i != j (§5.2)
        perm = torch.randperm(m, generator=g).tolist()
        i, j = masked_span_pos[perm[0]], masked_span_pos[perm[1]]

        # 路径: l2r / r2l / n_random 条 seeded random (§6.3); confidence 稍后由 pretrained 生成
        paths = dict(
            l2r=sorted(masked_span_pos),
            r2l=sorted(masked_span_pos, reverse=True),
        )
        random_path_seeds = []
        for k in range(n_random):
            pg = torch.Generator().manual_seed(seed * 100000 + sample_id * 10 + k)
            order = torch.randperm(m, generator=pg).tolist()
            paths[f"random_{k}"] = [masked_span_pos[t] for t in order]
            random_path_seeds.append(seed * 100000 + sample_id * 10 + k)

        records.append(dict(
            sample_id=sample_id,
            source_offset=sample_id,
            context_tokens=[x0[p].item() for p in range(seq_len) if p not in span],
            target_span=[x0[p].item() for p in span],
            span_start=span_start,
            span_end=span_start + span_len,
            initial_masked_positions=masked_span_pos,
            initial_state=x_t.tolist(),
            x0=x0.tolist(),
            i=i, j=j,
            a=x0[i].item(), b=x0[j].item(),
            sigma=sigma,
            seq_len=seq_len,
            span_len=span_len,
            mask_ratio=m / span_len,
            revealed_target_count=0,
            pair_distance=abs(i - j),
            paths=paths,
            random_path_seeds=random_path_seeds,
        ))
    return records, n_skipped


def add_confidence_paths(records, model, score_fn, D):
    """§6.4: pretrained 生成的固定置信路径 π_conf-pre, 所有 checkpoint 共享。

    TF 揭示: 每步策略选当前 mask 的 span 位置中 p_hat 最大者 (§3.8 tie → 最小 index),
    揭示 gold token。按 sigma 分组批量跑 strict_reveal。
    """
    groups = {}
    for r in records:
        groups.setdefault(r["sigma"], []).append(r)

    tie_total, tie_steps = 0, 0
    for sigma, rs in groups.items():
        for start in range(0, len(rs), 8):
            chunk = rs[start:start + 8]
            xs = torch.tensor([r["initial_state"] for r in chunk], device="cuda")
            gold = torch.tensor([r["x0"] for r in chunk], device="cuda")
            _, info = og.strict_reveal(score_fn, xs, gold, sigma, D,
                                       order="confidence", mode="teacher_forced")
            for local, r in enumerate(chunk):
                r["paths"]["confidence"] = info["pos_history"][:, local].tolist()
                r["confidence_tie_count"] = int(info["tie_count"])
            tie_total += info["tie_count"]
            tie_steps += info["steps"]
    tie_fraction = tie_total / max(tie_steps, 1)
    print(f"confidence 路径生成完成 (pretrained): tie_count={tie_total}, tie_fraction={tie_fraction:.4f} (§3.8)")
    return records, tie_fraction


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=os.path.join(ROOT, "protocol/regime_a_protocol.yaml"))
    parser.add_argument("--model_path", default="louaaron/sedd-small")
    parser.add_argument("--with_confidence_path", "--with-confidence-path", action="store_true")
    args = parser.parse_args()

    proto = OmegaConf.load(args.protocol)
    manifest_path = os.path.join(ROOT, proto.eval_manifest.file)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    g = torch.Generator().manual_seed(proto.eval_manifest.eval_manifest_seed)
    ds = get_dataset("wikitext103", "test", cache_dir="data", block_size=proto.data.seq_len, num_proc=4)
    print(f"wikitext103 test 256-chunk 总数: {len(ds)}, 目标样本数: {proto.eval_manifest.n_samples}")

    records, n_skipped = build_records(ds, proto, g)
    print(f"构建样本: {len(records)}; 跳过 (m<2): {n_skipped} (§21)")

    if args.with_confidence_path:
        device = torch.device("cuda")
        model, graph, noise = load_model(args.model_path, device)
        score_fn = mutils.get_score_fn(model, train=False, sampling=False)
        records, tie_frac = add_confidence_paths(records, model, score_fn, graph.dim)

    with open(manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    digest = sha256_file(manifest_path)
    with open(manifest_path + ".sha256", "w") as f:
        f.write(digest + "\n")
    print(f"manifest 已冻结: {manifest_path} (n={len(records)})")
    print(f"SHA-256: {digest} (§5A: 同一版本号内容不可变)")


if __name__ == "__main__":
    main()
