#!/usr/bin/env python
"""P1 preflight gate 2 辅助: v4.1 formal checkpoint vs P1 checkpoint 的 tensor-level 复现比较。

用法:
    .venv/bin/python scripts/compare_checkpoints.py <p1_ckpt> <v41_ckpt>

退出码:
    0 = 逐字节一致, 或结构一致且全部数值差异在 kernel 非确定性噪声量级 (max rel < 1e-5)
        → PASS / PASS-CAVEAT (证据见输出), 可继续评估
    1 = 结构不一致, 或数值差异超阈值 (训练轨迹实质分歧) → 硬停排查, 不要继续烧评估

比较对象: model (131 keys) / ema (decay, num_updates, shadow_params[130]) /
          optimizer (AdamW state: exp_avg / exp_avg_sq / step) /
          scaler (GradScaler state) / step。
RNG state: checkpoint 不序列化 RNG (save_ckpt 只存 model/ema/optimizer/scaler/step)
          → N/A; corruption 流的任何分歧会经 loss/梯度进入权重, 故权重一致即隐含
          corruption 流一致 (两 run 均 num_workers=0 + 固定 generator)。
阈值语义: fp32 kernel 非确定性累积误差典型 ≤1e-6 rel; 真实轨迹分歧在 1020 步内
          会放大到 ≥1e-3 rel。阈值 1e-5 rel 留足安全边际。
"""
import sys
import torch

THRESH = 1e-5


def compare(name, a, b, report, stats):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            report.append(f"{name}: shape/dtype 不一致 {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
            return
        if a.numel() == 0:
            return
        if a.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            aa, bb = a.float(), b.float()
            d = (aa - bb).abs()
            denom = torch.maximum(aa.abs(), bb.abs()).clamp_min(1e-12)
            mabs = d.max().item()
            mrel = (d / denom).max().item()
            stats["max_abs"] = max(stats["max_abs"], mabs)
            stats["max_rel"] = max(stats["max_rel"], mrel)
            if mrel > THRESH:
                report.append(f"{name}: max_abs={mabs:.3e} max_rel={mrel:.3e}  ← 超阈值")
        else:  # int / bool 张量要求完全一致
            if not torch.equal(a, b):
                report.append(f"{name}: 整数张量不一致 ({a.dtype})")
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            report.append(f"{name}: dict keys 不一致: {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a.keys()) & set(b.keys())):
            compare(f"{name}.{k}", a[k], b[k], report, stats)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            report.append(f"{name}: 长度不一致 {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(f"{name}[{i}]", x, y, report, stats)
    elif isinstance(a, (int, float, str)) and isinstance(b, (int, float, str)):
        if a != b:
            if isinstance(a, float) and isinstance(b, float):
                d = abs(a - b)
                denom = max(abs(a), abs(b), 1e-12)
                stats["max_abs"] = max(stats["max_abs"], d)
                stats["max_rel"] = max(stats["max_rel"], d / denom)
                if d / denom > THRESH:
                    report.append(f"{name}: {a!r} vs {b!r}")
            else:
                report.append(f"{name}: {a!r} vs {b!r}")
    elif a is None and b is None:
        pass
    else:
        report.append(f"{name}: 类型不一致 {type(a).__name__} vs {type(b).__name__}")


def main():
    p1_path, v41_path = sys.argv[1], sys.argv[2]
    A = torch.load(p1_path, map_location="cpu", mmap=True, weights_only=False)
    B = torch.load(v41_path, map_location="cpu", mmap=True, weights_only=False)

    print(f"P1:  {p1_path}")
    print(f"v4.1:{v41_path}")
    if A.get("step") != B.get("step"):
        print(f"FATAL: step 不一致 {A.get('step')} vs {B.get('step')}")
        sys.exit(1)
    print(f"step: {A['step']} == {B['step']} ✓")

    report, stats = [], {"max_abs": 0.0, "max_rel": 0.0}
    for sec in ["model", "ema", "optimizer", "scaler"]:
        if sec in A or sec in B:
            compare(sec, A.get(sec), B.get(sec), report, stats)

    print(f"\n全局 max_abs = {stats['max_abs']:.3e}   max_rel = {stats['max_rel']:.3e}  (阈值 {THRESH:.0e})")
    if report:
        print("差异明细 (前 20 条):")
        for r in report[:20]:
            print(" ", r)
    else:
        print("无任何差异记录。")

    if stats["max_rel"] < THRESH and not report:
        print("\nPASS: 结构一致, 全部数值差异在 kernel 非确定性噪声量级内。")
        print("      结论: P1 训练轨迹与 v4.1 formal 复现一致 (证据已入 preflight audit)。")
        sys.exit(0)
    print("\nFAIL: 训练轨迹实质分歧 → 硬停排查, 不要继续评估。")
    sys.exit(1)


if __name__ == "__main__":
    main()
