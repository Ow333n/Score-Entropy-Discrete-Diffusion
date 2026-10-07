# Phase 1.2 Bridge Verdict（FROZEN）

**状态：FINAL — FROZEN（2026-10-07）**
**前序**：precision-ladder 裁决 Level C（B FAIL → C = true-FP32 mirrored scoring forward）

## 冻结要素

- Bridge checkpoint set：**15**（FROZEN 协议 §4 枚举全部；协议文本"11"为起草计数失误，
  已在 bridge_manifest 披露，按"不增删枚举"冻结 15 个）
- bridge manifest sha256：`986dcf1d0c79e7ff6fd21f9d00cf782ce81707556d62f3ed2b0425ea98d6587b`
- old BF16 evaluator：`evaluation/eval_cpi.py` sha256 `e8e250ba3a13243380de14212278c27775a759d9406ee348525472e33bcee07c`
- Level C FP32 evaluator：`evaluation/eval_diag_fp32.py` sha256 `ef1ef03ea08da87fb75eda6049311e2dd4ae91ab1a9f29dfa9d57cdcc6064030`
- Phase 1 protocol sha256：`ec61491736dac5e54a3d857896585e2c0a49361405a705dd5b09a76dd8d46ee7`
- git commit（分析时代）：`9ccc7e8`
- sample set：full frozen manifest 500 样本（0..499），两套 evaluator 完全同样本
- 分析产物：`results/phase1_bridge/bridge_analysis.json` + `bridge_scatter.png`

## 结果摘要

- **A 方向保存**：核心 pretrained→SFT **14/14 对同号**（ΔCPI 全部为负 −0.033…−0.065，
  FP32 下 attenuation 保留）；全部 20 对 18/19 同号；2 个不同号均为 within-run 近零小效应
- **B ranking**：CPI_abs Spearman **0.9821** / Kendall 0.9596；CPI_RMS S=0.9821；
  CE/NLL S=0.9643
- **C absolute shift（FP32−BF16）**：CPI_abs mean −0.0032（range −0.0062…+0.0000）；
  CE mean −0.0007（range −0.011…+0.016）
- **D per-sample agreement**（pooled 7500）：|δ| Pearson **0.9963**、signed δ 0.9970、
  CE 1.0000——旧 evaluator 只是 coarse/quantized，不系统改变样本排序
- **E BF16 CE 量化**：old unique 597 vs new 7500、lattice spacing 0.0078125（bf16 网格
  实证）；量化真实存在但未破坏 checkpoint 级 CE ranking

## Verdict：**BRIDGE-A**

1. 历史 large compatibility effects（pretrained→SFT 衰减、v1.2/v2.1 Case 判定）在 FP32 下
   定性保持；
2. 小的 evaluator-dependent differential effects 不跨 evaluator 重新解释；
3. Phase 1 fine-grained CE/NLL dynamics 必须使用 FP32。

历史 frozen report（v1.2 / v2.1）**不修改、不重写**；旧结论保持原样。
