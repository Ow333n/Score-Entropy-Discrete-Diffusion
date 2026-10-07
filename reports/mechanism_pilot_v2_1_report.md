# Mechanism Pilot v2.1 — Complementary Exposure H/U/L：Final Report

**状态：FINAL — FROZEN（2026-10-07）**
**协议**：`protocol/mechanism_complementary_exposure_v2_1.md`（FINAL v1.1，sha256 `af312d86…`，本体零改动）
**执行**：`protocol/mechanism_v2_1_execution_manifest.json`（sha256 `5cb816f3…`）+ `protocol/mechanism_v2_1_errata.md`（sha256 `d099aaaa…`）
**分析**：`scripts/v21_g3_analysis.py` + `results/mechanism_pilot_v21/v21_g3_analysis.json`（commit `f9d9c3b`）
**pilot start commit**：`108147f5d8728e0546c0159e2e2fc86bba4be0e1`（= 2c72124 provenance patch + MEMORY.md docs）

---

## 0. 结论（预注册 Case 表判定）

**Case C**：无稳定证据支持 complementary partial-conditioning exposure 驱动 global reveal-order
compatibility 衰减；各 contrast 均 INCONCLUSIVE under the current pilot setting。

**All H/U/L variants still exhibit CPI attenuation relative to pretrained, despite the absence of a
stable policy-specific compatibility effect.**

措辞纪律（协议 §1.1）：不表述为"complementary exposure 已被证明无效"；统一表述
"no stable supporting evidence / INCONCLUSIVE under the current pilot setting"。

**§15.1 Confirmatory 进入条件评估**：① G3a PASS ✓ ② H-vs-L 两 replicate 同向 ✗ ③ effect size
无研究意义 ✗ ④ G3d matching 可行 ✓ → **不满足进入条件，不进入 confirmatory，不盲目加 seeds。**

---

## 1. Provenance

- 6 runs：U1→H1→L1→U2→H2→L2（v21pilot-*，各 2500 精确步，~19min，2.2 steps/s，零 NaN/OOM/dxg）
- pilot_start_commit = `108147f5…`（六 run 一致）；协议/errata/execution-manifest/maps sha 全部
  启动时硬校验通过；frozen 文件 git-clean
- schedule digest 逐位一致（r1 `5da669c2045d9eb5…` / r2 `3ee9c9dfc1dfdc38…`，与 dryrun 参考一致）；
  exact-K violations = **0**（80000 items × 6）
- recipe（execution manifest）：LR=3e-4（v1.2 actual executed setting；v1.2 协议文本 3e-5 记录不改写）、
  **dropout=0.1**（v1.2/v2.1 协议文本 0.0；hydra 合并语义下 small.yaml 默认值优先，全部既往 run
  实际均为 0.1；见 errata）、warmup 2500、AdamW(0.9,0.999,1e-8)、wd=0、clip 1.0、batch 32 eff、
  seq 256、span [10,50]、EMA 0.9999、checkpoints 500/1020/2500
- 评估 76 文件（global 36 + treated 12 + heldout 18 + subset 6 + step0 4），manifest sha `1897bd14…` 全一致

## 2. G3a — Intervention validity：**PASS（8/8）**

| 项 | r1 | r2 |
|---|---|---|
| 共享 v21 digest U==H==L | ✓ (`5da669c2…`) | ✓ (`3ee9c9df…`) |
| exact-K violations | 0 | 0 |
| E_comp（H/U/L） | 0.4800 / 0.3325 / 0.0397，**H>U>L** ✓ | 0.4834 / 0.3354 / 0.0396，**H>U>L** ✓ |
| E_comp 实测 vs 闭式 | ✓（<0.01） | ✓ |
| C̄_treated（H/U/L） | −0.0555 / −0.0063 / +0.0913，**H<U<L** ✓ | −0.0556 / −0.0062 / +0.0922，**H<U<L** ✓ |
| C̄_active（H/U/L） | −0.0821 / −0.0068 / +0.1432，H<U<L ✓ | −0.0822 / −0.0065 / +0.1450，H<U<L ✓ |
| 单位置 marginal（rel-pos 最大偏差） | 0.007–0.014（≈采样噪声界） | 0.011–0.012 |
| L/R 趋势（z） | U 1.35 / H 0.44 / L −0.41 | U −0.61 / H 0.14 / L −0.60 |
| degenerate K 计数表（三 policy） | 完全相等 ✓ | 完全相等 ✓ |
| pair-map hash + heldout 不重叠 | ✓ | ✓ |
| confound diagnostics 七项 | 全部产出 ✓ | 全部产出 ✓ |

干预在 mask 层面完全按设计生效。

## 3. G3b — Local manipulation check：**INCONCLUSIVE**

treated CPI（per-sample mean |δ| over treated edges；H/L 同 replicate 共享 map，可逐样本配对；
per-replicate paired bootstrap 10k seed=0 + 按 replicate 聚类 pooled CI）：

| step | H−L point | pooled CI | rep means | verdict |
|---|---|---|---|---|
| 500 | −0.0003 | [−0.0036, +0.0031] | −0.0002 / −0.0003 | INC |
| 1020 | +0.0016 | [−0.0029, +0.0062] | +0.0020 / +0.0013 | INC |
| 2500 | +0.0055 | [−0.0055, +0.0160] | −0.0009 / +0.0119 | INC |

H 的 treated-pair CPI 并未比 L 更低；所有 policy 的 treated CPI 从 step0（r1 0.3480 / r2 0.3427）
均匀衰减到 ~0.296–0.308（衰减 ~0.04–0.05），policy 间无稳定差异。

## 4. G3c — Global mechanism signal：**INCONCLUSIVE**

- **Primary H vs L @2500**：+0.0108，pooled CI [−0.0215, +0.0442]，两 replicate 异号
  （+0.0250 / −0.0035）→ INC（pooled 点估计方向与预测相反，但不稳定）
- Secondary H vs U @2500：−0.0082 CI[−0.0502, +0.0371]（异号）→ INC
- Secondary U vs L @2500：+0.0189 CI[−0.0114, +0.0483]（同号但 CI 含 0）→ INC
- **U 位置**（pooled）：H=0.2812 / U=0.2900 / L=0.2705 → U 不居中且 H<L 不成立。按 §11
  预注册规则：U 偏离作为 unexpected mask-structure / seed-noise 效应披露讨论，
  不重新解释 hypothesis。

## 5. G3d — Matched performance：MATCHED（评估器分辨率内）+ 重做 G3c INC

§5.3 匹配规则（参考=H 的 {500,1020,2500}，池=L 的同集合，最小 |ΔNLL| 且落 NLL±0.02/acc±0.01，
平局取更小 step；no-overlap ⇒ INC 非 FAIL）：

| rep | 配对 | ΔNLL | Δacc | ΔCPI | paired CI |
|---|---|---|---|---|---|
| 1 | H@500 → L@500 | 0.0000 | +0.0012 | +0.0098 | [−0.0051, +0.0280] |
| 1 | H@1020 → L@1020 | 0.0000 | −0.0030 | +0.0098 | [−0.0156, +0.0332] |
| 1 | H@2500 → L@2500 | 0.0000 | −0.0051 | +0.0254 | [−0.0136, +0.0649] |
| 2 | H@500 → L@500 | 0.0000 | +0.0006 | +0.0078 | [−0.0111, +0.0292] |
| 2 | H@1020 → L@1020 | 0.0000 | −0.0062 | −0.0078 | [−0.0312, +0.0166] |
| 2 | H@2500 → L@2500 | 0.0000 | +0.0026 | −0.0039 | [−0.0388, +0.0331] |

全部匹配成功（无 no-overlap）→ 匹配后重做 G3c：rep means [+0.0254, −0.0039]，CI [−0.0388, +0.0649]
→ **INCONCLUSIVE**。

### 5.1 ⚠ bf16 NLL 量化披露（预存在性质，非 v2.1 回归）

frozen evaluator 的数值链为 **bf16 端到端**：`model/transformer.py:282`（frozen 基线实现）在
model forward 内部启用 `autocast(bfloat16)` → score 输出 bf16 → `clean_log_probs` 的
log_softmax 保持 bf16 → δ 与 CE/NLL 及全部 summary 均值均为 bf16。实测量化网格：
masked NLL 2⁻⁵/2⁻⁶（~0.0156–0.031 nats）、δ 2⁻⁶/2⁻⁷、CPI_abs 2⁻⁹（~0.001）。
**该性质自 v4.1 pretrained 基线（local CE 4.281）起即存在**，v1.2、v4.2、v2.1 全部评估同此口径。
后果：G3d 的 ±0.02 NLL 容差低于量化粒度 → "匹配"实为"同桶"，应解读为
**评估器分辨率下无 NLL 分离**，而非精确性能相等。CPI 对比不受实质影响（δ_SD≈0.53 ≫ 量化噪声，
paired bootstrap 有效）。已列入后续 Phase 1.1 FP32 diagnostic evaluator 的动机。

## 6. H/U/L 全指标轨迹（@500 / 1020 / 2500）

CPI_abs（step0 全局基线 0.3281）：

| run | 500 | 1020 | 2500 |
|---|---|---|---|
| U1 | 0.2773 | 0.2910 | 0.2715 |
| U2 | 0.2676 | 0.2832 | 0.3086 |
| H1 | 0.2773 | 0.2891 | 0.2891 |
| H2 | 0.2832 | 0.2988 | 0.2734 |
| L1 | 0.2676 | 0.2793 | 0.2637 |
| L2 | 0.2754 | 0.3066 | 0.2773 |

CPI_RMS：

| run | 500 | 1020 | 2500 |
|---|---|---|---|
| U1 | 0.5234 | 0.5664 | 0.5273 |
| U2 | 0.5039 | 0.5586 | 0.6172 |
| H1 | 0.5312 | 0.5547 | 0.6133 |
| H2 | 0.5547 | 0.5742 | 0.5391 |
| L1 | 0.4980 | 0.5391 | 0.5430 |
| L2 | 0.5195 | 0.5898 | 0.5469 |

masked NLL（bf16 量化网格，见 §5.1）：

| run | 500 | 1020 | 2500 |
|---|---|---|---|
| U1/U2 | 4.00000 | 3.95312 | 4.06250 |
| H1/H2/L1/L2 | 4.00000 | 3.96875 | 4.03125 |

token accuracy：

| run | 500 | 1020 | 2500 |
|---|---|---|---|
| U1 | 0.3455 | 0.3497 | 0.3421 |
| U2 | 0.3413 | 0.3500 | 0.3399 |
| H1 | 0.3455 | 0.3478 | 0.3375 |
| H2 | 0.3462 | 0.3461 | 0.3436 |
| L1 | 0.3442 | 0.3508 | 0.3426 |
| L2 | 0.3456 | 0.3524 | 0.3411 |

@2500 δ 分布（signed mean 全程 ≈0）：

| run | δ̄ | δ_SD | p90 | p99 |
|---|---|---|---|---|
| U1 | −0.0081 | 0.5273 | 0.7602 | 2.2513 |
| U2 | +0.0149 | 0.6172 | 0.9160 | 2.7036 |
| H1 | −0.0098 | 0.6133 | 0.7984 | 2.5625 |
| H2 | −0.0120 | 0.5391 | 0.7844 | 2.7506 |
| L1 | +0.0022 | 0.5430 | 0.7039 | 2.5644 |
| L2 | +0.0017 | 0.5469 | 0.7977 | 2.2213 |

## 7. treated / heldout / global 一致性

step0 → 2500：treated 0.3480/0.3427 → 0.296–0.308；heldout 0.3444 → 0.297–0.307；
global 0.3281 → 0.264–0.309。**三口径一致均匀衰减，policy 间无系统性差异**。

heldout H vs L @2500：−0.0011 CI[−0.0084, +0.0060]（heldout 每样本边数多、CI 最紧，仍含 0）。

**U 与 v1.2-A 可比性**（统计同分布：同 recipe/schedule，仅 mask-choice 流种子不同）：
A1 0.2734 / A2 0.2715 / U1 0.2715 / U2 0.3086 → 四 run 全距 0.037，**run-to-run SD≈0.018–0.02**
与 H−L 假设效应同阶——当前 pilot 功效不足以分辨该量级的 policy 效应（Phase 1.0 power audit 的
直接输入）。

## 8. §9.5 OrderGap subset + path-score variance（step0 → 2500）

| | step0 | H | U | L |
|---|---|---|---|---|
| OrderGap_raw（64 样本子集，SE≈1.14） | 12.35 | 11.92 | 11.21 | 11.15 |
| path-score variance | 33.29 | 33.02 | 27.89 | 29.31 |

全部略低于 step0，policy 间无排序信号（64 样本子集噪声内）；与 global CPI 无一致性对应。

## 9. Confound diagnostics（§7.6 七项，全部产出）

- mask/visible run-length 均值 ≈2.8，三 policy 一致（U 2.813/2.808 → H 2.810/2.807 → L 2.819/2.815，r1）
- transition 数：H 略高（9.659 vs L 9.630）——intended（discordant 更多）
- context-visible 同 replicate 内逐 policy 逐位一致（共享性 sanity ✓）
- pair-distance / rel-pos 表 / 非 treated 协方差直方图：跨 policy 仅 intended 差异，无意外 confound

## 10. Anomalies

零。无 dxg/EOVERFLOW、无 NaN、无 OOM（VRAM 6.41GB alloc / 7.32GB resv）、无 checkpoint 损坏、
无 hash/schedule 不匹配。磁盘：D: free 27GB（pilot 前 42GB，消费 ~16.3GB 与预估一致）。

## 11. 相关 errata（详见 protocol/mechanism_v2_1_errata.md）

1. odd-m §6 cycle-edge active-phase 数：文本"2 phases"仅 m=5 成立；一般 (m−1)/2。explanatory-text
   error，实现与全部验证不受影响。
2. degenerate-K E_comp 度量 artifact（U vs H/L ≈2e-4）：P_H=P_U=P_L 分布已验证相同，不构成 G3a FAIL。
3. LR：v1.2 文本 3e-5 vs 实际执行 3e-4；v2.1 取实际执行值。
4. dropout：文本 0.0 vs 实际执行 0.1（hydra 合并语义）；v2.1 取实际执行值。

## 12. 与既有发现的关系（保留事实，不新增声明）

- pretrained CPI≈0.3281 → 全部 14 个 partial-reveal SFT run（v1.2×8 + v2.1×6）衰减到 0.264–0.309：
  **partial-reveal SFT 降低 CPI 是最稳定的 observation，且不依赖 mask-policy 细节**。
- 三个机制 pilot（fresh-random / directional / complementary-exposure）均未找到稳定 policy-specific
  效应 → 转向"为何普通 SFT 普遍降低 reveal-order incompatibility"的 estimation-dynamics 路线
  （见后续 Phase 1 protocol）。
