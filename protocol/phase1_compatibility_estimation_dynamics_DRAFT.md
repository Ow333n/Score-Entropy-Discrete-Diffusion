# Phase 1 — Compatibility–Estimation Dynamics：Protocol

**状态：FINAL v0.2 — FROZEN（2026-10-07；用户批准，冻结后禁止修改 hypothesis / gate / 判定规则）**
**修订记录**：v0.1（初版）→ v0.2（用户修订 1–6：precision-ladder 前置门、两层 inference
分离、leave-one-family-out、SESOI 敏感性、Option B DEFER）→ **v0.2 FROZEN（最终 3 小修）**：
① sample-level inference 措辞纪律句（英文原文入 §5）② B-vs-C 容差标记为
"pre-registered engineering equivalence tolerances"（非 statistical equivalence test）
③ Phase 2 grouped CV（见 Phase 2 协议）。
**前序**：v1.2 / v2.1 已关闭（v2.1 frozen 报告 sha256 `dd3cdb26…`，Case C）
**原则**：observation / mechanism-diagnostic study；CPI 与 CE 强相关也不能声称因果。

---

## 1. 核心研究问题

> SFT 过程中 conditional estimation improvement 是否与 CPI attenuation 稳定对齐？

- P1-1：FP32 conditional CE/NLL 是否随 SFT 稳定下降？
- P1-2：FP32 CPI_abs 是否随 SFT 稳定下降？
- P1-3：within-run ΔCE 与 ΔCPI 是否在多个 run/policy 中一致关联？
- P1-4（temporal）：CE improvement 是否系统性早于 CPI attenuation？（lead-lag，§8）

---

## 2. 资产审计结论（Phase 1.0，2026-10-07 已执行）

权重可用：pretrained step0；formal v4.1 s1/s2 × {1020,5100,10200}（lr=3e-5）；
v1.2 A/B/C/D ×2 × {500,1020,2500}；v2.1 U/H/L ×2 × {500,1020,2500}（均 lr=3e-4）；
RL snapshots。**p1 dense 中间权重已退役**（只剩 2500+meta；84 个旧 JSON 为 dense 轨迹唯一来源）。

方差分解：sample-level SE 0.020–0.024（N=500）；within-run SD ~0.008；between-seed
spread 0.002–0.037；14-run pooled SD = **0.0113**；pretrained→SFT 衰减 0.0495 ≈ 4.4×SD。

---

## 3. Phase 1.1 — Precision-ladder preflight（修订 #1，FP32 evaluator 的前置门）

**动机**：事后 `.float()` 不能恢复 bf16 forward 已舍入的 score 精度。必须先量化
"真正改善发生在哪一层"，再定正式 evaluator 的规格。

**已探针确认的结构事实（2026-10-07）**：
- frozen forward 是**混合精度**：`vocab_embed / sigma_map / rotary` 在 bf16 autocast **外**
  （fp32）；`blocks 循环 + output_layer` 在 `transformer.py:282` 的 bf16 autocast **内**；
  后续 scale_by_sigma/scatter 跟随 x dtype（bf16）
- 外层 `autocast(enabled=False)` **不能**覆盖内层（嵌套语义内层获胜）→ Level C 不能靠
  disabled 上下文实现
- **Level C 可行方案（已验证）**：evaluator 侧**镜像 forward 调用序列**（调用相同的
  frozen 子模块 vocab_embed/sigma_map/rotary/blocks/output_layer，仅去掉 bf16 autocast），
  不改任何 frozen 文件；**镜像忠实性校验已通过**：blocks 段包 bf16 时镜像输出与 frozen
  forward **逐位一致**；全 fp32 镜像输出超出 bf16 网格（真 fp32），峰值显存 0.84GB
  （2×64），模型级 bf16 量化冲击 median 相对差 2.0e-3 / max 2.9e-2

**Ladder 三级定义**：

| Level | forward | 提取/聚合 | 预期 |
|---|---|---|---|
| A（frozen 基线） | frozen 混合 forward（bf16 blocks） | 全 bf16（现状） | NLL 2⁻⁵/2⁻⁶、δ 2⁻⁶/2⁻⁷、CPI 2⁻⁹ 网格 |
| B | frozen 混合 forward（bf16 blocks） | score.float() → fp32 log_softmax/δ/CE → fp64 聚合 | 网格消失；score 仍含模型级 bf16 舍入 |
| C | 镜像 forward 全 fp32（blocks/output 也 fp32） | 同 B | 无任何量化；显存/速度需实测 |

**Preflight 规格**：
- 固定小样本集：frozen manifest 的 32 个均匀间隔样本（索引列表 + sha256，批量前冻结）；
  2 个模型（pretrained step0 + v2.1-H1@2500）
- 每 (level, model) 跑 delta-swap + CE 路径，输出：per-position NLL 原始值（网格检测：
  GCD-of-differences / 网格外值占比）、CPI_abs/RMS/δ 分布、per-sample CE、VRAM、wall-time
- **判定规则（预注册）**：
  1. B 必须证明主要 NLL lattice / quantization artifact 消失（否则 B 无意义）
  2. B 与 C 均落入以下 **pre-registered engineering equivalence tolerances**（明确标记：
     **不是正式 statistical equivalence test**，只是工程等价容差）：
     |ΔCPI_B−C| < 0.003 且 |ΔCE_B−C| < 0.01（两者在全部测试模型上成立）
  3. **选择规则**：B 消除主要 NLL lattice 量化 **且** B 与 C 落入上述 engineering
     tolerance → 正式 bulk evaluator = **Level B**（bf16 模型舍入对聚合统计影响在
     容差内，成本最低）；**否则 → Level C**（显存/速度须同时满足 8GB 预算与吞吐预算，
     否则按 chunk 降档并记录）
  4. A 始终为历史对照，不进正式 evaluator
- 产出：`results/phase1_preflight/precision_ladder_report.json`（+报告小节），
  **用户 review 通过后才允许批量运行**

---

## 4. Phase 1.2 — 新旧 evaluator bridge（保留 v0.1 设计）

Bridge set 11 checkpoint 预注册（pretrained + p1-s1/s2@2500 + formal-s1/s2@{1020,10200}
+ v2.1 U1/H1/L1@1020、@2500 + v1.2 A1@1020、@2500；精确清单批量前冻结落盘 sha256）。
每 checkpoint 跑 old frozen + 新（B 或 C 由 §3 判定）两套；比较 CPI_abs/RMS/signed δ/
CE-NLL/acc 的绝对差、rank preservation（Spearman）、Pearson、衰减方向一致性。
禁止为对齐两套数字调新 evaluator。

---

## 5. Phase 1.3 — Primary / Secondary 预注册（保留 v0.1 + 修订 #2）

**两层 inference 明确分离（修订 #2，硬性纪律）**：
- **Level-1 sample-level association**：固定 checkpoint 内 500 样本的 per-sample
  CE vs CPI 关联（独立样本，有效推断）。措辞纪律（2026-10-07 补充，FROZEN）：
  "Sample-level inference estimates within-checkpoint/sample heterogeneity conditional
  on a fixed trained model. It does not increase the number of independent training
  replicates."——500 samples / 多 checkpoints 不能被解释为增加 training-seed N。
- **Level-2 training-run trajectory association**：checkpoint 间 Δ 配对（within-run），
  **checkpoints 不是独立 training replicates**、跨 checkpoint 相关——仅 exploratory
- **禁止**：把 500 samples × N checkpoints 拼成 500N 个"独立"样本做显著性

Primary（P1-1/2/3 判据不变：同向 + CI 排除 0 / 一致关联 bootstrap CI）。

Secondary：CPI_RMS、δ 分布（§6）、finite-path OrderGap（§7）、path-score variance、
score scale、entropy、calibration（reliability）、token acc。

---

## 6. δ 分布报告规范（保留）

median |δ|、P75/P90/P95、RMS、signed mean、positive/negative fraction、outlier fraction
（|δ| > 2×δ_SD 与 > P95 双口径）；per-quantile 的 step0→SFT 变化表 → 判读
"A. 整分布左移 vs B. 尾部 outlier 减少"。

---

## 7. finite-path OrderGap（保留，命名固定）

固定 6 冻结路径（l2r/r2l/random×3/confidence，manifest paths）；OrderGap = max−min 于
该有限 path set；报告 n_paths、mean/SEM、path-score variance、sampling uncertainty；
明确不声称枚举 m!；小 span exhaustive（m≤7）当前 manifest 无此子集 → deferred。

---

## 8. Phase 1.4 — Lead-lag / temporal ordering diagnostic（修订 #5：Option B DEFER）

- **正式证据链（零训练）**：FP32 dynamics 于现有权重——v2.1/v1.2 的 {500,1020,2500}
  三点轨迹 ×14 runs + formal v4.1 的 {1020,5100,10200} ×2 seeds（更长程）
- **旧 p1 dense JSON（84 文件，bf16 口径 CE）只作 secondary historical lead-lag
  evidence**，明确标注口径限制
- **Option B（dense 确定性重训）= DEFER**：只有 Phase 1 在上述证据中出现值得追的
  temporal signal 后，才向用户提案是否重训（Gate-2 保证可逐字节复现，2×~18min+~33GB）
- 统计（同 v0.1）：within-run lag-1 cross-correlation，corr(ΔCE_t, ΔCPI_{t+1}) vs
  corr(ΔCPI_t, ΔCE_{t+1})；per-run + 跨 run bootstrap；命名纪律：lead-lag / temporal
  ordering diagnostic，**禁称 Granger causality**
- 判读三分支不变（CE 先 / 同步 / CPI 先）

---

## 9. 统计计划（修订 #2/#3）

1. 每条 training run 独立 trajectory（主报告元素）
2. effect size + bootstrap CI（10k seed=0）
3. pooled / mixed-effects 回归（CPI ~ CE + checkpoint + (1|run)）**仅 exploratory**；
   n_runs 小，p-value 不作主结论
4. within-run centering：CPI_rt − mean_r(CPI) vs CE_rt − mean_r(CE)
5. **Leave-one-experiment-family-out robustness（修订 #3）**：family =
   {formal(v4.1), v1.2-AB, v1.2-CD, v2.1-HUL, p1-dense(historical)}；逐 family 拿掉后
   重新估计 CE–CPI association（斜率/Spearman/Δ 配对方向），报告估计范围；
   primary 结论须对 leave-out 稳定

---

## 10. 多重比较（保留）

Primary 不扩散；secondary 用 BH-FDR；报告 raw p / adjusted q / effect size / CI；
禁止把 secondary 中挑出的显著结果升级成 primary。

---

## 11. SESOI 与 stopping rule（修订 #4）

- **research-scale SESOI = 0.02 CPI_abs**（≈40% 衰减尺度；依据 §2 方差分解）
- **敏感性分析**：0.01 / 0.02 / 0.03 三档分别报告可检性与结论敏感性
- CE 侧对应 SESOI = 0.05 nats（pretrained→SFT CE 变化的 ~20%）
- 三结局保留：A 明确稳定 association / B 明确无 association / C 现有 power 不足
  （CI 宽度 > SESOI 无法区分）→ 交用户裁定

---

## 12. 执行清单与 review gates

1. [ ] 用户 review v0.2 → 修订 → FROZEN
2. [ ] precision-ladder preflight（§3：32 样本 × 2 模型 × 3 levels，~10min GPU）
   → 报告 + **停下 review（定 B 或 C）**
3. [ ] 正式 FP32 evaluator 实现（`evaluation/eval_diag_fp32.py`，versioned，不改 frozen）
   + 单测 + 1-checkpoint 冒烟 → **停下 review**
4. [ ] bridge 精确清单冻结（sha256）→ bridge 批量（~2–3h，用户 tmux）
5. [ ] P1-1/2/3 + δ 分布 + OrderGap + lead-lag（正式证据链）分析 + leave-out 稳健性
6. [ ] 报告（三结局 + SESOI 敏感性）→ 停下，用户裁定 Phase 2 / Option B

**禁止**：新训练（Option B 明确 DEFER）、改 frozen 资产、把多 checkpoint/sample 当独立
training replicates、FP32 bulk 评估在 review gate 前运行。
