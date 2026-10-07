# Phase 1 — Compatibility–Estimation Dynamics：Protocol DRAFT v0.1

**状态：DRAFT（待用户 review；不执行、不冻结）**
**前序**：v1.2（G1/G2 INC）、v2.1（G3a PASS、G3b-d INC、Case C）已关闭；frozen 报告
`reports/mechanism_pilot_v2_1_report.md`（sha256 `dd3cdb26…`）
**路线**：停止第四种 mask policy；转向"为何普通 SFT 普遍降低 reveal-order incompatibility"
**原则**：observation / mechanism-diagnostic study——即使 CPI 与 CE 强相关，也不能直接声称
"CE improvement causes CPI reduction"。

---

## 1. 核心研究问题

> SFT 过程中 conditional estimation improvement 是否与 CPI attenuation 稳定对齐？

子问题：
1. FP32 conditional CE/NLL 是否随 SFT 稳定下降？
2. FP32 CPI_abs 是否随 SFT 稳定下降？
3. within-run 的 ΔCE 与 ΔCPI 是否在多个 run/policy 中呈一致关联？
4. （temporal）CE improvement 是否系统性早于 CPI attenuation？

---

## 2. 资产审计（Phase 1.0，已执行，2026-10-07）

### 2.1 权重资产（可重跑新 evaluator）

| 资产 | checkpoints | 容器 | 备注 |
|---|---|---|---|
| pretrained（step0） | mechpilot_shared/checkpoint_0000.pth | {model, ema, step} | 全体共享 init |
| formal v4.1 | s1/s2 × {1020, 5100, 10200} | 全量 {model,ema,opt,scaler,step} | lr=3e-5，10200 步 |
| p1 v4.2 dense | s1/s2 × {2500} + meta | 全量 | **中间 dense（50–1020）权重已退役删除** |
| v1.2 mechpilot | A/B/C/D ×2 × {500,1020,2500} + raw | {ema}/{model} | lr=3e-4 |
| v2.1 pilot | U/H/L ×2 × {500,1020,2500} + raw | {ema}/{model} | lr=3e-4 |
| RL | rlpilot eval_snapshot ×{50,100,250,500} + step500 | {model,ema,...} | RL 阶段 |

### 2.2 仅 frozen JSON（旧 bf16 evaluator，无权重）

- results/p1_dense/：84 文件（s1/s2 × 7 dense steps × cpi/og/g1 × ema/raw）——dense 轨迹的
  唯一现存来源（CE 为 bf16 量化口径）
- results/vanilla/（formal v4.1 18 评估）、results/mechanism_pilot/（v1.2 53）、
  results/mechanism_pilot_v21/（v2.1 76+分析）、results/pretrained/（基线）

### 2.3 方差分解（基于现有 per-sample 数据，N=500）

| 成分 | 估计 | 来源 |
|---|---|---|
| D. sample-level | per-sample \|δ\| SD = 0.45–0.54；500-sample SE = 0.020–0.024 | 各 run JSON |
| A. within-run（500/1020/2500 三点） | SD 0.002–0.017（中位 ~0.008） | v1.2/v2.1 轨迹 |
| B. between-seed @2500 | spread 0.002–0.037（U 类 0.037 为最大） | 7 组 policy×seed |
| C. between-policy @2500（同 rep） | spread 0.025 / 0.035 | v2.1 |
| pooled run-to-run | **14 runs SD = 0.0113**（range 0.264–0.309） | v1.2+v2.1 |
| pretrained→SFT 衰减 | 0.3281 → 0.2786（−0.0495 ≈ 4.4× pooled SD） | 稳健可检 |

### 2.4 功效结论

- paired policy contrast（H vs L 型）的 per-seed 差估计 SE ≈ 0.010–0.017（sampling），
  between-seed 差的 SD ≈ 0.02 → 2 seeds 下可检效应 ≳ 0.035；4 seeds ≳ 0.025；8 seeds ≳ 0.018。
- **SESOI（smallest effect size of interest）= 0.02 CPI_abs**（≈40% 的衰减尺度 0.05；
  低于此的 policy 效应在本设置下无科学意义）。
- 结论：**现有 2-seed 资产功效不足**；Phase 1 以 within-run 关联（N 大）为主统计，
  between-run 结论仅 exploratory；不把多个 checkpoint 当独立 training replicate。

---

## 3. Phase 1.1 — FP32 diagnostic evaluator 设计（新 versioned evaluator）

**定位**：新 diagnostic evaluator，**绝不替换** frozen v1.2/v2.1 evaluator；旧结果保持原样。

**精度边界（已探针确认，2026-10-07）**：
- frozen 实现的 model forward 内部启用 bf16（`model/transformer.py:282` autocast，Day-1 修复
  的一部分，FROZEN 不可改）→ score 输出 bf16 → 旧 evaluator 全链 bf16（实测：NLL 2⁻⁵/2⁻⁶、
  δ 2⁻⁶/2⁻⁷、CPI_abs 2⁻⁹ 网格）
- 新 evaluator 规格：
  1. model forward **保持原样**（bf16 输出，不修改 frozen model 代码）
  2. `score.float()` 后立即提取：log_softmax **fp32**、δ **fp32**、CE/NLL **fp32**
  3. aggregation（均值/分位数/SD）**fp64**
  4. 明确记录剩余限制：score 值本身仍为 bf16 舍入（相对 ~2⁻⁸–2⁻⁹）——**不假装 fp32 evaluator
     = ground truth**
  5. 确定性检查（同输入逐位一致）+ provenance：git commit / manifest sha / 自身文件 sha256
     sidecar / DIAG_PROTOCOL_VERSION=v1
- 模块：`evaluation/eval_diag_fp32.py`（新文件，不动任何 frozen 文件）；输出
  `results/phase1_diag/`
- **review gate：实现 + 单测 + 1 checkpoint 冒烟后停下，用户 review 通过才允许批量运行**

---

## 4. Phase 1.2 — 新旧 evaluator bridge（预注册）

**Bridge set（11 checkpoints，训练前冻结）**：
1. pretrained（step0）
2. p1-s1：2500（+ formal-s1：1020 / 10200 作长程补充）
3. p1-s2：2500（+ formal-s2：1020 / 10200）
4. v2.1：U1/H1/L1 @1020、@2500（6 个）
5. v1.2：A1 @1020、@2500（2 个）

（最终 11 个精确清单在批量运行前冻结落盘。）

每 checkpoint 同时跑：old frozen evaluator（eval_cpi + eval_task）与 new FP32 evaluator。
比较（全部落盘）：CPI_abs / CPI_RMS / signed δ / conditional CE-NLL / token acc 的
绝对差、rank/order preservation（Spearman ρ）、Pearson r、**衰减方向是否一致**。
判定目的：旧 evaluator 的历史结论与新 evaluator 的 dynamics trend 是否可衔接。
**禁止**为了让两套数字一致去调新 evaluator。

---

## 5. Phase 1.3 — Primary / Secondary 预注册

**Primary（三个，不扩散）**：
- **P1-1**：FP32 conditional CE/NLL trajectory 是否随 SFT 稳定下降？（within-run 轨迹 +
  pretrained→SFT 方向；判定=同向 + CI 排除 0）
- **P1-2**：FP32 CPI_abs trajectory 是否随 SFT 稳定下降？（同上）
- **P1-3**：within-run ΔCE vs ΔCPI 是否在多个 run/policy 中一致关联？（per-run 相邻
  checkpoint 对的 Δ 配对，跨 run pooled Spearman/Pearson + bootstrap CI；**观测性关联，
  不声称因果**）

**Secondary**：CPI_RMS、signed δ、δ 分布（见 §6）、OrderGap（§7）、path-score variance、
score scale、entropy、calibration（reliability）、token acc。

---

## 6. δ 分布报告规范（机制意义）

不得只报 mean |δ|。每 checkpoint 必须报告：median |δ|、P75、P90、P95、RMS、signed mean、
positive/negative fraction、outlier fraction（定义：|δ| > 2×δ_SD，或 > P95，二者都报）。
判读问题：CPI attenuation 是 **A. 整分布左移** 还是 **B. 少数 extreme curl outlier 减少**——
对机制解释有本质区别。方法：per-quantile 的 step0→SFT 变化表（P50/P75/P90/P95/RMS 各自
的相对变化）。

---

## 7. OrderGap 定义固定（命名规范）

- 名称：**finite-path OrderGap**（非"全排列 OrderGap"）
- 定义：在 **固定 6 条冻结路径**（l2r / r2l / random_0/1/2 / confidence，manifest paths 字段，
  构建期冻结）上计算 Q_π，OrderGap = max_π Q − min_π Q；**明确不声称枚举 m! 排列**。
- 报告：n_paths=6、mean/SEM、per-sample path-score variance、sampling uncertainty
  （path-set 的有限性说明）；span length 分布（m̄≈21.7）。
- 小 span exhaustive 子集（m ≤ 7 才可枚举）单独报告（仅当 manifest 中存在小 span 样本；
  当前 m∈[10,50] 无 exhaustive 子集——记为 deferred）。

---

## 8. Phase 1.4 — Lead-lag / temporal ordering diagnostic（重要新分析）

**资产现实**：p1 dense 中间权重（50/100/250/500/750/1020）已退役删除；现存：
- 选项 A（零训练）：用 results/p1_dense 的旧 JSON（bf16 口径 CE，量化 0.0156–0.031；
  ΔCE 信号 ~0.05–0.1/间隔，勉强可用）做 lead-lag——作为 **exploratory pre-analysis**
- 选项 B（需用户批准）：Gate-2 已证明 P1 recipe 逐字节可复现（checkpoint_1020 位级一致）→
  **确定性重训恢复 dense 权重**（2 × ~18min、~33GB 磁盘）→ 跑 FP32 evaluator 得精口径
  dense 轨迹。这不是新实验，是已退役资产的确定性再生；但属训练时间，**待用户裁定**。

**定义（选项 A/B 通用）**：
- 时间轴 = checkpoint step（50/100/250/500/750/1020/2500），7 点
- per-run：ΔCE(t) = CE(t) − CE(t−1)，ΔCPI(t) 同理（t 为相邻 checkpoint 索引）
- 统计：within-run lag-1 cross-correlation——corr(ΔCE_t, ΔCPI_{t+1}) vs
  corr(ΔCPI_t, ΔCE_{t+1})，per-run 计算 + 跨 run 汇总（bootstrap）
- **命名纪律**：lead-lag / temporal ordering diagnostic；**禁止称 Granger causality**
  （checkpoint 数少、非时间序列设定）
- 判读：CE 先降且 CPI 后降 → 支持 "estimation improvement precedes compatibility
  attenuation"；同步 → 只支持 co-evolution；CPI 先降 → generic estimation-error
  hypothesis 被削弱。

---

## 9. 统计计划

1. 每条 training run 独立 trajectory（主报告元素，不折叠）
2. effect size + bootstrap CI（10k，seed=0，v1.2 同款）
3. pooled exploratory regression：CPI ~ CE + checkpoint + (1|run)（mixed-effects 仅
   exploratory；n_runs 小，p-value 不作主结论）
4. within-run centering：CPI_rt − mean_r(CPI) vs CE_rt − mean_r(CE)，减少 seed-level
   offset 干扰

---

## 10. 多重比较

- Primary 三问：**不做 FDR 扩散**（各自独立预注册判据）
- Secondary exploratory：大量 correlations/metrics 时用 **BH-FDR**；报告 raw p、adjusted q、
  effect size、CI；**禁止从 secondary 中挑显著结果升级成 primary**。

---

## 11. SESOI 与 stopping rule

- SESOI = **0.02 CPI_abs**（依据 §2.4）；CE 侧 SESOI = 0.05 nats（对应 pretrained→SFT
  CE 变化 4.28→~4.03 的同阶 ~20%）。
- Phase 1 结束三结局（全部允许）：
  - **A. 明确稳定 association**（P1-3 一致关联 + lead-lag 一致方向）→ 进入 Phase 2 设计深化
  - **B. 明确无明显 association**（P1-3 与 lead-lag 均无一致信号）→ 机制假设修正，重新设计
  - **C. 现有 power 不足**（CI 宽度 > SESOI 且无法区分）→ 记录 power 需求，交用户裁定
    （加 seeds / 恢复 dense / 放弃该路线）

---

## 12. 执行清单与 review gates

1. [ ] 用户 review 本 DRAFT → 修订 → FROZEN
2. [ ] FP32 evaluator 实现 + 单测 + 1-checkpoint 冒烟 → **停下 review**
3. [ ] bridge set 精确清单冻结落盘（sha256）
4. [ ] bridge 批量运行（11 × 2 evaluators，~2–3h GPU，用户 tmux）
5. [ ] P1-1/P1-2/P1-3 + δ 分布 + OrderGap + lead-lag（选项 A 先行）分析
6. [ ] 报告（三结局判定 + SESOI 对照）→ 停下，用户裁定 Phase 2 / 选项 B

**GPU/磁盘估算**：bridge ≈2–3h；FP32 evaluator 与旧 evaluator 同为 inference-only；
lead-lag 选项 B（若批准）= 2×~18min + ~33GB。
**禁止**：新训练（除选项 B 明确批准）、改 frozen 资产、把多 checkpoint 当独立 replicate。
