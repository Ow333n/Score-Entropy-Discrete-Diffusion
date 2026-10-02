# Experiment Protocol v4.2 — Attenuation Early Dynamics & Cross-Setting Reproduction

> **与 v4.1 的关系**：本协议是独立的新研究阶段。**不修改、不重新解释 protocol v4.1**；
> v4.1 的 Stage-4 判决 **B_DIAGNOSTIC_ONLY 永久固定**。v4.2 的回答对象是 v4.1 判决
> 所暴露的新问题，不是对判决的上诉。
>
> 协议版本：`v4.2`（本文件 + `regime_a_protocol_v4.2.yaml` 共同构成；规则冲突时以本文件为准）

## 1. 固定的核心现象（v4.1 产物，作为 v4.2 的输入事实）

> 在当前 SEDD-small + WikiText103 span-infilling 设置下，Vanilla SFT 使
> CPI 从 0.328 降到约 0.283–0.289、OrderGap 从 10.24 降到约 8.8，
> 而且主要变化在第一个 1020-step checkpoint 之前已经出现。
> （两 seed 一致，paired bootstrap 95% CI 均不含 0；v4.1 判决 B。）

## 2. 研究问题

1. **P1（早期动力学）**：这个 attenuation 究竟在训练的哪一步发生？step 0 → 1020 之间的
   变化形态是什么（连续衰减 vs 瞬时阶跃）？
2. **P2（跨设置可复现性）**：普通 post-training 是否在独立任务/数据分布/模型规模上
   **系统性重塑** reveal-order compatibility？变化方向是否依赖 task/data/model？
   —— 核心问题不是"证明 SFT 一定降低 incompatibility"，而是判断重塑是否存在与方向依赖。

## 3. P1 设计：密集 checkpoint 早期动力学

### 3.1 训练
- 从同一 pretrained checkpoint（`louaaron/sedd-small`）重新 Vanilla SFT，与 v4.1 相同 recipe
  （LR 3e-5、warmup 2500、eff batch 32、dropout 0、wd 0、AdamW、span 10-50、mean_over_masked_span）。
- seeds：复用 v4.1 正式 seeds **(1,1,1) 与 (2,2,2)**（与 v4.1 直接可比）。
- **dense analysis checkpoints**（预注册，在结果前冻结）：**50, 100, 250, 500, 750, 1020, 2500**
  （step 0 = pretrained 已有评估，不重训不重评）。每 checkpoint 保存 raw + EMA 双权重。
- **循环语义修正（v4.2 起）**：`n_iters = N` 表示**精确 N 个 optimizer steps**
  （v4.1 的 `step < n_iters+1` 实际跑了 N+1=10201 步，审计 ① 已记录；v4.2 训练脚本改为
  `step < n_iters`，run_metadata 记录 `loop_semantics: exact_N`）。
- n_iters = 2500（含 warmup 终点之后的 0 步，warmup 内的动力学是本研究的对象）。

### 3.2 评估（同一 frozen manifest `manifests/regime_a_eval_v1.jsonl`，N=500）
指标：CPI_abs、CPI_RMS、signed δ、TF OrderGap（固定 6 路径口径）、G1（masked NLL / token
acc / span 贪心命中）、buckets（pair-distance / sigma / span-length）。权重：**raw 与 EMA 都评**。

**预注册的 compute-tiered 评估顺序**（在结果前冻结，不做事后挑选）：
- **Tier 1（先跑）**：EMA 权重 × 全部 dense 点 × 2 seeds —— CPI 全指标 + buckets
  （先回答"何时发生"，最便宜）。
- **Tier 2**：EMA 权重 × 全部 dense 点 × 2 seeds —— TF OrderGap + G1。
- **Tier 3**：raw 权重 × 全部 dense 点 × 2 seeds —— CPI + TF OrderGap + G1。
- Tier 顺序只控制执行先后；**三个 Tier 全部预注册、全部要跑**，不用 Tier 1 结果裁剪 Tier 2/3。

### 3.3 P1 判读（预注册）
- 报告 per-checkpoint 轨迹（raw 与 EMA 分开），不预设"连续 vs 阶跃"。
- 关键比较：相邻 dense 点的 paired Δ（同一 manifest 样本），报告 bootstrap 95% CI；
  变化定位 = 第一个相邻点对 (s, s') 的 paired Δ 的 CI 不含 0 的位置。
- 两 seed 轨迹都要报告；方向不一致记为 `unstable`（沿用 v4.1 统计纪律，不加 seed 追显著）。

## 4. P2 设计：跨设置可复现性

**顺序**：P1 完成后进行。**go 条件**：无（P2 是复现性检验，不依赖 P1 的具体形态；
P1 的形态只影响叙事）。

- **P2a（数据分布变化，必做）**：openwebtext span-infilling——同一任务结构、同一模型、
  不同数据分布。复用 v4.1 recipe 与 manifest 协议（为 openwebtext 建新冻结 manifest，
  命名 `manifests/regime_a_openwebtext_eval_v1.jsonl`，σ 网格与样本构造规则与 v4.1 相同）。
  训练 N=10200、seeds (1,1,1)/(2,2,2)、analysis checkpoints 1020/5100/10200
  （沿用 v4.1 的三个分析点，P2 只回答"跨设置是否存在稳定 Δ"，不做密集动力学）。
- **P2b（任务族变化，尽力做）**：一个非 span-infilling 的独立任务（候选：code infilling、
  或构造的 synthetic 条件任务）。数据集与 G1 门槛按 v4.1 §15/§19 纪律在 pilot 前冻结，
  写 `protocol/regime_b_protocol.yaml`（若做）。若资源不允许，显式记录放弃理由。
- **P2c（模型规模，stretch）**：SEDD-medium 在本机（8GB）不可训练（静态 AdamW+模型+EMA
  ≈ 8.4GB，v4.1 已记录）。可选替代：自定义更小模型（如 hidden 512 / 8 blocks）从零
  wikitext 预训练数 epoch 后重复 P1-P2a 流程。是否执行在 P2a 完成后决定并预注册。

**P2 判读（预注册）**：每个新设置独立报告 ΔCPI（paired bootstrap CI，方向如实记录）。
"系统性重塑"的判据 = 该设置的 ΔCPI paired CI 不含 0（任意方向）；"方向依赖"的判据 =
  跨设置 ΔCPI 方向出现分歧（各设置 CI 均不含 0 且符号不同）。若新设置 CI 含 0，记录为
  未检测到重塑。

## 5. 机制消融（延迟，不在本协议执行）

data distribution / objective-reduction / LR / 训练长度等消融 **只在跨设置复现成立后**
另立 protocol 执行。PAPL/Swap 2×2 **不进入优先队列**：Vanilla 已自行降低 incompatibility，
"Swap 修复 SFT 制造的不兼容"的原始动机已被 v4.1 结果弱化（记录为科学结论，不作方法路线）。

## 6. 文献核查（持续任务）

重点：是否已有工作直接研究 **vanilla downstream SFT 前后** 的 reveal-order
compatibility / CPI / OrderGap 变化。核查结果记录于 `reports/literature_check.md`，
每条目标注：论文、设置、是否直接测量同一对象、与我们的发现的异同。
当前叙事口径：**"potentially novel empirical observation"**——不声称普适规律、
不提前确定创新。

## 7. 冻结规则（继承 v4.1 §19/§18A 精神，独立适用）

- 本协议在 P1 训练启动前冻结；结果出现后不改 checkpoint 列表、Tier 结构、判读规则。
- 实现 bug 修复需 protocol 修订号递增（v4.2 → v4.2.1）并重跑受影响部分。
- 统计纪律同 v4.1：per-seed 报告、paired bootstrap 仅代表评估样本不确定性、
  2 seeds 不作显著性、方向分歧记 `unstable`。
