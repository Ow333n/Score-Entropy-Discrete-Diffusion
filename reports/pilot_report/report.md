# Vanilla Pilot 过程性报告 — Regime A span-infilling SFT（protocol v4.1）

> **用途**：本报告是实验过程的证据文档，供外部 AI / 审阅者诊断后再启动正式实验。
> 所有原始数据在 `data/`（含 SHA-256 清单 `data/sha256s.txt`），图在 `figs/`。
> 协议版本 v4.1；正式 confirmatory 实验尚未开始，本报告覆盖 calibration 阶段全部产出。

**日期**：2026-10-01　**机器**：RTX 5060 Ti 8GB（WDDM/WSL2，单卡）　**模型**：SEDD-small（169.6M，absorbing）

---

## 0. 一分钟摘要

1. **LR 校准**：3e-4 发散（eval 7.17→8.26@4500）、1e-4 摇摆（warmup 后 7.30→7.31→7.26）、**3e-5 单调下降（7.27→7.19@3500）→ 选定 3e-5**。
2. **Pilot**（lr 3e-5，30k 步）：eval 7.55@500 → **7.05@6500 最优** → plateau 检测触发于 8500 → **N = min(30000, ⌈1.2×8500⌉) = 10200**；20000 步后出现晚段过拟合（+0.3 nats/10k 步），N=10200 恰好避开。
3. **G1 pilot 检查通过**（step-30000 EMA vs pretrained，冻结 manifest N=500）：masked NLL 4.281→3.828（−10.6%）、token acc 0.322→0.362、贪心命中 0.443→0.472。
4. 协议已冻结：N=10200、LR=3e-5、analysis checkpoints [1020, 5100, 10200]、2 formal seeds。正式实验**未开始**。

---

## 1. 协议上下文

- Canonical 协议：`protocol/experiment_protocol.md`（v4.1，执行规则唯一权威）+ 主计划 `post_training_reveal_order_compatibility_local_plan_v4.1_FROZEN_FINAL.md`。
- Regime A 冻结文件：`protocol/regime_a_protocol.yaml`（副本在 `data/regime_a_protocol.yaml`）。
- Pilot 的角色（§3.1/§9.2）：calibration——确定稳定 LR、总预算 N、span 难度、256 管线吞吐；pilot 本身不进入最终方法比较，不用于 seed variance 估计。
- **Pilot 参数在运行前已冻结**：N_pilot_max=30000、eval_interval M=500、plateau_window K=6、相对改善阈值 ε=0.5%、安全系数 α=1.2、pilot_seed=0。

## 2. 任务定义与管线

- **任务**（§7.2）：wikitext103 span-infilling——seq_len 256，span 10~50 token；span 外上下文**恒可见**，span 内 partial-absorbing 腐蚀（每位置以 1−e^{−σ} 概率变 [MASK]），σ 走训练分布 t~U[ε,1]→loglinear。
- **损失**（§7.3，reduction 已冻结为 `mean_over_masked_span`）：(dsigma·SE) 在 span 内 mask 支撑集上按序列均值 → batch 均值。只对 span 内当前 mask 位置计算。
- **评估**：验证集 64 块、每次 eval 同一腐蚀种子（确定性、跨 checkpoint 可比）、EMA 权重。
- **显存修复**（Day 4）：SEDD train 模式每 block 保留 ~540MB 激活（fp32 LayerNorm 输出+MLP 中间量+autograd 图），12 blocks 在 batch 32×256 下 ≈6.5GB 必超 8GB 并触发 WDDM 换页（实测 0.2 steps/s）。修复：12 个 block 加梯度检查点（train 专用；dropout=0 使 backward 重算逐位一致，forward 输出不变）。修复后 **5.73GB / 2.1 steps/s**。
- **Batch 语义更正**（G0 §4.1 运行时核实）：data.py 把 `config.batch_size // (ngpus×accum)` 当 micro-batch → **config batch_size 本身就是 effective batch**。本管线 config batch 32 = micro 32 × accum 1 = eff 32（每步 8192 tokens）。

## 3. LR 校准（探针数据，图 `figs/fig2_lr_probe_comparison.png`）

| LR | 探针长度 | warmup 后 eval 轨迹 | 判定 |
|---|---|---|---|
| 3e-4 | 5000 步（被终止的 pilot） | 7.17@1500 → 8.01@2500 → **8.26@4500 发散** | ✗ 弃用 |
| 1e-4 | 3500 步 | 7.30@2500 → 7.31@3000 → 7.26@3500（摇摆） | 备选 |
| **3e-5** | 3500 步（+30k pilot） | 7.27@2500 → 7.24@3000 → **7.19@3500 单调** | ✓ **选定** |

判据：warmup 结束后单调下降 + 最终值最优。3e-4 的发散签名（warmup 结束即过拟合上升）与 1e-4 的摇摆都排除；3e-5 动力学最健康。

## 4. Pilot 结果与 N 确定（图 `figs/fig1_pilot_learning_curve.png`）

- 吞吐 2.2 steps/s（17.8k tokens/s），峰值显存 6.41GB，全程无 NaN/Inf。
- **eval 轨迹**：7.55@500 → 7.05@6500（最优）→ 7.13~7.19（12500-20000 平台）→ 7.48@29500（晚段过拟合，+0.3 nats/10k 步）。
- **Plateau 检测**（§3.1）：step 8500 时连续 K=6 次 eval 相对改善全部 <0.5% → s_p=8500 → **N = min(30000, ⌈1.2×8500⌉) = 10200**。
- N=10200 落在 eval 良好区域（~7.2），且避开 20k 后的过拟合段。early/mid/late analysis checkpoints（§3.2）= **1020 / 5100 / 10200**。

### 观察 1（供诊断）：plateau 规则对 eval 噪声的敏感性
ε=0.5% 相对改善阈值对 eval 噪声（±0.5-1%）偏严：噪声改善（如 1500→2000 变差后出现的 +0.6%）会重置 K 窗口，使检测推迟。本 pilot 在 8500 触发纯属噪声窗口恰好满足。协议已有兜底分支（不触发则 N=N_max=30000），未修订。**问题**：ε 或 K 是否需要 protocol revision（v4.2）？我的立场：不动（已冻结 + 结果合理），但欢迎外部意见。

### 观察 2（供诊断）：晚段过拟合
lr 3e-5、wd=0、eff batch 32 下，20k 步后 eval 缓慢上升 +0.3 nats。N=10200 避开该区域。若未来需要更长训练（如 Regime B），wd>0 或更小 LR 值得校准。

## 5. G1 pilot 检查（图 `figs/fig3_g1_comparison.png`）

协议 §9.4 三项：非平凡提升、非 near-zero、非饱和——全部满足。

| 指标（冻结 manifest，N=500） | pretrained | pilot SFT（step-30000 EMA） | 变化 |
|---|---|---|---|
| masked-token NLL (nats) | 4.281 ± 0.081 | **3.828 ± 0.079** | **−10.6%** |
| token accuracy | 0.322 ± 0.010 | **0.362 ± 0.010** | +4.0pp |
| span 贪心 per-pos 命中率 (l2r, n=100) | 0.443 | **0.472** | +2.9pp |
| span 精确匹配率（辅助） | 0.02 | 0.01 | — |

注：这是 step-30000 的数字；正式 N=10200 的 checkpoint 处于 eval 更优区域，预期更好。span 精确匹配率对 10-50 token span 预期近零，仅作辅助信号，不作为 gate。

## 6. 已冻结的正式协议（protocol v4.1，`data/regime_a_protocol.yaml` 的 `training_frozen:`）

| 项 | 冻结值 |
|---|---|
| N（optimizer steps） | **10200**（plateau 规则：s_p=8500, α=1.2） |
| analysis checkpoints | **1020 / 5100 / 10200**（§3.2 显式保存） |
| LR / schedule | **3e-5** / linear warmup 2500 |
| optimizer | AdamW(β 0.9/0.999, eps 1e-8), wd=0, grad_clip 1.0 |
| EMA | 0.9999（eval 用 EMA 权重） |
| dropout | 0.0（§9.1 所有条件统一） |
| batch | eff 32（config 32, micro 32, accum 1） |
| seq/span | 256 / [10, 50]，σ 网格评估用 {0.5,1.5,3.0} |
| formal seeds | [model, data_order, corruption] = (1,1,1) 与 (2,2,2)（§3.11；eval manifest 不变） |
| λ（Swap） | 待 §14 梯度范数校准后冻结 |

## 7. 既有冻结基线（上下文，非本报告新产出）

- pretrained Data CPI（manifest 同版）：CPI_abs = 0.328 ± 0.023（N=500）；δ 无偏（+0.003±0.027）；δ_SD=0.605（效应量基线）；CPI 随 pair_distance 强烈下降（距离 1-4→0.68，15-41→0.11）。
- pretrained TF OrderGap：raw = 10.24 ± 0.35 nats；per-token 0.605 ± 0.031；confidence 路径 Q 最差（−65.8 vs 其他 −61~−62）。
- 评估 manifest：`manifests/regime_a_eval_v1.jsonl`（SHA-256 冻结，§5A）。

## 8. 过程中踩过的坑（已修复，供审计）

1. **EMA 加载**：EMA state_dict 是 `{decay, num_updates, shadow_params:[tensor...]}` 非按参数名索引；`model.load_state_dict(ema_dict, strict=False)` 会静默空载 → 输出层零初始化 → NLL=ln(50257)=10.81 零方差（该签名=均匀分布）。已改用 `ema.load_state_dict + copy_to`。
2. **SEDD train 显存**：见 §2，梯度检查点修复。
3. **batch 语义**：见 §2，Day 3 记录的 "batch 4/accum 2=eff 8" 有误，实际 eff=4。
4. **plateau 参数未传参 bug**：训练脚本一度忽略 CLI overrides（pilot 误跑成 300 步 smoke），已修复。

## 9. 文件清单（全部 SHA-256 见 `data/sha256s.txt`）

```
report.md                                 本报告
figs/fig1_pilot_learning_curve.png        pilot 学习曲线 + N 确定标注
figs/fig2_lr_probe_comparison.png         三 LR 探针对比
figs/fig3_g1_comparison.png               G1 pretrained vs pilot
data/pilot_learning_curve.csv             pilot 全曲线（60 个 eval 点）
data/pilot_run_metadata.json              §27 协议字段（吞吐/显存/seeds/曝光量）
data/lrprobe_{3e-4,1e-4,3e-5}_learning_curve.csv
data/smoke256_run_metadata.json           256 管线 smoke 校准
data/g1_{pretrained,pilot}.json           G1 数字
data/cpi_pretrained_baseline.json         冻结 CPI 基线
data/order_gap_pretrained_baseline.json   冻结 OrderGap 基线
data/regime_a_protocol.yaml               冻结协议（副本）
```

## 10. 请外部诊断的问题

1. **N 的选择**：plateau 规则给出 N=10200（eval 最优在 6500）。这个预算对 4 条件 × 2 seeds 的 2×2 是否合理？有无更好的预算规则？
2. **晚段过拟合**（20k 步后 +0.3 nats）：170M 模型 + wd=0 + eff 32 的典型行为？正式实验是否应加 wd 或降 LR（若改，按 §3.3B 所有条件共享）？
3. **plateau 检测的 ε/K**：ε=0.5% 对 eval 噪声 ±0.5-1% 是否过严（观察 1）？是否值得 protocol revision v4.2？
4. **G1 提升幅度**：NLL −10.6%、acc +4pp 在 170M 模型 + span infilling 上是健康的 SFT 信号吗？对后续 CPI/OrderGap 的动力学测量（G2）够用吗？
5. **损失 reduction 选择**（mean_over_masked_span）：与 repo 原生 sum-reduction 的差异是否会影响 ΔCPI 的解释？
6. 其他任何方法论/实现疑点。

---

*报告生成：Claude Code（dataviz 规范配图）；数据可复现（所有 run 的 seeds/命令记录于 run_metadata.json）。*
