# Stage-4 Gate 过程性报告 — 正式 Vanilla SFT 与兼容性动力学（protocol v4.1）

> **用途**：正式实验阶段 (G0.5 → G1 → G2 → Stage-4) 的完整证据文档，供外部 AI / 审阅者审批。
> 所有原始数据在 `data/`（含 SHA-256 清单），图在 `figs/`。
> **Stage-4 判决：B_DIAGNOSTIC_ONLY**（判据预先注册于 `protocol/regime_a_protocol.yaml`，结果后未修改）。

**日期**：2026-10-01 ~ 10-02　**机器**：RTX 5060 Ti 8GB 单卡　**模型**：SEDD-small（169.6M，absorbing）　**任务**：wikitext103 span-infilling（256-seq，span 10-50）

---

## 0. 一分钟摘要

1. 两个正式 Vanilla run（seeds (1,1,1)/(2,2,2)，N=10200，LR=3e-5）从 pretrained 重新训练完成，各保存 1020/5100/10200 三个分析点（raw + EMA 双权重），全程零 NaN。
2. G0.5 state-support gate **通过**（6/6）：训练态与冻结评估 manifest 在 σ、mask ratio、K、span、pair 规则上覆盖合理。
3. **G1 通过**：SFT 后 masked NLL 4.281→3.875（−9.5%）、token acc 0.322→0.362、贪心命中 0.443→0.47（两 seed 几乎一致）。
4. **Stage-4 判决 B_DIAGNOSTIC_ONLY**：
   - **ΔCPI stable** ✅：SFT 使 CPI_abs 0.328→~0.286（**−12~14%**），两 seed 同负、paired bootstrap 95% CI 均不含 0。
   - **ΔOrderGap stable** ✅：OrderGap_raw 10.24→~8.8（**−14%**），两 seed 同负、CI 不含 0。
   - **联动判据不成立** ❌：early→late 轨迹平坦（CPI_el CI 含 0）——效应的实际模式是**"warmup 期内瞬时阶跃 + 之后平台"**，而非随训练逐步演化。
5. **方向与原始 H1 假设相反**：Vanilla SFT（无任何兼容性正则）本身减弱了揭示顺序不兼容与顺序敏感度。

---

## 1. 预注册判据（结果前冻结，未修改）

`protocol/regime_a_protocol.yaml` 的 `stage4_gate_pre_registered`（完整副本在 `data/regime_a_protocol.yaml`）：

```
stable_criterion:    paired bootstrap 95% CI (10k) of mean delta 不含 0, 两个 seed 符号一致
co_movement_criterion: sign(CPI_late - CPI_early) == sign(OG_late - OG_early),
                     两者的 paired bootstrap 95% CI 均不含 0, 两个 seed 判断一致
gate_outcomes:       A_HARD_STOP / B_DIAGNOSTIC_ONLY / C_PROCEED
seed_disagreement:   方向相反 → unstable, 不加 seed
```

Primary 口径（同样预注册）：EMA 权重、late checkpoint (10200)、固定 6 路径 OrderGap（与 pretrained 同口径）、adaptive 路径分开报告、sampled rollout greedy/T=1.0 为 §3.9 主结果。

## 2. 正式训练 run 审计

| 项 | s1 (formal-vanilla-s1-191414) | s2 (formal-vanilla-s2-204215) |
|---|---|---|
| optimizer steps | 10201 | 10201 |
| 时长 / 吞吐 | 4474s / 2.3 steps/s | 5175s / 2.0 steps/s |
| 峰值显存 | 6.41 GB | 6.41 GB |
| NaN/Inf | 0 | 0 |
| analysis checkpoints | 1020 / 5100 / 10200 ✓ | 1020 / 5100 / 10200 ✓ |
| 权重 | raw + EMA 双份 | raw + EMA 双份 |

### 审计注记 1：训练期 eval 监控器的腐蚀种子噪声
s1 的训练期 eval 曲线 (~8.6-8.7) 显著高于 pilot (~7.1-7.2)，交叉诊断（`scripts/cross_eval_check.py`）证明是 **eval 腐蚀种子噪声**而非模型差异：同一模型换 eval 种子后 eval 7.05↔8.67；同一 eval 种子下 s1-10200 (7.049) 反而优于 pilot-30000 (7.420)。根因：训练期 eval 只抽 64 个 t，loglinear 调度的 dsigma 权重跨 t 变化 3 个数量级。**正式评估用 frozen manifest 的固定 σ 网格 {0.5,1.5,3.0}，不受影响**。

### 审计注记 2：实现核查（正式实验前完成 + 事后复核修正）
- Scheduler：warmup=2500 后 LR 位级恒定 3e-5；trajectory 与 n_iters 无关（`scripts/check_scheduler.py`）。
- EMA：存在 decay warmup（effective_decay = min(0.9999, (1+n)/(10+n))），**修正后的正确数值**：n=1 时 0.1818、n=1020（early ckpt）0.9913、n=5100 时 0.9982、n=10200（late）0.9991（此前报告"0.9990→0.99912"有误，已更正）；无 bias correction；eval/checkpoint 用 shadow 权重（`scripts/check_ema.py`，torch 镜像逐位验证）。含义：EMA 在 early checkpoint 以 decay≈0.991（半衰期 ~70 步）紧密跟踪模型，在 step 50 时 decay=0.85（EMA≈近 6 步均值）——对 v4.2 密集 checkpoint 研究重要。
- **审计更正：两个 formal run 实际 optimizer steps = 10201（非 10200）**。根因：训练循环条件 `step < n_iters + 1`（继承 repo run_train.py 惯例）→ 步数 0..10200 共 10201 步。分析 checkpoint 保存于精确的 1020/5100/10200 ✓。偏离 0.01%，不影响任何分析结论，v4.2 将循环改为精确 N 步。
- pilot 24k 步 train-loss spike 与 late-stage validation degradation：仅作审计记录，不据此改正式 recipe（预注册规定）。

### 审计注记 3：评估中断
正式评估（18 项）曾被用户开游戏导致的 GPU 争抢中断一次；每项评估独立可重跑，最终 18/18 全部完成落盘，无半截文件。

## 3. G0.5 state-support gate — PASS ✅

| 检查 | 结果 |
|---|---|
| σ 网格 {0.5,1.5,3.0} ⊂ 训练支持 [1e-3, 6.9] | ✅（P(σ≤3.0)=95.1%） |
| manifest C 状态 (K=0) 训练可达 | ✅ P(K=0)=3.8% |
| C+a/C+b 状态 (K=1) 训练可达 | ✅ P(K=1)=3.9% |
| span 长度同分布 U[10,50] | ✅ |
| pair 规则一致（masked span 内随机对） | ✅ |
| partial-reveal 任意子集可达 | ✅（按构造） |

limitation 注记：K=0/1 状态在训练中占比 ~3.8-3.9%（低 σ 区），但 manifest σ 网格全部位于训练常见区；CPI_K sensitivity 已按 bucket 报告（§8.3）。

## 4. G1 正式结果（frozen manifest，EMA 权重，N=500）

| 指标 | pretrained | s1-1020 | s1-5100 | s1-10200 | s2-1020 | s2-5100 | s2-10200 |
|---|---|---|---|---|---|---|---|
| masked NLL | 4.281±0.081 | 4.031 | 3.906 | 3.875 | 4.031 | 3.906 | 3.875 |
| token acc | 0.322±0.010 | 0.344 | 0.359 | 0.362 | 0.342 | 0.354 | 0.363 |
| span 贪心命中 | 0.443 | 0.454 | 0.467 | 0.467 | 0.454 | 0.468 | 0.471 |

§9.4 三项（非平凡提升、非 near-zero、非饱和）全部满足；两 seed 高度一致（max |Δ| < 0.01）。pilot step-30000 的 G1（NLL 3.828）仅为 learnability sanity，正式 G1 以本表为准。

## 5. G2 兼容性动力学（图 `figs/fig1_cpi_trajectory.png`、`figs/fig2_ordergap_trajectory.png`）

### CPI 轨迹（图 1）

| | pretrained | early 1020 | mid 5100 | late 10200 |
|---|---|---|---|---|
| s1 | 0.3281 | 0.2871 | — | 0.2891 |
| s2 | 0.3281 | 0.2793 | — | 0.2832 |

**关键模式：pretrained→early 的阶跃就是全部效应；early→late 平台（Δ ≈ +0.002）。**

### OrderGap 轨迹（图 2，固定 6 路径口径）

| | pretrained | early 1020 | mid 5100 | late 10200 |
|---|---|---|---|---|
| s1 | 10.244 | 8.595 | — | 8.756 |
| s2 | 10.244 | 8.825 | — | 8.817 |

同样模式：阶跃发生在 pretrained→early，early→late 在噪声内（s1 +0.16 / s2 −0.01，符号甚至相反但均在 CI 内）。

### 分桶（late，pair_distance）——SFT 后距离结构保留

| 距离桶 | pretrained | s1-10200 | s2-10200 |
|---|---|---|---|
| 1-4 | 0.680 | 0.594 | 0.625 |
| 4-8 | 0.357 | 0.334 | 0.287 |
| 8-15 | 0.171 | 0.128 | 0.128 |
| 15-41 | 0.109 | 0.097 | 0.092 |

相邻 mask 对不兼容 > 远距离对的结构在 SFT 后保持，所有桶同步下降。

### Adaptive 置信路径（分开报告，不与 pretrained 直接比）

含 adaptive 路径的 OrderGap（7 路径）：s1-10200 = 9.08，s2-10200 = 9.16（vs 固定 6 路径的 8.76/8.82）——本 checkpoint 自己的置信路径比 pretrained 生成的固定置信路径更差，与 pretrained 阶段的观察（confidence 路径 Q 最差）一致。

## 6. Stage-4 gate 执行（图 `figs/fig3_gate_evidence.png`）

| 判据 | s1 | s2 | 结论 |
|---|---|---|---|
| Δ\|δ\|_abs 均值 (late vs pre) | −0.040 | −0.046 | — |
| paired bootstrap 95% CI | [−0.072, −0.007] | [−0.078, −0.014] | **CPI stable** ✅（同负） |
| ΔOG 均值 (late vs pre) | −1.489 | −1.427 | — |
| paired bootstrap 95% CI | [−2.123, −0.936] | [−2.052, −0.869] | **OG stable** ✅（同负） |
| ΔCPI(early→late) CI | [−0.024, +0.026] | [−0.024, +0.031] | 含 0 |
| ΔOG(early→late) CI | [−0.096, +0.422] | [−0.309, +0.288] | 含 0 |
| early→late 符号一致 | +/+ | +/− | ❌ **联动不成立** |

**GATE VERDICT: B_DIAGNOSTIC_ONLY**（`data/stage4_gate.json`）

## 7. 科学解读

1. **方向与 H1 原始假设相反**：proposal H1 猜测 "SFT 制造/放大不兼容"；实测 Vanilla SFT（无任何兼容性正则）**减弱**不兼容（CPI −12~14%）并**降低**顺序敏感度（OG −14%）。对应 proposal §24 Outcome B：*"ordinary post-training attenuates incompatibility"*。
2. **效应动力学是"瞬时阶跃 + 平台"**：全部效应在 warmup 期（≤1020 步）内完成。这解释了联动判据为何失败——预注册判据测的是 early→late 轨迹同动，而实际没有轨迹可动。这是一个**判据设计与现象模式不匹配**的情况（不是"无效应"）。
3. **为什么不能据此改判据直接进 C**：预注册纪律（§19 Formal Freeze Rule）要求按原判据执行；判据修订需要 protocol revision v4.2 + 设计缺陷论证 + 受影响分析重跑，不能因为"结果模式不在预期内"而事后改。
4. **对方法路线（PAPL/Swap）的含义**：B 判决下，Swap-SFT 的动机弱化——Vanilla SFT 已免费获得兼容性收益，Swap 的增量价值存疑。这本身就是 proposal §13.2 预想的 major falsification 的变体（"compatibility 可能只是普通 post-training 的副产物"）。

## 8. 审计注记汇总

- [x] 评估口径（EMA/late/固定路径）预注册且未变
- [x] 18/18 评估完成，全部带 per-sample 数据（paired 口径）
- [x] pretrained 基线在加 per-sample 字段后重跑（同 manifest、同协议，数值与首次一致：CPI 0.3281、OG 10.244）
- [x] 训练期 eval 噪声、游戏中断、24k spike 等均已记录且不影响正式结论
- [x] 无任何临时改超参/选 checkpoint 行为

## 9. 文件清单（SHA-256 见 `data/sha256s.txt`）

```
report.md                          本报告
figs/fig1_cpi_trajectory.png       CPI 轨迹（阶跃+平台）
figs/fig2_ordergap_trajectory.png  OrderGap 轨迹
figs/fig3_gate_evidence.png        判决证据（paired Δ + bootstrap CI）
figs/fig4_g1_trajectory.png        G1 任务指标轨迹
data/stage4_gate.json              判决与全部判据数字
data/cpi_pretrained.json / og_pretrained.json / g1_pretrained.json
data/cpi_{s1,s2}_{1020,5100,10200}.json    (6, 含 per-sample δ)
data/og_{s1,s2}_{1020,5100,10200}.json     (6, 含 per-sample OG 与 adaptive)
data/g1_{s1,s2}_{1020,5100,10200}.json     (6)
data/g05_state_support.json
data/regime_a_protocol.yaml        冻结协议（含预注册判据）
data/sha256s.txt
```

## 10. 请外部审批的问题

1. **B 判决的执行是否正确**？判据应用、bootstrap 方法、seed 处理是否符合预注册文本？
2. **"阶跃+平台"模式下联动判据的设计缺陷**：预注册判据测 early→late 轨迹同动；实测效应在 early 前完成。是否应 protocol revision v4.2 将联动判据改为"pretrained→SFT 阶跃的一致性"（ΔCPI 与 ΔOG 在 pretrained→late 上同向且 CI 不含 0——实测两者均为负且稳定）？此修订的科学正当性如何？
3. **诊断论文叙事的价值**：*"Vanilla post-training attenuates reveal-order incompatibility and order sensitivity; the effect completes within warmup"* —— 作为现象级发现（proposal §26 Level 1-2）的发表前景？与 Path-Dependent Denoising / Decoding in Order-Agnostic LMs 的定位差异？
4. **后续消融方向**：什么训练因素决定兼容性方向？（lr / weight decay / 数据分布 / 训练时长）——哪些最值得做、预算如何？
5. 其他方法论疑点。

---

*报告生成：Claude Code；所有 run 的 seeds/命令/config 记录于各 run_metadata.json；协议 v4.1。*
