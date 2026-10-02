# Post-Training RL Plan v0.1 — DRAFT（未冻结，审查中）

> **状态：DRAFT（不是 FROZEN / FINAL）**。任何 RL 代码、训练、评估在协议 freeze 前不得启动。
> 前置 Git 快照：`pre-rl-compatibility-v1` @ `de69bb9`（origin main）。
> 技术依据：`reports/rl_reverse_transition_analysis.md`、`reports/repo_audit_summary.md`。

---

## 1. Research Motivation

SEDD 离散扩散 LM 的 reveal-order compatibility 已被证明受 post-training 影响：Vanilla SFT 使
CPI 0.328→~0.28、OrderGap 10.24→~8.8（v4.1），且 P1 定位该衰减为 warmup 后期（~250–1020 步）
的连续过程。训练信号是"监督"。当前未知：当训练信号切换为 **on-policy reward optimization** 时，
compatibility 会继续衰减、保持、还是反弹？这决定 post-training 信号类型是否是 compatibility
的一个自由度。

## 2. Current Frozen Findings（v4.1 + P1，不可改动）

- Vanilla SFT（SEDD-small、WikiText103 span-infilling、2 seeds）使 CPI_abs 0.3281→0.283–0.289（−12~14%）、OrderGap_raw 10.24→~8.8（−14%），paired bootstrap CI 不含 0。
- Stage-4 判决 **B_DIAGNOSTIC_ONLY（永久固定）**。
- P1：衰减定位在 250–750 窗口（s1=(250,500)、s2=(500,750)，EMA 口径），0→250 平台、1020→2500 平台；OrderGap 更早启动（0→50 已显著）；signed δ 全程 ≈0（对称收缩）。
- 措辞纪律：attenuation temporally overlaps with later warmup/rising LR —— **不声称 LR 因果**。

## 3. Why RL Is Introduced Now

课程 take-home 明确要求探索离散扩散 LM 的 RL 路线。已完成：SEDD 复现 → SFT → 兼容性评估 → 独立研究（P1）。
未覆盖：RL integration。故现在进入 **full-model RL post-training**（非 frozen-model scheduler）。

## 4. Main Research Question

> Does reward-based post-training continue, preserve, or reverse the reveal-order compatibility
> attenuation induced by SFT?

θ_pretrained → θ_SFT → θ_RL（θ_RL 是真正被 reward 更新过的 SEDD 参数）。观察 CPI / OrderGap 如何随 task-reward 优化变化。

## 5. Hypotheses / Possible Outcomes（三种都是合法科学结果）

- H-A：RL 后 CPI / OrderGap 继续下降（compatibility attenuation 延续）
- H-B：RL 后基本不变（task reward 改善不改变 compatibility）
- H-C：RL 后反弹升高（reward specialization 重新引入/放大 reveal-order 依赖）
- H-D：CPI 与 OrderGap 方向分歧（局部 vs 全局度量捕获不同结构效应）

## 6. Scope

- 模型：SEDD-small（~170M）；GPU：RTX 5060 Ti 8GB；WSL2（已迁 D:）。
- 数据/任务：WikiText103 partial-reveal span-infilling（沿用 v4.1 任务结构）。
- 评估：frozen manifest `regime_a_eval_v1`（SHA `1897bd14…`）+ 既有 CPI/OrderGap/G1 三件套。

## 7. Non-goals

- 不在第一版做：frozen-model reveal scheduler、semantic reward、reward model、critic、KL 约束到 reference model、multi-GPU、SEDD-medium。
- 不把 CPI/OrderGap 放进 reward（§20）。
- 不做 PAPL/Swap 2×2（已降级，v4.1 记录）。

## 8. Relationship to Existing SFT Work

SFT 管线（training/vanilla.py、corruption.py、vanilla_256.yaml、EMA/checkpoint 格式）全部复用；RL 训练循环另写 `training/rl.py`。SFT 的 checkpoint_10200 EMA（seed 1）是 RL 唯一初始化（§12）。

## 9. Relationship to P1

P1 已 FREEZE（84/84、Gate 1/2 双 PASS、位级复现证据 2/2）。RL 的 CPI 轨迹在 SFT 后的新动力学段上延续同一度量与同一 manifest，可直接与 P1 曲线拼接（step 0 = SFT-10200）。P1 结论不因 RL 结果改写。

## 10. Deferred P2 Plan

**P2 = DEFERRED, NOT CANCELLED**。优先级：P1 收尾 → RL → take-home/interview → P2 → potential paper。
P2a（openwebtext 复现）/ P2b（任务族）/ P2c（规模）设计见 protocol v4.2 §4，执行条件与 RL 结果无关。

## 11. Existing Assets to Reuse

见 `reports/repo_audit_summary.md` §1（模型/图/噪声/采样器/EMA/checkpoint 格式/评估三件套/frozen manifest/Swap 显存配方）。

## 12. RL Initialization Checkpoint

`exp_local/regime_a/formal-vanilla-s1-191414/checkpoint_10200.pth` 的 **EMA 权重**
（加载路径与 eval_task.py 一致：`ema.load_state_dict + copy_to`，禁止直接 `model.load_state_dict(ema_dict)`）。
raw 权重仅作对照，RL 更新 raw；EMA 在 RL 期间照常维护。

## 13. SFT-to-RL Transition

- model weights ← SFT EMA-10200；optimizer ← **fresh AdamW**（不继承 SFT moments：objective 已变）。
- scaler ← fresh GradScaler；EMA ← 以 RL 初始权重重新初始化（decay 参数沿用 0.9999）。
- RL step 0 由 RL-G0 gate 验证（§33）。

## 14. Definition of Policy in SEDD

见 `reports/rl_reverse_transition_analysis.md`（全文是本章的 technical annex）。要点：
- model 输出 **log-score**（[B,L,50258]，x_t 位置 logit 恒 0）；一切概率从 exp(score) 构造。
- policy 是 **per-position categorical**：给定 x_t，各位置独立决定"保持"或"跳到哪个 token"；总 log π = Σ_positions log π_pos。
- 预注册口径：**rollout 采样器与 log π 公式同源**。默认候选 A：analytic predictor 的转移权重 → 正规化 softmax → categorical 采样 + log π；候选 B：Euler/τ-leaping（需记录负"概率"诊断量）。freeze 前用 §34 单测锁定其一。

## 15. SEDD Reverse-Transition Derivation

`reports/rl_reverse_transition_analysis.md` §1–§4：score 最优解、Absorbing rate 矩阵、reverse_rate 构造、
Euler 与 analytic 一步转移公式、staggered-score 近似、末尾 denoiser。此处不重复。

## 16. Action Definition

Action a_t = (masked positions 集合, 每位置选中的 token 或"保持")。实现为与 x_t 同形的 token 张量
+ "是否变化" mask。action 的 log π 由 §14 公式按位置求和。

## 17. Trajectory Representation

Rollout（no_grad、eval 模式）为每条轨迹落盘：
`sample_id, rng_seed, step_idx, t, sigma, x_t, x_next, changed_mask, action_tokens, old_logπ, span_mask, generated_span, gt_span, reward, 诊断量（负概率计数/权重和偏离）`。**不保留任何 autograd 图。**

## 18. Reward Definition

Primary：position-wise token reconstruction accuracy over the target span：
`R = #(x̂_i == x_i, i ∈ span) / |span| ∈ [0,1]`。与 G1 口径对齐、长度归一、无需 reward model。

## 19. Reward Limitations

position-wise 口径对语义等价输出（bought→purchased）给低分。第一版不混入 semantic score（保持 reward
纯净，避免把变化归因复杂化）；secondary 记录 exact-match / edit distance / BERTScore 供观察。semantic
reward 留作后续 ablation。

## 20. Why CPI / OrderGap Are Diagnostics, Not Rewards

R = −CPI / −OrderGap / −|δ| 一律禁止（含 hidden regularizer 形式）。目的是观察 task-reward 优化是否
自然改变 compatibility；直接 reward compatibility 会使结论失去科学价值。

## 21. Simplified Policy-Gradient Smoke Test

形式（概念参考，实际按 §14 公式）：
```
τ ~ π_θ (rollout), R(τ) → A = R − batch_mean(R)
L_PG = −A · Σ_{sampled t} log π_θ(a_t | x_t)
```
目标（全部通过才进入 RL-2/RL-3）：rollout 完整、reward 正常、logπ 可重算、对 θ 有梯度、PG 更新非零、
无 NaN、reward 出现学习信号、模型不立刻 collapse、SFT 任务性能无灾难性下降。

## 22. Group-Relative Advantage

同一 corrupted span 采样 G=4 条独立 rollouts：`A_i = (r_i − mean(r)) / (std(r) + 1e-4)`。无 critic。
同一 prompt 下高于平均 → 正 advantage，低于 → 负。

## 23. diffu-GRPO Discussion

- 思想：把扩散生成整体当作一条序列似然做 group-relative PG，使用 sequence/one-step likelihood 近似。
- 与 SEDD 匹配难点：SEDD 的 exact 生成似然是逐位置 CTMC 转移的乘积（§14），diffu-GRPO 的近似 likelihood
  与真实 sampler 分布不一致 → old/new logπ 口径错位、estimator bias 风险。
- 复杂度/成本：需要完整轨迹 likelihood 重算，8GB 下负担重。
- 用途：related method / optional baseline，**不优先**。若实现，必须按 §33 的单测验证其近似口径。

## 24. AGRPO-style Main Method

- Markov transition formulation：直接利用 SEDD 逐位置 categorical 转移（§14）—— AGRPO 的
  step-level PG 与 SEDD 结构天然对应（每 reverse step 是一个 Markov transition）。
- Step-level PG：完整 rollout no_grad；每轨迹采样 K 个 timestep（§25）→ 重算 log π → PG。
- 不照搬 LLaDA AGRPO 代码：先按 §14 的 SEDD 公式从零推导。
- 数学有效性：log π 可微可重算（§6 of reverse-transition 分析），single-state backward 显存可行。

## 25. Timestep Sampling

- Pilot 预注册：每轨迹 **uniform 采样 1 个 reverse step**（不含最终 denoiser 步，pilot 先排除其
  非标准分布）；K=1。
- 消融候选（后续 protocol 修订）：noise-weighted（∝ dσ）、多步 K=2/4、含 denoiser 步。
- estimator weighting / unbiasedness：PG 估计为采样 step 的无偏估计（每个采样 step 贡献
  A·∇logπ / P(sampled)）；importance correction 与 group advantage 的相互作用在 §26 定义。

## 26. Importance Ratio / Clipping

- ratio = exp(new_logπ − old_logπ)（同一 transition、同一公式重算）。
- 目标：`L = −A · clip(ratio, 1−ε, 1+ε) · (sampled-step indicator)`，ε=0.2（pilot 先冻结）。
- 多 epoch 重放同一 rollout 时：old_logπ 固定于 rollout 时刻（off-policy 修正只经 ratio），
  但每 epoch 后 ratio 失真加剧 → pilot 限制 ≤2 epochs，超限/ratio 均值爆炸进 abort（§44）。
- 不做 KL 约束（第一版无 reference model；用 clipping + 小 LR 控制漂移；记录参数漂移诊断量）。

## 27. Optimizer Strategy

Fresh AdamW（β=(0.9,0.999), eps=1e-8, wd=0），grad_clip=1.0，warmup 暂定 0（从 SFT 已训权重起步，
RL 阶段是否需要 warmup 在 LR pilot 中观察决定）。scaler 沿用 amp。

## 28. LR Pilot

候选 {1e-6, 3e-6, 1e-5}，优先 **3e-6**（SFT 的 3e-5 对高方差 PG 过激，先验）。选择依据：grad norm、
reward 趋势、ratio 分布、NLL/重建 acc 漂移、稳定性。**不得在 pilot 前把任何 LR 写死为正式值。**

## 29. GPU / VRAM Plan

- Rollout：eval 模式 no_grad，batch 小（如 4-8 prompts × G=4），峰值 ~2-3GB（CPI 评估同量级）。
- Training：单 transition 重算 forward+backward（gradient checkpointing 沿用），微 batch 8–16，
  accum 2–4；峰值目标 ≤6.4GB（SFT 基线）。
- 无 critic、无常驻 reference model、无 rollout 图驻留。old_logπ 存 detached 数值（非旧模型副本）。

## 30. Rollout Memory Plan

每步只落盘数值字段（§17）。轨迹缓存目录 `exp_local/regime_a/rl-*/trajectories/`（按 batch 分文件，
JSONL/pickle 均可，写入时用 .tmp+rename 防半截文件——**吸取 P1 崩溃教训**）。训练后缓存可删（协议
§43 预算）。

## 31. Checkpoint Plan

- RL pilot save_at：**0, 50, 100, 250, 500**（预注册）。
- 内容审计：SFT checkpoint 含 model+EMA+optimizer+scaler（~2.7GB）。RL pilot 建议：
  - save_at 全量保存（与 SFT 同格式，可被 eval 三件套直接读取）；
  - 中间诊断点（每 25 步）只存 `{step, model, ema}`（省空间，~1.4GB）供轨迹观察；
  - optimizer 不单独持久化到中间点（断点续训从最近全量点恢复）。
- raw + EMA 双权重照旧（评估口径统一）。

## 32. RL Pilot Horizon

~500 optimizer steps；检查点 0/50/100/250/500。若 reward 无学习信号 / NaN / collapse / NLL 灾难性
退化 / ratio 爆炸 → debug，不扩大训练。pilot 通过后才设计 formal horizon。

## 33. Preflight Gates

- **RL-G0（step-0 gate）**：RL 初始化的模型（EMA-10200 加载后）在 frozen manifest 上复算 CPI/OrderGap/G1，
  与 v4.1 formal s1-10200 结果逐字段一致（容忍浮点噪声，预注册阈值）；state_dict 与 SFT checkpoint
  EMA shadow 逐 tensor 一致。不一致 → RL 停止。
- **RL-G1（transition gate）**：§34 单测 1–13 全过 + rollout/recompute logπ 一致性（单测 5）。
- **RL-G2（pilot 放行）**：G0+G1 通过 + LR probe 完成 + 磁盘预算满足。

## 34. Correctness Unit Tests（预注册清单）

1. transition 概率正规化；2. sampler 采样分布与 logπ 公式一致（大样本频率对拍）；3. 采样动作 logπ 有限；
4. old logπ detached；5. 更新前重算 logπ == rollout logπ（逐位）；6. PG 梯度有限；7. 随机轨迹 reward
无关性 sanity；8. group 内 advantage 均值 ≈0；9. 零 advantage → 更新 ≈0；10/11. toy 上正/负 advantage
分别增/减选中转移概率；12. reward 无梯度路径（reward 张量 no_grad 验证）；13. 固定 seed 复现；
14. RL step0 == SFT init。另：toy finite-state Markov chain 上验证 PG 实现。

## 35. Task Evaluation

每个 save_at 点：mean rollout reward、token reconstruction acc、exact-match rate、masked NLL、
greedy reconstruction acc（eval_task.py 口径）、可选 semantic 指标。

## 36. Compatibility Evaluation

沿用 frozen manifest + eval_cpi.py / eval_order_gap.py（EMA/raw 双权重）：CPI_abs、CPI_RMS、signed δ、
OrderGap_raw、per-token；若成本允许加 buckets。与 Pretrained/SFT 阶段直接可比（同一 manifest）。

## 37. Seed Strategy

Pilot/take-home：**SFT seed1 → RL seed1**（单链）。signal 存在后再加 SFT seed2 → RL seed2。
单个 RL seed 不作 universal conclusion。

## 38. Statistical Analysis

延续 v4.1 纪律：同一 manifest 上的 paired sample bootstrap（SFT vs RL 同样本差），只代表 sample
uncertainty，不代表 training-seed significance。两 seed（若做）分别报告 + 均值，不夸大两点均值。

## 39. Failure / Abort Criteria（预注册）

Abort → debug（不当科学结果）：transition 概率无法与 sampler 一致定义；rollout 与重算 logπ 意外
不一致；梯度 NaN/Inf；reward collapse / variance≈0 / group 内全同；ratio 爆炸（max>20 或 mean>5）；
NLL 灾难性退化（Δ>+1 nat）；token acc 崩塌；生成无效状态（含 MASK/越界）；RL-G0 不通过；8GB 下
OOM 无法用合理显存策略解决。

## 40. Result Interpretation Matrix（预注册）

| Scenario | 信号 | 解读 |
|---|---|---|
| A | Reward↑ CPI↓ OG↓ | RL 延续 compatibility attenuation |
| B | Reward↑ CPI≈ OG≈ | reward 改善不改变 compatibility |
| C | Reward↑ CPI↑ OG↑ | reward specialization 重新引入/放大 reveal-order 依赖 |
| D | CPI 与 OG 方向分歧 | 局部 vs 全局度量捕获不同结构效应，需进一步分析，不强行合并 |
| E | Reward 不提升 | 先判定 RL 实现/reward/exploration/optimization，不回答科学问题 |

## 41. Logging / Reproducibility

run_metadata.json 沿用 vanilla.py 字段 + 新增：rl_protocol_version、predictor、steps、G、K、LR、clip ε、
rollout RNG seeds、trajectory 缓存位置、诊断量汇总。所有随机源显式 seeded；每步日志含 reward/advantage/
ratio/近似诊断。

## 42. Demo Plan

最终展示：同一 corrupted span 上 SFT vs RL 输出对比 + token acc / reward / NLL + 三阶段（pretrained→SFT→RL）
CPI/OrderGap/task 指标图。Gradio 后置，先保证科学正确。

## 43. Disk Budget

- RL pilot：5 全量 ckpt × 2.7GB ≈ 13.5GB + 25 诊断点 × ~1.4GB ≈ 35GB？→ **过多**。收紧：诊断点每
  50 步（10 个 × 1.4GB ≈ 14GB）+ 5 全量 ≈ 13.5GB → 合计 ~28GB；或诊断点不落盘只打日志（推荐 pilot：
  5 全量 ≈ 13.5GB + trajectory 缓存 ~2-5GB）。**当前 D: 仅 ~19GB free，必须先行清理/压缩/迁移
  （Optimize-VHD + 清 lrprobe/pilot-132632 ~7.8GB，或 vhdx 迁 E:/F:）—— P1 后遗留的已知 blocker。**
- GitHub 只存源码/config/protocol/结果 JSON/报告/hash，不存 checkpoint（`.gitignore` 已挡）。

## 44. Git / Versioning Plan

- 一切 RL 代码在 `rl/`、`training/rl.py`、`tests/test_rl_*` 新文件内进行；不动 v4.1/v4.2/P1 相关文件。
- 协议版本号：本 DRAFT → 审查 → `post_training_rl_protocol_v1.0_FROZEN.md`（freeze 后任何改动递增
  版本并重跑受影响部分，纪律同 v4.2 §7）。

## 45. Implementation Roadmap

1. 存储：清 junk / compact / 迁盘（先决）。
2. `rl/transition.py` + 单测 1–13（CPU 可跑）。
3. RL-G0（EMA-10200 加载对拍）。
4. `rl/rollout.py` + `rl/reward.py`（小批量 smoke）。
5. `training/rl.py`：simplified PG（RL-1）→ group-relative（RL-2）→ AGRPO-style step-level（RL-3）。
6. LR probe（1e-6/3e-6/1e-5，各 ~100-200 步）。
7. RL pilot（save_at 0/50/100/250/500）→ 全评估（task + compatibility）。
8. 判读（§40 矩阵）→ 决定 formal horizon / seed2 / P2。

## 46. Open Questions（freeze 前必须定案）

1. rollout predictor 定案：analytic（候选 A）vs euler/128（候选 B）——单测 2 的大样本对拍结果决定。
2. steps 数：1024（run_sample 同款）vs 128（config 默认）——按 rollout 成本与质量 pilot 前测。
3. 是否含 denoiser 步进 PG 采样（pilot 排除，formal 再议）。
4. K=1 是否足够（variance/学习速度的 probe 数据决定 K）。
5. 多 epoch 重放上限（pilot 先 2）。
6. warmup 是否需要（LR probe 决定）。
7. 诊断点 checkpoint 策略（§43）。
8. advantage 归一化 epsilon 与 std 下限（组内 std≈0 时 advantage 置 0，预注册常数）。
9. 采样温度：rollout 是否用 τ<1 或 argmax 混合（第一版 τ=1 纯采样，记录 diversity 诊断）。

## 47. Conditions Required Before Protocol Freeze

- §46 全部定案；§34 单测 1–13 全过；RL-G0 通过；LR probe 完成；存储先决满足；
- 本 DRAFT 经用户审查无异议 → 提升为 `post_training_rl_protocol_v1.0_FROZEN.md`。
- freeze 前禁止任何 RL training 启动。
