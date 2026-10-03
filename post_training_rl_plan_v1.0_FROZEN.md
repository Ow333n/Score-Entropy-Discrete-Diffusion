# Post-Training RL Plan v1.0 — FROZEN

> **状态：FROZEN**。任何改动需递增版本号（v1.1+）并重跑受影响部分（纪律同 protocol v4.2 §7）。
> 本文件由 `post_training_rl_plan_v0.2_DRAFT.md` 经 LR review + freeze review 修订而来。
> 前置 Git 快照：`pre-rl-compatibility-v1` @ `de69bb9`；freeze 时 HEAD：见 §0.5。
> 技术依据：`reports/rl_reverse_transition_analysis.md`、`reports/repo_audit_summary.md`、
> `reports/lr_probe_summary.json`。
> 协议文件 SHA-256：`post_training_rl_plan_v1.0_FROZEN.md.sha256`（侧车文件，与 manifest 同约定）。

---

## 0. Freeze Record（v1.0 新增）

### 0.1 LR 裁定（LR review 结论，逐字）

> Among the three stable learning rates, 3e-6 showed the clearest short-horizon deterministic
> reconstruction improvement with moderate parameter drift and no NLL degradation, and was
> selected for the formal pilot.

- **不声称 3e-6 statistically / significantly 优于其他 LR**。三者全部 150/150 PASS、无 OOM /
  NaN/Inf / dxg error / collapse、grad norm 稳定、zvg 相近、无 NLL degradation。
- **LR calibration 到此结束**：不继续搜索 5e-6 / 7e-6 / 其他中间 LR。
- **formal LR = 3e-6（constant, 无 warmup）**；禁止用 CPI/OrderGap 重新选择 LR。

### 0.2 三条 150-step probe 关键证据（见 `reports/lr_probe_summary.json`）

| metric (step150, step0 锚点) | step0 | 1e-6 | 3e-6 | 1e-5 |
|---|---|---|---|---|
| drift ‖Δθ‖/‖θ0‖ (raw) | 0 | 5.34e-5 | 1.54e-4 | 5.28e-4 |
| NLL raw（冻结 corruption realization） | 7.0490 | 7.0424 | 7.0125 | 7.0228 |
| eval8 greedy raw | 0.2375 | 0.2625 | **0.2803** | 0.2622 |
| eval8 sampled raw | 0.1979 | 0.1933 | 0.1837 | 0.1709 |
| zvg cumulative | — | 38/600 | 43/600 | 45/600 |
| soft_neg_elements / soft_steps | — | 18/18 | 20/19 | 15/15 |

离线补算与各 run logged 值 **7/7 逐位吻合**。drift 随 LR 近似线性标度（无异常放大）。

### 0.3 v0.2 §46 / §47 冻结前条件 —— 全部满足

- §46.1 predictor = **analytic** ✓（§14 定案；Euler 仅 debug baseline）
- §46.2 steps = **128** ✓（128vs1024 gate 已裁决）
- §46.3 denoiser 步不进 PG 采样 ✓（pilot 维持排除）
- §46.4 **K=1** ✓（probe 数据：3 LR 全部稳定、无 variance 爆炸）
- §46.5 多 epoch ≤2 ✓（RL-3 条款；pilot/RL-1 无 replay）
- §46.6 warmup = **无** ✓（3 LR constant 全程稳定）
- §46.7 |M0| 下限 = 无人工门槛 ✓（仅 |M0|=0 重采样）
- §46.8 advantage 处理已定 ✓（mean-centered，不除 std）
- §46.9 rollout 采样 τ=1 纯 gumbel ✓；greedy 仅作 eval 口径（§35），不进 rollout
- §46.10 OrderGap 表述已按 frozen 数据修正 ✓（§2）
- §34 单测：24 项全过 ✓；RL-G0 通过 ✓；LR probe 完成 ✓；存储前置 ≥30GB：D: 现余 35GB ✓

### 0.4 已完成的预冻结里程碑（证据归档）

- RL-G0 / 128vs1024 gate / analytic rollout 验证（零硬负权重）—— MEMORY.md + 单测记录
- RL-1 smoke（150 步, 3e-6）：PASS，零 OOM/NaN/dxg error，无 memory leak（§21 记录）
- LR probe 1e-6 / 3e-6 / 1e-5 各 150 步：全部 PASS（§28 记录）
- 三个 audit（step0 基线 / sigma diagnostics / drift）—— scripts/rl1_post_smoke_audit.py
- Windows Restart 后 dxgkrnl/WDDM allocation 异常已根治（默认 allocator model-load PASS 因果验证）

### 0.5 Freeze 元数据

- freeze 日期：2026-10-03；freeze 前 HEAD：3f88959；本文件 SHA-256：见侧车文件。
- 本协议 freeze 后，formal pilot 启动前还需：freeze review PASS（用户裁定）。

---

## 1. Research Motivation

SEDD 离散扩散 LM 的 reveal-order compatibility 已被证明受 post-training 影响：Vanilla SFT 使
CPI 0.328→~0.28、OrderGap 10.24→~8.8（v4.1），且 P1 定位 CPI 衰减为 warmup 后期（~250–1020 步）
的连续过程。训练信号是"监督"。当前未知：当训练信号切换为 **on-policy reward optimization** 时，
compatibility 会继续衰减、保持、还是反弹？这决定 post-training 信号类型是否是 compatibility
的一个自由度。

## 2. Current Frozen Findings（v4.1 + P1 + RL 预冻结证据，不可改动）

- Vanilla SFT（SEDD-small、WikiText103 span-infilling、2 seeds）使 CPI_abs 0.3281→0.283–0.289（−12~14%）、OrderGap_raw 10.24→~8.8（−14%），paired bootstrap CI 不含 0。
- Stage-4 判决 **B_DIAGNOSTIC_ONLY（永久固定）**。
- P1（CPI）：衰减定位在 250–750 窗口（s1=(250,500)、s2=(500,750)，EMA 口径），0→250 平台、1020→2500 平台；signed δ 全程 ≈0（对称收缩）。
- P1（OrderGap）：OrderGap 在 step 50 已出现小幅可检测下降（10.244→10.216，Δ≈−0.03 nats），CPI 此时仍平台 → 两度量早期动力学不同步；随后 100→1020 连续大幅下降。frozen 数据口径不变。
- **RL-1 smoke（150 步, 3e-6）**：150/150 PASS、零 OOM/NaN/dxg error；train reward 噪声内平（0.07–0.36，无趋势）；fixed eval8 reward 0.1883→0.1837→0.1837（vs step0 0.1979）；NLL 7.0186→7.0130→7.0171（vs step0 7.0490，无退化）；zvg 43/600；drift raw 1.54e-4 / ema 1.44e-4；显存平台稳定（5.84GB max_alloc，无单调增长）。
- **sigma 诊断（audit 定案）**：J support（σ≥0.05）= mean **91.9%**（per-prompt 112–121/128 步）；被截断 objective support = mean **8.1%**（7–16 步/128，全在 σ→0 尾段）；σ≥0.05 区三条 run 全程零硬失败（nan_rollouts=0）；软区负权重事件率 ≈0.15%（chunk-step 级）/ ≈4e-8（element 级）。
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
- 评估：frozen manifest `regime_a_eval_v1`（SHA-256 侧车校验）+ 既有 CPI/OrderGap/G1 三件套。
- **RL method 名称（v1.0 定案）**：**safe-region / truncated transition policy gradient**。
  不得称为 "full reverse-trajectory unbiased PG"（理由见 §25：J 采样支撑被截断 8.1%）。

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
**不复制该 checkpoint**：RL run 目录记录其绝对路径 + SHA-256（run_metadata），step-0 评估直接引用。

## 13. SFT-to-RL Transition

- model weights ← SFT EMA-10200；optimizer ← **fresh AdamW**（不继承 SFT moments：objective 已变）。
- scaler ← fresh GradScaler。
- **EMA policy（明确）**：
  - shadow 初始化 = RL 初始权重（即 SFT EMA-10200 权重本身）；
  - `num_updates` **reset 到 0**（RL 阶段独立计数；不复用 SFT 的 10201）；
  - decay 参数 0.9999，但 pytorch_ema 存在 decay warmup：`effective_decay = min(0.9999, (1+n)/(10+n))`
    —— RL 早期 EMA 处于 warmup 区间，报告中必须引用 effective decay 而非 0.9999；
  - **评估口径（v1.0 重申）：RL early dynamics 以 raw 为 PRIMARY、EMA 为 SECONDARY**
    （理由：EMA 在 decay warmup 中，早期不能代表慢移平均；raw 直接反映 PG 更新轨迹）。
- RL step 0 由 RL-G0 gate 验证（§33）。

## 14. Definition of Policy in SEDD

见 `reports/rl_reverse_transition_analysis.md`（技术 annex）。policy 要点（定稿口径）：

- model 输出 **log-score**（[B,L,50258]，x_t 位置 logit 恒 0）；一切概率从 exp(score) 构造。
- policy 是 **per-position categorical**。给定 x_t，每个位置在 D=50258 个状态上有一个转移权重向量 w_θ(x_t, σ)：
  - **primary RL-policy = analytic predictor（定案）**：`w = staggered_score(score, dσ) × transp_transition(x_t, dσ)`
    （元素乘，见 reverse-transition 分析 §4.2；staggered_score 内部含 exp(dσ) 缩放与 MASK 列补偿）。
  - **Euler/τ-leaping 降级为 inference / debug baseline**：除非 Euler 单独通过 §14.1 的权重有效性 /
    非负性 / normalization / sampler-frequency matching 全套检查，否则不作为 primary RL policy。
- **policy 分布与 rollout sampler 完全同源**（同一 w 向量）：
  ```
  π_θ(v | s) = w_v / Σ_u w_u            （正规化 categorical）
  logπ_θ(v | s) = log(w_v) − logsumexp(log w)
  ```
  rollout 采样用 `sample_categorical(w)`（gumbel 技巧按 positive weights proportional sampling，
  **不需要显式归一化**）；logπ 按上式用同一 w 计算。禁止"采样用一套分布、logπ 用另一套"。
- **动作语义**：对 MASK 位置，动作 = 保持 MASK 或跳到某干净 token；**MASK→MASK (stay) 若其权重依赖 θ
  则属于 stochastic action，必须计入 logπ 与 PG**。非 MASK 位置是确定性吸收，**logπ=0，从 PG 中排除**。
- 每位置 logπ 求和得 transition-level logπ；joint trajectory logπ = Σ_steps Σ_positions。

### 14.1 Analytic-weight validity gate（每次 rollout step 强制执行）

- 对每个位置的权重向量 w 检查：**全部 finite、全部 ≥ 0、总质量 Σw > 0**。
- 违反且该步 σ ≥ SIGMA_GATE_MIN → **abort**（不静默 clamp）。违反且 σ < SIGMA_GATE_MIN → 软门
  放行 + 计入统计（§25：safe-region 定义与数值有效性边界的实证依据）。
- numerical negative 处理：|w_min|/Σw ≤ 预注册 tolerance（**1e-6**，仅浮点噪声）允许放行但计入统计；
  超过 tolerance → abort 并 debug。
- **Euler 口径资格条款**：若 Euler 的 effective categorical weights（含 keep 项）出现真实负数，
  **取消 Euler 作为 RL-policy candidate 的资格**，不是仅记录诊断。

## 15. SEDD Reverse-Transition Derivation

`reports/rl_reverse_transition_analysis.md` §1–§4：score 最优解、Absorbing rate 矩阵、reverse_rate 构造、
Euler 与 analytic 一步转移公式、staggered-score 近似、末尾 denoiser。此处不重复。

## 16. Action Definition

Action a_t = (MASK 位置集合, 每位置选中的 token 或"保持")。实现为与 x_t 同形的 token 张量
+ "是否变化" mask。log π 按 §14 公式逐位置求和（含 stay 动作）。

## 17. Trajectory Representation（K=1 精简版，见 §43 磁盘约束）

Rollout（no_grad、eval 模式）**预先采样 timestep J**，只缓存训练所需字段：
`sample_id, rng_seed, J, t_J, sigma_J, x_J, x_{J+1}, changed_mask, action_tokens, old_logπ, M0, generated_span, gt_span, reward, 权重合法性诊断量`。
完整 trajectory 只保留小型 debug subset（每 batch 1 条，上限 ~20 步字段）。**不保留任何 autograd 图。**

## 18. Reward Definition（M0 partial-reveal 口径）

- **M0 = rollout 初始时 span 内实际被 mask 的位置集合**（`initial_state == MASK` 且 `span_mask`）。
- Primary reward：
  ```
  R = |{ i ∈ M0 : x̂_i == x_i }| / |M0|        （在初始 mask 位置上评估 token 重建正确率）
  ```
  不在 M0 上的 span 位置**不参与 reward**。
- **Group 同源约束**：同一 prompt 的 G 条 rollouts 必须共享 clean sample / span / σ / initial mask
  （同一次 `corrupt_span_batch` 产物），只改变 rollout RNG。
- 退化情形定案：**只有 |M0| = 0 时重采样**（record 计数）；不设 |M0| ≥ 4/8 之类的人为门槛。
  zero-variance group（r 全同 → A 全 0）自然不产生 policy update，属合理行为，不强制重采样。
  logging 要求：|M0| 分布、zero-variance group fraction、reward variance vs |M0|、mean reward vs |M0|。
- R ∈ [0,1]，长度归一，无需 reward model。secondary 指标（exact-match / edit distance / semantic）仅记录。

## 19. Reward Limitations

position-wise 口径对语义等价输出给低分。第一版不混入 semantic score（保持 reward 纯净）；
semantic reward 留作后续 ablation。

## 20. Why CPI / OrderGap Are Diagnostics, Not Rewards

R = −CPI / −OrderGap / −|δ| 一律禁止（含 hidden regularizer 形式）。目的是观察 task-reward 优化是否
自然改变 compatibility；直接 reward compatibility 会使结论失去科学价值。

## 21. RL-1 Smoke Test —— 已完成（PASS，证据冻结）

- exact joint transition logprob：`logπ(τ) = Σ_{steps∈τ} Σ_{i∈masked} logπ_θ(a_{t,i} | s_t)`（含 stay 动作）。
- 纯 on-policy：不 replay、不 clipping、不 ratio。L_PG = −A · logπ(τ)（A 为 §22 group mean-centered advantage）。
- **结果（rl1-smoke-022319, 150 步, 3e-6）**：目标全部通过 —— rollout 完整、reward 正常、logπ 可重算（单测 5）、
  对 θ 有梯度、PG 更新非零、无 NaN、零 OOM / dxg error、无 memory leak、SFT 任务性能无灾难性下降
  （NLL 7.0490→7.0171）。细节见 §2 frozen findings。

## 22. Group-Relative Advantage

- 同一 prompt（共享初始腐蚀，§18）采样 G=4 条独立 rollouts。
- **Primary advantage = mean-centered**：`A_i = r_i − mean_{g∈G}(r_g)`。**不除 std**（std 仅 logging）。
- 无 critic。组内 std≈0 时 A 全 0 → 该组贡献 ≈0（PG 无害），记录计数（zvg）。

## 23. diffu-GRPO Discussion

- 用途：related method / optional baseline，**不优先**。若实现，必须按 §34 的单测验证其近似口径。

## 24. AGRPO-style Main Method（RL-3）

- Markov transition formulation：SEDD 逐位置 categorical 转移（§14）与 step-level PG 天然对应。
- **per-position ratio / clipping**（不是 joint ratio）；ρ_i = π_new(a_i|s)/π_old(a_i|s)，逐 position clip（§26）。
- Step-level PG：完整 rollout no_grad；每轨迹按 §25 采样 K 个 timestep → 重算 logπ → PG。
- 不照搬 LLaDA AGRPO 代码：先按 §14 的 SEDD 公式从零推导。

## 25. Timestep Sampling —— safe-region / truncated PG（v1.0 重定义）

- **Primary MC target：safe-region uniform timestep expectation**
  ```
  g(θ) = E_{J ~ Uniform({i : σ_i ≥ SIGMA_GATE_MIN})} [ ∇_θ logπ_θ(a_J | s_J) ]
  ```
  q(J) = uniform over the safe region → 采样 J 后直接贡献 ∇logπ_J，**无需 1/q 乘子**。
- **方法命名（v1.0 定案）**：本 RL method 为 **safe-region / truncated transition policy gradient**。
  - J support = mean **91.9%**（per-prompt 112–121/128 步）；被截断 objective support = mean **8.1%**
    （7–16 步/128，全部在 σ→0 尾段，取决于 prompt σ0）。
  - **不得描述为 full reverse-trajectory unbiased PG**。unbiasedness 声明限定于：
    unclipped、on-policy 的 **safe-region truncated** timestep-MC estimator（对 g(θ) 的定义域）。
- **SIGMA_GATE_MIN = 0.05 的语义（v1.0 定案）**：
  - **完整 rollout 仍然执行全部 128 步**（含 σ<0.05 尾段 + 末尾 denoiser 收尾）；0.05 只排除
    K=1 PG timestep 采样，不截断生成过程。
  - 0.05 是 **SEDD-small + analytic predictor + 128-step schedule 下经验验证的 numerical-validity
    boundary**（证据：σ≥0.05 区三条 run 全程零硬失败；σ<0.05 软区负权重事件率 ≈0.15% chunk-step 级、
    ≈4e-8 element 级），**不是理论通用常数**。换模型/采样器/schedule 必须重新验证。
- **K=1 正式定案**：每条 trajectory 每次 update 只随机抽 1 个 safe-region timestep。
  仅当出现：gradient variance 明显过高 / reward 完全无法学习 / K=1 与 larger-K toy estimator
  偏差异常 / correctness test 表明 K=1 不稳定，才考虑增加 K。
- Pilot：K=1、safe-region uniform、不含最终 denoiser 步。消融候选：noise-weighted q、K=2/4、
  含 denoiser 步（需协议修订 v1.1+）。

## 26. Importance Ratio / Clipping（PPO min 形式，RL-3 用）

- ratio（per-position）：ρ_i = exp(new_logπ_i − old_logπ_i)。
- **Objective（标准 PPO clamp）**：`L_i = −min( ρ_i · A ,  clip(ρ_i, 1−ε, 1+ε) · A )`；ε = 0.2（pilot 冻结）。
- **RL-1（smoke/pilot）与 RL-3 的明确区分**：RL-1 = joint logπ、纯 on-policy、无 ratio、无 clip；
  RL-3 = per-position ρ_i + clip；**禁止 joint ratio**。
- 多 epoch 重放同一 rollout：pilot 限制 ≤2 epochs，超限/ratio 均值爆炸进 abort（§39）。
  不做 KL 约束（第一版无 reference model；用 clipping + 小 LR 控制漂移；记录参数漂移诊断量）。

## 27. Optimizer Strategy

Fresh AdamW（β=(0.9,0.999), eps=1e-8, wd=0），grad_clip=1.0，**constant LR = 3e-6 from step 1
（无 warmup，v0.2 审查定案 + LR review 裁定，§0.1）**。scaler 沿用 amp。

## 28. LR Pilot —— 已完成，formal LR 已定（3e-6）

- 候选 {1e-6, 3e-6, 1e-5} 三条 150-step probe 全部完成且通过稳定性检查（§0.1/§0.2）。
- 选择依据：grad norm、reward 趋势、NLL/重建 acc 漂移、drift、稳定性 —— **禁止按 CPI/OrderGap 美观度选 LR**。
- 裁定表述与限制见 §0.1。**不再进行任何 LR 搜索。**

## 29. GPU / VRAM Plan

- Rollout：eval 模式 no_grad，batch 小（4-8 prompts × G=4），峰值 ~2-3GB。
- Training：单 transition 重算 forward+backward，微 batch 8–16，accum 2–4；峰值目标 ≤6.4GB。
- 无 critic、无常驻 reference model、无 rollout 图驻留。old_logπ 存 detached 数值（非旧模型副本）。
- **Allocator（v1.0 定案，engineering 配置）**：
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  —— 分类为 **memory / allocator engineering configuration**，**不是 algorithmic change**。
  证据：Windows Restart 后默认 allocator model-load PASS（恢复与 allocator 无关的因果验证）；
  expandable_segments 独立 PASS 且无 warning；reserved 略低于默认 allocator。必须记入 run_metadata。
- 既有内存工程优化保留：float32 one_hot、chunk=2、phase 间 empty_cache。

## 30. Rollout Memory Plan

每步只落盘 §17 的数值字段。轨迹缓存目录 `exp_local/regime_a/rl-*/trajectories/`（按 batch 分文件，
写入用 .tmp+rename 防半截文件）。**每 batch 训练完成后释放缓存**（§43）。

## 31. Checkpoint Plan（最小化磁盘）

- **step 0**：不复制 SFT checkpoint；run_metadata 记录源路径 + SHA-256 指针，评估直接引用源文件。
- **step 50 / 100 / 250**：**eval snapshots**（只含 `{model, ema, step}` ≈ 1.4GB/个，供 raw+EMA 评估）。
- **step 500**：full checkpoint（`{model, ema, optimizer, scaler, step}` ≈ 2.7GB）。
- **checkpoint_latest**：单个 **atomic rolling** 全量检查点（写 .tmp → rename；每 25 步覆盖一次）。
- raw + EMA 双权重评估口径照旧（§35：raw PRIMARY / EMA SECONDARY）。

## 32. RL Pilot Horizon（formal）

**500 optimizer steps**；评估点 **0 / 50 / 100 / 250 / 500**。若 reward 无学习信号 / NaN / collapse /
NLL 灾难性退化 / ratio 爆炸 → debug，不扩大训练。pilot 通过后才设计 formal horizon。

## 33. Preflight Gates

- **RL-G0（step-0 gate）**：RL 初始化的模型（EMA-10200 加载后）在 frozen manifest 上复算 CPI/OrderGap/G1，
  与 v4.1 formal s1-10200 结果逐字段一致（容忍浮点噪声，预注册阈值）；state_dict 与 SFT checkpoint
  EMA shadow 逐 tensor 一致。不一致 → RL 停止。
- **RL-G1（transition gate）**：§34 单测全过 + rollout/recompute logπ 一致性（单测 5）。
- **RL-G2（pilot 放行）**：G0+G1 通过 + LR probe 完成 + freeze review PASS + 磁盘前置满足（§43）。

## 34. Correctness Unit Tests（24 项，全部通过，清单冻结）

1–22 项见 v0.2 §34 清单（transition 正规化 / sampler frequency 对拍 / logπ 有限 / detached /
重算逐位一致 / PG 梯度有限 / 随机轨迹 reward 无关 / 组内 advantage 均值≈0 / 零 advantage 更新≈0 /
toy 正负 advantage 方向 / reward 无梯度路径 / 固定 seed 复现 / step0==SFT init / weight positivity /
non-mask logπ=0 / stay-action included / G 共享初始腐蚀 / M0-only reward / PPO negative-advantage
clipping / K=1 MC estimator toy 对拍 / toy Markov PG）；23/24：RL rollout 口径补充测试
（validity gate 防御语义、sampler factorization、K=1 解析对拍、PPO clip 手算、M0 reward、G 共享腐蚀）。

## 35. Task Evaluation（v1.0 修订：formal fixed task eval）

- **评估样本（v1.0 定案）**：**64-sample subset，独立于 training prompt pool**。
  从 frozen eval manifest `regime_a_eval_v1.jsonl`（500 条，SHA 校验）取 **record 索引 64–127**
  （training pool = 索引 0–63）。同一 64 样本在 **step 0 / 50 / 100 / 250 / 500** 全部评估点
  **完全复用相同 samples / corruption / decoding protocol**。
  成本估计：每评估点 ≈ 4–5 min（64 samples × {sampled, greedy} × {raw, EMA} + 64-chunk NLL），
  5 个评估点合计 ≈ 20–25 min，计算预算允许（64 samples 不缩减）。
- **evaluation hierarchy（v1.0 定案）**：**RAW weights = PRIMARY，EMA = SECONDARY**。
- **sampled reconstruction** 保留，但**不能单独作为短 horizon task-learning 判据**
  （实证：小 drift 下离散 trajectory 对参数变化分辨率有限——gumbel-stable，多数 per-prompt
  sampled 轨迹在 5e-5~5e-4 drift 下与 step0 完全相同）。
- **greedy reconstruction 作为 deterministic supplementary task metric**：argmax 变体 rollout
  （与 sampled rollout 逐行同源、同 seed 同 J，仅 argmax vs gumbel），无采样噪声，提供
  sampled 口径之外的补充分辨率。greedy 只进 eval，不进 rollout 采样（§0.3）。
- **NLL evaluation（v1.0 定案）**：所有 checkpoint / raw / EMA 评估**必须使用同一个冻结
  corruption realization**（corruption_seed+1000 流，smoke 同流；raw/EMA 各用独立 generator 对象
  配对该流 → 两者看到完全相同的 corruption，NLL 差纯粹来自权重）。
  实证依据：独立 corruption stream（seed+2000）曾出现 σ≈0 极端低-sigma realization，单个
  high-variance / pathological chunk 足以显著改变有限样本 NLL mean（64 块均值 7.05→22.84）。
  该 realization **定性为 high-variance / pathological，不是"错误数据"**（证据归档
  /tmp/lrprobe-1e6-aborted-artifact.log）。跨 run NLL 比较禁止换 corruption 流。
- 每评估点指标：mean rollout reward（sampled + greedy，raw/EMA）、token reconstruction acc（M0 口径）、
  exact-match rate、masked NLL（raw/EMA，冻结 corruption realization）。

## 36. Compatibility Evaluation

沿用 frozen manifest + eval_cpi.py / eval_order_gap.py（raw primary / EMA secondary）：CPI_abs、CPI_RMS、
signed δ、OrderGap_raw、per-token；若成本允许加 buckets。与 Pretrained/SFT 阶段直接可比（同一 manifest）。
**CPI/OrderGap 只作 compatibility 诊断，不作为 RL 训练判据或 LR 选择依据（§20/§28）。**

## 37. Seed Strategy

Pilot/take-home：**SFT seed1 → RL seed1**（单链）。signal 存在后再加 SFT seed2 → RL seed2。
单个 RL seed 不作 universal conclusion。

## 38. Statistical Analysis

延续 v4.1 纪律：同一 manifest 上的 paired sample bootstrap（SFT vs RL 同样本差），只代表 sample
uncertainty，不代表 training-seed significance。两 seed（若做）分别报告 + 均值，不夸大两点均值。

## 39. Failure / Abort Criteria（预注册）

Abort → debug（不当科学结果）：transition 权重不通过 §14.1 validity gate（σ≥0.05 区真实负权重/非 finite/
零总质量）；rollout 与重算 logπ 意外不一致；梯度 NaN/Inf；reward collapse / variance≈0 / group 内全同；
ratio 爆炸（max>20 或 mean>5，RL-3）；NLL 灾难性退化（Δ>+1 nat）；token acc 崩塌；生成无效状态；
RL-G0 不通过；OOM 无法用合理显存策略解决；Euler 候选触发 §14.1 资格取消后未切换口径。
**OOM 规则：第一次 OOM 立即停止，不自动重启**，保存 optimizer step / 阶段 / requested / allocated /
reserved / max / mem_get_info / nvidia-smi / dmesg / |M0| / rollout index / J / σ 后回报。

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
rollout RNG seeds、trajectory 缓存位置、§14.1 负权重统计、M0 分布统计、**allocator_env
（expandable_segments:True，§29）**、**git_head**、**冻结 corruption realization seed**。
所有随机源显式 seeded；每 10 步日志含 reward / advantage mean/std/mean|A| / zero-variance groups /
grad norm / torch.cuda 全内存指标（allocated/reserved/peak/free）。

## 42. Demo Plan

最终展示：同一 corrupted span 上 SFT vs RL 输出对比 + token acc / reward / NLL + 三阶段（pretrained→SFT→RL）
CPI/OrderGap/task 指标图。Gradio 后置，先保证科学正确。

## 43. Disk / Trajectory / Checkpoint Budget

- **轨迹（禁止全量落盘）**：K=1 预采样 J，只存 §17 字段；完整 trajectory 仅 debug subset（每 batch ≤1 条、
  ≤20 步）。每 batch 训练后释放缓存 → 稳态 <1GB。
- **checkpoint（§31）**：step0 = 源指针（0 磁盘）；eval snapshots 50/100/250 ≈ 3×1.4 = 4.2GB；
  step500 full ≈ 2.7GB；rolling checkpoint_latest ≈ 2.7GB（原子覆盖）。
  **pilot 新增合计 ≈ 9.6GB < 10GB**。
- **前置条件**：pilot 开始前 free disk ≥ 30GB（freeze 时 D: = 35GB ✓，满足；pilot 启动前复核）。
  所有写入走 .tmp+rename（防半截文件）。
- GitHub 只存源码/config/protocol/结果 JSON/报告/hash，不存 checkpoint（`.gitignore` 已挡）。

## 44. Git / Versioning Plan

- 一切 RL 代码在 `rl/`、`training/rl.py`、`tests/test_rl_*` 新文件内进行；不动 v4.1/v4.2/P1 相关文件。
- 本文件为 v1.0 FROZEN；任何改动递增版本（v1.1+）并重跑受影响部分。

## 45. Implementation Roadmap（状态更新）

- **A.** 磁盘问题：free disk ≥ 30GB —— **已满足**（D: 35GB）。
- **B/C.** rl/transition.py + 24 项单测 —— **已完成**（全过）。
- **D.** 128vs1024 gate —— **已完成**（128 定案）。
- **E.** RL-G0 —— **已完成**（PASS）。
- **F.** RL-1 smoke —— **已完成**（150 步 PASS，§21）。
- **G.** LR probe（1e-6/3e-6/1e-5 × 150 步）—— **已完成**（§28，formal LR=3e-6）。
- **H.** review 回报 —— 已完成（LR review：SMOKE=PASS + LR=3e-6 裁定）。
- **I.** freeze v1.0 —— **本文件**。
- **J.** 500-step formal RL pilot —— **待 freeze review PASS 后启动**（当前禁止启动）。

## 46. Open Questions —— 全部定案（freeze 后新增问题需 v1.1+）

1. predictor = analytic ✓
2. steps = 128 ✓
3. denoiser 不进 PG 采样 ✓
4. K=1 ✓
5. 多 epoch ≤2（RL-3 条款）✓
6. warmup = 无 ✓
7. |M0| 无人工门槛 ✓
8. advantage = mean-centered 不除 std ✓
9. rollout τ=1 纯采样；greedy 仅 eval ✓
10. OrderGap 表述已按 frozen 数据修正 ✓

## 47. Conditions Required Before Formal Pilot Launch

- freeze review PASS（用户裁定，当前等待中）；
- pilot 启动前复核 free disk ≥ 30GB；
- 以下实现项（freeze 后、启动前完成，不改协议）：
  - training/rl.py：formal fixed eval 切换为 §35 的 64-sample 独立子集（manifest 索引 64–127）+
    step-0 评估点 + greedy eval 口径（复用 audit 脚本的 argmax 变体，同 seed 同 J）；
  - run_metadata 增补 §41 字段；
  - 启动命令：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python training/rl.py
    rl.n_steps=500 rl.lr=3e-6 rl.name=rlpilot`（recipe 唯一 override = n_steps=500 与 name；
    其余全部 config 冻结默认值）。

---

## Appendix A. v0.2 → v1.0 Change Log（freeze 修订）

| # | 章节 | 修订 |
|---|---|---|
| 1 | §0 新增 | Freeze Record：LR=3e-6 裁定（含逐字表述与"不声称统计显著"限制）、三条 probe 证据表、§46/§47 条件全部满足的证据、freeze 元数据 |
| 2 | §6/§25 | RL method 更名为 **safe-region / truncated transition policy gradient**；J support=91.9%、truncation=8.1% 写入协议；禁止"full reverse-trajectory unbiased PG"表述 |
| 3 | §25 | SIGMA_GATE_MIN=0.05 语义定案：完整 rollout 仍执行全部 128 步，0.05 只排除 K=1 timestep 采样；0.05 = SEDD-small + analytic + 128-step 下经验验证的 numerical-validity boundary，非理论通用常数 |
| 4 | §35 | NLL evaluation 定案：所有 checkpoint/raw/EMA 必须使用同一冻结 corruption realization（seed+1000）；seed+2000 极端低-σ realization 定性为 high-variance/pathological，非"错误数据" |
| 5 | §35/§13 | evaluation hierarchy：RAW=PRIMARY / EMA=SECONDARY；sampled reconstruction 不得单独作短 horizon task-learning 判据（gumbel-stable 分辨率限制）；greedy reconstruction = deterministic supplementary task metric |
| 6 | §35/§47 | formal fixed task eval：64-sample 独立子集（manifest 索引 64–127），评估点 0/50/100/250/500，完全复用相同 samples/corruption/decoding；成本估算 ≈20–25 min 总计，预算允许不缩减 |
| 7 | §29 | allocator 定案：PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，分类 memory/allocator engineering configuration，非 algorithmic change；记入 run_metadata |
| 8 | §27/§28 | formal LR=3e-6（constant 无 warmup）冻结；LR calibration 终止（不再搜索 5e-6/7e-6/中间值）；禁止 CPI/OrderGap 选 LR |
| 9 | §21/§45 | 里程碑状态更新：smoke PASS、24 单测全过、RL-G0、128 gate、LR probe 全部完成 |
| 10 | §39 | OOM 规则显式化：第一次 OOM 立即停止不自动重启 + 完整诊断清单 |
