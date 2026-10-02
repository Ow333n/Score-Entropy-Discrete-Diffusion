# Post-Training RL Plan v0.2 — DRAFT（未冻结，审查中）

> **状态：DRAFT（不是 FROZEN / FINAL）**。任何 RL 代码、训练、评估在协议 freeze 前不得启动。
> v0.1 → v0.2 changelog 见文末 Appendix A；external review 的 11 条 mandatory corrections 已全部纳入。
> 前置 Git 快照：`pre-rl-compatibility-v1` @ `de69bb9`。
> 技术依据：`reports/rl_reverse_transition_analysis.md`、`reports/repo_audit_summary.md`。

---

## 1. Research Motivation

SEDD 离散扩散 LM 的 reveal-order compatibility 已被证明受 post-training 影响：Vanilla SFT 使
CPI 0.328→~0.28、OrderGap 10.24→~8.8（v4.1），且 P1 定位 CPI 衰减为 warmup 后期（~250–1020 步）
的连续过程。训练信号是"监督"。当前未知：当训练信号切换为 **on-policy reward optimization** 时，
compatibility 会继续衰减、保持、还是反弹？这决定 post-training 信号类型是否是 compatibility
的一个自由度。

## 2. Current Frozen Findings（v4.1 + P1，不可改动）

- Vanilla SFT（SEDD-small、WikiText103 span-infilling、2 seeds）使 CPI_abs 0.3281→0.283–0.289（−12~14%）、OrderGap_raw 10.24→~8.8（−14%），paired bootstrap CI 不含 0。
- Stage-4 判决 **B_DIAGNOSTIC_ONLY（永久固定）**。
- P1（CPI）：衰减定位在 250–750 窗口（s1=(250,500)、s2=(500,750)，EMA 口径），0→250 平台、1020→2500 平台；signed δ 全程 ≈0（对称收缩）。
- P1（OrderGap）：**CPI 与 OrderGap 的早期动力学不完全同步**——OrderGap 在 0→50 已出现小幅显著变化（≈−0.03 nats，四口径一致，方向为下降；P1 报告 paired Δ 表以 pretrained−ckpt 记为正号 +0.03，注意该正号表示下降），而 CPI 在 0→250 完全平台；OrderGap 随后 100→1020 连续大幅下降，s2 在 1020→2500 仍显著下降而 CPI 已平台。
  > ⚠️ external review 曾表述为 "small initial increase before later decline"；P1 数据方向为 **decrease**（10.24→10.22）。v0.2 按数据写，若 review 持有不同数据请提供（frozen 数据禁止改写）。
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
**不复制该 checkpoint**：RL run 目录记录其绝对路径 + SHA-256（run_metadata），step-0 评估直接引用（§31/§43）。

## 13. SFT-to-RL Transition

- model weights ← SFT EMA-10200；optimizer ← **fresh AdamW**（不继承 SFT moments：objective 已变）。
- scaler ← fresh GradScaler。
- **EMA policy（明确）**：
  - shadow 初始化 = RL 初始权重（即 SFT EMA-10200 权重本身）；
  - `num_updates` **reset 到 0**（RL 阶段独立计数；不复用 SFT 的 10201）；
  - decay 参数 0.9999，但 pytorch_ema 存在 decay warmup：`effective_decay = min(0.9999, (1+n)/(10+n))`
    （v4.1 审计 ②）——RL 早期 EMA 处于 warmup 区间，报告中必须引用 effective decay 而非 0.9999；
  - **评估口径：RL early dynamics 以 raw 为 primary、EMA 为 secondary**（理由：EMA 在 decay warmup 中，
    早期不能代表慢移平均；raw 直接反映 PG 更新轨迹）。
- RL step 0 由 RL-G0 gate 验证（§33）。

## 14. Definition of Policy in SEDD

见 `reports/rl_reverse_transition_analysis.md`（技术 annex）。policy 要点（v0.2 定稿口径）：

- model 输出 **log-score**（[B,L,50258]，x_t 位置 logit 恒 0）；一切概率从 exp(score) 构造。
- policy 是 **per-position categorical**。给定 x_t，每个位置在 D=50258 个状态上有一个转移权重向量 w_θ(x_t, σ)：
  - **analytic predictor 口径（候选 A，推荐）**：`w = staggered_score(score, dσ) × transp_transition(x_t, dσ)`
    （元素乘，见 reverse-transition 分析 §4.2；staggered_score 内部含 exp(dσ) 缩放与 MASK 列补偿）。
  - Euler/τ-leaping 口径（候选 B）：`w_v = dt·dσ·exp(score_v)`（v≠x_t）、`w_keep = 1 − Σ_v w_v` —— 见 §14.1 资格限制。
- **policy 分布与 rollout sampler 完全同源**（同一 w 向量）：
  ```
  π_θ(v | s) = w_v / Σ_u w_u            （正规化 categorical）
  logπ_θ(v | s) = log(w_v) − logsumexp(log w)
  ```
  rollout 采样用 `sample_categorical(w)`（gumbel 技巧按 positive weights proportional sampling，
  **不需要显式归一化**）；logπ 按上式用同一 w 计算。禁止"采样用一套分布、logπ 用另一套"。
- **动作语义**：对 MASK 位置，动作 = 保持 MASK 或跳到某干净 token；**MASK→MASK (stay) 若其权重依赖 θ
  则属于 stochastic action，必须计入 logπ 与 PG**（stay 权重 = staggered 公式的 MASK 列项，依赖 θ）。
  非 MASK 位置是确定性吸收（转移权重只集中在当前 token），**logπ=0，从 PG 中排除**。
- 每位置 logπ 求和得 transition-level logπ；joint trajectory logπ = Σ_steps Σ_positions（RL-1 smoke 用）。

### 14.1 Analytic-weight validity gate（v0.2 新增，每次 rollout step 强制执行）

- 对每个位置的权重向量 w 检查：**全部 finite、全部 ≥ 0、总质量 Σw > 0**。
- 违反 → 记录 (batch, step, position, w_min, Σw) 并 abort 该 step 所在 batch 的 rollout（不静默 clamp）。
- numerical negative 处理：若 w_min < 0 但 |w_min|/Σw ≤ 预注册 tolerance（**1e-6**，仅浮点噪声），
  允许放行但计入统计（负权重位置计数随 run_metadata 落盘）；超过 tolerance → abort 并 debug。
- **Euler 口径资格条款**：若 Euler 的 effective categorical weights（含 keep 项）出现真实负数
  （keep 概率 < 0，即 dt·dσ·Σexp(score_v) > 1），**取消 Euler 作为 RL-policy candidate 的资格**，
  不是仅记录诊断。该判定由 §34 测试在 rollout 统计上做出（正式 rollout 参数下的负权重率）。

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

## 18. Reward Definition（v0.2 修正：partial-reveal 口径）

- **M0 = rollout 初始时 span 内实际被 mask 的位置集合**（`initial_state == MASK` 且 `span_mask`）。
- Primary reward：
  ```
  R = |{ i ∈ M0 : x̂_i == x_i }| / |M0|        （在初始 mask 位置上评估 token 重建正确率）
  ```
  不在 M0 上的 span 位置（partial-reveal 下初始就可见）**不参与 reward**——模型从未被要求重建它们，
  计入会稀释信号。
- **Group 同源约束**：同一 prompt 的 G 条 rollouts 必须共享 clean sample / span / σ / initial mask
  （同一次 `corrupt_span_batch` 产物），只改变 rollout RNG（采样随机性）。
- 退化情形预注册：|M0| = 0 的 prompt 排除并重采样（record 计数）；|M0| ≥ 1 即有效，pilot 阶段
  统计 |M0| 分布并在正式协议中定下限。
- R ∈ [0,1]，长度归一，无需 reward model。secondary 指标（exact-match / edit distance / semantic）仅记录。

## 19. Reward Limitations

position-wise 口径对语义等价输出（bought→purchased）给低分。第一版不混入 semantic score（保持 reward
纯净，避免把变化归因复杂化）；semantic reward 留作后续 ablation。

## 20. Why CPI / OrderGap Are Diagnostics, Not Rewards

R = −CPI / −OrderGap / −|δ| 一律禁止（含 hidden regularizer 形式）。目的是观察 task-reward 优化是否
自然改变 compatibility；直接 reward compatibility 会使结论失去科学价值。

## 21. Simplified Policy-Gradient Smoke Test（RL-1）

- **exact joint transition logprob**：`logπ(τ) = Σ_{steps∈τ} Σ_{i∈masked} logπ_θ(a_{t,i} | s_t)`（含 stay 动作）。
- 纯 on-policy：**不 replay、不 clipping、不 ratio**。L_PG = −(R − batch_mean(R)) · logπ(τ)。
- 目标（全部通过才进入 RL-2/RL-3）：rollout 完整、reward 正常、logπ 可重算（单测 5）、对 θ 有梯度、
  PG 更新非零、无 NaN、reward 出现学习信号、模型不立刻 collapse、SFT 任务性能无灾难性下降。

## 22. Group-Relative Advantage（v0.2 修正）

- 同一 prompt（共享初始腐蚀，§18）采样 G=4 条独立 rollouts。
- **Primary advantage = mean-centered**：
  ```
  A_i = r_i − mean_{g∈G}(r_g)
  ```
  **不除 std**（贴近 AGRPO official；std 仅 logging）。standardized advantage（除 std）留作后续 ablation。
- 无 critic。组内 std≈0 时 A 全 0 → 该组贡献 ≈0（PG 无害），记录计数。

## 23. diffu-GRPO Discussion

- 思想：把扩散生成整体当作一条序列似然做 group-relative PG，使用 sequence/one-step likelihood 近似。
- 与 SEDD 匹配难点：SEDD 的 exact 生成似然是逐位置 CTMC 转移的乘积（§14），diffu-GRPO 的近似 likelihood
  与真实 sampler 分布不一致 → old/new logπ 口径错位、estimator bias 风险。
- 复杂度/成本：需要完整轨迹 likelihood 重算，8GB 下负担重。
- 用途：related method / optional baseline，**不优先**。若实现，必须按 §34 的单测验证其近似口径。

## 24. AGRPO-style Main Method（RL-3）

- Markov transition formulation：直接利用 SEDD 逐位置 categorical 转移（§14）—— AGRPO 的
  step-level PG 与 SEDD 结构天然对应（每 reverse step 是一个 Markov transition）。
- **per-position ratio / clipping**（不是 joint ratio）：
  joint ratio 是几十个 position ratio 的乘积，极易爆炸；RL-3 在每个 position 上独立
  ρ_i = π_new(a_i|s)/π_old(a_i|s) 并逐 position clip（§26）。
- Step-level PG：完整 rollout no_grad；每轨迹按 §25 采样 K 个 timestep → 重算 logπ → PG。
- 不照搬 LLaDA AGRPO 代码：先按 §14 的 SEDD 公式从零推导。
- 数学有效性：logπ 可微可重算（reverse-transition 分析 §6），single-state backward 显存可行。

## 25. Timestep Sampling（v0.2 修正：MC target 明确定义）

- **Primary MC target：uniform timestep expectation**
  ```
  g(θ) = E_{J ~ Uniform(steps)} [ ∇_θ logπ_θ(a_J | s_J) ]
  ```
  q(J) = uniform → 采样 J 后直接贡献 ∇logπ_J，**无需 1/q 乘子**。
- 若后续使用 non-uniform sampling 分布 q(t)（noise-weighted 等消融）：
  ```
  weight = p_target(t) / q_sample(t)
  ```
  其中 p_target = uniform。
- **unbiasedness 声明范围（严格限定）**：仅对 **unclipped、on-policy 的 timestep-MC estimator** 声明
  （q 覆盖 target 支撑时它是 g(θ) 的 IS 估计）。**不对 clipped / group-relative / off-policy replay
  的完整算法做 unbiased 声明**。
- Pilot：K=1、uniform、不含最终 denoiser 步（denoiser 的分布口径单独审计后再议）。消融候选：
  noise-weighted q、K=2/4、含 denoiser 步（需 protocol 修订）。

## 26. Importance Ratio / Clipping（v0.2 修正：PPO min 形式）

- ratio（per-position）：ρ_i = exp(new_logπ_i − old_logπ_i)（同一 transition、同一 §14 公式重算）。
- **Objective（标准 PPO clamp，修正 v0.1 的 −A·clip(ρ) 错误）**：
  ```
  L_i = −min( ρ_i · A ,  clip(ρ_i, 1−ε, 1+ε) · A )
  ```
  A 为 trajectory-level group advantage（§22）广播到该轨迹所有采样 position；
  L = Σ_{sampled positions} L_i。ε = 0.2（pilot 冻结）。
- **RL-1 与 RL-3 的明确区分**：
  - RL-1（smoke）：joint logπ、纯 on-policy、无 ratio、无 clip（§21）。
  - RL-3：per-position ρ_i + clip；**禁止 joint ratio**（几十个 position 的乘积爆炸）。
- 多 epoch 重放同一 rollout：old_logπ 固定于 rollout 时刻；ratio 随 epoch 失真 → pilot 限制 ≤2 epochs，
  超限/ratio 均值爆炸进 abort（§39）。不做 KL 约束（第一版无 reference model；用 clipping + 小 LR
  控制漂移；记录参数漂移诊断量）。

## 27. Optimizer Strategy

Fresh AdamW（β=(0.9,0.999), eps=1e-8, wd=0），grad_clip=1.0，warmup 暂定 0（从 SFT 已训权重起步，
RL 阶段是否需要 warmup 在 LR pilot 中观察决定）。scaler 沿用 amp。

## 28. LR Pilot

候选 {1e-6, 3e-6, 1e-5}，优先 **3e-6**（SFT 的 3e-5 对高方差 PG 过激，先验）。选择依据：grad norm、
reward 趋势、ratio 分布、NLL/重建 acc 漂移、稳定性。**不得在 pilot 前把任何 LR 写死为正式值。**

## 29. GPU / VRAM Plan

- Rollout：eval 模式 no_grad，batch 小（4-8 prompts × G=4），峰值 ~2-3GB（CPI 评估同量级）。
- Training：单 transition 重算 forward+backward（gradient checkpointing 沿用），微 batch 8–16，
  accum 2–4；峰值目标 ≤6.4GB（SFT 基线）。
- 无 critic、无常驻 reference model、无 rollout 图驻留。old_logπ 存 detached 数值（非旧模型副本）。

## 30. Rollout Memory Plan

每步只落盘 §17 的数值字段。轨迹缓存目录 `exp_local/regime_a/rl-*/trajectories/`（按 batch 分文件，
写入用 .tmp+rename 防半截文件——吸取 P1 崩溃教训）。**每 batch 训练完成后释放缓存**（§43）。

## 31. Checkpoint Plan（v0.2 重做：最小化磁盘）

- **step 0**：不复制 SFT checkpoint；run_metadata 记录源路径 + SHA-256 指针，评估直接引用源文件。
- **step 50 / 100 / 250**：**eval snapshots**（只含 `{model, ema, step}` ≈ 1.4GB/个，供 raw+EMA 评估）。
- **step 500**：full checkpoint（`{model, ema, optimizer, scaler, step}` ≈ 2.7GB）。
- **checkpoint_latest**：单个 **atomic rolling** 全量检查点（写 .tmp → rename；每 25 步覆盖一次，
  用于断点续训；断电/崩溃时只有它可能损坏，且损坏只丢最近进度）。
- raw + EMA 双权重评估口径照旧（v4.2）。
- 磁盘目标见 §43（pilot 新增 <10GB）。

## 32. RL Pilot Horizon

~500 optimizer steps；评估点 0/50/100/250/500。若 reward 无学习信号 / NaN / collapse / NLL 灾难性
退化 / ratio 爆炸 → debug，不扩大训练。pilot 通过后才设计 formal horizon。

## 33. Preflight Gates

- **RL-G0（step-0 gate）**：RL 初始化的模型（EMA-10200 加载后）在 frozen manifest 上复算 CPI/OrderGap/G1，
  与 v4.1 formal s1-10200 结果逐字段一致（容忍浮点噪声，预注册阈值）；state_dict 与 SFT checkpoint
  EMA shadow 逐 tensor 一致。不一致 → RL 停止。
- **RL-G1（transition gate）**：§34 单测全过 + rollout/recompute logπ 一致性（单测 5）。
- **RL-G2（pilot 放行）**：G0+G1 通过 + LR probe 完成 + 磁盘前置满足（free disk ≥30GB，§43）。

## 34. Correctness Unit Tests（v0.2 清单 = 原 14 项 + external review 8 项）

1. transition 概率正规化（Σπ=1）；
2. **exact normalized-weight sampler frequency test**：用 `sample_categorical(w)` 大样本对拍 π=w/Σw 频率；
3. 采样动作 logπ 有限；
4. old logπ detached；
5. 更新前重算 logπ == rollout logπ（逐位）；
6. PG 梯度有限；
7. 随机轨迹 reward 无关性 sanity；
8. group 内 advantage 均值 ≈0（mean-centered，§22）；
9. 零 advantage → 更新 ≈0；
10/11. toy 上正/负 advantage 分别增/减选中转移概率；
12. reward 无梯度路径（reward 张量 no_grad 验证）；
13. 固定 seed 复现；
14. RL step0 == SFT init；
15. **analytic weight positivity test**（w 全 finite/≥0/Σw>0，§14.1）；
16. **deterministic non-mask position logπ=0 test**；
17. **masked stay-action included test**（stay 动作计入 logπ 且对 θ 有梯度）；
18. **same initial corruption across G rollout test**（G 条轨迹共享 x0/span/σ/M0，仅 RNG 不同）；
19. **initially-masked-only reward test**（reward 只在 M0 上计算）；
20. **PPO negative-advantage clipping test**（A<0 时 min 形式正确作用、梯度方向正确）；
21. **K=1 timestep MC estimator toy test**（有限状态 toy 上 uniform 采样估计 E_J[∇logπ_J]，无 1/q 乘子，对拍解析值）；
22. toy finite-state Markov chain 上验证 PG 实现。

## 35. Task Evaluation

每个评估点：mean rollout reward、token reconstruction acc（M0 口径）、exact-match rate、masked NLL、
greedy reconstruction acc（eval_task.py 口径）、可选 semantic 指标。

## 36. Compatibility Evaluation

沿用 frozen manifest + eval_cpi.py / eval_order_gap.py（raw primary / EMA secondary）：CPI_abs、CPI_RMS、
signed δ、OrderGap_raw、per-token；若成本允许加 buckets。与 Pretrained/SFT 阶段直接可比（同一 manifest）。

## 37. Seed Strategy

Pilot/take-home：**SFT seed1 → RL seed1**（单链）。signal 存在后再加 SFT seed2 → RL seed2。
单个 RL seed 不作 universal conclusion。

## 38. Statistical Analysis

延续 v4.1 纪律：同一 manifest 上的 paired sample bootstrap（SFT vs RL 同样本差），只代表 sample
uncertainty，不代表 training-seed significance。两 seed（若做）分别报告 + 均值，不夸大两点均值。

## 39. Failure / Abort Criteria（预注册）

Abort → debug（不当科学结果）：transition 权重不通过 §14.1 validity gate（真实负权重/非 finite/零总质量）；
rollout 与重算 logπ 意外不一致；梯度 NaN/Inf；reward collapse / variance≈0 / group 内全同；ratio 爆炸
（max>20 或 mean>5）；NLL 灾难性退化（Δ>+1 nat）；token acc 崩塌；生成无效状态（含 MASK/越界）；
RL-G0 不通过；8GB 下 OOM 无法用合理显存策略解决；Euler 候选触发 §14.1 资格取消后未切换口径。

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
rollout RNG seeds、trajectory 缓存位置、§14.1 负权重统计、M0 分布统计。所有随机源显式 seeded；
每步日志含 reward/advantage/ratio/近似诊断。

## 42. Demo Plan

最终展示：同一 corrupted span 上 SFT vs RL 输出对比 + token acc / reward / NLL + 三阶段（pretrained→SFT→RL）
CPI/OrderGap/task 指标图。Gradio 后置，先保证科学正确。

## 43. Disk / Trajectory / Checkpoint Budget（v0.2 重做）

- **轨迹（禁止全量落盘）**：K=1 时 rollout 前预采样 timestep J，只缓存 §17 的字段（单 transition +
  final output + reward）；完整 trajectory 仅小型 debug subset（每 batch ≤1 条、≤20 步）。
  每 batch 训练后释放缓存 → 轨迹缓存稳态占用 <1GB。
- **checkpoint（§31）**：step0 = 源指针（0 磁盘）；eval snapshots 50/100/250 ≈ 3×1.4 = 4.2GB；
  step500 full ≈ 2.7GB；rolling checkpoint_latest ≈ 2.7GB（原子覆盖，不增长）。
  **合计稳态 ≈ 9.6GB < 10GB**（不含已有 SFT 源 checkpoint，其磁盘占用已在 P1 阶段存在）。
- **前置条件**：pilot 开始前 free disk ≥ 30GB（当前 D: 仅 ~19GB —— 必须先清理/compact/迁盘，
  见 blockers）。所有写入走 .tmp+rename（防半截文件）。
- GitHub 只存源码/config/protocol/结果 JSON/报告/hash，不存 checkpoint（`.gitignore` 已挡）。

## 44. Git / Versioning Plan

- 一切 RL 代码在 `rl/`、`training/rl.py`、`tests/test_rl_*` 新文件内进行；不动 v4.1/v4.2/P1 相关文件。
- 协议版本号：本 DRAFT → 审查 → `post_training_rl_protocol_v1.0_FROZEN.md`（freeze 后任何改动递增
  版本并重跑受影响部分，纪律同 v4.2 §7）。

## 45. Implementation Roadmap

1. 存储：清 junk / compact / 迁盘（先决，§43）。
2. `rl/transition.py`（§14 的 w/logπ/validity gate）+ 单测 1–21（CPU 可跑）。
3. RL-G0（EMA-10200 加载对拍）。
4. `rl/rollout.py` + `rl/reward.py`（§17/§18，小批量 smoke）。
5. `training/rl.py`：RL-1 joint PG（§21）→ RL-2 group-relative（§22）→ RL-3 per-position PPO（§24/§26）。
6. LR probe（1e-6/3e-6/1e-5，各 ~100-200 步）。
7. RL pilot（评估点 0/50/100/250/500）→ 全评估（task + compatibility）。
8. 判读（§40 矩阵）→ 决定 formal horizon / seed2 / P2。

## 46. Open Questions（freeze 前必须定案）

1. rollout predictor 定案：analytic（候选 A）vs euler/128（候选 B，受 §14.1 资格条款约束）——单测 2/15 的
   大样本对拍与负权重统计决定。
2. steps 数：1024（run_sample 同款）vs 128（config 默认）——按 rollout 成本与质量 pilot 前测。
3. denoiser 步是否进 PG 采样（pilot 排除，formal 再议）。
4. K=1 是否足够（variance/学习速度的 probe 数据决定 K）。
5. 多 epoch 重放上限（pilot 先 2）。
6. warmup 是否需要（LR probe 决定）。
7. |M0| 下限（pilot 统计分布后定）。
8. 组内 std≈0 的 advantage 处理已定（全 0），无需再议；standardized advantage 留 ablation。
9. 采样温度：rollout 是否用 τ<1 或 argmax 混合（第一版 τ=1 纯采样，记录 diversity 诊断）。
10. external review 的 OrderGap "initial increase" 表述与 P1 数据方向（decrease）矛盾——待与 review
    确认（§2 的 ⚠️），frozen 数据不改写。

## 47. Conditions Required Before Protocol Freeze

- §46 全部定案；§34 单测 1–21 全过；RL-G0 通过；LR probe 完成；存储前置满足（free disk ≥30GB）；
- 本 DRAFT 经用户 + external review 无异议 → 提升为 `post_training_rl_protocol_v1.0_FROZEN.md`。
- freeze 前禁止任何 RL training 启动。

---

## Appendix A. v0.1 → v0.2 Change Log（external review corrections）

| # | 章节 | 修正 |
|---|---|---|
| 1 | §14 | policy 定义改为直接对 analytic 权重 w 正规化：π = w/Σw、logπ = log(w) − logsumexp(log w)；明确 rollout sampler（sample_categorical 按 positive weights proportional sampling）与 logπ 完全同源（同一 w 向量），删除 "logits→softmax" 表述 |
| 2 | §14.1 新增 | analytic-weight validity gate：finite / ≥0 / Σw>0 检查；不静默 clamp；numerical negative 统计 + tolerance=1e-6 预注册；Euler 出现真实负权重 → 取消其 RL-policy candidate 资格（不只记录） |
| 3 | §18 | reward 改为 M0 口径（只评初始 mask 位置），G rollouts 共享 clean/span/σ/initial mask、仅 RNG 不同 |
| 4 | §25 | MC target 明确定义为 uniform timestep expectation；q=uniform 无 1/q 乘子；non-uniform 时 weight=p_target/q_sample；unbiasedness 声明限定于 unclipped on-policy timestep estimator |
| 5 | §26 | PPO objective 改为标准 min 形式：L = −min(ρ·A, clip(ρ,1±ε)·A)（v0.1 的 −A·clip(ρ) 为错误形式） |
| 6 | §21/§24/§26 | 明确 RL-1 = joint logπ 纯 on-policy 无 replay 无 clip；RL-3 = per-position ratio+clip（禁止 joint ratio 爆炸）；MASK→MASK stay 属 stochastic action 计入 PG；非 MASK 位置 logπ=0 排除 |
| 7 | §22 | advantage 改为 mean-centered（A=r−mean_G(r)），std 仅 logging；standardized 留 ablation |
| 8 | §13 | EMA policy 明确：shadow init、num_updates reset、decay warmup 引用 effective decay、评估 raw primary / EMA secondary |
| 9 | §2 | P1 OrderGap 描述修正：CPI 与 OrderGap 早期动力学不同步；⚠️ external review 的 "initial increase" 方向与 P1 数据（−0.03 nats decrease）相反，v0.2 按 frozen 数据写，待与 review 确认（见 §46.10） |
| 10 | §31/§43 | 禁止全量轨迹落盘（K=1 预采样 J、只存单 transition + debug subset）；checkpoint 改为 step0 指针 + eval snapshots 50/100/250 + step500 full + 单个 atomic rolling latest；pilot 新增 <10GB；前置 free disk ≥30GB |
| 11 | §34 | 新增 8 项单测（weight positivity、normalized-weight sampler frequency、non-mask logπ=0、stay-action included、G 共享初始腐蚀、M0-only reward、PPO negative-advantage clipping、K=1 MC estimator toy）→ 共 22 项 |
