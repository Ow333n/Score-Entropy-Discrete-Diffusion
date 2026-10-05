# Mechanism Pilot Protocol — Why Does Partial-Reveal SFT Reduce Reveal-Order Compatibility?

> 版本：**v1.2**（2026-10-06，含 v1.2 四项修正，见文末修订记录）
> 状态：**未启动训练**；pilot 须在本协议冻结并完成 §9 实现与测试后方可开始。
> 上游现象（v4.1/v4.2 已冻结）：Pretrained → SFT 后 CPI 0.3301→0.2871、OrderGap
> 10.2246→8.7475（两 seed 方向一致）；SFT→RL-500 后 compatibility 基本保持。
> 本协议只做机制归因，不新增泛化实验。

---

# 0. 研究问题与范围

**核心问题**：vanilla partial-reveal SFT 降低 CPI/OrderGap 的机制是什么？

- 本阶段（pilot）只检验一个特定机制假说：**mask-context diversity（fresh 重采样）
  是否通过多上下文监督隐式耦合不同揭示路径的条件预测**（§2–§3）。
- 本阶段**不做** explicit compatibility loss、compatibility curriculum、新 RL 路线。
- 本阶段**只跑 pilot**；pilot gate（§5）通过后，另立 confirmatory 协议并增加训练 seeds。

# 1. 数学表述（与仓库实际实现逐项一致）

Source of truth = 仓库当前代码。公式与代码的对应关系如下，**禁止**将训练描述为
"直接 CE 监督四个 conditional probabilities"。

## 1.1 噪声调度（noise_lib.LogLinearNoise）

t = (1−ε)·U[0,1] + ε，ε = 1e-3（corruption.py `EPS`）：

\[
\sigma(t) = -\log(1 - (1-\varepsilon)t) \qquad (\texttt{LogLinearNoise.total\_noise}, \ \text{noise\_lib.py})
\]
\[
\mathrm{d}\sigma(t) = \frac{1-\varepsilon}{1-(1-\varepsilon)t} \qquad (\texttt{LogLinearNoise.rate\_noise})
\]

由定义直接得 **q_t = 1 − e^{−σ_t} = (1−ε)·t_t**（吸收转移的 mask 概率）。

## 1.2 单 token 的 score entropy（graph_lib.Absorbing.score_entropy）

模型对 partial-reveal 状态 x_t 输出 **log-score ℓ_θ = s_θ(x_t) ∈ R^{B×L×(V+1)}**
（`model/utils.py get_score_fn → log_score_fn`）。对每个被 mask 的位置 i：

\[
\mathrm{SE}_i = \underbrace{\sum_{v \ne \text{[MASK]}} \exp(\ell_{i,v})}_{\text{pos\_term}}
- \underbrace{r(\sigma)\,\ell_{i,x_0^i}}_{\text{neg\_term}}
+ \underbrace{r(\sigma)\big(\log r(\sigma) - 1\big)}_{\text{const}},
\qquad
r(\sigma) = \frac{1}{e^{\sigma}-1}
\]

对应 `graph_lib.py:244-269` 的 `Absorbing.score_entropy`：
`ratio = 1/expm1(σ)`；`pos_term = score[rel_ind][:, :-1].exp().sum(-1)`
（即 sum exp(log-score)，排除最后一列 [MASK]）；`neg_term = ratio · gather(ℓ, x0)`
（即 −r(σ)·log-score_GT）；`const = ratio · (ratio.log() − 1)`。三个项的逐字对应关系
由 `tests/test_policy_loss_formula.py` 在 CPU 上数值对拍（≤1e-6）。

## 1.3 单步训练目标（losses.py + training/vanilla.py）

\[
L_{\text{step}}(\theta) =
\frac{1}{B}\sum_{b} \frac{1}{|M_b|}
\sum_{i \in M_b} \mathrm{d}\sigma(\sigma_t)\,\mathrm{SE}_i,
\]

其中 M_b 为该序列 span 内被 mask 的位置集合（`vanilla.py span_task_loss`：
`weighted = (dsigma[:, None] * se) * support`，`per_seq = weighted.sum(-1)/n_support`，
再 batch 均值；`losses.py:31` 的 dσ 加权口径同源）。即：**dσ 加权 + masked-span
支撑集按序列取均值 + batch 均值**，非全序列 CE。

## 1.4 δ / CPI / OrderGap 是模型蕴含量，不是训练目标

对 span 内有序 pair (a, b) 与共同上下文 C，评估器构造三个 partial-reveal 状态并读取
**同一个 denoiser** 的隐含条件概率（与 `compatibility/cpi.py evaluate_delta_swap_batch`
同源）：

- **C-状态**（a, b 均 mask）→ 模型蕴含量 p_θ(a|C)、p_θ(b|C)
- **C+a 状态**（a clean，b mask）→ 模型蕴含量 p_θ(b|C, a)
- **C+b 状态**（b clean，a mask）→ 模型蕴含量 p_θ(a|C, b)

\[
\delta_{a,b|C} = \log p_\theta(a|C) + \log p_\theta(b|C,a)
- \log p_\theta(b|C) - \log p_\theta(a|C,b)
\]

CPI = E|δ|（frozen 评估口径）；OrderGap = max_π Q_π − min_π Q_π。
**训练中不存在**任何"四个条件概率各自 CE"的头或项；C / C+a / C+b 只用于描述
不同 partial-reveal states 下的 score-ratio supervision 与 implied conditionals。

## 1.5 训练对四项蕴含量的监督事件（policy 访问哪些状态）

| 监督事件 | 状态 | 被监督的位置 | 对应蕴含量 |
|---|---|---|---|
| E_both | a, b 均 mask | 位置 a 和 b | p_θ(a|C)、p_θ(b|C) |
| E_a→b | a clean，b mask | 位置 b | p_θ(b|C, a) |
| E_b→a | b clean，a mask | 位置 a | p_θ(a|C, b) |
| E_neither | a, b 均 clean | 无 | — |

机制假说（§2）的核心是：**三类监督事件的暴露频率与结构**决定共享参数下四项蕴含量
被耦合的程度。

# 2. Mechanism Hypothesis（v1.2 冻结版）

## 2.1 直觉版

Fresh random partial-reveal SFT 让同一段 ground truth 在多个互补可见性上下文下被反复
denoising 监督：对 pair (a,b)，有时两个都被 mask（同时监督两个边际）、有时 a 可见只
监督 b、有时 b 可见只监督 a。共享参数 + 同一 GT 目标使不同揭示路径的条件读数互相
拉近 → |δ| 下降；路径级同理 → OrderGap 下降。

## 2.2 形式版（q(1−q) 口径，统一）

设某步 mask 概率为 q（=1−e^{−σ}）。对任意有序 pair：

- **单侧互补事件**（a mask ∧ b 可见）概率 = **q(1−q)**；
- **exactly-one-masked 总暴露**（两个有序方向之和）= **2q(1−q)**，关于 q 对称且在
  **q = 0.5 最大**。

机制主张（可证伪）：在 matched step 与 matched task performance 下，CPI/OrderGap 的
衰减幅度与互补可见性暴露量（2q(1−q) 加权累计）正相关——而非仅与总步数、总
masked-token 数或总 task 损失下降相关。

**P3 检验方式（v1.2 修正）**：训练 span 每步随机（§4.1），训练期 pair 与评估期 pair
无法按绝对位置对齐，per-pair co-visibility 矩阵在随机 span 下**没有良定义**——v1.2
删除 co-visibility 矩阵日志。P3 改由**可选常数-q 探针**检验（§5.1）；confirmatory 若
冻结 span，可恢复 co-visibility 逐对追踪。

# 3. 实验条件：A/B/C/D/E（v1.2 修订）

**共同冻结项**：同一 pretrained init（louaaron/sedd-small）；同一训练数据（wikitext103
train 256-chunk + shared data order）；同一 (sample, span, σ, K) 共享 schedule
（§4）；同一 optimizer / LR（3e-5）/ batch（32 effective，accum 1）/ 步数（2500）/
梯度检查点配置；仅 mask 位置选择 policy 不同。

**K_t 生成（保持原始 corruption 分布）**：原始 `corrupt_span_batch` 对 span 内每个
位置独立 Bernoulli(q_t)（corruption.py:33）。v1.2 采用与其**同分布**的分解：
**共享 K_t ~ Binomial(m_t, q_t)**，m_t = 该步该序列的 span_len（随机，§4.1），
q_t = 1−e^{−σ_t}。**禁止**用 deterministic round(m·q) 替代。条件于 K 时，iid
Bernoulli 的 mask 集是 K 子集上的均匀分布——这是 A 的构造依据（§3-A）。

| 条件 | Policy | 语义 |
|---|---|---|
| **A Fresh Random** | 给定 (m_t, K_t)，从 span 内均匀抽 size-K 子集（policy RNG 每次访问重抽） | 与原始 Bernoulli masking **逐分布等价**；最大 mask-pattern resampling / context diversity |
| **B Fixed Random** | 每 replicate 固定一条 [0,L) 的全序列随机排列 π_L（policy RNG 在 run 开头抽一次，全部样本共享）；给定 (s0, m_t, K_t)，mask span 内 **π_L 秩最小的 K_t 个位置** | 去掉"每次重采样"；**仍有** nested context diversity（σ_t/K_t 变化产生嵌套上下文链；K1<K2 时 mask(K1)⊂mask(K2)） |
| **C L2R Ordered** | mask span 最右侧 K_t 个位置（可见=左侧） | **directional stress test**（非机制因果条件） |
| **D R2L Ordered** | mask span 最左侧 K_t 个位置（可见=右侧） | **directional stress test** |
| **E Full-span（辅助）** | 整 span 全 mask | 仅辅助观察，不参与 gate |

**关键因果声明**：

1. **A vs B 是本协议的唯一天机制因果对照**。两者 (σ, span, K) schedule 逐位相同、
   单步边际分布相同（均匀 K 子集）；唯一差异是 A 每次访问重采样、B 固定排列。
   B **不是"无 diversity"**——它保留 σ/K 变化带来的 nested context diversity。
2. **C/D 降级为 directional stress test**。存在结构性 **positional exposure confound**：
   C 中 span 左侧位置从不被 mask（永不受 denoising 监督），右侧位置总被 mask
   （总受监督）；D 反之。C/D 结果**不得**单独用于机制因果结论（gate G2）。
3. 结构注记（v1.2 口径）：D ≡ B 在 π_L = identity 的特例；C ≡ B 在 π_L = reverse
   的特例。C/D 属于 B 的"固定排列"家族。

# 4. Paired-Seed / Common-Random-Number（CRN）方案（v1.2 修订）

## 4.1 共享 schedule 的完整组成（含 span 随机性）

**训练 span 是每步随机的，不是 frozen manifest 的 span**（vanilla.py:169 训练数据为
wikitext103 train 分块；corruption.py:23-27 每步采样 span_len ~ U[10,50] 与
span_start ~ U[0, L−span_len]；manifest 仅用于**评估**，SHA-256 `1897bd14...` 冻结）。

因此 replicate 内四 policy 共享的 schedule 是**完整五元组**：

\[
\text{shared schedule} = \big\{(\text{sample\_id}_j,\ \text{span\_start}_j,\ \text{span\_len}_j,\ \sigma_j,\ K_j)\big\}_{j=1}^{2500 \times 32}
\]

（sample_id 来自共享 data order；σ_j、span、K_j 来自共享随机流，§4.2）。每 run 落盘
该 schedule 的 SHA-256；replicate 内四 run 必须逐位一致，不一致硬停。评估用 frozen
manifest + frozen evaluator，与训练 schedule 完全分离。

## 4.2 随机流表（全部显式 torch.Generator，禁止全局 RNG）

| 随机流 | 种子 | 语义 |
|---|---|---|
| data_order | `1000 + r` | 每 epoch 训练分块洗牌顺序（DataLoader generator，replicate 内共享） |
| sigma_schedule | `2000 + r` | 逐步 t → σ、dσ（每 item 恰好消耗一次，与 policy 无关） |
| span_schedule | `3000 + r` | 逐步 span_len ~ U[span_min, span_max]、span_start（每 item 恰好消耗一次） |
| k_schedule | `4000 + r` | 逐步 K ~ Binomial(m, q_t)（显式 generator 实现，§3） |
| mask_policy_A | `5000 + r` | A 的 K 子集采样 |
| mask_policy_B | `6000 + r` | B 的 π_L（run 开头抽一次，全样本共享） |
| mask_policy_C/D | 无 RNG | 由 (s0, m_t, K_t) 确定性构造 |
| dropout | `torch.cuda.manual_seed(7000 + r)` | dropout=0 断言后为惰性流（若改 >0 须修订本协议） |
| evaluator | 固定独立种子 | frozen evaluator 全部随机流，与训练流永不共享（j_rng 隔离先例） |

- 模型初始化无随机性：8 个 run 均从同一 pretrained checkpoint 加载，逐位一致。
- 消耗顺序在 `task_data/policy_corruption.py` 中固定，任何改动须修订协议。
- 校验：dry-run（§9）先于训练，CPU 重放全部流并比对 digest。

# 5. 预注册预测与 Gate（v1.2 修订：G1 三值化）

## 5.1 预测

- **H1（diversity mechanism）**：matched-step 与 matched-performance 双口径下，
  CPI(A) < CPI(B)（OrderGap 同向，secondary）。
- **H2（directional bias，辅助）**：C 与 D 的 mean signed δ 符号相反；A/B 的
  mean signed δ 接近 0 且 |δ| 更低。不使用任何人为阈值；检验见 G2。
- **H3（generic task-learning alternative）**：matched-performance 下所有 policy
  的 CPI 相近 → 衰减来自通用监督适配（Path-Dependent Denoising Theorem 3 通道）。
- **H4（performance confound）**：matched-step 有差异但 matched-performance 后
  消失 → 衰减是 task learning 副产品。
- **P3（q(1−q) 通道）**：**可选探针**——固定 σ 的常数 q ∈ {0.2, 0.5, 0.8} 微型 run
  （每条件 500 步 × 1 replicate，仅 A 型 fresh policy），预测 q=0.5 衰减最大
  （2q(1−q) 在 0.5 最大）。非 gate 条件；仅作机制佐证。

## 5.2 Gate（G1 三值：PASS / FAIL / INCONCLUSIVE）

**G1 — diversity mechanism gate（primary）**

- G1a（matched-step）：step 2500 EMA checkpoint，frozen evaluator 500 样本逐样本
  配对差 Δ_i = CPI_i(A) − CPI_i(B)。判定：
  - **PASS**：pooled paired bootstrap 95% CI（10k 重采样，按 replicate 聚类）完全
    位于 0 以下，且两 replicate 点估计同号（均为负）。
  - **FAIL**：CI 完全位于 0 以上，或两 replicate 点估计均为正。
  - **INCONCLUSIVE**：CI 含 0，或两 replicate 点估计异号。
- G1b（matched-performance）：按 §5.3 规则配对 checkpoint 后重做 G1a。判定同 G1a；
  另加一条：**若 checkpoint 池过稀导致任何 policy 与 A 无 NLL±0.02 交集（no-
  overlap），G1b 判定 INCONCLUSIVE 而非 FAIL**——只允许在 confirmatory 加密
  checkpoint 后重新判定。
- **G1 总判定 = G1a ∧ G1b**（PASS 需两者皆 PASS；任一 INCONCLUSIVE → G1
  INCONCLUSIVE）。OrderGap 同法作 secondary 佐证，不单独构成 gate。

**G2 — directional auxiliary gate（辅助）**

- 检验：mean signed δ 的 C vs D 差异——paired bootstrap 95% CI（10k，按 replicate
  聚类）排除 0，且 sign(δ̄_C) = −sign(δ̄_D) 在两 replicate 一致。三值判定同 G1a
  （CI 含 0 或符号不一致 → INCONCLUSIVE）。预期符号方向不预先指定正负，只预注册
  "相反"。
- **G2 单独通过不得宣称 diversity mechanism 成立**。

**Gate 结论矩阵（预注册）**：

| G1 | G2 | 结论 | 下一步 |
|---|---|---|---|
| PASS | PASS | Case 1（最强） | confirmatory 规划 |
| PASS | FAIL/INC | diversity 成立但无方向证据 | confirmatory 加强 H2 设计 |
| INC | 任意 | **机制未决** | confirmatory：加密 checkpoint / 加训练 seeds 后重判 |
| FAIL | PASS | **不可宣称 diversity mechanism**；仅 directional bias | 停止或重设计 |
| FAIL | FAIL | Case 2/3/4 之一 | **停止，不进长跑** |

- 任何 FAIL/INCONCLUSIVE → pilot 停止或仅按上述矩阵进入 confirmatory 修正；
  **禁止看到结果后修改 hypothesis 或 gate 判据**（沿用 v4.1 Stage-4 裁定纪律）。
- **不确定性口径**：pilot 的 2 replicates 只用于 gate 判定；所有 CI 为 500 样本的
  sample-level 不确定性（bootstrap），**不代表 training-seed 不确定性**。
  G1 PASS 后，confirmatory 必须增加训练 seeds（目标 ≥4）后才可作机制结论。

## 5.3 Matched-performance 规则（预注册冻结）

- **primary task metric**：masked-span NLL（frozen manifest 500 样本，M0 位置均值，
  `evaluation/eval_task.py` G1 口径）；secondary：masked-position token accuracy。
- **容差**：NLL ±0.02 nats；acc ±0.01。
- **checkpoint 池**：每 run EMA @ {500, 1020, 2500} + 最终 weights @2500（§7）。
- **配对规则**：对 A 的每个参考 checkpoint，选另一 policy 的 NLL 最接近且落入容差的
  checkpoint；多候选取 step 更小者。
- **no-overlap 处理**：判定该 matched-performance 对比 **INCONCLUSIVE**（非 FAIL），
  confirmatory 加密 checkpoint 后重判。
- 配对一经选定不再调整；分析脚本与所选 pair 全部落盘。

# 6. 评估协议（与训练 policy 分离）

- **Frozen evaluator（主）**：与 v4.1/v4.2 完全相同——同一 clean 样本、同一 span、
  同一 pair 集合、同一上下文 C、同一揭示顺序集合。训练 policy 信息不进入评估器。
  指标：CPI_abs、CPI_RMS、signed δ mean、δ 分布、OrderGap_raw、task NLL、token acc。
- **OOD evaluator（次）**：不同 pair-distance 分桶（1–2 / 3–8 / 9+）；不同 mask-ratio
  分桶；额外揭示顺序集。
- **预注册分析**：
  1. CPI–OrderGap 相关性：sample-level、checkpoint-level、temporal 三层；
  2. pair-distance 分桶衰减差异；
  3. P3 常数-q 探针（若执行）；
  4. G1/G2 的 bootstrap 检验（§5.2）。
- **训练期日志**：σ/K 直方图（schedule 保真度）+ schedule SHA-256（§4.1）。
  完整 mask 集不落盘；co-visibility 矩阵**删除**（随机 span 下无良定义，§2.2）。

# 7. 存储计划（v1.2）

| 资产 | 数量/大小 |
|---|---|
| step0 共享 artifact（model+EMA，fp32，~170M 参数） | 1 份 × ~1.36GB |
| 每 run：EMA-only @500/1020/2500 + weights @2500 | 8 runs × (3×0.68 + 0.68) ≈ **21.8GB** |
| schedule 摘要/SHA-256 + run metadata + σ/K 直方图 | <0.1GB |
| **合计** | **≈ 23.3GB** |

- optimizer state 不保存（2500 步 ≈ 20 min/run，崩溃即重跑，不需 resume）。
- 紧张档：删 500 档 → ≈17.6GB（代价：G1b no-overlap 概率上升 → INCONCLUSIVE 增多）。
- 存放 D: 盘（C: 爆满教训）；评估产物入 `results/mechanism_pilot/`，不入 exp_local。

# 8. 结果解释模板（预注册冻结）

- **Case 1**：G1 PASS（含 matched-performance）→ 支持 mask-context diversity as
  implicit compatibility regularization；P3 正相关为机制佐证。
- **Case 2**：G1 FAIL/INC 但 ordered（C/D）与 random 族有差异迹象 → 机制更像
  non-directional masking / bidirectional exposure，而非 fresh resampling。
- **Case 3**：G1 FAIL 且 matched-performance 下各 policy 无差异 → generic
  supervised adaptation。
- **Case 4**：G1a PASS 但 G1b FAIL → attenuation 是 task learning 副产品的解释主导；
  G1b INCONCLUSIVE → Case 4 保持未决。
- **措辞纪律**：pilot 结论一律以"在 2 replicates、8GB、2500 步、SEDD-small 的
  pilot 设置下"开头；不得外推到规模、任务族或训练长度。

# 9. 实现与验证顺序（本阶段范围，不含训练）

1. 本协议冻结（用户复核通过）→
2. `task_data/policy_corruption.py`：共享 schedule 五元组生成（§4）+ A/B/C/D mask
   选择 + 显式 generator 的 Binomial 实现（不触碰任何 frozen 代码路径）→
3. 单元测试（CPU）：
   - `tests/test_policy_corruption.py`：A/B/C/D 的 mask 集合性质（§3，含
     K 子集均匀性、B 嵌套性、C/D 确定性、K=0 边界）、A 与原始 Bernoulli 掩码的
     分布等价、shared schedule 跨 policy 逐位一致与 SHA-256 相等；
   - `tests/test_policy_loss_formula.py`：§1.2 公式与 `Absorbing.score_entropy`
     的 CPU 数值对拍（≤1e-6）；
   - 既有测试回归：`scripts/run_tests.py`（compatibility 套件）与 RL 套件全绿；
     `losses.py / graph_lib.py / noise_lib.py / data.py / training/vanilla.py /
     task_data/corruption.py` 保持零改动（git diff 验证）。
4. Dry-run（CPU，无模型）：`scripts/mechanism_pilot_dryrun.py` 重放 2500×32 步
   schedule，输出每 replicate 四 policy 的 schedule digest 并验证跨 policy 相等 →
5. 交付 v1.2 协议 + loss↔代码对应 + 测试结果 + hash 检查结果 → 用户裁定 →
6. 训练（>2h 任务由用户 tmux 执行；8 run）→ Tier-1 评估 → Gate 判定 → 报告。

**未冻结 §9.2-§9.5 前，禁止启动任何 2500-step 训练。**

---

# 修订记录

- **v1.1（2026-10-05）**：① 数学表述改为 SEDD absorbing score-entropy，δ 四项为
  "同一 denoiser 在 C/C+a/C+b 状态下的模型蕴含量"；② A vs B 核心因果对照，B 保留
  nested context diversity；③ C/D 降级为 directional stress test + positional
  exposure confound；④ CRN 设计 + 四随机流分离；⑤ H2 废除 2×max(SD)，改 bootstrap
  CI + 预期符号；⑥ 统一 q(1−q)/2q(1−q) 口径；⑦ 冻结 matched-performance 规则；
  ⑧ 2 replicates 仅 pilot、CI 为 sample-level；⑨ 存储重做（≈23.5GB）；⑩ gate 拆
  G1/G2 + 结论矩阵。
- **v1.2（2026-10-06）**：
  ① §1 公式改为与仓库实现逐项一致：显式给出 pos_term=sum exp(log-score)、
  neg_term=−r(σ)·log-score_GT、const=r(σ)(log r(σ)−1)、r(σ)=1/(e^σ−1)、dσ 加权
  与 masked-span 均值，并逐项引用 `graph_lib.py:244-269` / `losses.py:31` /
  `vanilla.py span_task_loss` / `noise_lib.LogLinearNoise`，新增 CPU 数值对拍测试
  作为 source of truth 验证；
  ② K_t 改为共享 **K_t ~ Binomial(m_t, q_t)**（q_t=1−e^{−σ_t}，显式 generator 实现），
  禁止 deterministic round(mq)；A 的条件分布即原始 iid Bernoulli 掩码的精确条件
  分布（逐分布等价）；新增独立共享 k_schedule 流并落盘 SHA-256；
  ③ 明确**训练 span 每步随机**（vanilla.py:169 + corruption.py:23-27 引用），共享
  schedule 扩展为完整五元组 (sample_id, span_start, span_len, σ, K) 并单独哈希；
  manifest 仅用于评估（SHA `1897bd14...`）；
  ④ G1 改为三值 PASS / FAIL / INCONCLUSIVE；matched-performance no-overlap →
  INCONCLUSIVE（非 FAIL），仅允许 confirmatory 加密 checkpoint 后重判；sample-level
  bootstrap 不代表 training-seed uncertainty 的口径保留；
  ⑤ 连带修正：co-visibility 矩阵在随机 span 下无良定义 → 删除，P3 仅由可选常数-q
  探针检验；B 的定义改为"π_L 秩最小 K 个（span 内）"，D≡B(identity)、C≡B(reverse)；
  存储计划微调至 ≈23.3GB。
