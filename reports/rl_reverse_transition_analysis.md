# SEDD Reverse-Policy / Transition-Probability Analysis（RL 前置技术审计）

> 目的：回答"policy π_θ(a_t | s_t) 在 SEDD 里究竟是什么"——从 model 输出一路追踪到采样出的 next state，
> 并明确哪些部分可微、RL 的 log π 该如何定义与重算。所有引用以 repo 快照 `pre-rl-compatibility-v1` 为准。

## 1. Model 输出：log-score（不是 logits）

`SEDD.forward(indices, sigma)`（`model/transformer.py`）输出形状 `[B, L, D]`，D = vocab + 1 = **50258**（absorbing，D−1 = MASK）。

- 尾部 `scale_by_sigma` 分支减 `log(e^σ−1) − log(D−1)`（吸收态的 score 时间标量 centering，数学上是 r(t) = 1/(e^σ−1) 的对数项）。
- **最后一行**：`x = torch.scatter(x, -1, indices[..., None], zeros)` —— 把每个位置当前 token（包括 MASK）的 logit 置 0。所以 `score[..., x_t] = 0` 恒成立，MASK 列的 0 不是有效 logit。
- 训练路径（`model/utils.py::get_score_fn(train=True)`）返回该 **log-score**；`graph.score_entropy` 内部对干净词表取 `.exp()` 后计算交叉熵型 loss。
- 采样路径（`sampling=True`）返回 `score.exp()` —— **真 score**（用于 reverse rate 构造）。

**理论最优解**（absorbing + score-entropy 最优，RADD 桥）：在 MASK 位置，
```
exp(score_v) = p̂_0(v | C) / p̂_0(MASK | C) = p̂_0(v | C) · r(t),   r(t) = 1/(e^{σ} − 1)
```
评估口径 `compatibility/posterior.py::clean_log_probs` 正是按此提取条件概率：
```
log p̂(v|C) = log_softmax(score[..., :D−1])        # 只对干净词表 D−1 个条目 softmax，排除 MASK 列
```
**这是 CPI/OrderGap 度量的概率口径，也是 RL policy log-prob 的首选口径（见 §5）。**

## 2. 状态 / 转移结构：Absorbing CTMC

`graph_lib.py::Absorbing`，D = 50258，token 50257 = MASK（吸收态）。

- forward rate（`rate`）：非 MASK 位置 i → 只有 i→MASK 的 rate 1。
- `transp_rate(i)`（第 i 行）：**非 MASK 位置**：行 = −one_hot(i)（只有对角线 −1）；**MASK 位置**：行 = 全 1、MASK 列 0（从 MASK 到每个干净 token 的 rate 1）。
- exact forward transition（`transp_transition(i, σ)`）：非 MASK i 行 = e^{−σ}·one_hot(i)（保持）；MASK 行 = 非 MASK 列各 (1−e^{−σ})、MASK 列 e^{−σ}。

## 3. Reverse rate 的构造（`Graph.reverse_rate`）

```python
normalized_rate = transp_rate(i) * score          # score 是 exp 后的真 score（采样路径）
normalized_rate.scatter_(-1, i, 0)                # 对角线置 0
normalized_rate.scatter_(-1, i, -row_sum)         # 对角线 = −行和（保证行和 0）
```

代入 MASK 位置（x_t = MASK，score[MASK]=0 已由 scatter 置 0）：
```
R̂(MASK → v) = 1 · exp(score_v) = p̂_0(v|C) · r(t)      （v ∈ 干净词表）
R̂(MASK → MASK) = −Σ_v R̂(MASK → v)
```
非 MASK 位置：全 0（已揭晓位置不再变化，吸收语义）。这与 absorbing 反向 CTMC 的 rate 公式
`R̂_v(x_t) = Q^f_v(x_t)·exp(score_v − score_{x_t})` 一致（x_t=MASK 时 Q^f=1、score 差为 score_v−0）。

## 4. 两个采样器：一步转移分布（RL action 分布的唯一来源）

`sampling.py::get_pc_sampler`（predictor–corrector 框架，实际实现只有 predictor 循环 + 末尾 denoiser，**无 corrector 步**）。
t 从 1 → ε（默认 `configs/config.yaml`: predictor=euler, steps=128, noise_removal=True；`run_sample.py` 用 analytic/1024）。

### 4.1 EulerPredictor（τ-leaping，默认）

```python
sigma, dsigma = noise(t)                                  # σ(t) = −log(1−(1−ε)t), dσ = g(t)·dt
score = score_fn(x, sigma)                                # exp 后的真 score
rev_rate = step_size · dsigma[...,None] · graph.reverse_rate(x, score)
x = graph.sample_rate(x, rev_rate)                        # categorical(one_hot(x) + rev_rate)
```

每位置一步转移概率（对 dt·dσ·R̂ 的**一阶 τ 近似**，未 clamp）：
```
π(v | x_t)  = dt · dσ · exp(score_v)          （v ≠ x_t, 仅 MASK 位置有非零项）
π(keep)     = 1 − dt · dσ · Σ_v exp(score_v)
```
- **可微**：exp(score_v) 是 model 输出的光滑函数，反向可穿过。
- **近似性质**：dt·dσ·exp(score_v) 可能 > 1、keep 概率可能 < 0（σ 大/score 大时）。`sample_categorical` 的 gumbel-argmax 在负权重下仍会运行（隐式非正规化权重采样），但 log π 没有正规化语义。**这是 Euler 采样器的固有近似，不是 bug。**

### 4.2 AnalyticPredictor（staggered-score 一步转移）

```python
stag_score = graph.staggered_score(score, dsigma)         # p_{σ−dσ}(z)/p_σ(x) 的近似
probs = stag_score * graph.transp_transition(x, dsigma)   # × exact forward transition
x = sample_categorical(probs)
```

absorbing 的 `staggered_score`：
```
s'_v = e^{dσ}·score_v                       （干净词表）
s'_MASK = e^{dσ}·score_MASK + (1−e^{dσ})·Σ_v score_v
```
MASK 位置一步转移权重（乘上 exact forward transition 行）：
```
w(v)  = s'_v · (1 − e^{−dσ})                （跳到干净 token v）
w(MASK) = s'_MASK · e^{−dσ}
```
- 相对 Euler 优点：无 rate·dt 一阶截断（dt 再大也不会出现负"概率"）；仍含 staggered-score 的 p 比值近似。
- 末尾 `Denoiser` 用同一公式（σ=σ_ε）并截断 MASK 列后 categorical，保证终态无 MASK。

## 5. RL policy 定义（预注册候选，供协议 §14 定稿）

RL 的"action"= 一个 reverse step 内每个位置并行做出的局部决定（保持 / 跳到哪个 token）。
SEDD 是 per-position 独立的 categorical 结构（给定 x_t，各位置转移条件独立）。

**候选口径 A（推荐）**：以 AnalyticPredictor 的分布为 rollout 采样器，policy 取正规化 categorical：

```
logits_v = log( stag_score_v · transp_transition(x, v) )      # v ∈ 全部 D 个状态
π_θ(· | x_t) = softmax(logits)                                # 正规化后采样 + log π
```
- 与 CPI 度量（softmax 条件概率）同族：在 dt→0 极限与 clean_log_probs 口径一致。
- 每步 log π 有界、正规化 → importance ratio 有定义、可 clip。
- 采样用 `sample_categorical(π)`（gumbel argmax，无需另写采样器）。

**候选口径 B**：rollout 用默认 Euler/128（与 config 完全一致），log π 按 τ-leaping 公式重算（不正规化，允许负 keep 概率时记录诊断量）。优点是与"默认生成管线"零差异；缺点是 IS 口径带近似、正规化性无保证。

**协议定稿原则**：rollout 采样器与 log π 公式必须**同源**（同一 predictor 的同一公式），old/new logπ 用同一代码路径重算；不一致即触发 abort（协议 §33/§44）。推荐 pilot 用口径 A + analytic predictor（每 rollout steps 数与 DRAFT 一起冻结）。

## 6. 梯度路径与重算可行性

- log π_θ 是 score 的 log_softmax 组合，score = model(x_t, σ)，σ 与 x_t 在训练步是**给定常数**（detached rollout 数据）→ 重算 forward 时对 θ 的梯度路径干净、无采样器内的不可微操作（gumbel 只用于 rollout 采样，不进图）。
- rollout 全程 `no_grad` + eval 模式（`get_score_fn(train=False, sampling=True)`），只落盘 `(x_t, σ, action, old_logπ)` 与 reward 所需字段——不保留任何 autograd 图（协议 §19）。
- 训练步：取 trajectory 中采样的 transition，(re)forward 单状态 → 重算 log π → PG loss → backward。单状态峰值显存 ≈ SFT 训练量级（gradient checkpointing 已开，见 `training/swap.py` 的逐状态 backward 先例）。

## 7. 关键结论（RL 代码动工前的判据）

1. model 输出 log-score，policy 的一切概率从 `exp(score)` 构造 —— 禁止写 `log_softmax(model(x))` 当"policy logits"。
2. per-position categorical 转移 + 吸收语义 ⇒ action 空间是"每个 MASK 位置 × (D−1 个 token + 保持)"的并行局部决定；总 log π = 各位置求和。
3. rollout 采样器与 logπ 公式必须同源、可重算、可微 —— 这是 RL-G0 之后第一个 correctness gate（协议 §33）。
4. Euler/τ-leaping 的近似性、staggered-score 的近似性都要在 rollout 元数据中记录诊断量（负"概率"计数、权重和偏离 1 的幅度），进协议 §44 的 abort 阈值。
