# Phase 2 — Parallel Decoding Mechanism Decomposition：Protocol

**状态：FINAL v0.2 — FROZEN（2026-10-07；用户批准，冻结后禁止修改指标定义 / 协议 / 判定规则）**
**修订记录**：v0.1（初版）→ v0.2（用户修订 7–14：JS dependence proxy、D_ab 降级、NFE 两栏、
primary endpoints 固定两个、三报告、嵌套模型）→ **v0.2 FROZEN（最终 3 小修）**：
① sample-level 措辞纪律（见 Phase 1 协议 §5）② B-vs-C engineering tolerances 标记
（见 Phase 1 协议 §3）③ **grouped 5-fold CV**（group key = 底层 evaluation sample /
masked example ID；同一样本派生的 tokens / block sizes / paths / scheduler observations
全部同 fold，禁止跨 fold 泄漏）+ 分 endpoint 评价指标（token recovery：held-out R²/MSE；
exact-span：held-out log-loss/Brier，AUROC 仅 secondary；**binary exact-span 禁用普通 R²
作主指标**）。
**前序**：Phase 1 给出方向后才执行
**核心问题**：更低的 CPI/curl 是否真的意味着更好的 decoding-order robustness /
parallel decoding？CPI（compatibility）≠ tokens conditionally independent——即使 CPI≈0，
parallel block generation 仍可能因 token dependence 失败。

---

## 1. 三类 explanatory factors

### A. Compatibility / curl
|δ|、CPI（|δ| 样本均值）、δ 分布 quantiles——FP32 evaluator 口径（Phase 1.1 判定后）。

### B. Conditional estimation error
GT conditional NLL / CE；entropy、confidence 为伴生诊断。

### C. Conditional dependence proxy（修订 #7/#8/#9）

**C.1 数学基础：SEDD score → normalized conditional distribution 的严格映射**

给定状态 x_t（时间 σ），模型输出 score(x_t, σ) ∈ R^{L×D}（D=50258，absorbing：
50257 干净词表 + 1 MASK）。对 mask 位置 i，模型对干净词表的条件预测分布定义为
（与 frozen evaluator 的 `clean_log_probs` 完全同源）：

```
p̂(v | C) = softmax(score[i, 0:D−1])_v ， v ∈ {0, …, D−2}
```

这是一个**严格归一化的分布**（log_softmax 于干净词表），映射无近似、无自由参数。
下文所有 distribution-level proxy 均建立在该映射上。

**C.2 primary dependence proxy：JS 分布口径（最终提案）**

对 treated pair (i, j)，GT tokens (a, b)，冻结 corruption 状态 C（两端均 mask）：

- reveal a 前后，位置 j 的预测分布：
  Q_j = p̂(· | C)|_j ；P_j^a = p̂(· | C, a)|_j
- 对称地：Q_i = p̂(· | C)|_i ；P_i^b = p̂(· | C, b)|_i
- JSD(P ‖ Q) = ½·KL(P ‖ M) + ½·KL(Q ‖ M)，M = ½(P + Q)（自然对数，值域 [0, ln 2]）

**JS_dep(i, j) = ½ · [ JSD(P_j^a ‖ Q_j) + JSD(P_i^b ‖ Q_i) ]**

性质（单测逐条验证）：
1. **对称**：JS_dep(i,j) = JS_dep(j,i)（定义即两方向平均）
2. 非负、上界 ln 2
3. JS_dep = 0 ⟺ 两个 reveal 均不改变对方位置的预测分布
4. **JS_dep = 0 ⟹ δ = 0**（分布不变 ⇒ GT logp 差为 0 ⇒ 两方向 PMI 为 0 ⇒ δ = 0）；
   **δ = 0 ⇏ JS_dep = 0**（GT logp 不变但分布其他部分移动）→ JS proxy 严格强于 δ
5. 与 δ 无代数函数关系（δ 只含 4 个 GT logp；JS 含完整分布）
6. **数值稳定性**：由 logp 向量计算——KL(P‖Q) = Σ_v exp(lp_P[v])·(lp_P[v] − lp_Q[v])，
   fp64 聚合；log 域运算避免 P(v)→0 的下溢问题
7. **成本**：零额外 forward（与 δ 同一 3-forward quartet，完整 logp 已在手）
8. **命名纪律**："JS dependence proxy"；**明确不是 exact total correlation**（TC 需
   联合分布全分解；JS 只测单 token reveal 对另一位置预测分布的位移）——报告与代码中
   禁止出现 TC 措辞

**C.3 secondary：GT-conditioned D_ab（修订 #7，明确降级）**

v0.1 的 D_ab = 0.5[(p̂(b|C,a) − p̂(b|C)) + (p̂(a|C,b) − p̂(a|C))] 保留为
**secondary GT-conditioned diagnostic**：其与 δ 的分解关系（δ = 两方向 PMI 之差、
D_ab = 两者之均值、实证 corr(δ, D_ab)≈0.02、mean +0.20 sharpening 偏移）本身有诊断
价值（"dependence 均值 vs 不对称度"），但不作 primary dependence，原因：GT-token
pointwise 口径受 calibration/sharpening 偏移污染。

**C.4 冻结要求（修订 #9）**：三个 proxy（|δ|、D_ab、JS_dep）定义在正式 decoding
stress test **之前冻结**；非循环定义（定义只依赖模型输出，不依赖 decoding 结果）；
JS_dep 与 D_ab 均有数值稳定性单测（对称性、恒等零、界、toy 解析值、逐位确定性）。

---

## 2. Block decoding protocol（冻结规格）

| 项 | 规格 |
|---|---|
| block size | {1, 2, 4, 8}；block=1 = sequential reference |
| 位置选择 | 三种模式各跑：confidence-first / random（seed 0/1/2）/ fixed-order（l2r 块序）；主报告 confidence-first |
| block 内更新 | 独立并行更新（同轮内各 token 条件于同一上下文 C，互不见对方的揭示） |
| 每轮重算 logits | 是（每轮一次 model forward，全部 mask 位置同批打分） |
| 模式 | greedy 为主（deterministic）；sampled 为 secondary（temperature=1.0，报告用） |
| stopping rule | span 内全部揭示；或 max rounds = ⌈m/block⌉ + 1 自然上限 |
| NFE | 见 §3 |

---

## 3. NFE / compute accounting（修订 #10，严格定义）

**NFE = decoder 实际消耗的 model forward 总次数。**
- 每轮 1 次 forward 产生全部 mask 位置打分；confidence-first 的 block 位置选择使用
  **同一轮的分数**（per-position argmax-prob 排序）→ **选择不产生额外 forward**
- 任何变体若需要独立的选择 forward（如 lookahead 重打分、tie-break 重算），
  **逐次计入 NFE**
- 报告格式强制：`NFE_total = NFE_predict + NFE_select`（两栏分别列，禁止隐藏选择成本）
- greedy 主口径下 NFE_predict = 轮数、NFE_select = 0（预注册断言，运行时校验）

**三种比较分别报告（修订 #12，禁混用）**：
1. **fixed-NFE quality**：同 NFE 预算下各 block size 的质量
2. **fixed-quality NFE**：达到同一质量（如 90% 的 block-1 最终命中率）所需 NFE
3. **quality–NFE frontier**：每 block size 一条质量-轮数曲线（主产物）
避免"block-8 只是用了更少 forward 结果更差却被解释成 compatibility failure"的误读。

---

## 4. Decoding endpoints（修订 #11，primary 固定两个）

**Primary（两个，均固定）**：
1. **token recovery accuracy**——最终揭示序列的 span 内 per-position GT 命中率
   （greedy 确定性、零方差、最便宜最可解释；v4.1 task 口径同源）
2. **exact-span recovery**——span 全序列精确匹配率（预期低，作为严格 endpoint）

**Secondary**：
1. path divergence——不同 block size / block 序 / 采样 seed 下最终文本的两两不一致率
   （decoding-order robustness 直接度量）
2. sequential-vs-block disagreement——block-k 与 block-1 在相同位置预测不同 token 的
   比例（按轮次追踪）
3. pseudo-NLL / final NLL——最终揭示序列在各条件顺序下的 logp 之和

---

## 5. 增量预测价值（修订 #13 + FROZEN 修订 ③，预注册嵌套模型）

不是问"CPI 与 decoding failure 有没有 correlation"，而是已知 CE / Dependence 之后
CPI 是否仍有额外解释力。Sample-level（500 样本），Failure = per-position GT recovery
miss（primary 1 的样本级版本）：

- **Base**：Failure ~ CE + Dependence(JS)
- **Extended**：Failure ~ CE + Dependence(JS) + CPI(|δ|)

**CV 设计（FROZEN 修订 ③：grouped 5-fold CV）**：
- **group key = 底层 evaluation sample / masked example ID**——同一原始 sample 派生出的
  全部 tokens、block sizes、paths、scheduler observations 及一切相关 measurement
  **必须留在同一个 fold**；禁止任何跨 train/test fold 泄漏
- 分 endpoint 评价（primary 分开）：
  - **A. token recovery accuracy（回归型）**：held-out **R²** + held-out **MSE**；
    增量 = **ΔR² / ΔMSE**
  - **B. exact-span recovery（二元型）**：held-out **log-loss** + held-out **Brier score**；
    增量 = **Δlog-loss / ΔBrier**；AUROC 仅 secondary
  - **禁止对 binary exact-span 使用普通 R² 作主指标**
- 不作 in-sample 显著性主结论
- **额外 baseline**：entropy、confidence、token_distance 单独加入 Extended（每次加一，
  控制 predictor 总数 ≤ 6，防小样本过拟合）

---

## 6. 循环解释禁令（保留）

所有指标定义（JS_dep、D_ab、entropy、confidence、CPI、Failure、decode 协议）必须在
decoding stress test 正式运行**之前**冻结；禁止先看哪个 proxy 与 failure 最相关再把它
定义成"真正 dependence"。本 DRAFT FROZEN 后定义锁定。

---

## 7. Phase 3：DEFER（修订 #14）

不训练 Full-mask、不恢复 dense checkpoints、不增加 seeds。Full-mask vs Partial-reveal
仅保留为候选 causal control；两个 matching axis（compute/optimizer-step matched、
effective supervised-token budget matched）+ matched-performance + CPI ~ condition + NLL
residual 分析，仅作未来记录。

---

## 8. 执行限制

现在：只设计。不跑大规模 decoding、不训练、不实现 Phase 3、不做 FP32 bulk 评估。
前置：Phase 1 的 precision-ladder 判定 + FP32 evaluator review gate + Phase 1 结论。
全部定义冻结 → 用户 review → 才允许 decoding stress test 批量运行。
