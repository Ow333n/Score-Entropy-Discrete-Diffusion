# Phase 2 — Parallel Decoding Mechanism Decomposition：Protocol DRAFT v0.1

**状态：DRAFT（待用户 review；只设计，不运行）**
**前序**：Phase 1（Compatibility–Estimation Dynamics）给出方向后才执行 Phase 2
**核心问题**：

> 更低的 CPI/curl 是否真的意味着更好的 decoding-order robustness / parallel decoding？
> CPI（compatibility）并不等于 tokens conditionally independent——即使 CPI≈0，
> parallel block generation 仍可能因 token dependence 失败。

---

## 1. 三类 explanatory factors（预注册定义）

### A. Compatibility / curl
- 定义：|δ|（per-pair swap residual）与 CPI（|δ| 的样本均值）；δ 分布 quantiles。
- 口径：FP32 evaluator（Phase 1.1）同款提取路径。

### B. Conditional estimation error
- 定义：GT conditional NLL / CE（masked-position 的 −log p̂(gold|C) 均值）；entropy、
  confidence（argmax prob）为伴生诊断。

### C. Conditional dependence proxy（不叫 total correlation）

**用户提议的 D_ab 定义检查（已做代数 + 实证验证，2026-10-07）**：

D_ab = 0.5·[log p(b|C,a) − log p(b|C) + log p(a|C,b) − log p(a|C)]

- **对称性**：✓ 对 (a,b) 对称。
- **代数结构（精确算术下）**：定义方向化 pointwise 项
  PMI_a→b = log p̂(b|C,a) − log p̂(b|C)、PMI_b→a = log p̂(a|C,b) − log p̂(a|C)，
  则 **δ = PMI_a→b − PMI_b→a（两项之差），D_ab = 0.5·(PMI_a→b + PMI_b→a)（两项之均值）**。
  → D_ab **不**代数等价于 δ；两者是 (PMI_a→b, PMI_b→a) 的一对正交分量。
- **实证（step0 treated 11302 pairs）**：corr(δ, D_ab) = **0.018**（近正交）；
  corr(PMI_a→b, PMI_b→a) = 0.78；D_ab mean = +0.20、SD = 0.89（显著正偏移 =
  conditioning sharpening：已知真 token 后预测整体变锐）；|δ| vs |D_ab| corr = 0.54。
- **数值稳定性**：✓（logp 差，量级 ~±3）；但 GT-token pointwise 口径受 calibration 偏移
  （正均值 0.2 即为证据）→ **bias 存在，判读时必须与 CE/calibration 一并控制**。
- **提案**：
  - **primary dependence proxy 改用 KL 分布口径**：D_KL_ab = 0.5·[KL(p̂(b|C,a) ‖ p̂(b|C))
    + KL(p̂(a|C,b) ‖ p̂(a|C))]——非负、对称、全分布（非仅 GT token）、**不是 δ 的函数**；
    **零额外 forward**（与 δ 同一 3-forward quartet，logp 全分布在手）。
  - D_ab 保留为 **secondary pointwise dependence proxy**（与 δ 的分解关系本身有诊断价值：
    "dependence 均值 vs 不对称度"）。
  - 命名纪律：一律称 "conditional-dependence proxy"，**禁止称 total correlation**。

---

## 2. Block decoding protocol（冻结规格，DRAFT 待确认）

| 项 | 规格 |
|---|---|
| block size | {1, 2, 4, 8}；block=1 = sequential reference |
| 每 block 位置选择 | 三种模式各跑：confidence-first / random / fixed-order（l2r 块序）；主报告 confidence-first，random×3 seeds 作稳健性 |
| block 内更新 | 独立并行更新（同轮内各 token 条件于同一上下文 C，互不看到对方的揭示） |
| 每轮是否重算 logits | 是（每轮一次 model forward，全部 mask 位置同批打分） |
| 模式 | greedy 为主（deterministic）；sampled 为 secondary（temperature=1，报告用） |
| temperature | sampled 模式固定 1.0 |
| random seed | random 模式的 block 序/采样 seed 冻结（0/1/2） |
| stopping rule | 所有 span 内位置揭示完毕；或 max rounds 到达 |
| max reverse steps | 不设（mask 数有限，block-k 至多 ⌈m/k⌉+1 轮自然终止）；σ 网格沿用 manifest |
| NFE 定义 | NFE = model forward 次数 = 轮数（每轮一次 forward，与 block size 无关） |

---

## 3. NFE / compute matching（重点）

三种比较必须分开报告，禁止混用：
- **A. same decoding rounds**：block-k 与 block-1 各跑 R 轮（R ≤ ⌈m/k⌉），比质量
- **B. same NFE**：block-k 预算 = block-1 的 1/k（block-1 m 轮 vs block-k ⌈m/k⌉ 轮）
- **C. same wall-clock**（可选，仅报告）
- 主产物：**quality-vs-NFE frontier**（每 block size 一条质量-轮数曲线）——避免
  "block-8 只是用了更少 forward 结果更差却被解释成 compatibility failure" 的误读。

---

## 4. Decoding endpoints（预注册）

**Primary（1 个，最稳定/最便宜/最可解释）**：
**GT recovery accuracy**——最终揭示序列的 span 内 per-position 命中率（与 v4.1 task 口径同源，
deterministic greedy 下零方差，成本 = 每样本每轮一次 forward）。

**Secondary**：
1. final NLL / pseudo-likelihood（最终揭示序列在各自条件顺序下的 logp 之和）
2. exact span recovery（全 span 精确匹配率，预期低，仅辅助）
3. **path divergence**：不同 block size / block 序 / 采样 seed 下最终文本的
   两两不一致率（decoding-order robustness 的直接度量，与 OrderGap 对应）
4. sequential-vs-block disagreement（block-k 与 block-1 在相同位置预测不同 token 的比例，
   按轮次追踪）
5. task quality（greedy 命中，见 primary）

---

## 5. 增量预测价值（Phase 2 的真正问题）

不是"CPI 与 decoding failure 有没有 correlation"，而是：已知 entropy / confidence /
token distance / estimation error / dependence proxy 之后，**CPI 是否还提供额外解释力**。

预注册嵌套模型（sample-level，500 样本）：
- M_base：Failure ~ CE + Dependence(KL) 
- M_full：Failure ~ CE + Dependence(KL) + CPI(|δ|)
- M_extra：Failure ~ CE + Dependence(KL) + CPI + entropy + confidence + token_distance
  （≤6 个 predictor，防小样本过拟合）

比较：**ΔR² + 5-fold CV prediction error**（CV 必须；不作 in-sample 显著性主结论）。
Failure 定义：per-position GT recovery miss（primary endpoint 的 sample 级版本）。

---

## 6. 循环解释禁令

所有指标定义（dependence proxy、entropy、confidence、CPI、failure）**必须在 decoding
stress test 正式运行之前冻结**；禁止先看哪个 proxy 与 failure 最相关再把它定义成
"真正 dependence"。本 DRAFT FROZEN 后定义即锁定。

---

## 7. Phase 3 注意事项（现在只记录，不设计不执行）

Full-mask vs Partial-reveal 仅保留为候选 causal control（Phase 1/2 有清晰方向后才设计）。
未来必须处理（预注册提醒）：Full/Partial 天然改变 mask ratio / effective supervision /
difficulty / conditional entropy；至少两个 matching axis：
**A. compute / optimizer-step matched；B. effective supervised-token budget matched**；
再加 matched-performance；并探索 CPI ~ condition + NLL，看 controlling for performance 后
conditioning regime 是否还有 residual effect。做不到这些则不能宣称 partial conditioning
是必要原因。

---

## 8. 执行限制

- 现在：只设计，不跑大规模 decoding、不训练、不实现 Phase 3。
- 前置：Phase 1 的 FP32 evaluator（review 通过后）+ Phase 1 结论。
- 全部指标定义冻结 → 用户 review → 才允许 decoding stress test 批量运行。
