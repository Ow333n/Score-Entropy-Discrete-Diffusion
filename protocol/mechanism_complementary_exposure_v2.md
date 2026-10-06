# Mechanism-v2 Protocol（DRAFT，待复核）— Complementary Partial-Conditioning Exposure

> 版本：**v2 DRAFT**（2026-10-06；未冻结、未启动任何 v2 训练）
> 上游裁定：v1.2 P1 的 H1（fresh resampling 机制）在 2500 步 gate 上
> **unsupported / inconclusive**；A/B 共同的部分揭示训练都明显低于 pretrained
> （CPI 0.3281 → 0.2637~0.2793），机制问题收紧为：
> **"Does complementary partial-conditioning exposure drive reveal-order
> compatibility attenuation during SFT?"**

---

# 0. 问题与假设

## 0.1 新核心 hypothesis

> Repeated exposure to complementary partial-conditioning states
> （a masked & b visible 或 a visible & b masked），rather than fresh mask
> resampling itself, drives the reduction of reveal-order incompatibility
> during SFT.

对 pair (a,b)，swap defect δ = log p(a|C) + log p(b|C,a) − log p(b|C) − log p(a|C,b)。
真正可能重要的训练事件是模型反复在 **C+a 与 C+b 两类 complementary
partial-reveal 状态**下接受 score-ratio supervision（同一 denoiser、同一 GT、
互补可见性）。v2 直接操纵 **P(exactly one of a,b masked)**，
不再间接操纵 fresh-random 概率。

## 0.2 与 v1.2 的关系

- v1.2 的 A/B 改变 fresh resampling，但 B 的 K 随 σ 变化仍产生 nested
  conditioning states → 两者都暴露于大量 partial contexts；
- v2 直接操纵互补暴露量本身：期望 **E_H > E_U > E_L**，同时
  **K_H = K_U = K_L**，并保持每位置长期 marginal mask rate 一致。

## 0.3 明确禁止的表述（无论结果如何）

- "fresh random 被证明有效/无效"（v1.2 P1 裁定）；
- v2 通过前声称 "complementary exposure 是机制"；
- Opposite 结果出现时重写解释包装成成功（§6 Case-矩阵预注册）。

# 1. 三个条件：U / H / L

**共享**：pretrained init、sample order、target span、σ、K、optimizer、LR、
batch、步数、loss 实现、evaluator、checkpoint schedule、**pairing schedule**。
**唯一操纵变量**：pairwise mask-state correlation（exactly-one-masked 暴露量）。

## 1.1 共同构件：共享随机 pairing（消除位置 confound）

每个 training item/step，用 **pairing_rng** 生成一个 span 内随机 perfect
matching：
`π = randperm(m)` → 配对 `(π[0],π[1]), (π[2],π[3]), …`；
m 为奇数时最后一个位置为 **singleton**（不成对，严格定义见 §1.5）。
**不使用固定相邻配对**（(1,2),(3,4)…），否则引入 locality/position confound。

同一 replicate 内 U/H/L 共享完全相同的
**(sample, span, σ, K, pairing) 六元组 schedule**；U 即使不用 pairing 生成
mask，也必须生成并记录相同 pairing（用于计算其 realized E_comp）。

## 1.2 U — Uniform partial baseline

给定 (m, K)：从 m 个 span 位置均匀选 K 个 mask（= v1.2 的 A，fresh K-subset）。
E_U 的理论 per-pair 值 = 超几何 2K(m−K)/(m(m−1))；每步用共享 pairing
**记录实际** exactly-one-masked 计数。

## 1.3 H — High Complementary（max exactly-one-masked，exact K）

记 n = ⌊m/2⌋、s = m mod 2（singleton 存在性）。构造（贪心、可证最优）：

1. `c = min(K, n)` 个互补对（每对 exactly-one-masked，1 mask/对）；
2. `R = K − c`；若 `s==1 且 R>0`：singleton 置 mask，`R ← R−1`；
3. 剩余 `u = R` 个 mask 把 pairing 顺序中**最后 u 个互补对**升级为
   co-masked（每对 +1 mask、−1 互补计数）；
4. 每个互补对的**方向随机对称**（policy RNG 决定 pair 中谁被 mask，50/50）。

最终：exactly-one-masked 数 = `c − u`；mask 总数恒等于 K
（(c−u)·1 + u·2 + s_mask = K，逐项可验证）。位置偏置：pairing 均匀随机 +
"按 pairing 顺序选择"固定规则 → 每位置长期 marginal = K/m（§4 实测验证）。
**最优性**：小 m 全域 brute-force 单测验证（§4）。

## 1.4 L — Low Complementary / Co-mask（min exactly-one-masked，exact K）

1. `co_pairs = K // 2` 个 co-masked 对（2 masks/对，互补计数 0）；
2. `rem = K % 2`：若 rem==1 且 s==1 → singleton 置 mask（互补计数 0）；
   若 rem==1 且 s==0 → pairing 顺序中第一个未用对置为互补对
   （方向随机，互补计数 1）——仅在 K 为奇、m 为偶时被奇偶性强制。

最终：exactly-one-masked 数 = `rem·(1−s)`（0 或 1）；mask 总数恒等于 K。
**最优性**：小 m 全域 brute-force 单测验证。

## 1.5 odd m / arbitrary K 的严格定义（§0 需求）

- m 奇：最后位置为 singleton，其 mask 状态由 §1.3/§1.4 规则严格决定
  （H：吸收剩余 1 mask 且不减互补计数；L：吸收奇数剩余 1 mask）。
- K ∈ [0, m] 全域：K=0 → 全 visible（互补 0）；K=m → 全 mask（互补 0）；
  中间值按 §1.3/§1.4 公式，单测覆盖所有 (m,K) 小域组合。
- 任意 m,K 的互补计数上/下界由 brute-force 校准后冻结为单测断言。

# 2. RNG Pairing 方案（v2）

| 随机流 | 种子 | 语义 |
|---|---|---|
| data_order | `1000 + r` | 训练分块顺序（replicate 内三条件共享） |
| sigma_schedule | `2000 + r` | σ_t（共享） |
| span_schedule | `3000 + r` | span_len/span_start（共享） |
| k_schedule | `4000 + r` | K_t ~ Binomial(m, q_t)（共享） |
| **pairing_schedule** | **`4500 + r`** | **每 item 的 randperm(m) pairing（共享，v2 新增）** |
| mask_policy_U | `5000 + r` | U 的 K-subset 采样 |
| mask_policy_H | `5500 + r` | H 的互补对方向随机 |
| mask_policy_L | `6000 + r` | L 的强制互补对方向随机 |
| dropout | `torch.cuda.manual_seed(7000 + r)` | dropout=0 断言 |
| evaluator | 固定独立种子 | 永不共享 |

- pairing 流**不得污染** data/span/sigma/k/mask-policy/evaluator 任何流；
- 共享 schedule 升级为**六元组** `(sample_id, span_start, span_len, σ, K, pairing)`，
  每 run 落盘 SHA-256；replicate 内三条件必须逐位一致，不一致硬停；
- pairing 哈希单独记录（G3a 校验项）。

# 3. 干预强度记录（G3a 的原始证据）

每步每 item 记录：`n_pairs`、`n_exactly_one_masked`。
**E_comp = Σ n_exactly_one_masked / Σ n_pairs**（按 replicate 聚合，并另按
q/K bucket 输出——baseline 下理论 per-pair 互补概率 ≈ 2q(1−q)，q≈0.5 最大）。
Preflight（dry-run，CPU）必须验证 **E_H > E_U > E_L 逐 replicate 成立**，
且给出 U/H/L 的 E_comp 理论值与经验值对照。

# 4. Preflight / Unit-test 设计（v2）

**Dry-run（无模型，CPU）**：
1. 六元组 schedule 重放：三条件 hash 相同（replicate 内）、r1≠r2；
2. E_comp：E_H > E_U > E_L 逐 replicate 成立（经验值 + 理论对照）；
3. K 恒等：每步 |M| = K，Σ|M| 三条件相同；
4. **位置边际表**：span-relative 每位置 empirical mask frequency 在三条件
   间近似一致（max deviation < 3·SE 判定），且无 left/right 单调趋势；
5. K sanity（经验 mean(K) vs mean(m·q) z-score，沿用 v1.2 preflight）。

**Unit tests（CPU）**：
1. H/L 对任意 (m, K)（小域 m∈{2..8} 全域）：exact K、互补计数 = brute-force
   最优值、singleton 规则、方向 50/50 对称、无位置偏置（随机 pairing 平均下
   marginal 均匀）；
2. U 回归（K-subset 均匀性，沿用 v1.2 测试）；
3. pairing 生成均匀性（matching 分布）；
4. 流隔离（pairing/mask-policy 不消耗共享流）。

**若 H/L 出现明显位置 marginal bias → 先修 intervention，禁止开始训练（§G3a）。**

# 5. Pilot 配置与执行

- 6 runs = U/H/L × 2 replicates；SEDD-small / 256 / batch 32 / LR 3e-5；
- 训练 2500 步；checkpoint 500 / 1020 / 2500（EMA-only 容器，含 step）；
- step0 共用 pretrained baseline（复用 v1.2 的 mechpilot_shared 评估值）；
- 优先指标：masked-span NLL、token accuracy、CPI_abs、CPI_RMS、signed δ mean、
  realized E_comp；**暂不跑 OrderGap**（CPI gate 有信号后再算）；
- matched-step + matched-performance 双口径（primary NLL ±0.02、secondary
  acc ±0.01；no-overlap → INCONCLUSIVE，不得强行匹配）；
- 硬停：frozen 文件改动 / NaN / hash mismatch / E_comp 次序不成立（G3a）。

# 6. 预注册预测与 Gate

**Primary prediction**：若互补暴露是机制 → CPI_H < CPI_U < CPI_L；
最低要求：**CPI_H < CPI_L 且两 replicate 同方向**。

- **最强支持**：两 replicate 均 H<U<L，且 H−L paired-sample bootstrap CI
  排除 0，matched-performance 后仍成立；
- **Partial support**：H<L 两 replicate 一致、U 位置不稳定 → 只能写
  "complementary exposure 有信号"，不得声称 dose ordering 完整成立；
- **Null**：H≈U≈L → complementary mechanism 不受支持 → 转向 generic
  supervised adaptation 解释；
- **Opposite**：H>L → **直接反驳当前 mechanism prediction，不得重写解释
  包装成成功**。

**G3a — Intervention validity**（先决门）：E_H > E_U > E_L（逐 replicate）
∧ 位置边际 / K / schedule / pairing 哈希 parity 全 PASS。失败 → 实验无效，
禁止用 CPI 结果作任何解释。

**G3b — CPI mechanism signal**（step 2500）：两 replicate 的
CPI_H − CPI_L < 0，且至少 H−L 的 paired bootstrap CI 提供一致支持；
U 的单调位置作 secondary evidence。

**G3c — matched-performance**：task 对齐后 H−L 差异仍保留；
无匹配 checkpoint → INCONCLUSIVE（confirmatory 加密后再判）。

**结论授权**：仅 **G3a ∧ G3b ∧ G3c 全 PASS** 才允许写：
"Results support complementary partial-conditioning exposure as a driver of
reveal-order compatibility attenuation."（2 seeds 仍仅 pilot；
confirmatory 需增加训练 seeds ≥4。）

# 7. 成本估算（v2 pilot）

| 项目 | 估算 |
|---|---|
| 训练 | 6 runs × ~18 min ≈ **1.8h** |
| 评估 | step0 复用 + 18 CPI + 18 task ≈ **1.5–2.5h** |
| 磁盘 | 6 × 2.72GB + step0 共享 1.36GB + 日志 ≈ **17.7GB**（D: 盘） |
| **合计** | **≈ 3.5–4.5 GPU-hours** |

# 8. 暂不执行（v2 gate 通过前）

explicit L_compat、GRPO/RL、fresh-random dose-response、compatibility-aware
scheduler、parallel decoding、更大模型、GSM8K 等扩展任务——一律等待。

# 9. v2 成功后的路线（提前写明，不执行）

Stage 3 Controllability（连续 dose-response）→ Stage 4 Utility（order
robustness / parallelism frontier）→ Stage 5 Boundary（普通文本 vs
reasoning/structured）→ Stage 6 Method（compatibility-aware curriculum /
regularization / decoding scheduler）。

---

# 修订记录

- **v2 DRAFT（2026-10-06）**：基于 v1.2 P1 裁定起草；待用户复核后冻结。
  核心设计决定：① 直接操纵 exactly-one-masked 暴露（非 fresh 概率）；
  ② 共享随机 pairing（无固定相邻对）；③ H/L 构造算法可证最优（brute-force
  单测）；④ 六元组共享 schedule + pairing_rng 独立流；⑤ G3a/b/c 三 gate。
