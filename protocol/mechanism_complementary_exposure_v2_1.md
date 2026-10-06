# Mechanism Pilot v2.1 — Complementary Exposure Protocol

**状态：FINAL v1.1 — FROZEN（冻结后禁止修改 hypothesis / gate / Case 定义 / 措辞纪律）**
**修订记录（v0.1 DRAFT → v1.0 FINAL，用户裁定）**：
1. odd-m singleton 硬币明确 `s ~ Bernoulli(K/m)`；补 K=0 / K=m 边界与 marginal 推导（§5/§6）
2. G3a 的 strict E_H>E_U>E_L 仅作用于 non-degenerate K∈[2,m−2]；K∈{0,1,m−1,m} 单独报告、不得构成 FAIL（§11 G3a-5、§7.3）
3. treated/heldout CPI evaluator 训练前冻结；训练 RNG 完全隔离；H/U/L 同 replicate 共享 sample/span/context；heldout edges 与 training treated edges 不重叠（§9、§14）
**修订记录（v1.0 FINAL → v1.1 FINAL — FROZEN，专家 review 吸收）**：
4. **1A 措辞纪律**：P1/P2 结论统一表述（§1.1），禁止"已被证明无效"类措辞
5. **1B mask-structure confound diagnostics**：run-length / adjacency / transition / pair-distance 分桶等 7 项只读诊断（§7.6、G3a），区分 intended 与 confound，不新增 gate 要求
6. **1C calibration / scale diagnostics**：score scale、masked-token calibration、temperature-scaled CPI 的严谨定义与限制（§9.4，secondary，不入 G3 primary；退化为 δ/T 时标记 deferred）
7. **1D 小型 global path metric**：step2500 专属 frozen 子集 OrderGap + path-score variance（§9.5，secondary，不入 G3c primary）
8. **1E U baseline 分析预注册**：primary contrast H vs L；secondary H vs U、U vs L；dose-like H<U<L 非 PASS 必需，U 不居中按预注册规则解释（§11 G3c）
9. **1F 训练期机制诊断 logging**：loss 轨迹、gradient norm(+rolling variance 近似)、per-checkpoint task NLL/acc（§10，零/低成本，不做 per-example gradient）
10. **1G 两阶段设计**：pilot（6 runs）→ Confirmatory（≥4 total paired replicates + power analysis）的进入条件预注册（§15.1）
11. **1H dose-response follow-up**：λ-interpolation（H/U 混合 policy）仅在 Future Confirmatory Stage 定义，本 pilot 不执行（§15.2）
12. **1I 明确 defer 的大型扩展**：full-mask SFT / AR SFT / 多规模 / 跨域 / 多 K 独立训练 / 5 seeds 首跑——全部归入"if mechanism survives pilot" future work（§15.3）
**前身**：`protocol/mechanism_complementary_exposure_v2.md`（DRAFT，已废弃——pairing 每步 fresh random 会被边缘化回 uniform，identifiability 失效）
**v1.2 结论依据**：`reports/mechanism_pilot_v1_2_report.md`（FROZEN，tag `mechanism-pilot-v1.2-frozen`）

---

## 1. 背景与研究问题

v1.2 pilot（G1=INC, G2=INC）之后，研究问题收窄为：

> **Does complementary partial-conditioning exposure itself drive the reduction of reveal-order incompatibility during SFT?**

动机：partial-reveal SFT 中，当一个 pair 的两个位置恰好只有其一被 mask 时，模型必须"用已揭示的一端条件化预测被 mask 的一端"——这正是 delta-swap CPI 度量的条件打分结构。若该暴露本身是隐式 compatibility regularizer，则训练中 exactly-one-masked pair 频率越高的 policy，SFT 后 global CPI 衰减应越强。

v2.1 直接操纵该频率（而不是再间接操纵 fresh-random 程度）：

- **U** = Uniform Partial baseline（size-K 均匀子集）
- **H** = High Complementary Exposure（exactly-one-masked pair 数最大化）
- **L** = Low Complementary / Co-mask Exposure（该数最小化）

核心变量：

> 对训练中定义的一组 treated pair (a,b)：**P(exactly one of a,b is masked)**，H 最大 / U 自然 / L 最小；其余（K、样本、span、σ、单位置 marginal）尽量一致。

## 1.1 措辞纪律（1A，FROZEN）

对 v1.2 的 P1（Fresh vs Fixed Random）与 P2（L2R vs R2L directional）：

- **禁止**："Fresh random 已被证明无效"、"Directional exposure 已被证明无效"、"已被证明不影响 order" 及任何等价措辞。
- **统一表述**："no stable supporting evidence under the current pilot" / "INCONCLUSIVE under the current pilot setting"。
- 语义：pilot 只是**没有检测到稳定证据**，不是证明 null；两者在统计与科学含义上不可互换。

## 2. 预注册假设（FROZEN）

- **H_c（complementary-exposure mechanism）**：E_comp 更高的 policy 在 SFT 后产生更强的 global reveal-order compatibility attenuation：CPI_global(H) < CPI_global(L)，U 介于两者（dose-like，非强制单调）。
- **H_local**：intervention 首先作用于被直接操纵的 treated pairs：CPI_treated(H) < CPI_treated(L)。
- **H_gen**：若机制是全局的，heldout pairs（训练从未使用）应同方向衰减：CPI_heldout(H) < CPI_heldout(L)。
- **备择**：H≈U≈L → complementary exposure 不成立，generic supervised adaptation 成为更强解释；task performance 不可匹配 → 机制判定 INCONCLUSIVE。

## 3. 核心设计原则

1. **Persistent / frozen pair structure**：pair 结构不是模型输入；若每步从所有 matching 对称随机采样，边缘化后 P_H(M)=P_U(M)=P_L(M)，干预为零。因此 pair map 必须 frozen（每 replicate × span length 生成一次、落盘、sha256、训练加载不重生成）。
2. **单位置 marginal 不变**：P(i masked | m,K) = K/m 对所有 policy——干预只改变 pairwise correlation，不改变 marginal（§5 证明）。
3. **共享 schedule**：U/H/L 同 replicate 共享 sample/span/σ/K 四元流与 odd-m phase 流；policy 专属 RNG 流独立种子、互不影响。
4. **identifiability 必须先证明**：exact TV > 0 + treated-pair covariance 排序，未通过则 protocol 不成立（§7）。

## 4. Even m = 2n：精确算法

**Pair map**：每 replicate r × span length m（even），frozen random perfect matching `P_m^r = {(i_1,j_1),…,(i_n,j_n)}`（random permutation 连续配对，落盘+sha256）。

给定 exact K：

### H（max complementary exposure）
- `b_max = min(K, m−K)`（discordant pair 数；m even ⇒ b_max ≡ K mod 2 ✓）
- `a = (K−b_max)/2`（both-masked pair 数）
- 采样：① 从 n 个 pair 均匀选 b_max 个 discordant；② 每个 discordant pair 独立 fair coin 决定 mask 左端/右端；③ 从剩余 n−b_max 个 pair 均匀选 a 个 both-mask；④ 其余 both-visible。
- exact K：b_max·1 + a·2 = K ✓。最优性：任意 mask set 满足 b ≤ min(K, m−K) 且 b ≡ K (mod 2)，故 b_max 是可达上界 → H 逐 item 最大化 exactly-one-masked 数。

### L（min complementary exposure）
- `b_min = K mod 2`
- K even：b=0，随机选 K/2 个 pair both-mask；
- K odd：b=1，随机选 1 个 discordant（orientation fair coin），再选 (K−1)/2 个 pair both-mask；
- 其余 both-visible。exact K ✓。

### U（uniform baseline）
- 从 m 个位置均匀抽 size-K 子集（不使用 pair map 决定 mask，但同一 frozen map 仍生成/记录，供 treated-pair 测量与 CPI_treated）。

### K 边界（even m，H/L/U 同规则）
- `K=0`：b_max=0、a=0 → 空 mask 集（唯一可能）；`K=m`：b_max=0、a=m/2 → 全集（唯一可能）。三 policy 分布相同（单点分布）。
- `K=1`：H 与 L 均为"1 个 discordant pair + orientation coin" ⇒ P(i masked)=1/m 且三者分布逐位相同；`K=m−1` 对称（等价 K=1 的可见集）。⇒ **degenerate K 集合 = {0, 1, m−1, m}**（H=U=L 分布完全相同），G3a 对此单独报告（§11）。

## 5. 单位置 marginal 不变性证明

**Even m**：设 pair 标签上的均匀置换群 + 每 pair 独立 orientation flip 构成自同构群 G，作用于 m 个位置**传递**（任意 i 可到任意 j：pair 移过去 + 按需 flip）。H/L 采样过程对 G 不变（pair 选择均匀、orientation fair coin）⇒ 分布 G-不变 ⇒ 所有位置 marginal 相等；又 Σ_i P(i masked) = K（exact K）⇒ **P_H(i)=P_L(i)=K/m**。U 平凡 K/m。∎

**Odd m（§6 bank）**：完整推导（按用户修订 #1 展开）：
1. **每 item exact K**：singleton 硬币 s ∈ {0,1} 取值后，pairs 内需 K−s 个 mask；0 ≤ K−s ≤ m−1 恒可被 H/L 的 b/a 分解表示（§6 可行性）⇒ 总 mask 数 = s + (K−s) = K 恒成立。
2. **轮换对称**：phase t 下 pair 内位置对称（pair 选择均匀 + orientation fair coin）⇒ 条件边际相等；且每位置在 m 个 phase 中恰 1 次 singleton、m−1 次 paired ⇒ 对位置求期望时全体位置地位相同 ⇒ 边际 P(i masked) 与 i 无关。
3. **取值**：设 singleton 被 mask 概率 q（硬币流参数，U/H/L 共享），paired 位置边际 α。对任意位置 i：`P(i masked) = (m−1)/m · α + (1/m) · q`；同时期望总 mask = Σ_i P(i masked) = K ⇒ `(m−1)α + q = K`。
4. **取 q = K/m**（`s ~ Bernoulli(K/m)`，U/H/L 共享流；U 消耗但不用）⇒ `α = (K − K/m)/(m−1) = K/m` ⇒ **全体位置 marginal 精确 K/m**。∎
5. **K 边界**：K=0 ⇒ s≡0（Bernoulli(0)），空集唯一；K=m ⇒ s≡1（Bernoulli(1)），pairs 全 both-mask + singleton mask ⇒ 全集唯一。三 policy 单点分布相同（degenerate，§4/§7.3）。

**L/R 趋势**：discordant pair 的 orientation 为 fair coin（H/L）⇒ 任意 treated pair 左端/右端 mask 率差 = 0；U 均匀 ⇒ 0。exact（小 case 枚举验证，§7）。

## 6. Odd m = 2n+1：balanced persistent matching bank

不能有固定 singleton 相对位置。设计（DRAFT）：

- **Bank 构造**：每 replicate × odd m，frozen random base permutation π_r；phase t = 0..m−1 的 matching：`singleton = π_r(t)`，pairs = `(π_r(t+1), π_r(t+2)), (π_r(t+3), π_r(t+4)), …, (π_r(t+m−2), π_r(t+m−1))`（下标 mod m）。n = (m−1)/2 对。
  - 性质：每个位置在 m 个 phase 中**恰好一次** singleton；bank 的 pair 并集 = π_r 的 cycle 边 `{(π_r(i), π_r(i+1)) mod m}`（m 条边，每条边恰在 2 个 phase active）。
- **Phase schedule（U/H/L 完全共享，确定性）**：对 span 内第 s 个 item，`phase = (o_span + s) mod m`，`o_span` 为该 span 的 frozen offset（由 span 流确定，非 RNG、非每步 fresh）。长 span 下每 phase 频率 → 1/m ⇒ 每位置 singleton 长期频率相等 ✓。
- **Singleton 硬币（共享流，修订 #1）**：每 item 消耗一次 **`s ~ Bernoulli(K/m)`**（显式：P(s=1)=K/m、P(s=0)=1−K/m；seed 8500+r，U/H/L 共享同一硬币流；U 消耗但不用该值）。边界：K=0 ⇒ s≡0；K=m ⇒ s≡1。marginal 推导见 §5。
- **给定 (phase t, s) 的 pair 内 mask**（exact K 保持：pairs 内需 K−s 个 mask，0 ≤ K−s ≤ m−1 恒可行）：
  - H：`b = min(K−s, m−1−(K−s))`，`a = (K−s−b)/2`；选 b 个 discordant（fair coin orientation）+ a 个 both-mask。
  - L：`b = (K−s) mod 2`，`a = (K−s−b)/2`；同上。
  - U：均匀 size-K 子集（与 phase/s 无关）。
- **Treated pair 集合（odd m）**：bank 的 cycle 边（m 条）；CPI_treated / covariance 用它们（每条边在 2 个 phase active）。
- **不退化检查**：bank 为确定性有限结构（frozen permutation 的旋转），不是每步从所有 matching 对称采样 ⇒ 不会重新边缘化回 uniform（§7 的 m=5 exact TV > 0 验证）。

## 7. Identifiability preflight（HARD，DRAFT 已执行设计期验证）

**脚本**：`scripts/v21_design_validation.py`（exact Fraction 枚举 + 闭式 + MC；只读，无训练）。

### 7.1 Small exact cases

**m=4, K=2，frozen matching {(0,1),(2,3)}**（6 个 size-2 子集）：

| M | P_U | P_H | P_L |
|---|---|---|---|
| {0,1} | 1/6 | 0 | 1/2 |
| {0,2} | 1/6 | 1/4 | 0 |
| {0,3} | 1/6 | 1/4 | 0 |
| {1,2} | 1/6 | 1/4 | 0 |
| {1,3} | 1/6 | 1/4 | 0 |
| {2,3} | 1/6 | 0 | 1/2 |

TV(H,U)=1/3，TV(L,U)=2/3，TV(H,L)=1（support 不相交）；marginals 全 1/2 ✓；E_comp：H=1 > U=2/3 > L=0 ✓；C̄_treated：H=−1/4 < U=−1/12 < L=1/4 ✓；L/R=0 ✓。

**m=6, K=2/3/4，matching {(0,1),(2,3),(4,5)}**：

| case | TV(H,U) | TV(L,U) | TV(H,L) | E_comp H/U/L | C̄_treated H/U/L |
|---|---|---|---|---|---|
| K=2 | 1/5 | 4/5 | 1 | 2/3 > 8/15 > 0 | −1/9 < −2/45 < 2/9 |
| K=3 | 3/5 | 2/5 | 1 | 1 > 3/5 > 1/3 | −1/4 < −1/20 < 1/12 |
| K=4 | 1/5 | 4/5 | 1 | 2/3 > 8/15 > 0 | −1/9 < −2/45 < 2/9 |

marginals 全 K/m ✓；L/R=0 ✓。H 只落在 cross-pair masks，L 只落在 co-pair masks。

**Odd bank m=5, K=2 / K=3**（5 phases，singleton 轮换）：

| case | TV(H,U) | TV(L,U) | TV(H,L) | E_comp H/U/L | C̄_treated H/U/L |
|---|---|---|---|---|---|
| K=2 | 3/20 | 3/10 | 9/20 | 4/5 > 3/5 > 1/5 | −4/25 < −3/50 < 7/50 |
| K=3 | 3/20 | 3/10 | 9/20 | 4/5 > 3/5 > 1/5 | −4/25 < −3/50 < 7/50 |

marginals 全 K/m ✓。**P_H ≠ P_U ≠ P_L 成立，不再退化。**

### 7.2 Even m 大 case TV 闭式（无稀疏估计问题）

H 是 pair-type pattern (b_max, a) 上的均匀分布，U 在该 pattern 类上均匀，两类不相交 ⇒

- `TV(H,U) = 1 − C(n,b_max)·2^b_max·C(n−b_max,a) / C(m,K)`
- `TV(L,U) = 1 − C(n,b_min)·2^b_min·C(n−b_min,a_L) / C(m,K)`
- `TV(H,L) = 1`（b_max ≠ b_min；degenerate K∈{0,m} 时为 0）

| m | K | TV(H,U) | TV(L,U) |
|---|---|---|---|
| 10 | 3 | 1/3 | 2/3 |
| 10 | 5 | 55/63 | 16/21 |
| 20 | 7 | 259/323 | 316/323 |
| 50 | 10 | ≈0.674 | ≈0.99999 |
| 50 | 25 | ≈0.9999997 | ≈0.999999 |
| 50 | 40 | ≈0.674 | ≈0.99999 |

全部严格 > 0 ✓。odd m 大 case 以 7.4 的 covariance 诊断 + MC 验证（见下）。

### 7.3 全范围闭式 sweep（pilot 范围 m∈[10,50]）

- even：567/609 个 (m,K) 严格 E_comp H>U>L 且 C̄ H<U<L；其余 42 个为 K∈{1,m−1} 的**退化全相等**（非违反）。违反：**0**。
- odd：540/580 严格；40 个退化（K∈{1,m−1}）。违反：**0**。
- **degenerate K 集合 = {0, 1, m−1, m}**（修订 #2）：K=0/m 为单点分布（空/全集）；K=1/m−1 时 H/L 均为"1 个 discordant + coin"，与 U 分布逐位相同（exact 枚举验证，§7.1 补充）。这些 K 下三 policy 分布完全相同，strict 排序不适用。
- 退化 K 的处置（gate 口径，修订 #2）：**strict E_comp H>U>L 仅对 non-degenerate K∈[2,m−2] 判**；K∈{0,1,m−1,m} 单独报告（计数/占比表），**不得构成 FAIL**。K 流由共享 q_t-Binomial 驱动，non-degenerate K 有正概率 ⇒ pooled 口径排序仍严格。

### 7.4 二阶诊断：treated-pair covariance（large-m 稳定指标）

`C_ij = P(i,j both masked) − P(i)P(j)`，treated pairs 上平均：

- Even m 闭式：C̄_H = a/n − (K/m)²，C̄_L = a_L/n − (K/m)²，C̄_U = K(K−1)/(m(m−1)) − (K/m)²
- Odd m 闭式（phase/s 条件口径）：C̄ = E_s[a(s)]/n − (K/m)²（H/L），U 同上
- 设计排序：**C̄_H < C̄_U < C̄_L**（sweep 验证 1107/1189 严格，其余退化全相等，0 违反）

### 7.5 Monte Carlo（H policy，200k items）

- m=50,K=25：max_i |rate_i − K/m| = 0.0027（采样噪声量级，非偏差）
- m=49,K=24（odd bank）：0.0027 ✓

### 7.6 Mask-structure confound diagnostics（1B，只读，不入 gate 阈值）

专家 review 指出：即使 exact K 与单位置 marginal 相同，H/U/L 仍可能连带改变更高阶 mask structure。除 intended pairwise correlation 外，必须**记录**这些连带几何差异（`design 期闭式/枚举 + 训练流实测 + 分桶报告`），以便 G3 判读时区分 intended 与 confound：

1. **mask run-length distribution**（连续被 mask 位置的 run 长度直方图）
2. **visible-token run-length / adjacency statistics**（连续可见 run 长度、相邻可见对频率）
3. **masked-visible transition count**（沿位置的 mask↔visible 切换数，≈2×discordant 相关量）
4. **treated pair distance distribution**——**H/U/L 必须逐位相同**（同 replicate 共享同一 frozen map，pair 距离结构由 map 决定、与 policy 无关）；按 near / medium / far 分桶报告
5. **relative-position / L-R trend**（与 G3a-4 同一口径，扩展为逐相对位置 mask 率表）
6. **context-visible-count**（应由 exact K 与共享 corruption context 决定，H/U/L 相同——作为共享性 sanity check）
7. **non-treated pair correlation summary**（非 treated 边对的 mask 协方差分布摘要：mean / P10 / P50 / P90）

**解释规则（FROZEN）**：不要求三 policy 的高阶结构完全一致——intervention 本身就必须改变 treated-pair joint structure。目标仅是**知道**除 intended pairwise correlation 外还有哪些 mask geometry 被连带改变；若发现意外的大幅 confound（如某 policy 的 mask 空间聚类显著偏离另两者且无法由设计解释），在报告中披露并讨论，**不追溯改 gate**。

## 8. Exposure metric E_comp（intervention diagnostic，非机制证明）

- **定义**：`E_comp = E[# exactly-one-masked active pairs / # active pairs]`（active pairs：even m 恒 n=m/2；odd m 当前 phase 的 n=(m−1)/2，singleton 不计）。
- **闭式**（设计目标值）：even：H=b_max/n，L=(K mod 2)/n，U=2K(m−K)/(m(m−1))；odd：H=E_s[min(K−s, m−1−K+s)]/n，L=E_s[(K−s) mod 2]/n，U=2K(m−K)/(m(m−1))。
- **报告口径**：overall（pooled）+ by K/m bucket + by σ bucket；训练流实测值须与闭式一致（G3a）。
- **地位**：只证明干预强度正确；**不能代替** mask-distribution identifiability（§7）。

## 9. CPI 定义（三口径）

1. **CPI_global（primary，frozen 不动）**：现有 frozen evaluator 全口径 CPI_abs / CPI_RMS（manifest `1897bd14…`，500 样本，v4.2）。**任何修改禁止。**
2. **CPI_treated（secondary，新 evaluator 模块）**：只评估训练 pair map 的 treated pairs。per sample：取该 replicate 的 frozen map 在 span 内的相对位置对，逐对 delta-swap（复用 frozen corruption/eval 打分路径，3 forward/pair），per-sample = mean |δ| over treated pairs。H_r 与 L_r 同 replicate 共享同一 map ⇒ 可逐样本配对。step0 baseline 按各 replicate map 分别给出。
3. **CPI_heldout（secondary）**：一套独立 frozen heldout map（fresh random permutation 的 **skip-2 边**；生成时强制与 rep1/rep2 训练 map 边集不相交；落盘+sha256，训练永不加载）。区分 local pair-specific vs generalized effect。

**冻结与隔离要求（修订 #3，全部为硬性要求）**：
- **训练前冻结**：treated/heldout CPI evaluator 的代码、heldout map 文件、pair 定义在**任何训练开始之前**完成并落盘 sha256；训练期间与训练后**禁止修改** evaluator（与 frozen evaluator 同等待遇）。
- **训练 RNG 完全隔离**：training 的 policy RNG 流（§10 seeds）与 evaluator RNG 永不共享；treated/heldout evaluator 的 corruption 流沿用 frozen evaluator 的既有隔离实现（manifest 冻结 corruption realization，训练流 seed 域与 evaluator seed 域不相交），pair 选择为确定性 map（无 RNG）。
- **H/U/L 同 replicate 共享 sample/span/context**：三 policy 同 replicate 使用逐位相同的 sample 流、span 结构（长度/起点）、σ、K 流与 corruption context（x_t 的初始揭示状态）；policy 只决定 mask subset 的选取（§4/§6）。共享性由 schedule digest 校验（§11 G3a-2）。
- **heldout edges 与 training treated edges 不重叠**：heldout map 生成时对每个 span length m 逐边校验与 rep1/rep2 treated 边集的交集为空（确定性 rejection 循环，m≥10 时几乎一次通过）；不重叠关系落盘记录（JSON：每 m 的三方边集 + 交集断言结果）。

**成本注**：treated/heldout 每 sample 需 m/2（even）或 m（odd）对 × 3 forward；span m≤50 ⇒ ≤150 forward/sample，500 样本 ⇒ ≤75k forward，pilot 规模可接受。

### 9.4 Calibration / scale diagnostics（1C，secondary/exploratory，不入 G3 primary）

Primary 不变：CPI_global_abs；继续报告 signed mean δ、CPI_RMS、δ quantiles（P10/P25/P50/P75/P90/P95）、masked-token NLL、token accuracy。

新增：

1. **probability / score scale summary**：local CE（masked-position NLL）、log-prob 对 {log p(a|C), log p(b|C,a), log p(b|C), log p(a|C,b)} 的均值/SD、δ_SD 相对 local CE 的比值——刻画 compatibility 信号的绝对量级。
2. **masked-token calibration diagnostic**（若 frozen evaluator 路径可可靠实现）：per-position 预测置信度（argmax token 的预测概率）分桶 vs 经验命中率（reliability 曲线）。由 treated/heldout evaluator 模块的 per-position 打分路径顺带产出；若实现成本显著或不可靠，标记 deferred 并说明原因（不硬凑）。
3. **post-hoc temperature-scaled CPI**：数学定义（SEDD score-ratio 结构下）：`log p_T(x|·) = log p(x|·)/T − log Z_T` ⇒ δ-swap 中 log Z_T 相消 ⇒ **δ_T = δ/T，CPI_abs(T) = CPI_abs(1)/T**——纯尺度退化，不新增分布形状信息。因此：
   - 标准 softmax temperature scaling **无法为 δ-swap 提供非平凡校准**；本协议将 temperature-scaled CPI 标记为 **deferred / exploratory**，理由是数学退化（δ_T = δ/T），不是实现困难；
   - 替代做法：报告 2 的 reliability 曲线 + δ quantiles（形状信息），以及 δ_SD/local CE 尺度比；
   - 若未来对单边条件概率（log p(a|C) 等）做 calibration，须用独立 calibration split（与 CPI test samples 不相交）、H/U/L 用同一 fitting procedure、只作 secondary、不入 G3 primary gate。当前不执行。

### 9.5 小型 global path metric（1D，secondary，不入 G3c primary）

为验证 local swap metric 与 global reveal-order behavior 的对应关系：

- **在 step2500 最终 checkpoint（+step0 baseline）上**，对一个**训练前冻结**的小型 evaluation subset（frozen manifest 的固定子集，例如按 frozen 规则抽取的 64 样本；索引列表落盘 + sha256，H/U/L、rep1/rep2 完全共享）额外计算：
  - **OrderGap**（frozen 6 路径口径，`evaluation/eval_order_gap.py` 复用于子集）；
  - **path-score variance**：per sample 6 条冻结路径得分的方差，跨样本平均——量化单一样本内揭示顺序的 score 离散度。
- 要求：子集训练前冻结；evaluator RNG 独立于 training；只在 step2500（+step0）跑，不跑全部 checkpoint；secondary，**不进入 G3c primary gate**。
- 目标：检查 CPI 改善是否伴随 global path sensitivity 下降（相关性报告，非因果 gate）。

## 10. Pilot 结构（审批后才执行）

- **Runs**：U/H/L × 2 training replicates = **6 runs**；SEDD-small，seq 256，span [10,50]，batch 32，2500 steps，dropout=0，同一 pretrained 初始化，共享 step0。
- **Checkpoints**：EMA @500/1020/2500 + raw @2500（同 v1.2 容器契约）。
- **Seeds**：共享四元流沿用 1000+r…4000+r 约定；policy 流：H pair-choice 8001+r / H orient 8101+r / L pair-choice 8201+r / L orient 8301+r / U uniform 8401+r / singleton coin（仅 odd m）8501+r；dropout 7001+r。frozen evaluator 随机流不动。
- **Metrics**：primary = masked-span NLL、token acc、CPI_global_abs、CPI_global_RMS；secondary = signed δ、δ quantiles（P10/P25/P50/P75/P90/P95）、CPI_treated、CPI_heldout、E_comp、C̄_treated、§9.4 校准/尺度诊断、§9.5 step2500 frozen 子集 OrderGap + path-score variance。OrderGap **仅**在 step2500（+step0）frozen 子集上跑，不跑全 checkpoint。
- **训练期 logging（1F，零/低成本，不改 objective）**：train loss 轨迹；gradient norm（每 100 步记录 + rolling variance 近似，**不做 per-example gradient**）；H/U/L 每 checkpoint 的 task NLL/acc（评估计划已含）。
- **存储**：6 × ~2.6GB ≈ 15.7GB + maps/结果 <0.1GB（磁盘计划见 §13）。

## 11. Gate（四层，FROZEN）

**G3a — Intervention validity（全部必须 PASS，任一 FAIL ⇒ 禁止机制解释）**
1. exact K per item（训练流断言；含 odd-m singleton 硬币后的 K−s 分解可行性断言）；
2. U/H/L 同 replicate 共享 sample/span/σ/K 流与 phase/singleton 硬币流（schedule digest sha256 一致）；
3. 单位置 marginal parity（实测 per-position mask rate ≈ K/m，容差=采样噪声界）；
4. 无 L/R 趋势（discordant pair 左/右 mask 计数差 ≈ 0）；
5. E_comp：**strict H > U > L 仅对 non-degenerate K∈[2,m−2] 判**（修订 #2）；K∈{0,1,m−1,m} 单独报告（计数/占比表，三 policy 应全相等），**不得构成 FAIL**；pooled overall 口径 H > U > L 且分桶与闭式一致；
6. mask distributions 可区分（§7 设计期 exact TV > 0 + 实测 C̄_treated 排序 H<U<L；K∈{0,1,m−1,m} 除外，单独报告）；
7. pairing/bank hash 正确（frozen map 文件 sha256 与加载校验一致）；heldout 与 training treated edges 不重叠断言通过；
8. **confound diagnostics 记录（1B，非 gate 阈值）**：§7.6 的 7 项全部产出并分桶报告；判读时区分 intended structural difference 与 unwanted confound——**不要求** H/U/L 的所有 pairwise/joint mask statistics 相同（intervention 本身就要改变 treated-pair correlation），只要求逐项记录与披露。

**G3b — Local manipulation check**：CPI_treated_H < CPI_treated_L，两 training replicate 点估计同向；per-replicate paired bootstrap（10k, seed=0）+ 按 replicate 聚类 pooled CI（同 v1.2 口径）。PASS 仅允许"intervention locally affects treated pairs"表述。

**G3c — Global mechanism signal（FROZEN，1E）**
- **Primary causal contrast：H vs L**（intervention contrast 最大的一组）：CPI_global_abs_H < CPI_global_abs_L，两 replicate 同向；CI 判据同 G1a。
- **Secondary 预注册对比**：H vs U、U vs L（各自 per-rep paired bootstrap + 聚类 CI）；最理想的 dose-like ordering 为 CPI_H < CPI_U < CPI_L——**严格 H<U<L 不是 primary PASS 必需条件**。
- **U 位置的解释规则（预注册，禁止事后改）**：若 H<L 成立且 U 居中 → dose-like support；若 H<L 但 U 不居中（U ≤ H 或 U ≥ L）→ H vs L primary contrast 仍按原判据判定，U 偏离作为 unexpected mask-structure effect 在报告中披露讨论，不重新解释 hypothesis。
- **其他 secondary 证据**：CPI_RMS、signed δ、δ quantiles、CPI_heldout 同向（generalization evidence）、§9.5 OrderGap subset / path-score variance 同向（global path sensitivity 对应性）、§9.4 诊断。均不入 primary PASS 条件。

**G3d — Matched performance**：masked NLL ±0.02 / token acc ±0.01（沿用 §5.3 配对规则与 checkpoint 池，**匹配规则事前冻结，禁止后验选 checkpoint**）；同时报告 CI；no overlap ⇒ **INCONCLUSIVE（非 FAIL）**；匹配后重做 G3c。

**结论语言（预注册）**：
- G3a+G3c+G3d 全 PASS ⇒ "Results support complementary partial-conditioning exposure as a driver of global reveal-order compatibility attenuation."
- 仅 G3a+G3b ⇒ 只能报告 local pair-specific effect。
- 其他 ⇒ 见 Case 表。

## 12. Case 表（预注册冻结，禁止事后重写）

| Case | 观察 | 结论 |
|---|---|---|
| A | H global CPI < L，matched-performance 后仍成立 | supports complementary-exposure mechanism |
| B | treated CPI 有差异，global/heldout 无差异 | local pair-specific effect only |
| C | H≈U≈L | complementary exposure not supported；generic supervised adaptation 增强 |
| D | H > L | 与当前机制预测相反 |
| E | task performance 不可匹配 | mechanism verdict INCONCLUSIVE |

## 13. 磁盘与预算（v2.1 训练前必须复核）

当前（2026-10-06）：D: free ≈ **18GB**；VHDX ≈ 116.7GB；exp_local ≈ 95GB。
v2.1 新增：≈15.7GB checkpoints + <0.1GB 其他 ≈ **16GB**；最低安全要求 = 16 + ≥10GB margin = 26GB。
**正式训练前目标（FROZEN）：D: free ≥ 30GB，最好 ≥ 35GB**——不以 26GB 卡边。
路径：先做**只读 disk retirement audit**（候选 p1-s1/p1-s2：优先"保留 final checkpoint + logs + metadata + frozen analysis，只删 dense intermediate"，见 preflight plan P4）→ 用户逐项批准 → Linux 内删除 → DiskPart compact → 复核（D: free 达标 + v1.2 资产 sha256 抽查）。**禁止边跑边赌磁盘；禁止碰 mechpilot v1.2 / v1.2 report / P1/P2 JSON / protocol / manifests / git / 正式 SFT-RL 关键资产。**

## 14. 执行顺序（FROZEN）

1. 本协议冻结（v1.1 sha256 落盘，不再改动）
2. frozen pair/bank map 生成 + sha256（rep1/rep2/heldout）+ heldout 不重叠断言落盘
3. **treated/heldout CPI evaluator 实现并冻结（训练前，修订 #3）**：代码 + heldout map + pair 定义全部落盘 sha256；此后禁止修改；§9.5 OrderGap frozen 子集（索引列表 + sha256）同期冻结
4. G3a 设计期 preflight 复跑（§7 脚本 + 训练流 dry-run：digest / exact-K / phase 消耗核对）
5. **磁盘前置**：只读 retirement audit → 用户批准删除清单 → 清理 → DiskPart compact → D: free ≥30GB（最好 ≥35GB）复核 + v1.2 资产完整性抽查
6. **Implementation + 测试**：mask policy / odd-m bank / H-U-L 采样 / evaluator（treated/heldout/OrderGap 子集）/ logging——先 **unit tests**，再 **50–100 step smoke test**（检查：exact K、schedule hash、map hash、H/U/L 同 sample/span/σ/K、无 RNG 污染、marginal parity、E_comp 排序、covariance 排序、loss finite、VRAM、磁盘增长、checkpoint 契约）。**全部 PASS 才允许正式 6-run pilot。**
7. 6 run 训练（driver：manifest hash / frozen 文件 / map hash 守卫）→ global CPI/task 评估 + treated/heldout CPI + §9.5 step2500 OrderGap subset
8. G3a/b/c/d 分析 → 报告 → 冻结

**当前禁止**：实现 mask policy 代码、实现 treated/heldout evaluator、生成大 checkpoint、启动任何训练、rm/compact——须用户按本表逐项放行。

## 15. 两阶段设计与未来工作（1G/1H/1I，FROZEN）

### 15.1 Pilot → Confirmatory 进入条件（1G）

v2.1 当前仍是 pilot：H/U/L × 2 paired replicates = 6 runs，**不扩到 3–5 seeds**。
进入 Confirmatory Stage 的预注册条件（须同时满足）：
1. G3a PASS；
2. H vs L global CPI 两 replicate 同方向；
3. effect size 有研究意义（以 pilot 的 CI 宽度与 δ_SD 效应量基线评估，判定在报告中给出）；
4. performance matching 可行（G3d 非 INCONCLUSIVE）。
满足后：Confirmatory 扩展到 **≥4 total paired replicates**（基于 pilot effect size / CI 做 power analysis 后再定总数）。
若 pilot 无稳定 signal：**不盲目烧 15 个 runs**，按 Case 表收束。

### 15.2 Dose-response follow-up（1H，仅 Future Confirmatory Stage 定义，本 pilot 不执行）

若 H/L intervention 过强或 H/L performance 分离明显，下一阶段用 λ interpolation：
`P(use H-style policy) = λ，P(use U-style policy) = 1−λ`（λ 由低到高），核心问题：λ ↑ 是否导致 CPI ↓。v2.1 pilot 仍只做 H/U/L。

### 15.3 Deferred 大型扩展（1I，全部归入 "if mechanism survives pilot" future work）

- full-mask SFT baseline、AR SFT baseline
- 多模型规模、跨数据域
- 大量不同 K 的独立训练
- 一开始就跑 5 seeds

以上均**不进入当前 pilot**，避免 scope explosion。
