# Mechanism Pilot v2.1 — Complementary Exposure Protocol

**状态：DRAFT v0.1（非冻结，待用户复核；未经审批禁止实现/训练）**
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

## 2. 预注册假设（DRAFT）

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

## 5. 单位置 marginal 不变性证明

**Even m**：设 pair 标签上的均匀置换群 + 每 pair 独立 orientation flip 构成自同构群 G，作用于 m 个位置**传递**（任意 i 可到任意 j：pair 移过去 + 按需 flip）。H/L 采样过程对 G 不变（pair 选择均匀、orientation fair coin）⇒ 分布 G-不变 ⇒ 所有位置 marginal 相等；又 Σ_i P(i masked) = K（exact K）⇒ **P_H(i)=P_L(i)=K/m**。U 平凡 K/m。∎

**Odd m（§6 bank）**：paired marginal α 与 singleton 概率 q 满足 (m−1)α + q = K；取 q = K/m（singleton 硬币流，U/H/L 共享）⇒ α = K/m ⇒ **全体位置 marginal 精确 K/m**。∎

**L/R 趋势**：discordant pair 的 orientation 为 fair coin（H/L）⇒ 任意 treated pair 左端/右端 mask 率差 = 0；U 均匀 ⇒ 0。exact（小 case 枚举验证，§7）。

## 6. Odd m = 2n+1：balanced persistent matching bank

不能有固定 singleton 相对位置。设计（DRAFT）：

- **Bank 构造**：每 replicate × odd m，frozen random base permutation π_r；phase t = 0..m−1 的 matching：`singleton = π_r(t)`，pairs = `(π_r(t+1), π_r(t+2)), (π_r(t+3), π_r(t+4)), …, (π_r(t+m−2), π_r(t+m−1))`（下标 mod m）。n = (m−1)/2 对。
  - 性质：每个位置在 m 个 phase 中**恰好一次** singleton；bank 的 pair 并集 = π_r 的 cycle 边 `{(π_r(i), π_r(i+1)) mod m}`（m 条边，每条边恰在 2 个 phase active）。
- **Phase schedule（U/H/L 完全共享，确定性）**：对 span 内第 s 个 item，`phase = (o_span + s) mod m`，`o_span` 为该 span 的 frozen offset（由 span 流确定，非 RNG、非每步 fresh）。长 span 下每 phase 频率 → 1/m ⇒ 每位置 singleton 长期频率相等 ✓。
- **Singleton 硬币（共享流）**：每 item 消耗一次 `s ~ Bern(K/m)`（seed 8500+r，U/H/L 共享；U 不用于 mask 但流一致）。
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

- even：567/609 个 (m,K) 严格 E_comp H>U>L 且 C̄ H<U<L；其余 42 个为 K=1 或 m−1 的**退化全相等**（非违反）。违反：**0**。
- odd：540/580 严格；40 个退化（K=1 / m−1）。违反：**0**。
- 退化 K 的处置：逐-K 严格性在 K∈{1,m−1} 不成立（该 K 下所有 policy 同分布），但 K 流由共享 q_t-Binomial 驱动，非退化 K 有正概率 ⇒ **pooled E_comp 排序严格**。gate 用 pooled 口径 + 分桶报告。

### 7.4 二阶诊断：treated-pair covariance（large-m 稳定指标）

`C_ij = P(i,j both masked) − P(i)P(j)`，treated pairs 上平均：

- Even m 闭式：C̄_H = a/n − (K/m)²，C̄_L = a_L/n − (K/m)²，C̄_U = K(K−1)/(m(m−1)) − (K/m)²
- Odd m 闭式（phase/s 条件口径）：C̄ = E_s[a(s)]/n − (K/m)²（H/L），U 同上
- 设计排序：**C̄_H < C̄_U < C̄_L**（sweep 验证 1107/1189 严格，其余退化全相等，0 违反）

### 7.5 Monte Carlo（H policy，200k items）

- m=50,K=25：max_i |rate_i − K/m| = 0.0027（采样噪声量级，非偏差）
- m=49,K=24（odd bank）：0.0027 ✓

## 8. Exposure metric E_comp（intervention diagnostic，非机制证明）

- **定义**：`E_comp = E[# exactly-one-masked active pairs / # active pairs]`（active pairs：even m 恒 n=m/2；odd m 当前 phase 的 n=(m−1)/2，singleton 不计）。
- **闭式**（设计目标值）：even：H=b_max/n，L=(K mod 2)/n，U=2K(m−K)/(m(m−1))；odd：H=E_s[min(K−s, m−1−K+s)]/n，L=E_s[(K−s) mod 2]/n，U=2K(m−K)/(m(m−1))。
- **报告口径**：overall（pooled）+ by K/m bucket + by σ bucket；训练流实测值须与闭式一致（G3a）。
- **地位**：只证明干预强度正确；**不能代替** mask-distribution identifiability（§7）。

## 9. CPI 定义（三口径）

1. **CPI_global（primary，frozen 不动）**：现有 frozen evaluator 全口径 CPI_abs / CPI_RMS（manifest `1897bd14…`，500 样本，v4.2）。**任何修改禁止。**
2. **CPI_treated（secondary，新 evaluator 模块，实现待审批后）**：只评估训练 pair map 的 treated pairs。per sample：取该 replicate 的 frozen map 在 span 内的相对位置对，逐对 delta-swap（复用 frozen corruption/eval 打分路径，3 forward/pair），per-sample = mean |δ| over treated pairs。H_r 与 L_r 同 replicate 共享同一 map ⇒ 可逐样本配对。step0 baseline 按各 replicate map 分别给出。
3. **CPI_heldout（secondary）**：一套独立 frozen heldout map（fresh random permutation 的 **skip-2 边**；生成时强制与 rep1/rep2 训练 map 边集不相交；落盘+sha256，训练永不加载）。区分 local pair-specific vs generalized effect。

**成本注**：treated/heldout 每 sample 需 m/2（even）或 m（odd）对 × 3 forward；span m≤50 ⇒ ≤150 forward/sample，500 样本 ⇒ ≤75k forward，pilot 规模可接受。

## 10. Pilot 结构（审批后才执行）

- **Runs**：U/H/L × 2 training replicates = **6 runs**；SEDD-small，seq 256，span [10,50]，batch 32，2500 steps，dropout=0，同一 pretrained 初始化，共享 step0。
- **Checkpoints**：EMA @500/1020/2500 + raw @2500（同 v1.2 容器契约）。
- **Seeds**：共享四元流沿用 1000+r…4000+r 约定；policy 流：H pair-choice 8001+r / H orient 8101+r / L pair-choice 8201+r / L orient 8301+r / U uniform 8401+r / singleton coin（仅 odd m）8501+r；dropout 7001+r。frozen evaluator 随机流不动。
- **Metrics**：primary = masked-span NLL、token acc、CPI_global_abs、CPI_global_RMS；secondary = signed δ、CPI_treated、CPI_heldout、E_comp、C̄_treated。**OrderGap 暂不跑。**
- **存储**：6 × ~2.6GB ≈ 15.7GB + maps/结果 <0.1GB（磁盘计划见 §13）。

## 11. Gate（四层，DRAFT）

**G3a — Intervention validity（全部必须 PASS，任一 FAIL ⇒ 禁止机制解释）**
1. exact K per item（训练流断言）；
2. U/H/L 同 replicate 共享 sample/span/σ/K 流（schedule digest sha256 一致）；
3. 单位置 marginal parity（实测 per-position mask rate ≈ K/m，容差=采样噪声界）；
4. 无 L/R 趋势（discordant pair 左/右 mask 计数差 ≈ 0）；
5. E_comp：实测 overall H > U > L（且分桶与闭式一致）；
6. mask distributions 可区分（§7 设计期 exact TV > 0 + 实测 C̄_treated 排序 H<U<L）；
7. pairing/bank hash 正确（frozen map 文件 sha256 与加载校验一致）。

**G3b — Local manipulation check**：CPI_treated_H < CPI_treated_L，两 training replicate 点估计同向；per-replicate paired bootstrap（10k, seed=0）+ 按 replicate 聚类 pooled CI（同 v1.2 口径）。PASS 仅允许"intervention locally affects treated pairs"表述。

**G3c — Global mechanism signal（primary）**：CPI_global_H < CPI_global_L，两 replicate 同向；CI 判据同 G1a。U 的 H<U<L 为 dose-like secondary evidence（不强制完美单调）；CPI_heldout 同向为 generalization evidence。

**G3d — Matched performance**：masked NLL ±0.02 / token acc ±0.01（沿用 §5.3 配对规则与 checkpoint 池）；no overlap ⇒ **INCONCLUSIVE（非 FAIL）**；匹配后重做 G3c。

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
v2.1 新增：≈15.7GB checkpoints + <0.1GB 其他 ≈ **16GB**；安全要求 = 16 + ≥10GB margin ⇒ **需 D: free ≥ 26GB**。
⇒ 当前不满足。训练前需受控清理（候选：p1-s1/p1-s2 等已退役 stage 资产，**须用户逐项确认，禁止碰 v1.2 frozen 资产**）+ Windows DiskPart compact（既有流程）。禁止边跑边赌磁盘。

## 14. 执行顺序（审批后）

1. 本协议用户复核 → FROZEN（sha256 落盘）
2. frozen pair/bank map 生成 + sha256（rep1/rep2/heldout）
3. G3a 设计期 preflight 复跑（§7 脚本 + 训练流 dry-run）
4. 磁盘清理 + compact
5. 6 run 训练 → 18 checkpoint 评估（global CPI/task）+ treated/heldout CPI
6. G3a/b/c/d 分析 → 报告 → 冻结

**当前禁止**：实现 mask policy 代码、生成大 checkpoint、启动任何训练。
