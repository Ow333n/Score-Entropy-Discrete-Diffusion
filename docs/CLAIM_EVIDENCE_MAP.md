# CLAIM EVIDENCE MAP — 结论分级、专项审计与证据覆盖审计

> 状态:**REVIEWED(2026-10-08,证据体系与一致性修订已经用户审查)**。
> REVIEWED 仅指导航层 / 证据体系的组织通过审查,**不意味着任何底层科研假设获得
> 证明**;各 claim 的证据等级见本文件分类体系(§0)。
> 本文只做"结论 → 证据 → 局限"映射;**不追溯修改任何 frozen 报告/protocol/results**
> 的原始结论。每条 claim 的 `分类` 是全项目统一口径,`修订史` 记录该结论曾被谁修正。

## 0. 分类体系(重复声明,与 START_HERE/REGISTRY 一致)

- `VALIDATED_OBSERVATION`:复现过、gate PASS、跨 evaluator 代际稳健
- `SUPPORTED_BUT_NOT_CAUSAL`:观察到的关联/效应,机制因果未建立
- `INCONCLUSIVE`:**实验已执行**、但证据不足以区分 hypothesis(**≠ 证明无效**)
- `UNSUPPORTED_HYPOTHESIS`:预注册检验未获支持(CI 含 0 / 方向反向)
- `SUPERSEDED_MEASUREMENT`:被新 evaluator / dense 分辨率取代
- `INVALID_OR_VOIDED_RUN`:失败/作废 run(provenance 保留)
- `NOT_EXECUTED`:**计划过 / 已定义 RQ,但尚未正式执行实验**(区别于
  INCONCLUSIVE:后者是"执行了但无法区分")

三条纪律:技术 gate PASS ≠ 研究假设成立;复现成功 ≠ 机制得到证明;
INCONCLUSIVE ≠ 证明无效。

## 1. 当前研究树(RQ,用户 2026-10-08 口径,全项目统一)

- **Main empirical phenomenon**:partial-reveal SFT reduces reveal-order
  incompatibility / sensitivity(CPI↓、OrderGap↓,两 seed 复现)。
- **RQ1:Why does CPI decrease during early SFT?**
  - Current answer:**generic conditional-estimation improvement is the current
    most parsimonious explanation for the early attenuation**(作用域限定为
    early SFT-associated CPI attenuation),but **causal independence remains
    untested**;不得写成"已证明 CE 导致 CPI 下降"。
  - Rejected / unsupported candidate explanations:
    - Fresh vs Fixed:INCONCLUSIVE
    - Directional exposure:INCONCLUSIVE
    - Complementary exposure:intervention valid,effect INCONCLUSIVE
    - JS_dep:descriptive / timing mismatch
    - D_ab:replicated correlate,not independent mechanism
- **RQ2:Why does CPI stop improving / remain non-zero while CE continues
  improving?**
  - Current status:**NOT_EXECUTED**。
  - 无正式 causal / lower-bound 实验;**尚未证明存在 positive lower bound**;
    当前 empirical saturation ≠ theoretical lower bound;现有数据只构成
    下一阶段研究动机。
- **RL branch(独立扩展问题,不属 RQ1 机制树)**:Does post-SFT reward
  optimization further alter compatibility?Current 500-step result:no
  detectable change under the tested setting(Scenario B)。

## 2. Claim 清单

### C-01 SFT 稳健降低 CPI(|δ|)vs pretrained
- **分类**:VALIDATED_OBSERVATION
- **证据**:Stage-4 paired bootstrap 两 seed CI 排除 0(s1 −0.0402[−0.0722,−0.0066],
  s2 −0.0461[−0.0779,−0.0143],`results/vanilla/stage4_gate.json`);FP32 bulk 55
  checkpoint(rel −14.7%/RMS −9.3%/q50 −27.8%,`results/phase1_diag/phase1_dynamics_analysis.json`);
  v2.1 保留事实"All H/U/L variants still exhibit CPI attenuation"
  (`reports/mechanism_pilot_v2_1_report.md`);dense 轨迹两 seed 一致
  (`results/phase2/optionb_dense_analysis.json`)。
- **局限**:全部在同一 pretrained init + 同一 manifest 上;BF16 历史值与 FP32 值
  有量级 ~2e-3 差异(已 BRIDGE-A harmonize)。
- **是否保留**:是(论文/面试主结论)。

### C-02 SFT 改善 masked-span NLL 与 token acc(G1 任务可学性)
- **分类**:VALIDATED_OBSERVATION
- **证据**:G1 PASS(NLL 4.281→3.875、acc 0.322→0.362,`reports/stage4_report/report.md`
  §4;`results/vanilla/g1_s{1,2}_*.json`);v2.1 六 run fixed_eval@2500 masked NLL
  3.18–3.27(全部学会任务)。
- **局限**:BF16 口径为主;FP32 core 的 CE 同向(`results/phase1_diag/core/`)。
- **是否保留**:是。

### C-03 CPI 衰减的早期时间窗(250–750 main window;后期 saturation;2500 rebound 仅 descriptive)
- **证据分级**(三段不同等级,**不得混为同一证据强度**):
  - **250–750 main attenuation window = VALIDATED_OBSERVATION**:EXP-05 定位
    (`results/p1_dense/` 84 JSON)+ EXP-19 dense 轨迹两 seed 一致 −9.61%/−10.42%
    (`results/phase2/optionb_dense_analysis.json`),tensor-exact 复现(EXP-18)。
  - **Later saturation / near-bottom = supported descriptive trajectory**:
    CPI reaches its lowest observed mean around 1020–1500(0.2749/0.2717)。
  - **2500 rebound = descriptive only**:s1 1500→2500 ≈ +0.007、s2 ≈ +0.012,
    **未经 paired bootstrap significance / effect-size / reversal 检验**。
- **标准措辞**:CPI reaches its lowest observed mean around 1020–1500 and shows
  a small descriptive rebound by step 2500; the rebound itself has not been
  established as a statistically reliable reversal.(中文:CPI 的最低观测均值
  出现在约 1020–1500 步,2500 步时均值出现小幅 descriptive rebound;目前尚未
  证明该 rebound 是统计上可靠的 reversal。)
- **局限**:750–1020 无 dense 内点(只有端点);1020→1500 只有两个点。
- **是否保留**:是。

### C-04 OrderGap 随 SFT 下降
- **分类**:VALIDATED_OBSERVATION
- **证据**:BF16:OG_raw 10.2445→8.76/8.82,ΔOG s1 −1.4888 CI[−2.1225,−0.9357]
  (`results/vanilla/stage4_gate.json`);FP32 finite-path:12.2559→10.24–11.41
  (`results/phase1_diag/phase1_dynamics_analysis.json`)。
- **局限**:OG 是 6 路径近似,非 m! 全排列;FP32 与 BF16 基线数值不可直接互比
  (口径差异),但方向一致。
- **是否保留**:是。

### C-05 Stage-4 判定:CPI↔OrderGap late-stage co-movement 不成立 → B_DIAGNOSTIC_ONLY(永久冻结)
- **分类**:VALIDATED_OBSERVATION(gate verdict 本身;冻结,禁止上诉)
- **证据**:`results/vanilla/stage4_gate.json`(两 seed cpi_stable/og_stable=true、
  co_movement=false);`protocol/experiment_protocol_v4.2.md` L3–4 永久固定声明。
- **局限**:co-movement 只测 early→late 区间;不构成对任何机制的单点否定。
- **是否保留**:是(冻结结论,引用时不得改述)。

### C-06 P1-B:late-stage CE 改善与 CPI 衰减无稳定共变("单纯 estimation 改善不足以解释 late-stage")
- **分类**:SUPPORTED_BUT_NOT_CAUSAL
- **证据**:pooled Spearman(ΔCE,ΔCPI)=−0.105、run-clustered slope CI[−0.618,−0.029]
  (`results/phase1_diag/phase1_dynamics_analysis.json`,VERDICT=P1-B)。
- **局限**:措辞已 nuance 化(只表述"late-stage CE improvement 不与 CPI
  attenuation 稳定共变";不写"完全无关/机制被排除",commit `7dad392`);相关
  性结论,不提供机制。
- **是否保留**:是。

### C-07 早期窗口 CE–CPI 共变(旧 BF16 p1_dense 的 within-run 强正相关 + ce_leads)
- **分类**:SUPERSEDED_MEASUREMENT
- **证据**:旧值 within-run +1.0/+0.89(`results/p1_dense/`,BF16,CE 量化
  0.0156–0.031);被 FP32 3 点 Spearman −0.105 与 dense FP32 轨迹取代。
- **修订史**:`de69bb9`(P1 dense 判读)→ `ce14103`(FP32 反驳)→ `41d73cb`
  (dense 轨迹最终取代)。旧值仍存档,仅标注 secondary historical。
- **是否保留**:档案保留,不作结论使用。

### C-08 v1.2 fresh vs fixed(A vs B)机制假说
- **分类**:INCONCLUSIVE
- **证据**:ΔCPI(A−B)两 replicate 方向相反(rep1 −0.0059 CI[−0.0382,0.0235]、
  rep2 +0.0078 CI[−0.0211,0.0373]);G1b INCONCLUSIVE
  (`results/mechanism_pilot/p1_stage1_analysis.json`、`p1_g1b_analysis.json`)。
- **局限**:INCONCLUSIVE ≠ 无效;gate 设计无法区分"无效应"与"效应小于检出能力"。
- **是否保留**:是。

### C-09 v1.2 directional(C vs D)假说
- **分类**:INCONCLUSIVE
- **证据**:G2 primary(2500)=INCONCLUSIVE、1020=FAIL(定义为 early transient 不
  升格,`results/mechanism_pilot/p2_g2_analysis.json`)。
- **局限**:C/D 含 positional exposure confound(协议明示不得单独因果)。
- **是否保留**:是。

### C-10 v2.1 complementary exposure(H/U/L)机制假说
- **分类**:INCONCLUSIVE(机制)+ VALIDATED_OBSERVATION(干预实现有效)
- **证据**:G3a **PASS 8/8**(intervention validity,`results/mechanism_pilot_v21/
  v21_g3_analysis.json`);G3c H vs L pooled +0.01078 CI[−0.02154,0.04417]、两
  replicate 反向 → Case C。
- **局限**:G3d 的 NLL±0.02 匹配容差低于 BF16 量化粒度 → 已 errata 化为
  "评估器分辨率下无 NLL 分离";Case C 官方措辞保留衰减事实,禁写"已被证明无效"。
- **是否保留**:是。

### C-11 RL-500 的观测结果(Scenario B)
- **分类**:SUPPORTED_BUT_NOT_CAUSAL(观测结果 = 该设定下近似稳定的
  compatibility;因果外推未建立)
- **Observed outcome**:Scenario B — no detectable compatibility change under
  the current setting(16 post evals CI 全含 0,`results/rl_pilot/`)。
- **Hypothesis status**:predicting substantial further compatibility change 的
  假设在该精确设定下未获支持;**这不等价于** "RL does not affect compatibility
  in general"。**This is not an equivalence test and does not establish a zero
  effect.**(observed outcome 与 inferential status 是两回事:16 个 CI 含 0 是
  观测结果,不是零效应证明。)禁用措辞:"RL 无效"、"RL 不改变 CPI"、
  "RL hypothesis failed"(除非加 under this exact setting / horizon / reward /
  seed 限定)。
- **是否保留**:是(负面结果的准确表述)。

### C-12 K-ablation:K=4 ≈ K=1
- **分类**:INCONCLUSIVE
- **证据**:reward 0.2691 vs 0.2732、drift ~1.6e-4(`results/rl_k_ablation/
  comparison.json`);面试口径已降级("没有证据表明是主要瓶颈")。
- **是否保留**:是(诊断记录)。

### C-13 BF16 历史结论定性稳健(BRIDGE-A);精细 dynamics 用 FP32
- **分类**:VALIDATED_OBSERVATION
- **证据**:direction 14/14、CPI Spearman 0.9821、per-sample |δ| Pearson≥0.9
  (`results/phase1_bridge/bridge_analysis.json`,case=BRIDGE-A);precision ladder
  B-C 工程等价 H1-2500 PASS(`results/phase1_preflight/precision_ladder_report.json`)。
- **是否保留**:是(口径转换的权威依据)。

### C-14 BF16 evaluator 存在量化缺陷(lattice 已按真实 artifact 重新验证)
- **分类**:VALIDATED_OBSERVATION(技术事实)
- **证据(2026-10-08 重新计算,修正原 "1/256" 表述)**:
  - **Empirical quantization statement(限定在已观察的 Level-A 数值范围,
    非全值域全局证明)**:已观察的全部 Level-A CPI_abs 值均落在 1/512(2⁻⁹)
    网格上——`results/phase1_preflight/precision_ladder_report.json` 实测:
    step0 = 0.255859375 = **131/512**(奇数分子,不落在 1/256 上)、
    H1-2500 = 0.20703125 = 106/512、pretrained = 0.328125 = 168/512。
    故"1/256"表述错误;但 1/512 是 **empirical quantization statement**,
    仅对已观察数值成立。
  - **δ per-sample:无单一 2 的幂网格**——4 项 bf16 logp 的 fp32 组合
    (实测 `results/vanilla/cpi_s1_10200.json` per_sample delta 200 值,
    最小相邻间距 ≈3.8e-5,~2⁻¹⁵ 级)。
  - **CE/NLL per-sample:落在 2⁻¹⁰ 网格、相邻间距最小 2⁻⁷**(同文件实测);
    raw bf16 logp 本身的 ulp 为 0.0156–0.031 nats(值域相关,2⁻⁶~2⁻⁵)。
  - 根因:transformer.py:282 的 bf16 autocast(v2.1 errata +
    `protocol/errata_v4.2_eval_provenance.md`);ladder Level A
    off_bf16_grid_fraction=0.0、B/C=1.0。
- **修正结论**:原 claim map "CPI 落 1/256 网格" **错误**;正确表述为"已观察
  Level-A CPI_abs 值均落在 1/512(2⁻⁹)网格"(empirical,限定已观察范围);
  精细 dynamics 一律用 FP32。
- **是否保留**:是。

### C-15 pretrained 基线数值
- **分类**:SUPERSEDED_MEASUREMENT(BF16 旧值)+ VALIDATED_OBSERVATION(FP32 现值)
- **证据**:旧随机对口径 0.0393(`results/pilot/cpi_pretrained_baseline.json`,已
  移出主链)→ BF16 frozen 0.328125(`results/pretrained/cpi.json`)→ FP32
  0.3241808526828245(`results/phase1_diag/core/pretrained-step0.json`)。
- **修订史**:`de69bb9`(口径切换)→ `ce14103`(FP32)。
- **是否保留**:FP32 现值保留;旧值存档。

### C-16 JS_dep 的变化与定位
- **分类**:SUPPORTED_BUT_NOT_CAUSAL(descriptive)
- **证据**:全程 −0.002890(−6.27%,s1)/−0.002786(−6.04%,s2);0–250 已占全程
  ~48%/~45%;active window −3.39%/−4.15%(vs CPI −9.61%/−10.42%)→ timing
  mismatch、无独立 transition(`results/phase2/optionb_case_adjudication.md` §6)。
- **措辞纪律**:写成"JS_dep changes modestly and reproducibly, but its temporal
  profile does not align with the CPI active window…"——**不要**写"几乎没变/2%"。
- **是否保留**:是。

### C-17 D_ab 的变化与定位
- **分类**:SUPPORTED_BUT_NOT_CAUSAL
- **证据**:两 seed 复现性上升(active +0.0223/+0.0233);与 CE per-seed Spearman
  **ρ = −1**(各 14 点,严格单调;pooled 28 点 −0.996);1020 后继续上升(+0.006/
  +0.011)不 plateau(`optionb_case_adjudication.md` §5;`optionb_dense_analysis.json`)。
- **措辞纪律**:定位为 "a robust correlate of training progress / compatibility
  improvement whose independence from conditional-estimation improvement is not
  established";**不称 mechanism**。
- **是否保留**:是(与 Case A 伴生)。

### C-18 最终裁定 Case A(作用域限定为 early CPI attenuation)
- **分类**:SUPPORTED_BUT_NOT_CAUSAL
- **标准措辞**(用户 F + 2026-10-08 作用域修订,引用时不得改写):
  > "Generic conditional-estimation improvement is the current most parsimonious
  > explanation for the early SFT-associated CPI attenuation; causal independence
  > remains untested, and this explanation does not by itself account for the
  > later compatibility saturation / residual gap."
  > (中文:generic conditional-estimation improvement 是当前对 SFT 前期 CPI 衰减
  > 最简洁的解释;其因果独立性尚未验证,而且该解释本身不能解释后期 CE 持续改善
  > 但 CPI 不再继续下降的 compatibility saturation / residual gap。)
- **语义边界**(必须强调):Case A = most parsimonious explanation **among current
  candidates,且仅对 early attenuation**;不是 global monotonic CE→CPI law;
  不是 causal proof;不得写成 "CE improvement causes CPI reduction" 或
  "CE 已被证明是机制"。
- **证据**:`results/phase2/optionb_case_adjudication.md`(FINAL);机械脚本已合规化
  (`scripts/optionb_dense_analysis.py`:independence 未实现时只输出
  B_candidate_unadjudicated,`93c8e44`)。
- **修订史**:机械 Case B(`0c54822` 的脚本逻辑)→ 用户裁定(`93c8e44`)→ 人工终裁
  Case A(`41d73cb`)→ 作用域限定修订(`docs` 一致性修订,2026-10-08)。
- **是否保留**:是(当前项目终态)。

### C-19 RQ2(residual gap / non-zero saturation)
- **RQ2 正式表述**:研究 **finite-step SFT 下的 residual compatibility gap 与
  CE–CPI decoupling**——即"后期 CE 持续改善而 CPI 不再继续下降"这一现象
  的结构性来源。
- **分类**:**NOT_EXECUTED**(计划过 / 已定义 RQ,但尚未正式执行实验——不是
  INCONCLUSIVE,因为没有任何已执行的实验可供"无法区分")
- **明确事实**:
  - 当前无正式 causal / lower-bound experiment;
  - **CPI_abs 的理论下界为 0(由定义 E|δ|≥0 直接给出);尚未证明任何
    正下界(no positive lower bound has been established)**;
  - 当前 empirical saturation ≠ theoretical lower bound;
  - 现有数据(CE 持续改善而 CPI 平台/微反弹,后者仅 descriptive)只构成
    下一阶段研究动机。
- **纪律**:**CE-adjusted independence 若未来被实现,不得自动重开 frozen
  Case A**(Case A 裁定文件与 gate verdict 均已冻结;任何重开需用户显式裁定)。
- **是否保留**:作为下一阶段研究问题。

### Reconciliation:C-06 / C-07 / C-18 三者不矛盾(关系澄清)

> Relationship between C-06 and C-18:This does not contradict the Case-A
> adjudication. C-06 establishes that CE improvement is not a sufficient
> monotonic explanation across late-stage / cross-family dynamics. C-18 states
> only that, within the canonical early CPI-active window, no independent
> structural transition was found that explains the attenuation better than
> generic conditional-estimation improvement.
> (中文:二者不矛盾。C-06 表明 CE improvement 不能作为整个训练过程、跨
> checkpoint family 的充分单调解释;C-18 只表示在 canonical SFT 的 early
> CPI-active window 内,目前没有发现一个比 generic conditional-estimation
> improvement 更有独立证据支持的结构 transition。)

- C-07(旧 BF16 早期强正相关被取代)与 C-18 的关系:C-07 是测量层的取代
  (BF16 → FP32/dense),不是对早期共变现象的否定;dense FP32 轨迹显示早期
  窗口 CE 与 |δ| 同向改善(与 Case A 一致),后期二者脱钩(与 C-06 一致)。
- 必须强调:Case A = most parsimonious explanation among current candidates;
  不是 global monotonic CE→CPI law;也不是 causal proof。

### C-20 本机 SFT 训练 tensor-exact 可复现
- **分类**:VALIDATED_OBSERVATION
- **证据**:Gate-2 链(p1@1020 == formal@1020 逐字节,`scripts/compare_checkpoints.py`
  定案);Option B s2 重跑 10 checkpoint EMA 130/130 + model 131/131 torch.equal、
  worst_rel=0(`results/phase2/optionb_s2_repro_gate.json`);跨日跨 commit 的独立
  run loss 曲线逐位一致。
- **局限**:本机(GPU/driver/torch 2.14)证据;不承诺跨机 bitwise。
- **是否保留**:是。

### C-21 执行事实与审计结论(dropout / LR / 步数 / provenance)
- **分类**:VALIDATED_OBSERVATION(provenance 事实)
- **证据**:
  - **dropout 实际 0.1**(全部 SFT/mechpilot/v21/optionb run):hydra 合并语义下
    small.yaml 0.1 覆盖 vanilla_256.yaml 的 0.0 override(实测 compose 验证);
    记录于 v2.1 execution manifest `deviations_recorded[1]` 与 errata;
    **注意:对 v4.1/v4.2 vanilla/formal/p1/optionb 同样成立,但仅 v2.1 errata
    事后披露**。
  - **v1.2 LR 实际 3e-4**(文本 3e-5):8 个 run train.log step2500 lr=3.00e-04
    实证;用户裁定按实际执行口径,`protocol/mechanism_v2_1_execution_manifest.json`。
  - **formal 实际 10201 optimizer steps**(n_iters=10200 + 1 收尾步),
    `reports/audit_optimizer_step_count_10201.md` 定案;checkpoint_10200 恰为
    10200 次 update 后状态,不重跑。
  - **EMA decay 报告错误更正**:0.9990→0.99912 为误,实际 n=1 时 0.1818。
  - **84dc4b4 时代评估代码未提交**:历史数值不可 bit-reproduce(aggregate
    diff CPI ~2e-3、OG ~8e-3–2e-2;`protocol/errata_v4.2_eval_provenance.md`)。
- **是否保留**:是(所有历史结论引用的前提条件)。

### C-22 机械 Case B verdict(已作废的脚本输出)
- **分类**:SUPERSEDED_MEASUREMENT
- **证据**:机械输出从未入库(首次入库即 B_candidate_unadjudicated);其存在由
  `93c8e44`/`41d73cb` commit message 与 MEMORY.md 佐证;人工终裁 = Case A。
- **是否保留**:仅作修订史记录。

## 3. 专项审计(用户点名项)

### 审计 1:BF16 vs true FP32 / harmonized metrics
- 旧 BF16 系(2026-10-07 前主链):eval_cpi/eval_task/eval_order_gap,forward 内
  bf16 autocast → 数值落 BF16 网格,CE 量化 0.0156–0.031 nats。
- 新 FP32 系:true_fp32_mirrored_forward(eval_diag_fp32 冻结 sha ef1ef03e…、
  pairs 冻结 sha 22d80c65…),聚合 fp64。
- Harmonization:precision ladder(B-C 工程等价 gate;step0 FAIL 0.00328 →
  H1-2500 PASS 0.00115)+ bridge(BRIDGE-A:定性稳健)。
- **引用规则**:compatibility 结论方向可用任何一代;精细数值一律 FP32。

### 审计 2:v1.2/v2.1 LR 和 dropout 配置偏差
- 见 C-21。要点:两处偏差均**有书面 provenance、经用户裁定、未静默改写旧文本**;
  确定性未受损(dropout 流有 seed,复现门逐位通过)。
- 引用纪律:写"实际执行 lr=3e-4(dropout=0.1)",并注明 protocol 文本为 3e-5(0)。

### 审计 3:frozen Stage-4 B_DIAGNOSTIC_ONLY
- 定义:`protocol/regime_a_protocol.yaml` L98–101;判决:`results/vanilla/
  stage4_gate.json`;永久固定:`protocol/experiment_protocol_v4.2.md` L3–4。
- 含义:ΔCPI 稳定且 ΔOG 稳定,但 early→late co-movement 不成立 → 现象级结论,
  机制声明弱化。**不是**"CPI 与 OrderGap 无关"的证明。
- 后续所有阶段(v4.2/v1.2/v2.1/Phase1/2)均继承此口径。

### 审计 4:P1 dense / FP32 bridge
- P1 dense(EXP-05):84 个 BF16 评估定位衰减窗口 250–750;中间权重已删
  (仅 2500 full-state 幸存)——这是 Option B 重训的动因。
- Bridge(EXP-13):15 checkpoint 成对;BRIDGE-A 判定历史结论定性稳健;
  计数披露 11 vs 15 已入 manifest。

### 审计 5:Fresh vs Fixed / directional tests(v1.2)
- A vs B 是 **the primary matched intervention designed for causal
  discrimination**(设计有 causal-discrimination intent,但最终 gate
  INCONCLUSIVE,不能表述为 causal identification 已成立);G1a/G1b =
  INCONCLUSIVE(方向相反);C/D 是 directional stress test,含位置暴露混杂,
  不得单独因果;G2 = INCONCLUSIVE(1020 FAIL 定义为 early transient)。
  结论:**两条干预路线均未获得机制支持(INCONCLUSIVE ≠ proven ineffective)**。

### 审计 6:H/U/L complementary exposure(v2.1)
- 干预实现正确(G3a PASS 8/8:exact-K、E_comp 严格序 0.48>0.33>0.04、map hash);
- 但 CPI 无策略间差异(G3c INC,两 replicate 反向)→ Case C;
- 保留事实:所有变体仍相对 pretrained 衰减;INCONCLUSIVE ≠ 无效。

### 审计 7:RL-500 与 K-ablation(RL 是独立分支,不是 SFT mechanism test)
- RL-500(EXP-10):Observed outcome = Scenario B(该设定下无可检测的
  CPI/OG 变化,16 evals CI 全含 0);hypotheses predicting substantial further
  change 在该精确设定下未获支持,**不构成** "RL does not affect compatibility
  in general";**This is not an equivalence test and does not establish a zero
  effect**;分类 SUPPORTED_BUT_NOT_CAUSAL;dxg deviation 已记录。
- K-ablation(EXP-11):K4≈K1(100 步诊断)→ INCONCLUSIVE;RAM gate 修订记录在案。
- 结构注意:RL branch 问的是 "After SFT, does reward-based post-training
  further alter compatibility?",与 RQ1 的 SFT 机制探索(Fresh/Fixed、
  Directional、Complementary Exposure)是不同的问题,不要混称"三条机制路线"。

### 审计 8:Phase 2A predictor analysis
- coarse 分辨率 Case C:CE 与 |δ| **反相**(两个窗口方向相反,与"CE 驱动 |δ|"
  的简单假设矛盾);JS_dep 的 grouped-CV 预测力(M0 0.2142→M1 0.1368)是
  descriptive,协议明确"CV 增益≠因果"。
- **SUPERSEDED FOR EARLY-WINDOW ADJUDICATION BY EXP-19**:raw descriptive 结果
  与 grouped-CV 结果保留;coarse 下的 Case C 不再作为当前 early-window
  verdict;当前 active-window 裁决以 EXP-19 dense trajectory 为准。
- 该阶段的教训:幸存 checkpoint 分辨率不足以裁决早期窗口 → 触发 Option B。

### 审计 9:Option B deterministic retrain / Case A adjudication
- 复现:双 gate PASS(tensor-exact;worst_rel=0;指标 diff 0.0)。
- 轨迹:dense 13 点 × 2 seeds;机械 verdict 合规化(B_candidate_unadjudicated);
- 人工终裁:**Case A(限定 early CPI attenuation)**;D_ab = replicated but
  non-independent correlate(ρ=−1);JS_dep = modest/reproducible but
  timing-mismatched;2500 rebound 仅 descriptive。
- 引用纪律(用户 F + 作用域修订):"Generic conditional-estimation improvement
  is the current most parsimonious explanation for the early SFT-associated CPI
  attenuation; causal independence remains untested, and this explanation does
  not by itself account for the later compatibility saturation / residual gap."

## 4. Evidence Coverage Audit(覆盖缺口与风险)

### 4.1 覆盖良好(每个 claim 均有 SHA 可验证的 artifact)
- C-01/C-04/C-05:C-01 有 3 代独立证据(BF16 gate、FP32 bulk、dense)+ BRIDGE。
- C-03:dense trajectory + Gate-2 复现链。
- C-20:两个独立 gate(bitwise 级)。

### 4.2 覆盖缺口(引用前须知)
1. **exp_local/ 不入 git**:权重与 train.log 只在磁盘;git 里无权重。若磁盘丢失,
   EXP-03/05/06/07/08/10/11/17 的权重将不可恢复(结果 JSON 仍在)。
2. **p1-s1/s2 中间 checkpoint 已删**(50…1020):EXP-05 的中间权重缺失;由
   EXP-17 重建(tensor-exact 复现已证明,见 C-20),但"原始文件"不存在。
3. **84dc4b4 时代评估代码未提交**:历史 formal 数值不可 bit-reproduce;harmonized
   主链以当前代码为准(errata 已披露)。
4. **早期阶段无逐事件 git 粒度**:2026-10-01~02 的 pilot/formal/Stage-4/P1 dense
   压缩在 `de69bb9` 一个 commit;时间线依赖 MEMORY.md。
5. **机械 Case B 输出从未入库**:首次入库即修正版;修订链靠 commit message 佐证。
6. **750–1020 窗口无 dense 内点**;1020–1500 只有两个点。
7. **v21smoke-U1-221648** 无任何训练数据(3 行 cfg)。
8. **RL acceptance 输出在 /tmp**(不在 results/ 体系)。
9. **`4f855b5` 孤儿 commit**(RL plan FROZEN 曾改写,主线为 bc479ac)。
10. **Option B s2 事故目录已删**(215621/230404/230455):事故证据 = 已保留的
    failed_run 日志 + MEMORY.md + adjudication §8。

### 4.3 建议(不自动执行)
- 如要长期保存权重:为 exp_local 生成 manifest(文件名+SHA)入库,权重本体留在
  磁盘/外部存储。
- 如要回答 RQ2:需要先设计"positive lower bound"的预注册研究(当前 NOT_EXECUTED)。
- 如需补 750–1020 内点:只能通过新 dense schedule(当前禁跑,需用户裁定)。

## 5. 修订史汇总(25 条,详见 git 审计;最重要 8 条见 START_HERE §7)
