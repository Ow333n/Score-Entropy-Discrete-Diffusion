# EXPERIMENT REGISTRY — SEDD Order Compatibility 全实验登记册

> 状态:**REVIEWED(2026-10-08,证据体系与一致性修订已经用户审查)**。
> REVIEWED 仅指导航层 / 证据体系的组织通过审查,**不意味着任何底层科研假设获得
> 证明**;各实验的结论分类见本文件第 0 节。
> 依据:exp_local/regime_a/ 全部 28 个 run 目录的 train.log / run_metadata.json /
> learning_curve.csv / checkpoint 实测 + 93 个 git commit + protocol/errata 原文。
> 每条 recipe 均为**日志实证**(不是 protocol 文本);protocol-vs-实际偏差在
> `docs/CLAIM_EVIDENCE_MAP.md` §专项审计 2 汇总。

## 0. 唯一实验编号与结论分类

- 实验用稳定 ID `EXP-01`…`EXP-19`(不再用 Case A/B/C 之类模糊名称指代实验;
  Case 字样只出现在"该实验的 gate verdict"字段)。
- 结论分类(全项目统一):
  `VALIDATED_OBSERVATION` / `SUPPORTED_BUT_NOT_CAUSAL` / `INCONCLUSIVE` /
  `UNSUPPORTED_HYPOTHESIS` / `SUPERSEDED_MEASUREMENT` / `INVALID_OR_VOIDED_RUN` /
  `NOT_EXECUTED`。
- 通用冻结背景:model=louaaron/sedd-small(169,627,218 参数)、wikitext103
  256-chunk、seq 256、span [10,50]、batch 32 eff(accum 1)、warmup 2500 linear、
  AdamW(0.9,0.999,1e-8)、wd=0、grad_clip 1.0、ema 0.9999、评估 manifest
  500 样本(sha `1897bd14…`)。

## 1. 快速索引

| ID | 实验 | 一句话结论 | 分类 |
|---|---|---|---|
| EXP-01 | 环境适配 + 训练 smoke | 本机可跑;PPL 40.19(1000-t) | VALIDATED_OBSERVATION |
| EXP-02 | LR 探针 + vanilla pilot | LR 3e-5 定案(3e-4 发散);N=10200 | VALIDATED_OBSERVATION |
| EXP-03 | formal vanilla SFT ×2 | 完成;10201 步审计定案 | VALIDATED_OBSERVATION |
| EXP-04 | 冻结基线 + Stage gate 链 | Stage-4 = **B_DIAGNOSTIC_ONLY**(永久) | VALIDATED_OBSERVATION |
| EXP-05 | P1 dense canonical ×2 | CPI 衰减定位 250–750;Gate-2 逐字节 | VALIDATED_OBSERVATION(+部分 SUPERSEDED_MEASUREMENT) |
| EXP-06 | v1.2 Stage P1(A vs B) | fresh/fixed 无方向差异 | INCONCLUSIVE |
| EXP-07 | v1.2 Stage P2(C vs D) | directional 无稳定差异 | INCONCLUSIVE |
| EXP-08 | v2.1 H/U/L ×2 | 干预有效(G3a PASS)但 CPI 无差异 → Case C | INCONCLUSIVE(机制)+ VALIDATED_OBSERVATION(干预) |
| EXP-09 | RL 前置(plan/G0/128vs1024) | RL-G0 PASS;lr=3e-6 定案 | VALIDATED_OBSERVATION |
| EXP-10 | RL-500 pilot | Scenario B:该设定下无可检测的 CPI/OG 变化(非"RL 无效") | SUPPORTED_BUT_NOT_CAUSAL |
| EXP-11 | RL K-ablation | K4≈K1;无证据 K 是瓶颈 | INCONCLUSIVE |
| EXP-12 | Phase 1 precision ladder | B-C 工程等价 gate:step0 FAIL → Level C FP32 定案 | VALIDATED_OBSERVATION |
| EXP-13 | Phase 1 bridge(15 ckpt) | **BRIDGE-A**:BF16 历史结论定性稳健 | VALIDATED_OBSERVATION |
| EXP-14 | Phase 1 bulk FP32(55×3) | **P1-B**:late-stage CE 不解释 CPI 衰减 | SUPPORTED_BUT_NOT_CAUSAL |
| EXP-15 | Phase 2 preflight | 工具层全 PASS(dependence/decoder/CV) | VALIDATED_OBSERVATION |
| EXP-16 | Phase 2A pairs(55×500) | coarse 分辨率 Case C;raw/grouped-CV 保留,early-window 裁决已由 EXP-19 supersede | SUPERSEDED_FOR_EARLY_WINDOW(原始数据保留) |
| EXP-17 | Option B dense retrain s1/s2 | 恢复 dense 轨迹;s2 经磁盘事故后 tensor-exact 重跑 | VALIDATED_OBSERVATION |
| EXP-18 | Option B reproduction gates | 双 gate **PASS**(worst_rel=0 / 指标 diff 0.0) | VALIDATED_OBSERVATION |
| EXP-19 | dense pairs + trajectory + 终裁 | 机械 B → **人工 Case A(限定 early attenuation)**(non-independent D_ab correlate) | SUPPORTED_BUT_NOT_CAUSAL |

---

## 2. 实验档案

### EXP-01 环境适配 + 256-seq 训练 smoke(2026-09-30,commit `84dc4b4`)

- **RQ**:本机(RTX 5060 Ti 8G / torch 2.14+cu130)能否跑通 SEDD 训练与评估;复现上游 PPL。
- **实际 recipe(日志实证)**:smoke batch 4 / accum 2(B_eff 语义审计:eff=4,早期
  "eff 8"为记录错误)、300 步、flash-attn 缺失走 SDPA fallback、rotary 重写。
- **Seeds**:model=0/data=0/corruption=0(校准用)。
- **Evaluator**:早期 harness(BF16;1-t PPL 与 1000-t MC 两版)。
- **主要发现**:300 步 5 条验收 PASS;16-block PPL **40.19**(1000-t MC 定口径;
  1-t 值 38.5 被取代);loss 全程 finite。
- **Gate verdict**:smoke 验收 PASS。
- **Confounds/偏差**:batch 语义记录错误(已修正,`scripts/check_batch_semantics.py`)。
- **Artifacts**:`exp_local/regime_a/pilot_console.log`;MEMORY.md Day 2–3 段。
- **分类**:VALIDATED_OBSERVATION(工程校准级)。**保留**:是(环境事实)。

### EXP-02 LR 探针 + vanilla pilot N 定案(2026-10-01,压缩于 `de69bb9`)

- **RQ**:选稳定 LR;用 plateau 检测确定正式训练预算 N。
- **实际 recipe**:n_iters=30000 探针(pilot-132632,lr=3e-4,发散记录在案)→
  1e-4(摇摆)→ **3e-5 定案**;plateau K=6/eps 0.005/alpha 1.2 → N_hat。
- **主要发现**:3e-4 发散、1e-4 摇摆、3e-5 稳定 → 冻结 base_lr=3e-5、N=10200。
- **Artifacts**:`protocol/regime_a_protocol.yaml`(L40–62)、`exp_local/regime_a/pilot_console.log`。
- **分类**:VALIDATED_OBSERVATION。**保留**:是。

### EXP-03 formal vanilla SFT ×2 seeds(2026-10-01,`de69bb9`)

- **RQ**:partial-reveal span-infilling SFT 是否让模型"学会"任务(G1)并产生
  compatibility 变化(G2/Stage-4 输入)。
- **实际 recipe(日志实证)**:batch 32/accum 1/ngpus 1、**n_iters=10200**、
  **lr=3e-5**、warmup 2500、ema 0.9999、dropout **实际 0.1**(文本 0,见专项审计 2);
  实际 optimizer steps **10201**(多 1 步收尾,审计定案:checkpoint_10200 恰为
  10200 次 update 后状态,不重跑)。
- **Seeds**:s1=(1,1,1)、s2=(2,2,2)。运行时:4474s / 5175s。
- **Evaluator**:BF16(eval_cpi/eval_task/eval_order_gap,protocol v4.1,manifest 1897bd14…);
  checkpoints 1020/5100/10200 full-state(2.53GiB/个)。
- **主要发现**:G1 PASS(masked NLL 4.281→3.875、acc 0.322→0.362);G2 轨迹
  (CPI 0.328→early 0.287/0.279→late 0.289/0.283;OG 10.24→8.6/8.8→8.76/8.82)。
- **Gate verdict**:G0.5 PASS;G1 PASS;Stage-4 见 EXP-04。
- **Confounds/偏差**:训练期 eval 腐蚀种子噪声(交叉评估证明,非模型差异);
  EMA decay 报告错误(更正 0.1818@n=1);10201 步。
- **Artifacts**:`exp_local/regime_a/formal-vanilla-s1-191414/`、`-s2-204215/`;
  `results/vanilla/{cpi,og,g1}_s*_*.json`、`g1_pilot.json`。
- **分类**:VALIDATED_OBSERVATION。**保留**:是(全部主链结论的载体)。

### EXP-04 冻结基线 + Stage gate 链(2026-10-01~02,`de69bb9`)

- **RQ**:pretrained 基线下,SFT 后的 CPI/OG 变化是否满足进入机制研究的条件
  (A_HARD_STOP / B_DIAGNOSTIC_ONLY / C_PROCEED)。
- **评估口径**:per-sample paired Δ|δ| 与 ΔOG,per-seed bootstrap 95% CI(10k),
  两 seed 同号;co-movement 判据 sign(ΔCPI)==sign(ΔOG) 且 CI 均排除 0。
- **主要发现**:pretrained CPI=0.328±0.023(BF16)、OG_raw=10.2445;s1 Δ|δ|=−0.0402
  CI[−0.0722,−0.0066]、ΔOG=−1.4888 CI[−2.1225,−0.9357];s2 Δ|δ|=−0.0461
  CI[−0.0779,−0.0143]、ΔOG=−1.4270;**两 seed cpi_stable/og_stable=true 但
  co_movement=false**。
- **Gate verdict**:**B_DIAGNOSTIC_ONLY**(`results/vanilla/stage4_gate.json`;
  v4.2 协议 L3–4 宣告"永久固定,不是对判决的上诉")。
- **Artifacts**:`results/vanilla/stage4_gate.json`、`g05_state_support*.json`、
  `results/pretrained/{cpi,order_gap,g1_task,residual_time}.json`、
  `protocol/experiment_protocol_v4.2.md`。
- **分类**:VALIDATED_OBSERVATION(gate verdict 本身)。**保留**:是(冻结)。

### EXP-05 P1 dense canonical s1/s2(2026-10-02,`de69bb9`)

- **RQ**:用密集 checkpoint 定位 CPI/OG 衰减发生在训练的哪个阶段。
- **实际 recipe(日志实证)**:n_iters=2500 exact_N、lr=3e-5、save_at
  [50,100,250,500,750,1020,2500]、dropout 实际 0.1。
- **Seeds**:s1=(1,1,1)、s2=(2,2,2)。运行时 1231s / 1138s。
- **主要发现**:**连续衰减定位 250–750**(s1 (250,500)、s2 (500,750));OrderGap
  0→50 已启动;84 个 BF16 评估 JSON(results/p1_dense/)。
- **Gate verdict**:Gate-2 链 = p1@1020 与 formal@1020 **逐字节一致**(s1/s2 均
  PASS;`scripts/compare_checkpoints.py`,max rel<1e-5 阈)。
- **Confounds**:中间 checkpoint(50…1020)磁盘已删,仅 2500 full-state 幸存;
  其 BF16 CE 数值后被 FP32 取代(SUPERSEDED_MEASUREMENT 部分)。
- **Artifacts**:`exp_local/regime_a/p1-s1-020240/`、`p1-s2-110355/`;
  `results/p1_dense/`(84 文件)。
- **分类**:VALIDATED_OBSERVATION(轨迹定位)+ SUPERSEDED_MEASUREMENT(旧 BF16 数值)。
  **保留**:是;权重缺失部分由 EXP-17 重建。

### EXP-06 mechanism pilot v1.2 Stage P1(A vs B)(2026-10-06,`0f8bcc2`)

- **RQ**:fresh random resampling(A)vs fixed permutation(B)——每次重采样 vs
  固定 mask 模式是否改变 CPI 学习?(设计意图:the primary matched intervention
  designed for causal discrimination;注:设计有 causal-discrimination intent,
  但最终 gate INCONCLUSIVE,不能表述为 causal identification 已成立)
- **实际 recipe(日志实证)**:batch 32、2500 步、**lr 实际 3e-4**(文本 3e-5,
  用户裁定按实际口径)、dropout 实际 0.1;共享 (sample,span,σ,K) 流;stream
  seeds 1000+r…7000+r;save 500/1020/2500 EMA-only + 2500 raw。
- **Seeds**:rep1 streams 1001/2001/3001/4001/5001/6001/7001;rep2 各 +1。
  每 run ~1065s。
- **主要发现**:step2500 ΔCPI(A−B):rep1 −0.0059 CI[−0.0382,0.0235]、rep2
  +0.0078 CI[−0.0211,0.0373] → **方向相反**;step0 两 replicate d_cpi=0;
  G1b matched-performance(±0.02 NLL)INCONCLUSIVE。
- **Gate verdict**:G1a/G1b = INCONCLUSIVE(机制未决,非 FAIL)。
- **Artifacts**:`exp_local/regime_a/mechpilot-A{1,2}-*`、`mechpilot-B{1,2}-*`;
  `results/mechanism_pilot/p1_stage1_analysis.json`、`p1_g1b_analysis.json`。
- **分类**:INCONCLUSIVE。**保留**:是(对照组的必要档案)。

### EXP-07 mechanism pilot v1.2 Stage P2(C vs D)(2026-10-06,`9cc93c5`)

- **RQ**:directional stress test(L2R vs R2L)——揭示方向是否影响 CPI?
  (协议明示 C/D 含 positional exposure confound,不得单独用于机制因果)
- **实际 recipe**:同 EXP-06(C/D 确定性选择、不耗 policy RNG)。
- **主要发现**:G2 primary(step2500)=INCONCLUSIVE;step1020=FAIL(定义为 early
  transient,不升格);v1.2 报告冻结,结论矩阵"机制未决"。
- **Gate verdict**:G2 INCONCLUSIVE。
- **Artifacts**:`exp_local/regime_a/mechpilot-C{1,2}-*`、`mechpilot-D{1,2}-*`;
  `results/mechanism_pilot/p2_g2_analysis.json`;
  `reports/mechanism_pilot_v1_2_report.md`。
- **分类**:INCONCLUSIVE。**保留**:是。

### EXP-08 mechanism pilot v2.1 H/U/L(2026-10-06~07,`f9d9c3b`/`23de20e`)

- **RQ**:complementary exposure 程度(同 pair 内恰一个被 mask 的比例)是否调制
  CPI?(H=互补最大化 E_comp≈0.48 / U=均匀≈0.33 / L=共掩≈0.04)
- **实际 recipe(日志实证)**:lr=0.0003(manifest+errata 裁定)、warmup 2500、
  batch 32、2500 步、dropout 0.1;frozen pair maps(matching/cycle/heldout
  skip-2);E_comp 实测 U 0.3325/H 0.4800/L 0.0397(与闭式一致)。
- **Seeds**:共享流 1000+r…4000+r;policy 流 8001+r…8501+r。每 run ~1110s。
- **主要发现**:G3a **PASS 8/8**(干预实现有效:exact-K、marginal parity、
  E_comp 严格 H>U>L、分布可区分、map hash);G3b/G3c INCONCLUSIVE(G3c H vs L
  pooled +0.01078 CI[−0.02154,0.04417],两 replicate 方向相反);G3d 在评估器
  分辨率内 matched;fixed_eval@2500 masked NLL 三 policy 均 ~3.18–3.27(任务
  都学会了)。
- **Gate verdict**:**Case C**(官方措辞保留事实:"All H/U/L variants still
  exhibit CPI attenuation relative to pretrained";禁写"已被证明无效")。
- **Confounds**:BF16 量化(G3d 的 ±0.02 容差低于量化粒度 0.0156–0.031 →
  "同桶即匹配"的解读已 errata 化);LR/dropout 偏差(manifest 记录)。
- **Artifacts**:`exp_local/regime_a/v21pilot-{U,H,L}{1,2}-*`、
  `mechpilot_v21_maps/`;`results/mechanism_pilot_v21/v21_g3_analysis.json`(76
  评估 JSON);`protocol/mechanism_complementary_exposure_v2_1.md` + errata +
  execution manifest;`reports/mechanism_pilot_v2_1_report.md`。
- **分类**:INCONCLUSIVE(机制假说)+ VALIDATED_OBSERVATION(干预有效性与衰减事实)。
  **保留**:是。

### EXP-09 RL 前置:plan v1.0 冻结 + RL-G0 + 128vs1024(2026-10-02~03)

- **RQ**:RL 管线的正确性门(G0)与 D-step 长度选择(128 vs 1024)。
- **主要发现**:RL-G0 **PASS**(loader shadow/model mismatch=0、step=10200;
  复跑 CPI/G1/OG 与 vanilla s1-10200 全字段 diff<1e-9);128vs1024 analytic
  (acc/reward 0.2035 vs 0.2192,残存 MASK=0,n_neg=0);lr=3e-6 定案
  (lrprobe);"staggered_score 负权重 blocker"澄清为**非 blocker**。
- **Artifacts**:`results/rl_gates/{rl_g0,128_vs_1024}.json`;
  `post_training_rl_plan_v1.0_FROZEN.md`;`protocol/errata_v1.0.md`。
- **分类**:VALIDATED_OBSERVATION。**保留**:是。

### EXP-10 RL-500 pilot(2026-10-03~04,`5be2fdf`)

- **RQ**:在 SFT 收敛点上做 500 步 RL(σ-safe-region 奖励)是否会进一步改变
  CPI/OrderGap?
- **实际 recipe(日志实证)**:lr=3e-6、n_steps=500、G=4/P=4/K=1、rollout_steps=128、
  pool=64、init=formal-s1-10200 EMA、chunk=2、eval_points [0,50,100,250,500];
  14131s(~3.9h);final reward 0.2889±0.1110、zvg 117/2000、nan_rollouts 0。
- **主要发现**:16 个 post evals(4 snapshots × raw/ema × CPI/OG)**全部 CI 含 0**
  → **Observed outcome:Scenario B — no detectable compatibility change under
  the current setting**。
- **Hypothesis status**:预测"substantial further compatibility change"的假设在
  **该精确设定下**未获支持;这**不**构成 "RL does not affect compatibility in
  general"。**This is not an equivalence test and does not establish a zero
  effect.**(observed outcome 与 inferential status 分离。)禁用措辞:"RL 无效"、
  "RL 不改变 CPI"、"RL hypothesis failed"(除非加 under this exact setting /
  horizon / reward / seed 限定)。
- **Confounds**:dxg EOVERFLOW 瞬时异常 27 条(~112s,step 410–450),无 OOM/NaN/
  训练不连续(Deviation #1 记录在案)。
- **Artifacts**:`exp_local/regime_a/rlpilot-185545/`;`results/rl_pilot/`(16 evals);
  `protocol/deviation_formal_rlpilot_dxg.md`。
- **分类**:SUPPORTED_BUT_NOT_CAUSAL(观测结果 = 该设定下近似稳定的
  compatibility;因果外推未建立)。**保留**:是。

### EXP-11 RL K-ablation(2026-10-04,`a32290d`)

- **RQ**:K(gradient accumulation 步数)=1 vs 4 是否改变 RL 学习(诊断性)。
- **实际 recipe(日志实证)**:100 步、lr=3e-6、G=4/P=4/rollout 128、j_rng 隔离
  (seed=rl.seed+step×10000+chunk_offset)、K 从 safe support 无放回采样。
- **主要发现**:final reward K1 0.2691±0.1942 vs K4 0.2732±0.2011;drift
  ~1.6e-4;zvg 同(26/400);→ **K4≈K1,无证据 K 是主要瓶颈**。
- **Confounds**:preflight 曾因 RAM gate 物理不可满足 STOP(amendment #1 修订
  gate:MemAvailable≥5GiB 等,机器适配)。
- **Artifacts**:`exp_local/regime_a/rl-k-ablation/{k1-100,k4-100,smoke-k4}`;
  `results/rl_k_ablation/comparison.json`;`protocol/rl_k_ablation_protocol.md` +
  `rl_k_ablation_preflight_stop.md`。
- **分类**:INCONCLUSIVE(100 步诊断;K 效应未排除)。**保留**:是。

### EXP-12 Phase 1 precision ladder preflight(2026-10-07,`809e7dc`/`7e1167c`)

- **RQ**:三种精度口径(A=frozen BF16 全链、B=BF16 forward+FP32 提取、C=true-FP32
  mirrored forward)哪个能作为 bulk dynamics 的标准?
- **主要发现**:B-C 工程等价 gate(预注册:CPI 差<0.003、CE 差<0.01):step0
  d_cpi=0.00328 → **FAIL**(禁放宽)→ H1-2500 d_cpi=0.00115 → PASS;
  Level B/C 的 NLL off-bf16-grid=1.0(量化消除);A 保真 diff=0.0。
  → **Level C(true-FP32 mirrored forward)定案**,B 只作工程对照。
- **Artifacts**:`results/phase1_preflight/precision_ladder_report.json`、
  `ladder_subset_indices.json`(sha f03515fb…);`evaluation/eval_diag_fp32.py`
  (冻结 sha ef1ef03e…);`scripts/test_diag_fp32.py`。
- **分类**:VALIDATED_OBSERVATION。**保留**:是。

### EXP-13 Phase 1 bridge BF16↔FP32(2026-10-07,`9ccc7e8`)

- **RQ**:历史 BF16 结论在 FP32 口径下是否定性稳健?
- **设计**:15 checkpoint 成对(old BF16 evaluator vs new FP32 diag);manifest
  冻结(sha 986dcf1d…);计数披露 11 vs 15(起草失误,按不增删枚举冻结 15)。
- **主要发现**:**BRIDGE-A**:pretrained→SFT 方向 14/14 保存、CPI Spearman 0.9821、
  per-sample |δ| Pearson≥0.9、CE 排名变化未达阈值 → **历史 compatibility 结论
  定性稳健;后续精细 dynamics 用 FP32**。
- **Artifacts**:`results/phase1_bridge/bridge_analysis.json`、`bridge_manifest.json`、
  old_*/new_* 成对 30 文件、`bridge_scatter.png`。
- **分类**:VALIDATED_OBSERVATION。**保留**:是。

### EXP-14 Phase 1 bulk FP32 dynamics(2026-10-07,`ce14103`)

- **RQ**:跨 55 checkpoint 的 FP32 动力学——CE 改善能否解释 CPI 衰减?(P1-A/B/C)
- **设计**:55 checkpoint × {core,ordergap,aux} = 165 文件;bulk manifest 冻结
  (sha 3b9701…);within-run centered ΔCE vs ΔCPI 为 primary。
- **主要发现**:FP32 基线 CPI 0.3242→SFT finals 0.2766(rel **−14.7%**、RMS
  −9.3%、q50 −27.8%);OG 12.2559→10.24–11.41;pooled Spearman(ΔCE,ΔCPI)=
  **−0.105**、run-clustered slope CI[−0.618,−0.029];旧 BF16 p1_dense 的强正
  相关不成立 → **VERDICT=P1-B:单纯 conditional estimation improvement 不足以
  解释(late-stage)CPI 衰减**。
- **Gate verdict**:P1-B(措辞已 nuance 化:"只表述 late-stage CE improvement
  不与 CPI attenuation 稳定共变",不写"完全无关/机制被排除")。
- **Artifacts**:`results/phase1_diag/{core,ordergap,aux}/`(165)、
  `phase1_dynamics_analysis.json`、`bulk_manifest.json`。
- **分类**:SUPPORTED_BUT_NOT_CAUSAL(相关性结论)。**保留**:是。

### EXP-15 Phase 2 preflight(2026-10-07,`7ca921c`)

- **RQ**:dependence 度量(JS_dep/D_ab/PMI)、block decoder、grouped CV 是否正确
  实现并可运行?
- **主要发现**:46/46 单测;smoke 全过(block1 recovery 0.26–0.28、NFE
  accounting ok、JS_dep 0.0416、VRAM 3.94G);grouped CV 泄漏双向断言。
- **Artifacts**:`results/phase2/preflight_pairs.json`、`preflight_smoke.json`;
  `compatibility/dependence.py`;`evaluation/phase2_block_decoder.py`;
  `scripts/phase2_grouped_cv.py`。
- **分类**:VALIDATED_OBSERVATION(工具层)。**保留**:是。

### EXP-16 Phase 2A pairs diagnostics(2026-10-07,`fb79783`)

- **RQ**:55 checkpoint × 500 pairs 的 FP32 per-sample 结构量(δ/|δ|/JS_dep/D_ab/
  PMI/CE)——早期窗口有没有独立结构信号?
- **主要发现**:**CE 与 |δ| 反相**(500→1020:CE 改善(−0.033,100% runs 同向)但
  |δ| 不动(+0.0098,仅 7% 下降);1020→2500:CE 恶化(+0.092,0% 改善)但 |δ| 下降
  (−0.0089,71% 同向));checkpoint 级 corr(|δ|,CE)=−0.332、corr(|δ|,JS_dep)=
  +0.466、偏相关 +0.322;grouped CV M0(CE)0.2142→M1(CE+JS)0.1368(CV 增益≠因果);
  D_ab 全平 ≈+0.2。
- **Gate verdict**:**Case C(coarse 分辨率)**:决定性 250–1020 窗口在幸存
  checkpoint(500/1020/2500)下不可分辨 → 触发 Option B dense 重训。
- **Artifacts**:`results/phase2/pairs/`(55)、`phase2a_analysis.json`。
- **分类**:**SUPERSEDED FOR EARLY-WINDOW ADJUDICATION BY EXP-19**。
  含义:raw descriptive 结果与 grouped-CV descriptive 结果**全部保留**;coarse
  分辨率下的 Case C **不再作为当前 early-window verdict**;当前 active-window
  裁决以 EXP-19(dense trajectory)为准。**保留**:是。

### EXP-17 Option B dense retrain s1/s2(2026-10-07,`1a4446e`/`55f26db`)

- **RQ**:恢复 P1 canonical 两个 seed 在 250–1020 窗口的 dense 可分析权重。
- **实际 recipe(日志实证)**:与 EXP-05 完全一致(n_iters=2500 exact_N、lr=3e-5、
  batch 32、同一 vanilla.py 循环;seed tuple 见下);dense save_at 13 点;anchor
  (500/1020/2500)存 model+ema、其余 EMA-only;s2 使用 atomic-save + 磁盘 guard
  wrapper(训练数学零改动)。
- **Seeds(seed tuple,run_metadata 实证)**:
  s1 = (model=1, data=1, corruption=1);
  s2 = (model=2, data=2, corruption=2)。
  运行时 1216s / 1126s,峰值 6.40GB。
- **事件链(s2)**:首跑 215621 因宿主 D 盘写满死于 step~1020(事故日志
  failed_run_215621.log 保留)→ 首启 230404 因 CUDA OOM 碎片化死于 step 2 →
  加 expandable_segments(纯分配器,数学零改动)后 230610 完成。
- **主要发现**:s1/s2 全 13 点 dense checkpoint 产出;loss 曲线与 frozen P1
  逐位一致。
- **Artifacts**:`exp_local/regime_a/optionb-dense-s1-213551/`、
  `optionb-dense-s2-230610/`(含 save_sizes.csv、failed_run 日志)。
- **分类**:VALIDATED_OBSERVATION(复现)+ INVALID_OR_VOIDED_RUN(215621/230404,
  provenance 已保留)。**保留**:是。

### EXP-18 Option B reproduction gates(2026-10-07~08,`d2e8b7c`/`41d73cb`)

- **RQ**:重训轨迹是否 tensor-exact 复现原轨迹?(决定 dense 轨迹能否代表原 run)
- **主要发现**:s2 deterministic gate:10 个共享 checkpoint(50–750)EMA
  shadow_params **130/130 torch.equal**、step500 model **131/131**、step/decay/
  num_updates 全一致、43 条重叠 loss 日志逐位一致、worst_max_rel=**0**;
  official gate:s1/s2@2500 EMA+model 与 frozen p1 逐位一致、@1020 与 formal
  EMA 一致、4 组指标 diff(cpi/rms/ce/acc)全 **0.0**;文件级 SHA256 不同已实证
  为 torch.save 序列化层差异(tensor 内容全等)。
- **Gate verdict**:双 gate **PASS**。
- **Artifacts**:`results/phase2/optionb_s2_repro_gate.json`、
  `optionb_repro_gate.json`;`scripts/optionb_s2_repro_gate.py`(优先级:
  tensor equality > loss 逐位 > SHA256 辅助)。
- **分类**:VALIDATED_OBSERVATION。**保留**:是。

### EXP-19 dense pairs + trajectory + 最终裁定(2026-10-08,`93c8e44`/`41d73cb`)

- **RQ**:在 dense 分辨率下,CPI active window 内是否存在独立结构 transition?
  (预注册 Case A/B/C)
- **设计**:26 = 2 seeds × 13 dense steps,FP32 pairs(evaluator sha 22d80c65…);
  窗口 0–250/250–750(active)/750–1020/1020–2500。
- **主要发现**:|δ| active window −9.61%/−10.42%;CPI reaches its lowest observed
  mean around 1020–1500 and shows a small descriptive rebound by step 2500
  (**the rebound itself has not been established as a statistically reliable
  reversal**;中文:CPI 的最低观测均值出现在约 1020–1500 步,2500 步时均值出现
  小幅 descriptive rebound;目前尚未证明该 rebound 是统计上可靠的 reversal);
  CE 同窗 −0.145/−0.149 且全程单调改善;JS_dep 全程 −6.27%/−6.04%(modest but
  reproducible,但 0–250 已占全程下降 ~48%/45% → timing mismatch,无独立
  transition);D_ab 两 seed 复现性上升(+0.022/+0.023)但**与 CE per-seed
  Spearman ρ=−1 完全共单调**、1020 后继续上升不 plateau。
- **Gate verdict**:机械脚本曾输出正式 Case B → 用户裁定 independence 判据
  (CE-adjusted)从未实现 → 脚本改为只输出 **B_candidate_unadjudicated**
  (case_b_full_criteria_met=false,missing_criterion=independence_from_ce)→
  人工终裁 **FINAL Case A(作用域限定为 early CPI attenuation)**:
  "Generic conditional-estimation improvement is the current most parsimonious
  explanation for the early SFT-associated CPI attenuation; causal independence
  remains untested, and this explanation does not by itself account for the
  later compatibility saturation / residual gap."
  中文:generic conditional-estimation improvement 是当前对 SFT 前期 CPI 衰减
  最简洁的解释;其因果独立性尚未验证,而且该解释本身不能解释后期 CE 持续改善
  但 CPI 不再继续下降的 compatibility saturation / residual gap。
  D_ab 定位:"a robust correlate of training progress / compatibility improvement
  whose independence from conditional-estimation improvement is not established"
  (**不称 mechanism**)。
- **Artifacts**:`results/phase2/dense_pairs/`(26)、`optionb_dense_analysis.json`、
  `optionb_case_adjudication.md`。
- **分类**:SUPPORTED_BUT_NOT_CAUSAL(最简解释,非因果证明)。**保留**:是(最终
  裁定文件)。

---

## 3. NOT_EXECUTED 清单(计划过、从未执行)

| 项目 | 出处 | 状态 |
|---|---|---|
| Regime B:PAPL / Swap / PAPL+Swap / 2×2 matrix | v4.1 协议 §15A/§29 | NOT_EXECUTED(Stage-4=C_PROCEED 才进入;B_DIAGNOSTIC_ONLY 未触发) |
| Phase 2B:block decoding bulk(55×4 block×3 选择×500)、random×3 全解码 | phase2 协议 | NOT_EXECUTED(等 Case 裁定;Case A 后仍未执行) |
| Phase 2C causal interventions | phase2 协议 §8 | NOT_EXECUTED(Case B 才允许;最终 Case A) |
| v2.1 Confirmatory Stage:扩展到 **≥4 total paired replicates**(基于 pilot effect size / CI 做 power analysis 后定总数;pilot 本身即 6 runs = H/U/L × 2 paired replicates,非"6-run 扩大") | v2.1 协议 §15.1 | NOT_EXECUTED(进入条件 = G3a PASS + H vs L 两 replicate 同向 + effect size 有研究意义 + G3d 匹配;G3 结果 Case C,条件未满足,不进入) |
| v2.1 dose-response λ-interpolation(H/U 混合,§15.2)与 deferred 大型扩展(full-mask / AR SFT / 多规模 / 多 K / 5 seeds,§15.3) | v2.1 协议 §15.2/§15.3 | NOT_EXECUTED(仅 Future Confirmatory Stage 定义) |
| RQ2 正式研究:finite-step SFT 下的 residual compatibility gap 与 CE–CPI decoupling | 用户 2026-10-08 定义 | NOT_EXECUTED(无正式 causal / lower-bound 实验;**CPI_abs 理论下界为 0(定义),尚未证明正下界**;当前 empirical saturation ≠ theoretical lower bound,现有数据只构成下一阶段研究动机;CE-adjusted independence 不得自动重开 frozen Case A) |
| RL 128/1024 的正式训练对比(非 analytic) | RL plan | NOT_EXECUTED(仅 analytic gate) |
| Regime B protocol 文件(regime_b_protocol.yaml) | v4.1 §15A | 未创建 |

## 4. INVALID_OR_VOIDED_RUN 清单(保留 provenance)

| Run | 原因 | 证据保留 |
|---|---|---|
| optionb-dense-s2-215621 | D 盘写满,step~1020 中断 | failed_run_215621.log/curve 已复制进新 s2 目录;原目录已删(用户批准) |
| optionb-dense-s2-230404 | CUDA OOM 碎片化,step 2 死亡 | 已删(用户批准);事件记录于 MEMORY.md 与 adjudication §8 |
| optionb-dense-s2-230455 | 前台测试 25s 超时终止 | 已删(用户批准) |
| optionb-dense-s1-213031 | 与完整 s1 重复的失败 run(已验证逐字节同源) | failed_run_213031.log 在 s1 目录内 |
| v21smoke-U1-221648 | 启动中止(仅 3 行 cfg) | train.log 仍在原目录 |
| pilot-132632(LR 探针 3e-4) | 按设计"发散"探针,非事故 | pilot_console.log |
| mechp2(D1 首试) | 训练中被 Terminated | mechp2_driver.log;resume 后完成 |

## 5. 全 run 完成状态矩阵(28 个磁盘目录,计数已核对)

**28 = 24 个完成的单目录 run**(formal×2、p1×2、optionb×2、mechpilot×8、
v21pilot×6、v21smoke 完成×3、rlpilot×1)+ **1 个目录含 3 个已完成 sub-run**
(rl-k-ablation:k1-100 / k4-100 / smoke-k4)+ **1 个中断**(v21smoke-U1-221648)
+ **2 个辅助**(mechpilot_shared、mechpilot_v21_maps)。

计数差异说明(此前"24+1+2+2=29"的由来):①"失败后重试成功 2"(optionb
213031→213551、215621→230610)与"完成 24"中的 optionb×2 **重叠**——失败
目录已被用户批准删除,只以 failed_run_*.log 保留 provenance,不计入磁盘
目录数;②rl-k-ablation 此前未单列(其 3 个 sub-run 已全部完成)。
逐 run 的 train.log cfg 行与 seeds 行是"实际配置"的唯一权威来源
(registry 已转录)。
