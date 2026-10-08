# ASSET INDEX — EXP-ID → 资产路径总索引

> **This document is an asset-location index, not an independent scientific
> authority.** 科学结论、数值、Case 裁定一律以
> `docs/CLAIM_EVIDENCE_MAP.md`(结论分级)与 `docs/EXPERIMENT_REGISTRY.md`
> (实验档案)为权威;本文件只回答"证据文件在哪里"。
> 所有相对路径从 repository root 解析。状态 REVIEWED(2026-10-08,导航层审查)。

## 1. EXP-01～EXP-19 资产映射

### EXP-01 环境适配 + 256 训练 smoke
- Primary entrypoint:`train.py` / `run_train.py`(上游训练入口)
- Eval entrypoint:`eval_ppl.py`(1000-t MC PPL 协议)
- Result artifact:MEMORY.md Day 2–3 段(**无独立原始报告**)
- Report / verdict:smoke 5 条验收 PASS(MEMORY.md 记录)
- Local checkpoint:无(已清)
- Claim-IDs:C-20 背景
- Asset status:HISTORICAL(工程校准)
- Provenance limitations:原始 smoke 输出未落盘为独立报告;batch 语义早期记录错误已修正

### EXP-02 LR 探针 + vanilla pilot N 定案
- Primary entrypoint:`training/vanilla.py`(n_iters=30000 探针)
- Eval entrypoint:`eval_cpi.py`(旧 harness 口径)
- Result artifacts:`reports/pilot_report/data/lrprobe_{1e-4,3e-4,3e-5}_learning_curve.csv`、
  `reports/pilot_report/data/{pilot_learning_curve,pilot_run_metadata}.csv/.json`、
  `exp_local/regime_a/pilot_console.log`(截断于 step 2600)、`results/pilot/`
- Report / verdict:LR=3e-5 定案、N=10200(`protocol/regime_a_protocol.yaml` L40–62)
- Local checkpoint:无(探针已清)
- Claim-IDs:C-21 背景
- Asset status:HISTORICAL(LR 定案的唯一原始曲线)
- Provenance limitations:pilot_console.log 仅 110 行截断;lrprobe CSV 为最完整证据

### EXP-03 formal vanilla SFT ×2
- Primary entrypoint:`training/vanilla.py`
- Eval entrypoint:`evaluation/eval_cpi.py` · `eval_task.py` · `eval_order_gap.py`
  (驱动:`scripts/run_formal_evals.sh`)
- Result artifacts:`results/vanilla/{cpi,og,g1}_s{1,2}_{1020,5100,10200}.json`
- Report / verdict:`reports/stage4_report/report.md`;Stage-4 见 EXP-04
- Local checkpoints:`exp_local/regime_a/formal-vanilla-s1-191414/`、
  `formal-vanilla-s2-204215/`(full-state 1020/5100/10200 + meta)
- Claim-IDs:C-01 C-02 C-04 C-05
- Asset status:ACTIVE_EVIDENCE(结果)/ 权重无备份
- Provenance limitations:dropout 实际 0.1(文本 0);实际 10201 步(审计定案);
  84dc4b4 时代评估代码未提交

### EXP-04 冻结基线 + Stage gate 链
- Primary entrypoint:`evaluation/stage4_gate.py` · `evaluation/check_state_support.py`
- Result artifacts:`results/vanilla/stage4_gate.json`、
  `g05_state_support{,_distributional}.json`、`results/pretrained/{cpi,order_gap,g1_task,residual_time}.json`
- Report / verdict:B_DIAGNOSTIC_ONLY(永久冻结)
- Local checkpoint:无
- Claim-IDs:C-05 C-15
- Asset status:ACTIVE_EVIDENCE(frozen)
- Provenance limitations:BF16 口径;FP32 对应值见 EXP-14

### EXP-05 P1 dense canonical ×2
- Primary entrypoint:`training/vanilla.py`(v4.2 exact_N)
- Eval entrypoint:`eval_cpi.py` 等(驱动:`scripts/run_p1_evals.sh`)
- Result artifacts:`results/p1_dense/`(84 文件,BF16)
- Report / verdict:`reports/p1_early_dynamics_analysis.md`;Gate-2 链逐字节
  (`scripts/compare_checkpoints.py`)
- Local checkpoints:`exp_local/regime_a/p1-s1-020240/`、`p1-s2-110355/`
  (**仅 checkpoint_2500 幸存**;中间 50–1020 已删)
- Claim-IDs:C-03 C-07 C-20
- Asset status:ACTIVE_EVIDENCE + SUPERSEDED_MEASUREMENT(BF16 数值部分)
- Provenance limitations:中间权重缺失(由 EXP-17 tensor-exact 重建)

### EXP-06 v1.2 Stage P1(A vs B)
- Primary entrypoint:`training/pilot_mechanism.py`(驱动:`scripts/run_p1_mech.sh`)
- Eval entrypoint:`eval_cpi.py` · `eval_task.py`
- Result artifacts:`results/mechanism_pilot/`(A/B 12 文件)、`p1_stage1_analysis.json`、
  `p1_g1b_analysis.json`
- Report / verdict:`reports/mechanism_pilot_v1_2_report.md`;G1 = INCONCLUSIVE
- Local checkpoints:`exp_local/regime_a/mechpilot-{A1,A2,B1,B2}-*/`(EMA 500/1020/2500 + raw 2500)
- Claim-IDs:C-08
- Asset status:ACTIVE_EVIDENCE(INCONCLUSIVE 档案)
- Provenance limitations:LR 实际 3e-4(文本 3e-5);dropout 实际 0.1

### EXP-07 v1.2 Stage P2(C vs D)
- Primary entrypoint:`training/pilot_mechanism.py`(驱动:`scripts/run_p2_mech.sh`)
- Result artifacts:`results/mechanism_pilot/`(C/D 12 文件)、`p2_g2_analysis.json`
- Report / verdict:v1.2 report;G2 = INCONCLUSIVE(1020 FAIL 定义为 early transient)
- Local checkpoints:`exp_local/regime_a/mechpilot-{C1,C2,D1,D2}-*/`
- Claim-IDs:C-09
- Asset status:ACTIVE_EVIDENCE
- Provenance limitations:同 EXP-06;D1 首试被 Terminated(resume 后完成)

### EXP-08 v2.1 H/U/L ×2
- Primary entrypoint:`training/pilot_v21.py`(驱动:`scripts/run_v21_pilot.sh`)
- Eval entrypoint:`evaluation/eval_cpi_pairs.py`(treated/heldout)、`eval_order_gap_subset.py`
  (驱动:`scripts/run_v21_evals.sh`)
- Result artifacts:`results/mechanism_pilot_v21/`(76 文件)、`v21_g3_analysis.json`
- Report / verdict:`reports/mechanism_pilot_v2_1_report.md`;G3a PASS / G3b-c INC → Case C
- Local checkpoints:`exp_local/regime_a/v21pilot-{U,H,L}{1,2}-*/`;
  **frozen maps:`exp_local/regime_a/mechpilot_v21_maps/`(见 §4)**
- Claim-IDs:C-10
- Asset status:ACTIVE_EVIDENCE + maps = **BACKUP_PRIORITY_1 / RUNTIME_DEPENDENCY**
- Provenance limitations:BF16 量化(G3d 解释已 errata 化)

### EXP-09 RL 前置
- Primary entrypoint:`scripts/rl_g0_gate.py` · `scripts/rl_128_vs_1024_gate.py`
- Result artifacts:`results/rl_gates/{rl_g0,128_vs_1024}.json`
- Report / verdict:RL-G0 PASS
- Local checkpoint:无
- Claim-IDs:C-21 背景
- Asset status:ACTIVE_EVIDENCE
- Provenance limitations:128vs1024 为 analytic,非训练对比

### EXP-10 RL-500 pilot
- Primary entrypoint:`training/rl.py`
- Eval entrypoint:`eval_cpi.py` · `eval_order_gap.py`(驱动:`scripts/run_rl_pilot_postevals.sh`)
- Result artifacts:`results/rl_pilot/`(16 evals)
- Report / verdict:`reports/final_findings_and_lessons.md`(RL 阶段历史快照);Scenario B
- Local checkpoints:`exp_local/regime_a/rlpilot-185545/`(2 full-state + 4 eval snapshots)
- Claim-IDs:C-11
- Asset status:ACTIVE_EVIDENCE(Scenario B 档案)
- Provenance limitations:dxg EOVERFLOW deviation #1;单 seed 设定

### EXP-11 RL K-ablation
- Primary entrypoint:`scripts/run_k_ablation.py`
- Eval entrypoint:`eval_cpi.py` · `eval_order_gap.py`
- Result artifacts:`results/rl_k_ablation/`(7 文件)
- Report / verdict:`reports/rl_k_ablation.md`;K4≈K1
- Local checkpoints:`exp_local/regime_a/rl-k-ablation/{k1-100,k4-100,smoke-k4}/`
- Claim-IDs:C-12
- Asset status:ACTIVE_EVIDENCE
- Provenance limitations:100 步诊断;RAM gate amendment #1

### EXP-12 precision ladder
- Primary entrypoint:`scripts/phase1_precision_ladder.py`
- Eval entrypoint:`evaluation/eval_diag_fp32.py`(frozen sha `ef1ef03e…`)
- Result artifacts:`results/phase1_preflight/{precision_ladder_report,ladder_subset_indices}.json`
- Report / verdict:Level B-C 工程等价 gate:step0 FAIL / H1-2500 PASS → Level C 定案
- Local checkpoint:无
- Claim-IDs:C-13 C-14
- Asset status:ACTIVE_EVIDENCE
- Provenance limitations:gate 为工程容差,非统计等价

### EXP-13 bridge BF16↔FP32
- Primary entrypoint:`scripts/phase1_bridge_analysis.py`(驱动:`scripts/run_phase1_bridge.sh`)
- Result artifacts:`results/phase1_bridge/`(bridge_manifest + old/new 成对 30 + bridge_analysis.json + scatter.png)
- Report / verdict:`reports/phase1_bridge_verdict.md`;BRIDGE-A
- Local checkpoint:无(复用既有权重)
- Claim-IDs:C-13
- Asset status:ACTIVE_EVIDENCE
- Provenance limitations:manifest 计数披露 11 vs 15

### EXP-14 bulk FP32 dynamics(55×3)
- Primary entrypoint:`scripts/phase1_bulk_combined.py`
- Eval entrypoint:`evaluation/eval_diag_fp32.py` · `_aux.py` · `_ordergap.py`
- Result artifacts:`results/phase1_diag/{core,aux,ordergap}/`(165)+
  `bulk_manifest.json` + `phase1_dynamics_analysis.json`
- Report / verdict:VERDICT = P1-B
- Local checkpoint:无(复用 55 个既有 checkpoint)
- Claim-IDs:C-06
- Asset status:ACTIVE_EVIDENCE
- Provenance limitations:core/ 曾被裸 `core` gitignore 规则误伤(已修复,Phase 1A)

### EXP-15 Phase 2 preflight
- Primary entrypoint:`scripts/phase2_preflight_smoke.py`(GPU smoke)
- Result artifacts:`results/phase2/preflight_pairs.json`、`preflight_smoke.json`
- Report / verdict:工具层 PASS(46/46 单测 + smoke)
- Local checkpoint:无
- Claim-IDs:C-16 C-17 工具层
- Asset status:ACTIVE_EVIDENCE(工具)
- Provenance limitations:smoke 用 16 ladder 样本

### EXP-16 Phase 2A pairs(55×500)
- Primary entrypoint:`scripts/run_phase2a_pairs.py` + `scripts/phase2a_analysis.py`
- Eval entrypoint:`evaluation/eval_diag_fp32_pairs.py`(frozen sha `22d80c65…`)
- Result artifacts:`results/phase2/pairs/`(55)、`phase2a_analysis.json`
- Report / verdict:coarse Case C —— **SUPERSEDED FOR EARLY-WINDOW ADJUDICATION BY EXP-19**
- Local checkpoint:无
- Claim-IDs:C-16 背景
- Asset status:HISTORICAL(原始数据保留)+ SUPERSEDED(verdict 部分)
- Provenance limitations:coarse 分辨率

### EXP-17 Option B dense retrain s1/s2
- Primary entrypoint:`scripts/optionb_dense_retrain.py` /
  `scripts/optionb_dense_retrain_safe.py`(atomic save + disk guard)
- Result artifacts:train.log / learning_curve.csv / run_metadata.json(exp_local)
- Report / verdict:无独立报告(见 EXP-18/19)
- Local checkpoints:`exp_local/regime_a/optionb-dense-s1-213551/`、
  `optionb-dense-s2-230610/`(dense 13 点;anchor=model+ema,其余 EMA-only;
  含 failed_run_*.log 事故记录)
- Claim-IDs:C-03 C-20
- Asset status:ACTIVE_EVIDENCE(**Case A 裁定的唯一 dense 权重载体**)
- Provenance limitations:事故目录(213031/215621/230404/230455)已删,日志保留

### EXP-18 Option B reproduction gates
- Primary entrypoint:`scripts/optionb_s2_repro_gate.py`(tensor equality 优先)、
  `scripts/optionb_repro_gate.py`(bitwise + 指标)
- Result artifacts:`results/phase2/optionb_s2_repro_gate.json`(PASS,worst_rel=0)、
  `optionb_repro_gate.json`(PASS,指标 diff 全 0.0)
- Report / verdict:双 gate PASS
- Local checkpoint:无(只读引用)
- Claim-IDs:C-20
- Asset status:ACTIVE_EVIDENCE
- Provenance limitations:SHA256 仅辅助(序列化层差异已实证)

### EXP-19 dense pairs + trajectory + 终裁
- Primary entrypoint:`scripts/run_optionb_dense_pairs.py` +
  `scripts/optionb_dense_analysis.py`
- Eval entrypoint:`evaluation/eval_diag_fp32_pairs.py`
- Result artifacts:`results/phase2/dense_pairs/`(26)、`optionb_dense_analysis.json`
  (B_candidate_unadjudicated)、**`optionb_case_adjudication.md`(FINAL Case A)**
- Report / verdict:人工终裁 Case A(限定 early attenuation)
- Local checkpoint:无(只读引用 EXP-17 权重)
- Claim-IDs:C-03 C-16 C-17 C-18
  (**C-19 关联仅为 descriptive motivation**:EXP-19 的 saturation/rebound 观测
  是 RQ2 的研究动机来源;**RQ2 本身状态 = NOT_EXECUTED**,EXP-19 不是 RQ2 的
  正式机制验证,见 claim map C-19)
- Asset status:ACTIVE_EVIDENCE
- Provenance limitations:750–1020 窗口无内点;机械 Case B 输出从未入库

## 2. results/ 目录分类(指针)

| 目录 | 分类 |
|---|---|
| `vanilla/` `pretrained/` `p1_dense/` `mechanism_pilot/` `mechanism_pilot_v21/` `rl_gates/` `rl_pilot/` `rl_k_ablation/` `phase1_diag/` `phase1_bridge/` `phase1_preflight/` `phase2/` | ACTIVE_EVIDENCE |
| `mechanism_pilot_dryrun/dryrun_summary.json`、`mechanism_pilot_v21_dryrun/dryrun_summary.json` | **RUNTIME_DEPENDENCY(MUST_KEEP)**——训练结束守卫硬读 |
| `mechanism_pilot_v21_dryrun/diagnostics_*.json` ×6 | HISTORICAL_EVIDENCE(暂保留) |
| `pilot/` | SUPERSEDED_BUT_PRESERVED |
| `phase2/phase2a_analysis.json` 的 verdict 部分 | SUPERSEDED(raw 数据保留) |

## 3. scripts/ 状态摘要(指针)

完整清单与引用关系见 `docs/EXPERIMENT_REGISTRY.md` 各 EXP 条目;本索引不重复。
原则:所有脚本保留原路径——包括 refs=0 的一次性执行器,它们是复现证据。

## 4. Frozen v2.1 maps — BACKUP_PRIORITY_1 / RUNTIME_DEPENDENCY

- 路径:`exp_local/regime_a/mechpilot_v21_maps/`(共 10 文件,≈1.6MB;
  **已纳入 git(2026-10-08 Phase 2C)——exp_local 的唯一例外**,带
  `.gitattributes` byte-exact 保护)
- 内容:maps_rep1/rep2/heldout.json + order_gap_subset_indices.json + maps_manifest.json
  + 5 个 .sha256 sidecar
- **完整性验证(2026-10-08 实测)**:4 个数据文件 + manifest 的 SHA256 与各自
  sidecar 全部 MATCH;manifest 内 file_sha256 与实测一致;manifest sha
  `4ee8c94f…` 与 v21pilot train.log 记录一致
- Runtime paths:
  - 读:`evaluation/eval_cpi_pairs.py:177`(--maps_dir 默认)、`training/pilot_v21.py:244`
  - 写:`scripts/v21_generate_maps.py:34`(OUT_DIR 硬编码)
  - 结论:移动该目录会破坏 v2.1 treated/heldout 评估复现 → **必须保持原路径**
- 备份方案比较:

| | 方案 A:git add -f 入库 | 方案 B:外部校验备份 |
|---|---|---|
| 操作 | 原路径保留,force-add 10 文件(需 .gitignore 豁免 `mechpilot_v21_maps/`) | 原路径保留,复制到外部盘 + 逐文件 SHA 清单入库 |
| 优点 | 版本化、随 clone 分发、diff 可见(JSON 文本可 diff) | 零 git 增量、不动 .gitignore |
| 风险 | git 增量 ≈1.6MB(可忽略);需豁免规则维护 | 外部副本无版本化;损坏检测靠人工比对 |
| 预计 git 增量 | ≈1.6MB(10 文件) | 0(仅清单文件 ~1KB) |
| 建议 | **首选**(体积极小、JSON 文本、与 git 哲学一致) | 作为 A 之外的冗余第二副本 |

## 5. exp_local 权重盘点(统一 GiB 口径,2026-10-08 实测并复核)

总计 **106.83 GiB(114.70 GB decimal)**(分项复核:mechpilot 20.22 +
formal 20.22 + optionb 21.49 + v21pilot 15.19 + p1 10.11 + rlpilot 10.11 +
其余 9.48 = 106.83 ✓)。此前 "107G" 为 du -sh 的十进制 GB 约数。
除 frozen maps 已纳入 git(见 §4)外,无任何权重有独立备份;各实验保留了相应
训练配置、代码和历史记录,原则上可尝试重新训练;但只有部分 checkpoint 通过了
指定范围内的复现验证,不能保证所有模型逐位重建(验证级别见下表)。SHA256 全量
成本 ≈10–15 分钟(仅磁盘读)。

**Replay 验证四级标签**(防止把局部 gate 外推为全量 replay):
① `tensor-exact` = checkpoint 张量逐位复现(仅覆盖实际比对的 checkpoint 范围);
② `matched gate verification` = 预注册守卫逐位通过(digest/E_comp/C̄ 等);
③ `metric-level agreement` = loss/指标逐位一致(不含权重逐位);
④ `only theoretically reproducible` = 未执行任何 replay 验证。

| 资产 | 大小 | 备份 | replay 验证级别(范围限定) | 保留优先级 |
|---|---|---|---|---|
| optionb dense ×2 | 10.74 GiB×2 | ❌ | ① tensor-exact(仅共享的 10 个 ckpt 50–750 + 500 model;1020/1500/2500 无旧副本可比)+ ③ curve 逐位 | **P1(终裁唯一载体)** |
| formal ×2 | 10.11 GiB×2 | ❌ | ② 锚点 bitwise(1020 EMA 经 official gate)+ ③ curve 逐位;5100/10200 未 replay | P1 |
| p1 ×2 | 5.06 GiB×2 | ❌ | ② 锚点 bitwise(2500 EMA+model 经 official gate)+ ③ curve 逐位;中间 ckpt 已删无法 replay | P1 |
| rlpilot | 10.11 GiB | ❌ | ④ only theoretically reproducible | P2 |
| mechpilot ×8 | 2.53 GiB×8 = 20.22 | ❌ | ④ only theoretically reproducible | P2 |
| v21pilot ×6 | 2.53 GiB×6 = 15.19 | ❌ | ② matched gate verification(结束守卫 digest/E_comp/C̄ vs dryrun 逐位)+ ④ 权重本身未 replay | P2 |
| k-ablation | 2.53 GiB | ❌ | ④ only theoretically reproducible | P3 |
| v21smoke ×4 | 1.90 GiB×3 + 8KB = 5.7 | ❌ | ② matched gate verification(smoke digest= dryrun prefix) | P4(归档候选) |
| mechpilot_shared(step0 锚点) | 1.26 GiB | ❌ | ④ 可由 load_model 重建(未 replay) | P4 |
| frozen maps | 1.6MB | **git ✓(2026-10-08 Phase 2C,byte-exact)** | ② SHA 实测全 MATCH(sidecar+manifest+train.log 三方) | **P1(见 §4)** |

## 6. 纪律声明

- 本索引**不复制**任何 CPI/CE/JS_dep/D_ab 数值、Case 裁定与研究假设解释;
  需要结论时回到 `docs/CLAIM_EVIDENCE_MAP.md`(权威)与
  `docs/EXPERIMENT_REGISTRY.md`(实验档案)。
- 路径若在本索引与 registry 间冲突,以 registry 的 Artifacts 字段为准并报告。
