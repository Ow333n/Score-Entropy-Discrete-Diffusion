# START HERE — SEDD Post-Training Order Compatibility 项目导航

> 状态：**DRAFT(待用户审查,2026-10-08)**。本文件是项目证据体系的入口;
> 生成依据:4 个只读审计(protocol/gates、全部 28 个实际 run、evaluation/results、
> git 93 commit 修订史)。**本文件不修改任何 frozen 结论**——它只建立
> "结论 → 实验 → 证据文件"的可追溯链条。

---

## 1. 项目一句话

在 partial-reveal(span 内逐步揭示)SFT 下,研究 **Order Compatibility(CPI / OrderGap)**
为什么会变化:训练改变了什么,哪些机制假设成立、哪些不成立。

## 2. 当前科学状态(最终前沿,2026-10-08)

**研究树(全项目统一口径,与 REGISTRY/CLAIM_MAP 一致):**

- **Main empirical phenomenon**:partial-reveal SFT reduces reveal-order
  incompatibility / sensitivity(CPI↓、OrderGap↓,两 seed 复现)。
- **RQ1:Why does CPI decrease during early SFT?**
  当前回答 = **Case A(作用域限定为 early CPI attenuation)**:
  > "Generic conditional-estimation improvement is the current most parsimonious
  > explanation for the early SFT-associated CPI attenuation; causal independence
  > remains untested, and this explanation does not by itself account for the
  > later compatibility saturation / residual gap."
  > (中文:generic conditional-estimation improvement 是当前对 SFT 前期 CPI 衰减
  > 最简洁的解释;其因果独立性尚未验证,而且该解释本身不能解释后期 CE 持续改善
  > 但 CPI 不再继续下降的 compatibility saturation / residual gap。)
  候选解释状态:Fresh vs Fixed = INCONCLUSIVE;Directional exposure =
  INCONCLUSIVE;Complementary exposure = 干预实现有效(G3a PASS)、机制
  INCONCLUSIVE;JS_dep = descriptive / timing mismatch;D_ab = replicated
  correlate、非独立 mechanism。
- **RQ2:Why does CPI stop improving / remain non-zero while CE continues
  improving?** 研究对象 = **finite-step SFT 下的 residual compatibility gap
  与 CE–CPI decoupling**。分类 = **NOT_EXECUTED**(未正式执行任何 causal /
  lower-bound 实验;**CPI_abs 理论下界为 0(由定义),尚未证明正下界**;当前
  empirical saturation ≠ theoretical lower bound,只是下一阶段研究动机;
  CE-adjusted independence 若未来实现,**不得自动重开 frozen Case A**)。
- **RL branch(独立的扩展问题,不属于 RQ1 机制树)**:post-SFT reward
  optimization 是否进一步改变 compatibility?当前 500-step 结果:在测试的
  单 seed / 500 步 / 特定奖励设定下无可检测变化(Scenario B;**not an
  equivalence test,does not establish a zero effect**)。

**伴生结论**:D_ab 是 replicated but non-independent correlate(与 CE per-seed
Spearman ρ = −1),不称 mechanism;JS_dep 是 modest/reproducible 但 timing
mismatch 的 descriptive 变化。

**冻结禁区**:不启动新 seed / 新 hand-crafted metric / Phase 2C / 新 intervention /
block decoding bulk / PAPL / Swap / PAPL+Swap / 2×2 matrix(v1.2/v2.1/Phase2
各协议与 Option B adjudication §9 的联合裁定)。

## 3. 你的 8 个问题 → 去哪里查

| 问题 | 答案位置 |
|---|---|
| 1. 我到底做过哪些实验? | `docs/EXPERIMENT_REGISTRY.md`(EXP-01…EXP-19 + NOT_EXECUTED 清单) |
| 2. 每个实验原本想验证什么? | Registry 每条 `RQ / Hypothesis` 字段 |
| 3. 实际训练配置(不读 protocol)? | Registry 每条 `实际执行 recipe(日志实证)` 字段;偏差清单见 `docs/CLAIM_EVIDENCE_MAP.md` §专项审计 2 |
| 4. 得到了什么结果? | Registry 每条 `主要发现` + `Artifacts`;数值摘要在 `docs/CLAIM_EVIDENCE_MAP.md` |
| 5. 哪些结论真正成立? | Claim map 中 `VALIDATED_OBSERVATION` 类;最短清单见 §6 |
| 6. 哪些假设未获支持/不确定? | Claim map 中 `INCONCLUSIVE` / `UNSUPPORTED_HYPOTHESIS` 类 |
| 7. 哪些历史结论被修正? | Claim map 每条 `修订史` 字段;完整 25 条修订清单见 §7 |
| 8. 哪些可用于论文/面试? | Claim map `是否保留` 字段;§6 最可靠发现清单 |

## 4. 文档与资产地图

```
docs/
  START_HERE.md             ← 你在这里
  EXPERIMENT_REGISTRY.md    ← 19 个实验的完整档案(唯一 EXP-ID)
  CLAIM_EVIDENCE_MAP.md     ← 结论分级 + 专项审计 + evidence coverage audit
protocol/                   ← 全部 frozen 协议/errata/manifest(22 文件,sha256 sidecar 锁定)
reports/                    ← stage4 报告、v1.2/v2.1 报告、审计报告(定案口径)
results/                    ← 全部评估 JSON 与 gate 判定(入库、可 git 追溯)
exp_local/regime_a/         ← 28 个实际训练 run(不入 git,权重/日志在磁盘)
manifests/regime_a_eval_v1.jsonl  ← 冻结评估 manifest(500 样本,sha 1897bd14…)
MEMORY.md                   ← 项目逐日任务记录(审计时间线的主要来源之一)
```

## 5. 证据层级与结论分类(全项目统一口径)

```
VALIDATED_OBSERVATION       复现过、gate PASS、跨 evaluator 代际稳健
SUPPORTED_BUT_NOT_CAUSAL    观察到的关联/效应,机制因果未建立
INCONCLUSIVE                gate 无法区分(≠ 证明无效)
UNSUPPORTED_HYPOTHESIS      预注册检验未获支持(如 CI 含 0、方向反向)
SUPERSEDED_MEASUREMENT      被新 evaluator / dense 分辨率取代的旧测量
INVALID_OR_VOIDED_RUN       失败/作废的 run(保留 provenance 记录)
NOT_EXECUTED                计划过但从未执行
```

三条纪律:
1. **技术 gate PASS ≠ 研究假设成立**(例:G3a PASS 只说明干预实现有效)。
2. **复现成功 ≠ 机制得到证明**(例:Option B 双 gate PASS 只保证 trajectory
   可信,不保证任何因果解释)。
3. **INCONCLUSIVE ≠ 证明无效**(例:v2.1 Case C 的官方措辞保留
   "All H/U/L variants still exhibit CPI attenuation relative to pretrained",
   禁止写成"机制已被证明无效")。

## 6. 发现总览(三类,30 秒可读)

### A. Scientific Findings

1. Partial-reveal SFT robustly reduces CPI relative to pretrained(两 seed paired
   95% CI 排除 0:s1 −0.0402 / s2 −0.0461;FP32 rel −14.7%、RMS −9.3%、q50 −27.8%)。
2. SFT also reduces OrderGap(BF16 ΔOG CI 排除 0;FP32 finite-path 方向一致)。
3. SFT improves masked conditional estimation / task performance(G1:masked NLL
   4.281→3.875、token acc 0.322→0.362)。
4. CPI attenuation is concentrated in an early training window(mainly 250–750,
   −9.6%/−10.4%,VALIDATED_OBSERVATION),followed by later saturation /
   near-bottom behavior(supported descriptive trajectory);step-2500 rebound =
   **descriptive only**(未建立统计可靠的 reversal)。
5. Stage-4 verdict B_DIAGNOSTIC_ONLY(永久冻结):CPI 与 OrderGap 各自稳定下降但
   late-stage co-movement 不成立——compatibility 变化是现象,机制声明弱化。

### B. Mechanism Status

1. Fresh vs Fixed(v1.2 A/B):INCONCLUSIVE。
2. Directional exposure(v1.2 C/D):INCONCLUSIVE。
3. Complementary exposure(v2.1 H/U/L):intervention implementation validated
   (G3a PASS 8/8),mechanism effect INCONCLUSIVE(Case C)。
4. Current explanation for **early** attenuation:generic conditional-estimation
   improvement is the most parsimonious current explanation(作用域限定为 early
   CPI attenuation),but causal independence is untested;该解释不覆盖后期
   saturation / residual gap。
5. RQ2(residual gap / non-zero saturation):**NOT_EXECUTED**。
6. RL extension(独立分支,非 SFT mechanism test):under the current single-seed /
   500-step / specific-reward setting,no detectable further CPI / OrderGap
   change(Scenario B;**not an equivalence test,does not establish a zero
   effect**)。

### C. Evidence Integrity

1. BF16 → FP32 bridge preserves main qualitative findings(BRIDGE-A:direction
   14/14、CPI Spearman 0.9821)。
2. Option B deterministic reproduction passes tensor-exact gates(s2:EMA 130/130 +
   model 131/131 torch.equal,worst_rel=0;official:bitwise 6/6 + 指标 diff 0.0)。
3. LR(3e-5 文本 vs 3e-4 实际)与 dropout(0 文本 vs 0.1 实际)等执行偏差已显式
   文档化(execution manifest + errata);84dc4b4 评估代码 provenance 断裂已披露。
4. Historical BF16 values and current FP32 values must not be mixed in one
   numerical trajectory(BF16 CE 量化粒度见 CLAIM_EVIDENCE_MAP C-14)。

## 7. 25 条历史修订(结论被修正/取代的完整清单)

`docs/CLAIM_EVIDENCE_MAP.md` §修订史汇总 + git 审计。最重要的 8 条:
1. pretrained 基线口径:旧随机对(CPI 0.0393)→ v4.1 冻结 manifest(0.328)→ FP32(0.3242)
2. PPL 估计器:1-t(38.5)→ 1000-t MC(40.19)
3. EMA decay 报告错误 0.9990→0.99912 → 更正 n=1 时 0.1818
4. 10201 optimizer steps 审计定案(checkpoint_10200 恰为 10200 次 update 后状态)
5. **LR 偏差**:v1.2 文本 3e-5 → 实际 3e-4(用户裁定按实际执行口径,manifest 记录)
6. **dropout 偏差**:全部 run 文本 0 → 实际 0.1(hydra 合并语义;v2.1 errata 披露)
7. 旧 BF16 p1_dense 的 ce_leads 早期共变信号 → 被 FP32 dense trajectory 取代
8. Phase 2A coarse Case C → Option B dense 机械 Case B → **人工裁定 Case A
   (作用域限定为 early CPI attenuation)**(independence 判据从未实现,脚本已
   改为只输出 B_candidate_unadjudicated)

## 8. 已知坑(引用任何结论前必读)

- **exp_local/ 不入 git**:权重与 train.log 在磁盘,git 里只有结果 JSON 与 gate
  证明;引用"实际配置"必须回到 exp_local 日志(registry 已代劳)。
- **BF16 vs FP32**:2026-10-07 之前的 CPI/CE/OG 数值多为 BF16 口径;精细
  dynamics 一律引用 `results/phase1_diag/core/`、`results/phase2/*` 的 FP32 值。
- **protocol 文本 ≠ 实际执行**:LR(3e-5 vs 3e-4)与 dropout(0 vs 0.1)两处偏差,
  有 errata/execution manifest 背书;引用时写"实际执行"并注明。
- **84dc4b4 时代评估代码未提交**:历史 formal 数值不可 bit-reproduce
  (`protocol/errata_v4.2_eval_provenance.md`);harmonized 主链以当前代码为准。
- **750–1020 窗口无 dense 内点**(dense schedule 只有 750 与 1020 两个端点)。

## 9. 建议阅读顺序

1. 本文 §2/§6(结论快照)
2. `docs/EXPERIMENT_REGISTRY.md`(实验全貌)
3. `docs/CLAIM_EVIDENCE_MAP.md`(结论分级 + 专项审计 + coverage audit)
4. 需要细节时:registry 条目 → Artifacts 路径 → 具体 JSON/协议原文
