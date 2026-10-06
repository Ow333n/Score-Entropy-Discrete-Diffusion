# Mechanism Pilot v1.2 — 冻结报告（FROZEN）

- **状态**：FROZEN（2026-10-06 冻结，禁止事后修改 hypothesis / gate / Case 定义）
- **Protocol**：`protocol/mechanism_pilot_protocol.md`（v1.2）
- **Frozen evaluator**：`evaluation/eval_cpi.py` + `evaluation/eval_task.py`；manifest `manifests/regime_a_eval_v1.jsonl`（sha256 `1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3`，500 样本）
- **分析脚本**：`scripts/p1_mech_analysis.py`、`scripts/p1_g1b_analysis.py`（P1，已冻结期运行）；`scripts/p2_g2_analysis.py`（P2 G2，本报告生成时运行）
- **分析产物**：`results/mechanism_pilot/p1_stage1_analysis.json`、`p1_g1b_analysis.json`、`p2_g2_analysis.json`

---

## 1. 研究问题与预注册假设

研究问题：为什么 partial-reveal SFT 降低 reveal-order sensitivity（CPI / OrderGap 衰减）。

预注册假设（protocol §5.1，v1.2 冻结）：

- **H1（diversity mechanism，primary）**：matched-step 与 matched-performance 双口径下，CPI(A) < CPI(B)。
- **H2（directional bias，辅助）**：C 与 D 的 mean signed δ 符号相反；A/B 的 mean signed δ 接近 0 且 |δ| 更低。
- **H3（generic task-learning alternative）** / **H4（performance confound）**：衰减来自通用监督适配 / task learning 副产品。

## 2. 设计与执行摘要

- **Policy**：A = fresh random partial（每步重选 K 个 mask 位置）；B = fixed random partial（固定 π_L 取前 K）；C = L2R ordered；D = R2L ordered。
- **规模**：8 runs（A/B/C/D × 2 replicates），2500 optimizer steps，batch 32，span [10,50]，seq 256，SEDD-small。
- **共享**：同 replicate 共享 step0 pretrained 初始化、sample/span/σ/K 四元流 schedule（rep1 sha256 `7b76dcfb…`，rep2 `ace3eec9…`）；dropout=0；frozen evaluator 与训练流完全隔离。
- **Checkpoints**：EMA @500/1020/2500 + raw @2500（EMA-only 容器含 step 字段）。
- **执行异常记录**：D1 首次运行在维护停机时被 TERM（aborted/pre-maintenance，**voided**，partial 已删除）；compact 后从 step0 以原 seed / 原 schedule 完整重跑，schedule hash 与 preflight 一致。其余 7 run 一次通过。所有 run loss 无 NaN（每 run 26 个采样点），VRAM peak 5.73GB，runtime 1037–1060s（2.36–2.41 steps/s）。

## 3. 结果

### 3.1 G1a — matched-step（primary）

step 2500 EMA，frozen evaluator 500 样本逐样本配对差 Δ_i = CPI_i(A) − CPI_i(B)：

| step | rep | CPI(A) | CPI(B) | ΔCPI | per-rep paired 95% CI（10k） |
|---|---|---|---|---|---|
| 2500 | 1 | 0.2734 | 0.2793 | −0.0059 | [−0.0382, +0.0235] |
| 2500 | 2 | 0.2715 | 0.2637 | +0.0078 | [−0.0211, +0.0373] |

两 replicate CI 均含 0，点估计异号 → **G1a = INCONCLUSIVE**。

Temporal（step 0/500/1020/2500，A/B × 2 reps，NLL / acc / CPI）：

| step | rep | NLL A/B | acc A/B | CPI A/B | ΔCPI | CI |
|---|---|---|---|---|---|---|
| 0 | — | 4.2812 | 0.3223 | 0.3281 | +0.0000 | — |
| 500 | 1 | 4.0000/4.0000 | 0.3439/0.3446 | 0.2812/0.2617 | **+0.0195** | **[+0.0029, +0.0357]** |
| 500 | 2 | 4.0000/4.0000 | 0.3443/0.3453 | 0.2871/0.2754 | +0.0117 | [−0.0013, +0.0269] |
| 1020 | 1 | 3.9688/3.9531 | 0.3478/0.3522 | 0.2852/0.2852 | +0.0000 | [−0.0234, +0.0253] |
| 1020 | 2 | 3.9688/3.9688 | 0.3523/0.3477 | 0.2910/0.2910 | +0.0000 | [−0.0237, +0.0250] |
| 2500 | 1 | 4.0625/4.0312 | 0.3370/0.3403 | 0.2734/0.2793 | −0.0059 | [−0.0382, +0.0235] |
| 2500 | 2 | 4.0625/4.0312 | 0.3391/0.3408 | 0.2715/0.2637 | +0.0078 | [−0.0211, +0.0373] |

### 3.2 G1b — matched-performance

primary NLL ±0.02 / secondary acc ±0.01；A 的参考 checkpoint 与 B 配对：

- rep1 / rep2 均在 step 500、1020 有匹配对；**step 2500 两 replicate 均无 NLL±0.02 overlap**。
- 全部匹配对的 ΔCPI CI 均含 0。
- 按 §5.3 no-overlap 规则 → **G1b = INCONCLUSIVE**（非 FAIL）。

### 3.3 G2 — directional auxiliary gate

mean signed δ 的 C vs D 配对差（δ = log p(a|C) + log p(b|C,a) − log p(b|C) − log p(a|C,b)，frozen evaluator 固定 pair 方向）；CI 为 per-replicate paired bootstrap（10k, seed=0）+ 按 replicate 聚类的 pooled CI（gate 口径）：

| step | rep | δ̄(C) | δ̄(D) | dCD | per-rep CI | 符号相反 |
|---|---|---|---|---|---|---|
| 500 | 1 | −0.0145 | −0.0220 | +0.0075 | [−0.0095, +0.0245] | ✗ |
| 500 | 2 | −0.0093 | −0.0092 | −0.0001 | [−0.0263, +0.0239] | ✗ |
| 1020 | 1 | −0.0383 | −0.0060 | −0.0324 | **[−0.0600, −0.0049]** | ✗ |
| 1020 | 2 | −0.0154 | −0.0021 | −0.0133 | [−0.0395, +0.0128] | ✗ |
| 2500 | 1 | −0.0193 | −0.0074 | −0.0120 | [−0.0590, +0.0330] | ✗ |
| 2500 | 2 | +0.0142 | −0.0275 | +0.0416 | [−0.0007, +0.0841] | ✓ |

聚类 CI：500 [−0.0136, +0.0181]（INC）；1020 [−0.0462, −0.0004]（排除 0 但两 replicate 同符号 → FAIL）；2500 [−0.0324, +0.0617]（INC）。

**G2 primary @2500 = INCONCLUSIVE**（CI 含 0；rep1 同符号、rep2 相反符号，replicate 间不一致）。

H2 的 A/B 辅助口径（step 2500 signed δ）：A1 +0.0026，B1 −0.0058，A2 +0.0206，B2 −0.0156。与 C/D（−0.0193 / −0.0074 / +0.0142 / −0.0275）相比，**无一致的 |δ| 排序**（D2 的 |δ̄|=0.0275 为 8 run 最大，A2 0.0206 次之）——仅为描述性观察，不构成 gate。

### 3.4 跨 policy 观察（描述性，非确认性）

1. **SFT 衰减一致出现**：pretrained CPI_abs = 0.3281；8 个 SFT run step 2500 全部低于此（A/B：0.2637–0.2793；C/D：0.2773–0.2910）。衰减不依赖 policy 族（A/B/C/D 都出现）→ 与"互补暴露 / partial-reveal 监督本身"共因假说一致，但本 pilot 未验证该机制。
2. **Early transient（500 步，禁止升级为确认性结果）**：两个 replicate 均出现 CPI(A) > CPI(B)，rep1 CI 排除 0；1020 消失、2500 不稳定。
3. **Early transient（1020 步，signed δ）**：两 replicate 均 δ̄(C) < δ̄(D)，聚类 CI 排除 0；2500 消失。同样标记为 transient observation。

## 4. Gate 判定与结论

| Gate | 判定 |
|---|---|
| G1a（matched-step） | INCONCLUSIVE |
| G1b（matched-performance） | INCONCLUSIVE |
| **G1（primary）** | **INCONCLUSIVE** |
| G2（directional auxiliary） | **INCONCLUSIVE** |

预注册结论矩阵（protocol §5.2）：**G1 = INC，G2 任意 → 机制未决**。

正式表述（措辞纪律，protocol §8）：

> 在 2 replicates、8GB、2500 步、SEDD-small 的 pilot 设置下，没有证据支持 fresh random resampling（A）相对 fixed nested partial（B）稳定产生更低 CPI（G1 = INCONCLUSIVE）；directional exposure（C/D）也未能在两 replicate 一致地塑造相反符号的 signed δ（G2 = INCONCLUSIVE）。Fresh-vs-Fixed hypothesis 的正式状态为 **INCONCLUSIVE / unsupported at pilot scale**——既不能说 fresh random 已被证明无效，也不能说已被证明有效。观察到的一致现象是：所有 partial-reveal SFT policy（A/B/C/D）相对 pretrained 均出现 CPI 衰减，提示衰减可能与 A/B 共有的 partial-reveal 暴露相关；该机制尚未验证。

## 5. 下一步（confirmatory 规划，未冻结）

按矩阵：**confirmatory = 加密 checkpoint / 加训练 seeds 后重判**（G1 需 ≥4 training seeds 才可作机制结论，protocol §5.2 不确定性口径）。研究假设已从"fresh resampling 是原因"收窄为"partial-reveal / complementary conditioning exposure 共因"，后续 protocol 版本（v2.x draft 见 `protocol/mechanism_complementary_exposure_v2.md`）须先解决 identifiability 问题（隐藏 pairing 对模型不可见 → 边缘化后 mask-set distribution 不变）后方可实施；**当前不启动任何 v2 训练**。

## 6. 资产与 provenance

- 8 run 资产：`exp_local/regime_a/mechpilot-{A,B,C,D}{1,2}-*/`（EMA 500/1020/2500 + raw 2500 + metadata + train.log）
- 评估产物：`results/mechanism_pilot/cpi_*.json`、`task_*.json`（P1 24 + P2 24 + step0 2）
- Schedule hash：rep1 `7b76dcfb3ae785feb91c86e1e4381cd6e6da674c79f1fb72c12bdc9f660d017b`；rep2 `ace3eec9434ca668c254e0d281e4c4be25b3494611b1d344e893780bd5f6c1a2`（与 preflight 一致）
- C1 训练 git commit：`f8c2f34b`；D1/C2/D2：`b601346b`（driver 恢复后）
- 分析时 git HEAD：见报告生成 commit
