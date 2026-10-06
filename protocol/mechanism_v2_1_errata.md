# Mechanism Pilot v2.1 — Errata & Execution Notes

**状态：随 FINAL v1.1 协议生效的补充记录（2026-10-06）**
**原则：不修改 `protocol/mechanism_complementary_exposure_v2_1.md` 本体（FINAL v1.1 — FROZEN，
sha256 `af312d864d41d8f67da343d55a8dee8a29a91c37dea622e6b2c4b22ef441fc61` 不变）。
本文件为事后补充的透明度记录，不追溯修改任何 frozen gate / Case / 措辞纪律。**

---

## Erratum 1：odd-m cycle edge 的 active phase 数（§6 explanatory-text error）

- **原 §6 文本**："每条边恰在 2 个 phase active"（"each cycle edge is active in exactly 2 phases"）。
- **仅在 m=5 时成立**。一般 odd m = 2n+1 下正确为：**每条 cycle edge 恰在 (m−1)/2 = n 个 phase active**。
  （phase t 的 matching 取 cycle edges `{(π(i), π(i+1)): i ≡ t+1 (mod 2)}`；固定边 e_i 满足
  `t ≡ i−1 (mod 2)` 的 phase 共 n 个。）
- **定性**：explanatory-text error（旁注性表述），非构造定义错误。
- **影响评估**：implementation 一直按 frozen matching-bank construction 定义（§6 主体：
  π 旋转 matching、singleton 轮换、phase schedule、singleton 硬币）执行；exact-K、单位置
  marginal parity、identifiability（exact TV > 0）、E_comp、covariance 排序的
  设计期验证与全部 unit test / dryrun 均按该构造执行并通过——**均不受此 erratum 影响**。
- **处置**：不追溯修改 frozen protocol 本体；本记录即处置。

## Erratum 2（说明，非 erratum）：degenerate-K item 上 U 与 H/L 的 E_comp 度量差异

- **现象**：K ∈ {0, 1, m−1, m}（degenerate，§7.3）的 item 上，U 的 pooled E_comp 比 H/L 低
  ≈2e-4（dryrun 实测：r1 U=0.05136 vs H/L=0.05112；r2 同量级）。
- **原因**：metric-definition artifact——E_comp 的分母只含 active pairs（§8）；K=1 时
  U 的 mask 可落在当前 phase 的 singleton 位置（不在任何 active pair 内），该 item 的
  exactly-one-masked active pair 计数为 0；H/L 的 K=1 mask 必为 discordant pair 成员
  （计数恒为 1）。K=m−1 对称。
- **关键事实**：exact mask-set distributions 已验证逐位相同
  **P_H(M) = P_U(M) = P_L(M)**（设计期 exact 枚举 §7.3 + 单测
  `test_g_degenerate_k_same_distribution`：三 policy support 完全相等）。差异仅在
  E_comp 度量定义层面，非分布差异。
- **gate 处置**：degenerate K 不参与 strict E_H > E_U > E_L gate（§11 G3a-5 修订 #2），
  单独报告、**不构成 G3a FAIL**。frozen gate 不修改。

## Execution notes

- **LR 决策**：v1.2 protocol 文本写 3e-5；8 个实际 v1.2 run 全部使用 config 实际值 3e-4
  （train.log 实证 step 2500 时 lr=3.00e-04）。v2.1 选择 **3e-4 = actual executed setting**，
  保持与真实 empirical baseline 连续、U 基线与 v1.2-A 可比。旧 v1.2 protocol 不静默改写；
  全部实际训练超参见 `protocol/mechanism_v2_1_execution_manifest.json`（+sha256 sidecar）。
- **dropout 实际执行值 = 0.1**（2026-10-06 provenance patch 期发现）：v1.2 §9 与 v2.1 §10
  文本均写 dropout=0，但 hydra 合并语义下 `defaults` 中 `model: small` 排在 `_self_` 之后，
  `model/small.yaml` 的 `dropout: 0.1` 覆盖 `vanilla_256.yaml` 的 `model: {dropout: 0.0}`
  → **全部既往 SFT/mechpilot run 与 v2.1 smoke 实际均在 dropout=0.1 下训练**（模型代码
  train 模式确实应用 dropout，transformer.py bias_dropout_add_scale_fused_train）。
  确定性不因此受损：torch.utils.checkpoint 保存/恢复 RNG state（backward 重算 dropout
  掩码与 forward 一致），且 p1 Gate-2 的 checkpoint_1020 逐字节复现已实证全管线确定。
  v2.1 按与 LR 同一裁定原则固定 **dropout=0.1 = actual executed setting**，透明记录，
  不静默改写任何旧 protocol 文本。
- **phase 实现解释**：phase = (span_start + span_len) mod m（s≡0：每 item 即其 span 的唯一
  item；由共享 span 流确定、无额外 RNG；实现于 task_data/v21_policy.py 并记录）。q=K/m 使
  单位置 marginal 对 phase 分布不敏感（marginal_i = K/m + P(phase=φ(i))·(q−K/m)，q=K/m 时精确）。
- **C̄_treated 双口径**：treated edges（§6 主口径）与 active pairs（§7.4 闭式口径）均实测，
  dryrun 证明两者均满足 H < U < L 排序。
