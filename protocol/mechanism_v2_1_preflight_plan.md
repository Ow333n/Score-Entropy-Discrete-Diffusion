# Mechanism Pilot v2.1 — Preflight Plan（执行清单）

**依据**：`protocol/mechanism_complementary_exposure_v2_1.md`（FINAL v1.1 — FROZEN）
**状态**：计划文档（随 FINAL 协议生效；未经用户逐项放行禁止执行）
**原则**：每步先验后做；任何 FAIL → 硬停回报；不碰 v1.2 frozen 资产；措辞纪律按协议 §1.1。

---

## P0. 协议冻结（前置）

- [ ] 用户对 FINAL v1.0 作冻结确认；此后协议内容零改动（sha256 已附，冻结 = 声明，不再改文件）

## P1. Frozen pair/bank map 生成（训练前一次性）

- [ ] 每 replicate r∈{1,2}：对 even m∈{10,12,…,50} 生成 perfect matching（frozen permutation π_r 连续配对）；对 odd m∈{11,13,…,49} 生成 balanced bank（π_r 旋转 matching，singleton 轮换）
- [ ] heldout map：fresh permutation π_h 的 skip-2 边；对每个 m 逐边校验与 rep1/rep2 treated 边集交集为空（确定性 rejection，m≥10 几乎一次通过）
- [ ] 全部 map 落盘 `exp_local/regime_a/mechpilot_v21_maps/` + 每文件 sha256 + 三方边集/交集断言 JSON
- [ ] 校验脚本断言：每 m 的 map 完整（even：n=m/2 边；odd：m cycle 边）、heldout 不重叠、singleton 轮换性质（odd）

## P2. treated/heldout CPI evaluator 冻结（训练前，修订 #3 硬性）

- [ ] 实现 `evaluation/eval_cpi_pairs.py`：读取 frozen manifest（500 样本、span、corruption realization）+ map 文件 → 逐 treated/heldout 边 delta-swap（复用 `compatibility/cpi.py` 打分路径，3 forward/边，pair 选择确定性无 RNG）
- [ ] RNG 隔离：corruption 流沿用 frozen evaluator 既有隔离实现；与训练 policy 流（§10 seeds）seed 域不相交
- [ ] 代码 + heldout map + pair 定义全部落盘 sha256；**此后（含训练期间/训练后）禁止修改**
- [ ] §9.5 OrderGap frozen 子集（索引列表 + sha256）同期冻结
- [ ] §9.4 校准诊断数据口径（reliability 分桶、scale summary）与 evaluator 同期冻结
- [ ] smoke：step0 模型上跑 treated（rep1/rep2 两 map）+ heldout，输出三套 step0 基线 JSON，数值有限性检查

## P3. G3a 设计期 preflight 复跑

- [ ] `scripts/v21_design_validation.py` 全量重跑（exact 枚举 m=4/6/5、闭式 sweep m∈[10,50]、MC marginal、degenerate K∈{0,1,m−1,m} 同分布断言）；输出归档
- [ ] 训练流 dry-run（CPU replay，沿用 v1.2 dryrun 机制）：U/H/L 同 replicate 的共享流 digest 一致、exact-K 断言、odd phase 流/singleton 硬币消耗一致、policy 流 seed 隔离

## P4. 磁盘前置（FROZEN 目标：D: free ≥ 30GB，最好 ≥35GB；当前 18GB，不满足）

- [ ] **第一步（只读）**：p1-s1/p1-s2 retirement audit——精确路径、总大小、逐 checkpoint 大小/step、哪些已被最终 analysis 使用、哪些结果 JSON 已冻结、哪些是纯 intermediate/redundant、是否有最终 EMA/raw checkpoint、logs/metadata/schedule hash 是否独立保存、删除中间 checkpoint 后能否完全支持已有结论/provenance
- [ ] **第二步**：产出候选删除表（优先"保留 final checkpoint + logs + metadata + frozen analysis，只删 dense intermediate"，目标安全释放 ≥15–20GB）；**逐项经用户批准后才 rm**
- [ ] 删除后 DiskPart compact（既有流程：wsl --shutdown → attach readonly → compact → detach）
- [ ] 复核：D: free ≥30GB（最好 ≥35GB）；v1.2 资产抽查 sha256 未变
- [ ] **禁止**：mechpilot v1.2、v1.2 report、P1/P2 JSON、protocol/、manifests/、git、正式 SFT/RL 关键资产

## P5. Implementation + 测试（先 unit tests，再 smoke）

- [ ] unit tests：mask policy（H/U/L、even/odd）、frozen map/bank 加载、exact-K、schedule digest、evaluator pair 模式、OrderGap 子集
- [ ] **50–100 step smoke test**（1 run 或 3 policy 各短跑）：exact K、schedule hash、map hash、H/U/L 同 sample/span/σ/K、无 RNG 污染、marginal parity、E_comp 排序、covariance 排序、loss finite、VRAM、磁盘增长、checkpoint 契约
- [ ] 全部 PASS 才允许正式 6-run pilot

## P6. 训练（6 runs）

- [ ] driver 守卫：manifest hash、frozen 文件零改动、map hash、协议 sha256（同 v1.2 driver 语义）
- [ ] 顺序 U1→H1→L1→U2→H2→L2（或交错），2500 steps、checkpoints 500/1020/2500 EMA + raw 2500、共享 step0
- [ ] 每 run metadata：policy、replicate、schedule digest、map sha、实测 E_comp（overall + K/m 桶 + σ 桶）、exact-K 断言计数、L/R 计数、§7.6 confound diagnostics、loss 轨迹、gradient norm(+rolling variance)、VRAM/runtime、anomaly

## P7. 评估

- [ ] global：6 runs × 3 steps × (CPI + task) = 36 文件（frozen evaluator）
- [ ] treated：H/L × 2 reps × 3 steps + step0 ×（rep1/rep2 两 map）= 12+2；heldout：U/H/L × 2 reps × 3 steps + step0 = 18+1（evaluator 冻结版）
- [ ] §9.5 OrderGap subset + path-score variance：仅 step2500（+step0），6 runs
- [ ] §9.4 校准/尺度诊断（reliability 分桶、scale summary；temperature-scaled CPI 按 §9.4 标记 deferred）
- [ ] 全部落盘 results/mechanism_pilot_v21/；逐样本配对口径（同一 manifest 顺序）校验

## P8. G3 分析

- [ ] G3a：8 项逐项判定（strict E_comp 仅 non-degenerate K∈[2,m−2]；degenerate K∈{0,1,m−1,m} 单独报告不得 FAIL；confound diagnostics 记录与披露、不入阈值）
- [ ] G3b：CPI_treated H<L，per-rep paired bootstrap（10k, seed=0）+ 聚类 CI
- [ ] G3c：**primary H vs L**（判据同 G1a）；secondary H vs U、U vs L、CPI_RMS、signed δ、δ quantiles、heldout、OrderGap subset；U 不居中按 §11 预注册规则解释
- [ ] G3d：NLL±0.02 / acc±0.01 matched performance + CI；no overlap ⇒ INCONCLUSIVE 非 FAIL；匹配规则事前冻结、禁后验选 checkpoint；匹配后重做 G3c
- [ ] Case A–E 判定（预注册，禁止事后重写）；§15.1 Confirmatory 进入条件评估 → 报告 → 冻结

## 成本与安全

- GPU：smoke ~15min + 训练 ≈1.75h + 评估（含 treated/heldout ≤75k forward + OrderGap 子集）≈1.5–2.5h → 全程 ≈4–4.5h（>2h 由用户 tmux 跑，沿用惯例）
- 磁盘：checkpoints ≈15.7GB + maps/结果 <0.1GB；**D: free ≥30GB（最好 ≥35GB）才启动**
- 任何一步 FAIL → 硬停回报，不自动绕过
