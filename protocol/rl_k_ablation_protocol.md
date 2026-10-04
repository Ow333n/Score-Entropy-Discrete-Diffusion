# RL K-Ablation Diagnostic Protocol（append-only，2026-10-04 夜 unattended）

> 本文件为本次 K=1 vs K=4 diagnostic ablation 的预注册协议。
> 不影响、不修改任何已有 formal RL protocol / result。

## 1. Hypothesis

RL-1（pure on-policy REINFORCE + K=1）task learning signal 弱。
假设：K=1 每条 trajectory 只采一个 differentiable timestep，可能带来较高 estimator
variance / 较弱 credit assignment。
今晚唯一问题：**保持其他 recipe 不变，K=1 → K=4 是否改善 short-horizon task-learning signal？**
（diagnostic，非新 formal result。）

## 2. Fixed variables（两组完全一致）

- init：SFT formal seed1 checkpoint_10200 EMA；optimizer fresh AdamW；lr=3e-6 constant
- P=4、G=4、128 analytic reverse steps、M0-only exact-token reward、group mean-centered
  advantage、safe-region sigma threshold 0.05（rollout 完整保留低-σ 段，仅 PG timestep 截断）
- model / dataset / training sample ordering（rl.seed=0 Generator）/ eval subset（formal64，
  manifest 64–127）/ allocator（expandable_segments:True）/ raw-primary EMA-secondary
- 训练长度：**100 optimizer steps**；评估点 step 0 / 50 / 100（不扩展到 150/250/500）
- 唯一 scientific variable：**K = 1 vs 4**

## 3. K=4 数学语义

每条 trajectory：一次完整 128-step no_grad rollout → 同一 final reward / advantage；
从合法 safe timestep support 采 **K=4 个 J（uniform without replacement，记录在案）**；
对 4 个 J 逐一 differentiable recomputation，loss_j = −A·Σlp_j / (G·K)；
sequential backward（每 J 独立 graph，backward 后立即释放）；4 个 J 完成后**只 optimizer.step
一次**。K=4 是更密集的 MC timestep estimate，**不是**等效放大 4× LR。
objective normalization 与 K=1 一致（总梯度 = 4 个 timestep estimator 的平均）。

## 4. RNG 隔离

- **j_rng 独立**：J timestep 采样使用专用 torch.Generator（CPU，seed = rl.seed + step*10000
  + chunk_offset，落盘记录），与 action-sampling 全局 RNG 隔离。
- 结果保证：K=1 与 K=4 的 dataset sampling / corruption / rollout action randomness
  **逐位相同**；K=1 的 J = K=4 的第 1 个 J（同 seed 同流）。差异只来自 K 本身。
- 注：与 formal RL-1 的唯一实现差异 = J 采样改用 j_rng（diagnostic 隔离需要）；
  K=1 control 与 formal RL-1 除 J-RNG 来源外 scientific 设置全同。此差异记录在案。

## 5. 显存策略（8GB 硬约束）

禁止同时保留 K=4 张 autograd graph。sequential recomputation / gradient accumulation：
每个 J 独立 graph → backward → 立即释放；峰值目标接近 K=1。
不为此改 P/G/seq_len/reward/loss definition。

## 6. Preflight gates（含 amendment #1）

| Gate | 要求 |
|---|---|
| GPU | 无 heavy GPU process；记录 total/used/free |
| RAM（amendment #1 修订） | MemAvailable ≥ 5.0 GiB 且 MemAvailable/MemTotal ≥ 0.65 且 swap used < 1.0 GiB 且无异常 heavy CPU/RAM process |
| Disk | ≥ 20 GB free（/mnt/d 宿主与 / 均记录） |
| dxg | 记录 EOVERFLOW baseline count 与时间戳 |

## 7. Runtime STOP rules（unattended）

- CUDA OOM / NaN/Inf loss 或 grad / model-state corruption / transition hard failure /
  checkpoint-write corruption → graceful stop + failure report，不重试、不改 recipe。
- RAM：MemAvailable < 2.5 GiB 连续多个采样点；或 swap used > 1.5 GiB；或 swap 持续快速增长；
  或 OOM killer / allocation failure；或 Python RSS 异常持续增长 → graceful stop。
- Disk free < 10 GB → stop。
- dxg：相对 baseline 出现连续/密集新增（burst）→ stop；单条孤立日志 → 记录并继续观察。

## 8. 磁盘策略（lightweight diagnostic）

每 run 仅：JSON/CSV/log/metadata + 1 个 rolling recovery checkpoint（覆盖式）+ 最终
lightweight raw/EMA snapshot（step100 eval 用）。成功后删除 rolling checkpoint。
不保存多个 2.7GB full checkpoint；不删除任何既有文件。

## 9. 评估与比较

- formal64（64–127）step 0/50/100：NLL（冻结 corruption realization）、sampled64、
  greedy，RAW primary / EMA secondary。step0 两组共享（同一 init）。
- Compatibility secondary（inference-only）：step0（= harmonized SFT 0.2871/8.7475）、
  K1-100、K4-100 RAW：CPI_abs / CPI_RMS / signed δ mean / OrderGap_raw / OrderGap/token。
- 成本比较：runtime / sec per optimizer step / extra recomputation cost /
  task gain per GPU-hour。
- 统计措辞：one training seed per condition；bootstrap 只表述 sample-level uncertainty，
  不写 seed-level significance。

## 10. 最终判读模板（四选一）

- A: K=4 task learning > K=1 且稳定 → 支持"K=1 timestep estimator density 是 RL-1 限制因素之一"
- B: K=4 ≈ K=1 → 单纯提高 K 不足以解决 weak task signal
- C: K=4 更差/更不稳定 → 更密集 recomputation 不能自动转化成更有效学习
- D: K=4 task 更强但成本不成比例 → estimator efficiency / wall-clock tradeoff 不理想

## 11. 禁止事项

RL-2 / GRPO / PPO / 第二 seed / 新 reward / cross-model / P2 / LR sweep / K=2/K=8 /
步骤扩展 / 修改 frozen formal protocol / 覆盖结果 / 修改 Demo 结论。

---

## Amendment #1（2026-10-04）：RAM preflight gate 修订

- **原因**：原 gate "available RAM ≥ 8 GiB" 为机器无关保守阈值；本 WSL 实例 MemTotal 仅
  7.3 GiB，该条件物理上不可满足。首次 preflight 因此 STOP；该 STOP 属阈值与本机容量
  不匹配，不代表 memory pressure。
- **修订**（PASS iff 全部满足）：
  1. MemAvailable ≥ 5.0 GiB
  2. MemAvailable / MemTotal ≥ 0.65
  3. swap used < 1.0 GiB
  4. 无异常 heavy CPU/RAM process
- 修订时实测：MemAvailable 6.36 GiB / ratio 0.866 / swap used 0.47 GiB / 无 heavy 进程 → PASS。
- 原 preflight STOP 记录：`protocol/rl_k_ablation_preflight_stop.md`（保留，不动）。
- 最终回报措辞（用户指定）：
  "Original preflight stopped because the absolute 8-GiB available-memory threshold
  exceeded the WSL instance's total 7.3-GiB memory. The diagnostic was resumed only
  after replacing it with a machine-capacity-aware safety gate."
