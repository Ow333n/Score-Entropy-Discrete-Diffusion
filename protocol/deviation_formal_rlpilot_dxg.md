# Infrastructure Deviation Record（append-only）

> Run: rlpilot-185545（protocol v1.0-rl1, 500-step formal RL pilot）
> Git HEAD at run start: 2c59a6f
> 本文件为 append-only infrastructure deviation record，不修改已 hash 的协议正文。

## Deviation #1（2026-10-03）：transient dxgkrnl/WDDM allocation anomaly

- **时间窗口**：kernel time 71402–71514s（≈ 本地 22:10–22:12），时长约 112 秒，
  对应训练约 step 410–450。
- **现象**：dmesg 共 **27 条** `misc dxg: dxgk: dxgkio_create_allocation: Ioctl failed: -75`
  （EOVERFLOW），集中在该窗口内，窗口后自行消失，run 期间无复发。
- **运行影响（全部为事实陈述）**：
  - 无 OOM；
  - 无 NaN / Inf（train 指标全程 finite，nan_rollouts=0）；
  - 无 observable training discontinuity（窗口前后 step 指标连续，grad_norm / reward /
    VRAM 均无异常跳变）；
  - run 后续正常完成 500/500 steps 并通过 checkpoint verification
    （checkpoint_step500.pth / eval snapshots 50/100/250/500 / 5 个 eval JSON 全部验证通过）。
- **root cause：未确认**。
  - 不是"已证明是 NVIDIA driver bug"；
  - 不是"已证明由游戏导致"。
  - 唯一可写定性：**transient dxgkrnl/WDDM allocation anomaly without observed
    model-state corruption**。
- **后续处理**：post-run CPI/OrderGap evaluation 在 GPU 空闲状态下执行；
  开始前记录 dmesg dxg 基线，结束后复查。
