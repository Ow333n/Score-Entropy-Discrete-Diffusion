"""正式实验前核查 ①: LR scheduler 行为 (CPU, 无需 GPU)。

协议要求: warmup=2500 结束后 LR 恒定保持 3e-5, 且 LR trajectory 不因
total_steps 从 30000 改为 10200 而改变。

实现路径: losses.optimization_manager (训练循环每 optimizer step 调用一次):
    g['lr'] = lr * np.minimum(step / warmup, 1.0)
step 是当前 optimizer step 计数, 不依赖 n_iters —— 本脚本在 n_iters=10200 与
n_iters=30000 两种配置下逐 step 驱动同一 optimize_fn, 断言两条 trajectory 逐点一致,
且 step>=2500 时 LR == 3e-5 精确相等 (float 位级)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

import losses


class FakeScaler:
    """记录调用、不做实际缩放的 scaler 替身 (GradScaler 需要 CUDA, 本检查只在 CPU)。"""
    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        pass

    def update(self):
        pass


def lr_trajectory(n_iters):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256",
                      overrides=[f"training.n_iters={n_iters}", "optim.lr=3e-5"])
    p = torch.nn.Parameter(torch.zeros(1))
    optimizer = losses.get_optimizer(cfg, [p])
    p.grad = torch.ones_like(p)          # grad_clip 需要 grad
    optimize_fn = losses.optimization_manager(cfg)
    scaler = FakeScaler()

    traj = {}
    for step in [1, 1000, 2499, 2500, 2501, 5000, n_iters]:
        # 与训练循环一致: 每个 optimizer step 调一次 optimize_fn(step)
        # (重复调用同一 step 不改变结论; 这里逐 step 采样关键点)
        optimize_fn(optimizer, scaler, [p], step=step)
        traj[step] = optimizer.param_groups[0]["lr"]
    return traj


def main():
    traj_10200 = lr_trajectory(10200)
    traj_30000 = lr_trajectory(30000)

    print("=" * 60)
    print("核查 ①: LR scheduler (warmup=2500, lr=3e-5)")
    print(f"{'step':>7} {'lr (n_iters=10200)':>20} {'lr (n_iters=30000)':>20}")
    for k in sorted(set(traj_10200) | set(traj_30000)):
        v1 = traj_10200.get(k)
        v2 = traj_30000.get(k)
        print(f"{k:>7} {str(v1):>20} {str(v2):>20}")

    # 断言 1: 共享 step 上两条 trajectory 逐点位级一致; 且 30000 轨迹在 10200 之后仍恒定
    for k in set(traj_10200) & set(traj_30000):
        assert traj_30000[k] == traj_10200[k], f"step {k} LR 不一致!"
    assert traj_30000[30000] == 3e-5, "30000 步时 LR 漂移!"
    # 断言 2: warmup 结束后恒定 3e-5
    for step, lr in traj_10200.items():
        if step >= 2500:
            assert lr == 3e-5, f"step {step} LR={lr} != 3e-5"
    # 断言 3: warmup 内是线性 ramp
    assert abs(traj_10200[1] - 3e-5 * (1 / 2500)) < 1e-30
    assert abs(traj_10200[2500] - 3e-5) < 1e-30
    assert traj_10200[2501] == 3e-5

    print("✅ 核查 ① 通过: warmup 后 LR 恒定 3e-5 (位级); trajectory 与 n_iters 无关")
    print("   实现细节: g['lr'] = lr * min(step/warmup, 1.0), step 为 optimizer step 计数,")
    print("   n_iters 只出现在训练循环终止条件, 不进入 scheduler。")


if __name__ == "__main__":
    main()
