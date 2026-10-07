"""Option B seed2 deterministic rerun 安全 wrapper（用户 2026-10-07 批准）。

与 scripts/optionb_dense_retrain.py 的训练执行路径完全一致：
- 相同 overrides（seeds / n_iters / lr / save_at / training.name）
- 相同 vanilla.main() 训练循环
- 相同 checkpoint payload 构造逻辑与 key 顺序（tensor 内容与旧产物可比）

唯一差异在 checkpoint 保存路径（I/O only，不改变训练数学）：
- atomic save：写 <path>.tmp → flush + fsync → os.replace
- free-space guard：宿主 F 盘（动态 VHD 所在盘；不用 df / 的 VHD logical capacity）；
  启动前 < 32GB 拒绝启动，且 free 须覆盖预计 footprint（≈11.6GB）；
  每次保存前 < 20GB 安全停止（已有 checkpoint 不受影响）
- 每次保存后打印并记录文件大小与 host free GB（save_sizes.csv）
- 保存失败：清理残留 .tmp、FATAL 日志、干净退出，绝不破坏已有 checkpoint

约束（用户 2026-10-07 修正案）：
- 不修改 vanilla.py / frozen protocol / frozen evaluator
- 不修改 optimizer / LR / batch / seeds / dataloader / corruption
- 不新增任何消费 torch / numpy / python RNG 的操作
  （disk_usage / git rev-parse / open / fsync / replace / torch.save 均无 RNG；
   preflight 不初始化 CUDA，GPU 状态由启动方用 nvidia-smi 检查）
- 不增加额外 forward/backward；不改变 save_at；不改变 payload tensor 内容与 key 顺序
- 磁盘检查、save_sizes.csv、日志只发生在 checkpoint 保存路径（及启动前 preflight）

用法（tmux 跑，约 20 分钟）:
  .venv/bin/python scripts/optionb_dense_retrain_safe.py 2
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import training.vanilla as vanilla

DENSE_SAVE_AT = [50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1020, 1500, 2500]
ANCHORS = {500, 1020, 2500}

# --- 宿主 F 盘 guard（动态 VHD 所在盘；不要用 df / 的 VHD logical capacity 判断）---
HOST_F = "/mnt/f"
STARTUP_MIN_GB = 32.0
SAVE_MIN_GB = 20.0
EXPECTED_FOOTPRINT_GB = 11.6  # 10×648M + 3×1.30G + meta 648M


def host_free_bytes():
    return shutil.disk_usage(HOST_F).free


def save_ckpt_safe(path, state):
    """vanilla.save_ckpt 的 I/O 安全替代；payload 构造与原 Option B 完全一致。"""
    free = host_free_bytes()
    if free < SAVE_MIN_GB * 1e9:
        print(f"FATAL: 保存 {path} 前 host F free={free / 1e9:.2f}GB "
              f"< {SAVE_MIN_GB:.0f}GB guard，安全停止（已有 checkpoint 完好）", flush=True)
        sys.exit(1)

    # payload 构造与 scripts/optionb_dense_retrain.py 逐字一致（key 顺序不变）
    payload = dict(ema=state["ema"].state_dict(), step=state["step"])
    if "checkpoints-meta" not in path and state["step"] in ANCHORS:
        payload["model"] = state["model"].state_dict()

    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        print(f"FATAL: 保存 {path} 失败：{e!r}（残留 .tmp 已清理，干净退出）", flush=True)
        sys.exit(1)

    size = os.path.getsize(path)
    free_after = host_free_bytes()
    work_dir = os.path.dirname(path)
    if os.path.basename(work_dir) == "checkpoints-meta":
        work_dir = os.path.dirname(work_dir)
    rel = os.path.relpath(path, work_dir)
    with open(os.path.join(work_dir, "save_sizes.csv"), "a") as f:
        f.write(f"{rel},{size},{free_after}\n")
    print(f"saved {rel} size={size / 1e9:.3f}GB hostF_free={free_after / 1e9:.2f}GB",
          flush=True)


def main():
    seed = int(sys.argv[1])
    assert seed in (1, 2), "canonical seeds 1/2"

    # --- preflight（不初始化 CUDA、不消费任何 RNG）---
    free = host_free_bytes()
    print(f"preflight: host F free={free / 1e9:.2f}GB "
          f"(startup guard={STARTUP_MIN_GB:.0f}GB, expected footprint≈{EXPECTED_FOOTPRINT_GB}GB)",
          flush=True)
    if free < STARTUP_MIN_GB * 1e9:
        print(f"FATAL preflight: host F free < {STARTUP_MIN_GB:.0f}GB，拒绝启动", flush=True)
        sys.exit(1)
    if free < (STARTUP_MIN_GB + EXPECTED_FOOTPRINT_GB) * 1e9:
        print(f"FATAL preflight: host F free 不足以容纳预计 footprint "
              f"(需 ≥ {STARTUP_MIN_GB + EXPECTED_FOOTPRINT_GB:.1f}GB)，拒绝启动", flush=True)
        sys.exit(1)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    print(f"preflight: seed={seed} git_commit={commit} "
          f"save_at={DENSE_SAVE_AT} anchors={sorted(ANCHORS)}", flush=True)

    vanilla.save_ckpt = save_ckpt_safe
    overrides = [
        f"training.n_iters=2500",
        f"training.save_at_steps={DENSE_SAVE_AT}",
        "optim.lr=3e-5",
        f"seeds.model_seed={seed}",
        f"seeds.data_order_seed={seed}",
        f"seeds.corruption_seed={seed}",
        f"training.name=optionb-dense-s{seed}",
    ]
    sys.argv = ["training/vanilla.py"] + overrides
    vanilla.main()


if __name__ == "__main__":
    main()
