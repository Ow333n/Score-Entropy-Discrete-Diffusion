"""RL-1 smoke pre-flight: SEDD model-load CUDA memory test (post WSL restart).

复刻 training/rl.py main() 的初始化序列直到 step loop 之前:
  config compose → graph/noise → load_rl_init(EMA-10200) → score fns → prompt pool
记录 model-load 完成时点 (pre-rollout) 的 CUDA 内存快照 + 中等 allocation 测试
+ 单条 forward 冒烟 (batch=1, 非 rollout)。纯诊断脚本, 不改任何 recipe。
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

import graph_lib
import noise_lib
from model import utils as mutils
from rl import loader


def snap(tag):
    a = torch.cuda.memory_allocated() / 1e6
    r = torch.cuda.memory_reserved() / 1e6
    free, total = torch.cuda.mem_get_info()
    print(f"[{tag}] allocated={a:.1f}MB reserved={r:.1f}MB free={free / 1e6:.1f}MB "
          f"total={total / 1e6:.1f}MB max_alloc={torch.cuda.max_memory_allocated() / 1e6:.1f}MB "
          f"max_reserved={torch.cuda.max_memory_reserved() / 1e6:.1f}MB")


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    device = torch.device("cuda")
    torch.manual_seed(cfg.seeds.model_seed)
    torch.cuda.reset_peak_memory_stats()
    snap("start")

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    snap("after graph+noise")

    model, info = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    print("load info:", json.dumps(info))
    assert info["shadow_vs_model_mismatches"] == 0, "EMA shadow vs model mismatch!"
    assert info["step"] == 10200, f"unexpected checkpoint step {info['step']}"
    model.train()
    sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
    train_score_fn = mutils.get_score_fn(model, train=True, sampling=False)

    # manifest prompt pool (与 training/rl.py 同口径)
    recs = [json.loads(l) for l in open(cfg.rl.manifest)][: cfg.rl.pool]
    x0_pool = torch.stack([torch.tensor(r["x0"]) for r in recs]).to(device)
    xt0_pool = torch.stack([torch.tensor(r["initial_state"]) for r in recs]).to(device)
    sig0_pool = torch.tensor([r["sigma"] for r in recs], device=device)
    span_pool = torch.zeros(cfg.rl.pool, cfg.data.seq_len, dtype=torch.bool, device=device)
    m0_pool = torch.zeros(cfg.rl.pool, cfg.data.seq_len, dtype=torch.bool, device=device)
    for k, r in enumerate(recs):
        span_pool[k, r["span_start"]:r["span_end"]] = True
        if r["initial_masked_positions"]:
            m0_pool[k, r["initial_masked_positions"]] = True
    print(f"prompt pool: {cfg.rl.pool} 样本 |M0| mean={m0_pool.sum(-1).float().mean().item():.1f}")

    # --- pre-rollout 内存快照 (用户要求记录时点) ---
    snap("AFTER MODEL LOAD (pre-rollout checkpoint)")
    print("--- nvidia-smi at pre-rollout checkpoint ---")
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.strip())

    # 中等 allocation 测试 (512MB, 复刻 512MB allocation PASS)
    x = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device=device)
    x.fill_(1.0)
    ok = bool((x.sum() > 0).item())
    print(f"512MB allocation test: {'PASS' if ok else 'FAIL'}")
    snap("during 512MB test")
    del x
    torch.cuda.empty_cache()
    snap("after free 512MB")

    # 单条 forward 冒烟 (batch=1, 验证 CUDA kernel 执行, 非 rollout)
    model.eval()
    with torch.no_grad():
        xt = torch.randint(0, cfg.tokens + 1, (1, cfg.data.seq_len), device=device)
        sig = torch.full((1, 1), 1.0, device=device)
        s = sampling_score_fn(xt, sig.squeeze(-1))
        finite = bool(torch.isfinite(s).all().item())
        print(f"forward smoke: score shape={tuple(s.shape)} finite={finite}")
        assert finite
    snap("after forward smoke (peak includes forward)")
    print("MODEL_LOAD_TEST_PASS")


if __name__ == "__main__":
    main()
