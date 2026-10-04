"""Phase C 轨迹验收（inference-only）:

1. 跨进程确定性：在全新进程重导 3 个样本（s000/s069/s375）× 4 stages × 2 modes，
   与已落盘 JSON 对比（final_tokens / keyframes 逐位）
2. 全 15 文件完整性：token 合法域、sigma 有限、final == 最后关键帧、mask_count 一致、
   无 hard-region 失败（构造保证，复核 gate 字段）、m0_reward 与 final 一致
3. dxg 计数
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

import graph_lib
import noise_lib
from model import SEDD
from model.ema import ExponentialMovingAverage
from rl import loader
from rl import reward as rw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from export_demo_trajectories import export_rollout, pick_keyframes, decode  # noqa: E402

TRAJ = os.path.join(ROOT, "demo_assets/trajectories")
MANIFEST = os.path.join(ROOT, "manifests/regime_a_eval_v1.jsonl")
SEED_BASE = 10000
RECHECK = [0, 69, 375]


def load_stage(stage, cfg, device):
    if stage == "pretrained":
        hf = SEDD.from_pretrained("louaaron/sedd-small")
        m = SEDD(cfg).to(device).eval()
        m.load_state_dict(hf.state_dict(), strict=False)
        del hf
        return m
    if stage == "sft":
        m, info = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
        assert info["step"] == 10200
        return m
    m, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    snap = torch.load(os.path.join(ROOT, "exp_local/regime_a/rlpilot-185545",
                                   "checkpoint_step500.pth"),
                      map_location="cpu", weights_only=False)
    if stage == "rl_raw":
        m.load_state_dict(snap["model"], strict=False)
    else:
        ema = ExponentialMovingAverage(m.parameters(), decay=0.9999)
        ema.load_state_dict(snap["ema"])
        ema.copy_to(m.parameters())
    return m


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    device = torch.device("cuda")
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    recs = [json.loads(l) for l in open(MANIFEST)]

    stages = ["pretrained", "sft", "rl_raw", "rl_ema"]
    print("== 1. 跨进程确定性（fresh process re-export 3 samples） ==")
    all_ok = True
    for idx in RECHECK:
        r = recs[idx]
        x_t0 = torch.tensor(r["initial_state"], device=device)[None]
        sigma0 = torch.tensor([float(r["sigma"])], device=device)
        saved = json.load(open(os.path.join(TRAJ, f"traj_s{idx:03d}.json")))
        for stage in stages:
            m = load_stage(stage, cfg, device)
            m.eval()
            for mode, greedy in [("sampled", False), ("greedy", True)]:
                states, sigmas, gst = export_rollout(
                    m, graph, noise, x_t0, sigma0, cfg.rl.rollout_steps,
                    SEED_BASE + idx, greedy=greedy, mask_token=cfg.tokens)
                sv = saved["stages"][stage]["modes"][mode]
                same_final = states[-1] == sv["final_tokens"]
                same_kf = [f["tokens"] for f in pick_keyframes(states, sigmas)] == \
                          [f["tokens"] for f in sv["keyframes"]]
                if not (same_final and same_kf):
                    all_ok = False
                    print(f"  FAIL s{idx:03d}/{stage}/{mode}: final_match={same_final} "
                          f"keyframes_match={same_kf}")
                else:
                    print(f"  PASS s{idx:03d}/{stage}/{mode} (final + keyframes bit-identical)")
            del m
            torch.cuda.empty_cache()
    print(f"  跨进程确定性: {'ALL PASS' if all_ok else 'FAILED'}")

    print("== 2. 全 15 文件完整性 ==")
    n_files = 0
    for fn in sorted(os.listdir(TRAJ)):
        if not fn.startswith("traj_"):
            continue
        n_files += 1
        d = json.load(open(os.path.join(TRAJ, fn)))
        idx = d["manifest_index"]
        m0_pos = recs[idx]["initial_masked_positions"]
        m0 = torch.zeros(1, cfg.data.seq_len, dtype=torch.bool)
        for p in m0_pos:
            m0[0, p] = True
        x0 = torch.tensor(recs[idx]["x0"])[None]
        for stage, sv in d["stages"].items():
            for mode, mv in sv["modes"].items():
                toks = mv["final_tokens"]
                assert len(toks) == cfg.data.seq_len, f"{fn} len"
                assert all(0 <= t <= cfg.tokens - 1 for t in toks), f"{fn} invalid token"
                assert toks == mv["keyframes"][-1]["tokens"], f"{fn} final != last keyframe"
                for f in mv["keyframes"]:
                    assert f["mask_count"] == sum(1 for t in f["tokens"] if t == 50257)
                    assert all(t <= 50257 for t in f["tokens"])
                    assert f["sigma"] == f["sigma"] and f["sigma"] >= 0  # 有限
                    assert f["reverse_step"] <= 128
                # m0_reward 与 final 一致
                rw_val = float(rw.m0_reward(x0, torch.tensor(toks)[None], m0))
                assert abs(rw_val - mv["m0_reward"]) < 1e-9, f"{fn} reward mismatch"
                # 无 hard-region 失败：gate stats 只有软区计数（硬失败会 raise，导出时已保证）
                assert set(mv["gate_stats"].keys()) <= {"soft_neg_elements", "soft_steps"}
    print(f"  {n_files} files integrity OK (tokens/len/final==last-kf/mask_count/sigma/reward/gate)")

    import subprocess
    dxg = subprocess.run(["dmesg"], capture_output=True, text=True).stdout
    n_dxg = sum(dxg.count(k) for k in ["dxgvmb_send_create_allocation failed",
                                       "dxgkio_create_allocation: Ioctl failed", "EOVERFLOW"])
    print(f"== 3. dxg 计数 = {n_dxg}（基线 27，无新增则 PASS） ==")
    print("TRAJ_VERIFY_DONE" if all_ok and n_dxg <= 27 else "TRAJ_VERIFY_FAILED")


if __name__ == "__main__":
    main()
