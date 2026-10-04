"""Phase C1: Demo trajectory keyframe export（inference-only，无 backward，无 PG J 语义）。

对 demo_assets/examples.json 的每个样本 × 4 stages:
  pretrained (HF louaaron/sedd-small) / sft (s1-10200 EMA) / rl_raw (rlpilot-185545
  checkpoint_step500.pth raw) / rl_ema (同 checkpoint EMA, secondary)
× 2 decoding modes（sampled 固定 RNG seed / greedy 天然 deterministic）导出:

  - 128 步 rollout 的关键帧（~10 帧：step0 + top-8 |Δmask_count| + final）
    每帧: reverse_step / sigma / tokens / mask_positions / newly_revealed_positions / mask_count
  - final sampled / greedy 输出文本 + M0 exact-token reward

Reproducibility 冻结: sample_id + x0 + span + initial corruption + sigma/schedule +
checkpoint + decoding mode + sampled RNG seed(=10000+manifest_index)。不含 PG timestep J。

计算与 rl/rollout.py rollout_chunk 同源代码（staggered/validity/denoiser 逐行一致，
仅去掉 J 采样 + 记录关键帧）；σ≥0.05 区硬门失败 → 直接抛错（export 失败）。
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
from model import utils as mutils
from model.ema import ExponentialMovingAverage
from rl import loader
from rl import transition as T
from rl import reward as rw
from catsample import sample_categorical
from transformers import GPT2TokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "demo_assets/examples.json")
OUT_DIR = os.path.join(ROOT, "demo_assets/trajectories")
MANIFEST = os.path.join(ROOT, "manifests/regime_a_eval_v1.jsonl")
SIGMA_GATE_MIN = 0.05
EPS = 1e-3
SEED_BASE = 10000          # sampled RNG seed = SEED_BASE + manifest_index（冻结）
N_KEYFRAMES = 10
tok = GPT2TokenizerFast.from_pretrained("gpt2")


def decode(ids, mask_positions=None):
    out, mset = [], set(mask_positions or [])
    for p, t in enumerate(ids):
        out.append("[MASK]" if p in mset else tok.decode([t]))
    return "".join(out).replace("Ġ", " ").strip()


def export_rollout(model, graph, noise, x_t0, sigma0, steps, seed, greedy, mask_token):
    """与 rollout_chunk 同源代码的 inference rollout（无 J），返回关键帧 + final。"""
    torch.manual_seed(seed)
    device = x_t0.device
    x_t0 = x_t0.clone()
    t0 = ((1 - (-sigma0).exp()) / (1 - EPS)).clamp(min=EPS + 1e-6)
    x = x_t0
    t_cur = t0
    score_fn = mutils.get_score_fn(model, train=False, sampling=True)
    states = []   # 每 iteration 后状态
    sigmas = []
    gate_stats = dict(soft_neg_elements=0, soft_steps=0)
    with torch.no_grad():
        for i in range(steps):
            t_next = EPS + (t0 - EPS) * (1 - (i + 1) / steps)
            sigma_cur = noise.total_noise(t_cur)
            sigma_next = noise.total_noise(t_next)
            dsigma = (sigma_cur - sigma_next)[:, None]
            s = score_fn(x, sigma_cur.squeeze(-1))
            stag = T.staggered_score_fn(s, dsigma)
            trans = graph.transp_transition(x, dsigma)
            w = stag * trans
            ok, stats = T.validity_check(w)
            if not ok:
                if (sigma_cur >= SIGMA_GATE_MIN).all():
                    raise RuntimeError(
                        f"hard-region validity failure step={i}: {stats} | σ={float(sigma_cur)}")
                gate_stats["soft_neg_elements"] += stats["n_neg"]
                gate_stats["soft_steps"] += 1
            x_next = w.argmax(dim=-1) if greedy else sample_categorical(w)
            x = x_next
            t_cur = t_next
            states.append(x[0].cpu().tolist())
            sigmas.append(float(noise.total_noise(t_next)[0]))
        # denoiser 收尾（与 sampling.py / rollout_chunk 一致）
        s = score_fn(x, sigma_next.squeeze(-1))
        stag = T.staggered_score_fn(s, sigma_next[:, None])
        probs = stag * graph.transp_transition(x, sigma_next[:, None])
        probs = probs[..., :-1]
        final = (probs.argmax(dim=-1) if greedy else sample_categorical(probs))[0].cpu().tolist()
    states.append(final)
    sigmas.append(float(noise.total_noise(torch.tensor([EPS], device=device))[0]))
    return states, sigmas, gate_stats


def pick_keyframes(states, sigmas, n_kf=N_KEYFRAMES):
    """step0 + top-(n_kf−2) |Δmask_count| + final，按 step 排序。"""
    mask_counts = [sum(1 for t in s if t == 50257) for s in states]
    deltas = [abs(mask_counts[i] - mask_counts[i - 1]) for i in range(1, len(states))]
    top = sorted(range(len(deltas)), key=lambda k: (-deltas[k], k))[: n_kf - 2]
    kf = sorted({0, *[i + 1 for i in top], len(states) - 1})
    frames = []
    prev_tokens = None
    for i in kf:
        tokens = states[i]
        newly = []
        if prev_tokens is not None:
            newly = [p for p in range(len(tokens))
                     if prev_tokens[p] == 50257 and tokens[p] != 50257]
        frames.append(dict(
            reverse_step=i, sigma=round(sigmas[i], 6),
            tokens=tokens,
            mask_positions=[p for p, t in enumerate(tokens) if t == 50257],
            newly_revealed_positions=newly,
            mask_count=sum(1 for t in tokens if t == 50257)))
        prev_tokens = tokens
    return frames


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    device = torch.device("cuda")
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    recs = [json.loads(l) for l in open(MANIFEST)]
    ex = json.load(open(EXAMPLES))["examples"]

    # 4 stages 的模型加载器
    def load_pretrained():
        hf = SEDD.from_pretrained("louaaron/sedd-small")
        m = SEDD(cfg).to(device).eval()
        m.load_state_dict(hf.state_dict(), strict=False)
        del hf
        return m

    def load_sft():
        m, info = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
        assert info["step"] == 10200
        return m

    def load_rl(w):
        m, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
        snap = torch.load(os.path.join(ROOT, "exp_local/regime_a/rlpilot-185545",
                                       "checkpoint_step500.pth"),
                          map_location="cpu", weights_only=False)
        if w == "raw":
            m.load_state_dict(snap["model"], strict=False)
        else:
            ema = ExponentialMovingAverage(m.parameters(), decay=0.9999)
            ema.load_state_dict(snap["ema"])
            ema.copy_to(m.parameters())
        return m

    stages = [("pretrained", load_pretrained), ("sft", load_sft),
              ("rl_raw", lambda: load_rl("raw")), ("rl_ema", lambda: load_rl("ema"))]

    # 权重零突变守卫（整个 export 前后 checksum 一致）
    probe, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    checksum0 = sum(p.detach().float().sum().item() for p in probe.parameters())
    del probe
    torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_files = []
    for st in ex:
        idx = int(st["manifest_index"])
        r = recs[idx]
        x_t0 = torch.tensor(r["initial_state"], device=device)[None]
        sigma0 = torch.tensor([float(r["sigma"])], device=device)
        m0 = torch.zeros(1, cfg.data.seq_len, dtype=torch.bool, device=device)
        for p in r["initial_masked_positions"]:
            m0[0, p] = True
        x0 = torch.tensor(r["x0"], device=device)[None]

        import subprocess
        payload = dict(sample_id=st["sample_id"], manifest_index=idx,
                       seed_base=SEED_BASE, n_reverse_steps=cfg.rl.rollout_steps,
                       export_git_head=subprocess.run(
                           ["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True).stdout.strip() or "unknown",
                       stages={})
        for sname, loader_fn in stages:
            model = loader_fn()
            model.eval()
            modes = {}
            for mode, greedy in [("sampled", False), ("greedy", True)]:
                seed = SEED_BASE + idx
                states, sigmas, gst = export_rollout(
                    model, graph, noise, x_t0, sigma0, cfg.rl.rollout_steps,
                    seed, greedy=greedy, mask_token=cfg.tokens)
                final_tokens = states[-1]
                final = torch.tensor(final_tokens, device=device)[None]
                reward = float(rw.m0_reward(x0, final, m0))
                modes[mode] = dict(
                    seed=seed,
                    keyframes=pick_keyframes(states, sigmas),
                    final_tokens=final_tokens,
                    final_text=decode(final_tokens),
                    m0_reward=reward,
                    gate_stats=gst)
            payload["stages"][sname] = dict(modes=modes)
            del model
            torch.cuda.empty_cache()
        path = os.path.join(OUT_DIR, f"traj_s{idx:03d}.json")
        with open(path, "w") as f:
            json.dump(payload, f)
        out_files.append(path)
        print(f"exported {os.path.basename(path)}: "
              f"sampled r={payload['stages']['rl_raw']['modes']['sampled']['m0_reward']:.4f} "
              f"greedy r={payload['stages']['rl_raw']['modes']['greedy']['m0_reward']:.4f}")

    probe2, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    checksum1 = sum(p.detach().float().sum().item() for p in probe2.parameters())
    assert abs(checksum0 - checksum1) < 1e-6, "weight mutation detected!"
    print(f"weight checksum guard: OK ({checksum0:.6f})")
    print(f"EXPORT_DONE: {len(out_files)} trajectory files")


if __name__ == "__main__":
    main()
