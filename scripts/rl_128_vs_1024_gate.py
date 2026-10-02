"""D 步: analytic 128-step vs 1024-step gate (review 定案顺序 D, 不训练)。

同一 SFT s1 EMA-10200、frozen manifest 前 N 样本的 initial_state,
从样本自身 σ 出发 reverse 到 eps, 比较:
  M0 重建 acc / 残存 MASK 率 / reward 分布 / runtime / 无效状态频率
输出: results/rl_gates/128_vs_1024.json
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hydra import initialize, compose
from model import SEDD
from model.ema import ExponentialMovingAverage
from model import utils as mutils
import graph_lib, noise_lib
from catsample import sample_categorical
from rl import transition as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_SAMPLES = 32
SEED = 20261002

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    model = SEDD(cfg).to(device).eval()
    ckpt = torch.load(os.path.join(ROOT, "exp_local/regime_a/formal-vanilla-s1-191414/checkpoint_10200.pth"),
                      map_location="cpu", weights_only=False)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    ema.load_state_dict(ckpt["ema"]); ema.copy_to(model.parameters())
    del ckpt; torch.cuda.empty_cache()

    graph = graph_lib.Absorbing(cfg.tokens)
    noise = noise_lib.LogLinearNoise()
    score_fn = mutils.get_score_fn(model, train=False, sampling=True)
    D = cfg.tokens + 1; MASK = cfg.tokens; eps = 1e-3

    recs = [json.loads(l) for l in open(os.path.join(ROOT, "manifests/regime_a_eval_v1.jsonl"))][:N_SAMPLES]
    x0 = torch.stack([torch.tensor(r["x0"]) for r in recs]).to(device)
    x_t0 = torch.stack([torch.tensor(r["initial_state"]) for r in recs]).to(device)
    sigma0 = torch.tensor([r["sigma"] for r in recs], device=device)
    span = torch.stack([torch.tensor(r["span_mask"]) for r in recs]).to(device) if "span_mask" in recs[0] else None
    if span is None:  # manifest 无 span_mask 字段则用 MASK 位置近似 M0
        m0 = (x_t0 == MASK)
    else:
        m0 = span & (x_t0 == MASK)
    print(f"N={N_SAMPLES}, |M0| per sample: {m0.sum(-1).tolist()}")

    def rollout(steps, rng_seed):
        torch.manual_seed(rng_seed)
        t0 = ((1 - (-sigma0).exp()) / (1 - eps)).clamp(min=eps + 1e-6)  # σ→t 逆
        t0i = time.time()
        n_neg = 0; n_invalid = 0
        accs, resids = [], []
        with torch.no_grad():
            # 分 chunk 处理 (one_hot int64 [B,256,50258] 大, batch 32 会 OOM)
            for lo in range(0, N_SAMPLES, 8):
                hi = min(lo + 8, N_SAMPLES)
                x = x_t0[lo:hi].clone()
                t_cur = t0[lo:hi]
                for i in range(steps):
                    t_next = eps + (t0[lo:hi] - eps) * (1 - (i + 1) / steps)
                    sigma_cur = noise.total_noise(t_cur)
                    sigma_next = noise.total_noise(t_next)
                    dsigma = (sigma_cur - sigma_next)[:, None]
                    s = score_fn(x, sigma_cur.squeeze(-1))
                    stag = graph.staggered_score(s, dsigma)
                    trans = graph.transp_transition(x, dsigma)
                    w = stag * trans
                    ok, stats = T.validity_check(w)
                    n_neg += stats["n_neg"]
                    x = sample_categorical(w)
                    t_cur = t_next
                # denoiser (末尾步, 与 sampling.py 一致)
                s = score_fn(x, sigma_next.squeeze(-1))
                stag = graph.staggered_score(s, sigma_next[:, None])
                probs = stag * graph.transp_transition(x, sigma_next[:, None])
                probs = probs[..., :-1]
                x = sample_categorical(probs)
                accs.append(((x == x0[lo:hi]) & m0[lo:hi]).sum(-1).float()
                            / m0[lo:hi].sum(-1).clamp(min=1).float())
                resids.append((x == MASK).sum(-1).float())
        dt = time.time() - t0i
        acc = torch.cat(accs); residual_mask = torch.cat(resids)
        return dict(
            steps=steps, acc=acc.tolist(), residual_mask=residual_mask.tolist(),
            reward=acc.tolist(), runtime_s=dt, n_neg=n_neg, n_invalid=n_invalid,
            reward_mean=float(acc.mean()), reward_std=float(acc.std()),
            residual_mask_mean=float(residual_mask.mean()),
        )

    out = {}
    for steps in (128, 1024):
        r = rollout(steps, SEED)
        out[str(steps)] = r
        print(f"steps={steps}: M0-acc={r['reward_mean']:.4f}±{r['reward_std']:.4f} "
              f"residual_MASK={r['residual_mask_mean']:.3f} runtime={r['runtime_s']:.1f}s "
              f"neg_weights={r['n_neg']} invalid={r['n_invalid']}")

    os.makedirs(os.path.join(ROOT, "results/rl_gates"), exist_ok=True)
    out["meta"] = dict(model="formal-vanilla-s1-191414/checkpoint_10200 EMA",
                       n_samples=N_SAMPLES, seed=SEED, predictor="analytic",
                       manifest_sha="1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3")
    json.dump(out, open(os.path.join(ROOT, "results/rl_gates/128_vs_1024.json"), "w"), indent=2)
    print("saved results/rl_gates/128_vs_1024.json")

if __name__ == "__main__":
    main()
