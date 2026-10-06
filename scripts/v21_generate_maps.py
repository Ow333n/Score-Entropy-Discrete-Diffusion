"""v2.1 frozen pair/bank map 生成（protocol mechanism_complementary_exposure_v2_1 §3.1/§6/§9，
preflight P1）。

一次性生成（训练前，生成后冻结）：
- 每 replicate r∈{1,2} × even m∈{10,12,…,50}：frozen perfect matching
  （frozen permutation π_r^{(m)} 连续配对）
- 每 replicate r × odd m∈{11,13,…,49}：balanced persistent matching bank
  （π_r^{(m)} 旋转 matching、singleton 轮换；每位置恰 1 次 singleton）
- heldout map：fresh permutation π_h^{(m)} 的 skip-2 边（m 条边/m）；
  与 rep1/rep2 treated 边集逐 m 交集断言为空（确定性 rejection 循环）
- §9.5 OrderGap frozen 子集索引（64 个，均匀间隔规则，冻结）

种子（冻结，记录于 maps_manifest）：rep r × m → 10000 + 100*r + m；
heldout m → 11000 + m（rejection 时 seed 递增，实际使用值落盘）。

输出：exp_local/regime_a/mechpilot_v21_maps/
  maps_rep1.json / maps_rep2.json / maps_heldout.json + 各 .sha256
  + maps_manifest.json（三方边集、交集断言、生成元数据）
  + order_gap_subset_indices.json + .sha256

用法: .venv/bin/python scripts/v21_generate_maps.py
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "exp_local", "regime_a", "mechpilot_v21_maps")

MAP_SEED_BASE_REP = 10000   # rep r × m: 10000 + 100*r + m
MAP_SEED_BASE_HELDOUT = 11000  # heldout m: 11000 + m（rejection 递增）
SPAN_MIN, SPAN_MAX = 10, 50
SEQ_LEN = 256
ORDERGAP_SUBSET_N = 64


def randperm_m(m, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(m, generator=g).tolist()


def make_even_map(m, seed):
    pi = randperm_m(m, seed)
    pairs = [[pi[2 * c], pi[2 * c + 1]] for c in range(m // 2)]
    return dict(pi=pi, pairs=pairs, treated=pairs)


def make_odd_bank(m, seed):
    """balanced persistent matching bank（§6）：phase t: singleton=π[t]，
    pairs=(π[t+1],π[t+2]),(π[t+3],π[t+4]),…（下标 mod m）。treated = m 条 cycle 边。"""
    pi = randperm_m(m, seed)
    phases = {}
    for t in range(m):
        seq = [(t + 1 + kk) % m for kk in range(m - 1)]
        pairs = [[pi[seq[2 * c]], pi[seq[2 * c + 1]]] for c in range((m - 1) // 2)]
        phases[str(t)] = dict(singleton=pi[t], pairs=pairs)
    treated = [[pi[i], pi[(i + 1) % m]] for i in range(m)]
    return dict(pi=pi, phases=phases, treated=treated)


def make_heldout_map(m, seed, treated_sets_rep1, treated_sets_rep2):
    """skip-2 边：{(π(i), π(i+2)) mod m}；与 rep1/rep2 treated 边集交集为空
    （确定性 rejection，seed 递增，实际 seed 落盘）。"""
    used = seed
    while True:
        pi = randperm_m(m, used)
        edges = [[pi[i], pi[(i + 2) % m]] for i in range(m)]
        eset = {frozenset(e) for e in edges}
        if not (eset & treated_sets_rep1[m]) and not (eset & treated_sets_rep2[m]):
            return dict(pi=pi, pairs=edges, seed_used=used), eset
        used += 1


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_with_sidecar(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    digest = sha256_file(path)
    with open(path + ".sha256", "w") as f:
        f.write(digest + "\n")
    return digest


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()

    files = {}
    treated_sets = {}  # r -> {m -> set(frozenset edges)}

    for r in (1, 2):
        rep = dict(replicate=r, seq_len=SEQ_LEN, generated_by=os.path.basename(__file__))
        rep["even"] = {}
        rep["odd"] = {}
        treated_sets[r] = {}
        for m in range(SPAN_MIN, SPAN_MAX + 1):
            if m % 2 == 0:
                seed = MAP_SEED_BASE_REP + 100 * r + m
                rep["even"][str(m)] = make_even_map(m, seed)
            else:
                seed = MAP_SEED_BASE_REP + 100 * r + m
                rep["odd"][str(m)] = make_odd_bank(m, seed)
            treated_sets[r][m] = {frozenset(e) for e in rep[
                "even" if m % 2 == 0 else "odd"][str(m)]["treated"]}
        # 结构断言（P1）
        for m, entry in rep["even"].items():
            mi = int(m)
            covered = sorted(p for e in entry["pairs"] for p in e)
            assert covered == list(range(mi)), f"rep{r} m={m}: matching 未完美覆盖"
            assert len(entry["pairs"]) == mi // 2
            assert len(entry["treated"]) == mi // 2
        for m, entry in rep["odd"].items():
            mi = int(m)
            singletons = [entry["phases"][str(t)]["singleton"] for t in range(mi)]
            assert sorted(singletons) == list(range(mi)), \
                f"rep{r} m={m}: singleton 轮换不完整"
            assert len(entry["treated"]) == mi
            for t in range(mi):
                covered = [entry["phases"][str(t)]["singleton"]] + [
                    p for e in entry["phases"][str(t)]["pairs"] for p in e]
                assert sorted(covered) == list(range(mi)), \
                    f"rep{r} m={m} phase={t}: 未覆盖全体位置"
        fname = f"maps_rep{r}.json"
        digest = write_with_sidecar(os.path.join(OUT_DIR, fname), rep)
        files[fname] = digest
        print(f"maps_rep{r}.json: even={len(rep['even'])}m odd={len(rep['odd'])}m "
              f"sha256={digest[:16]}…")

    # heldout map（与 rep1/rep2 treated 边集不重叠）
    heldout = dict(heldout=True, skip=2, seq_len=SEQ_LEN,
                   generated_by=os.path.basename(__file__))
    heldout["even"] = {}
    heldout["odd"] = {}
    disjointness = {}
    for m in range(SPAN_MIN, SPAN_MAX + 1):
        seed = MAP_SEED_BASE_HELDOUT + m
        hm, eset = make_heldout_map(m, seed, treated_sets[1], treated_sets[2])
        if m % 2 == 0:
            heldout["even"][str(m)] = hm
        else:
            heldout["odd"][str(m)] = hm
        disjointness[str(m)] = dict(
            seed_used=hm["seed_used"],
            n_heldout_edges=len(hm["pairs"]),
            overlap_with_rep1=len(eset & treated_sets[1][m]),
            overlap_with_rep2=len(eset & treated_sets[2][m]),
            ok=(len(eset & treated_sets[1][m]) == 0 and len(eset & treated_sets[2][m]) == 0),
        )
        assert disjointness[str(m)]["ok"], f"m={m}: heldout 与训练 treated 边重叠"
    heldout["disjointness"] = disjointness
    fname = "maps_heldout.json"
    files[fname] = write_with_sidecar(os.path.join(OUT_DIR, fname), heldout)
    print(f"maps_heldout.json: 全部 m 不重叠断言 PASS, sha256={files[fname][:16]}…")

    # §9.5 OrderGap frozen 子集索引（64，均匀间隔规则，冻结）
    idx = [round(i * (499) / (ORDERGAP_SUBSET_N - 1)) for i in range(ORDERGAP_SUBSET_N)]
    assert len(set(idx)) == ORDERGAP_SUBSET_N
    subset = dict(rule="evenly_spaced_round", n=ORDERGAP_SUBSET_N,
                  manifest_n=500, indices=idx)
    fname = "order_gap_subset_indices.json"
    files[fname] = write_with_sidecar(os.path.join(OUT_DIR, fname), subset)
    print(f"order_gap_subset_indices.json: {ORDERGAP_SUBSET_N} 索引, "
          f"sha256={files[fname][:16]}…")

    manifest = dict(
        protocol="mechanism_complementary_exposure_v2_1",
        protocol_sha256="af312d864d41d8f67da343d55a8dee8a29a91c37dea622e6b2c4b22ef441fc61",
        generated_at="2026-10-06",
        git_commit=git_commit,
        map_seed_base_rep=MAP_SEED_BASE_REP,
        map_seed_base_heldout=MAP_SEED_BASE_HELDOUT,
        span_range=[SPAN_MIN, SPAN_MAX],
        seq_len=SEQ_LEN,
        order_gap_subset_rule="evenly_spaced_round(0..499, 64)",
        files=sorted(files),
        file_sha256={k: files[k] for k in sorted(files)},
    )
    mf = write_with_sidecar(os.path.join(OUT_DIR, "maps_manifest.json"), manifest)
    print(f"maps_manifest.json sha256={mf[:16]}…")
    print("全部 map 生成完成、落盘 + sha256 sidecar。")


if __name__ == "__main__":
    main()
