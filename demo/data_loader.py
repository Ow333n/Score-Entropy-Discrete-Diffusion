"""Demo data loader（PRECOMPUTED MODE，纯 JSON，不 import torch / transformers，无 GPU）。

所有数据来自 demo_assets/ 预计算资产。数值 provenance 见各资产文件内字段。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "demo_assets")

_cache = {}


def _load(rel):
    if rel not in _cache:
        with open(os.path.join(ASSETS, rel)) as f:
            _cache[rel] = json.load(f)
    return _cache[rel]


def load_vocab():
    return _load("vocab_decode.json")


def decode_tokens(tokens, mask_positions=None, vocab=None):
    """token ids → 展示文本（纯 Python 解码，MASK → [MASK]）。"""
    v = vocab or load_vocab()
    mset = set(mask_positions or [])
    out = []
    for p, t in enumerate(tokens):
        if p in mset:
            out.append("[MASK]")
        else:
            out.append(v.get(str(t), "<?>").replace("Ġ", " "))
    return "".join(out)


def load_examples():
    return _load("examples.json")["examples"]


def example_by_index(idx):
    for e in load_examples():
        if e["manifest_index"] == idx:
            return e
    raise KeyError(idx)


def load_trajectory(idx):
    return _load(f"trajectories/traj_s{idx:03d}.json")


_manifest_cache = None


def load_manifest_tokens():
    """解析 frozen manifest，返回 {sample_id: {"x0": [gt token ids],
    "initial_state": [...], "initial_masked_positions": [...]}}（纯数据，不跑模型）。"""
    global _manifest_cache
    if _manifest_cache is None:
        path = os.path.join(ROOT, "manifests", "regime_a_eval_v1.jsonl")
        out = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                out[d["sample_id"]] = {
                    "x0": d["x0"],
                    "initial_state": d["initial_state"],
                    "initial_masked_positions": d["initial_masked_positions"],
                }
        _manifest_cache = out
    return _manifest_cache


def load_pair(stage, idx):
    return _load(f"compatibility_examples/pair_{stage}_s{idx:03d}.json")


def load_og_example(idx):
    return _load(f"compatibility_examples/og_example_s{idx:03d}.json")


def load_harmonized(stage, kind):
    return _load(f"metrics/harmonized/{stage}_{kind}_current.json")


def load_curves():
    return _load("metrics/formal_curves.json")


def stage_labels():
    return [("pretrained", "Pretrained"), ("sft", "SFT (s1-10200 EMA)"),
            ("rl_raw", "RL-500 (RAW)"), ("rl_ema", "RL-500 (EMA, secondary)")]


def main_chain_labels():
    return [("pretrained", "Pretrained"), ("sft", "SFT"),
            ("rl_raw", "RL-500")]
