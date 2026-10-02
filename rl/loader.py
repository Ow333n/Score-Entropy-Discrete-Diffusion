"""RL 模型加载 (plan v0.2 §12/§13): SFT EMA-10200 → RL init。

与 evaluation/eval_*.py 同款加载口径 (Day 4 踩坑: EMA.state_dict() 是
{decay, num_updates, shadow_params} 非按参数名索引, 必须走 ema.load_state_dict
+ copy_to, 直接 model.load_state_dict(ema_dict) 会静默空载)。
"""
import os

import torch
from hydra import initialize, compose

from model import SEDD
from model.ema import ExponentialMovingAverage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_model(device):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    return SEDD(cfg).to(device), cfg


def load_rl_init(model_path, ckpt, device):
    """加载 checkpoint 的 EMA shadow 到新模型 (RL 初始化)。

    返回 (model, info): info 含 step / ema_decay / num_updates / n_shadow /
    shadow_vs_model 逐 tensor 一致校验结果。
    """
    model, cfg = build_model(device)
    model.eval()
    ckpt_path = os.path.join(model_path, ckpt)
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    ema.load_state_dict(loaded["ema"])
    ema.copy_to(model.parameters())
    shadow = ema.shadow_params
    model_params = list(model.parameters())
    mismatch = sum(
        0 if torch.equal(s.to("cpu"), p.detach().to("cpu")) else 1
        for s, p in zip(shadow, model_params))
    info = dict(
        step=int(loaded["step"]),
        ema_decay=float(loaded["ema"]["decay"]),
        ema_num_updates=int(loaded["ema"]["num_updates"]),
        n_shadow=len(shadow),
        n_model_params=len(model_params),
        shadow_vs_model_mismatches=mismatch,
        raw_step=int(loaded["step"]),
    )
    del loaded
    torch.cuda.empty_cache()
    return model, info
