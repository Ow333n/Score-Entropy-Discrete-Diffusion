"""RL correctness 单测运行器 (plan v0.2 §34, 全部 CPU)。

用法: .venv/bin/python scripts/run_rl_tests.py
覆盖 §34 的 1-13, 15-22 (CPU 部分); §34.14 (RL step0) 在 RL-G0 阶段执行。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_MODULES = [
    "tests.test_rl_transition",
    "tests.test_rl_reward",
]


def main():
    n_pass = 0
    for mod_name in TEST_MODULES:
        mod = importlib.import_module(mod_name)
        tests = sorted((k, v) for k, v in vars(mod).items() if k.startswith("test_"))
        for name, t in tests:
            t()
            n_pass += 1
            print(f"PASS {mod_name.split('.')[-1]}::{name}")
    print(f"\n全部 {n_pass} 个 RL 单测通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
