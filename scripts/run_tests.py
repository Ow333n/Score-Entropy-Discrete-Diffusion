"""G0 单元测试运行器 (全部 CPU)。

用法: .venv/bin/python scripts/run_tests.py
覆盖 protocol §4.2/§4.4/§4.5/§4.6/§4.8/§4.9 的玩具级验证。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_MODULES = [
    "compatibility.tests.test_posterior",
    "compatibility.tests.test_swap_delta",
    "compatibility.tests.test_strict_reveal",
    "compatibility.tests.test_swap_gradient",
    "compatibility.tests.test_v21_policy",
]


def main():
    n_pass = 0
    for mod_name in TEST_MODULES:
        mod = importlib.import_module(mod_name)
        tests = sorted((k, v) for k, v in vars(mod).items() if k.startswith("test_"))
        tests = [v for _, v in tests]
        for t in tests:
            t()
            n_pass += 1
            print(f"PASS {mod_name.split('.')[-1]}::{t.__name__}")
    print(f"\n全部 {n_pass} 个 G0 单元测试通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
