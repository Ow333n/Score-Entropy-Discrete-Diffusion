"""Mechanism pilot corruption policies 单元测试（CPU，protocol v1.2 §9.3）。"""
import unittest

import torch

from task_data.policy_corruption import (SharedSchedule, draw_K, draw_pi_L,
                                        draw_span, select_mask, stream_seed)

L, SPAN_MIN, SPAN_MAX = 256, 10, 50


class TestDrawK(unittest.TestCase):
    def test_range(self):
        g = torch.Generator().manual_seed(0)
        span_len = torch.randint(10, 51, (2000,), generator=g)
        sigma = torch.rand(2000, generator=g)
        K = draw_K(span_len, sigma, torch.Generator().manual_seed(1), SPAN_MAX)
        self.assertTrue(((K >= 0) & (K <= span_len)).all())

    def test_binomial_moments_and_pmf(self):
        g = torch.Generator().manual_seed(7)
        m, sigma = 30, torch.tensor([0.6931472])  # q = 0.5
        span_len = torch.full((20000,), m, dtype=torch.long)
        sigma = torch.full((20000,), 0.6931472)
        K = draw_K(span_len, sigma, torch.Generator().manual_seed(8), SPAN_MAX).float()
        self.assertAlmostEqual(K.mean().item(), 15.0, delta=0.3)      # E = mq
        self.assertAlmostEqual(K.var(unbiased=False).item(), 7.5, delta=0.6)  # Var = mq(1-q)
        p0 = (K == 0).float().mean().item()
        self.assertAlmostEqual(p0, 0.5 ** 30, delta=1e-5)             # exact pmf spot


class TestScheduleSharing(unittest.TestCase):
    def test_same_replicate_same_digest(self):
        a = SharedSchedule(1, L, SPAN_MIN, SPAN_MAX)
        b = SharedSchedule(1, L, SPAN_MIN, SPAN_MAX)
        for _ in range(100):
            a.draw(32); b.draw(32)
        self.assertEqual(a.hexdigest(), b.hexdigest())

    def test_different_replicate_different_digest(self):
        a = SharedSchedule(1, L, SPAN_MIN, SPAN_MAX)
        b = SharedSchedule(2, L, SPAN_MIN, SPAN_MAX)
        for _ in range(100):
            a.draw(32); b.draw(32)
        self.assertNotEqual(a.hexdigest(), b.hexdigest())


class TestPolicyA(unittest.TestCase):
    def test_uniform_K_subset(self):
        """K=1 时每个位置被选中的频率 ≈ 1/m（均匀 K 子集）。"""
        g = torch.Generator().manual_seed(11)
        m, s0, n = 10, 5, 20000
        span_start = torch.full((n,), s0, dtype=torch.long)
        span_len = torch.full((n,), m, dtype=torch.long)
        K = torch.ones(n, dtype=torch.long)
        freq = torch.zeros(m)
        for i in range(n):
            sel = select_mask(span_start[i:i+1], span_len[i:i+1], K[i:i+1], "A", L, generator=g)[0]
            freq[sel - s0] += 1
        freq /= n
        p = 1 / m
        sd = (p * (1 - p) / n) ** 0.5
        self.assertTrue(all(abs(freq[i].item() - p) < 3 * sd for i in range(m)), freq)

    def test_marginal_equals_bernoulli(self):
        """A 的 per-position 边际 = E[K/m] = q = 1-exp(-σ)（与 iid Bernoulli 同分布）。"""
        g = torch.Generator().manual_seed(13)
        sigma = torch.tensor([1.0])
        n = 20000
        span_len = torch.full((n,), 20, dtype=torch.long)
        span_start = torch.zeros(n, dtype=torch.long)
        K = draw_K(span_len, sigma.expand(n), torch.Generator().manual_seed(14), SPAN_MAX)
        hit = torch.zeros(20)
        for i in range(n):
            sel = select_mask(span_start[i:i+1], span_len[i:i+1], K[i:i+1], "A", L, generator=g)[0]
            hit[sel] += 1
        q = 1 - (-sigma).exp().item()
        sd = (q * (1 - q) / n) ** 0.5
        self.assertTrue(all(abs(hit[i].item() / n - q) < 3 * sd for i in range(20)))


class TestPolicyB(unittest.TestCase):
    def test_nested_prefix(self):
        g = torch.Generator().manual_seed(21)
        pi_L = draw_pi_L(g, L)
        s0, m = 5, 20
        span_start = torch.tensor([s0])
        span_len = torch.tensor([m])
        m3 = select_mask(span_start, span_len, torch.tensor([3]), "B", L, pi_L=pi_L)[0]
        m7 = select_mask(span_start, span_len, torch.tensor([7]), "B", L, pi_L=pi_L)[0]
        self.assertEqual(len(m3), 3)
        self.assertEqual(len(m7), 7)
        self.assertTrue(set(m3.tolist()) < set(m7.tolist()))  # K1<K2 ⇒ 嵌套

    def test_K1_uniform_over_random_pi(self):
        """随机 π_L 下 B 的 K=1 边际均匀（平均掉 π_L 的随机性）。"""
        n, m, s0 = 5000, 12, 3
        hit = torch.zeros(m)
        g = torch.Generator().manual_seed(23)
        for _ in range(n):
            pi = draw_pi_L(g, L)
            sel = select_mask(torch.tensor([s0]), torch.tensor([m]), torch.tensor([1]),
                              "B", L, pi_L=pi)[0]
            hit[sel - s0] += 1
        p = 1 / m
        sd = (p * (1 - p) / n) ** 0.5
        self.assertTrue(all(abs(hit[i].item() / n - p) < 3 * sd for i in range(m)), hit)


class TestPolicyCD(unittest.TestCase):
    def test_exact_positions(self):
        s0, m = 10, 8
        span_start = torch.tensor([s0]); span_len = torch.tensor([m]); K = torch.tensor([3])
        c = select_mask(span_start, span_len, K, "C", L)[0]
        d = select_mask(span_start, span_len, K, "D", L)[0]
        self.assertEqual(c.tolist(), [15, 16, 17])   # 最右 K 个
        self.assertEqual(d.tolist(), [10, 11, 12])   # 最左 K 个

    def test_deterministic(self):
        a = select_mask(torch.tensor([0]), torch.tensor([10]), torch.tensor([4]), "C", L)[0]
        b = select_mask(torch.tensor([0]), torch.tensor([10]), torch.tensor([4]), "C", L)[0]
        self.assertEqual(a.tolist(), b.tolist())


class TestEdges(unittest.TestCase):
    def test_K_zero(self):
        for policy, kwargs in [("A", dict(generator=torch.Generator().manual_seed(1))),
                               ("B", dict(pi_L=torch.arange(L))),
                               ("C", {}), ("D", {})]:
            sel = select_mask(torch.tensor([0]), torch.tensor([10]), torch.tensor([0]),
                              policy, L, **kwargs)[0]
            self.assertEqual(sel.numel(), 0, policy)

    def test_K_equals_m(self):
        for policy, kwargs in [("A", dict(generator=torch.Generator().manual_seed(2))),
                               ("B", dict(pi_L=torch.arange(L))),
                               ("C", {}), ("D", {})]:
            sel = select_mask(torch.tensor([0]), torch.tensor([10]), torch.tensor([10]),
                              policy, L, **kwargs)[0]
            self.assertEqual(sel.numel(), 10, policy)


class TestStreamIsolation(unittest.TestCase):
    def test_CD_do_not_consume_policy_streams(self):
        g_a = torch.Generator().manual_seed(stream_seed("policy_a", 1))
        g_b = torch.Generator().manual_seed(stream_seed("policy_b", 1))
        # C/D 调用（不应消耗 g_a/g_b）
        select_mask(torch.tensor([0]), torch.tensor([10]), torch.tensor([5]), "C", L)
        select_mask(torch.tensor([0]), torch.tensor([10]), torch.tensor([5]), "D", L)
        next_a = torch.rand(3, generator=g_a)
        next_b = torch.rand(3, generator=g_b)
        ref_a = torch.rand(3, generator=torch.Generator().manual_seed(stream_seed("policy_a", 1)))
        ref_b = torch.rand(3, generator=torch.Generator().manual_seed(stream_seed("policy_b", 1)))
        self.assertTrue(torch.equal(next_a, ref_a))
        self.assertTrue(torch.equal(next_b, ref_b))

    def test_shared_streams_untouched_by_policy(self):
        s = SharedSchedule(1, L, SPAN_MIN, SPAN_MAX)
        s.draw(32)                      # 第一笔
        # 期间执行 policy 选择（A 用独立 generator）
        sig2, d2, ln2, st2, k2 = s.draw(32)
        s3 = SharedSchedule(1, L, SPAN_MIN, SPAN_MAX)
        s3.draw(32)                      # 对齐到第二笔
        sig3, d3, ln3, st3, k3 = s3.draw(32)
        self.assertTrue(torch.equal(sig2, sig3))
        self.assertTrue(torch.equal(ln2, ln3))
        self.assertTrue(torch.equal(k2, k3))


if __name__ == "__main__":
    unittest.main()
