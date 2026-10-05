"""协议 v1.2 §1.2 公式与 graph_lib.Absorbing.score_entropy 的 CPU 数值对拍。"""
import unittest

import torch

import graph_lib

V = 7          # clean vocab 7 → dim 8（含 [MASK]）


def se_reference(score, sigma, x, x0, dim):
    return graph_lib.Absorbing(V).score_entropy(score, sigma, x, x0)


def se_manual(score, sigma, x, x0, dim):
    """v1.2 §1.2：pos_term = Σ_{v≠MASK} exp(ℓ_v)；neg = r·ℓ_GT；const = r(log r −1)，r=1/(e^σ−1)。"""
    mask = dim - 1
    rel = x == mask
    esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
    r = 1 / esigm1
    pos = score[..., :-1].exp().sum(-1)
    neg = r * torch.gather(score, -1, x0[..., None]).squeeze(-1)
    const = r * (r.log() - 1)
    out = torch.zeros_like(score[..., 0])
    out[rel] = (pos - neg + const)[rel]
    return out


class TestLossFormula(unittest.TestCase):
    def _case(self, B=2, L=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        dim = V + 1
        score = torch.randn(B, L, dim, generator=g)
        x0 = torch.randint(0, V, (B, L), generator=g)
        sigma = torch.rand(B, 1, generator=g) * 2
        x = x0.clone()
        x[0, 1], x[0, 3], x[1, 2], x[1, 5] = dim - 1, dim - 1, dim - 1, dim - 1
        return score, sigma, x, x0, dim

    def test_masked_positions_match(self):
        score, sigma, x, x0, dim = self._case()
        ref = se_reference(score, sigma, x, x0, dim)
        man = se_manual(score, sigma, x, x0, dim)
        self.assertTrue(torch.allclose(ref, man, atol=1e-6), (ref - man).abs().max())

    def test_unmasked_positions_zero(self):
        score, sigma, x, x0, dim = self._case(seed=1)
        ref = se_reference(score, sigma, x, x0, dim)
        self.assertTrue((ref[x != dim - 1] == 0).all())

    def test_multiple_seeds(self):
        for s in range(5):
            score, sigma, x, x0, dim = self._case(seed=s + 100)
            ref = se_reference(score, sigma, x, x0, dim)
            man = se_manual(score, sigma, x, x0, dim)
            self.assertTrue(torch.allclose(ref, man, atol=1e-6), s)


if __name__ == "__main__":
    unittest.main()
