"""Phase 6 core gate: prove the rejection sampler is lossless (§8.3, §14).

Model-independent Monte-Carlo test: for arbitrary p^d != p^t, the accepted-or-resampled
token distribution must equal p^t exactly. We draw many samples and assert empirical KL
to p^t is ~0, for several adversarial p^d (uniform, peaked-wrong, shifted). If this holds,
routing (which only changes p^d) can never change the output -> every variant is lossless.

Run: CUDA_VISIBLE_DEVICES="" $PY scripts/test_losslessness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.sampler import accept_or_resample, sample_from  # noqa: E402


def empirical_kl(counts: torch.Tensor, pt: torch.Tensor) -> float:
    q = counts / counts.sum()
    m = pt > 0
    return float((pt[m] * (pt[m] / q[m].clamp_min(1e-12)).log()).sum())


def run(V=32, N=400_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    pt = torch.rand(V, generator=g) + 0.05
    pt = pt / pt.sum()

    drafts = {
        "uniform": torch.ones(V) / V,
        "peaked_wrong": torch.softmax(torch.arange(V).float() * 0.3, 0).flip(0),
        "pt_itself": pt.clone(),
        "noisy_pt": (pt + 0.3 * torch.rand(V, generator=g)).clamp_min(1e-6),
    }
    print(f"V={V} N={N} samples per draft")
    worst = 0.0
    for name, pd in drafts.items():
        pd = pd / pd.sum()
        ptb = pt.unsqueeze(0).expand(N, V)
        pdb = pd.unsqueeze(0).expand(N, V)
        u_tok = torch.rand(N, generator=g)
        u_acc = torch.rand(N, generator=g)
        u_res = torch.rand(N, generator=g)
        tok = sample_from(pdb, u_tok)                        # draw t ~ p^d
        _, out = accept_or_resample(ptb, pdb, tok, u_acc, u_res)
        counts = torch.bincount(out, minlength=V).float()
        kl = empirical_kl(counts, pt)
        acc = float((out == tok).float().mean())
        worst = max(worst, kl)
        print(f"  {name:14s} accept_rate={acc:.3f}  empirical_KL(p^t||emp)={kl:.5f}")
    ok = worst < 1e-2
    print(f"\nworst KL={worst:.5f}  RESULT: {'LOSSLESS OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
