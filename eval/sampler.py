"""Speculative rejection sampling — the lossless core (§14).

For each drafted token t ~ p^d, accept with prob min(1, p^t(t)/p^d(t)); on rejection,
resample from the normalized residual (p^t - p^d)_+ . This makes the accepted-token
distribution EXACTLY p^t for any draft p^d. Routing/draft quality only affect how many
tokens are accepted (tau), never the output distribution.
"""

from __future__ import annotations

import torch


def residual_dist(pt: torch.Tensor, pd: torch.Tensor) -> torch.Tensor:
    """Normalized (p^t - p^d)_+ over the vocab (last dim)."""
    r = (pt - pd).clamp(min=0)
    s = r.sum(-1, keepdim=True)
    return torch.where(s > 0, r / s, pt)  # degenerate -> fall back to p^t


def accept_or_resample(pt: torch.Tensor, pd: torch.Tensor, token: torch.Tensor, u: torch.Tensor,
                       resample_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One position. token ~ p^d already drawn; u,resample_u ~ U(0,1).

    Returns (accepted: bool, out_token). If accepted, out_token=token; else resampled
    from the residual using resample_u (inverse-CDF).
    """
    ptok = pt.gather(-1, token.unsqueeze(-1)).squeeze(-1)
    pdtok = pd.gather(-1, token.unsqueeze(-1)).squeeze(-1)
    ratio = (ptok / pdtok.clamp_min(1e-20)).clamp(max=1.0)
    accepted = u < ratio
    resamp = _inverse_cdf(residual_dist(pt, pd), resample_u)
    out = torch.where(accepted, token, resamp)
    return accepted, out


def _inverse_cdf(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Sample token indices by inverse-CDF with uniforms u (shape = probs[:-1])."""
    cdf = probs.cumsum(-1)
    return torch.searchsorted(cdf, u.unsqueeze(-1)).squeeze(-1).clamp(max=probs.shape[-1] - 1)


def sample_from(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Draw a token per row from `probs` using uniforms `u` (inverse-CDF)."""
    return _inverse_cdf(probs, u)
