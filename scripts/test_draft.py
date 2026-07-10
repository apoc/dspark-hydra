"""Phase 3 exit: forward + backward for every §6 draft variant on toy data.

Uses a small toy vocab + frozen stand-in embed/LM head so it runs on CPU in seconds.
Validates output shapes, gradient flow to trainable params, and reports parameter
counts (total & active) per variant. The architecture is vocab-agnostic; training
(Phase 4) uses the real borrowed embedding + LM head (V=248320).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from draft.config import DraftConfig  # noqa: E402
from draft.model import DraftModel  # noqa: E402

VARIANTS = {
    "B3_dense": "dense",
    "E1_hard": "hard",
    "E2_soft": "soft",
    "C1_scratch": "scratch",
}


def active_params(model: DraftModel, cfg: DraftConfig) -> int:
    """Params touched per token (attn + heads + k' experts + shared), excluding inactive experts."""
    total = 0
    for name, p in model.named_parameters():
        if ".experts." in name:
            continue  # count separately
        total += p.numel()
    if cfg.is_moe:
        # k' active experts + shared, out of K
        per_expert = sum(p.numel() for n, p in model.named_parameters() if ".experts.0." in n)
        total += cfg.k_prime * per_expert
    return total


def run_variant(mode: str, V: int = 512, B: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    cfg = DraftConfig(hidden_size=256, n_layers=3, n_heads=8, gamma=5, K=16, k_prime=2,
                      moe_intermediate_size=128, vocab_size=V, markov_rank=64,
                      num_target_experts=256, router_mode=mode)
    model = DraftModel(cfg)

    # frozen borrowed stand-ins
    embed = nn.Embedding(V, cfg.hidden_size)
    lm_head = nn.Linear(cfg.hidden_size, V, bias=False)
    for m in (embed, lm_head):
        m.requires_grad_(False)

    E = cfg.num_target_experts
    anchor = torch.randint(0, V, (B,))
    tgt = torch.randint(0, V, (B, cfg.gamma))
    prev = torch.cat([anchor[:, None], tgt[:, :-1]], dim=1)
    hidden_inject = torch.randn(B, cfg.n_inject, cfg.hidden_size)
    d = torch.randn(B, E)
    C = None
    if mode in ("hard", "soft"):
        labels = torch.randint(0, cfg.K, (E,))
        C = torch.zeros(cfg.K, E)
        C[labels, torch.arange(E)] = 1.0

    out = model(anchor, prev, hidden_inject, d, embed, lm_head, C)
    p_logits, conf = out["p_logits"], out["conf"]
    assert p_logits.shape == (B, cfg.gamma, V), p_logits.shape
    assert conf.shape == (B, cfg.gamma), conf.shape

    # toy composite loss: CE + TV(to random p^t) + conf BCE
    pt = torch.softmax(torch.randn(B, cfg.gamma, V), dim=-1)
    logp = F.log_softmax(p_logits, dim=-1)
    pd = logp.exp()
    w = torch.exp(-torch.arange(cfg.gamma).float() / cfg.gamma)[None, :]
    L_ce = -(w * logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).mean()
    L_tv = (w * 0.5 * (pd - pt).abs().sum(-1)).mean()
    cstar = (1 - 0.5 * (pd.detach() - pt).abs().sum(-1)).clamp(0, 1)
    L_conf = F.binary_cross_entropy(conf, cstar)
    loss = 0.1 * L_ce + 0.9 * L_tv + 1.0 * L_conf
    if mode == "soft" and out["aux"]["group_logits"]:
        tgt_g = torch.softmax(out["aux"]["group_logits"][0], dim=-1)      # (N,K) distill target
        pred_g = torch.log_softmax(out["aux"]["router_logits"][0], dim=-1)  # (N,K) draft router
        loss = loss + 0.01 * F.kl_div(pred_g, tgt_g, reduction="batchmean")

    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in trainable)
    total = sum(p.numel() for p in trainable)
    return {
        "mode": mode, "loss": float(loss), "shapes_ok": True,
        "trainable_tensors": len(trainable), "with_grad": n_grad,
        "total_params": total, "active_params": active_params(model, cfg),
    }


def main():
    print(f"{'variant':12s} {'loss':>7s} {'total':>10s} {'active':>10s} {'grad/tensors':>14s}")
    ok = True
    for name, mode in VARIANTS.items():
        r = run_variant(mode)
        gt = f"{r['with_grad']}/{r['trainable_tensors']}"
        print(f"{name:12s} {r['loss']:7.3f} {r['total_params']:10d} {r['active_params']:10d} {gt:>14s}")
        ok = ok and r["shapes_ok"] and r["with_grad"] > 0
    print("\nRESULT:", "ALL VARIANTS FWD/BWD OK" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
