"""Drafting losses (§7.2).

L = 0.1 L_ce + 0.9 L_tv + 1.0 L_conf  (+ lambda_route L_route for soft; + lambda_bal L_bal)

  L_ce   = -sum_k w_k log p_k^d(x_k*)
  L_tv   =  sum_k w_k || p_k^d - p_k^t ||_1
  L_conf =  BCE(c_k, c_k*),  c_k* = 1 - 1/2 || p_k^d - p_k^t ||_1   (detached)
  L_route=  KL( softmax(C@softmax(d)) || softmax(R_d(h)) )          (soft reuse, §4.2)
  L_bal  =  K * sum_e f_e * P_e   (switch load balance on the draft router)

  w_k = exp(-(k-1)/gamma)   (position decay; 0-indexed: exp(-k/gamma))

p^t is reconstructed from the stored pre-LM-head hidden via the frozen LM head (§7.3).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def position_weights(gamma: int, device) -> torch.Tensor:
    return torch.exp(-torch.arange(gamma, device=device).float() / gamma)  # (gamma,)


@torch.no_grad()
def reconstruct_pt(pt_hidden: torch.Tensor, lm_head, chunk: int = 4) -> torch.Tensor:
    """pt_hidden:(B,gamma,H) -> p^t:(B,gamma,V) via frozen LM head (softmax, temp 1)."""
    B, g, H = pt_hidden.shape
    outs = []
    flat = pt_hidden.reshape(B * g, H)
    for i in range(0, flat.shape[0], chunk * g):
        outs.append(torch.softmax(lm_head(flat[i:i + chunk * g]).float(), dim=-1))
    return torch.cat(outs, 0).reshape(B, g, -1)


def drafting_loss(out: dict, targets: torch.Tensor, pt: torch.Tensor, cfg,
                  w_ce=0.1, w_tv=0.9, w_conf=1.0, lambda_route=0.01, lambda_bal=0.001) -> dict:
    p_logits = out["p_logits"]                  # (B,g,V)
    conf = out["conf"]                           # (B,g)
    B, g, V = p_logits.shape
    w = position_weights(g, p_logits.device)     # (g,)

    logpd = F.log_softmax(p_logits.float(), dim=-1)
    pd = logpd.exp()

    ce_tok = -logpd.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)   # (B,g)
    L_ce = (w[None] * ce_tok).sum(1).mean()

    l1 = (pd - pt).abs().sum(-1)                 # (B,g) full L1
    L_tv = (w[None] * l1).sum(1).mean()

    cstar = (1.0 - 0.5 * l1).clamp(0, 1).detach()
    L_conf = F.binary_cross_entropy(conf.float(), cstar)

    loss = w_ce * L_ce + w_tv * L_tv + w_conf * L_conf
    logs = {"L_ce": L_ce.detach().item(), "L_tv": L_tv.detach().item(), "L_conf": L_conf.detach().item()}

    aux = out.get("aux", {})
    if cfg.router_mode == "soft" and aux.get("group_logits") and aux.get("router_logits"):
        kl = 0.0
        for gl, rl in zip(aux["group_logits"], aux["router_logits"]):
            tgt = torch.softmax(gl.float(), dim=-1)
            pred = F.log_softmax(rl.float(), dim=-1)
            kl = kl + F.kl_div(pred, tgt, reduction="batchmean")
        L_route = kl / len(aux["group_logits"])
        loss = loss + lambda_route * L_route
        logs["L_route"] = L_route.detach().item()

    if cfg.router_mode in ("soft", "scratch") and aux.get("gate"):
        lb = 0.0
        for gate in aux["gate"]:                 # (N,K) softmax gate
            P = gate.mean(0)                     # mean prob per group
            f = torch.zeros_like(P)
            sel = gate.topk(cfg.k_prime, dim=-1).indices
            f.scatter_add_(0, sel.reshape(-1), torch.ones(sel.numel(), device=gate.device))
            f = f / sel.numel()   # fraction of routed slots per group (sums to 1)
            lb = lb + cfg.K * (f * P).sum()
        L_bal = lb / len(aux["gate"])
        loss = loss + lambda_bal * L_bal
        logs["L_bal"] = L_bal.detach().item()

    logs["loss"] = loss.detach().item()
    return {"loss": loss, "logs": logs}
