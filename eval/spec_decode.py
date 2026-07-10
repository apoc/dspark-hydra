"""Speculative decoding engine + accepted-length (tau) measurement (§8.1).

Offline protocol (scheduler OFF, fixed block gamma, chain drafting, temperature 1.0):
each round the draft proposes gamma tokens from the current anchor (using the target's
full-attn hiddens + router logits at the anchor, borrowed for free), the target verifies
them in one forward, and speculative rejection sampling accepts a prefix. tau = accepted
tokens per round (+1 bonus when all gamma accept). Output is lossless for any draft.
"""

from __future__ import annotations

import torch

from target.hooks import extract
from .sampler import accept_or_resample, sample_from


@torch.no_grad()
def _anchor_signals(target, res, pos, inject_layers, l_star):
    """Extract (hidden_inject:(1,L,H), d:(1,E)) at sequence position `pos`."""
    hs = res.hidden_states
    inj = torch.stack([hs[l + 1][0, pos] for l in inject_layers], dim=0).unsqueeze(0)  # (1,L,H)
    d = res.router_at(l_star)[pos].unsqueeze(0).float()                                 # (1,E)
    return inj, d


@torch.no_grad()
def spec_decode(target, draft, cfg, prompt_ids, C=None, max_new=128, inject_layers=(19, 31, 39),
                l_star=39, gen=None, greedy_draft=False):
    """Run speculative decoding from prompt_ids:(1,L0). Returns (out_ids, taus:list)."""
    device = next(target.model.parameters()).device
    embed = target.model.get_input_embeddings()
    lm_head = target.model.lm_head if hasattr(target.model, "lm_head") else target.model.get_output_embeddings()
    seq = prompt_ids.to(device)
    taus, n_accs_all = [], []
    C = C.to(device) if C is not None else None

    while seq.shape[1] - prompt_ids.shape[1] < max_new:
        res = extract(target, seq)
        L = seq.shape[1]
        anchor = seq[0, -1:].clone()                                  # (1,)
        inj, d = _anchor_signals(target, res, L - 1, list(inject_layers), l_star)
        proposed, pd = draft.propose(anchor, inj, d, embed, lm_head, C, greedy=greedy_draft, gen=gen)
        proposed, pd = proposed[0], pd[0]                             # (g,),(g,V)

        vseq = torch.cat([seq, proposed.unsqueeze(0)], dim=1)
        vres = extract(target, vseq)
        pt = torch.softmax(vres.logits[0, L - 1: L - 1 + cfg.gamma].float(), dim=-1)  # (g,V)

        n_acc = 0
        new_tokens = []
        for k in range(cfg.gamma):
            u = torch.rand((), generator=gen, device=device)
            ru = torch.rand((), generator=gen, device=device)
            acc, tok = accept_or_resample(pt[k].unsqueeze(0), pd[k].unsqueeze(0),
                                          proposed[k].unsqueeze(0), u.unsqueeze(0), ru.unsqueeze(0))
            new_tokens.append(tok[0])
            if bool(acc[0]):
                n_acc += 1
            else:
                break
        if n_acc == cfg.gamma:
            # all accepted -> free bonus token from the target dist at the last position
            ub = torch.rand((), generator=gen, device=device)
            bonus = sample_from(pt[cfg.gamma - 1].unsqueeze(0), ub.unsqueeze(0))[0]
            new_tokens.append(bonus)
        taus.append(n_acc + (1 if n_acc == cfg.gamma else 0))
        n_accs_all.append(n_acc)

        add = torch.stack(new_tokens).to(device).unsqueeze(0)
        seq = torch.cat([seq, add], dim=1)
        eos = getattr(target.tokenizer, "eos_token_id", None)
        if eos is not None and (add[0] == eos).any():
            break

    return seq, taus, n_accs_all
