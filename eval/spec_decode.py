"""Speculative decoding engine + accepted-length (tau) measurement (§8.1).

Offline protocol (scheduler OFF, fixed block gamma, chain drafting, temperature 1.0):
each round the draft proposes gamma tokens from the current anchor (using the target's
full-attn hiddens + router logits at the anchor, borrowed for free), the target verifies
them in one forward, and speculative rejection sampling accepts a prefix. tau = accepted
tokens per round (+1 bonus when all gamma accept). Output is lossless for any draft.
"""

from __future__ import annotations

import time
import torch

from target.hooks import extract
from .sampler import accept_or_resample, sample_from


@torch.no_grad()
def _anchor_signals(target, res, pos, inject_layers, l_star, d_source="star", full_attn_layers=None):
    """Extract (hidden_inject:(1,L,H), d:(1,E)) at position `pos`. d_source 'star' = router
    logits at l*; 'agg' = log of the mean-softmax over full_attn_layers (so softmax(d)=agg,
    matching the C built from router_agg)."""
    hs = res.hidden_states
    inj = torch.stack([hs[l + 1][0, pos] for l in inject_layers], dim=0).unsqueeze(0)  # (1,L,H)
    if d_source == "agg":
        d = res.router_aggregate(full_attn_layers)[pos].clamp_min(1e-9).log().unsqueeze(0)  # (1,E)
    else:
        d = res.router_at(l_star)[pos].unsqueeze(0).float()                                 # (1,E)
    return inj, d


def verify_bonus_dists(logits_row, L, gamma):
    """Split target logits over vseq into (verify, bonus) distributions.

    logits_row: (T,V) target logits over vseq = seq(len L) + proposed(len gamma), T=L+gamma.
    Returns (pt_verify:(gamma,V), pt_bonus:(V,)):
      - pt_verify[k] (k=0..gamma-1) predicts proposed[k]; logits at index L-1+k.
      - pt_bonus is the all-accept bonus, ONE past the last proposal: logits at index L-1+gamma.
        Drawing the bonus from any earlier index is NOT lossless (it samples a distribution
        that predicts an earlier token, not the token that follows the accepted block).
    """
    pt = torch.softmax(logits_row[L - 1: L + gamma].float(), dim=-1)  # (gamma+1, V)
    return pt[:gamma], pt[gamma]


@torch.no_grad()
def spec_decode(target, draft, cfg, prompt_ids, C=None, max_new=128, inject_layers=(19, 31, 39),
                l_star=39, gen=None, greedy_draft=False, time_it=False, d_source="star"):
    """Run speculative decoding from prompt_ids:(1,L0).

    Returns (out_ids, taus, n_accs, timing). Per round tau = tokens generated = n_acc + 1
    (DSpark bonus-token convention, Sec 4.1 fn4): every round emits n_acc accepted draft
    tokens plus exactly one target token -- a residual-corrected token on rejection, or a
    free bonus on all-accept. Lossless for any draft. `timing` holds per-round wall-clock
    sums {t_draft, t_verify, rounds} when time_it=True; t_verify is the single verification
    forward over vseq (the anchor-signal forward is an offline artifact, excluded from L).
    """
    device = next(target.model.parameters()).device
    embed = target.model.get_input_embeddings()
    lm_head = target.model.lm_head if hasattr(target.model, "lm_head") else target.model.get_output_embeddings()
    seq = prompt_ids.to(device)
    taus, n_accs_all = [], []
    t_draft = t_verify = t_anchor = 0.0
    rounds = 0
    C = C.to(device) if C is not None else None
    fa = target.full_attention_layers() if d_source == "agg" else None

    def _sync():
        if time_it and device.type == "cuda":
            torch.cuda.synchronize()

    while seq.shape[1] - prompt_ids.shape[1] < max_new:
        _sync(); t0 = time.perf_counter()
        res = extract(target, seq)                                    # anchor-signal forward
        _sync(); t_anchor += time.perf_counter() - t0
        L = seq.shape[1]
        anchor = seq[0, -1:].clone()                                  # (1,)
        inj, d = _anchor_signals(target, res, L - 1, list(inject_layers), l_star, d_source=d_source, full_attn_layers=fa)
        _sync(); t0 = time.perf_counter()
        proposed, pd = draft.propose(anchor, inj, d, embed, lm_head, C, greedy=greedy_draft, gen=gen)
        _sync(); t_draft += time.perf_counter() - t0
        proposed, pd = proposed[0], pd[0]                             # (g,),(g,V)

        vseq = torch.cat([seq, proposed.unsqueeze(0)], dim=1)         # (1, L+g)
        _sync(); t0 = time.perf_counter()
        vres = extract(target, vseq)
        _sync(); t_verify += time.perf_counter() - t0
        # gamma+1 target dists at indices L-1 .. L-1+gamma: first gamma verify the proposals,
        # the last (index gamma) is the all-accept bonus distribution.
        pt, pt_bonus = verify_bonus_dists(vres.logits[0], L, cfg.gamma)  # (g,V),(V,)

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
            # all accepted -> free bonus token from the target dist AFTER the last proposal
            ub = torch.rand((), generator=gen, device=device)
            bonus = sample_from(pt_bonus.unsqueeze(0), ub.unsqueeze(0))[0]
            new_tokens.append(bonus)
        taus.append(n_acc + 1)
        n_accs_all.append(n_acc)
        rounds += 1

        add = torch.stack(new_tokens).to(device).unsqueeze(0)
        seq = torch.cat([seq, add], dim=1)
        eos = getattr(target.tokenizer, "eos_token_id", None)
        if eos is not None and (add[0] == eos).any():
            break

    timing = {"t_anchor": t_anchor, "t_draft": t_draft, "t_verify": t_verify, "rounds": rounds}
    return seq, taus, n_accs_all, timing
