"""Draft training loop (§7.2).

Borrows the target's frozen token embedding + LM head, trains the draft (backbone,
MoE experts, Markov head, confidence head; router for soft/scratch) on windowed dump
examples. The collapse map C (hard/soft) is frozen and passed in.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from .data import WindowDataset, collate
from .losses import drafting_loss, reconstruct_pt


def train_draft(model, cfg, dump_dir, embed, lm_head, C=None, *, device="cuda",
                epochs=10, batch_size=16, lr=3e-4, weight_decay=0.0, max_steps=None,
                log_every=20, seed=0):
    torch.manual_seed(seed)
    ds = WindowDataset(dump_dir, cfg.gamma)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    model.to(device).train()
    for p in list(embed.parameters()) + list(lm_head.parameters()):
        p.requires_grad_(False)
    if C is not None:
        C = C.to(device)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    total = (max_steps or (len(dl) * epochs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)

    print(f"train: {len(ds)} windows, {len(dl)} steps/epoch, {total} total steps, "
          f"variant={cfg.router_mode} K={cfg.K} k'={cfg.k_prime}")
    step, t0 = 0, time.time()
    for ep in range(epochs):
        for batch in dl:
            b = {k: v.to(device) for k, v in batch.items()}
            pt = reconstruct_pt(b["pt_hidden"], lm_head)
            out = model(b["anchor_id"], b["prev_tokens"], b["hidden_inject"], b["d"], embed, lm_head, C)
            res = drafting_loss(out, b["targets"], pt, cfg)
            opt.zero_grad(set_to_none=True)
            res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % log_every == 0 or step == 1:
                lg = res["logs"]
                extra = " ".join(f"{k}={lg[k]:.3f}" for k in ("L_route", "L_bal") if k in lg)
                print(f"  step {step:5d}/{total} loss={lg['loss']:.4f} ce={lg['L_ce']:.3f} "
                      f"tv={lg['L_tv']:.3f} conf={lg['L_conf']:.3f} {extra} ({time.time()-t0:.0f}s)", flush=True)
            if max_steps and step >= max_steps:
                return model
    return model
