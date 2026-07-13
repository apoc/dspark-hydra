"""Draft training loop (§7.2).

Borrows the target's frozen token embedding + LM head, trains the draft (backbone,
MoE experts, Markov head, confidence head; router for soft/scratch) on windowed dump
examples. The collapse map C (hard/soft) is frozen and passed in.
"""

from __future__ import annotations

import time
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import WindowDataset, collate
from .losses import drafting_loss, reconstruct_pt


def _val_loss(model, cfg, val_batches, embed, lm_head, C, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in val_batches:
            pt = reconstruct_pt(b["pt_hidden"], lm_head)
            out = model(b["anchor_id"], b["prev_tokens"], b["hidden_inject"], b["d"], embed, lm_head, C)
            tot += drafting_loss(out, b["targets"], pt, cfg)["logs"]["loss"]
            n += 1
    model.train()
    return tot / max(n, 1)


def train_draft(model, cfg, dump_dir, embed, lm_head, C=None, *, device="cuda",
                batch_size=16, lr=3e-4, weight_decay=0.0, warmup=200,
                eval_interval=200, patience=6, min_delta=1e-3, min_steps=1000,
                val_seq_fraction=0.08, val_batches_cap=20, max_steps=100_000,
                log_every=50, seed=0, d_source="star", ckpt_path=None):
    """Train until knowledge saturation (validation-loss plateau), not a fixed step count.

    Early-stop when the validation loss fails to improve by `min_delta` (relative) for
    `patience` consecutive evals, but never before `min_steps` (avoids stopping on the
    initial descent). Validation is split by SEQUENCE (windows from one sequence share
    rows, so a window-level split would leak). `max_steps` is a safety cap; the best-val
    checkpoint is restored at the end.
    """
    torch.manual_seed(seed)
    ds = WindowDataset(dump_dir, cfg.gamma, d_source=d_source)
    # leak-free split: hold out whole sequences
    wseq = torch.tensor(ds.window_seq)
    uniq = torch.unique(wseq)
    g = torch.Generator().manual_seed(seed)
    perm = uniq[torch.randperm(len(uniq), generator=g)]
    n_val_seq = max(1, int(len(uniq) * val_seq_fraction))
    val_seqs = set(perm[:n_val_seq].tolist())
    val_w = [i for i, s in enumerate(ds.window_seq) if s in val_seqs]
    train_w = [i for i, s in enumerate(ds.window_seq) if s not in val_seqs]
    train_set = torch.utils.data.Subset(ds, train_w)
    val_set = torch.utils.data.Subset(ds, val_w)
    dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    vdl = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True)
    val_batches = [{k: v.to(device) for k, v in b.items()} for b in vdl][:val_batches_cap]

    model.to(device).train()
    for p in list(embed.parameters()) + list(lm_head.parameters()):
        p.requires_grad_(False)
    if C is not None:
        C = C.to(device)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    def lr_at(s):
        return s / max(warmup, 1) if s < warmup else 1.0     # warmup then hold (horizon unknown)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    print(f"train: {len(train_set)} train / {len(val_set)} val windows, "
          f"variant={cfg.router_mode} K={cfg.K} k'={cfg.k_prime}; saturation-stop "
          f"(patience={patience} eval_interval={eval_interval} min_delta={min_delta}) cap={max_steps}", flush=True)

    best, best_state, bad, step, t0 = float("inf"), None, 0, 0, time.time()
    if ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best, bad, best_state = ck["step"], ck["best"], ck["bad"], ck["best_state"]
        print(f"  resumed {ckpt_path} @ step {step} (best={best:.4f} bad={bad})", flush=True)
    stop = False
    while not stop:
        for batch in dl:
            b = {k: v.to(device) for k, v in batch.items()}
            pt = reconstruct_pt(b["pt_hidden"], lm_head)
            out = model(b["anchor_id"], b["prev_tokens"], b["hidden_inject"], b["d"], embed, lm_head, C)
            res = drafting_loss(out, b["targets"], pt, cfg)
            opt.zero_grad(set_to_none=True)
            res["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step(); sched.step(); step += 1

            if step % log_every == 0 or step == 1:
                lg = res["logs"]
                extra = " ".join(f"{k}={lg[k]:.3f}" for k in ("L_route", "L_bal") if k in lg)
                print(f"  step {step:6d} loss={lg['loss']:.4f} ce={lg['L_ce']:.3f} tv={lg['L_tv']:.3f} "
                      f"conf={lg['L_conf']:.3f} {extra} ({time.time()-t0:.0f}s)", flush=True)

            if step % eval_interval == 0:
                vl = _val_loss(model, cfg, val_batches, embed, lm_head, C, device)
                improved = vl < best * (1 - min_delta)
                if improved:
                    best, bad = vl, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                print(f"  [eval] step {step:6d} val_loss={vl:.4f} best={best:.4f} "
                      f"bad={bad}/{patience} {'*' if improved else ''}", flush=True)
                if ckpt_path is not None:
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "step": step, "best": best,
                                "bad": bad, "best_state": best_state}, str(ckpt_path) + ".tmp")
                    os.replace(str(ckpt_path) + ".tmp", ckpt_path)
                if bad >= patience and step >= min_steps:
                    print(f"  SATURATED at step {step} (val plateau, >= min_steps {min_steps}); "
                          f"restoring best (val={best:.4f})", flush=True)
                    stop = True; break
            if step >= max_steps:
                print(f"  hit max_steps cap {max_steps}", flush=True)
                stop = True; break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"training done: {step} steps, best val_loss={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model
