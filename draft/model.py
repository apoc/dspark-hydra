"""Assembled draft model (§4).

Backbone input sequence per anchor: [emb(anchor), mask_1, ..., mask_{gamma-1}] — the
backbone is parallel (masks stand in for unknown future tokens); the semi-AR token
dependency enters only through the Markov head's B[x_{k-1}] bias. Borrowed embedding +
LM head are frozen and passed in.

forward returns, for each of the gamma positions:
  p_logits : (B, gamma, V)  draft logits (U_k + B_k), softmax = p_k^d
  conf     : (B, gamma)     confidence prob c_k
  aux      : dict for training (group logits for distillation, gate for balance)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import DraftBackbone
from .conf_head import ConfidenceHead
from .config import DraftConfig
from .kv_inject import KVInjector
from .markov_head import MarkovHead


class DraftModel(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.cfg = cfg
        self.injector = KVInjector(cfg)
        self.mask_emb = nn.Parameter(torch.zeros(cfg.gamma - 1, cfg.hidden_size))
        nn.init.normal_(self.mask_emb, std=0.02)
        self.backbone = DraftBackbone(cfg)
        self.markov = MarkovHead(cfg)
        self.conf = ConfidenceHead(cfg)

    def forward(self, anchor_ids, prev_tokens, hidden_inject, d, embed, lm_head, C=None):
        """
        anchor_ids:    (B,)            token at the anchor position
        prev_tokens:   (B, gamma)      x_{k-1} for k=0..gamma-1 (= [anchor, tgt_0..tgt_{g-2}])
        hidden_inject: (B, n_inject, H) or (B, T_ctx, n_inject, H)  target context
        d:             (B, E)          target router logits at l* (descriptor)
        embed, lm_head: frozen borrowed modules
        C:             (K, E)          collapse map (hard/soft); None for dense/scratch
        """
        B = anchor_ids.shape[0]
        cfg = self.cfg
        anchor_emb = embed(anchor_ids).unsqueeze(1)                  # (B,1,H)
        masks = self.mask_emb.unsqueeze(0).expand(B, -1, -1)          # (B,g-1,H)
        x = torch.cat([anchor_emb, masks], dim=1)                    # (B,g,H)

        ctx = self.injector(hidden_inject)                           # (B,Tc,H)
        # broadcast descriptor d to per-position if a single vector is given
        d_seq = d.unsqueeze(1).expand(B, cfg.gamma, -1) if (d is not None and d.dim() == 2) else d
        h = self.backbone(x, ctx, d_seq, C)                          # (B,g,H)

        p_logits, token_feat = self.markov(h, prev_tokens, lm_head)  # (B,g,V),(B,g,r)
        conf_logit, conf = self.conf(h, token_feat)                  # (B,g)

        aux = {"group_logits": [], "router_logits": [], "gate": []}
        for layer in self.backbone.layers:
            f = layer.ffn
            if getattr(f, "last_group_logits", None) is not None:
                aux["group_logits"].append(f.last_group_logits)
            if getattr(f, "last_router_logits", None) is not None:
                aux["router_logits"].append(f.last_router_logits)
            if getattr(f, "last_gate", None) is not None:
                aux["gate"].append(f.last_gate)
        return {"p_logits": p_logits, "conf_logit": conf_logit, "conf": conf, "hidden": h, "aux": aux}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
