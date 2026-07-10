"""Native MTP-1 head (baseline B0).

The checkpoint ships an MTP-1 head (``mtp.*``, 19 tensors) but HF's loader drops
it (``_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]``). We load those weights
directly from the safetensors shards and wire them into a module that reuses the
target's own ``Qwen3_5MoeDecoderLayer`` + rotary embedding + LM head.

Structure (EAGLE / DeepSeek-MTP style, one layer):
    e = pre_fc_norm_embedding(embed_tokens(next_ids))     # (B,T,H)
    h = pre_fc_norm_hidden(last_hidden_state)             # (B,T,H)
    x = fc(cat([e, h], -1))                               # (B,T,2H)->(B,T,H)
    x = decoder_layer(x, ...)                             # full-attn + MoE
    logits = lm_head(norm(x))
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

MTP_PREFIX = "mtp."


def load_mtp_state_dict(model_path: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Read every ``mtp.*`` tensor from the sharded checkpoint."""
    model_path = os.path.expanduser(model_path)
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    weight_map = json.load(open(index_file))["weight_map"]
    mtp_keys = [k for k in weight_map if k.startswith(MTP_PREFIX)]
    by_shard: dict[str, list[str]] = {}
    for k in mtp_keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    sd: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device=device) as f:
            for k in keys:
                sd[k] = f.get_tensor(k).to(dtype)
    return sd


def validate_mtp(sd: dict[str, torch.Tensor], text_config) -> list[str]:
    """Assert the MTP tensors are present and shape-consistent with the config.

    Returns a list of human-readable check lines; raises AssertionError on mismatch.
    """
    H = text_config.hidden_size
    E = text_config.num_experts
    I = text_config.moe_intermediate_size
    checks: list[str] = []

    def chk(key: str, shape: tuple[int, ...]):
        assert key in sd, f"missing MTP tensor {key}"
        got = tuple(sd[key].shape)
        assert got == shape, f"{key}: expected {shape}, got {got}"
        checks.append(f"  ok  {key:52s} {list(shape)}")

    chk("mtp.fc.weight", (H, 2 * H))
    chk("mtp.pre_fc_norm_embedding.weight", (H,))
    chk("mtp.pre_fc_norm_hidden.weight", (H,))
    chk("mtp.norm.weight", (H,))
    chk("mtp.layers.0.mlp.gate.weight", (E, H))
    chk("mtp.layers.0.mlp.experts.gate_up_proj", (E, 2 * I, H))
    chk("mtp.layers.0.mlp.experts.down_proj", (E, H, I))
    chk("mtp.layers.0.input_layernorm.weight", (H,))
    chk("mtp.layers.0.post_attention_layernorm.weight", (H,))
    return checks


class Qwen35MTPHead(nn.Module):
    """Runnable native MTP-1 head, weights loaded from the checkpoint.

    Reuses the target's decoder-layer class, rotary embedding, embedding table and
    LM head (all frozen). Constructed against the target so no architecture is
    re-derived by hand.
    """

    def __init__(self, target, full_attn_layer_idx: int | None = None):
        super().__init__()
        self.target = target
        cfg = target.text_config
        # locate transformers building blocks from the already-imported module
        lm = target.model.model.language_model
        DecoderLayer = type(lm.layers[0])
        RMSNorm = type(lm.norm)
        self.rotary_emb = lm.rotary_emb
        self.embed_tokens = lm.embed_tokens
        self.lm_head = target.model.lm_head if hasattr(target.model, "lm_head") else target.model.get_output_embeddings()

        if full_attn_layer_idx is None:
            full_attn_layer_idx = target.full_attention_layers()[0]
        device = next(lm.parameters()).device
        dtype = next(lm.parameters()).dtype

        self.pre_fc_norm_embedding = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device, dtype)
        self.pre_fc_norm_hidden = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device, dtype)
        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False).to(device, dtype)
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device, dtype)
        self.layer = DecoderLayer(cfg, full_attn_layer_idx).to(device, dtype)
        self._device, self._dtype = device, dtype

    @torch.no_grad()
    def load_weights(self, sd: dict[str, torch.Tensor]):
        def cp(mod_param: torch.Tensor, key: str):
            mod_param.copy_(sd[key].to(mod_param.device, mod_param.dtype))

        cp(self.pre_fc_norm_embedding.weight, "mtp.pre_fc_norm_embedding.weight")
        cp(self.pre_fc_norm_hidden.weight, "mtp.pre_fc_norm_hidden.weight")
        cp(self.fc.weight, "mtp.fc.weight")
        cp(self.norm.weight, "mtp.norm.weight")
        layer_sd = {k[len("mtp.layers.0."):]: v for k, v in sd.items() if k.startswith("mtp.layers.0.")}
        missing, unexpected = self.layer.load_state_dict(layer_sd, strict=False)
        # experts params are nn.Parameter tensors; strict=False tolerates naming, assert none critical missing
        assert not [m for m in missing if "experts" in m or "self_attn" in m], f"MTP layer missing: {missing}"
        return missing, unexpected

    @torch.no_grad()
    def forward(self, last_hidden_state: torch.Tensor, next_input_ids: torch.Tensor) -> torch.Tensor:
        """Predict the token after each position.

        last_hidden_state: (B,T,H) final hidden from the target main model.
        next_input_ids:    (B,T) the token sequence shifted by the MTP offset.
        """
        device = self._device
        e = self.pre_fc_norm_embedding(self.embed_tokens(next_input_ids.to(device)))
        h = self.pre_fc_norm_hidden(last_hidden_state.to(device, self._dtype))
        x = self.fc(torch.cat([e, h], dim=-1))

        B, T, _ = x.shape
        position_ids = torch.arange(T, device=device).view(1, 1, -1).expand(3, B, -1)
        pos_emb = self.rotary_emb(x, position_ids)
        text_position_ids = torch.arange(T, device=device).view(1, -1).expand(B, -1)
        x = self.layer(
            x,
            position_embeddings=pos_emb,
            attention_mask=None,  # None => full causal within the block
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
        )
        return self.lm_head(self.norm(x))
