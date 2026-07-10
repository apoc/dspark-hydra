"""Load the frozen Qwen3.6-35B-A3B target, text-only.

The checkpoint's architecture is ``Qwen3_5MoeForConditionalGeneration`` (multimodal);
text weights live under ``model.language_model.*`` and vision under ``model.visual.*``.
We load the full multimodal class and drive only the text path (``pixel_values=None``);
the vision tower stays idle. This keeps the router / hidden extraction on the exact
production graph rather than a hand-remapped text-only copy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_YAML = _REPO_ROOT / "configs" / "model.yaml"


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def resolve_model_path(model_yaml: str | os.PathLike | None = None) -> str:
    """Return the on-disk checkpoint path from configs/model.yaml (``local_path``)."""
    cfg_path = Path(model_yaml) if model_yaml else _DEFAULT_MODEL_YAML
    cfg = yaml.safe_load(cfg_path.read_text())
    local = cfg["target"].get("local_path")
    if local:
        local = _expand(local)
        if Path(local).exists():
            return local
    # fall back to hub id (will hit the HF cache / download)
    return cfg["target"]["repo_id"]


@dataclass
class Target:
    model: Any
    tokenizer: Any
    text_config: Any
    path: str

    @property
    def num_layers(self) -> int:
        return self.text_config.num_hidden_layers

    def full_attention_layers(self) -> list[int]:
        """0-based indices of full-attention layers (the KV-injection source set)."""
        lt = getattr(self.text_config, "layer_types", None)
        if lt is not None:
            return [i for i, t in enumerate(lt) if t == "full_attention"]
        interval = self.text_config.full_attention_interval
        return [i for i in range(self.num_layers) if (i + 1) % interval == 0]


def load_embed_lm_head(model_yaml=None, device: str = "cuda", dtype=torch.bfloat16):
    """Load ONLY the frozen token embedding + LM head (~2GB), not the full 67GB target.

    Training the draft needs only these borrowed tensors. Returns (embed, lm_head,
    text_config) with the modules on `device` and frozen.
    """
    import json
    import os as _os

    from safetensors import safe_open
    from transformers import AutoConfig

    path = _os.path.expanduser(resolve_model_path(model_yaml))
    cfg = AutoConfig.from_pretrained(path).get_text_config()
    wm = json.load(open(_os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    emb_key = "model.language_model.embed_tokens.weight"
    lm_key = "lm_head.weight"

    def _get(key):
        with safe_open(_os.path.join(path, wm[key]), framework="pt", device="cpu") as f:
            t = f.get_tensor(key).to(dtype)
        return t.pin_memory().to(device, non_blocking=True) if device == "cuda" else t.to(device)

    ew = _get(emb_key)          # (V,H)
    lw = _get(lm_key)           # (V,H)
    # meta-init the modules (no host alloc), then bind the already-on-device weights
    embed = torch.nn.Embedding(ew.shape[0], ew.shape[1], device="meta")
    embed.weight = torch.nn.Parameter(ew, requires_grad=False)
    lm_head = torch.nn.Linear(lw.shape[1], lw.shape[0], bias=False, device="meta")
    lm_head.weight = torch.nn.Parameter(lw, requires_grad=False)
    if device == "cuda":
        torch.cuda.synchronize()
    return embed, lm_head, cfg


def move_to_cuda_pinned(model, device: str = "cuda", chunk_bytes: int = 512 * 1024 * 1024) -> None:
    """Stream a CPU-resident model to CUDA via pinned memory, in place.

    On GB10 unified memory, pageable (unpinned) H2D runs ~0.16 GB/s while pinned
    runs ~18 GB/s. We pin each tensor into a reusable staging buffer, copy to GPU,
    and free the CPU copy so peak memory stays ~model-size (both halves live in the
    same 128GB pool).
    """
    staging = torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)

    def move(t: torch.Tensor) -> torch.Tensor:
        if t.device.type == "cuda":
            return t
        n = t.numel() * t.element_size()
        src = t.contiguous()
        if n <= chunk_bytes:
            buf = staging[:n].view(t.dtype).view(t.shape)
            buf.copy_(src)
            out = buf.to(device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            # tensor larger than staging buffer: pin it directly (rare: big experts)
            out = src.pin_memory().to(device, non_blocking=True)
            torch.cuda.synchronize()
        return out

    with torch.no_grad():
        for _, mod in model.named_modules():
            for name, p in list(mod._parameters.items()):
                if p is None:
                    continue
                mod._parameters[name] = torch.nn.Parameter(move(p.data), requires_grad=False)
            for name, b in list(mod._buffers.items()):
                if b is None:
                    continue
                mod._buffers[name] = move(b)
    torch.cuda.synchronize()


def load_target(
    model_yaml: str | os.PathLike | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "cuda",
    drop_vision: bool = True,
) -> Target:
    """Load model + tokenizer. Frozen (eval, requires_grad=False).

    For a CUDA target, load on CPU (near-free from warm page cache) then stream to
    GPU through pinned memory (~113x faster than transformers' default unpinned H2D
    on GB10).
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    path = resolve_model_path(model_yaml)
    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    want_cuda = device_map == "cuda" and torch.cuda.is_available()
    load_map = "cpu" if want_cuda else device_map
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, device_map=load_map)
    model.eval()
    model.requires_grad_(False)

    if drop_vision and hasattr(model.model, "visual"):
        model.model.visual = None

    if want_cuda:
        move_to_cuda_pinned(model)
        torch.cuda.empty_cache()

    text_config = config.get_text_config()
    return Target(model=model, tokenizer=tokenizer, text_config=text_config, path=path)
