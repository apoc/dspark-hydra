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


def load_target(
    model_yaml: str | os.PathLike | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "cuda",
    drop_vision: bool = True,
) -> Target:
    """Load model + tokenizer. Frozen (eval, requires_grad=False)."""
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    path = resolve_model_path(model_yaml)
    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=dtype, device_map=device_map
    )
    model.eval()
    model.requires_grad_(False)

    if drop_vision and hasattr(model.model, "visual"):
        # free the vision tower; text drafting never touches it
        model.model.visual = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    text_config = config.get_text_config()
    return Target(model=model, tokenizer=tokenizer, text_config=text_config, path=path)
