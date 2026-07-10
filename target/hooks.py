"""Extract the two free signals the experiment reuses from the target's
verification forward pass:

  * full-attention layer hidden states  -> KV-injection context (H_ctx)
  * per-layer 256-way router logits      -> domain descriptor (d)

Both come from a single forward with ``output_hidden_states=True`` and
``output_router_logits=True``. transformers records router logits via the
model's ``_can_record_outputs`` (OutputRecorder on Qwen3_5MoeTopKRouter),
so no manual module hooks are needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ExtractionResult:
    # hidden_states[i] = output of layer i-1; index 0 is the embedding layer.
    hidden_states: tuple[torch.Tensor, ...]
    # router_logits[l] = (num_tokens, num_experts) logits for layer l.
    router_logits: tuple[torch.Tensor, ...]
    logits: torch.Tensor

    def full_attn_hiddens(self, layers: list[int]) -> dict[int, torch.Tensor]:
        """Hidden state *emitted by* layer `l` -> hidden_states[l + 1]."""
        return {l: self.hidden_states[l + 1] for l in layers}

    def router_at(self, layer: int) -> torch.Tensor:
        return self.router_logits[layer]

    def router_aggregate(self, layers: list[int]) -> torch.Tensor:
        """Mean of softmaxed router logits over `layers` (RQ5 aggregate descriptor)."""
        probs = [torch.softmax(self.router_logits[l].float(), dim=-1) for l in layers]
        return torch.stack(probs, 0).mean(0)


@torch.no_grad()
def extract(target, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> ExtractionResult:
    """Run one text-only forward, returning hiddens + router logits + LM logits."""
    model = target.model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_router_logits=True,
        use_cache=False,
        return_dict=True,
    )
    return ExtractionResult(
        hidden_states=out.hidden_states,
        router_logits=out.router_logits,
        logits=out.logits,
    )
