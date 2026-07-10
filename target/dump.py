"""Phase 1 instrumentation dump (§7.4).

For each calibration sequence we (1) generate a target response (non-thinking), then
(2) teacher-force a single forward to extract, per response position i:

  * hidden_final[i]  : pre-LM-head hidden (H,)      -> p^t via frozen LM head (§7.3), draft target
  * hidden_inject[i] : hiddens at inject layers (L,H) -> KV-injection context
  * router_star[i]   : router logits at l* (E,)      -> domain descriptor d
  * router_agg[i]    : mean-softmax router over full-attn layers (E,) -> robust descriptor
  * next_token[i]    : x_{i+1}, the token position i predicts (int32)
  * seq_id, pos, domain_id

We store the pre-LM-head hidden (not logits/dist): p^t is reconstructed with the shared
frozen LM head at train time -> O(H) storage, exact TV target, no truncation.

Shards are safetensors (stacked tensors) + a JSON sidecar of metadata.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .hooks import extract

DOMAINS = ["math", "code", "chat", "prose"]
DOMAIN_ID = {d: i for i, d in enumerate(DOMAINS)}

@torch.no_grad()
def generate_response(target, prompt_text: str, gen_cfg: dict, mode: str = "chat", prompt_tokens: int = 128) -> tuple[torch.Tensor, int]:
    """Return (full_ids[1,T], response_start).

    mode="chat": apply chat template (non-thinking), generate an assistant response.
    mode="completion": feed a raw text chunk (first `prompt_tokens` tokens) and continue.
    """
    tok = target.tokenizer
    device = next(target.model.parameters()).device
    if mode == "completion":
        ids = tok(prompt_text, return_tensors="pt", truncation=True, max_length=prompt_tokens).input_ids
    else:
        messages = [{"role": "user", "content": prompt_text}]
        try:
            ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                enable_thinking=gen_cfg.get("enable_thinking", False),
            )
        except TypeError:
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = ids.to(device)
    response_start = ids.shape[1]

    out = target.model.generate(
        input_ids=ids,
        max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        do_sample=True,
        temperature=gen_cfg.get("temperature", 0.7),
        top_p=gen_cfg.get("top_p", 0.8),
        top_k=gen_cfg.get("top_k", 20),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return out, response_start


@torch.no_grad()
def extract_records(
    target,
    full_ids: torch.Tensor,
    response_start: int,
    inject_layers: list[int],
    l_star: int,
) -> dict[str, torch.Tensor]:
    """Teacher-force one forward; return per-response-position record tensors (CPU)."""
    res = extract(target, full_ids)
    T = full_ids.shape[1]
    # positions i in [response_start-1 .. T-2]: each predicts token i+1 (the first response
    # token is predicted from the last prompt position, so start at response_start-1).
    lo = max(response_start - 1, 0)
    hi = T - 1  # last position T-1 has no next token
    if hi <= lo:
        return {}
    idx = torch.arange(lo, hi)

    hs = res.hidden_states  # tuple, hs[l+1] = output of layer l
    hidden_final = hs[-1][0, idx].float().to(torch.bfloat16).cpu()          # (N,H)
    inj = torch.stack([hs[l + 1][0, idx] for l in inject_layers], dim=1)     # (N,L,H)
    inj = inj.float().to(torch.bfloat16).cpu()
    router_star = res.router_at(l_star)[idx].float().to(torch.bfloat16).cpu()  # (N,E)
    fa = target.full_attention_layers()
    router_agg = res.router_aggregate(fa)[idx].to(torch.bfloat16).cpu()        # (N,E)
    next_token = full_ids[0, idx + 1].to(torch.int32).cpu()                    # (N,)
    pos = idx.to(torch.int32)

    return {
        "hidden_final": hidden_final,
        "hidden_inject": inj,
        "router_star": router_star,
        "router_agg": router_agg,
        "next_token": next_token,
        "pos": pos,
    }


@dataclass
class ShardWriter:
    """Accumulate per-token records; flush fixed-size safetensors shards."""

    out_dir: str
    tokens_per_shard: int = 65536
    _buf: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    _n: int = 0
    _shard: int = 0
    _seq: int = 0
    _manifest: list[dict] = field(default_factory=list)

    def __post_init__(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def add(self, rec: dict[str, torch.Tensor], domain: str):
        if not rec:
            return
        n = rec["next_token"].shape[0]
        rec = dict(rec)
        rec["seq_id"] = torch.full((n,), self._seq, dtype=torch.int32)
        rec["domain_id"] = torch.full((n,), DOMAIN_ID[domain], dtype=torch.int32)
        for k, v in rec.items():
            self._buf.setdefault(k, []).append(v)
        self._n += n
        self._seq += 1
        while self._n >= self.tokens_per_shard:
            self._flush(self.tokens_per_shard)

    def _flush(self, count: int):
        cat = {k: torch.cat(v, dim=0) for k, v in self._buf.items()}
        take = {k: v[:count].contiguous() for k, v in cat.items()}
        rest = {k: v[count:] for k, v in cat.items()}
        path = os.path.join(self.out_dir, f"shard_{self._shard:05d}.safetensors")
        save_file(take, path)
        self._manifest.append({"shard": self._shard, "n": count, "file": os.path.basename(path)})
        self._buf = {k: [v] for k, v in rest.items() if v.shape[0] > 0}
        self._n = int(next(iter(rest.values())).shape[0]) if rest and next(iter(rest.values())).shape[0] else 0
        self._shard += 1

    def close(self):
        if self._n > 0:
            self._flush(self._n)
        meta = {
            "shards": self._manifest,
            "sequences": self._seq,
            "total_tokens": sum(m["n"] for m in self._manifest),
            "keys": list(self._buf.keys()) if self._buf else None,
        }
        json.dump(meta, open(os.path.join(self.out_dir, "manifest.json"), "w"), indent=2)
        return meta


def load_shards(out_dir: str):
    """Yield (shard_dict, meta) for each shard in a dump directory."""
    meta = json.load(open(os.path.join(out_dir, "manifest.json")))
    for m in meta["shards"]:
        yield load_file(os.path.join(out_dir, m["file"])), m


class DumpDataset(torch.utils.data.Dataset):
    """Flat per-token view over all shards (lazy-loads one shard at a time)."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.meta = json.load(open(os.path.join(out_dir, "manifest.json")))
        self.offsets, tot = [], 0
        for m in self.meta["shards"]:
            self.offsets.append((tot, tot + m["n"], m["file"]))
            tot += m["n"]
        self.total = tot
        self._cache_file = None
        self._cache = None

    def __len__(self):
        return self.total

    def _shard_for(self, i: int):
        for lo, hi, f in self.offsets:
            if lo <= i < hi:
                return lo, f
        raise IndexError(i)

    def __getitem__(self, i: int):
        lo, f = self._shard_for(i)
        if f != self._cache_file:
            self._cache = load_file(os.path.join(self.out_dir, f))
            self._cache_file = f
        j = i - lo
        return {k: v[j] for k, v in self._cache.items()}
