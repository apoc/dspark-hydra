"""Training-data assembly: turn the per-token dump into (anchor, gamma-window) examples.

Within a sequence (contiguous response positions), a window of gamma+1 consecutive rows
[b .. b+gamma] yields one training example predicting x_{a+1..a+gamma} from anchor a=b+1:

  anchor_id     = next_token[b]                       # x_a
  prev_tokens   = next_token[b .. b+gamma-1]          # [x_a .. x_{a+gamma-1}]
  targets       = next_token[b+1 .. b+gamma]          # [x_{a+1} .. x_{a+gamma}]
  hidden_inject = hidden_inject[b+1]                  # target context at the anchor
  d             = router_star[b+1]                    # descriptor at the anchor
  pt_hidden     = hidden_final[b+1 .. b+gamma]        # p^t sources (softmax(lm_head(.)))
"""

from __future__ import annotations

import torch
from safetensors.torch import load_file

from target.dump import load_shards


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, dump_dir: str, gamma: int, d_source: str = "star"):
        self.gamma = gamma
        self.d_source = d_source
        # load all shards into memory and concatenate (calib dumps fit in RAM)
        parts: dict[str, list[torch.Tensor]] = {}
        for sd, _ in load_shards(dump_dir):
            for k, v in sd.items():
                parts.setdefault(k, []).append(v)
        self.data = {k: torch.cat(v, 0) for k, v in parts.items()}

        # group row indices by sequence, ordered by position
        seq = self.data["seq_id"]
        pos = self.data["pos"]
        order = torch.argsort(seq * (int(pos.max()) + 1) + pos)
        self.windows: list[torch.Tensor] = []
        self.window_seq: list[int] = []            # seq_id per window (for leak-free splits)
        cur, rows = None, []
        for r in order.tolist():
            s = int(seq[r])
            if s != cur:
                self._emit(rows, cur)
                cur, rows = s, []
            rows.append(r)
        self._emit(rows, cur)

    def _emit(self, rows: list[int], seq_id):
        g = self.gamma
        for b in range(0, len(rows) - g):
            self.windows.append(torch.tensor(rows[b:b + g + 1], dtype=torch.long))
            self.window_seq.append(int(seq_id) if seq_id is not None else -1)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i: int):
        w = self.windows[i]                 # (gamma+1,) row indices
        d = self.data
        nt = d["next_token"][w].long()      # (gamma+1,)
        return {
            "anchor_id": nt[0],
            "prev_tokens": nt[:self.gamma],
            "targets": nt[1:self.gamma + 1],
            "hidden_inject": d["hidden_inject"][w[1]],       # (n_inject,H)
            "d": (d["router_agg"][w[1]].float().clamp_min(1e-9).log() if self.d_source == "agg"
                  else d["router_star"][w[1]].float()),   # softmax(d)=agg when logged
            "pt_hidden": d["hidden_final"][w[1:self.gamma + 1]],  # (gamma,H)
            "domain_id": d["domain_id"][w[1]],
        }


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], 0)
    return out
