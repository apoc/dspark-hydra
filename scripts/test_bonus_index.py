"""Regression test for the all-accept bonus index (§8.3 losslessness, engine side).

The all-accept bonus token MUST be sampled from the target distribution ONE position
past the last proposal (logits index L-1+gamma), not from the distribution that predicts
the last proposal (L-1+gamma-1). An earlier bug used the latter; this test crafts logits
whose per-position argmax is distinct, so it FAILS under the old `pt[gamma-1]` indexing and
PASSES with `verify_bonus_dists` returning `pt[gamma]`.

Run: $PY scripts/test_bonus_index.py   (pure CPU, no model)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from eval.spec_decode import verify_bonus_dists  # noqa: E402


def run():
    V, L, gamma = 16, 4, 5
    T = L + gamma  # vseq length
    logits = torch.full((T, V), -10.0)
    for i in range(T):
        logits[i, i] = 10.0  # position i -> argmax token i (i < V, no wrap)

    pt_verify, pt_bonus = verify_bonus_dists(logits, L, gamma)
    assert tuple(pt_verify.shape) == (gamma, V), pt_verify.shape
    assert tuple(pt_bonus.shape) == (V,), pt_bonus.shape

    # verify dist k predicts proposed[k]; it lives at logits index L-1+k
    for k in range(gamma):
        got = int(pt_verify[k].argmax())
        assert got == L - 1 + k, f"verify[{k}] argmax={got}, want {L - 1 + k}"

    correct_idx = L - 1 + gamma          # what the bonus MUST use
    buggy_idx = L - 1 + gamma - 1        # what the old code used (== last verify dist)
    assert correct_idx != buggy_idx
    got = int(pt_bonus.argmax())
    assert got == correct_idx, f"bonus argmax={got}, want {correct_idx} (old bug -> {buggy_idx})"
    # confirm the old index is a genuinely different distribution (test discriminates)
    assert int(pt_verify[gamma - 1].argmax()) == buggy_idx

    print(f"bonus-index OK: bonus from logits[{correct_idx}] (L-1+gamma), "
          f"distinct from old buggy logits[{buggy_idx}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
