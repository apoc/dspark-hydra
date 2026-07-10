"""Phase 4–5: train and evaluate all variants in §6 with identical hparams.

Dispatches B1–E2 (and ablations) sequentially or via a job scheduler.
Writes results to reports/.

Usage:
    python scripts/run_matrix.py --config configs/variants.yaml \
        --variants B3 E1 E2 C1 --eval-only
"""
# TODO (Phase 4): implement training dispatch; (Phase 5): evaluation dispatch
