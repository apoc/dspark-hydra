"""Phase 2: build the 256→K expert collapse map C.

Methods (§5):
  co_activation   -- co-activation clustering (default)
  weight_cluster  -- weight-similarity clustering + optional centroid init
  learned         -- end-to-end trainable C (upper bound)

Also runs balance stats and emits a cluster report.

Usage:
    python scripts/build_C.py --calib-dir data/calib/ --method co_activation \
        --K 16 --source-layer 39 --out collapse/C.pt
"""
# TODO (Phase 2): implement Methods 1–3 + rebalancing
