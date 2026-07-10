"""Pure-torch clustering primitives (no sklearn/scipy on Spark).

k-means (Lloyd), balanced k-means (capacity-constrained assignment), and spectral
embedding via the normalized graph Laplacian. Used by the collapse-map builders.
"""

from __future__ import annotations

import torch


def kmeans(x: torch.Tensor, k: int, iters: int = 100, seed: int = 0, tol: float = 1e-5):
    """Lloyd's k-means. x:(N,D) -> (labels:(N,), centroids:(k,D))."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = x.shape[0]
    # k-means++ init
    centroids = x[torch.randint(0, N, (1,), generator=g)].clone()
    for _ in range(k - 1):
        d2 = torch.cdist(x, centroids).min(dim=1).values ** 2
        probs = d2 / (d2.sum() + 1e-12)
        nxt = torch.multinomial(probs, 1, generator=g)
        centroids = torch.cat([centroids, x[nxt]], dim=0)

    labels = torch.zeros(N, dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(x, centroids)
        new = d.argmin(dim=1)
        if torch.equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centroids[c] = x[m].mean(0)
            else:  # reseed empty cluster to the farthest point
                far = torch.cdist(x, centroids).min(dim=1).values.argmax()
                centroids[c] = x[far]
    return labels, centroids


def balanced_kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 0):
    """Capacity-constrained k-means: each cluster gets ~N/k points (§5.4 rebalance).

    Greedy balanced assignment per iteration: sort points by assignment-regret and
    fill clusters up to capacity. Approximate but deterministic.
    """
    N = x.shape[0]
    cap = -(-N // k)  # ceil
    _, centroids = kmeans(x, k, iters=10, seed=seed)
    labels = torch.zeros(N, dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(x, centroids)  # (N,k)
        order = (d.min(1).values - d.kthvalue(2, dim=1).values).argsort()  # most-certain first
        counts = torch.zeros(k, dtype=torch.long)
        new = torch.full((N,), -1, dtype=torch.long)
        for i in order.tolist():
            pref = d[i].argsort()
            for c in pref.tolist():
                if counts[c] < cap:
                    new[i] = c
                    counts[c] += 1
                    break
        if torch.equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centroids[c] = x[m].mean(0)
    return labels, centroids


def spectral_labels(affinity: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Spectral clustering on a symmetric non-negative affinity matrix -> labels:(N,)."""
    A = affinity.clone().double()
    A = 0.5 * (A + A.t())
    A.fill_diagonal_(0)
    deg = A.sum(1)
    dinv = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
    L = torch.eye(A.shape[0], dtype=torch.double) - dinv[:, None] * A * dinv[None, :]
    evals, evecs = torch.linalg.eigh(L)
    emb = evecs[:, :k]  # k smallest eigenvectors
    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-12)  # row-normalize (Ng-Jordan-Weiss)
    labels, _ = kmeans(emb.float(), k, seed=seed)
    return labels


def labels_to_C(labels: torch.Tensor, k: int, num_experts: int) -> torch.Tensor:
    """One-hot collapse map C:(k, num_experts); C[g,i]=1 iff expert i in group g."""
    C = torch.zeros(k, num_experts)
    C[labels, torch.arange(num_experts)] = 1.0
    return C
