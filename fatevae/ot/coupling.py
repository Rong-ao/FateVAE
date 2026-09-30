"""OT coupling: larval disc cells (source) -> adult terminal fates (target).

Per-cell fate mass = row of the (row-normalized) transport plan aggregated
by target group. This is the moscot/Waddington-OT-style prospective
validation layer of proposal v2 (Aim 5) - the marker-independent teacher.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sinkhorn import row_normalized, sinkhorn_unbalanced


def build_ot_problem(
    merged_pca: np.ndarray,
    is_larva: np.ndarray,
    adult_group: pd.Series,  # group label per adult cell (NaN for others)
    balance: dict[str, int] | None = None,
    seed: int = 0,
):
    """Balanced target construction + scaled cost matrix.

    Returns (emb_source, emb_target, target_group_labels, C) where C is
    squared-Euclidean cost on standardized PCs, rescaled by its median.
    """
    rng = np.random.default_rng(seed)
    emb = np.asarray(merged_pca, dtype=np.float64)
    emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-9)

    idx_t = []
    labs = []
    for g, m in adult_group.groupby(adult_group):
        ids = np.where(m.to_numpy())[0]
        cap = (balance or {}).get(g, len(ids))
        if cap < len(ids):
            ids = rng.choice(ids, cap, replace=False)
        idx_t.append(ids)
        labs.extend([g] * len(ids))
    idx_t = np.concatenate(idx_t)
    idx_s = np.where(is_larva)[0]

    Es, Et = emb[idx_s], emb[idx_t]
    # memory-safe squared distance
    sq_s = (Es ** 2).sum(1)[:, None]
    sq_t = (Et ** 2).sum(1)[None, :]
    d2 = sq_s + sq_t - 2.0 * (Es @ Et.T)
    np.maximum(d2, 0.0, out=d2)
    C = d2 / np.median(d2)
    return Es, Et, np.asarray(labs), C


def ot_fate_mass(
    C: np.ndarray,
    target_groups: np.ndarray,
    eps: float = 0.05,
    tau: float = 5.0,
    n_iter: int = 1500,
    growth_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run unbalanced OT and aggregate the plan per target group.

    Returns per-source-cell fate-mass DataFrame (columns = groups)."""
    n, m = C.shape
    b = (growth_weight / growth_weight.sum()) if growth_weight is not None else None
    sol = sinkhorn_unbalanced(C, a=None, b=b, eps=eps, tau=tau, n_iter=n_iter)
    Rn = row_normalized(sol["T"])
    out = pd.DataFrame(0.0, index=range(n), columns=sorted(set(target_groups)))
    for g in out.columns:
        out[g] = Rn[:, target_groups == g].sum(1)
    out.attrs["mass"] = sol["mass"]
    return out


def kNN_anchor_baseline(
    Es: np.ndarray, Et: np.ndarray, target_groups: np.ndarray, k: int = 30
) -> pd.DataFrame:
    """The naive baseline from v0.1 (for head-to-head comparison)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k).fit(Et)
    _, idx = nn.kneighbors(Es)
    nb = target_groups[idx]
    out = pd.DataFrame(0.0, index=range(Es.shape[0]),
                       columns=sorted(set(target_groups)))
    for g in out.columns:
        out[g] = (nb == g).mean(1)
    return out
