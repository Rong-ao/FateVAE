"""FateOT teacher: data-only pseudo-fates for larval cells via optimal
transport to *unlabeled* adult cell clusters.

No experimental annotation and no marker gene is used:
  - adult cells are clustered data-driven (Leiden on their PCA);
  - cluster gene sets are data-derived (differential expression);
  - larval pseudo-fate scores come from an entropy-regularized OT plan
    (Sinkhorn) between larval cells and adult clusters in the joint PCA
    space of the merged larva+adult object.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from fatevae import config


def cluster_adults(
    merged: ad.AnnData,
    resolution: float = 0.5,
    min_size: int = 80,
) -> pd.Series:
    """Leiden clusters of adult cells only (data-derived terminal states)."""
    adult = merged[merged.obs["sample"] != "larva"].copy()
    sc.pp.pca(adult, n_comps=30, svd_solver="arpack")
    sc.pp.neighbors(adult, n_neighbors=15, random_state=0)
    sc.tl.leiden(adult, resolution=resolution, key_added="adult_leiden")
    counts = adult.obs["adult_leiden"].value_counts()
    keep = counts[counts >= min_size].index
    lab = adult.obs["adult_leiden"].astype(str).where(
        adult.obs["adult_leiden"].isin(keep), other="small")
    return pd.Series(lab.astype(str).to_numpy(), index=adult.obs_names,
                     name="adult_cluster")


def cluster_genesets(
    merged: ad.AnnData,
    adult_labels: pd.Series,
    n_genes: int = 30,
) -> dict[str, list[str]]:
    """Top differentially expressed genes per adult cluster (data-derived)."""
    adult = merged[adult_labels.index].copy()
    adult.obs["adult_cluster"] = adult_labels.to_numpy()
    sc.tl.rank_genes_groups(adult, "adult_cluster", method="wilcoxon")
    sets: dict[str, list[str]] = {}
    for cl in adult.obs["adult_cluster"].unique():
        genes = sc.get.rank_genes_groups_df(adult, group=cl)["names"].head(n_genes)
        sets[str(cl)] = list(genes)
    return sets


def sinkhorn(
    cost: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float = 0.05,
    n_iter: int = 200,
) -> np.ndarray:
    """Entropy-regularized OT plan (cells x clusters)."""
    K = np.exp(-cost / reg)
    K = K / K.sum(axis=1, keepdims=True)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        u = a / np.maximum(K @ v, 1e-12)
        v = b / np.maximum(K.T @ u, 1e-12)
    return (K * u[:, None]) * v[None, :]


def larva_pseudo_fates(
    merged_h5ad=config.DMEL_MERGED_H5AD,
    resolution: float = 0.5,
    reg: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.Series]:
    """Returns (larval cell x cluster OT soft-assignment, cluster gene sets,
    adult cluster labels)."""
    merged = ad.read_h5ad(merged_h5ad)
    adult_labels = cluster_adults(merged, resolution=resolution)
    genesets = cluster_genesets(merged, adult_labels)

    is_larva = (merged.obs["sample"] == "larva").to_numpy()
    emb = np.asarray(merged.obsm["X_pca"], dtype=np.float64)
    emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-9)

    clusters = sorted(c for c in adult_labels.unique() if c != "small")
    adult_pos = merged.obs_names.get_indexer(adult_labels.index)
    emb_adult = emb[adult_pos]
    cents = np.stack([emb_adult[adult_labels.to_numpy() == c].mean(0)
                      for c in clusters])
    cost = np.linalg.norm(emb[is_larva][:, None, :] - cents[None, :, :],
                          axis=2)
    cost = cost / (cost.mean() + 1e-9)
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    sizes = np.array([(adult_labels == c).sum() for c in clusters], dtype=float)
    b = sizes / sizes.sum()
    plan = sinkhorn(cost, a, b, reg=reg)
    soft = pd.DataFrame(plan / plan.sum(axis=1, keepdims=True),
                        index=merged.obs_names[is_larva],
                        columns=clusters)
    return soft, genesets, adult_labels
