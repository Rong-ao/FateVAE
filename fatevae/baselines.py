"""Aim-1 baseline embeddings (all returned as cells x d arrays).

B0 hvg_pca       - variance-aware reference (the 'round UMAP' space)
B1 regulon_pca   - PCA on pySCENIC AUC matrix (fate-aware, unsupervised)
B2 marker_pca    - PCA on prior marker-gene expression (oracle-lite)
B3 plain_vae     - unsupervised branch of FateVAE (ablation via cfg)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def hvg_pca(adata, n_pcs: int = 30) -> np.ndarray:
    if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= n_pcs:
        return np.asarray(adata.obsm["X_pca"])[:, :n_pcs]
    from scanpy.tools import pca as sc_pca

    sc_pca(adata, n_comps=n_pcs, svd_solver="arpack")
    return np.asarray(adata.obsm["X_pca"])[:, :n_pcs]


def regulon_pca(auc: pd.DataFrame, n_pcs: int = 10) -> np.ndarray:
    X = auc.to_numpy(dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return PCA(n_components=n_pcs, random_state=0).fit_transform(X)


def marker_pca(adata, marker_genes: list[str], n_pcs: int = 6) -> np.ndarray:
    gidx = {g: i for i, g in enumerate(adata.var_names)}
    cols = [gidx[g] for g in marker_genes if g in gidx]
    X = adata.X[:, cols]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return PCA(n_components=min(n_pcs, X.shape[1]), random_state=0).fit_transform(X)
