"""Stage 1 (Seed): discover the fate code from detection-pattern structure.

The premise: compartment-defining genes form *mutually exclusive
co-detection groups* - a cell belongs to one primordium, so its genes'
detection indicators anti-correlate across cells, whereas housekeeping
genes co-detect positively everywhere. Global objectives (reconstruction,
variance, NMF) are blind to this; searching for it explicitly is a
data-driven way to write down the fate code without any experimental prior.

Method:
  1. restrict to an annotation-derived gene panel (TF list, optionally +
     HVGs), prevalence-filtered (detected in 5-70% of cells)
  2. drop nuisance genes whose detection tracks technical covariates
     (n_genes, pct_mt) - QC covariates are data-driven, not curated priors
  3. k-means over gene detection profiles for k = 2..k_max; score each
     partition by *exclusivity sharpness* = mean within-module phi
     correlation - mean between-module phi correlation (fate partitions
     have positive within, negative between)
  4. modules = clusters; per-cell module scores = mean standardized
     detection of member genes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _phi_matrix(D: np.ndarray) -> np.ndarray:
    """Pairwise phi (binary Pearson) correlation of detection columns."""
    p = D.mean(axis=0)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    S = (D - p) / np.sqrt(p * (1 - p))
    return (S.T @ S) / D.shape[0]


def detection_seed(
    X,
    var_names,
    panel: list[str],
    nuisance: dict[str, np.ndarray] | None = None,
    prevalence: tuple[float, float] = (0.05, 0.70),
    nuisance_corr: float = 0.45,
    k_range: range = range(2, 9),
    min_module_size: int = 3,
    seed: int = 0,
) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    """Return (modules, cell_scores, diagnostics).

    modules: {'M0': [genes], ...} anticoherent detection groups
    cell_scores: per-cell standardized module scores (soft pseudo-labels)
    diagnostics: sharpness per k
    """
    vidx = pd.Index(var_names)
    panel = [g for g in panel if g in vidx]
    cols = vidx.get_indexer(panel)
    import scipy.sparse as sp

    Dg = X[:, cols]
    D = (Dg.toarray() > 0) if sp.issparse(Dg) else (np.asarray(Dg) > 0)

    prev = D.mean(axis=0)
    keep = (prev >= prevalence[0]) & (prev <= prevalence[1])
    dropped_prev = int((~keep).sum())

    if nuisance:
        n = D.shape[0]
        for cov in nuisance.values():
            z = (cov - cov.mean()) / (cov.std() + 1e-9)
            r = ((D - D.mean(0)) * z[:, None]).sum(0) / n / (D.std(0) + 1e-9)
            keep &= np.abs(r) < nuisance_corr
    D, names = D[:, keep], [g for g, k in zip(panel, keep) if k]
    print(f"[seed] panel {len(panel)} genes -> {len(names)} after prevalence/"
          f"nuisance filters (dropped {dropped_prev} prevalence)", flush=True)
    if len(names) < min_module_size * 2:
        raise ValueError("seed panel too small after filtering")

    phi = _phi_matrix(D)
    diag_rows = []
    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(D.T)
        lab = km.labels_
        within, between, nw, nb = 0.0, 0.0, 0, 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if lab[i] == lab[j]:
                    within += phi[i, j]; nw += 1
                else:
                    between += phi[i, j]; nb += 1
        within /= max(nw, 1)
        between /= max(nb, 1)
        sharp = within - between
        diag_rows.append({"k": k, "within_phi": within,
                          "between_phi": between, "sharpness": sharp})
        if best is None or sharp > best[1]:
            best = (lab, sharp)
    diagnostics = pd.DataFrame(diag_rows)
    lab = best[0]

    modules: dict[str, list[str]] = {}
    order = np.argsort([np.median(prev[keep][lab == c]) for c in range(lab.max() + 1)])
    for mi, c in enumerate(order):
        genes = [names[i] for i in np.where(lab == c)[0]]
        if len(genes) >= min_module_size:
            modules[f"M{mi}"] = genes

    # per-cell module scores: mean standardized detection of member genes
    scores = pd.DataFrame(index=range(D.shape[0]))
    p = D.mean(axis=0)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    S = (D - p) / np.sqrt(p * (1 - p))
    name_pos = {g: i for i, g in enumerate(names)}
    for m, genes in modules.items():
        scores[m] = S[:, [name_pos[g] for g in genes]].mean(axis=1)
    return modules, scores, diagnostics
