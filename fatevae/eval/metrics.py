"""Evaluation metrics: the Aim-1 evaluation contract.

All metrics are computed identically for every method (PCA baselines,
ablations, FateVAE) so the benchmark table is directly comparable:

- ``embedding_gene_auroc``  - does an embedding carry the information of a
  held-out marker gene? (kNN prediction of the gene's detection pattern)
- ``embedding_silhouette``  - fate-label separation in the embedding
- ``chi_index``             - Cryptic Heterogeneity Index: kNN cross-fate
  contamination, fate-aware vs variance-aware
- ``fate_entropy``          - entropy of per-cell fate probabilities
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import silhouette_score


def embedding_gene_auroc(
    embedding: np.ndarray,
    gene_expr: np.ndarray,
    n_splits: int = 5,
    k: int = 25,
    seed: int = 0,
) -> float:
    """AUROC of predicting a gene's detection pattern from the embedding.

    gene_expr: per-cell log-normalized expression. The label is
    "gene detected above its 75th percentile among expressing cells" so the
    task is the same regardless of gene abundance. kNN classifier, CV'd.
    """
    y = (gene_expr > 0).astype(int)
    thr = np.quantile(gene_expr[gene_expr > 0], 0.75) if (y == 1).any() else 1.0
    y = (gene_expr > thr).astype(int)
    if y.sum() < 10 or (1 - y).sum() < 10:
        return np.nan
    aucs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(embedding, y):
        clf = KNeighborsClassifier(n_neighbors=min(k, len(tr)), n_jobs=-1)
        clf.fit(embedding[tr], y[tr])
        pos_rate = clf.predict_proba(embedding[te])[:, list(clf.classes_).index(1)]
        aucs.append(roc_auc_score(y[te], pos_rate))
    return float(np.mean(aucs))


def embedding_silhouette(
    embedding: np.ndarray,
    labels: np.ndarray,
    sample: int = 4000,
    seed: int = 0,
) -> float:
    labels = pd.Series(labels).astype(str).to_numpy()
    keep = pd.Series(labels).groupby(labels).transform("size").to_numpy() >= 5
    if keep.sum() < 100 or np.unique(labels[keep]).size < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = np.where(keep)[0]
    if len(idx) > sample:
        idx = rng.choice(idx, sample, replace=False)
    return float(silhouette_score(embedding[idx], labels[idx]))


def chi_index(
    embedding: np.ndarray,
    fate_labels: np.ndarray,
    variance_aware_embedding: np.ndarray,
    k: int = 15,
    seed: int = 0,
) -> dict:
    """Cryptic Heterogeneity Index.

    For an embedding E and ground-truth fate labels y:
        contamination(E) = fraction of kNN neighbours of a cell that carry a
        different fate label (higher = more mixed).
        CHI = contamination(variance-aware E0) - contamination(fate-aware E)
    A positive CHI means the fate-aware embedding separates fates better
    than the standard variance-dominated one. Reported per-fate too.
    """
    res = {}
    for tag, emb in (("variance_aware", variance_aware_embedding),
                     ("fate_aware", embedding)):
        cont = _knn_contamination(emb, fate_labels, k=k, seed=seed)
        res[f"contamination_{tag}"] = float(cont.mean())
        for f in np.unique(fate_labels):
            res[f"contamination_{tag}_{f}"] = float(cont[fate_labels == f].mean())
    res["CHI"] = res["contamination_variance_aware"] - res["contamination_fate_aware"]
    return res


def _knn_contamination(emb: np.ndarray, labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    labels = pd.Series(labels).astype(str).to_numpy()
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(emb)
    _, idx = nn.kneighbors(emb)
    nb = labels[idx[:, 1:]]  # drop self
    same = (nb == labels[:, None]).mean(axis=1)
    return 1.0 - same


def fate_entropy(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(p)).sum(axis=1)


def summarize_embedding(
    name: str,
    embedding: np.ndarray,
    fate_labels: np.ndarray,
    heldout_genes: dict[str, np.ndarray],
    variance_aware: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """One row of the benchmark table for a candidate embedding."""
    row: dict = {"method": name, "dim": embedding.shape[1]}
    row["silhouette"] = embedding_silhouette(embedding, fate_labels, seed=seed)
    aucs = {
        g: embedding_gene_auroc(embedding, expr, seed=seed)
        for g, expr in heldout_genes.items()
    }
    valid = [v for v in aucs.values() if not np.isnan(v)]
    row["heldout_marker_auroc"] = float(np.mean(valid)) if valid else np.nan
    row["heldout_marker_auroc_sd"] = float(np.std(valid)) if valid else np.nan
    for g, v in aucs.items():
        row[f"auc_{g}"] = v
    if variance_aware is not None:
        chi = chi_index(embedding, fate_labels, variance_aware, seed=seed)
        row.update({k: v for k, v in chi.items()})
    return row
