"""FateSeed: self-supervised discovery of fate seed modules (v0.3).

Philosophy: the principle is hidden in the data itself. Compartment-like
gene programs differ from housekeeping programs in three computable ways:

  1. TF predictability  - a fate TF's expression is predictable from its
     co-expression neighborhood (information flows both ways between a
     determinant TF and its program).
  2. Bimodality         - a compartment module's per-cell score is bimodal
     (cells are in the domain or not), unlike smooth housekeeping modules.
  3. Mutual exclusivity - compartment domains are spatially exclusive, so
     module scores anti-correlate across pairs.

Pipeline: co-expression neighborhoods for all genome-annotated TFs ->
score each by (1)x(2) -> greedy selection of non-redundant modules with
exclusivity preference -> per-cell pseudo-fate scores. These data-derived
seeds then replace experimental markers in the informed-mode FateVAE.
No experimental marker gene is used anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def tf_predictability(
    X: np.ndarray,
    var_names,
    tf_list: list[str],
    n_features: int = 2000,
    n_folds: int = 5,
    lambda_ridge: float = 1.0,
    seed: int = 0,
) -> pd.Series:
    """Cross-validated R^2 of predicting each TF from non-TF variable genes.

    Multi-target ridge: one solve per fold covers all TFs. Input features
    exclude all TFs (no self-leakage).
    """
    vidx = pd.Index(var_names)
    tf_cols = [vidx.get_loc(t) for t in tf_list if t in vidx]
    tfs = [t for t in tf_list if t in vidx]

    Xd = X if X.ndim == 2 else X.toarray()
    var = Xd.var(axis=0)
    non_tf = np.setdiff1d(np.arange(Xd.shape[1]), tf_cols)
    non_tf = non_tf[np.argsort(-var[non_tf])][:n_features]

    F = Xd[:, non_tf].astype(np.float64)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    Y = Xd[:, tf_cols].astype(np.float64)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)

    n = F.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    ss_res = np.zeros(Y.shape[1])
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(axis=0)
    eye = np.eye(F.shape[1])
    for f in folds:
        te = np.zeros(n, bool)
        te[f] = True
        tr = ~te
        W = np.linalg.solve(F[tr].T @ F[tr] + lambda_ridge * eye,
                            F[tr].T @ Y[tr])
        ss_res += ((Y[te] - F[te] @ W) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-9)
    return pd.Series(r2, index=tfs, name="r2")


def bimodality_gap(scores: np.ndarray, seed: int = 0) -> float:
    """DeltaBIC = BIC(1-comp) - BIC(2-comp); >~10 favours bimodality."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    if np.unique(np.round(s, 4)).size < 10:
        return -np.inf
    bic1 = GaussianMixture(1, random_state=seed).fit(s).bic(s)
    bic2 = GaussianMixture(2, random_state=seed,
                           reg_covar=1e-4).fit(s).bic(s)
    return bic1 - bic2


def module_scores_from_genesets(
    X, var_names, gene_sets: dict[str, list[str]]
) -> pd.DataFrame:
    from fatevae.priors.markers import module_scores

    return module_scores(X, var_names, gene_sets)


def select_fate_seeds(
    X,
    var_names,
    regulons: dict[str, list[str]],
    r2: pd.Series,
    n_seeds: int = 5,
    min_bic: float = 10.0,
    max_abs_corr: float = 0.45,
    min_module_genes: int = 5,
    seed: int = 0,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Greedy selection of compartment-like seed modules.

    Candidates ranked by predictability percentile * bimodality; selected
    greedily subject to non-redundancy (|corr| < max_abs_corr with every
    picked module; prefer negative correlation = spatial exclusivity).
    Returns (seed gene sets, candidate table with metrics).
    """
    from scipy.stats import spearmanr

    Xd = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    vidx = pd.Index(var_names)

    def gene_col(g):
        return Xd[:, vidx.get_loc(g)]

    rows = []
    scores_cache: dict[str, np.ndarray] = {}
    for tf, genes in regulons.items():
        genes = [g for g in genes if g in vidx]
        if len(genes) < min_module_genes or tf not in r2.index:
            continue
        ms = module_scores_from_genesets(X, var_names, {tf: genes})[tf].to_numpy()
        bic = bimodality_gap(ms, seed=seed)
        # coherence: mean |Spearman| between module score and member genes
        coh = [abs(spearmanr(gene_col(g), ms).statistic) for g in genes[:10]]
        rows.append({"TF": tf, "n_genes": len(genes), "r2": float(r2[tf]),
                     "delta_bic": bic, "coherence": float(np.mean(coh))})
        scores_cache[tf] = ms
    cand = pd.DataFrame(rows)
    if cand.empty:
        return {}, cand

    cand["r2_pct"] = cand.r2.rank(pct=True)
    cand["quality"] = cand.r2_pct * np.clip(cand.delta_bic, 0, 200) / 200 * \
        np.clip(cand.coherence, 0, 1)
    cand = cand.sort_values("quality", ascending=False).reset_index(drop=True)

    picked: list[str] = []
    picked_scores: list[np.ndarray] = []
    for _, row in cand.iterrows():
        if len(picked) >= n_seeds:
            break
        if row.delta_bic < min_bic:
            continue
        s = scores_cache[row.TF]
        if picked_scores:
            corrs = [abs(spearmanr(s, q).statistic) for q in picked_scores]
            if max(corrs) > max_abs_corr:
                continue
        picked.append(row.TF)
        picked_scores.append(s)

    seeds = {tf: [g for g in regulons[tf] if g in set(var_names)]
             for tf in picked}
    return seeds, cand
