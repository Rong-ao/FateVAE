"""Shared prior-recovery evaluation (proposal v2 §5 contract).

Known markers enter ONLY here - never in training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..priors.markers import FATE_MODULES
from .metrics import embedding_gene_auroc


def prior_recovery(zf: np.ndarray, dims: np.ndarray, X, var_names,
                   module_score_fn=None, nominees: dict[str, list[str]] | None = None,
                   verbose: bool = True) -> dict:
    """Evaluate a discovered fate latent against the known Dmel priors.

    zf: (cells, K) fate latent; dims: active dim indices.
    Returns dict with ARI, NMI, marker_auroc, mapping, per-gene AUROCs.
    """
    from ..priors.markers import module_scores

    scores = module_scores(X, var_names, FATE_MODULES)
    known = scores.idxmax(1).to_numpy()
    fate_names = list(FATE_MODULES)
    emb = zf[:, dims]

    corr = np.zeros((len(dims), len(fate_names)))
    for i, d in enumerate(dims):
        for j, f in enumerate(fate_names):
            corr[i, j] = spearmanr(zf[:, d], scores[f]).statistic
    ri, ci = linear_sum_assignment(-np.abs(corr))
    mapping = {int(dims[i]): fate_names[j] for i, j in zip(ri, ci)}

    # VAE dims are sign-arbitrary: sign-align each dim positively with its
    # matched fate before computing any partition metric
    sign = np.ones(emb.shape[1])
    for i, d in enumerate(dims):
        j = [jj for jj, (ii, jj2) in enumerate(zip(ri, ci)) if ii == i]
        if j:
            sign[i] = np.sign(corr[i, ci[j[0]]]) or 1.0
    emb_aligned = emb * sign[:, None].T

    probs = _softmax_std(emb_aligned)
    pred = probs.argmax(1)
    ari = adjusted_rand_score(known, pred)
    nmi = normalized_mutual_info_score(known, pred)

    markers = sorted({g for gs in FATE_MODULES.values() for g in gs})
    vidx = pd.Index(var_names)
    aucs = {}
    for g in markers:
        if g in vidx:
            col = X[:, vidx.get_loc(g)]
            col = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
            aucs[g] = embedding_gene_auroc(emb, col)
    valid = [v for v in aucs.values() if not np.isnan(v)]

    out = {"ARI": ari, "NMI": nmi, "marker_auroc": float(np.mean(valid)),
           "marker_auroc_sd": float(np.std(valid)), "per_gene_auroc": aucs,
           "mapping": mapping, "dim_fate_corr": corr}
    if verbose:
        print(f"[recovery] ARI={ari:.3f} NMI={nmi:.3f} "
              f"marker AUROC={np.mean(valid):.3f}±{np.std(valid):.3f}")
        for d, f in mapping.items():
            print(f"[recovery] dim {d} -> {f} (r={corr[list(dims).index(d), fate_names.index(f)]:.3f})")
    if nominees:
        rec_rows = []
        for m, genes in nominees.items():
            f = None
            for d, ff in mapping.items():
                if f"F{d}" == m or m == ff:
                    f = ff
            f = f or (m if m in FATE_MODULES else None)
            if f is None:
                continue
            known_tfs = FATE_MODULES.get(f, [])
            hits = [g for g in genes if g in known_tfs]
            rec_rows.append({"module": m, "matched_fate": f,
                             "nominees": ", ".join(genes[:12]),
                             "known_hits": ", ".join(hits),
                             "n_known_hits": len(hits)})
        out["nominee_recall"] = pd.DataFrame(rec_rows)
        if verbose and len(rec_rows):
            print("[recovery] nominee known-hits:")
            print(out["nominee_recall"][["module", "matched_fate",
                                         "known_hits", "n_known_hits"]]
                  .to_string(index=False))
    return out


def _softmax_std(emb: np.ndarray) -> np.ndarray:
    z = (emb - emb.mean(0)) / (emb.std(0) + 1e-9)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
