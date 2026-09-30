"""Data-driven co-expression regulons (marker-free).

The pySCENIC cisTarget-pruned regulons retain only TFs whose motifs pass
enrichment - on our disc data that dropped 10 of the 14 known compartment
TFs, which explains why discovery mode built on them cannot recover fate
structure. Here we rebuild regulons by co-expression alone:

    for every TF in a genome-derived TF list (allTFs file - annotation, not
    experimental prior) detected in >= min_frac of cells, take its top-n
    positively correlated genes (Pearson on log-normalized expression).

This is a GENIE3-lite substitute computable without external packages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def coexpression_regulons(
    X,
    var_names,
    tf_list: list[str],
    n_targets: int = 30,
    min_frac: float = 0.03,
    min_corr: float = 0.15,
    chunk: int = 2048,
) -> dict[str, list[str]]:
    """Return {TF: [target genes]} by top positive Pearson correlation."""
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    n_cells, n_genes = X.shape
    # gene standardization once
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    vidx = pd.Index(var_names)
    tfs = [t for t in tf_list if t in vidx]
    detected = (X > 0).mean(axis=0)
    tfs = [t for t in tfs if detected[vidx.get_loc(t)] >= min_frac]
    print(f"[coexp] {len(tfs)} TFs pass detection (of {len(tf_list)} listed, "
          f"{len([t for t in tf_list if t in vidx])} present in data)")

    out: dict[str, list[str]] = {}
    tf_rows = [vidx.get_loc(t) for t in tfs]
    for start in range(0, len(tf_rows), chunk):
        rows = tf_rows[start:start + chunk]
        corr = (Z[:, rows].T @ Z) / n_cells  # (n_tfs_chunk, n_genes)
        for i, r in enumerate(rows):
            c = corr[i]
            c[r] = -1.0  # no self-targeting
            top = np.argsort(-c)[: n_targets * 2]
            sel = [j for j in top if c[j] >= min_corr][:n_targets]
            if len(sel) >= 3:
                out[tfs[start + i]] = [var_names[j] for j in sel]
    print(f"[coexp] built {len(out)} regulons (>=3 targets at corr>={min_corr})")
    return out
