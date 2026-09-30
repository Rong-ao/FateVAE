"""Fate-aware GRN attribution (proposal §6.3, v0.1).

TF -> fate effects are estimated by multinomial logistic regression of the
model's calibrated fate probabilities on SCENIC regulon activities across
cells. In-silico perturbation shifts one regulon's activity to a chosen
quantile and re-predicts fate probabilities.

This is the interpretable, robust linear probe promised in §6.3.2; the
decoder-gradient attribution (§6.3.2 "importance(t,k)") plugs into the same
API once the decoder is trained (see Note at the bottom).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _fit_multinomial(auc: pd.DataFrame, hard: np.ndarray, seed: int = 0):
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, random_state=seed)
    clf.fit(auc.to_numpy(float), hard)
    return clf


def tf_fate_attribution(
    auc: pd.DataFrame,
    probs: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """(n_tfs x n_fates) signed attribution matrix.

    Row t, column f = change in log-odds of fate f per SD of regulon-t
    activity. Standardized AUCs make coefficients comparable across TFs.
    """
    Z = (auc - auc.mean()) / (auc.std() + 1e-9)
    hard = probs.to_numpy().argmax(axis=1)
    clf = _fit_multinomial(Z, hard, seed)
    coefs = clf.coef_.T  # (n_tfs, n_fates)
    out = pd.DataFrame(coefs, index=auc.columns, columns=probs.columns)
    out.index.name = "TF"
    return out


def rank_regulators(attr: pd.DataFrame, fate: str) -> pd.DataFrame:
    """Regulator ranking for one fate, signed."""
    r = attr[fate].sort_values(key=np.abs, ascending=False)
    return r.to_frame("coef").assign(abs_coef=lambda d: d.coef.abs())


def in_silico_perturb(
    auc: pd.DataFrame,
    probs: pd.DataFrame,
    tf: str,
    quantile: float = 0.9,
    seed: int = 0,
) -> pd.DataFrame:
    """Counterfactual: shift regulon-tf activity to `quantile` in all cells,
    return mean fate-probability shift (and per-fate means)."""
    Z = (auc - auc.mean()) / (auc.std() + 1e-9)
    hard = probs.to_numpy().argmax(axis=1)
    clf = _fit_multinomial(Z, hard, seed)
    Zp = Z.copy()
    Zp[tf] = Z[tf].quantile(quantile)
    p0 = clf.predict_proba(Z.to_numpy(float))
    p1 = clf.predict_proba(Zp.to_numpy(float))
    delta = p1 - p0
    return pd.DataFrame(
        {"mean_delta": delta.mean(axis=0), "fate": probs.columns}
    ).set_index("fate")


# Note (decoder-gradient attribution): once a FateVAE checkpoint is loaded,
# d logit_k / d activity_t can be computed by attaching regulon activities
# as encoder side-inputs (proposal §6.2.1 side inputs). The linear probe
# above is the robust v0.1 the proposal lists as the complementary method.
