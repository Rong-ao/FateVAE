"""Log-domain entropic (unbalanced) Sinkhorn, pure NumPy/SciPy.

We solve    min <T, C> + eps * KL(T | a b^T) + tau * KL(T1 | a) + tau * KL(T^T 1 | b)

where KL is the Kullback-Leibler divergence of the marginals (tau < inf =>
unbalanced OT: mass may be created/destroyed - appropriate when source and
target are cross-sectional snapshots with growth and cell death). tau = inf
reduces to balanced entropic OT.

Scaling iteration (Chizat et al. 2018), computed in the log domain for
numerical stability:

    r = tau / (tau + eps)
    log u_i <- r * (log a_i   - LSE_j[(g_j - C_ij)/eps])
    log v_j <- r * (log b_j   - LSE_i[(f_i - C_ij)/eps])   with f = eps*log u
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def sinkhorn_unbalanced(
    C: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    eps: float = 0.1,
    tau: float = 10.0,
    n_iter: int = 2000,
    tol: float = 1e-8,
    log_plan: bool = False,
) -> dict:
    """Returns dict with potentials (f, g), plan T (n x m), diagnostics.

    C: (n, m) nonnegative cost matrix (row-normalized scaling is caller's
       job - pass C / median(C) for a scale-free epsilon).
    a, b: marginal weights (will be normalized internally); default uniform.
    eps: entropic regularization; tau: marginal KL strength (np.inf=balanced).
    """
    n, m = C.shape
    a = np.full(n, 1.0 / n) if a is None else a / a.sum()
    b = np.full(m, 1.0 / m) if b is None else b / b.sum()
    la, lb = np.log(a), np.log(b)
    Mc = -C / eps  # (n, m)

    balanced = not np.isfinite(tau)
    r = 1.0 if balanced else tau / (tau + eps)

    f = np.zeros(n)
    g = np.zeros(m)
    prev = np.inf
    for it in range(n_iter):
        f = r * eps * (la - logsumexp(Mc + (g / eps)[None, :], axis=1))
        g = r * eps * (lb - logsumexp(Mc.T + (f / eps)[None, :], axis=1))
        if it % 10 == 0:
            # total-mass drift for monitoring
            row = np.exp(logsumexp((f[:, None] + g[None, :] - C) / eps, axis=1))
            err = np.abs(row.sum() - 1.0)
            if abs(prev - err) < tol:
                break
            prev = err

    logT = (f[:, None] + g[None, :] - C) / eps
    T = np.exp(logT)
    out = {"f": f, "g": g, "T": T, "logT": logT if log_plan else None,
           "iters": it, "mass": float(T.sum()),
           "row_mass": T.sum(axis=1), "col_mass": T.sum(axis=0)}
    return out


def row_normalized(T: np.ndarray) -> np.ndarray:
    """Each source cell's distribution over targets (rows sum to <=1; the
    deficit is mass the unbalanced coupling chose to drop)."""
    s = T.sum(axis=1, keepdims=True)
    return np.divide(T, s, out=np.zeros_like(T), where=s > 0)
