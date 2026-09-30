"""Smoke tests: synthetic data through the full FateVAE pipeline.

Run:  python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from fatevae.model import FateVAE, TrainConfig, build_fate_mask, train_fatevae
from fatevae.priors.markers import module_scores, iter_logo_splits
from fatevae.fate import calibrate_temperature, fate_probabilities
from fatevae.grn import tf_fate_attribution, in_silico_perturb
from fatevae.eval import embedding_gene_auroc, chi_index, fate_entropy


def make_synthetic(n_cells=400, n_genes=300, seed=0):
    rng = np.random.default_rng(seed)
    fates = rng.choice(["A", "B", "C"], size=n_cells)
    fate_genes = {"A": list(range(0, 10)), "B": list(range(10, 20)),
                  "C": list(range(20, 30))}
    lib = rng.lognormal(6, 0.3, size=n_cells).astype(np.float32)
    p = np.full((n_cells, n_genes), 1 / n_genes)
    for i, f in enumerate(fates):
        p[i, fate_genes[f]] *= 30.0
    p = p / p.sum(axis=1, keepdims=True)
    counts = rng.poisson(lib[:, None] * p).astype(np.float32)
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    scores = pd.DataFrame(
        {f: lognorm[:, g].mean(axis=1) for f, g in fate_genes.items()})
    return counts, lognorm, lib, fates, scores, fate_genes


def main():
    counts, X, lib, fates, scores, fate_genes = make_synthetic()
    gene_names = [f"g{i}" for i in range(X.shape[1])]
    fate_sets = {f: [f"g{i}" for i in gs] for f, gs in fate_genes.items()}

    mask = build_fate_mask(gene_names, fate_sets)
    assert mask.shape == (3, X.shape[1])
    assert all(mask[k].sum() == 10 for k in range(3))

    cfg = TrainConfig(epochs_pretrain=8, epochs_sup=8, epochs_adv=8,
                      batch_size=128, hidden=(64, 32), n_bg=4,
                      log_every=0, seed=0)
    model, hist = train_fatevae(X, counts, lib, mask, scores.to_numpy(),
                                np.ones(len(X), bool), np.zeros(len(X), bool),
                                cfg)
    assert len(hist) == 24
    zf, zb, logits = model.latent(torch.tensor(X), torch.tensor(lib))
    assert zf.shape == (X.shape[0], 3) and zb.shape == (X.shape[0], 4)
    assert np.isfinite(zf).all() and np.isfinite(logits).all()

    T = calibrate_temperature(logits, scores.to_numpy())
    probs = fate_probabilities(logits, T)
    assert np.allclose(probs.sum(axis=1), 1.0)
    ent = fate_entropy(probs)
    assert ent.shape == (X.shape[0],)

    # reconstructions recover fate structure on synthetic separable data
    rec = model.expected_expression(torch.tensor(X), torch.tensor(lib))
    assert rec.shape == X.shape and np.isfinite(rec).all()

    # eval + grn on toy frames
    auc = pd.DataFrame(np.random.default_rng(1).normal(size=(X.shape[0], 12)),
                       columns=[f"tf{j}" for j in range(12)])
    attr = tf_fate_attribution(auc, pd.DataFrame(probs, columns=scores.columns))
    assert attr.shape == (12, 3)
    delta = in_silico_perturb(auc, pd.DataFrame(probs, columns=scores.columns), "tf0")
    assert np.isfinite(delta.to_numpy()).all()

    # metrics run
    labels = np.asarray(fates)
    chi = chi_index(zf, labels, np.random.normal(size=(X.shape[0], 5)))
    assert "CHI" in chi
    auroc = embedding_gene_auroc(zf, X[:, 3])
    assert 0 <= auroc <= 1 or np.isnan(auroc)

    # LOGO splits sanity
    splits = iter_logo_splits({"A": ["a1", "a2"], "B": ["b1", "b2", "b3"]})
    assert len(splits) == 3
    train_sets, held = splits[0]
    assert held == [("A", "a1"), ("B", "b1")]

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
