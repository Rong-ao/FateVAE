"""Fate-marker prior modules and anti-circularity split machinery.

The gene sets below are the IF-validated compartment markers of the male
genital disc used in the lab's AUCell.R analysis (A8 / A9 / A10 primordia;
A9 split into two sub-domains). They are the biological ground truth that
the proposal splits into train / held-out portions so that every reported
metric measures *recovery of unseen signal*.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Compartment markers as defined in AUCell.R (kept verbatim; hh is shared
# between the two A9 sub-domains - biologically it is the A9 boundary signal)
FATE_MODULES: dict[str, list[str]] = {
    "A8": ["abd-A", "tsh"],
    "A9p1": ["btl", "Abd-B", "hh", "Dr", "dpp"],
    "A9p2": ["bnl", "en", "Delta", "dac", "hh"],
    "A10": ["cad", "Dll"],
}

# Human-readable fate names used across plots/tables
FATE_LABELS = {
    "A8": "A8 (accessory gland + ejac. duct)",
    "A9p1": "A9 domain 1",
    "A9p2": "A9 domain 2",
    "A10": "A10 (anal plate)",
}


def genes_present(genes: Iterable[str], var_names) -> list[str]:
    vn = set(var_names)
    return [g for g in genes if g in vn]


def module_scores(
    X,
    var_names,
    gene_sets: dict[str, list[str]],
    standardize: bool = True,
) -> pd.DataFrame:
    """Per-cell module activity: mean of z-scored log-expression.

    X: cells x genes (sparse or dense, log-normalized). Genes absent from
    var_names are skipped (reported by caller if needed).
    """
    var_index = {g: i for i, g in enumerate(var_names)}
    n_cells = X.shape[0]
    out = pd.DataFrame(np.nan, index=range(n_cells), columns=list(gene_sets))
    for name, genes in gene_sets.items():
        cols = [var_index[g] for g in genes if g in var_index]
        if not cols:
            continue
        block = X[:, cols]
        block = block.toarray() if sp.issparse(block) else np.asarray(block)
        if standardize:
            sd = block.std(axis=0, ddof=0)
            sd[sd == 0] = 1.0
            block = (block - block.mean(axis=0)) / sd
        out[name] = block.mean(axis=1)
    return out


def iter_logo_splits(
    gene_sets: dict[str, list[str]] | None = None,
) -> list[tuple[dict[str, list[str]], list[tuple[str, str]]]]:
    """Leave-one-gene-out (LOGO) splits over the marker modules.

    Returns a list of (train_sets, held_out) pairs, where held_out is a list
    of (module, gene) tuples - one gene per module per split. With the
    current modules this yields 2+5+5+2 = 14 splits. Because modules are
    tiny, holding out exactly one gene per module per split is the finest
    granularity that still leaves a usable training signal.
    """
    gene_sets = gene_sets or FATE_MODULES
    splits = []
    per_module_held = {
        name: [g for g in genes] for name, genes in gene_sets.items()
    }
    # round-robin: split i holds out the i-th gene of each module that has one
    max_len = max(len(v) for v in per_module_held.values())
    for i in range(max_len):
        held: list[tuple[str, str]] = []
        train: dict[str, list[str]] = {}
        for name, genes in gene_sets.items():
            if i < len(genes):
                held.append((name, genes[i]))
                rest = [g for j, g in enumerate(genes) if j != i]
            else:
                rest = list(genes)
            train[name] = rest
        splits.append((train, held))
    return splits


def extended_fate_genesets(
    regulons: dict[str, list[str]],
    gene_sets: dict[str, list[str]] | None = None,
    var_names=None,
    max_targets: int = 60,
) -> dict[str, list[str]]:
    """Fate gene sets extended with SCENIC regulon targets of module TFs.

    Used to build the group-sparse decoder mask: each fate latent dimension
    may only decode genes in its extended set. ``max_targets`` caps each
    regulon so that ubiquitous regulons do not flood a fate group.
    """
    gene_sets = gene_sets or FATE_MODULES
    out: dict[str, list[str]] = {}
    for name, genes in gene_sets.items():
        ext: list[str] = []
        for tf in genes:
            targets = regulons.get(tf, [])
            ext.extend(targets[:max_targets])
        ext = sorted(set(ext) | set(genes))
        if var_names is not None:
            ext = genes_present(ext, var_names)
        out[name] = ext
    return out
