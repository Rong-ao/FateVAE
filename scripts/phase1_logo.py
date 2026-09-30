#!/usr/bin/env python
"""Phase 1 anti-circularity check (proposal §7): leave-one-gene-out (LOGO).

For each LOGO split, supervision uses module scores computed WITHOUT the
held-out marker genes; evaluation asks whether the learned fate latent
nevertheless predicts the held-out genes' expression patterns (kNN AUROC on
the embedding). A high LOGO AUROC means the model recovers fate signal it
was not supervised on - the anti-circularity evidence.

Usage: python scripts/phase1_logo.py [--scale 0.5] [--tag logo]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from fatevae import config
from fatevae.eval import cell_splits, embedding_gene_auroc
from fatevae.io import load_disc, load_scenic
from fatevae.model import TrainConfig, build_fate_mask, train_fatevae
from fatevae.priors import (FATE_MODULES, extended_fate_genesets,
                            iter_logo_splits, module_scores)
from fatevae.baselines import hvg_pca

torch.set_num_threads(max(1, config.n_cpus()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--tag", default="logo")
    args = ap.parse_args()

    t0 = time.time()
    adata = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)
    auc, regulons = load_scenic(config.DMEL_SCENIC_LOOM, adata.obs_names)

    ext_sets = extended_fate_genesets(regulons, FATE_MODULES, adata.var_names)
    hvg = set(adata.var_names[adata.var["highly_variable"]])
    keep = sorted(hvg | {g for gs in ext_sets.values() for g in gs})
    keep = [g for g in keep if g in set(adata.var_names)]
    sub = adata[:, keep].copy()
    X = np.asarray(sub.X.todense(), dtype=np.float32)
    C = np.asarray(sub.layers["counts"].todense(), dtype=np.float32)
    lib = C.sum(axis=1).astype(np.float32)
    fate_mask = build_fate_mask(list(sub.var_names), ext_sets)
    train_m, val_m, _ = cell_splits(adata.obs["leiden_0.5"].to_numpy())
    hvg30 = hvg_pca(adata, 30)

    # held-out gene expression (full object gene space)
    def expr_of(g):
        return np.asarray(adata.X[:, adata.var_names.get_loc(g)].todense()).ravel()

    rows = []
    for si, (train_sets, held) in enumerate(iter_logo_splits()):
        held_in_data = [(f, g) for f, g in held if g in adata.var_names]
        print(f"\n-- LOGO split {si}: held out {held_in_data}", flush=True)
        scores = module_scores(sub.X, sub.var_names, train_sets)
        # gene set for the fate decoder mask keeps full extended sets (a
        # structural prior, not supervision); supervision is train-only
        cfg = TrainConfig(
            epochs_pretrain=int(100 * args.scale),
            epochs_sup=int(120 * args.scale),
            epochs_adv=int(80 * args.scale),
            batch_size=256, seed=config.SEED + si, log_every=0,
        )
        model, _ = train_fatevae(X, C, lib, fate_mask,
                                 scores[list(train_sets)].to_numpy(),
                                 train_m, val_m, cfg)
        zf, _, _ = model.latent(torch.tensor(X), torch.tensor(lib))
        for f, g in held_in_data:
            a_model = embedding_gene_auroc(zf, expr_of(g))
            a_hvg = embedding_gene_auroc(hvg30, expr_of(g))
            rows.append({"split": si, "fate": f, "gene": g,
                         "auroc_fatevae_zf": a_model, "auroc_hvg_pca30": a_hvg})
            print(f"   {f}/{g}: FateVAE-zf AUROC={a_model:.3f}  "
                  f"HVG-PCA={a_hvg:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(config.TABLES_DIR / f"phase1_logo_{args.tag}.csv", index=False)
    summary = pd.DataFrame({
        "FateVAE_zf": res.auroc_fatevae_zf.agg(["mean", "std"]),
        "HVG_PCA30": res.auroc_hvg_pca30.agg(["mean", "std"]),
    })
    summary.to_csv(config.TABLES_DIR / f"phase1_logo_summary_{args.tag}.csv")
    print("\nLOGO summary:")
    print(summary.round(3).to_string(), flush=True)
    print(f"\nLOGO done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
