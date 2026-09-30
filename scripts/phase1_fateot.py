#!/usr/bin/env python
"""FateOT (v0.3b): data-only fate discovery via optimal transport to
UNLABELED adult clusters. The adult cells' own structure is the teacher -
no annotations, no marker genes.

Stage 1  Leiden-cluster adult cells (data-derived terminal states)
Stage 2  Sinkhorn OT: larval cells -> adult clusters (joint PCA space)
Stage 3  informed-mode FateVAE trained on OT soft-assignments + cluster
         DE gene sets (all data-derived)
Stage 4  prior-recovery test (markers only at evaluation)

Usage: python scripts/phase1_fateot.py [--reg 0.05] [--tag ot1]
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
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from fatevae import config
from fatevae.eval import embedding_gene_auroc, fate_entropy
from fatevae.io import load_disc
from fatevae.model import TrainConfig, build_fate_mask, train_fatevae
from fatevae.ot_teacher import cluster_adults, cluster_genesets, larva_pseudo_fates
from fatevae.priors import FATE_MODULES, module_scores
from fatevae.viz import multi_scatter

torch.set_num_threads(max(1, config.n_cpus()))

KNOWN_MARKERS = sorted({g for gs in FATE_MODULES.values() for g in gs})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", type=float, default=0.05)
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--tag", default="ot1")
    ap.add_argument("--epochs-scale", type=float, default=1.0)
    args = ap.parse_args()
    t0 = time.time()

    adata = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)

    # ---------------- Stage 1+2: OT teacher -------------------------------
    print("== Stage 1-2: adult clusters + OT coupling ==", flush=True)
    soft, genesets, adult_labels = larva_pseudo_fates(
        resolution=args.resolution, reg=args.reg)
    print(f"adult clusters: {dict(adult_labels.value_counts())}", flush=True)
    print(f"OT soft-assignment: {soft.shape}; mean max-prob "
          f"{soft.max(1).mean():.3f}", flush=True)
    common = adata.obs_names.intersection(soft.index)
    if len(common) < 0.9 * len(adata):
        print(f"WARNING: only {len(common)}/{len(adata)} larval cells matched "
              "in merged object", flush=True)
    soft = soft.loc[common]
    adata = adata[common].copy()

    # ---------------- Stage 3: train on OT pseudo-fates --------------------
    print("\n== Stage 3: FateVAE on OT pseudo-fates ==", flush=True)
    clusters = list(soft.columns)
    genesets = {c: [g for g in genesets.get(c, []) if g in adata.var_names]
                for c in clusters}
    genesets = {c: gs for c, gs in genesets.items() if len(gs) >= 3}
    clusters = list(genesets)
    soft = soft[clusters]
    print(f"usable cluster gene sets: {len(clusters)} "
          f"(sizes { {c: len(g) for c, g in genesets.items()} })", flush=True)

    hvg = set(adata.var_names[adata.var["highly_variable"]])
    keep = sorted((hvg | {g for gs in genesets.values() for g in gs})
                  & set(adata.var_names))
    sub = adata[:, keep].copy()
    X = np.asarray(sub.X.todense(), dtype=np.float32)
    C = np.asarray(sub.layers["counts"].todense(), dtype=np.float32)
    lib = C.sum(axis=1).astype(np.float32)
    mask = build_fate_mask(keep, genesets)
    cfg = TrainConfig(epochs_pretrain=int(100 * args.epochs_scale),
                      epochs_sup=int(120 * args.epochs_scale),
                      epochs_adv=int(80 * args.epochs_scale),
                      batch_size=256, seed=config.SEED)
    model, hist = train_fatevae(X, C, lib, mask,
                                soft.to_numpy(),
                                np.ones(len(X), bool), np.zeros(len(X), bool),
                                cfg)
    zf, _, _ = model.latent(torch.tensor(X), torch.tensor(lib))
    torch.save({"state_dict": model.state_dict(), "genes": keep,
                "genesets": genesets,
                "cfg": {k: v for k, v in vars(cfg).items() if k != "history"}},
               config.MODELS_DIR / f"fateot_{args.tag}.pt")

    # ---------------- Stage 4: prior-recovery ------------------------------
    print("\n== Stage 4: prior-recovery test ==", flush=True)
    known_scores = module_scores(adata.X, adata.var_names, FATE_MODULES)
    known = known_scores.idxmax(1).to_numpy()
    fate_names = list(FATE_MODULES)
    dims = np.arange(len(clusters))
    corr = np.array([[spearmanr(zf[:, d], known_scores[f]).statistic
                      for f in fate_names] for d in dims])
    ri, ci = linear_sum_assignment(-corr)
    mapping = {int(d): fate_names[j] for d, j in zip(ri, ci)}
    for d, j in zip(ri, ci):
        print(f"  OT dim {d} (cluster {clusters[d]}) -> {fate_names[j]}: "
              f"r={corr[d, j]:.3f}", flush=True)

    zstd = (zf - zf.mean(0)) / (zf.std(0) + 1e-9)
    e = np.exp(zstd - zstd.max(1, keepdims=True))
    probs = e[:, dims] / e[:, dims].sum(1, keepdims=True)
    pred = probs.argmax(1)
    ari = adjusted_rand_score(known, pred)
    nmi = normalized_mutual_info_score(known, pred)
    print(f"\nARI={ari:.3f}  NMI={nmi:.3f}  (gate 0.3; v0.2 best 0.032; "
          f"FateSeed 0.026)", flush=True)

    heldout = {g: np.asarray(adata.X[:, adata.var_names.get_loc(g)].todense()).ravel()
               for g in KNOWN_MARKERS if g in adata.var_names}
    aucs = {g: embedding_gene_auroc(zf[:, dims], e2) for g, e2 in heldout.items()}
    val = [v for v in aucs.values() if not np.isnan(v)]
    print(f"marker AUROC (FateOT z_f): mean {np.mean(val):.3f} "
          f"(sd {np.std(val):.3f})  [informed 0.874, HVG-PCA 0.805]",
          flush=True)

    pd.DataFrame(probs, columns=clusters, index=adata.obs_names).assign(
        entropy=fate_entropy(probs)).to_csv(
        config.TABLES_DIR / f"fateot_probs_{args.tag}.csv")
    pd.DataFrame([{"method": "FateOT", "ARI": ari, "NMI": nmi,
                   "marker_auroc": float(np.mean(val)),
                   "n_clusters": len(clusters), "reg": args.reg,
                   "resolution": args.resolution}]).to_csv(
        config.TABLES_DIR / f"fateot_benchmark_{args.tag}.csv", index=False)
    pd.DataFrame(hist).to_csv(
        config.TABLES_DIR / f"fateot_history_{args.tag}.csv", index=False)

    umap = np.asarray(adata.obsm["X_umap"])
    panels = {c: soft[c].to_numpy() for c in clusters[:6]}
    multi_scatter(umap, panels,
                  config.FIGURES_DIR / f"fateot_soft_umap_{args.tag}.png",
                  cmap="coolwarm", ncol=3,
                  panel_titles={c: f"OT mass -> cluster {c}" for c in clusters[:6]})
    panels2 = {f"dim{d}": probs[:, i] for i, d in enumerate(dims[:6])}
    multi_scatter(umap, panels2,
                  config.FIGURES_DIR / f"fateot_probs_umap_{args.tag}.png",
                  cmap="coolwarm", ncol=3,
                  panel_titles={f"dim{d}": f"{clusters[d]} -> {mapping.get(d, '?')}" for d in dims[:6]})

    print(f"\nFateOT done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
