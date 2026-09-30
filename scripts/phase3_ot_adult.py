#!/usr/bin/env python
"""OT demo 1 (proposal v2 Aim 5): larval genital disc -> adult terminal
fates via unbalanced entropic OT in the integrated merged-PCA space.

Question: does an OT coupling give a sharper prospective fate assignment
than the naive kNN anchoring that was uninformative in v0.1?

Evaluation (markers enter HERE ONLY): cross-correlation between per-larval-
cell adult-fate mass and the disc compartment module scores (A8/A9p1/A9p2/
A10); OT vs kNN head-to-head; sensitivity over (eps, tau).

Usage: python scripts/phase3_ot_adult.py [--tag ot1]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fatevae import config
from fatevae.io import load_disc
from fatevae.ot import ot_fate_mass
from fatevae.ot.coupling import build_ot_problem, kNN_anchor_baseline
from fatevae.priors.markers import FATE_MODULES, module_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ot1")
    args = ap.parse_args()
    t0 = time.time()

    merged = ad.read_h5ad(config.DMEL_MERGED_H5AD)
    is_larva = (merged.obs["sample"] == "larva").to_numpy()
    ann = merged.obs["adult_annotation"].astype(str)
    groups = config.ADULT_FATE_GROUPS
    adult_group = pd.Series(np.nan, index=merged.obs_names)
    for g, cats in groups.items():
        adult_group[ann.isin(cats)] = g

    Es, Et, tgroups, C = build_ot_problem(
        np.asarray(merged.obsm["X_pca"]), is_larva, adult_group,
        balance={"accessory_gland": 1200}, seed=0)
    print(f"OT problem: {Es.shape[0]} larval x {Et.shape[0]} adult target "
          f"cells; groups {pd.Series(tgroups).value_counts().to_dict()}",
          flush=True)

    # disc compartment scores for evaluation
    disc = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)
    scores = module_scores(disc.X, disc.var_names, FATE_MODULES)

    # ---- OT grid -----------------------------------------------------------
    rows = []
    best = None
    for eps in (0.02, 0.05, 0.1, 0.2):
        for tau in (1.0, 5.0, 50.0):
            mass = ot_fate_mass(C, tgroups, eps=eps, tau=tau)
            cc = pd.DataFrame(
                {f: [spearmanr(mass[g], scores[f]).statistic
                     for g in mass.columns] for f in FATE_MODULES},
                index=mass.columns)
            maxabs = float(np.nanmax(np.abs(cc.to_numpy())))
            rows.append({"eps": eps, "tau": tau, "mass": mass.attrs["mass"],
                         "max_abs_corr": maxabs})
            print(f"eps={eps} tau={tau}: plan mass={mass.attrs['mass']:.3f} "
                  f"max|corr|={maxabs:.3f}", flush=True)
            if best is None or maxabs > best[0]:
                best = (maxabs, eps, tau, mass, cc)
    grid = pd.DataFrame(rows)
    grid.to_csv(config.TABLES_DIR / f"phase3_ot_grid_{args.tag}.csv", index=False)

    _, eps_b, tau_b, mass, cc = best
    print(f"\nBest config eps={eps_b} tau={tau_b}; cross-correlation "
          "(rows: adult fate mass, cols: disc module score):", flush=True)
    print(cc.round(3).to_string(), flush=True)
    cc.to_csv(config.TABLES_DIR / f"phase3_ot_crosscorr_{args.tag}.csv")

    # ---- kNN baseline head-to-head ----------------------------------------
    knn = kNN_anchor_baseline(Es, Et, tgroups, k=30)
    cc_knn = pd.DataFrame(
        {f: [spearmanr(knn[g], scores[f]).statistic for g in knn.columns]
         for f in FATE_MODULES}, index=knn.columns)
    print("\nkNN-30 baseline cross-correlation:", flush=True)
    print(cc_knn.round(3).to_string(), flush=True)
    cc_knn.to_csv(config.TABLES_DIR / f"phase3_knn_crosscorr_{args.tag}.csv")

    # ---- figures ------------------------------------------------------------
    umap = np.asarray(merged.obsm["X_umap"])[np.where(is_larva)[0]]
    fig, axes = plt.subplots(1, mass.shape[1] + 1,
                             figsize=(3 * (mass.shape[1] + 1), 2.8), dpi=170)
    for ax, g in zip(axes, mass.columns):
        sc = ax.scatter(umap[:, 0], umap[:, 1], c=mass[g], cmap="viridis",
                        s=2, alpha=0.7, linewidths=0)
        ax.set_title(f"OT mass -> {g}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.7)
    ent = -(mass.to_numpy() / mass.to_numpy().sum(1, keepdims=True)
            * np.log(mass.to_numpy() / mass.to_numpy().sum(1, keepdims=True)
                     + 1e-12)).sum(1)
    sc = axes[-1].scatter(umap[:, 0], umap[:, 1], c=ent, cmap="magma", s=2,
                          alpha=0.7, linewidths=0)
    axes[-1].set_title("OT fate entropy", fontsize=9)
    axes[-1].set_xticks([]); axes[-1].set_yticks([])
    fig.colorbar(sc, ax=axes[-1], shrink=0.7)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"phase3_ot_umap_{args.tag}.png",
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=170)
    x = np.arange(cc.size)
    ax.bar(x - 0.2, cc.to_numpy().ravel(), width=0.4, label="OT (best cfg)")
    ax.bar(x + 0.2, cc_knn.loc[cc.index, cc.columns].to_numpy().ravel(),
           width=0.4, label="kNN-30")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}-{c}" for r in cc.index for c in cc.columns],
                       rotation=60, ha="right", fontsize=7)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Spearman r")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"phase3_ot_vs_knn_{args.tag}.png",
                bbox_inches="tight")
    plt.close(fig)

    mass.to_csv(config.TABLES_DIR / f"phase3_ot_fate_mass_{args.tag}.csv")
    print(f"\nDemo 1 done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
