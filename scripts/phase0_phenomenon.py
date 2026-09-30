#!/usr/bin/env python
"""Phase 0 (proposal §6.1): quantify cryptic fate heterogeneity + baselines.

Outputs (results/):
  figures/phase0_umap_phenomenon.png  - round UMAP vs marker-space structure
  figures/phase0_regulon_umap.png     - SCENIC-AUC fate-aware UMAP
  tables/phase0_benchmark.csv         - embedding benchmark (Aim-1 contract)
  tables/phase0_variance_decomposition.csv  - H1 test (fate variance ratio)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from fatevae import config
from fatevae.baselines import hvg_pca, marker_pca, regulon_pca
from fatevae.eval import embedding_gene_auroc, embedding_silhouette, chi_index
from fatevae.io import load_disc, load_scenic
from fatevae.priors import FATE_MODULES, module_scores
from fatevae.viz import multi_scatter, scatter_embedding

import matplotlib.pyplot as plt
import scanpy as sc


def eta_squared(emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-dimension between-fate variance fraction (ANOVA eta^2)."""
    lab = pd.Series(labels).astype(str).to_numpy()
    out = np.zeros(emb.shape[1])
    grand = emb.mean(axis=0)
    ss_tot = ((emb - grand) ** 2).sum(axis=0)
    for f in np.unique(lab):
        m = lab == f
        out += m.sum() * (emb[m].mean(axis=0) - grand) ** 2
    out = np.divide(out, ss_tot, out=np.zeros_like(out), where=ss_tot > 0)
    return out


def main() -> None:
    np.random.seed(config.SEED)
    print("== Phase 0: phenomenon quantification ==", flush=True)

    adata = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)
    auc, regulons = load_scenic(config.DMEL_SCENIC_LOOM, adata.obs_names)
    print(f"disc data: {adata.shape[0]} cells x {adata.shape[1]} genes; "
          f"{auc.shape[1]} regulons", flush=True)

    # ---- marker availability + fate labels ---------------------------------
    avail = {f: [g for g in gs if g in adata.var_names] for f, gs in FATE_MODULES.items()}
    missing = {f: [g for g in gs if g not in adata.var_names] for f, gs in FATE_MODULES.items()}
    missing = {f: g for f, g in missing.items() if g}
    if missing:
        print(f"WARNING marker genes missing from data: {missing}", flush=True)
    scores = module_scores(adata.X, adata.var_names, FATE_MODULES)
    fate_label = scores.idxmax(axis=1).to_numpy()
    print("fate label composition (argmax of module scores):")
    print(scores.idxmax(axis=1).value_counts().to_string(), flush=True)

    # ---- H1 test: fate-variance ratio across representations --------------
    rows = []
    hvg = hvg_pca(adata, 30)
    reg = regulon_pca(auc, 10)
    mk_genes = sorted({g for gs in FATE_MODULES.values() for g in gs if g in adata.var_names})
    mk = marker_pca(adata, mk_genes, 6)
    var_share_hvg = np.asarray(adata.uns["pca"]["variance_ratio"][:30], dtype=float)
    for name, emb, share in [("hvg_pca30", hvg, var_share_hvg),
                             ("regulon_pca10", reg, None),
                             ("marker_pca", mk, None)]:
        etas = eta_squared(emb, fate_label)
        w = share if share is not None else np.full(emb.shape[1], 1 / emb.shape[1])
        rows.append({"representation": name,
                     "fate_variance_ratio": float((etas * w).sum()),
                     "max_dim_eta2": float(etas.max())})
    vd = pd.DataFrame(rows)
    vd.to_csv(config.TABLES_DIR / "phase0_variance_decomposition.csv", index=False)
    print("\nH1 variance decomposition (between-fate variance fraction):")
    print(vd.to_string(index=False), flush=True)

    # ---- phenomenon figure: existing UMAP (variance-aware) -----------------
    umap = np.asarray(adata.obsm["X_umap"]).copy()
    panels = {f: scores[f].to_numpy() for f in FATE_MODULES}
    multi_scatter(umap, panels, config.FIGURES_DIR / "phase0_umap_phenomenon.png",
                 cmap="coolwarm", ncol=2,
                 panel_titles={f: f"{f} marker module score" for f in FATE_MODULES})

    # ---- fate-aware UMAP from regulon AUC ----------------------------------
    adata.obsm["X_hvg_umap"] = umap  # keep the variance-aware UMAP
    adata.obsm["X_regulon_pca"] = reg
    sc.pp.neighbors(adata, use_rep="X_regulon_pca", n_neighbors=15, random_state=0)
    sc.tl.umap(adata, random_state=0)  # writes X_umap from regulon PCA
    reg_umap = np.asarray(adata.obsm["X_umap"])
    multi_scatter(reg_umap, panels, config.FIGURES_DIR / "phase0_regulon_umap.png",
                  cmap="coolwarm", ncol=2,
                  panel_titles={f: f"{f} marker module score" for f in FATE_MODULES})

    # ---- benchmark table ----------------------------------------------------
    heldout = {g: np.asarray(adata.X[:, adata.var_names.get_loc(g)].todense()).ravel()
               for f, gs in FATE_MODULES.items() for g in gs if g in adata.var_names}
    table = []
    for name, emb in [("hvg_pca30", hvg), ("regulon_pca10", reg), ("marker_pca6", mk)]:
        row = {"method": name, "dim": emb.shape[1]}
        row["silhouette"] = embedding_silhouette(emb, fate_label)
        aucs = {g: embedding_gene_auroc(emb, e) for g, e in heldout.items()}
        valid = [v for v in aucs.values() if not np.isnan(v)]
        row["heldout_marker_auroc"] = float(np.mean(valid))
        row["heldout_marker_auroc_sd"] = float(np.std(valid))
        row.update({f"auc_{g}": v for g, v in aucs.items()})
        chi = chi_index(emb, fate_label, hvg)
        row["CHI"] = chi["CHI"]
        table.append(row)
    bench = pd.DataFrame(table)
    bench.to_csv(config.TABLES_DIR / "phase0_benchmark.csv", index=False)
    cols = ["method", "dim", "heldout_marker_auroc", "heldout_marker_auroc_sd",
            "silhouette", "CHI"]
    print("\nPhase-0 benchmark:")
    print(bench[[c for c in cols if c in bench.columns]].to_string(index=False), flush=True)

    # ---- barplot of per-gene AUROC -----------------------------------------
    gene_cols = [c for c in bench.columns if c.startswith("auc_")]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=170)
    x = np.arange(len(gene_cols))
    w = 0.8 / len(bench)
    for i, (_, r) in enumerate(bench.iterrows()):
        ax.bar(x + i * w, [r[c] for c in gene_cols], width=w, label=r["method"])
    ax.axhline(0.5, color="grey", lw=0.6, ls="--")
    ax.set_xticks(x + w * (len(bench) - 1) / 2)
    ax.set_xticklabels([c[4:] for c in gene_cols], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUROC (marker-gene recovery)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "phase0_marker_auroc.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nPhase 0 done. Outputs in {config.RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
