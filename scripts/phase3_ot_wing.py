#!/usr/bin/env python
"""OT demo 2: temporal coupling of the wing disc (96h -> 120h AEL) from the
archived Everetts et al. 2021 data (GSE155543) - the Waddington-OT setting.

Questions:
  (a) fate continuity: where do 96h cells of each spatial compartment
      (pouch / notum / hinge / peripodial, by marker module scores) send
      their mass at 120h?
  (b) growth: which 120h cells receive above-uniform mass (proliferation
      signal)?

Usage: python scripts/phase3_ot_wing.py [--tag wing] [--eps 0.05] [--tau 10]
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

from fatevae import config
from fatevae.ot import sinkhorn_unbalanced
from fatevae.ot.sinkhorn import row_normalized

WING_DIR = ROOT / "data_archive" / "01_wing_disc_Everetts_2021" / "data"
FBGN_MAP: dict[str, str] | None = None  # built lazily in main()

# wing-disc spatial compartment markers (Everetts et al. 2021 and classic
# disc biology); used as EVALUATION-ONLY structure. omb and lms omitted -
# no FBgn mapping available from our in-house annotation subset.
WING_MODULES = {
    "pouch": ["vg", "nub"],
    "notum": ["tsh", "Doc2", "Doc3"],
    "hinge_distal": ["Dll", "kn", "crm"],
    "peripodial": ["Pdp1", "orb"],
    "myoblast": ["twi", "Mef2"],
}


def fbgn_to_symbol_map() -> dict[str, str]:
    """Map FlyBase accessions to gene symbols using the in-house Dmel
    annotation (var['gene_ids'] -> var_names). Unmapped genes keep FBgn."""
    import anndata as ad

    ref = ad.read_h5ad(config.DMEL_H5AD, backed="r")
    return dict(zip(ref.var["gene_ids"].astype(str), map(str, ref.var_names)))


def read_sample(stem: str) -> sc.AnnData:
    """Read one GSM's loose 10x mtx triplet."""
    with gzip.open(WING_DIR / f"{stem}_matrix.mtx.gz", "rb") as fh:
        M = scipy.io.mmread(fh).tocsr()
    genes = pd.read_csv(WING_DIR / f"{stem}_genes.tsv.gz", sep="\t",
                        header=None)
    barcodes = pd.read_csv(WING_DIR / f"{stem}_barcodes.tsv.gz", sep="\t",
                           header=None)[0].to_numpy()
    a = sc.AnnData(sp.csr_matrix(M.T),
                   obs=pd.DataFrame(index=pd.Index(barcodes.tolist())),
                   var=pd.DataFrame(index=pd.Index(
                       FBGN_MAP.get(g, g) for g in genes[1].astype(str))))
    a.var_names_make_unique()
    a.obs["sample"] = stem
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wing")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--resolve", action="store_true",
                    help="re-solve OT even if a cached plan exists")
    args = ap.parse_args()
    t0 = time.time()
    global FBGN_MAP
    FBGN_MAP = fbgn_to_symbol_map()

    samples = {
        "96h": ["GSM4705966_Dros96H1WingDisc", "GSM4705967_Dros96H2WingDisc"],
        "120h": ["GSM4705964_Dros120H1WingDisc", "GSM4705965_Dros120H2WingDisc"],
    }
    adatas = {tp: sc.concat([read_sample(s) for s in ss], join="inner")
              for tp, ss in samples.items()}
    for tp, a in adatas.items():
        sc.pp.filter_cells(a, min_genes=200)
        sc.pp.filter_genes(a, min_cells=10)
        print(f"{tp}: {a.shape}", flush=True)

    common = adatas["96h"].var_names.intersection(adatas["120h"].var_names)
    a96 = adatas["96h"][:, common].copy()
    a120 = adatas["120h"][:, common].copy()
    print(f"common genes: {len(common)}", flush=True)

    # normalize + pooled PCA
    for a in (a96, a120):
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3",
                                    layer="counts")
    hvg = (a96.var["highly_variable"] & a120.var["highly_variable"])
    hvg_names = common[hvg.to_numpy()]
    print(f"shared HVGs: {len(hvg_names)}", flush=True)
    pooled = sc.concat([a96[:, hvg_names].copy(), a120[:, hvg_names].copy()])
    sc.pp.scale(pooled, max_value=10)
    sc.tl.pca(pooled, n_comps=30, svd_solver="arpack")
    E = np.asarray(pooled.obsm["X_pca"], dtype=np.float64)
    E = (E - E.mean(0)) / (E.std(0) + 1e-9)
    n96 = a96.n_obs
    Es, Et = E[:n96], E[n96:]

    sq = (Es ** 2).sum(1)[:, None] + (Et ** 2).sum(1)[None, :] - 2 * Es @ Et.T
    np.maximum(sq, 0, out=sq)
    C = sq / np.median(sq)

    plan_path = config.MODELS_DIR / f"phase3_wing_plan_{args.tag}.npz"
    if plan_path.exists() and not args.resolve:
        z = np.load(plan_path, allow_pickle=True)
        R, growth = z["R"], z["growth"]
        print(f"reusing cached transport plan {plan_path}", flush=True)
    else:
        print(f"OT: {Es.shape[0]} x {Et.shape[0]}, eps={args.eps}, "
              f"tau={args.tau}", flush=True)
        sol = sinkhorn_unbalanced(C, eps=args.eps, tau=args.tau, n_iter=2000)
        R = row_normalized(sol["T"])
        growth = sol["col_mass"] * Et.shape[0]  # >1: receives extra mass
        np.savez_compressed(plan_path, R=R, growth=growth,
                            obs96=a96.obs_names.astype(str),
                            obs120=a120.obs_names.astype(str))

    # ---- (a) fate continuity by compartment module ------------------------
    from fatevae.priors.markers import module_scores

    s96 = module_scores(a96.X, a96.var_names, WING_MODULES)
    s120 = module_scores(a120.X, a120.var_names, WING_MODULES)
    miss = {m: [g for g in gs if g not in set(a96.var_names)]
            for m, gs in WING_MODULES.items()}
    miss = {m: g for m, g in miss.items() if g}
    if miss:
        print(f"missing markers: {miss}", flush=True)

    cont = pd.DataFrame(np.nan, index=list(WING_MODULES),
                        columns=list(WING_MODULES))
    for m in WING_MODULES:
        hi = s96[m].to_numpy() > np.nanquantile(s96[m], 0.75)
        if hi.sum() < 20:
            continue
        landed = R[hi].sum(0)  # mass landing profile of top-quartile cells
        for m2 in WING_MODULES:
            lhi = s120[m2].to_numpy() > np.nanquantile(s120[m2], 0.75)
            cont.loc[m, m2] = landed[lhi].sum() / max(landed.sum(), 1e-9)
    print("\nFate-continuity matrix (rows: 96h compartment top-quartile; "
          "cols: fraction of landed mass in 120h compartment top-quartile):",
          flush=True)
    print(cont.round(3).to_string(), flush=True)
    cont.to_csv(config.TABLES_DIR / f"phase3_wing_continuity_{args.tag}.csv")

    # ---- figures ------------------------------------------------------------
    tmp = sc.AnnData(E)
    sc.pp.neighbors(tmp, n_neighbors=15, random_state=0)
    sc.tl.umap(tmp, random_state=0)
    U = np.asarray(tmp.obsm["X_umap"])
    stage = np.array(["96h"] * n96 + ["120h"] * (E.shape[0] - n96))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), dpi=170)
    for ax, lab, col in zip(axes[:2], ("96h", "120h"),
                            (np.arange(n96), np.arange(n96, E.shape[0]))):
        ax.scatter(U[col, 0], U[col, 1], s=2, alpha=0.6, linewidths=0,
                   c="tab:blue" if lab == "96h" else "tab:orange")
        ax.set_title(lab); ax.set_xticks([]); ax.set_yticks([])
    sctr = axes[2].scatter(U[n96:, 0], U[n96:, 1], c=np.log2(growth + 1e-6),
                           cmap="coolwarm", s=2, alpha=0.7, linewidths=0)
    axes[2].set_title("120h log2 growth (OT mass)"); axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(sctr, ax=axes[2], shrink=0.75)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"phase3_wing_ot_{args.tag}.png",
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=170)
    im = ax.imshow(cont.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cont.columns)))
    ax.set_xticklabels(cont.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cont.index)))
    ax.set_yticklabels(list(cont.index), fontsize=8)
    for i in range(cont.shape[0]):
        for j in range(cont.shape[1]):
            if not np.isnan(cont.iloc[i, j]):
                ax.text(j, i, f"{cont.iloc[i, j]:.2f}", ha="center",
                        va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("fate continuity 96h -> 120h")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"phase3_wing_continuity_{args.tag}.png",
                bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"growth": growth}, index=a120.obs_names).to_csv(
        config.TABLES_DIR / f"phase3_wing_growth_{args.tag}.csv")
    print(f"\nDemo 2 done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
