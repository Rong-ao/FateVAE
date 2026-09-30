#!/usr/bin/env python
"""Discovery mode v2: Seed & Amplify (proposal v2 §9 'iterative marker
nomination', upgraded to a marker-FREE statistical seed).

Stage 1 (seed)  - anticoherent detection modules (fate code signature)
Stage 2 (amplify)- EM loop: informed-mode VAE on pseudo-modules ->
                  genome-wide nomination -> module update
Stage 3         - prior-recovery test (markers enter here only)

No experimental marker gene is used in stages 1-2. Data-driven inputs:
genome TF list (allTFs_dmel.txt), expression data, co-expression regulons.

Usage: python scripts/phase1_discovery2.py [--iters 3] [--tag sa]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from fatevae import config
from fatevae.discovery import amplify, detection_seed
from fatevae.eval.recovery import prior_recovery
from fatevae.io import load_disc, load_scenic
from fatevae.priors import coexpression_regulons, module_scores
from fatevae.priors.markers import FATE_MODULES
from fatevae.viz import multi_scatter

TF_LIST = str(config.DATA_ROOT / "Dmel_base" / "allTFs_dmel.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--tag", default="sa")
    ap.add_argument("--panel", default="tf", choices=["tf", "tf_hvg"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(max(1, config.n_cpus()))

    adata = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)
    var_names = list(adata.var_names)
    tf_list = [l.strip() for l in open(TF_LIST).read().splitlines() if l.strip()]

    # ---- Stage 1: statistical seed ----------------------------------------
    panel = list(dict.fromkeys(tf_list + ([] if args.panel == "tf" else
                                          list(adata.var_names[adata.var["highly_variable"]]))))
    nuisance = {
        "n_genes": adata.obs["n_genes_by_counts"].to_numpy(float),
        "pct_mt": adata.obs["pct_counts_mt"].to_numpy(float),
    }
    modules, scores, diag = detection_seed(adata.X, var_names, panel,
                                           nuisance=nuisance, seed=args.seed)
    print("\n[seed] sharpness by k:")
    print(diag.round(3).to_string(index=False), flush=True)
    print(f"\n[seed] {len(modules)} modules:")
    for m, gs in modules.items():
        print(f"  {m}({len(gs)}): {', '.join(gs[:14])}", flush=True)

    # seed-only recovery baseline (how good is the raw statistical seed?)
    seed_labels = scores.idxmax(1).to_numpy()
    known = module_scores(adata.X, var_names, FATE_MODULES).idxmax(1).to_numpy()
    from sklearn.metrics import adjusted_rand_score
    print(f"\n[seed] seed-only ARI vs known compartments: "
          f"{adjusted_rand_score(known, seed_labels):.3f}", flush=True)

    # ---- Stage 2: amplify ---------------------------------------------------
    X = np.asarray(adata.X.todense(), dtype=np.float32)
    C = np.asarray(adata.layers["counts"].todense(), dtype=np.float32)
    lib = C.sum(1).astype(np.float32)
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Zstd = ((X - mu) / sd).astype(np.float32)

    coexp = coexpression_regulons(adata.X, var_names, tf_list)
    hvg_mask = adata.var["highly_variable"].to_numpy()
    gene_set = sorted((set(np.asarray(var_names)[hvg_mask])
                       | {g for gs in coexp.values() for g in gs}) & set(var_names))
    sub = adata[:, gene_set].copy()
    Xs = np.asarray(sub.X.todense(), dtype=np.float32)
    Cs = np.asarray(sub.layers["counts"].todense(), dtype=np.float32)
    libs = Cs.sum(1).astype(np.float32)
    mus = Xs.mean(0); sds = Xs.std(0); sds[sds == 0] = 1
    Zstd_s = ((Xs - mus) / sds).astype(np.float32)

    model, zf, logits, history, stability = amplify(
        Xs, Cs, libs, modules, Zstd_s, list(sub.var_names), coexp,
        tf_panel=[t for t in tf_list if t in set(sub.var_names)],
        n_iter=args.iters, seed=args.seed)
    final_modules = history[-1]
    torch.save({"state_dict": model.state_dict(), "genes": gene_set,
                "modules_history": history, "stability": stability},
               config.MODELS_DIR / f"discovery2_{args.tag}.pt")

    # ---- Stage 3: prior-recovery test --------------------------------------
    dims = np.arange(zf.shape[1])
    res = prior_recovery(zf, dims, adata.X, var_names,
                         nominees={f"F{d}": final_modules.get(f"F{d}", [])
                                   for d in dims})
    rows = [{"metric": k, "value": v} for k, v in res.items()
            if isinstance(v, (int, float, np.floating))]
    pd.DataFrame(rows).to_csv(
        config.TABLES_DIR / f"discovery2_metrics_{args.tag}.csv", index=False)
    if "nominee_recall" in res:
        res["nominee_recall"].to_csv(
            config.TABLES_DIR / f"discovery2_nominees_{args.tag}.csv", index=False)

    # ---- figure: discovered dims on UMAP -----------------------------------
    tmp = sc.AnnData(zf)
    sc.pp.neighbors(tmp, n_neighbors=15, random_state=0)
    sc.tl.umap(tmp, random_state=0)
    zu = np.asarray(tmp.obsm["X_umap"])
    titles = {f"dim{d}": f"dim {d} -> {res['mapping'].get(d, '?')}" for d in dims}
    multi_scatter(zu, {f"dim{d}": zf[:, d] for d in dims},
                  config.FIGURES_DIR / f"discovery2_umap_{args.tag}.png",
                  cmap="coolwarm", ncol=3, panel_titles=titles)

    print(f"\n[done] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
