#!/usr/bin/env python
"""Phase 4 (Guidance.md §3.4): Graph VAE - disc-data-only optimization.

No external dataset, no adult reference, no OT target. The only new
ingredient is the CELL-CELL GRAPH (kNN in expression space), letting a GCN
encoder aggregate neighbor information to denoise the weak fate signal.

Modes:
  graph_informed   fate dims masked to marker-extended gene sets + weak
                   supervision (upper reference; beats 0.874?)
  graph_discovery  marker-free: fate dims masked to statistical-seed
                   (detection-exclusivity) modules; NO supervision
                   (beats DiscoveryFateVAE AUROC 0.775 / ARI <=0.06?)

Evaluation: the unchanged prior-recovery contract. Gene importance per fate
dim via decoder gradients (Guidance §4.5).

Usage: python scripts/phase4_graphvae.py --mode graph_informed --tag ginf
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
import scanpy as sc
import torch

from fatevae import config
from fatevae.discovery import detection_seed
from fatevae.discovery.amplify import extend_sets
from fatevae.eval.recovery import prior_recovery
from fatevae.io import load_disc, load_scenic
from fatevae.model import (GraphTrainConfig, build_fate_mask,
                           build_knn_graph, train_graphvae)
from fatevae.priors import FATE_MODULES, coexpression_regulons, module_scores
from fatevae.priors.markers import extended_fate_genesets

TF_LIST = str(config.DATA_ROOT / "Dmel_base" / "allTFs_dmel.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="graph_informed",
                    choices=["graph_informed", "graph_discovery"])
    ap.add_argument("--tag", default="ginf")
    ap.add_argument("--graph-k", type=int, default=15)
    ap.add_argument("--graph-space", default="hvg", choices=["hvg", "regulon"])
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}", flush=True)

    adata = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)
    auc, regulons = load_scenic(config.DMEL_SCENIC_LOOM, adata.obs_names)
    hvg_mask = adata.var["highly_variable"].to_numpy()
    var_names = list(adata.var_names)

    tf_list = [l.strip() for l in open(TF_LIST).read().splitlines() if l.strip()]
    coexp = coexpression_regulons(adata.X, var_names, tf_list)

    # ---- fate masks per mode ------------------------------------------------
    if args.mode == "graph_informed":
        ext = extended_fate_genesets(regulons, FATE_MODULES, var_names)
        genes = sorted(set(np.asarray(var_names)[hvg_mask])
                       | {g for gs in ext.values() for g in gs})
        soft = module_scores(adata.X, var_names, FATE_MODULES)[
            list(FATE_MODULES)].to_numpy()
        supervised = True
    else:
        nuisance = {"n_genes": adata.obs["n_genes_by_counts"].to_numpy(float),
                    "pct_mt": adata.obs["pct_counts_mt"].to_numpy(float)}
        seed_modules, seed_scores, _ = detection_seed(
            adata.X, var_names, tf_list, nuisance=nuisance, seed=args.seed)
        Xd = np.asarray(adata.X.todense(), dtype=np.float32)
        mu, sd = Xd.mean(0), Xd.std(0); sd[sd == 0] = 1
        Zstd = ((Xd - mu) / sd).astype(np.float32)
        ext = extend_sets(seed_modules, Zstd, var_names, coexp)
        genes = sorted(set(np.asarray(var_names)[hvg_mask])
                       | {g for gs in ext.values() for g in gs})
        soft = seed_scores.to_numpy()
        supervised = False
    genes = [g for g in genes if g in set(var_names)]
    sub = adata[:, genes].copy()
    mask = build_fate_mask(genes, ext)
    print(f"[{args.mode}] {len(genes)} genes; fate sets "
          f"{ {k: len(v) for k, v in ext.items()} }; supervised={supervised}",
          flush=True)

    X = np.asarray(sub.X.todense(), dtype=np.float32)
    C = np.asarray(sub.layers["counts"].todense(), dtype=np.float32)
    lib = C.sum(1).astype(np.float32)

    # ---- graph ----------------------------------------------------------------
    if args.graph_space == "hvg":
        emb = np.asarray(adata.obsm["X_pca"])[:, :30]
    else:
        from fatevae.baselines import regulon_pca
        emb = regulon_pca(auc, 10)
    A = build_knn_graph(emb, k=args.graph_k)

    cfg = GraphTrainConfig(
        steps_pretrain=int(2000 * args.steps_scale),
        steps_sup=int(3000 * args.steps_scale),
        steps_adv=int(2000 * args.steps_scale),
        seed=args.seed, device=dev)
    model, hist = train_graphvae(X, C, lib, A, mask, soft, cfg,
                                 supervised=supervised)
    torch.save({"state_dict": model.state_dict(), "genes": genes,
                "mode": args.mode, "cfg": vars(cfg)},
               config.MODELS_DIR / f"graphvae_{args.tag}.pt")
    pd.DataFrame(hist).to_csv(
        config.TABLES_DIR / f"phase4_history_{args.tag}.csv", index=False)

    zf, zb, logits = model.latent(torch.tensor(X, device=dev),
                                  torch.tensor(lib, device=dev),
                                  A.to(dev))
    dims = np.arange(zf.shape[1])

    # ---- prior-recovery (same contract as every variant) ---------------------
    res = prior_recovery(zf, dims, adata.X, var_names)
    row = {"method": args.mode, "tag": args.tag,
           "ARI": res["ARI"], "NMI": res["NMI"],
           "marker_auroc": res["marker_auroc"]}
    pd.DataFrame([row]).to_csv(
        config.TABLES_DIR / f"phase4_metrics_{args.tag}.csv", index=False)
    print(f"\n[summary] {args.mode}: ARI={res['ARI']:.3f} "
          f"AUROC={res['marker_auroc']:.3f}  "
          "[refs: informed-MLP 0.874, HVG-PCA 0.805, DiscoveryFateVAE 0.775]",
          flush=True)

    # ---- gene importance via the masked decoder (Guidance §4.5) --------------
    # Wf (genes x K) is the structural importance (fate dim k may only write
    # to its gene set); the per-dim logit spread across cells is the
    # functional sensitivity of each fate dimension.
    Wf = (model.core.dec_fate.weight
          * model.core._fate_full_mask.T).detach().cpu().numpy()  # genes x K
    logit_spread = logits.std(axis=0)
    nom_rows = []
    for k, dim in enumerate(dims):
        top = np.argsort(-np.abs(Wf[:, k]))[:15]
        nom_rows.append({"dim": int(dim),
                         "fate": res["mapping"].get(int(dim), "?"),
                         "top_decoder_genes": ", ".join(genes[t] for t in top),
                         "logit_spread": float(logit_spread[k])})
    nom = pd.DataFrame(nom_rows)
    nom.to_csv(config.TABLES_DIR / f"phase4_gene_importance_{args.tag}.csv",
               index=False)
    print("\nper-dim top decoder genes:", flush=True)
    print(nom.to_string(index=False), flush=True)

    # ---- UMAP figure -----------------------------------------------------------
    tmp = sc.AnnData(zf)
    sc.pp.neighbors(tmp, n_neighbors=15, random_state=0)
    sc.tl.umap(tmp, random_state=0)
    zu = np.asarray(tmp.obsm["X_umap"])
    from fatevae.viz import multi_scatter
    titles = {f"dim{d}": f"dim {d} -> {res['mapping'].get(int(d), '?')}"
              for d in dims}
    multi_scatter(zu, {f"dim{d}": zf[:, d] for d in dims},
                  config.FIGURES_DIR / f"phase4_graphvae_{args.tag}.png",
                  cmap="coolwarm", ncol=3, panel_titles=titles)

    print(f"\nPhase 4 done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
