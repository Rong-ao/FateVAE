#!/usr/bin/env python
"""FateSeed (v0.3): self-supervised fate discovery - "the principle is
hidden in the data, train the model with the data itself".

Stage 0  premise test  : do known fate TFs stand out by predictability /
                         bimodality / exclusivity? (honest go/no-go)
Stage 1  seed selection: compartment-like TF modules, fully data-computed
Stage 2  pseudo-fate training: informed-mode FateVAE on data-derived seeds
Stage 3  prior-recovery test: identical protocol to v0.2 (markers only
                         enter evaluation)

Usage: python scripts/phase1_fateseed.py [--n-seeds 4] [--tag seed1]
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
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from fatevae import config
from fatevae.discovery_seed import (bimodality_gap, select_fate_seeds,
                                    tf_predictability)
from fatevae.eval import embedding_gene_auroc, fate_entropy
from fatevae.io import load_disc
from fatevae.model import TrainConfig, build_fate_mask, train_fatevae
from fatevae.priors import FATE_MODULES, coexpression_regulons, module_scores
from fatevae.viz import multi_scatter

torch.set_num_threads(max(1, config.n_cpus()))

TF_LIST = str(config.DATA_ROOT / "Dmel_base" / "allTFs_dmel.txt")
KNOWN_MARKERS = sorted({g for gs in FATE_MODULES.values() for g in gs})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--tag", default="seed1")
    ap.add_argument("--epochs-scale", type=float, default=1.0)
    args = ap.parse_args()
    t0 = time.time()

    adata = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)
    tf_list = [l.strip() for l in open(TF_LIST).read().splitlines() if l.strip()]

    # ---------------- Stage 0: premise test -------------------------------
    print("== Stage 0: premise test (do fate TFs stand out?) ==", flush=True)
    r2 = tf_predictability(np.asarray(adata.X.todense(), dtype=np.float32),
                           adata.var_names, tf_list)
    r2.to_csv(config.TABLES_DIR / f"fateseed_tf_r2_{args.tag}.csv")
    known_r2 = r2[r2.index.isin(KNOWN_MARKERS)].sort_values(ascending=False)
    pct = r2.rank(pct=True)
    print("known fate TF predictability R^2 (percentile among "
          f"{len(r2)} TFs):", flush=True)
    for tf, v in known_r2.items():
        print(f"  {tf:8s} R2={v:+.3f}  pct={pct[tf]:.2f}", flush=True)
    print(f"mean percentile of known TFs: {pct[r2.index.isin(KNOWN_MARKERS)].mean():.2f}",
          flush=True)

    known_scores = module_scores(adata.X, adata.var_names, FATE_MODULES)
    print("\nknown marker-module bimodality (delta BIC, >10 = bimodal):")
    for f in FATE_MODULES:
        print(f"  {f:5s} deltaBIC={bimodality_gap(known_scores[f].to_numpy()):8.1f}",
              flush=True)
    cross = known_scores.corr(method="spearman")
    print("\ncross-compartment Spearman (exclusivity check):")
    print(cross.round(2).to_string(), flush=True)

    # ---------------- Stage 1: seed selection -----------------------------
    print("\n== Stage 1: data-driven seed selection ==", flush=True)
    coexp = coexpression_regulons(adata.X, adata.var_names, tf_list)
    seeds, cand = select_fate_seeds(adata.X, adata.var_names, coexp, r2,
                                    n_seeds=args.n_seeds)
    cand.to_csv(config.TABLES_DIR / f"fateseed_candidates_{args.tag}.csv",
                index=False)
    print(f"selected seed TFs: {list(seeds)}", flush=True)
    print("top-10 candidates by quality:")
    print(cand.head(10).to_string(index=False), flush=True)
    # where do known TFs sit among candidates?
    for tf in KNOWN_MARKERS:
        if tf in cand.TF.values:
            rk = int(cand.index[cand.TF == tf][0])
            print(f"  known TF {tf}: candidate rank {rk}, quality "
                  f"{cand.loc[cand.TF == tf, 'quality'].iloc[0]:.3f}", flush=True)
        else:
            print(f"  known TF {tf}: not a candidate (no/poor module)", flush=True)
    if not seeds:
        print("NO SEEDS SELECTED - premise failed at selection stage", flush=True)
        return

    # ---------------- Stage 2: pseudo-fate training ------------------------
    print("\n== Stage 2: informed-mode FateVAE on data-derived seeds ==",
          flush=True)
    pseudo_scores = module_scores(adata.X, adata.var_names, seeds)
    seed_names = list(seeds)
    hvg = set(adata.var_names[adata.var["highly_variable"]])
    keep = sorted((hvg | {g for gs in seeds.values() for g in gs})
                  & set(adata.var_names))
    sub = adata[:, keep].copy()
    X = np.asarray(sub.X.todense(), dtype=np.float32)
    C = np.asarray(sub.layers["counts"].todense(), dtype=np.float32)
    lib = C.sum(axis=1).astype(np.float32)
    mask = build_fate_mask(keep, seeds)
    cfg = TrainConfig(epochs_pretrain=int(100 * args.epochs_scale),
                      epochs_sup=int(120 * args.epochs_scale),
                      epochs_adv=int(80 * args.epochs_scale),
                      batch_size=256, seed=config.SEED)
    model, hist = train_fatevae(X, C, lib, mask,
                                pseudo_scores[seed_names].to_numpy(),
                                np.ones(len(X), bool), np.zeros(len(X), bool),
                                cfg)
    zf, _, logits = model.latent(torch.tensor(X), torch.tensor(lib))
    torch.save({"state_dict": model.state_dict(), "genes": keep,
                "seeds": seeds,
                "cfg": {k: v for k, v in vars(cfg).items() if k != "history"}},
               config.MODELS_DIR / f"fateseed_{args.tag}.pt")
    pd.DataFrame(hist).to_csv(
        config.TABLES_DIR / f"fateseed_history_{args.tag}.csv", index=False)

    # ---------------- Stage 3: prior-recovery test -------------------------
    print("\n== Stage 3: prior-recovery test ==", flush=True)
    known = known_scores.idxmax(1).to_numpy()
    fate_names = list(FATE_MODULES)
    dims = np.arange(len(seed_names))
    corr = np.array([[spearmanr(zf[:, d], known_scores[f]).statistic
                      for f in fate_names] for d in dims])
    ri, ci = linear_sum_assignment(-corr)
    mapping = {int(d): fate_names[j] for d, j in zip(ri, ci)}
    for d, j in zip(ri, ci):
        print(f"  seed dim {d} ({seed_names[d]}) -> {fate_names[j]}: "
              f"r={corr[d, j]:.3f}", flush=True)

    probs = np.exp((zf - zf.mean(0)) / (zf.std(0) + 1e-9))
    probs = probs[:, dims] / probs[:, dims].sum(1, keepdims=True)
    pred = probs.argmax(1)
    ari = adjusted_rand_score(known, pred)
    nmi = normalized_mutual_info_score(known, pred)
    print(f"\nARI={ari:.3f}  NMI={nmi:.3f} "
          f"(v0.2 best 0.032; G1' gate 0.3; informed-mode reference)", flush=True)

    heldout = {g: np.asarray(adata.X[:, adata.var_names.get_loc(g)].todense()).ravel()
               for g in KNOWN_MARKERS if g in adata.var_names}
    aucs = {g: embedding_gene_auroc(zf[:, dims], e) for g, e in heldout.items()}
    val = [v for v in aucs.values() if not np.isnan(v)]
    print(f"marker AUROC (FateSeed z_f): mean {np.mean(val):.3f} "
          f"(sd {np.std(val):.3f})  [informed 0.874, HVG-PCA 0.805, "
          f"v0.2 best 0.743]", flush=True)

    pd.DataFrame(probs, columns=[f"{seed_names[d]}" for d in dims],
                 index=adata.obs_names).assign(
        entropy=fate_entropy(probs),
        predicted=[seed_names[d] for d in pred],
        known=known).to_csv(
        config.TABLES_DIR / f"fateseed_probs_{args.tag}.csv")
    pd.DataFrame([{"method": "FateSeed", "ARI": ari, "NMI": nmi,
                   "marker_auroc": float(np.mean(val)),
                   "n_seeds": len(seeds)}]).to_csv(
        config.TABLES_DIR / f"fateseed_benchmark_{args.tag}.csv", index=False)

    # figures: seed scores and learned fate probs on the original UMAP
    umap = np.asarray(adata.obsm["X_umap"])
    panels = {f"seed_{s}": pseudo_scores[s].to_numpy() for s in seed_names}
    multi_scatter(umap, panels,
                  config.FIGURES_DIR / f"fateseed_scores_umap_{args.tag}.png",
                  cmap="coolwarm", ncol=3,
                  panel_titles={f"seed_{s}": f"seed {s} module score" for s in seed_names})
    panels2 = {f"dim{d}": probs[:, i] for i, d in enumerate(dims)}
    multi_scatter(umap, panels2,
                  config.FIGURES_DIR / f"fateseed_probs_umap_{args.tag}.png",
                  cmap="coolwarm", ncol=3,
                  panel_titles={f"dim{d}": f"{seed_names[d]} -> {mapping.get(d, '?')}" for d in dims})

    print(f"\nFateSeed done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
