#!/usr/bin/env python
"""Phase 1 (proposal §6.2): train FateVAE on the Dmel genital disc data.

Usage:
  python scripts/phase1_train_fatevae.py [--tag run1] [--scale 1.0]
      [--ablate {none,nosup,noadv,nosupadv}] [--adult-anchor]

Outputs (results/):
  models/fatevae_<tag>.pt            - state dict
  tables/phase1_benchmark_<tag>.csv  - embedding benchmark incl. baselines
  tables/phase1_fate_probs_<tag>.csv - per-cell fate probabilities/entropy
  figures/phase1_fate_umap_<tag>.png - fate-aware UMAP colored by probs
  figures/phase1_entropy_umap_<tag>.png
  tables/phase1_adult_anchor_<tag>.csv (if --adult-anchor)
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
from fatevae.baselines import hvg_pca, regulon_pca
from fatevae.eval import (cell_splits, embedding_gene_auroc,
                          embedding_silhouette, fate_entropy, summarize_embedding)
from fatevae.fate import calibrate_temperature, fate_probabilities
from fatevae.io import load_disc, load_scenic, map_larva_to_adult_fates
from fatevae.model import FateVAE, TrainConfig, build_fate_mask, train_fatevae
from fatevae.priors import (FATE_MODULES, extended_fate_genesets,
                            module_scores)
from fatevae.viz import multi_scatter

torch.set_num_threads(max(1, config.n_cpus() - 0))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run1")
    p.add_argument("--scale", type=float, default=1.0,
                   help="epoch scale factor (quick smoke tests use 0.1)")
    p.add_argument("--ablate", default="none",
                   choices=["none", "nosup", "noadv", "nosupadv"])
    p.add_argument("--adult-anchor", action="store_true")
    p.add_argument("--hidden", type=int, nargs=2, default=[256, 64])
    p.add_argument("--n-bg", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    np.random.seed(config.SEED)

    print("== Phase 1: FateVAE training ==", flush=True)
    adata = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)
    auc, regulons = load_scenic(config.DMEL_SCENIC_LOOM, adata.obs_names)

    # ---- gene selection: HVGs + markers + extended fate gene sets --------
    ext_sets = extended_fate_genesets(regulons, FATE_MODULES, adata.var_names)
    hvg = set(adata.var_names[adata.var["highly_variable"]]) if "highly_variable" in adata.var \
        else set(pd.Series(adata.var_names).sample(3000, random_state=0))
    keep = sorted(hvg | {g for gs in ext_sets.values() for g in gs})
    keep = [g for g in keep if g in set(adata.var_names)]
    adata_sub = adata[:, keep].copy()
    print(f"gene set: {len(keep)} genes "
          f"(HVG {len(hvg)}, extended fate sets "
          f"{ {k: len(v) for k, v in ext_sets.items()} })", flush=True)

    X = np.asarray(adata_sub.X.todense(), dtype=np.float32)
    C = np.asarray(adata_sub.layers["counts"].todense(), dtype=np.float32)
    lib = C.sum(axis=1).astype(np.float32)

    fate_mask = build_fate_mask(list(adata_sub.var_names), ext_sets)
    scores_full = module_scores(adata_sub.X, adata_sub.var_names, FATE_MODULES)
    fate_label = scores_full.idxmax(axis=1).to_numpy()

    train_m, val_m, test_m = cell_splits(adata.obs["leiden_0.5"].to_numpy())

    cfg = TrainConfig(
        epochs_pretrain=int(100 * args.scale),
        epochs_sup=int(120 * args.scale),
        epochs_adv=int(80 * args.scale),
        batch_size=256,
        hidden=tuple(args.hidden),
        n_bg=args.n_bg,
        seed=config.SEED,
    )
    if "nosup" in args.ablate:
        cfg.lambda_sup = 0.0
    if "noadv" in args.ablate or "nosup" in args.ablate:
        cfg.lambda_adv = 0.0  # adversary meaningless without supervision target
    print(f"TrainConfig: pre={cfg.epochs_pretrain} sup={cfg.epochs_sup} "
          f"adv={cfg.epochs_adv} ablate={args.ablate}", flush=True)

    model, hist = train_fatevae(X, C, lib, fate_mask,
                                scores_full[list(FATE_MODULES)].to_numpy(),
                                train_m, val_m, cfg)
    torch.save({"state_dict": model.state_dict(),
                "genes": list(adata_sub.var_names),
                "ext_sets": ext_sets,
                "cfg": {k: v for k, v in vars(cfg).items() if k != "history"}},
               config.MODELS_DIR / f"fatevae_{args.tag}.pt")
    pd.DataFrame(hist).to_csv(config.TABLES_DIR / f"phase1_history_{args.tag}.csv",
                              index=False)

    # ---- latents, calibration, probabilities ------------------------------
    zf, zb, logits = model.latent(torch.tensor(X), torch.tensor(lib))
    Y = scores_full[list(FATE_MODULES)].to_numpy()
    T = calibrate_temperature(logits[val_m], _row_softmax(Y[val_m]))
    probs = fate_probabilities(logits, T)
    ent = fate_entropy(probs)
    print(f"calibration temperature: {T:.3f}; "
          f"entropy quartiles: {np.quantile(ent, [0.25, 0.5, 0.75]).round(3)}",
          flush=True)

    pd.DataFrame(probs, columns=list(FATE_MODULES),
                 index=adata.obs_names).assign(entropy=ent).to_csv(
        config.TABLES_DIR / f"phase1_fate_probs_{args.tag}.csv")

    # ---- benchmark against Phase-0 baselines -------------------------------
    hvg30 = hvg_pca(adata, 30)
    reg10 = regulon_pca(auc, 10)
    heldout = {g: np.asarray(adata.X[:, adata.var_names.get_loc(g)].todense()).ravel()
               for f, gs in FATE_MODULES.items() for g in gs if g in adata.var_names}
    rows = []
    for name, emb in [("hvg_pca30", hvg30), ("regulon_pca10", reg10),
                      ("fatevae_zf", zf), ("fatevae_zb", zb),
                      ("fatevae_z", np.hstack([zf, zb]))]:
        rows.append(summarize_embedding(name, emb, fate_label, heldout,
                                        variance_aware=hvg30))
    bench = pd.DataFrame(rows)
    bench.to_csv(config.TABLES_DIR / f"phase1_benchmark_{args.tag}.csv",
                 index=False)
    show = [c for c in ["method", "dim", "heldout_marker_auroc",
                        "heldout_marker_auroc_sd", "silhouette", "CHI"]
            if c in bench.columns]
    print("\nPhase-1 benchmark:")
    print(bench[show].to_string(index=False), flush=True)

    # ---- fate-aware UMAP figures -------------------------------------------
    tmp = sc.AnnData(zf)
    sc.pp.neighbors(tmp, n_neighbors=15, random_state=0)
    sc.tl.umap(tmp, random_state=0)
    zu = np.asarray(tmp.obsm["X_umap"])
    panels = {f"{f}_prob": probs[:, i] for i, f in enumerate(FATE_MODULES)}
    panels["entropy"] = ent
    multi_scatter(zu, panels, config.FIGURES_DIR / f"phase1_fate_umap_{args.tag}.png",
                  cmap="coolwarm", ncol=3,
                  panel_titles={**{f"{f}_prob": f"{f} fate probability" for f in FATE_MODULES},
                                "entropy": "fate entropy"})

    # variance-aware original UMAP colored by model output (the money plot)
    umap = np.asarray(adata.obsm["X_umap"])
    multi_scatter(umap, panels, config.FIGURES_DIR / f"phase1_hvgumap_probs_{args.tag}.png",
                  cmap="coolwarm", ncol=3,
                  panel_titles={**{f"{f}_prob": f"{f} fate probability" for f in FATE_MODULES},
                                "entropy": "fate entropy"})

    # ---- adult-anchored prospective validation ------------------------------
    if args.adult_anchor:
        anchor = map_larva_to_adult_fates()
        common = adata.obs_names.intersection(anchor.index)
        ai = anchor.loc[common]
        pi = pd.DataFrame(probs, index=adata.obs_names,
                          columns=list(FATE_MODULES)).loc[common]
        conf = pi.max(axis=1) > 0.6
        # cross-correlation: disc fate probability vs adult-neighbour fraction
        # (rows: disc fates; columns: adult terminal fates). Left to the
        # biologist to read off the compartment->organ correspondence.
        cc = pd.DataFrame(
            {f: [np.corrcoef(pi.loc[conf, f], ai.loc[conf, a])[0, 1]
                 for a in ai.columns] for f in pi.columns},
            index=ai.columns,
        )
        print(f"\nAdult-anchored cross-correlation (confident cells n={conf.sum()}):",
              flush=True)
        print(cc.round(3).to_string(), flush=True)
        cc.to_csv(config.TABLES_DIR / f"phase1_adult_anchor_{args.tag}.csv")
        ai.assign(**{f"{f}_prob": pi[f] for f in pi.columns}).to_csv(
            config.TABLES_DIR / f"phase1_adult_anchor_cells_{args.tag}.csv")

    print(f"\nPhase 1 done in {(time.time() - t0) / 60:.1f} min. "
          f"Outputs tagged '{args.tag}' in {config.RESULTS_DIR}", flush=True)


def _row_softmax(y, t=0.5):
    s = y / t
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
