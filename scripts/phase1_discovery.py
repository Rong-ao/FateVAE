#!/usr/bin/env python
"""Phase 1 - Discovery mode (proposal v2 Aim 2/3): prior-independent fate
program discovery + the prior-recovery test on Dmel.

STRICT marker-free protocol: no marker gene or module is used for gene
selection, model structure, or training. pySCENIC regulons (co-expression +
motif pruning - data-driven) are the only fate-relevant structure. Known
markers enter ONLY the final evaluation.

Usage:
  python scripts/phase1_discovery.py [--k 8] [--epochs 300] [--tag disc]
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score

from fatevae import config
from fatevae.baselines import hvg_pca, regulon_pca
from fatevae.eval import embedding_gene_auroc, embedding_silhouette, fate_entropy
from fatevae.io import load_disc, load_scenic
from fatevae.model import (DiscoveryConfig, active_dims, fate_probabilities,
                           train_discovery)
from fatevae.priors import FATE_MODULES, module_scores
from fatevae.viz import multi_scatter

torch.set_num_threads(max(1, config.n_cpus()))

DATA_ROOT_TF_LIST = str(config.DATA_ROOT / "Dmel_base" / "allTFs_dmel.txt")

KNOWN_TFS = {f: [g for g in gs] for f, gs in FATE_MODULES.items()}
ALL_MARKERS = sorted({g for gs in FATE_MODULES.values() for g in gs})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tag", default="disc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="discovery",
                    choices=["discovery", "selfsup"],
                    help="discovery: pure unsupervised regulon-attention VAE; "
                         "selfsup: NMF-on-AUC pseudo-labels (data-driven) "
                         "training the informed architecture")
    ap.add_argument("--regulon-source", default="scenic",
                    choices=["scenic", "coexp", "both"],
                    help="source of the data-driven regulon prior: pySCENIC "
                         "cisTarget-pruned regulons, de-novo co-expression "
                         "regulons (genome TF list, no motif pruning), or both")
    # capacity-asymmetry variant knobs (proposal v2 G1' fallback 2)
    ap.add_argument("--n-bg", type=int, default=16)
    ap.add_argument("--beta-b", type=float, default=5e-4)
    ap.add_argument("--lambda-l1", type=float, default=2e-4)
    ap.add_argument("--lambda-ent", type=float, default=5e-3)
    args = ap.parse_args()
    t0 = time.time()

    adata = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)
    auc, regulons = load_scenic(config.DMEL_SCENIC_LOOM, adata.obs_names)

    if args.regulon_source in ("coexp", "both"):
        from fatevae.priors import coexpression_regulons

        tf_list = [l.strip() for l in
                   open(DATA_ROOT_TF_LIST).read().splitlines() if l.strip()]
        coexp = coexpression_regulons(adata.X, adata.var_names, tf_list)
        if args.regulon_source == "coexp":
            regulons = coexp
        else:  # union of target lists per TF
            merged = {tf: sorted(set(regulons.get(tf, [])) | set(gs))
                      for tf, gs in coexp.items()}
            for tf, gs in regulons.items():
                merged.setdefault(tf, gs)
            regulons = merged
        print(f"[regulon-source={args.regulon_source}] using {len(regulons)} "
              "regulons", flush=True)

    # ---- marker-free gene selection: HVGs + regulon target genes --------
    reg_targets = {g for gs in regulons.values() for g in gs}
    hvg = set(adata.var_names[adata.var["highly_variable"]])
    keep = sorted((hvg | reg_targets) & set(adata.var_names))
    sub = adata[:, keep].copy()
    print(f"[marker-free] gene set: {len(keep)} (HVG {len(hvg)}, regulon "
          f"targets {len(reg_targets)}); markers NOT used for selection",
          flush=True)

    # regulon membership matrix R (M x G) over the kept genes
    gidx = pd.Index(sub.var_names)
    R = np.zeros((len(regulons), len(keep)), dtype=np.float32)
    for m, (tf, genes) in enumerate(regulons.items()):
        cols = gidx.get_indexer([g for g in genes if g in gidx])
        R[m, cols] = 1.0
    coverage = (R.sum(0) > 0).mean()
    print(f"R: {R.shape[0]} regulons x {R.shape[1]} genes; regulon-covered "
          f"genes: {coverage:.1%}", flush=True)

    # ---- data-driven nuisance weights (QC covariates, no curation) -------
    lib = np.asarray(sub.layers["counts"].sum(1)).ravel()
    pct_mt = adata.obs["pct_counts_mt"].to_numpy()
    qc_w = np.zeros(len(regulons))
    for m, tf in enumerate(regulons):
        if tf not in auc.columns:
            continue  # co-expression-only TF: no AUC series
        a = auc[tf].to_numpy()
        c1 = abs(spearmanr(a, np.log(lib + 1)).statistic)
        c2 = abs(spearmanr(a, pct_mt).statistic)
        qc_w[m] = max(c1, c2)
    print(f"nuisance |corr|: median {np.median(qc_w):.2f}, "
          f"{(qc_w > 0.5).sum()} regulons > 0.5", flush=True)

    X = np.asarray(sub.X.todense(), dtype=np.float32)
    C = np.asarray(sub.layers["counts"].todense(), dtype=np.float32)

    if args.mode == "discovery":
        cfg = DiscoveryConfig(n_fate=args.k, epochs=args.epochs, seed=args.seed,
                              n_bg=args.n_bg, beta_b=args.beta_b,
                              lambda_l1=args.lambda_l1, lambda_ent=args.lambda_ent)
        model, hist = train_discovery(X, C, lib, torch.tensor(R), qc_weight=qc_w,
                                      cfg=cfg)
        torch.save({"state_dict": model.state_dict(), "genes": keep,
                    "regulons": list(regulons), "R": R, "qc_weight": qc_w,
                    "cfg": {k: v for k, v in vars(cfg).items() if k != "history"}},
                   config.MODELS_DIR / f"discovery_{args.tag}.pt")
        pd.DataFrame(hist).to_csv(
            config.TABLES_DIR / f"discovery_history_{args.tag}.csv", index=False)
        zf, zb = model.latent(torch.tensor(X), torch.tensor(lib))
        dims = active_dims(model, zf)
        print(f"active dims: {dims.tolist()} of {args.k}", flush=True)
    else:
        # ---- self-supervised: NMF on AUC (data-driven) pseudo-labels ------
        from fatevae.model import TrainConfig, build_fate_mask, train_fatevae

        nmf = NMF(n_components=args.k, init="nndsvda", random_state=0,
                  max_iter=800)
        W = nmf.fit_transform(auc.to_numpy(float))       # cells x k
        H = nmf.components_                              # k x regulons
        W = W / (W.sum(axis=1, keepdims=True) + 1e-9)
        comp_names = [f"P{c}" for c in range(args.k)]
        # data-driven fate gene sets: targets of each component's top regulons
        reg_list = list(regulons)
        pseudo_sets = {}
        for c in range(args.k):
            top = np.argsort(-H[c])[:8]
            genes: list[str] = []
            for m in top:
                genes.extend(regulons[reg_list[m]][:60])
            pseudo_sets[comp_names[c]] = sorted(set(genes) & set(keep))
        print("[selfsup] NMF pseudo-labels; per-component gene-set sizes "
              f"{ {n: len(s) for n, s in pseudo_sets.items()} }", flush=True)
        mask = build_fate_mask(keep, pseudo_sets)
        cfg = TrainConfig(epochs_pretrain=100, epochs_sup=120, epochs_adv=80,
                          batch_size=256, n_bg=args.n_bg, seed=args.seed,
                          hidden=(256, 64))
        tmodel, hist = train_fatevae(X, C, lib, mask, W, 
                                     np.ones(len(X), bool),
                                     np.zeros(len(X), bool), cfg)
        torch.save({"state_dict": tmodel.state_dict(), "genes": keep,
                    "pseudo_sets": pseudo_sets,
                    "cfg": {k: v for k, v in vars(cfg).items() if k != "history"}},
                   config.MODELS_DIR / f"discovery_{args.tag}.pt")
        pd.DataFrame(hist).to_csv(
            config.TABLES_DIR / f"discovery_history_{args.tag}.csv", index=False)
        zf, _, _ = tmodel.latent(torch.tensor(X), torch.tensor(lib))
        dims = np.arange(args.k)
        print(f"[selfsup] using all {args.k} dims", flush=True)
        model = None  # no attention to report; nominees come from mask weights
    probs = fate_probabilities(zf, dims)
    pred = probs.argmax(1)

    # ---- prior-recovery evaluation ---------------------------------------
    scores = module_scores(adata.X, adata.var_names, FATE_MODULES)
    known = scores.idxmax(1).to_numpy()
    fate_names = list(FATE_MODULES)

    # Hungarian mapping discovered dims -> known fates by Spearman corr
    corr = np.zeros((len(dims), len(fate_names)))
    for i, d in enumerate(dims):
        for j, f in enumerate(fate_names):
            corr[i, j] = spearmanr(zf[:, d], scores[f]).statistic
    ri, ci = linear_sum_assignment(-corr)
    mapping = {int(dims[i]): fate_names[j] for i, j in zip(ri, ci)}
    print("\ndim->fate mapping (Spearman):")
    for i, j in zip(ri, ci):
        print(f"  dim {dims[i]} -> {fate_names[j]}: r={corr[i, j]:.3f}",
              flush=True)

    ari = adjusted_rand_score(known, pred)
    nmi = normalized_mutual_info_score(known, pred)
    print(f"\nARI={ari:.3f}  NMI={nmi:.3f} (discovered {len(dims)} vs 4 known)",
          flush=True)

    # marker-gene recovery AUROC (embedding metric, comparable to v0.1)
    heldout = {g: np.asarray(adata.X[:, adata.var_names.get_loc(g)].todense()).ravel()
               for g in ALL_MARKERS if g in adata.var_names}
    emb = zf[:, dims]
    aucs = {g: embedding_gene_auroc(emb, e) for g, e in heldout.items()}
    val = [v for v in aucs.values() if not np.isnan(v)]
    print(f"\nmarker AUROC (discovery z_f): mean {np.mean(val):.3f} "
          f"(sd {np.std(val):.3f})   [v0.1 informed: 0.874, HVG-PCA: 0.805]",
          flush=True)

    # nominated regulons per active dim + known-TF recovery
    reg_names = np.array(list(regulons))
    if model is not None:  # discovery mode
        attn = model.attention().detach().numpy()
        loader = model.fate_gene_loader().detach().numpy()
    else:  # selfsup mode: NMF regulon loadings + masked decoder weights
        attn = H  # (k, M)
        loader = tmodel.dec_fate.weight.detach().numpy().T  # (K, G)
    nom_rows = []
    recall_rows = []
    for i, d in enumerate(dims):
        top_reg = reg_names[np.argsort(-attn[d])[:10]]
        f = mapping.get(int(d), "unassigned")
        hit = [tf for tf in top_reg if tf in KNOWN_TFS.get(f, [])]
        top_genes = [keep[j] for j in np.argsort(-loader[d])[:15]]
        gene_hit = [g for g in top_genes if g in ALL_MARKERS]
        nom_rows.append({"dim": int(d), "mapped_fate": f,
                         "corr": float(corr[i, fate_names.index(f)]) if f in fate_names else np.nan,
                         "top_regulons": ", ".join(top_reg),
                         "known_TF_hits": ", ".join(hit),
                         "top_genes": ", ".join(top_genes),
                         "known_marker_gene_hits": ", ".join(gene_hit)})
        if f in fate_names:
            for tf in KNOWN_TFS.get(f, []):
                rank = int(np.where(reg_names == tf)[0][0]) if tf in reg_names else -1
                arank = int(np.argsort(-attn[d]).tolist().index(rank)) if rank >= 0 else 999
                recall_rows.append({"fate": f, "TF": tf, "in_regulons": rank >= 0,
                                    "attention_rank": arank})
    nom = pd.DataFrame(nom_rows)
    nom.to_csv(config.TABLES_DIR / f"discovery_nominees_{args.tag}.csv", index=False)
    print("\nnominated regulons per dim:")
    print(nom[["dim", "mapped_fate", "corr", "top_regulons",
               "known_TF_hits"]].to_string(index=False), flush=True)
    rec = pd.DataFrame(recall_rows)
    rec.to_csv(config.TABLES_DIR / f"discovery_tf_recall_{args.tag}.csv",
               index=False)
    known_in_regs = rec[rec.in_regulons]
    top10 = (known_in_regs.attention_rank < 10).sum()
    print(f"\nknown-TF recall: {top10}/{len(known_in_regs)} available known "
          f"TFs rank in top-10 attention of their fate dim "
          f"(of {len(rec)} total, {len(rec) - len(known_in_regs)} have no regulon)",
          flush=True)

    # ---- data-driven baselines: NMF on AUC, Leiden on regulon-PCA --------
    base_rows = []
    A = auc.to_numpy(float)
    for k in (4, 8):
        Wf = NMF(n_components=k, init="nndsvda", random_state=0,
                 max_iter=800).fit_transform(A)
        nmf_pred = Wf.argmax(1)
        base_rows.append({"method": f"NMF_AUC_k{k}", "ARI": adjusted_rand_score(known, nmf_pred),
                          "NMI": normalized_mutual_info_score(known, nmf_pred),
                          "marker_auroc": np.nanmean(
                              [embedding_gene_auroc(Wf, e) for e in heldout.values()])})
    reg10 = regulon_pca(auc, 10)
    tmp = sc.AnnData(reg10)
    sc.pp.neighbors(tmp, n_neighbors=15, random_state=0)
    sc.tl.leiden(tmp, resolution=0.5, key_added="leiden")
    base_rows.append({"method": "leiden_regulonPCA", "ARI": adjusted_rand_score(known, tmp.obs["leiden"]),
                      "NMI": normalized_mutual_info_score(known, tmp.obs["leiden"]),
                      "marker_auroc": np.nanmean(
                          [embedding_gene_auroc(reg10, e) for e in heldout.values()])})
    base_rows.append({"method": "DISCOVERY_FateVAE", "ARI": ari, "NMI": nmi,
                      "marker_auroc": float(np.mean(val))})
    bench = pd.DataFrame(base_rows)
    bench.to_csv(config.TABLES_DIR / f"discovery_benchmark_{args.tag}.csv",
                 index=False)
    print("\nprior-recovery benchmark:")
    print(bench.round(3).to_string(index=False), flush=True)

    # ---- figures ----------------------------------------------------------
    tmpf = sc.AnnData(zf[:, dims])
    sc.pp.neighbors(tmpf, n_neighbors=15, random_state=0)
    sc.tl.umap(tmpf, random_state=0)
    zu = np.asarray(tmpf.obsm["X_umap"])
    panels = {f"dim{d}": zf[:, d] for d in dims}
    multi_scatter(zu, panels, config.FIGURES_DIR / f"discovery_umap_{args.tag}.png",
                  cmap="coolwarm", ncol=3,
                  panel_titles={f"dim{d}": f"dim {d} -> {mapping.get(d, 'unassigned')}" for d in dims})
    ent = fate_entropy(probs)
    multi_scatter(zu, {"fate_entropy": ent},
                  config.FIGURES_DIR / f"discovery_entropy_{args.tag}.png",
                  cmap="viridis", ncol=1)
    fig, ax = plt.subplots(figsize=(5, 8), dpi=170)
    im = ax.imshow(attn[dims], aspect="auto", cmap="magma")
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f"dim {d} -> {mapping.get(d, 'unassigned')}" for d in dims], fontsize=8)
    ax.set_xticks(range(len(reg_names)))
    ax.set_xticklabels(reg_names, rotation=90, fontsize=4)
    fig.colorbar(im, ax=ax, shrink=0.5, label="attention (softplus)")
    ax.set_title("fate-dim attention over regulons")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"discovery_attention_{args.tag}.png",
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nDiscovery done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
