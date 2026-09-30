#!/usr/bin/env python
"""Phase 2 (proposal §6.3, v0.1): fate-aware TF attribution + perturbation.

Reads the FateVAE fate probabilities from a Phase-1 run and the SCENIC AUC
matrix; outputs:
  tables/phase2_tf_attribution_<tag>.csv   - TF x fate signed coefficients
  tables/phase2_regulators_<fate>.csv      - per-fate ranked regulators
  tables/phase2_perturbation_<tag>.csv     - mean fate-probability shifts
  figures/phase2_attr_heatmap_<tag>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fatevae import config
from fatevae.grn import in_silico_perturb, rank_regulators, tf_fate_attribution
from fatevae.io import load_scenic

# known compartment regulators for a quick recall check (literature/our priors)
KNOWN_REGULATORS = {
    "A8": ["abd-A", "tsh"],
    "A9p1": ["Abd-B", "btl", "dpp", "hh", "Dr"],
    "A9p2": ["en", "dac", "Delta", "bnl", "hh"],
    "A10": ["cad", "Dll"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1")
    args = ap.parse_args()

    probs = pd.read_csv(
        config.TABLES_DIR / f"phase1_fate_probs_{args.tag}.csv", index_col=0
    ).drop(columns=["entropy"], errors="ignore")
    from fatevae.io import load_disc

    adata = load_disc(config.DMEL_H5AD, config.DMEL_RAW_H5AD)
    auc, _ = load_scenic(config.DMEL_SCENIC_LOOM, adata.obs_names)

    common = probs.index.intersection(auc.index)
    probs, auc = probs.loc[common], auc.loc[common]

    attr = tf_fate_attribution(auc, probs)
    attr.to_csv(config.TABLES_DIR / f"phase2_tf_attribution_{args.tag}.csv")

    for fate in probs.columns:
        r = rank_regulators(attr, fate)
        r.to_csv(config.TABLES_DIR / f"phase2_regulators_{fate}_{args.tag}.csv")
        known = [tf for tf in KNOWN_REGULATORS.get(fate, []) if tf in r.index]
        if known:
            ranks = [(tf, int(r.index.get_loc(tf)) + 1) for tf in known]
            print(f"{fate}: known regulator ranks {ranks}", flush=True)

    # heatmap of top regulators per fate
    top = pd.concat(
        [rank_regulators(attr, f).head(15) for f in probs.columns], axis=0
    ).index.unique()
    fig, ax = plt.subplots(figsize=(5.5, 7), dpi=170)
    im = ax.imshow(attr.loc[top].to_numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(attr.shape[1]))
    ax.set_xticklabels(attr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.6, label="log-odds / SD of regulon activity")
    ax.set_title("TF -> fate attribution (FateVAE + SCENIC probe)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"phase2_attr_heatmap_{args.tag}.png",
                bbox_inches="tight")
    plt.close(fig)

    # in-silico perturbation of the marker TFs (sanity: pushing a marker TF
    # toward high activity should raise its own fate's probability)
    rows = []
    for tf in ["abd-A", "tsh", "Abd-B", "en", "Dll", "cad"]:
        if tf in auc.columns:
            d = in_silico_perturb(auc, probs, tf, quantile=0.9)
            rows.append(d.assign(TF=tf))
    pert = pd.concat(rows)
    pert.to_csv(config.TABLES_DIR / f"phase2_perturbation_{args.tag}.csv")
    print("\nIn-silico perturbation (mean fate-probability delta, q90 activity):")
    print(pert.reset_index().to_string(index=False), flush=True)

    print(f"\nPhase 2 done. Outputs in {config.TABLES_DIR}", flush=True)


if __name__ == "__main__":
    main()
