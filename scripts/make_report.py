#!/usr/bin/env python
"""Assemble all phase outputs into results/RESULTS.md (missing pieces are
skipped with a note, so the report can be regenerated at any stage)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from fatevae import config


def _try_read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = ".3f") -> str:
    """Minimal markdown table (no tabulate dependency - HPC offline)."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    if index:
        d.insert(0, df.index.name or "", [str(i) for i in df.index])
    cols = list(d.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _fmt(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: float = 3) -> str:
    cols = [c for c in (cols or df.columns) if c in df.columns]
    return _md_table(df[cols], floatfmt=f".{floatfmt}f")


def main() -> None:
    lines = ["# FateVAE - results report (auto-generated)", ""]

    # ---- Phase 0 -------------------------------------------------------
    vd = _try_read(config.TABLES_DIR / "phase0_variance_decomposition.csv")
    bench0 = _try_read(config.TABLES_DIR / "phase0_benchmark.csv")
    lines += ["## Phase 0 - cryptic fate heterogeneity quantification", ""]
    if vd is not None:
        lines += ["**H1 test - between-fate variance fraction** "
                  "(fate-signal dilution):", "",
                  _fmt(vd), ""]
        lines += ["> HVG-PCA carries ~1% fate-related variance while the "
                  "marker subspace carries >20%: the fate signal exists but "
                  "is diluted in variance-dominated embeddings.", ""]
    if bench0 is not None:
        lines += ["**Embedding benchmark** (marker-gene recovery AUROC, "
                  "silhouette on marker-defined fate labels, CHI):", "",
                  _fmt(bench0, ["method", "dim", "heldout_marker_auroc",
                                "heldout_marker_auroc_sd", "silhouette", "CHI"]),
                  ""]
    for fig in ["phase0_umap_phenomenon.png", "phase0_regulon_umap.png",
                "phase0_marker_auroc.png"]:
        p = config.FIGURES_DIR / fig
        if p.exists():
            lines += [f"![{fig}](figures/{fig})", ""]

    # ---- Phase 1 -------------------------------------------------------
    for tag in ["run1", "smoke"]:
        b1 = _try_read(config.TABLES_DIR / f"phase1_benchmark_{tag}.csv")
        if b1 is not None:
            lines += [f"## Phase 1 - FateVAE training (`{tag}`)", "",
                      _fmt(b1, ["method", "dim", "heldout_marker_auroc",
                                "heldout_marker_auroc_sd", "silhouette", "CHI"]),
                      ""]
            anchor = _try_read(config.TABLES_DIR / f"phase1_adult_anchor_{tag}.csv",
                               )
            if anchor is not None:
                lines += ["**Adult-anchored cross-correlation** "
                          "(rows: adult neighbour fractions; columns: model "
                          "fate probabilities):", "",
                          _md_table(anchor, index=True), ""]
            for fig in [f"phase1_fate_umap_{tag}.png",
                        f"phase1_hvgumap_probs_{tag}.png"]:
                p = config.FIGURES_DIR / fig
                if p.exists():
                    lines += [f"![{fig}](figures/{fig})", ""]
            break

    logo = _try_read(config.TABLES_DIR / "phase1_logo_summary_logo.csv")
    if logo is not None:
        lines += ["## Phase 1 - LOGO anti-circularity check", "",
                  "> Supervision excludes the held-out marker genes; AUROC "
                  "measures whether the fate latent recovers them anyway.",
                  "", _md_table(logo, index=True), ""]

    # ---- Discovery mode (v2) ------------------------------------------------
    disc_tags = ["disc", "disc_k4", "discA", "ss_k4", "ss_k6", "coexpA",
                 "bothA"]
    disc_rows = []
    for tag in disc_tags:
        d = _try_read(config.TABLES_DIR / f"discovery_benchmark_{tag}.csv")
        if d is not None:
            disc_rows.append(d.assign(tag=tag))
    if disc_rows:
        dall = pd.concat(disc_rows, ignore_index=True)
        baselines = dall[dall.method != "DISCOVERY_FateVAE"].drop_duplicates(
            subset="method")
        variants = dall[dall.method == "DISCOVERY_FateVAE"][
            ["tag", "method", "ARI", "NMI", "marker_auroc"]]
        lines += ["## Discovery mode - prior-recovery test (v2)", "",
                  "> No experimental marker genes were used for gene selection,",
                  "> model structure, or training; markers enter only this",
                  "> evaluation. G1' gate: ARI >= 0.3 vs marker-defined",
                  "> compartments.", "",
                  "**Data-driven baselines**:", "",
                  _md_table(baselines[["method", "ARI", "NMI", "marker_auroc"]]),
                  "",
                  "**FateVAE discovery variants** (regulon source / training):",
                  "", _md_table(variants), "",
                  "References: informed mode marker AUROC 0.874, HVG-PCA30",
                  "0.805. **Gate G1' failed for every prior-free variant** -",
                  "diagnosis: cisTarget pruning removed 10/14 known fate TFs",
                  "from the regulon prior; co-expression regulons (226, TF",
                  "coverage restored) improved AUROC to 0.743 but not cluster",
                  "recovery. Unsupervised objectives are dominated by the",
                  "~99% non-fate variance; partial supervision (LOGO 0.789,",
                  "informed 0.874) is what lifts recovery.", ""]
        for fig in [f"discovery_attention_{t}.png" for t in disc_tags]:
            p = config.FIGURES_DIR / fig
            if p.exists():
                lines += [f"![{fig}](figures/{fig})", ""]
                break

    # ---- Discovery mode v2.1: Seed & Amplify (v0.3) -------------------------
    sa_tags = ["sa", "sa_hvg", "sa5", "sa5_s1"]
    sa_rows = []
    for tag in sa_tags:
        m = _try_read(config.TABLES_DIR / f"discovery2_metrics_{tag}.csv")
        if m is not None:
            sa_rows.append(m.assign(tag=tag).pivot(index="tag", columns="metric",
                                                   values="value").reset_index())
    if sa_rows:
        sal = pd.concat(sa_rows, ignore_index=True)
        cols = [c for c in ["tag", "ARI", "NMI", "marker_auroc",
                            "marker_auroc_sd"] if c in sal.columns]
        lines += ["## Discovery mode v2.1 - Seed & Amplify (v0.3, self-supervised)", "",
                  "> Stage 1: anticoherent TF detection modules (statistical seed,",
                  "> no markers). Stage 2: EM loop - informed-mode VAE on seed",
                  "> pseudo-modules -> genome-wide nomination -> damped module",
                  "> update. Stage 3: prior-recovery test (markers enter here only).",
                  "",
                  _md_table(sal[cols]), "",
                  "Best prior-free result so far: marker AUROC up to 0.775",
                  "(beats NMF/Leiden baselines 0.64-0.73, approaches HVG-PCA",
                  "0.805); amplified dim-fate |r| up to 0.50 (seed ~0.3, all",
                  "earlier variants <=0.3). Partition recovery (ARI) still fails",
                  "(gate 0.3). Diagnosis: the EM converges preferentially to the",
                  " strongest anticoherent program in the data - the ecdysone/",
                  "temporal axis (br, Eip75B, lola, Tet, chinmo, ftz-f1 are",
                  "nominated de novo) - while the spatial compartment code stays",
                  "below the unsupervised detection threshold in scRNA alone.", ""]
        for fig in ["discovery2_umap_sa5.png"]:
            p = config.FIGURES_DIR / fig
            if p.exists():
                lines += [f"![{fig}](figures/{fig})", ""]

    # ---- Phase 3: OT demos ---------------------------------------------------
    ot_grid = _try_read(config.TABLES_DIR / "phase3_ot_grid_ot1.csv")
    if ot_grid is not None:
        lines += ["## Phase 3 - OT demos (v0.4)", "",
                  "**Demo 1: larva -> adult unbalanced OT** (merged PCA space).",
                  "(eps, tau) sensitivity of the plan mass and the best "
                  "|Spearman| between OT fate mass and disc compartment "
                  "scores; kNN-30 baseline for reference (v0.1: max |r|~0.21):",
                  "", _md_table(ot_grid), ""]
        cc = _try_read(config.TABLES_DIR / "phase3_ot_crosscorr_ot1.csv",
                       )
        if cc is not None:
            lines += ["Best-config cross-correlation:", "",
                      _md_table(cc.set_index(cc.columns[0]), index=True), ""]
        for fig in ["phase3_ot_umap_ot1.png", "phase3_ot_vs_knn_ot1.png"]:
            p = config.FIGURES_DIR / fig
            if p.exists():
                lines += [f"![{fig}](figures/{fig})", ""]
    for wtag in ["wing2", "wing"]:
        wing = _try_read(config.TABLES_DIR
                         / f"phase3_wing_continuity_{wtag}.csv")
        if wing is not None:
            lines += [f"**Demo 2: wing disc 96h -> 120h temporal OT** "
                      f"(Everetts et al. 2021 archive, tag {wtag}): "
                      "fate-continuity matrix "
                      "(rows: 96h compartment top-quartile cells; cols: landed "
                      "mass fraction in 120h compartments):", "",
                      _md_table(wing.set_index(wing.columns[0]), index=True), ""]
            for fig in [f"phase3_wing_ot_{wtag}.png",
                        f"phase3_wing_continuity_{wtag}.png"]:
                p = config.FIGURES_DIR / fig
                if p.exists():
                    lines += [f"![{fig}](figures/{fig})", ""]
            break

    # ---- Phase 4: Graph VAE (Guidance.md 3.4) --------------------------------
    g_rows = []
    for tag in ["ginf", "ginf_full", "ginf_k30", "ginf_x2", "ginf_x4",
                "ginf_x8", "ginf_x16", "ginf_x32", "ginf_x64",
                "gdisc_full", "gdisc_reg", "gdisc_x4"]:
        m = _try_read(config.TABLES_DIR / f"phase4_metrics_{tag}.csv")
        if m is not None:
            g_rows.append(m)
    if g_rows:
        g = pd.concat(g_rows, ignore_index=True)
        lines += ["## Phase 4 - Graph VAE (disc-data-only; Guidance 3.4)", "",
                  "> GCN encoder (2-layer, sym-normalized kNN graph) + the",
                  "two-branch NB decoder. Same prior-recovery contract.",
                  "refs: informed-MLP 0.874, HVG-PCA 0.805, best prior-free",
                  "0.775.", "", _md_table(g), "",
                  "Findings: graph aggregation lifts per-dim fate",
                  "correlations to |r| 0.4-0.6 (vs 0.1-0.3 for MLP",
                  "encoders) - the denoising mechanism works; longer",
                  "training keeps improving ARI monotonically (0.069 ->",
                  "0.094 -> 0.120 -> 0.158 -> 0.241 -> 0.278 at",
                  "x1/x2/x4/x8/x16/x32, gate 0.3 within reach) while",
                  "denser graphs (k=30) over-smooth; unsupervised graph",
                  "mode degrades with longer training.", ""]
        for fig in ["phase4_graphvae_ginf_x32.png",
                    "phase4_graphvae_ginf_x16.png"]:
            p = config.FIGURES_DIR / fig
            if p.exists():
                lines += [f"![{fig}](figures/{fig})", ""]

    # ---- Phase 2 -------------------------------------------------------
    attr = _try_read(config.TABLES_DIR / "phase2_tf_attribution_run1.csv")
    if attr is not None:
        lines += ["## Phase 2 - TF -> fate attribution (v0.1 linear probe)",
                  "", _fmt(attr), ""]
        for fig in ["phase2_attr_heatmap_run1.png"]:
            p = config.FIGURES_DIR / fig
            if p.exists():
                lines += [f"![{fig}](figures/{fig})", ""]

    (config.RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines))
    print(f"Wrote {config.RESULTS_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
