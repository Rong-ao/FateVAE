#!/usr/bin/env python
"""FateVAE v2 framework graphic (graphical redesign).

Glyph-based overview: input cards -> encoder/latent capsules -> the two
fate-decoder modes drawn as wiring diagrams -> NB likelihood strip -> output
glyphs -> the marker-only prior-recovery test, with the prior-free label
front-ends (Seed & Amplify EM loop, FateOT) feeding MODE A from below.
Vector output (PDF) + PNG. Regenerate: scripts/make_algorithm_figure.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Circle,
                                Polygon, Rectangle)
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fatevae import config

# ------------------------------------------------------------------ palette
FATE = {"A8": "#E4572E", "A9p1": "#F3A712", "A9p2": "#2A9D8F", "A10": "#5B4B8A"}
FCOL = list(FATE.values())
C_DATA = "#2e7d32"
C_EXP = "#6a1b9a"
C_CORE = "#1565c0"
C_INF = "#c62828"
C_BG = "#607d8b"
C_EVAL = "#b71c1c"
TXT = "#212121"
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(18, 10.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


# ------------------------------------------------------------------ helpers
def panel(x, y, w, h, title, ec="#9e9e9e", fc="#fafafa", lw=1.3, ls="-",
          title_c="#616161", fs=9.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.15,rounding_size=1.0",
                                ec=ec, fc=fc, lw=lw, linestyle=ls, zorder=1))
    ax.text(x + 1.6, y + h - 1.9, title, fontsize=fs, fontweight="bold",
            color=title_c, ha="left", va="top", zorder=4)


def card(x, y, w, h, ec, ls="-", fc="white", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.12,rounding_size=0.7",
                                ec=ec, fc=fc, lw=lw, linestyle=ls, zorder=2))


def cap(x, y, s, c="#616161", fs=6.6, ha="center", weight="normal"):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va="center",
            zorder=5, fontweight=weight)


def arrow(p0, p1, color=C_CORE, lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color,
                                 lw=lw, linestyle=ls, mutation_scale=14,
                                 shrinkA=1, shrinkB=1, zorder=3,
                                 connectionstyle=f"arc3,rad={rad}"))


def dots(x0, y0, n, color, r=0.42, dx=1.15, z=4):
    xs = x0 + np.arange(n) * dx
    ax.scatter(xs, [y0] * n, s=r * 26, color=color, zorder=z)


# ------------------------------------------------------------------ header
ax.text(50, 99.0, "FateVAE v2 — framework", ha="center", va="top",
        fontsize=17, fontweight="bold")
ax.text(50, 96.2, "two-mode two-branch NB-VAE · prior-free label front-ends · "
                  "markers used for evaluation only (Drosophila male genital disc)",
        ha="center", va="top", fontsize=10, color="#757575", style="italic")

# ================================================================== 1 input
panel(1, 32, 15, 59, "1 · input", ec="#9e9e9e", fc="#f7f7f7")

# -- markers card (dashed purple)
card(3, 77, 12, 9.2, C_EXP, ls=(0, (4, 2.4)), fc="#faf7fc")
for i, c in enumerate(FCOL):
    dots(4.6, 84.2 - i * 1.9, 3, c, r=0.38, dx=1.5)
cap(9, 78.2, "markers (optional)", C_EXP)

# -- counts card (matrix glyph)
card(3, 58.5, 12, 12.3, C_DATA)
M = rng.random((5, 6)) ** 2
for i in range(5):
    for j in range(6):
        v = M[i, j]
        ax.add_patch(Rectangle((4.4 + j * 1.55, 61.6 + i * 1.6), 1.35, 1.4,
                               fc=plt.cm.Blues(0.15 + 0.8 * v), ec="white",
                               lw=0.4, zorder=3))
cap(9, 60.1, "scRNA-seq counts", C_DATA)

# -- regulons card (TF -> targets graph glyph)
card(3, 39.5, 12, 16.5, C_DATA)
ax.add_patch(Circle((5.6, 50.6), 0.95, fc="#a5d6a7", ec=C_DATA, lw=1.2, zorder=3))
ax.text(5.6, 50.6, "TF", fontsize=5.6, ha="center", va="center",
        color=C_DATA, zorder=4, fontweight="bold")
targ = [(10.6, 53.2), (12.2, 51.2), (12.6, 48.6), (11.8, 46.0), (10.2, 44.2),
        (8.2, 43.4)]
for tx, ty in targ:
    ax.plot([5.6, tx], [50.6, ty], color=C_DATA, lw=0.8, alpha=0.65, zorder=2)
    ax.scatter([tx], [ty], s=14, color=C_DATA, zorder=3)
cap(9, 41.1, "regulons R (data-driven)", C_DATA)

# ================================================================== 2 encode
panel(18, 32, 13, 54, "2 · encode", ec=C_CORE, fc="#f2f6fb", title_c=C_CORE)

ax.add_patch(Polygon([(20.5, 57.5), (28.5, 57.5), (27.2, 76), (21.8, 76)],
                     closed=True, fc="white", ec=C_CORE, lw=1.5, zorder=2))
for xl in (23.0, 24.5, 26.0):
    y1 = 58.6 + (xl - 20.5) * 0.19
    y2 = 75.2 - (28.5 - xl) * 0.19
    ax.plot([xl, xl], [y1, y2], color=C_CORE, lw=0.8, alpha=0.5, zorder=2)
cap(24.5, 77.6, "encoder  q$_{φ}$(z | x)", C_CORE)

card(19.5, 47, 11, 9, C_INF, fc="#fdf1f0")
dots(21.4, 52.4, 4, C_INF, r=0.45, dx=2.15)
cap(25, 48.5, "z$_f$   fate (K)", C_INF, weight="bold")

card(19.5, 34, 11, 9, C_BG, fc="#f0f3f5")
for r_ in range(2):
    dots(21.0, 40.4 - r_ * 2.1, 8, "#90a4ae", r=0.34, dx=1.15)
cap(25, 35.4, "z$_b$   background", "#455a64", weight="bold")

arrow((23.0, 57.2), (21.8, 56.4), color=C_CORE)
arrow((26.6, 57.2), (28.6, 43.4), color=C_CORE, rad=-0.25)

# ================================================================== 3 decode
panel(33, 32, 37, 59, "3 · decode — two modes", ec="#9e9e9e", fc="#fdfdfd")

# ---- MODE A -----------------------------------------------------------
card(35, 61, 33, 26.5, C_INF, fc="#fdf4f3", lw=1.4)
ax.text(36.5, 86.2, "MODE A — informed (markers)", fontsize=8.4,
        fontweight="bold", color=C_INF, va="top", zorder=5)
fates = list(FATE)
ydots = [80.5, 75.6, 70.7, 65.8]
yblk = [79.2, 74.4, 69.6, 64.8]
for i, f in enumerate(fates):
    ax.scatter([38.2], [ydots[i]], s=34, color=FCOL[i], zorder=5)
    ax.plot([39.0, 56.5], [ydots[i], yblk[i] + 1.1], color=FCOL[i],
            lw=1.4, zorder=3)
    ax.add_patch(FancyBboxPatch((56.5, yblk[i]), 10, 2.2,
                                boxstyle="round,pad=0.1,rounding_size=0.4",
                                fc=FCOL[i], ec="none", alpha=0.88, zorder=4))
    ax.text(61.5, yblk[i] + 1.1, f"{f} genes", fontsize=5.8, color="white",
            ha="center", va="center", fontweight="bold", zorder=5)
# one masked (forbidden) connection to show group sparsity
ax.plot([39.0, 56.5], [80.5, 75.5], color="#9e9e9e", lw=0.9,
        linestyle=(0, (2, 2)), zorder=3)
ax.text(47.5, 79.2, "✕", fontsize=8, color="#9e9e9e", ha="center",
        va="center", zorder=5)
cap(36.8, 62.4, "group-sparse decoder · supervision CE(fate head, labels) · "
                "GRL keeps z$_b$ fate-free", C_INF, fs=6.2, ha="left")

# ---- MODE B -----------------------------------------------------------
card(35, 40, 33, 18.5, C_DATA, fc="#f3f9f3", lw=1.4)
ax.text(36.5, 57.3, "MODE B — discovery (marker-free)", fontsize=8.4,
        fontweight="bold", color=C_DATA, va="top", zorder=5)
ydots_b = [46.6, 49.4, 52.2, 55.0 - 0.0]
ydots_b = [53.6, 51.0, 48.4, 45.8]
hprof = [2.2, 0.5, 1.0, 0.35, 1.5, 0.4]
for i in range(4):
    ax.scatter([38.2], [ydots_b[i]], s=30, color=C_DATA, zorder=5)
    ax.plot([39.0, 42.0], [ydots_b[i], ydots_b[i]], color=C_DATA, lw=1.2,
            zorder=3)
    for b in range(6):
        h = hprof[b] * (0.55 if i % 2 else 1.0) * rng.uniform(0.75, 1.1)
        ax.add_patch(Rectangle((42.4 + b * 1.75, ydots_b[i] - h / 2), 1.05, h,
                               fc=C_DATA, alpha=0.75, ec="none", zorder=4))
ax.text(55.4, 49.6, "×R", fontsize=9, fontweight="bold", color=C_DATA,
        ha="center", va="center", zorder=5)
# gene strip: segments across genes, colored by dominant fate
segs = [FCOL[0]] * 2 + ["#cfd8dc"] + [FCOL[1]] * 2 + [FCOL[2]] * 2 + \
    ["#cfd8dc"] + [FCOL[3]] * 2
for j, c in enumerate(segs):
    ax.add_patch(Rectangle((58.0 + j * 1.02, 48.4), 0.92, 2.6, fc=c,
                           ec="white", lw=0.3, alpha=0.9, zorder=4))
cap(62.2, 46.4, "gene rates", C_DATA, fs=5.8)
cap(36.8, 41.4, "attention over regulons · penalties L1 + row-entropy + "
                "QC-nuisance · KL warm-up · no labels", C_DATA, fs=6.2,
    ha="left")

# ---- background + likelihood strip ------------------------------------
card(35, 32.8, 33, 5.2, C_CORE, fc="white", lw=1.2)
cap(36.8, 36.6, "background decoder: dense W$_b$ — absorbs nuisance "
                "(cell cycle, stress, batch)", "#455a64", fs=6.2, ha="left")
ax.text(50.5, 34.3, r"NB likelihood:   log $\mu$ = log $\ell$ + "
        r"log softmax(w$_f$ + w$_b$ + b)", fontsize=7.4, color=C_CORE,
        ha="center", va="center", fontweight="bold", zorder=5)

# ================================================================== 4 output
panel(72, 63, 27, 28, "4 · output", ec="#9e9e9e", fc="#fafafa")
props = [(0.66, 0.12, 0.12, 0.10), (0.18, 0.52, 0.20, 0.10),
         (0.10, 0.12, 0.16, 0.62)]
for c, pr in enumerate(props):
    base = 70.0
    for k, p in enumerate(pr):
        h = p * 13
        ax.add_patch(Rectangle((76.0 + c * 3.1, base), 2.3, h, fc=FCOL[k],
                               ec="white", lw=0.5, zorder=3))
        base += h
cap(79.6, 66.6, "fate probabilities\n· entropy", "#455a64", fs=6.6)
ranks = [9.5, 7.4, 5.8, 4.4, 3.1]
names = ["Abd-B", "cad", "lola", "br", "…"]
for i, (L, nm) in enumerate(zip(ranks, names)):
    ax.add_patch(FancyBboxPatch((88.0, 80.6 - i * 2.3), L, 1.35,
                                boxstyle="round,pad=0.05,rounding_size=0.3",
                                fc=C_DATA, alpha=0.85 - i * 0.12, ec="none",
                                zorder=3))
    ax.text(88.4, 81.28 - i * 2.3, nm, fontsize=5.2, color="white",
            va="center", zorder=4)
cap(92.0, 66.6, "nominated regulators\n& marker genes", C_DATA, fs=6.6)

# ================================================================== 5 test
panel(72, 32, 27, 28, "5 · prior-recovery test", ec=C_EVAL, fc="white",
      ls=(0, (5, 3)), title_c=C_EVAL)
for i in range(4):
    for j in range(4):
        c = FCOL[i] if i == j else "#eceff1"
        ax.add_patch(Rectangle((74.6 + j * 2.45, 46.4 - i * 2.45), 2.25, 2.25,
                               fc=c, ec="white", lw=0.8,
                               alpha=0.9 if i == j else 1.0, zorder=3))
cap(79.4, 37.4, "vs known compartments", "#848484", fs=5.6)
for k, s in enumerate(["markers enter evaluation ONLY", "ARI · NMI",
                       "marker AUROC", "regulator recall@k"]):
    cap(87.2, 54.6 - k * 2.5, s, C_EVAL if k == 0 else "#455a64",
        fs=6.8, ha="left", weight="bold" if k == 0 else "normal")
cap(87.2, 42.0, "gate status: discovery G1′ failed\n→ modality constraint, "
                "see STAGE_REPORT", "#848484", fs=6.0, ha="left")

# ================================================================== front-ends
panel(1, 3, 69, 24, "prior-free label front-ends — feed MODE A's supervision "
      "slot", ec=C_DATA, fc="#f1f8f1", title_c=C_DATA, fs=8.6)

# -- F1 seed
card(3, 5.5, 20, 16.4, C_DATA)
g = [(8.2, 16.4, FCOL[0]), (14.6, 17.8, FCOL[2]), (11.4, 10.6, FCOL[3])]
for cx, cy, c in g:
    pts = rng.normal([cx, cy], [1.15, 0.95], (9, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=13, color=c, alpha=0.85, zorder=3)
cap(13, 7.0, "seed — exclusive detection modules", C_DATA, fs=6.6)

# -- F2 amplify EM loop
card(25, 5.5, 20, 16.4, C_DATA)
chips = [(28.6, 14.6, "VAE"), (37.4, 17.2, "nominate"), (38.6, 10.2, "update")]
for cx, cy, s in chips:
    ax.add_patch(FancyBboxPatch((cx - 2.6, cy - 1.2), 5.2, 2.4,
                                boxstyle="round,pad=0.1,rounding_size=0.5",
                                fc="white", ec=C_DATA, lw=1.1, zorder=3))
    ax.text(cx, cy, s, fontsize=5.8, color=C_DATA, ha="center", va="center",
            fontweight="bold", zorder=4)
arrow((31.4, 15.4), (34.4, 16.9), color=C_DATA, lw=1.2, rad=-0.2)
arrow((39.2, 15.8), (39.0, 11.8), color=C_DATA, lw=1.2, rad=-0.2)
arrow((35.8, 10.4), (30.2, 13.4), color=C_DATA, lw=1.2, rad=-0.2)
cap(35, 7.0, "amplify — EM ×3–5 (anti-drift update)", C_DATA, fs=6.6)

# -- F3 FateOT
card(47, 5.5, 21, 16.4, C_DATA)
src = rng.normal([52.0, 14.5], [0.9, 2.6], (16, 2))
ax.scatter(src[:, 0], src[:, 1], s=13, color="#90a4ae", alpha=0.85, zorder=3)
for k, (cy, c) in enumerate([(17.6, FCOL[0]), (14.0, FCOL[2]),
                             (10.4, FCOL[3])]):
    pts = rng.normal([62.6, cy], [1.5, 0.8], (7, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=13, color=c, alpha=0.85, zorder=3)
    ax.add_patch(FancyArrowPatch((54.3, 10.5 + k * 3.8), (60.2, cy),
                                 arrowstyle="-|>", color="#90a4ae", lw=0.9,
                                 mutation_scale=8, shrinkA=0, shrinkB=2,
                                 zorder=2, connectionstyle="arc3,rad=0.15"))
cap(57.5, 7.0, "FateOT — Sinkhorn to unlabeled adult clusters", C_DATA, fs=6.6)

# ================================================================== arrows
arrow((14.3, 64.5), (20.3, 64.5), color=C_CORE)                     # counts
arrow((14.3, 85.3), (34.9, 84.4), color=C_EXP, ls=(0, (4, 2.4)))    # markers
arrow((14.3, 45.5), (34.9, 45.5), color=C_DATA)                     # R
arrow((30.8, 52.6), (34.9, 69.0), color=C_INF)                      # zf -> A
arrow((30.8, 49.2), (34.9, 51.6), color=C_INF)                      # zf -> B
arrow((30.8, 38.3), (34.9, 36.2), color=C_BG)                       # zb
arrow((50.0, 39.8), (50.0, 38.2), color="#455a64", lw=1.2)          # B->NB
arrow((68.0, 72.0), (71.3, 72.0), color=C_INF)                      # A->NB
arrow((71.3, 72.0), (71.3, 34.5), color=C_INF)
arrow((71.3, 34.5), (68.4, 34.5), color=C_INF)
arrow((70.0, 76.5), (71.9, 76.5), color="#616161")                  # ->output
arrow((85.5, 62.8), (85.5, 60.4), color=C_EVAL)                     # ->test
# front-end corridor -> MODE A supervision slot
arrow((33.9, 27.4), (33.9, 66.0), color=C_DATA, lw=2.4)
arrow((33.9, 66.0), (34.9, 67.5), color=C_DATA, lw=2.4)
cap(36.0, 29.5, "soft pseudo-labels (marker-free)", C_DATA, fs=6.6, ha="left")

# ================================================================== legend
leg = [(C_DATA, "-", "data-driven"), (C_EXP, (0, (4, 2.4)), "experimental prior"),
       (C_INF, "-", "fate branch (informed)"), (C_DATA, "-", "fate branch (discovery)"),
       (C_BG, "-", "background"), (C_EVAL, (0, (5, 3)), "evaluation only")]
x = 19.0
for c, ls, s in leg:
    ax.plot([x, x + 2.6], [1.6, 1.6], color=c, lw=2.2, linestyle=ls)
    ax.text(x + 3.3, 1.6, s, fontsize=6.8, color="#616161", va="center")
    x += 3.3 + len(s) * 0.62 + 3.4

fig.savefig(config.FIGURES_DIR / "algorithm_overview.png", bbox_inches="tight")
fig.savefig(config.FIGURES_DIR / "algorithm_overview.pdf", bbox_inches="tight")
print("wrote", config.FIGURES_DIR / "algorithm_overview.png")
