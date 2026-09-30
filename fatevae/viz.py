"""Plotting helpers (matplotlib, consistent style across phases)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FATE_COLORS = {
    "A8": "#1b9e77",
    "A9p1": "#d95f02",
    "A9p2": "#7570b3",
    "A10": "#e7298a",
}


def scatter_embedding(
    xy: np.ndarray,
    color: np.ndarray,
    title: str,
    out: Path,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    cbar_label: str = "",
    s: int = 4,
    alpha: float = 0.7,
):
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=180)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=color, cmap=cmap, s=s, alpha=alpha,
                    vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if cbar_label:
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label(cbar_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def multi_scatter(
    xy: np.ndarray,
    panels: dict[str, np.ndarray],
    out: Path,
    cmap: str = "viridis",
    ncol: int = 3,
    panel_titles: dict[str, str] | None = None,
):
    n = len(panels)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), dpi=170)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, vals) in zip(axes, panels.items()):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap, s=3, alpha=0.7,
                        linewidths=0)
        ax.set_title((panel_titles or {}).get(name, name), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.75)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
