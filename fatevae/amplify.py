"""Stage 2 (Amplify): EM loop turning seed modules into fate programs.

Justification from v0.1/v0.2 results: given partial anchoring the informed
architecture recovers unseen markers (LOGO: hh 0.91, Dr 0.92); the loop
below manufactures that anchoring from the statistical seed alone.

    repeat:
      1. soft pseudo-labels = module scores of current modules
      2. mask = extended gene sets (module genes + their top-correlated
         genes + co-expression regulon targets of module TFs) - all
         data-derived
      3. train the informed-mode two-branch NB-VAE on the pseudo-labels
      4. nominate genome-wide: per fate dim, genes by |corr(z_f_k, gene)|
         (TF-panel genes and any-gene candidates)
      5. update modules from nominees; measure Jaccard stability
    until stable or n_iter reached.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..model import TrainConfig, build_fate_mask, train_fatevae
from ..priors.markers import module_scores


def _top_corr_genes(Zstd: np.ndarray, var_names, gene: str, topn: int) -> list[str]:
    j = pd.Index(var_names).get_loc(gene)
    r = Zstd.T @ Zstd[:, j] / Zstd.shape[0]
    r[j] = -1
    top = np.argsort(-r)[:topn]
    return [var_names[t] for t in top if r[t] > 0.1]


def extend_sets(modules, Zstd, var_names, coexp, topn: int = 20,
                cap: int = 80) -> dict[str, list[str]]:
    ext: dict[str, list[str]] = {}
    vset = set(var_names)
    for m, genes in modules.items():
        s = set(genes) & vset
        for g in genes:
            if g in vset and len(s) < cap:
                s.update(_top_corr_genes(Zstd, var_names, g, topn))
            if g in coexp and len(s) < cap:
                s.update(coexp[g][:40])
        ext[m] = sorted(s)[:cap]
    return ext


def nominate(zf: np.ndarray, Zstd: np.ndarray, var_names,
             tf_panel: list[str], n_tf: int = 8, n_any: int = 7,
             min_corr: float = 0.15) -> dict[str, list[str]]:
    vidx = pd.Index(var_names)
    tf_cols = vidx.get_indexer([t for t in tf_panel if t in vidx])
    out: dict[str, list[str]] = {}
    zs = (zf - zf.mean(0)) / (zf.std(0) + 1e-9)
    for k in range(zf.shape[1]):
        r = Zstd.T @ zs[:, k] / Zstd.shape[0]
        tfs = [var_names[t] for t in tf_cols[np.argsort(-np.abs(r[tf_cols]))[:n_tf]]
               if abs(r[t]) > min_corr]
        r_tf = set(tfs)
        anyg = [var_names[t] for t in np.argsort(-np.abs(r))[:n_any]
                if abs(r[t]) > min_corr and var_names[t] not in r_tf]
        out[f"F{k}"] = tfs + anyg
    return out


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def damped_update(old: list[str], new: list[str], keep_old: int = 6,
                  size: int = 15) -> list[str]:
    """Union of top new nominees with retained old members (anti-drift)."""
    merged = list(dict.fromkeys(new[: size - keep_old] + old[:keep_old]))
    return merged[:size]


def amplify(X, counts, lib, modules_init, Zstd, var_names, coexp,
            tf_panel, n_iter: int = 3, seed: int = 0, verbose: bool = True):
    """Returns (model, zf, logits, module_history, stability_history)."""
    modules = {k: list(v) for k, v in modules_init.items()}
    history, stability = [dict(modules)], []
    model = None
    zf = logits = None
    for it in range(n_iter):
        scores = module_scores(X, var_names, modules)
        mask = build_fate_mask(var_names, extend_sets(modules, Zstd, var_names, coexp))
        cfg = TrainConfig(epochs_pretrain=80, epochs_sup=100, epochs_adv=60,
                          batch_size=256, seed=seed + it, log_every=0)
        model, _ = train_fatevae(X, counts, lib, mask,
                                 scores[list(modules)].to_numpy(),
                                 np.ones(X.shape[0], bool),
                                 np.zeros(X.shape[0], bool), cfg)
        zf, _, logits = model.latent(torch.tensor(X), torch.tensor(lib))
        new_modules = nominate(zf, Zstd, var_names, tf_panel)
        # keep module order stable by best gene overlap with previous
        ordered = {}
        used = set()
        for m in modules:
            cand = max(new_modules, key=lambda n: jaccard(modules[m], new_modules[n]))
            if cand in used:
                continue
            ordered[m] = damped_update(modules[m], new_modules[cand])
            used.add(cand)
        for n, gs in new_modules.items():
            if n not in used and gs:
                ordered[f"{n}_x"] = gs
                used.add(n)
        stab = float(np.mean([jaccard(modules.get(m, []), ordered.get(m, []))
                              for m in set(modules) | set(ordered)]))
        stability.append(stab)
        if verbose:
            print(f"[amplify it{it}] stability(Jaccard)={stab:.3f}; modules: "
                  + "; ".join(f"{m}({len(g)})={','.join(g[:6])}"
                              for m, g in ordered.items()), flush=True)
        modules = ordered
        history.append(dict(modules))
    return model, zf, logits, history, stability
