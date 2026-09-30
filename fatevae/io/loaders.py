"""Data loaders: AnnData disc objects, pySCENIC loom regulon activity, and
the larva->adult fate anchor from the merged dataset.

Notes on conventions (see bio-single-cell-data-io skill):
- AnnData is cells x genes; the SCENIC loom matrix is genes x cells, and its
  per-cell AUC values live in col_attrs/RegulonsAUC (we read them directly
  with h5py to avoid a loompy dependency).
- Count matrices are kept sparse; only small AUC / embedding matrices are
  densified.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from .. import config


def load_disc(
    h5ad_path: Path | str = config.DMEL_H5AD,
    raw_h5ad_path: Path | str | None = None,
) -> ad.AnnData:
    """Load a processed disc AnnData and make slot conventions uniform.

    Guarantees on return:
    - ``layers['counts']`` holds raw integer UMI counts - taken from
      ``raw_h5ad_path`` when given (the processed files store a
      corrected/normalized matrix in their counts layer, not UMIs)
    - ``X`` holds log1p library-normalized values recomputed from counts
    - ``.obs_names`` unique
    """
    adata = ad.read_h5ad(h5ad_path)
    adata.var_names_make_unique()

    if raw_h5ad_path is not None:
        raw = ad.read_h5ad(raw_h5ad_path)
        raw.var_names_make_unique()
        common_obs = adata.obs_names.intersection(raw.obs_names)
        if len(common_obs) < 0.95 * adata.n_obs:
            raise ValueError(
                f"raw object barcodes do not match processed object "
                f"({len(common_obs)}/{adata.n_obs})"
            )
        raw = raw[list(common_obs)]
        missing_var = adata.var_names.difference(raw.var_names)
        if len(missing_var):
            warnings.warn(
                f"{len(missing_var)} genes of the processed object absent from "
                "raw counts; their counts set to 0"
            )
        counts = sp.csc_matrix((adata.n_obs, adata.n_vars), dtype=np.float32)
        take = [g for g in adata.var_names if g in raw.var_names]
        sub = raw[:, take].X.tocsc()
        gi = pd.Index(adata.var_names).get_indexer(take)
        counts[:, gi] = sub
        counts = counts.tocsr()
    else:
        counts = adata.layers["counts"].tocsr()

    data = counts.data
    if data.size and not np.allclose(data, np.round(data)):
        warnings.warn("counts are not integer; rounding (check raw source!)")
        counts.data = np.round(counts.data)
    adata.layers["counts"] = counts.astype(np.float32)
    adata.X = _lognorm(adata.layers["counts"])
    return adata


def _lognorm(counts) -> "sp.csr_matrix":
    counts = counts.tocsr().astype(np.float32)
    lib = np.asarray(counts.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = counts.multiply(1e4 / lib[:, None]).tocsr()
    out.data = np.log1p(out.data)
    return out


def load_scenic(
    loom_path: Path | str = config.DMEL_SCENIC_LOOM,
    obs_names: pd.Index | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Read pySCENIC final loom: (AUC matrix cells x regulons, regulon->genes).

    Regulon names are stripped of the trailing ``(+)``/``(-)`` annotation so
    they match plain TF symbols (e.g. ``Dll_(+)`` -> ``Dll``). If
    ``obs_names`` is given the AUC frame is reindexed to those cells
    (intersecting; loom barcodes usually carry the sample suffix the loom
    writer added).
    """
    with h5py.File(loom_path, "r") as f:
        cell_ids = f["col_attrs/CellID"][:].astype(str)
        auc_rec = f["col_attrs/RegulonsAUC"][:]
        auc_names = auc_rec.dtype.names
        auc = np.stack([np.asarray(auc_rec[n], dtype=np.float64) for n in auc_names], axis=1)
        reg_rec = f["row_attrs/Regulons"][:]
        genes = f["row_attrs/Gene"][:].astype(str)

    clean = lambda s: s.replace("_(+)", "").replace("_(-)", "").replace("(+)", "").replace("(-)", "")
    auc = pd.DataFrame(auc, index=pd.Index(cell_ids, name="obs_names"),
                       columns=[clean(n) for n in auc_names])

    regulons: dict[str, list[str]] = {}
    for name in reg_rec.dtype.names:
        members = np.asarray(reg_rec[name]).astype(bool)
        if members.sum() > 0:
            regulons[clean(name)] = sorted(genes[members].tolist())

    if obs_names is not None:
        auc = _align_cells(auc, pd.Index(obs_names))
    return auc, regulons


def _align_cells(auc: pd.DataFrame, obs_names: pd.Index) -> pd.DataFrame:
    """Match loom CellIDs to AnnData obs_names.

    The SCENIC loom was written from the same cells, but loom writers often
    append a sample tag. Strategy: exact match, else strip suffix after the
    last ``-`` / ``_`` token and match again.
    """
    if auc.index.duplicated().any():
        auc = auc[~auc.index.duplicated(keep="first")]
    common = auc.index.intersection(obs_names)
    if len(common) == len(obs_names):
        return auc.loc[obs_names]
    strip = lambda s: str(s).rsplit("-", 1)[0]
    remap = pd.Series(auc.index, index=[strip(s) for s in auc.index], dtype=object)
    remap = remap[~remap.index.duplicated(keep="first")]
    keys = pd.Index([strip(s) for s in obs_names])
    hits = remap.reindex(keys)
    if hits.isna().mean() > 0.5:
        raise ValueError(
            f"Cannot align loom cells to AnnData (matched {(~hits.isna()).sum()}/{len(keys)}); "
            "check barcode conventions."
        )
    if hits.isna().any():
        warnings.warn(f"{hits.isna().sum()} cells missing from SCENIC loom; filling AUC with NaN->0")
    out = auc.loc[hits.fillna(auc.index[0])].copy()
    out.index = obs_names
    if hits.isna().any():
        out.loc[hits.isna().values] = 0.0
    return out


def map_larva_to_adult_fates(
    merged_h5ad: Path | str = config.DMEL_MERGED_H5AD,
    n_neighbors: int = 25,
) -> pd.DataFrame:
    """Prospective fate anchoring: for each larval cell, its adult-neighbour
    composition in the integrated latent space of ``merged_scRNA.h5ad``.

    Returns a cells x fate-group DataFrame of neighbour fractions (rows sum
    to <= 1; the remainder are contaminant/unknown adult cells), indexed by
    the larval cell barcodes.
    """
    merged = ad.read_h5ad(merged_h5ad)
    is_larva = (merged.obs["sample"] == "larva").values
    emb = np.asarray(merged.obsm["X_pca"], dtype=np.float64)

    groups = config.ADULT_FATE_GROUPS
    adult_ann = merged.obs["adult_annotation"].astype(str)

    adult_mask = ~is_larva
    adult_labels = pd.Series(index=merged.obs_names[adult_mask], dtype=object)
    for gname, cats in groups.items():
        adult_labels[adult_ann[adult_mask].isin(cats)] = gname

    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=config.n_cpus())
    nn.fit(emb[adult_mask])
    _, idx = nn.kneighbors(emb[is_larva])

    lab = adult_labels.to_numpy(dtype=object)
    nb_labels = lab[idx]  # (n_larva, k)
    out = pd.DataFrame(0.0, index=merged.obs_names[is_larva], columns=list(groups))
    for g in groups:
        out[g] = (nb_labels == g).mean(axis=1)
    return out
