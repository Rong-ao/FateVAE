"""Cell-level train/validation/test splits (70/15/15, leiden-stratified)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def cell_splits(
    leiden: np.ndarray,
    seed: int = 0,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks (train, val, test) stratified by leiden cluster."""
    y = np.asarray(leiden).astype(str)
    n_test = test_size
    n_val = val_size / (1.0 - test_size)
    idx = np.arange(len(y))
    train_val, test = train_test_split(idx, test_size=n_test, stratify=y, random_state=seed)
    train, val = train_test_split(train_val, test_size=n_val, stratify=y[train_val], random_state=seed)
    masks = np.zeros(len(y), dtype=bool), np.zeros(len(y), dtype=bool), np.zeros(len(y), dtype=bool)
    masks[0][train] = True
    masks[1][val] = True
    masks[2][test] = True
    return masks
