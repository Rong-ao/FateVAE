"""Fate probability post-processing: calibration, entropy, uncertainty."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def calibrate_temperature(
    logits: np.ndarray,
    soft_labels: np.ndarray,
    max_iter: int = 300,
    lr: float = 0.05,
) -> float:
    """Temperature scaling on the validation labels; returns T."""
    lt = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(soft_labels, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=lr, max_iter=max_iter)
    y = yt.argmax(dim=1)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lt / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def fate_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = logits / max(temperature, 1e-9)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def uncertainty(
    model,
    X: np.ndarray,
    lib: np.ndarray,
    n_samples: int = 20,
    dropout_keys: tuple = (),
) -> np.ndarray:
    """MC-dropout / weight-perturbation free estimate placeholder: here we
    quantify predictive uncertainty as the mean JS-divergence across
    reparameterized samplings of z (model must be on same device as X).

    Returns per-cell JS divergence (nats)."""
    Xt = torch.tensor(X, dtype=torch.float32)
    libt = torch.tensor(lib, dtype=torch.float32)
    probs = []
    with torch.no_grad():
        model.eval()
        torch.manual_seed(0)
        for _ in range(n_samples):
            out = model(Xt, libt)
            p = F.softmax(torch.tensor(out["fate_logits"]), dim=1).numpy()
            probs.append(p)
    probs = np.stack(probs)  # (S, n, K)
    mean = probs.mean(axis=0)
    js = np.mean(
        np.sum(probs * (np.log(probs + 1e-12) - np.log(mean + 1e-12)[None]), axis=2),
        axis=0,
    )
    return js
