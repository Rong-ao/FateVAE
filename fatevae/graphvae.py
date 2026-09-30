"""GraphFateVAE: GCN-encoder two-branch NB-VAE (Guidance.md §3.4).

Rationale specific to our failure mode: dropout destroys the per-cell fate-
TF signal, but cells of one compartment are neighbors on the expression
graph - aggregating k neighbors' features recovers regional coherence
(~sqrt(k) SNR) that a per-cell MLP encoder cannot. The decoder, NB
likelihood, two-branch factorization and fate-mask machinery are inherited
from the informed-mode FateVAE; only the encoder becomes a graph
convolutional network over the cell-cell kNN graph.

Full-batch (transductive) training: the GCN needs the whole graph per step;
with ~6k cells this is small and GPU-friendly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fatevae import FateVAE, gaussian_kl


def build_knn_graph(embedding: np.ndarray, k: int = 15) -> torch.Tensor:
    """Symmetrically-normalized adjacency D^-1/2 (A + I) D^-1/2 as a sparse
    COO tensor (GCN convention). A = symmetrized kNN from scikit-learn."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(embedding)
    _, idx = nn.kneighbors(embedding)
    n = embedding.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].ravel()  # drop self
    pairs = set(zip(src.tolist(), dst.tolist()))
    rows, cols = [], []
    for (i, j) in pairs:
        rows.append(i); cols.append(j)
        if (j, i) not in pairs:
            rows.append(j); cols.append(i)
    rows = np.asarray(rows); cols = np.asarray(cols)
    deg = np.bincount(rows, minlength=n).astype(np.float64)
    # A_hat = A + I, D_hat = deg + 1; value(i,j) = 1/sqrt(D_i D_j)
    deg_hat = deg + 1.0
    d_is = 1.0 / np.sqrt(deg_hat)
    vals = d_is[rows] * d_is[cols]
    rows = np.concatenate([rows, np.arange(n)])  # self-loops: value 1/D_i
    cols = np.concatenate([cols, np.arange(n)])
    vals = np.concatenate([vals, d_is ** 2])
    indices = torch.from_numpy(np.vstack([rows, cols])).long()
    return torch.sparse_coo_tensor(indices, torch.from_numpy(vals).float(),
                                   (n, n)).coalesce()


class GCNLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f)

    def forward(self, x, A):
        return A @ self.lin(x)


class GraphFateVAE(nn.Module):
    """FateVAE with a 2-layer GCN encoder. Same interface as FateVAE for
    encode/decode/losses, plus the adjacency argument."""

    def __init__(self, n_genes: int, fate_mask: torch.Tensor, n_bg: int = 16,
                 hidden: tuple[int, ...] = (256, 64), n_extra_fate_dims: int = 0):
        super().__init__()
        self.core = FateVAE(n_genes, fate_mask, n_bg=n_bg, hidden=hidden,
                            n_extra_fate_dims=n_extra_fate_dims)
        # replace the MLP encoder with graph layers of the same widths
        dims = [n_genes, *hidden]
        self.gcn = nn.ModuleList()
        for a, b in zip(dims[:-1], dims[1:]):
            self.gcn.append(GCNLayer(a, b))

    def encode(self, x, A):
        h = x
        for i, layer in enumerate(self.gcn):
            h = layer(h, A)
            if i < len(self.gcn) - 1:
                h = F.silu(h)
        mu = self.core.enc_mu(h)
        logvar = self.core.enc_logvar(h)
        nf, nb = self.core.n_zf, self.core.n_zb
        return (mu[:, :nf], logvar[:, :nf], mu[:, nf:], logvar[:, nf:])

    def forward(self, x, lib, A):
        mu_f, lv_f, mu_b, lv_b = self.encode(x, A)
        zf = mu_f + torch.randn_like(mu_f) * torch.exp(0.5 * lv_f)
        zb = mu_b + torch.randn_like(mu_b) * torch.exp(0.5 * lv_b)
        w = self.core.gene_weights(zf, zb)
        log_mu = torch.log(lib)[:, None] + F.log_softmax(w, dim=1)
        fate_logits = self.core.fate_head(zf)
        return dict(mu_f=mu_f, lv_f=lv_f, mu_b=mu_b, lv_b=lv_b, zf=zf,
                    zb=zb, log_mu=log_mu, fate_logits=fate_logits)

    @torch.no_grad()
    def latent(self, X, lib, A):
        self.eval()
        out = self.forward(X, lib, A)
        return (out["mu_f"].cpu().numpy(), out["mu_b"].cpu().numpy(),
                out["fate_logits"].cpu().numpy())


@dataclass
class GraphTrainConfig:
    n_bg: int = 16
    hidden: tuple[int, ...] = (256, 64)
    steps_pretrain: int = 2000
    steps_sup: int = 3000
    steps_adv: int = 2000
    lr: float = 1e-3
    beta_f: float = 1e-3
    beta_b: float = 5e-4
    lambda_sup: float = 5.0
    lambda_adv: float = 0.3
    lambda_tc: float = 0.1
    sup_temperature: float = 0.5
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "cuda"
    log_every: int = 200


def train_graphvae(X, counts, lib, A, fate_mask, soft_scores,
                   cfg: GraphTrainConfig | None = None,
                   model: GraphFateVAE | None = None,
                   supervised: bool = True):
    """Full-batch training; returns (model, history)."""
    cfg = cfg or GraphTrainConfig()
    dev = torch.device(cfg.device if torch.cuda.is_available()
                       or cfg.device == "cpu" else "cpu")
    torch.manual_seed(cfg.seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    Ct = torch.tensor(counts, dtype=torch.float32, device=dev)
    libt = torch.tensor(lib, dtype=torch.float32, device=dev)
    At = A.to(dev).coalesce() if A.device != dev else A

    s = soft_scores / max(cfg.sup_temperature, 1e-6)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    Yt = torch.tensor(e / e.sum(axis=1, keepdims=True), dtype=torch.float32,
                     device=dev)

    if model is None:
        # construct on CPU (mask must match parameter devices), then move
        model = GraphFateVAE(X.shape[1], fate_mask.cpu(), n_bg=cfg.n_bg,
                             hidden=tuple(cfg.hidden)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def _cov_penalty(z):
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
        return ((cov - torch.diag(torch.diag(cov))) ** 2).mean()

    history = []
    schedule = [("pre", cfg.steps_pretrain), ("sup", cfg.steps_sup),
                ("adv", cfg.steps_adv)] if supervised else \
               [("pre", cfg.steps_pretrain)]
    for stage, n_steps in schedule:
        for t in range(n_steps):
            prog = (t + 1) / n_steps
            out = model(Xt, libt, At)
            losses = {}
            losses["nb"] = model.core.nb_nll(Ct, out["log_mu"])
            losses["kl"] = cfg.beta_f * gaussian_kl(out["mu_f"], out["lv_f"]) \
                + cfg.beta_b * gaussian_kl(out["mu_b"], out["lv_b"])
            if stage in ("sup", "adv"):
                lam = cfg.lambda_sup * prog
                losses["sup"] = lam * torch.mean(torch.sum(
                    -Yt * F.log_softmax(out["fate_logits"], dim=1), dim=1))
                losses["tc"] = cfg.lambda_tc * _cov_penalty(out["zf"])
            if stage == "adv":
                from .fatevae import grad_reverse
                adv = model.core.disc(
                    grad_reverse(out["zb"], cfg.lambda_adv * prog))
                losses["adv"] = F.cross_entropy(
                    adv, Yt.argmax(dim=1), label_smoothing=0.1)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if cfg.log_every and t % cfg.log_every == 0:
                history.append({"stage": stage, "step": t,
                                **{k: float(v) for k, v in losses.items()}})
                print(f"[{stage} {t:5d}] " + " ".join(
                    f"{k}={float(v):.3f}" for k, v in losses.items()),
                    flush=True)
    return model, history
