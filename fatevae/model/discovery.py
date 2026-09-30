"""DiscoveryFateVAE: prior-independent fate program discovery (proposal v2 §4 Aim 2).

The fate latent may influence gene rates ONLY through sparse non-negative
combinations of SCENIC regulon target sets (a fully data-driven structure -
co-expression + motif pruning, computable from any annotated genome):

    gene_weights_fate[i, g] = sum_k z_f[i, k] * sum_m softplus(A)[k, m] * R[m, g]

with A (K x M) the learned attention, R (M x G) the fixed regulon-membership
matrix. Sparse, decorrelated attention rows make each fate dimension a small
regulon module; the module composition is the model's *nomination* of fate
regulators, and the resulting gene loadings its nomination of markers.

No supervision term exists in this mode (no marker genes anywhere in the
objective). Cell-cycle/QC-correlated regulons are down-weighted through a
data-driven nuisance penalty vector (correlation of regulon activity with
library size / mt fraction), not through any curated list.

Training uses KL warm-up to avoid posterior collapse in the absence of
supervision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fatevae import gaussian_kl


class DiscoveryFateVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        R: torch.Tensor,
        n_fate: int = 8,
        n_bg: int = 16,
        hidden: tuple[int, ...] = (256, 64),
        attn_init: float = 0.1,
    ):
        """R: (M regulons, G genes) binary/weighted membership, fixed."""
        super().__init__()
        self.n_genes = n_genes
        self.n_fate = n_fate
        self.n_zb = n_bg
        self.register_buffer("R", R.float())
        self.M = R.shape[0]

        dims = [n_genes, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
        self.encoder = nn.Sequential(*layers)
        self.enc_mu = nn.Linear(hidden[-1], n_fate + n_bg)
        self.enc_logvar = nn.Linear(hidden[-1], n_fate + n_bg)

        self.attn_raw = nn.Parameter(
            torch.empty(n_fate, self.M).normal_(mean=math.log(math.expm1(attn_init)), std=0.3)
        )
        self.dec_bias = nn.Parameter(torch.zeros(n_genes))
        self.dec_bg = nn.Linear(n_bg, n_genes, bias=False)
        log_theta = torch.zeros(n_genes)
        self.log_theta = nn.Parameter(log_theta)

    # -- parts ---------------------------------------------------------------
    def attention(self) -> torch.Tensor:
        return F.softplus(self.attn_raw)

    def fate_gene_loader(self) -> torch.Tensor:
        """(K, G) gene program per fate dim = attention @ R."""
        return self.attention() @ self.R

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.enc_mu(h), self.enc_logvar(h)
        return (mu[:, : self.n_fate], logvar[:, : self.n_fate],
                mu[:, self.n_fate:], logvar[:, self.n_fate:])

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def gene_weights(self, zf, zb):
        w_f = zf @ self.fate_gene_loader()  # (N, G)
        return w_f + F.linear(zb, self.dec_bg.weight) + self.dec_bias

    def forward(self, x, lib):
        mu_f, lv_f, mu_b, lv_b = self.encode(x)
        zf, zb = self.reparam(mu_f, lv_f), self.reparam(mu_b, lv_b)
        w = self.gene_weights(zf, zb)
        log_mu = torch.log(lib)[:, None] + F.log_softmax(w, dim=1)
        return dict(mu_f=mu_f, lv_f=lv_f, mu_b=mu_b, lv_b=lv_b,
                    zf=zf, zb=zb, log_mu=log_mu)

    def nb_nll(self, x, log_mu):
        theta = torch.exp(self.log_theta).clamp(1e-4, 1e6)
        logits = log_mu - torch.log(theta)[None, :]
        nb = torch.distributions.NegativeBinomial(
            total_count=theta[None, :].expand_as(x), logits=logits
        )
        return -nb.log_prob(x).sum(dim=1).mean()

    @torch.no_grad()
    def latent(self, X, lib, batch=1024):
        zs_f, zs_b = [], []
        self.eval()
        for i in range(0, X.shape[0], batch):
            out = self.forward(X[i:i + batch], lib[i:i + batch])
            zs_f.append(out["mu_f"].cpu())
            zs_b.append(out["mu_b"].cpu())
        return torch.cat(zs_f).numpy(), torch.cat(zs_b).numpy()


@dataclass
class DiscoveryConfig:
    n_fate: int = 8
    epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    hidden: tuple[int, ...] = (256, 64)
    n_bg: int = 16
    beta_f: float = 1e-3
    beta_b: float = 5e-4
    lambda_l1: float = 2e-4      # attention row sparsity
    lambda_ent: float = 5e-3     # attention row entropy (peaky rows)
    lambda_tc: float = 0.1       # decorrelate z_f dims
    lambda_qc: float = 1e-2      # nuisance-regulon down-weighting
    kl_warmup_frac: float = 0.2  # beta warm-up fraction of epochs
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "cpu"
    log_every: int = 25
    history: list = field(default_factory=list)


def train_discovery(
    X: np.ndarray,
    counts: np.ndarray,
    lib: np.ndarray,
    R: torch.Tensor,
    qc_weight: np.ndarray | None = None,
    cfg: DiscoveryConfig | None = None,
    model: DiscoveryFateVAE | None = None,
) -> tuple[DiscoveryFateVAE, list[dict]]:
    """qc_weight: (M,) >= 0 nuisance score per regulon (e.g., max |corr| of
    its activity with log library size / mt fraction); penalizes attention
    onto nuisance regulons. Data-driven - no curation."""
    cfg = cfg or DiscoveryConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dev = torch.device(cfg.device)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    Ct = torch.tensor(counts, dtype=torch.float32, device=dev)
    libt = torch.tensor(lib, dtype=torch.float32, device=dev)
    Rt = R.float().to(dev)
    qc = (torch.tensor(qc_weight, dtype=torch.float32, device=dev)
          if qc_weight is not None else torch.zeros(R.shape[0], device=dev))

    if model is None:
        model = DiscoveryFateVAE(X.shape[1], Rt, n_fate=cfg.n_fate,
                                 n_bg=cfg.n_bg, hidden=tuple(cfg.hidden)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n_train = Xt.shape[0]

    warmup = max(1, int(cfg.epochs * cfg.kl_warmup_frac))
    for e in range(cfg.epochs):
        beta_scale = min(1.0, (e + 1) / warmup)
        model.train()
        perm = torch.randperm(n_train, device=dev)
        tot: dict = {}
        for b0 in range(0, n_train, cfg.batch_size):
            idx = perm[b0:b0 + cfg.batch_size]
            x, l = Xt[idx], libt[idx]
            out = model(x, l)
            attn = model.attention()
            row_norm = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-9)
            row_ent = -(row_norm * row_norm.clamp_min(1e-9).log()).sum(dim=1).mean()
            loss_d = {
                "nb": model.nb_nll(Ct[idx], out["log_mu"]),
                "kl": beta_scale * (cfg.beta_f * gaussian_kl(out["mu_f"], out["lv_f"])
                                    + cfg.beta_b * gaussian_kl(out["mu_b"], out["lv_b"])),
                "l1": cfg.lambda_l1 * attn.sum(),
                "ent": cfg.lambda_ent * row_ent,
                "qc": cfg.lambda_qc * (qc[None, :] * attn).sum() / attn.sum().clamp_min(1e-9) / attn.shape[0],
                "tc": cfg.lambda_tc * _cov_penalty(out["zf"]),
            }
            loss = sum(loss_d.values())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            for k, v in loss_d.items():
                tot[k] = tot.get(k, 0.0) + v.item()
        nb_b = math.ceil(n_train / cfg.batch_size)
        rec = {"epoch": e, **{k: v / nb_b for k, v in tot.items()}}
        cfg.history.append(rec)
        if cfg.log_every and e % cfg.log_every == 0:
            print(f"[discovery {e:4d}] " + " ".join(
                f"{k}={v:.3f}" for k, v in rec.items() if k != "epoch"), flush=True)
    return model, cfg.history


def _cov_penalty(z):
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).mean()


# --- post-hoc analysis helpers (all marker-free) --------------------------
def active_dims(model: DiscoveryFateVAE, zf: np.ndarray,
                z_frac: float = 0.1, attn_frac: float = 0.05) -> np.ndarray:
    """Dims that carry signal: latent variance and attention mass above a
    fraction of the max across dims."""
    zstd = zf.std(axis=0)
    attn = model.attention().detach().numpy()
    amass = attn.sum(axis=1)
    keep = (zstd > z_frac * zstd.max()) & (amass > attn_frac * amass.max())
    return np.where(keep)[0]


def fate_probabilities(zf: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Softmax over standardized active dims."""
    z = zf[:, dims]
    z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-9)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
