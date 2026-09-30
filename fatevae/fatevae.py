"""FateVAE: two-branch disentangled negative-binomial VAE (proposal §6.2).

Latent factorization
--------------------
    z = (z_f, z_b)
    z_f : fate latent, one dimension per fate module (default), anchored by
          (i) a group-sparse decoder - fate dim k may only decode genes in
          fate module k's extended gene set (markers + SCENIC regulon
          targets of marker TFs), and
          (ii) weak supervision - a linear fate head trained against soft
          marker-module scores (train gene split only).
    z_b : background latent, dense decoder, absorbs residual variation; an
          adversarial discriminator (gradient reversal) prevents fate
          information from leaking into z_b.

Likelihood: library-size-conditioned negative binomial with softmax gene
allocation (scVI convention: mean = library * softmax(gene weights)), per-gene
inverse dispersion theta.

Training schedule (proposal §6.2.3): stage A pretrain (NB + KL), stage B adds
supervision, stage C adds the adversary. All hyperparameters of the schedule
are constructor arguments so ablations can switch terms off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- utils
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


def gaussian_kl(mu, logvar):
    return 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1))


# ------------------------------------------------------------------ module
class FateVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        fate_mask: torch.Tensor,
        n_bg: int = 16,
        hidden: tuple[int, ...] = (512, 128),
        n_extra_fate_dims: int = 0,
        dispersion: float = 1.0,
    ):
        """fate_mask: (n_fate, n_genes) binary - fate dim k decodes genes
        where fate_mask[k] == 1. Built by ``build_fate_mask``."""
        super().__init__()
        self.n_genes = n_genes
        self.n_fate = fate_mask.shape[0]
        self.n_extra = n_extra_fate_dims
        self.n_zf = self.n_fate + self.n_extra
        self.n_zb = n_bg
        self.register_buffer("fate_mask", fate_mask.float())

        dims = [n_genes, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
        self.encoder = nn.Sequential(*layers)

        self.enc_mu = nn.Linear(hidden[-1], self.n_zf + self.n_zb)
        self.enc_logvar = nn.Linear(hidden[-1], self.n_zf + self.n_zb)

        # group-sparse fate decoder: extra fate dims (if any) get a full mask
        full_mask = torch.cat(
            [fate_mask, torch.ones(self.n_extra, n_genes, device=fate_mask.device)], dim=0
        )  # (n_zf, n_genes)
        self.dec_fate = nn.Linear(self.n_zf, n_genes, bias=False)
        with torch.no_grad():
            self.dec_fate.weight.mul_(full_mask.T)  # weight is (n_genes, n_zf)
        self.register_buffer("_fate_full_mask", full_mask)
        self.dec_bias = nn.Parameter(torch.zeros(n_genes))
        self.dec_bg = nn.Linear(self.n_zb, n_genes, bias=False)

        self.fate_head = nn.Linear(self.n_zf, self.n_fate, bias=True)
        self.disc = nn.Sequential(  # adversary on z_b
            nn.Linear(self.n_zb, 64), nn.SiLU(), nn.Linear(64, self.n_fate)
        )
        log_theta = torch.full((n_genes,), float(np.log(dispersion)))
        self.log_theta = nn.Parameter(log_theta)

    # -- encoder ----------------------------------------------------------
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.enc_mu(h), self.enc_logvar(h)
        return mu[:, : self.n_zf], logvar[:, : self.n_zf], \
            mu[:, self.n_zf:], logvar[:, self.n_zf:]

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # -- decoder ----------------------------------------------------------
    def gene_weights(self, zf, zb):
        w_f = F.linear(zf, self.dec_fate.weight * self._fate_full_mask.T)
        return w_f + F.linear(zb, self.dec_bg.weight) + self.dec_bias

    def forward(self, x, lib):
        mu_f, lv_f, mu_b, lv_b = self.encode(x)
        zf, zb = self.reparam(mu_f, lv_f), self.reparam(mu_b, lv_b)
        w = self.gene_weights(zf, zb)
        log_mu = torch.log(lib)[:, None] + F.log_softmax(w, dim=1)
        fate_logits = self.fate_head(zf)
        return dict(
            mu_f=mu_f, lv_f=lv_f, mu_b=mu_b, lv_b=lv_b,
            zf=zf, zb=zb, log_mu=log_mu, fate_logits=fate_logits,
        )

    def nb_nll(self, x, log_mu):
        theta = torch.exp(self.log_theta).clamp(1e-4, 1e6)
        logits = log_mu - torch.log(theta)[None, :]
        nb = torch.distributions.NegativeBinomial(
            total_count=theta[None, :].expand_as(x), logits=logits
        )
        return -nb.log_prob(x).sum(dim=1).mean()

    # -- inference helpers --------------------------------------------------
    @torch.no_grad()
    def latent(self, X, lib, batch=1024):
        zs_f, zs_b, logits = [], [], []
        self.eval()
        for i in range(0, X.shape[0], batch):
            out = self.forward(X[i:i + batch], lib[i:i + batch])
            zs_f.append(out["mu_f"].cpu())
            zs_b.append(out["mu_b"].cpu())
            logits.append(out["fate_logits"].cpu())
        return (torch.cat(zs_f).numpy(), torch.cat(zs_b).numpy(),
                torch.cat(logits).numpy())

    @torch.no_grad()
    def expected_expression(self, X, lib, batch=1024):
        """Decoder mean (cells x genes) - used for held-out gene recovery."""
        outs = []
        self.eval()
        for i in range(0, X.shape[0], batch):
            out = self.forward(X[i:i + batch], lib[i:i + batch])
            outs.append(out["log_mu"].exp().cpu())
        return torch.cat(outs).numpy()


def build_fate_mask(
    gene_names: list[str],
    fate_genesets: dict[str, list[str]],
    n_extra: int = 0,
) -> torch.Tensor:
    """(n_fate, n_genes) binary mask; genes not in any set stay connected to
    no fate dimension (background-only)."""
    order = list(fate_genesets)
    gidx = {g: i for i, g in enumerate(gene_names)}
    mask = torch.zeros(len(order), len(gene_names))
    for k, fate in enumerate(order):
        for g in fate_genesets[fate]:
            if g in gidx:
                mask[k, gidx[g]] = 1.0
    if mask.sum(1).min() < 2:
        raise ValueError("A fate dimension has <2 genes; check gene name matching")
    return mask


# ------------------------------------------------------------------ trainer
@dataclass
class TrainConfig:
    epochs_pretrain: int = 150
    epochs_sup: int = 200
    epochs_adv: int = 150
    batch_size: int = 256
    lr: float = 1e-3
    hidden: tuple[int, ...] = (256, 64)
    n_bg: int = 16
    beta_f: float = 1e-3      # KL weight fate branch (kept small: fate dims supervised)
    beta_b: float = 5e-4      # KL weight background branch
    lambda_sup: float = 5.0   # supervision weight (ramped within stage B)
    lambda_adv: float = 0.3   # adversary weight (ramped within stage C)
    lambda_tc: float = 0.1    # off-diagonal covariance penalty on z_f
    sup_temperature: float = 0.5  # soft-label softmax temperature
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "cpu"
    log_every: int = 25
    history: list = field(default_factory=list)


def _soft_targets(scores: np.ndarray, temperature: float) -> torch.Tensor:
    """Marker module scores (cells x fates) -> soft probability targets."""
    s = scores / max(temperature, 1e-6)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return torch.tensor(e / e.sum(axis=1, keepdims=True), dtype=torch.float32)


def train_fatevae(
    X: np.ndarray,
    counts: np.ndarray,
    lib: np.ndarray,
    fate_mask: torch.Tensor,
    soft_scores: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    cfg: TrainConfig | None = None,
    model: FateVAE | None = None,
) -> tuple[FateVAE, list[dict]]:
    """Train FateVAE; returns (model, history).

    X: log-normalized expression (dense, cells x genes) - encoder input
    counts: raw integer counts (same shape) - NB likelihood target
    lib: library size per cell (raw counts sum)
    soft_scores: per-cell marker module scores (train gene split only)
    train_mask/val_mask: boolean cell masks
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dev = torch.device(cfg.device)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    Ct = torch.tensor(counts, dtype=torch.float32, device=dev)
    libt = torch.tensor(lib, dtype=torch.float32, device=dev)
    Yt = _soft_targets(soft_scores, cfg.sup_temperature).to(dev)
    n = Xt.shape[0]
    train_idx = torch.tensor(np.where(train_mask)[0], device=dev)
    val_idx = torch.tensor(np.where(val_mask)[0], device=dev)

    if model is None:
        model = FateVAE(X.shape[1], fate_mask.to(dev), n_bg=cfg.n_bg,
                        hidden=tuple(cfg.hidden)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def _epoch(stage: str, epoch: int, prog: float):
        model.train()
        perm = torch.randperm(len(train_idx), device=dev)
        tot = {}
        for b0 in range(0, len(perm), cfg.batch_size):
            idx = train_idx[perm[b0:b0 + cfg.batch_size]]
            x, y, l = Xt[idx], Yt[idx], libt[idx]
            out = model(x, l)
            loss_d = {}
            loss_d["nb"] = model.nb_nll(Ct[idx], out["log_mu"])
            loss_d["kl"] = cfg.beta_f * gaussian_kl(out["mu_f"], out["lv_f"]) \
                + cfg.beta_b * gaussian_kl(out["mu_b"], out["lv_b"])
            zf = out["zf"]
            if stage in ("sup", "adv"):
                lam = cfg.lambda_sup * prog
                loss_d["sup"] = lam * _soft_ce(out["fate_logits"], y)
                if cfg.lambda_tc > 0:
                    loss_d["tc"] = cfg.lambda_tc * _cov_penalty(zf)
            if stage == "adv":
                lam_adv = cfg.lambda_adv * prog
                adv_logits = model.disc(grad_reverse(out["zb"], lam_adv))
                loss_d["adv"] = F.cross_entropy(
                    adv_logits, y.argmax(dim=1), label_smoothing=0.1
                )
            loss = sum(loss_d.values())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            for k, v in loss_d.items():
                tot[k] = tot.get(k, 0.0) + v.item()
        nb_batches = math.ceil(len(perm) / cfg.batch_size)
        rec = {"stage": stage, "epoch": epoch,
               **{k: v / nb_batches for k, v in tot.items()}}
        # quick val reconstruction loss for monitoring
        with torch.no_grad():
            model.eval()
            out = model(Xt[val_idx], libt[val_idx])
            rec["val_nb"] = model.nb_nll(Ct[val_idx], out["log_mu"]).item()
        cfg.history.append(rec)
        if cfg.log_every and epoch % cfg.log_every == 0:
            print(f"[{stage} {epoch:4d}] " + " ".join(
                f"{k}={v:.3f}" for k, v in rec.items() if k not in ("stage", "epoch")
            ), flush=True)
        return rec

    schedule = [("pre", cfg.epochs_pretrain), ("sup", cfg.epochs_sup),
                ("adv", cfg.epochs_adv)]
    for stage, n_ep in schedule:
        for e in range(n_ep):
            _epoch(stage, e, prog=(e + 1) / n_ep)
    return model, cfg.history


def _soft_ce(logits, soft_targets):
    return torch.mean(torch.sum(-soft_targets * F.log_softmax(logits, dim=1), dim=1))


def _cov_penalty(z):
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).mean()
