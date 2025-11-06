
"""
PRIME: Proxy-based Representation Learning for Imbalanced Regression (PyTorch)

Implements:
- ProxyBank: holds proxy targets y_p and learnable proxy features z_p
- prime_losses: L_proxy and L_align
- PrimeLoss: combines L_reg (user-provided), L_proxy, L_align into L_PRIME
- utility functions and example usage

Paper formulas mapped to code (scalar targets version):
- L_proxy = KL(P || Q) - sum_{i!=j} w_ij * (1 - cos(theta_{ij}))^2
    where P_ij ~ exp(-tau_t * d_t(y_i^p, y_j^p)), Q_ij ~ exp(-tau_f * d_f(z_i^p, z_j^p))
- L_align = - sum_j T_j * log A_j
    where A_j ~ exp(-tau_f * d_f(z, z_j^p)), T_j ~ exp(-tau_t * d_t(y, y_j^p))

Author: ChatGPT (GPT-5 Thinking)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Helper distance functions
# -------------------------

def pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared L2 distances between rows of a and b.
    a: [N, D], b: [M, D]
    returns: [N, M]
    """
    a2 = (a**2).sum(dim=1, keepdim=True)    # [N, 1]
    b2 = (b**2).sum(dim=1, keepdim=True).t()# [1, M]
    ab = a @ b.t()                           # [N, M]
    d2 = a2 + b2 - 2 * ab
    return d2.clamp_min(0.0)


def pairwise_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise L1 distances between rows of a and b.
    returns: [N, M]
    """
    return (a[:, None, :] - b[None, :, :]).abs().sum(dim=-1)


def cosine_similarity_matrix(z: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity for all pairs in z (rows).
    z: [N, D]
    returns: [N, N] cosine similarity in [-1, 1]
    """
    z_norm = F.normalize(z, dim=1, eps=1e-8)
    return z_norm @ z_norm.t()


# -------------------------
# Proxy Bank
# -------------------------

class ProxyBank(nn.Module):
    """
    Holds C proxies:
      - y_p: fixed proxy targets [C, T] (T=1 for scalar)
      - z_p: learnable proxy features [C, F]
    """
    def __init__(
        self,
        y_min: torch.Tensor,
        y_max: torch.Tensor,
        C: int,
        feature_dim: int,
        target_dim: int = 1,
        init_scale: float = 0.02,
        init_method: str = "normal",  # or "uniform"
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert target_dim >= 1, "target_dim must be >= 1"
        self.C = C
        self.feature_dim = feature_dim
        self.target_dim = target_dim

        device = device or y_min.device

        # Build uniformly spaced proxy targets for scalar targets,
        # or linearly spaced per-dim if target_dim > 1 (simple baseline).
        if target_dim == 1:
            y_lin = torch.linspace(y_min.item(), y_max.item(), steps=C, device=device)
            self.y_p = nn.Parameter(y_lin.view(C, 1), requires_grad=False)  # [C,1]
        else:
            mins = y_min.view(1, target_dim)
            maxs = y_max.view(1, target_dim)
            y_lin = torch.linspace(0.0, 1.0, steps=C, device=device).view(C, 1)
            self.y_p = nn.Parameter(mins + y_lin * (maxs - mins), requires_grad=False)  # [C,T]

        # Learnable proxy features
        if init_method == "normal":
            z_init = torch.randn(C, feature_dim, device=device) * init_scale
        else:
            z_init = (torch.rand(C, feature_dim, device=device) - 0.5) * 2 * init_scale
        self.z_p = nn.Parameter(z_init)  # [C,F]

    @torch.no_grad()
    def set_y_p(self, y_p: torch.Tensor):
        """
        Overwrite proxy targets (e.g., from k-means for multi-dim targets).
        y_p: [C, T]
        """
        assert y_p.shape == (self.C, self.target_dim)
        self.y_p.copy_(y_p)


# -------------------------
# PRIME Losses
# -------------------------

@dataclass
class PrimeConfig:
    tau_t: float = 5.0   # temperature for target space
    tau_f: float = 5.0   # temperature for feature space
    alpha: float = 0.1   # weight for cosine spread term in L_proxy
    lambda_p: float = 1.0  # weight for L_proxy
    lambda_a: float = 1.0  # weight for L_align
    target_distance: str = "l1"  # or "l2"
    feature_distance: str = "l2" # or "l1"


class PrimeLoss(nn.Module):
    """
    Computes L_PRIME = L_reg + lambda_p * L_proxy + lambda_a * L_align
    - L_reg must be provided by user (call .forward with reg_loss)
    """
    def __init__(self, proxy_bank: ProxyBank, config: PrimeConfig):
        super().__init__()
        self.pb = proxy_bank
        self.cfg = config

    def _dt(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.cfg.target_distance == "l1":
            return pairwise_l1(a, b)  # [N, M]
        else:
            return pairwise_l2(a, b).sqrt().clamp_min(1e-12)

    def _df(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.cfg.feature_distance == "l1":
            return pairwise_l1(a, b)
        else:
            return pairwise_l2(a, b).sqrt().clamp_min(1e-12)

    def l_proxy(self) -> torch.Tensor:
        """
        Build P and Q over proxies and compute:
        L_proxy = KL(P || Q) - sum_{i!=j} w_ij * (1 - cos(theta_{ij}))^2
        """
        y_p = self.pb.y_p  # [C,T]
        z_p = self.pb.z_p  # [C,F]
        C = y_p.size(0)

        # Pairwise distances
        dt = self._dt(y_p, y_p)  # [C,C]
        df = self._df(z_p, z_p)  # [C,C]

        # Mask out diagonals
        device = y_p.device
        eye = torch.eye(C, device=device, dtype=torch.bool)
        offdiag = ~eye

        # Global normalization over all i!=j as in paper
        P_logits = -self.cfg.tau_t * dt  # [C,C]
        Q_logits = -self.cfg.tau_f * df  # [C,C]

        P_exp = torch.exp(P_logits)[offdiag]  # [C*(C-1)]
        Q_exp = torch.exp(Q_logits)[offdiag]

        P = P_exp / (P_exp.sum() + 1e-12)     # normalized over all off-diag
        Q = Q_exp / (Q_exp.sum() + 1e-12)

        # KL divergence KL(P||Q) over off-diagonal pairs flattened
        kl = (P * (P.add(1e-12).log() - Q.add(1e-12).log())).sum()

        # Cosine spread regularizer: increase cosine distance proportionally to target distance
        cos = cosine_similarity_matrix(z_p)   # [C,C]
        cos_off = cos[offdiag]                # [C*(C-1)]
        # weights w_ij = alpha * d_t(y_i, y_j) on off-diagonal entries
        w = self.cfg.alpha * dt[offdiag]
        spread = (w * (1.0 - cos_off).pow(2)).sum()

        return kl - spread

    def l_align(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        For a batch of sample features z [B,F] and targets y [B, T]:
        Compute A_j and T_j per-sample, then cross-entropy between T and A.
        """
        z_p = self.pb.z_p          # [C,F]
        y_p = self.pb.y_p          # [C,T]
        B = z.size(0)

        # Distances to proxies
        df = self._df(z, z_p)      # [B,C]
        dt = self._dt(y, y_p)      # [B,C]

        # Associations
        A_logits = -self.cfg.tau_f * df
        T_logits = -self.cfg.tau_t * dt

        A = F.softmax(A_logits, dim=1)  # [B,C]
        T = F.softmax(T_logits, dim=1)  # [B,C]

        # Cross-entropy: - sum_j T_j * log A_j, averaged over batch
        loss = -(T * (A.add(1e-12).log())).sum(dim=1).mean()
        return loss

    def forward(
        self,
        reg_loss: torch.Tensor,
        z_batch: Optional[torch.Tensor] = None,
        y_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Combine losses. If z_batch and y_batch are provided, compute L_align.
        Always computes L_proxy (depends only on proxies).
        """
        l_proxy = self.l_proxy()
        if z_batch is not None and y_batch is not None:
            l_align = self.l_align(z_batch, y_batch)
        else:
            l_align = torch.zeros((), device=self.pb.z_p.device)

        total = reg_loss + self.cfg.lambda_p * l_proxy + self.cfg.lambda_a * l_align
        logs = {
            "L_reg": float(reg_loss.detach().cpu()),
            "L_proxy": float(l_proxy.detach().cpu()),
            "L_align": float(l_align.detach().cpu()),
            "L_total": float(total.detach().cpu()),
        }
        return total, logs
