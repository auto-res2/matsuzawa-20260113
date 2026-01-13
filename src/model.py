"""src/model.py
Complete model architectures implementing CaSE-Reg+-HierPrior and baselines.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaSEAdapter(nn.Module):
    """A lightweight CaSE-style channel adapter.
    It computes a per-sample per-channel gamma from the per-sample global context and
    scales the feature maps accordingly. It also emits a small reg_loss composed of
    anchor, variance and KL terms to regularize gamma.
    """

    def __init__(self, channels: int, reg_anchor: float = 0.0, reg_var: float = 0.0, reg_kl: float = 0.0, init_scale: float = 1.0):
        super(CaSEAdapter, self).__init__()
        self.C = channels
        self.reg_anchor = reg_anchor
        self.reg_var = reg_var
        self.reg_kl = reg_kl
        hidden = max(16, channels // 2)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        # Learnable prior parameter for KL term
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [N, C, H, W]
        N, C, H, W = x.shape
        # Global context per-sample per-channel
        ctx = x.mean(dim=(2, 3))  # [N, C]
        g = self.fc(ctx)  # [N, C], in (0,1)
        gamma = 0.5 + g  # map to [0.5, 1.5]
        x_scaled = x * gamma.view(N, C, 1, 1)
        # Regularization losses
        anchor_loss = self.reg_anchor * torch.mean((gamma - 1.0) ** 2)
        var_loss = self.reg_var * torch.var(gamma, dim=0).mean()
        mu = gamma.mean(dim=0)
        sigma_p = torch.exp(self.log_sigma)
        kl_loss = 0.5 * torch.sum(((mu - 1.0) ** 2) / (sigma_p ** 2) + torch.log(sigma_p ** 2) - 1)
        kl_loss = self.reg_kl * kl_loss / max(self.C, 1)
        reg_loss = anchor_loss + var_loss + kl_loss
        return x_scaled, reg_loss


class CaSERegBackbone(nn.Module):
    """A compact backbone (CNN-esque) with 2 blocks and CaSE adapters inserted.
    This serves as a lightweight, publication-friendly backbone for the CaSE-Reg+-HierPrior
    method.
    """

    def __init__(self, in_channels: int, num_classes: int, reg_anchor: float = 0.0, reg_var: float = 0.0, reg_kl: float = 0.0, num_caose_blocks: int = 2):
        super(CaSERegBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.num_caose_blocks = max(1, min(2, int(num_caose_blocks)))
        self.caose1 = CaSEAdapter(32, reg_anchor=reg_anchor, reg_var=reg_var, reg_kl=reg_kl)
        self.caose2 = CaSEAdapter(64, reg_anchor=reg_anchor, reg_var=reg_var, reg_kl=reg_kl)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_total = 0.0
        x = self.relu(self.bn1(self.conv1(x)))  # A: [N,32,H,W]
        if self.num_caose_blocks >= 1:
            x, reg1 = self.caose1(x)
            reg_total = reg_total + reg1
        x = self.relu(self.bn2(self.conv2(x)))  # B: [N,64,H,W]
        if self.num_caose_blocks >= 2:
            x, reg2 = self.caose2(x)
            reg_total = reg_total + reg2
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits, reg_total


class CaSE_RegPlus_Model(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, reg_anchor: float = 0.0, reg_var: float = 0.0, reg_kl: float = 0.0, num_caose_blocks: int = 2):
        super(CaSE_RegPlus_Model, self).__init__()
        self.backbone = CaSERegBackbone(in_channels, num_classes, reg_anchor, reg_var, reg_kl, num_caose_blocks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(x)


def _parse_num_blocks(v: object) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str) and '-' in v:
        a, b = v.split('-')
        try:
            return int(b)
        except Exception:
            return 2
    try:
        return int(v)
    except Exception:
        return 2


def build_model(cfg: dict, dataset_info: dict) -> nn.Module:
    name = cfg.get("name", "cae-regplus")
    in_channels = 3
    num_classes = int(dataset_info.get("num_classes", 10))
    reg_anchor = float(cfg.get("reg_anchor", 0.0))
    reg_var = float(cfg.get("reg_var", 0.0))
    reg_kl = float(cfg.get("reg_kl", 0.0))
    num_caose_blocks = _parse_num_blocks(cfg.get("num_caose_blocks", 2))

    model = CaSE_RegPlus_Model(in_channels, num_classes, reg_anchor=reg_anchor, reg_var=reg_var, reg_kl=reg_kl, num_caose_blocks=num_caose_blocks)
    return model

__all__ = ["CaSEAdapter", "CaSE_RegPlus_Model", "CaSERegBackbone", "build_model"]
