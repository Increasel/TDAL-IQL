from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
TensorBatch = List[torch.Tensor]
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

def mlp(in_dim: int, out_dim: int, hidden=(256, 256), act=nn.ReLU) -> nn.Module:
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

class CriticQ(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(s_dim + a_dim, 1, hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))

class ValueV(nn.Module):
    def __init__(self, s_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(s_dim, 1, hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

class GaussianPolicy(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden=(256, 256), max_action: float = 1.0):
        super().__init__()
        self.max_action = float(max_action)
        self.net = mlp(s_dim, a_dim * 2, hidden)

    def forward(self, s: torch.Tensor):
        mu_logstd = self.net(s)
        a_dim = mu_logstd.shape[-1] // 2
        mu = mu_logstd[..., :a_dim]
        log_std = mu_logstd[..., a_dim:]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    @torch.no_grad()
    def act(self, s: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if s.ndim == 1:
            s = s[None, :]
        device = next(self.parameters()).device
        st = torch.from_numpy(s.astype(np.float32)).to(device)
        mu, std = self.forward(st)
        if deterministic:
            a = torch.tanh(mu)
        else:
            eps = torch.randn_like(mu)
            a = torch.tanh(mu + std * eps)
        return (a * self.max_action).cpu().numpy()[0]

    def log_prob_of_data_action(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        a_scaled = torch.clamp(a / self.max_action, -1 + eps, 1 - eps)
        u = 0.5 * (torch.log1p(a_scaled) - torch.log1p(-a_scaled))
        mu, std = self.forward(s)
        logp_u = (
            -0.5 * (((u - mu) / (std + 1e-8)) ** 2)
            - torch.log(std + 1e-8)
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1, keepdim=True)
        log_det = torch.log(self.max_action * (1 - torch.tanh(u) ** 2) + 1e-8).sum(dim=-1, keepdim=True)
        return logp_u - log_det

class DarcDiscriminator(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, input_noise: float = 1.0):
        super().__init__()
        self.input_noise = float(input_noise)
        self.sa = nn.Sequential(nn.Linear(s_dim + a_dim, 32), nn.ReLU(), nn.Linear(32, 2))
        self.adv = nn.Sequential(nn.Linear(2 * s_dim + a_dim, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, s: torch.Tensor, a: torch.Tensor, s2: torch.Tensor, training: bool = True):
        sa_in = torch.cat([s, a], dim=-1)
        sas_in = torch.cat([s, a, s2], dim=-1)
        if training and self.input_noise > 0:
            sa_in = sa_in + torch.randn_like(sa_in) * self.input_noise
            sas_in = sas_in + torch.randn_like(sas_in) * self.input_noise
        sa_logits = self.sa(sa_in)
        adv_logits = self.adv(sas_in)
        sas_logits = sa_logits + adv_logits
        return sa_logits, sas_logits

    @torch.no_grad()
    def probs(self, s: torch.Tensor, a: torch.Tensor, s2: torch.Tensor):
        sa_logits, sas_logits = self.forward(s, a, s2, training=False)
        sa_probs = torch.softmax(sa_logits, dim=-1)
        sas_probs = torch.softmax(sas_logits, dim=-1)
        return sa_probs, sas_probs

def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    w = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (w * diff.pow(2)).mean()
