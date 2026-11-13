from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


Tensor = torch.Tensor


class Swish(nn.Module):


    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


def get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return Swish()
    raise ValueError(f"Unknown activation: {name}")


def init_weights_(
    module: nn.Module,
    init: str = "xavier_uniform",
    gain: float = 1.0,
) -> None:

    init = (init or "xavier_uniform").lower()

    if isinstance(module, nn.Linear):
        w = module.weight
        if init == "xavier_uniform":
            nn.init.xavier_uniform_(w, gain=gain)
        elif init == "xavier_normal":
            nn.init.xavier_normal_(w, gain=gain)
        elif init == "kaiming_uniform":
            nn.init.kaiming_uniform_(w, a=0.01, nonlinearity="leaky_relu")
        elif init == "kaiming_normal":
            nn.init.kaiming_normal_(w, a=0.01, nonlinearity="leaky_relu")
        elif init == "orthogonal":
            nn.init.orthogonal_(w, gain=gain)
        elif init == "trunc_normal":

            if hasattr(nn.init, "trunc_normal_"):
                nn.init.trunc_normal_(w, std=0.02)
            else:
                nn.init.normal_(w, std=0.02)
        else:
            raise ValueError(f"Unknown init method: {init}")

        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str = "relu",
    out_activation: Optional[str] = None,
    dropout: float = 0.0,
    layer_norm: bool = False,
    init: str = "xavier_uniform",
) -> nn.Sequential:

    layers: List[nn.Module] = []
    last = in_dim
    act = get_activation(activation)

    for h in hidden_dims:
        lin = nn.Linear(last, h)
        init_weights_(lin, init=init)
        layers.append(lin)
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act)
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        last = h

    lin_out = nn.Linear(last, out_dim)
    init_weights_(lin_out, init=init)
    layers.append(lin_out)


    if out_activation is not None:
        layers.append(get_activation(out_activation))

    return nn.Sequential(*layers)


class SaveLoadModule(nn.Module):


    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd)
