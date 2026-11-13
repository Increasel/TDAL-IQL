import torch
from torch import nn
from .ensemble import LinearEnsemble

def weight_init(m, gain=1):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            m.bias.data.fill_(0.)
    if isinstance(m, LinearEnsemble):
        for i in range(m.ensemble_size):
            nn.init.orthogonal_(m.weight.data[i], gain=gain)
        if m.bias is not None:
            m.bias.data.fill_(0.)
