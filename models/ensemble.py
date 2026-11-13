import math
from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int] = [256, 256],
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        output_act: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(nn.Linear(last_dim, dim))
            if dropout > 0.0:
                net.append(nn.Dropout(dropout))
            if normalization is not None:
                net.append(normalization(dim))
            net.append(act())
            last_dim = dim
        net.append(nn.Linear(last_dim, output_dim))
        if output_act is not None:
            net.append(output_act())
        self.net = nn.Sequential(*net)
        self._has_output_act = False if output_act is None else True

    @property
    def last_layer(self) -> nn.Module:
        if self._has_output_act:
            return self.net[-2]
        else:
            return self.net[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearEnsemble(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 3,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty((ensemble_size, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((ensemble_size, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:


        std = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 2:
            input = input.repeat(self.ensemble_size, 1, 1)
        elif len(input.shape) > 3:
            raise ValueError("LinearEnsemble layer does not support inputs with more than 3 dimensions.")
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "ensemble_size={}, in_features={}, out_features={}, bias={}".format(
            self.ensemble_size, self.in_features, self.out_features, self.bias is not None
        )


class LayerNormEnsemble(nn.Module):


    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: int,
        ensemble_size: int = 3,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert isinstance(normalized_shape, int), "Currently EnsembleLayerNorm only supports final dim int shapes."
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.ensemble_size = ensemble_size
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty((self.ensemble_size, 1) + self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty((self.ensemble_size, 1) + self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.repeat(self.ensemble_size, 1, 1)
        elif len(x.shape) > 3:
            raise ValueError("LayerNormEnsemble layer does not support inputs with more than 3 dimensions.")
        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int = 3,
        hidden_layers: List[int] = [256, 256],
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        output_act: Optional[Type[nn.Module]] = None,
    ):

        super().__init__()

        assert normalization is None or normalization is LayerNormEnsemble, "Ensemble only support EnsembleLayerNorm"
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(LinearEnsemble(last_dim, dim, ensemble_size=ensemble_size))
            if dropout > 0.0:
                net.append(nn.Dropout(dropout))
            if normalization is not None:
                net.append(normalization(dim, ensemble_size=ensemble_size))
            net.append(act())
            last_dim = dim
        net.append(LinearEnsemble(last_dim, output_dim, ensemble_size=ensemble_size))
        if output_act is not None:
            net.append(output_act())
        self.net = nn.Sequential(*net)
        self._has_output_act = False if output_act is None else True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def last_layer(self) -> torch.Tensor:
        if self._has_output_act:
            return self.net[-2]
        else:
            return self.net[-1]
