from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .base import SaveLoadModule, get_activation
except Exception:
    from base import SaveLoadModule, get_activation


Tensor = torch.Tensor


class Normalizer:


    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < self.eps, 1.0, std)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None, "Normalizer not fitted."
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform_mean(self, m: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None, "Normalizer not fitted."
        return (m * self.std + self.mean).astype(np.float32)

    def inverse_transform_var(self, v: np.ndarray) -> np.ndarray:

        assert self.std is not None, "Normalizer not fitted."
        return (v * (self.std[None, :] ** 2)).astype(np.float32)


class EnsembleLinear(nn.Module):


    def __init__(self, ensemble_size: int, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.E = int(ensemble_size)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.weight = nn.Parameter(torch.empty(self.E, self.in_dim, self.out_dim))
        self.bias = nn.Parameter(torch.empty(self.E, 1, self.out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for e in range(self.E):
            nn.init.kaiming_uniform_(self.weight[e], a=0.01, nonlinearity="leaky_relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:

        y = torch.bmm(x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class EnsembleMLP(nn.Module):


    def __init__(
        self,
        ensemble_size: int,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: str = "swish",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.E = int(ensemble_size)
        self.activation = get_activation(activation)
        self.dropout = float(dropout)

        dims = [int(in_dim)] + [int(h) for h in hidden_dims] + [int(out_dim)]
        self.layers = nn.ModuleList([EnsembleLinear(self.E, dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        return h


@dataclass
class DynamicsTrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 200
    patience: int = 10
    improve_ratio: float = 0.99
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    verbose: bool = True
    seed: int = 0


def _np_kfold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    return [f.astype(np.int64) for f in folds]


def _p_value_np(score: np.ndarray, calib_sorted: np.ndarray) -> np.ndarray:

    idx = np.searchsorted(calib_sorted, score, side="left")
    n = calib_sorted.size
    ge = (n - idx).astype(np.float32)
    p = (1.0 + ge) / (n + 1.0)
    return np.clip(p, 0.0, 1.0).astype(np.float32)


def p_to_weight(
    p: np.ndarray,
    mode: str = "power",
    power: float = 1.0,
    alpha: float = 0.1,
    temperature: float = 0.05,
) -> np.ndarray:
    mode = (mode or "power").lower()
    if mode == "power":
        return np.power(np.clip(p, 1e-12, 1.0), float(power)).astype(np.float32)
    if mode == "sigmoid":
        t = float(temperature)
        t = 1e-6 if t <= 0 else t
        return (1.0 / (1.0 + np.exp(-(p - float(alpha)) / t))).astype(np.float32)
    if mode == "hard":
        return (p >= float(alpha)).astype(np.float32)
    raise ValueError(f"Unknown weight mode: {mode}")


class CrossConformalEnsembleDynamics(SaveLoadModule):


    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k_folds: int = 5,
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "swish",
        dropout: float = 0.0,
        min_logvar: float = -10.0,
        max_logvar: float = 0.5,
        eps: float = 1e-8,
        score_y_start_idx: int = 0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.K = int(k_folds)
        self.eps = float(eps)
        self.score_y_start_idx = int(score_y_start_idx)

        self.net = EnsembleMLP(
            ensemble_size=self.K,
            in_dim=self.input_dim,
            hidden_dims=hidden_dims,
            out_dim=self.output_dim * 2,
            activation=activation,
            dropout=dropout,
        )


        self.min_logvar = nn.Parameter(torch.ones(1, self.output_dim) * float(min_logvar), requires_grad=True)
        self.max_logvar = nn.Parameter(torch.ones(1, self.output_dim) * float(max_logvar), requires_grad=True)


        self.x_norm = Normalizer()
        self.y_norm = Normalizer()


        self.folds: List[np.ndarray] = []
        self.train_pools: List[np.ndarray] = []
        self.train_subpools: List[np.ndarray] = []
        self.val_pools: List[np.ndarray] = []


        self.calib_sorted_scores: List[np.ndarray] = []


        self._fold_n: Optional[int] = None
        self._fold_seed: Optional[int] = None


        self._val_ratio: Optional[float] = None


        self._optimizer: Optional[torch.optim.Optimizer] = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _bound_logvar(self, logvar: Tensor) -> Tensor:
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return logvar

    def forward_all(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        out = self.net(x)
        mean, logvar = torch.chunk(out, 2, dim=-1)
        logvar = self._bound_logvar(logvar)
        return mean, torch.exp(logvar)

    def _ensure_folds(self, n: int, seed: int) -> None:

        if self.folds and self._fold_n == n:
            return

        self.folds = _np_kfold_indices(n, self.K, seed=seed)
        self._fold_n = n
        self._fold_seed = seed

        all_idx = np.arange(n, dtype=np.int64)
        self.train_pools = []
        for k in range(self.K):
            calib = self.folds[k]
            mask = np.ones(n, dtype=bool)
            mask[calib] = False
            self.train_pools.append(all_idx[mask])


        self.train_subpools = []
        self.val_pools = []
        self._val_ratio = None
        self.calib_sorted_scores = []

    def _ensure_train_holdouts(self, val_ratio: float, seed: int) -> None:

        val_ratio = float(val_ratio)
        if val_ratio <= 0.0:

            self.train_subpools = [tp for tp in self.train_pools]
            self.val_pools = [np.array([], dtype=np.int64) for _ in range(self.K)]
            self._val_ratio = 0.0
            return

        if self.train_subpools and self.val_pools and self._val_ratio == val_ratio:
            return

        rng = np.random.default_rng(seed + 1337)
        self.train_subpools = []
        self.val_pools = []
        for k in range(self.K):
            pool = self.train_pools[k]
            perm = rng.permutation(pool)
            n_val = max(1, int(len(pool) * val_ratio))
            val = perm[:n_val]
            tr = perm[n_val:]

            if tr.size == 0:
                tr = val
                val = np.array([], dtype=np.int64)
            self.train_subpools.append(tr.astype(np.int64))
            self.val_pools.append(val.astype(np.int64))

        self._val_ratio = val_ratio

    def _ensure_optimizer(self, cfg: DynamicsTrainConfig) -> None:

        lr = float(cfg.lr)
        wd = float(cfg.weight_decay)

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        else:
            for g in self._optimizer.param_groups:
                g["lr"] = lr
                g["weight_decay"] = wd


        dev = self.device
        dtype = next(self.parameters()).dtype
        for state in self._optimizer.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    if v.is_floating_point():
                        state[k] = v.to(device=dev, dtype=dtype)
                    else:
                        state[k] = v.to(device=dev)

    def _make_k_head_batch(self, Xn: np.ndarray, Yn: np.ndarray, batch_size: int, rng: np.random.Generator):

        idx = np.empty((self.K, batch_size), dtype=np.int64)
        for k in range(self.K):
            pool = self.train_subpools[k] if self.train_subpools else self.train_pools[k]
            idx[k] = rng.choice(pool, size=batch_size, replace=True)
        xb = Xn[idx]
        yb = Yn[idx]
        return xb, yb

    def _val_loss_per_head(self, Xn: np.ndarray, Yn: np.ndarray, norm: str = "mse") -> np.ndarray:

        self.eval()
        device = self.device
        dtype = next(self.parameters()).dtype
        losses = np.full((self.K,), np.nan, dtype=np.float64)

        with torch.no_grad():
            for k in range(self.K):
                idx = self.val_pools[k] if self.val_pools else np.array([], dtype=np.int64)
                if idx.size == 0:
                    continue
                xk = torch.as_tensor(Xn[idx], device=device, dtype=dtype).unsqueeze(0).repeat(self.K, 1, 1)
                yk = torch.as_tensor(Yn[idx], device=device, dtype=dtype)

                mean, var = self.forward_all(xk)
                mk = mean[k]
                vk = var[k]

                if norm == "nll":
                    inv = 1.0 / (vk + self.eps)
                    l = ((yk - mk) ** 2 * inv + torch.log(vk + self.eps)).mean()
                else:
                    l = ((yk - mk) ** 2).mean()

                losses[k] = float(l.detach().cpu().item())
        return losses

    @torch.no_grad()
    def _nonconformity_scores_head(
        self,
        Xn: np.ndarray,
        Yn: np.ndarray,
        head: int,
        idx: np.ndarray,
        score_norm: str = "l2",
    ) -> np.ndarray:

        self.eval()
        device = self.device
        dtype = next(self.parameters()).dtype

        x = torch.as_tensor(Xn[idx], device=device, dtype=dtype).unsqueeze(0).repeat(self.K, 1, 1)
        y = torch.as_tensor(Yn[idx], device=device, dtype=dtype)

        mean, var = self.forward_all(x)
        mu = mean[head]
        va = var[head]


        if self.score_y_start_idx > 0:
            y = y[:, self.score_y_start_idx:]
            mu = mu[:, self.score_y_start_idx:]
            va = va[:, self.score_y_start_idx:]


        if y.shape[-1] == 0:
            raise ValueError('score_y_start_idx sliced away all Y dimensions; check output_dim / score_y_start_idx.')
        z = (y - mu) / torch.sqrt(va + self.eps)

        if score_norm == "l2":
            s = torch.sqrt((z * z).sum(dim=-1) + self.eps)
        elif score_norm == "l1":
            s = z.abs().sum(dim=-1)
        elif score_norm == "max":
            s = z.abs().max(dim=-1).values
        else:
            raise ValueError(f"Unknown score_norm: {score_norm}")

        return s.detach().cpu().numpy().astype(np.float32)

    def fit_kfold(
        self,
        X: Union[np.ndarray, Tensor],
        Y: Union[np.ndarray, Tensor],
        cfg: DynamicsTrainConfig = DynamicsTrainConfig(),
        loss_type: str = "nll",
        score_norm: str = "l2",
        refit_normalizer: bool = False,
        train_val_ratio: float = 0.1,
    ) -> Dict[str, float]:


        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        n = X.shape[0]
        assert n >= self.K, f"Need at least K={self.K} samples."


        self._ensure_folds(n, seed=cfg.seed)


        self._ensure_train_holdouts(val_ratio=float(train_val_ratio), seed=cfg.seed)


        if refit_normalizer or self.x_norm.mean is None or self.y_norm.mean is None:
            self.x_norm.fit(X)
            self.y_norm.fit(Y)

        Xn = self.x_norm.transform(X)
        Yn = self.y_norm.transform(Y)


        device = torch.device(cfg.device)
        dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
        self.to(device=device, dtype=dtype)


        self._ensure_optimizer(cfg)

        rng = np.random.default_rng(cfg.seed)
        loss_type = (loss_type or "nll").lower()


        best_val = np.full((self.K,), np.inf, dtype=np.float64)
        best_state: Optional[Dict[str, Tensor]] = None
        bad_epochs = 0

        use_early_stop = float(train_val_ratio) > 0.0 and cfg.patience > 0

        for epoch in range(int(cfg.epochs)):
            self.train()


            steps = max(1, int(np.ceil(n / float(cfg.batch_size))))
            for _ in range(steps):
                xb, yb = self._make_k_head_batch(Xn, Yn, cfg.batch_size, rng)
                xt = torch.as_tensor(xb, device=device, dtype=dtype)
                yt = torch.as_tensor(yb, device=device, dtype=dtype)

                mean, var = self.forward_all(xt)

                if loss_type == "nll":
                    inv = 1.0 / (var + self.eps)
                    loss = ((yt - mean) ** 2 * inv + torch.log(var + self.eps)).mean()
                    loss = loss + 1e-3 * (self.max_logvar.sum() - self.min_logvar.sum())
                else:
                    loss = ((yt - mean) ** 2).mean()

                assert self._optimizer is not None
                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._optimizer.step()

            if not use_early_stop:

                if cfg.verbose and (epoch % 50 == 0 or epoch == int(cfg.epochs) - 1):
                    print(f"[CC-DynamicsV1] epoch={epoch:03d} (fixed-epochs)")
                continue


            val_losses = self._val_loss_per_head(Xn, Yn, norm=("nll" if loss_type == "nll" else "mse"))

            finite = np.isfinite(val_losses)
            improved = np.zeros_like(finite, dtype=bool)
            improved[finite] = val_losses[finite] < (best_val[finite] * float(cfg.improve_ratio))

            if improved.any():
                best_val[finite] = np.minimum(best_val[finite], val_losses[finite])
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if cfg.verbose and (epoch % 10 == 0 or bad_epochs == 0):
                msg = f"[CC-DynamicsV1] epoch={epoch:03d} val=" + ", ".join(
                    ["nan" if not np.isfinite(x) else f"{x:.4f}" for x in val_losses.tolist()]
                )
                print(msg)

            if bad_epochs >= int(cfg.patience):
                if cfg.verbose:
                    print(f"[CC-DynamicsV1] early stop at epoch={epoch}, patience={cfg.patience}")
                break

        if best_state is not None and use_early_stop:
            self.load_state_dict(best_state)


        self.calib_sorted_scores = []
        for k in range(self.K):
            idx = self.folds[k]
            s = self._nonconformity_scores_head(Xn, Yn, head=k, idx=idx, score_norm=score_norm)
            self.calib_sorted_scores.append(np.sort(s))

        return {
            "K": float(self.K),
            "fold_n": float(self._fold_n or 0),
            "train_val_ratio": float(train_val_ratio),
            "early_stop": 1.0 if use_early_stop else 0.0,
            "val_loss_mean": float(np.nanmean(best_val)) if use_early_stop else float("nan"),
        }

    @torch.no_grad()
    def predict(
        self,
        X: Union[np.ndarray, Tensor],
        factor_ensemble: bool = False,
        return_numpy: bool = True,
    ) -> Tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]:

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy().astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        Xn = self.x_norm.transform(X_np)
        device = self.device
        dtype = next(self.parameters()).dtype

        xb = torch.as_tensor(Xn, device=device, dtype=dtype).unsqueeze(0).repeat(self.K, 1, 1)
        mean_k, var_k = self.forward_all(xb)

        if factor_ensemble:
            mean = mean_k
            var = var_k
        else:
            mean = mean_k.mean(dim=0)
            var = var_k.mean(dim=0) + (mean_k - mean.unsqueeze(0)).pow(2).mean(dim=0)

        if factor_ensemble:
            mean_np = mean.detach().cpu().numpy().astype(np.float32)
            var_np = var.detach().cpu().numpy().astype(np.float32)
            mean_raw = np.stack([self.y_norm.inverse_transform_mean(mean_np[k]) for k in range(self.K)], axis=0)
            var_raw = np.stack([self.y_norm.inverse_transform_var(var_np[k]) for k in range(self.K)], axis=0)
        else:
            mean_np = mean.detach().cpu().numpy().astype(np.float32)
            var_np = var.detach().cpu().numpy().astype(np.float32)
            mean_raw = self.y_norm.inverse_transform_mean(mean_np)
            var_raw = self.y_norm.inverse_transform_var(var_np)

        if return_numpy:
            return mean_raw, var_raw
        return torch.as_tensor(mean_raw, device=device, dtype=dtype), torch.as_tensor(var_raw, device=device, dtype=dtype)

    @torch.no_grad()
    def p_values(
        self,
        X: Union[np.ndarray, Tensor],
        Y: Union[np.ndarray, Tensor],
        score_norm: str = "l2",
    ) -> np.ndarray:

        assert self.calib_sorted_scores and len(self.calib_sorted_scores) == self.K, "Call fit_kfold() first."

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        Xn = self.x_norm.transform(X)
        Yn = self.y_norm.transform(Y)

        p_acc = np.zeros((X.shape[0],), dtype=np.float32)

        bs = 4096
        for start in range(0, X.shape[0], bs):
            sl = slice(start, min(start + bs, X.shape[0]))
            idx = np.arange(sl.start, sl.stop, dtype=np.int64)

            p_sum = np.zeros((idx.size,), dtype=np.float32)
            for k in range(self.K):

                s = self._nonconformity_scores_head(Xn, Yn, head=k, idx=idx, score_norm=score_norm)

                p_k = _p_value_np(s, self.calib_sorted_scores[k])
                p_sum += p_k
            p_acc[sl] = p_sum / float(self.K)

        return p_acc

    @torch.no_grad()
    def weights(
        self,
        X: Union[np.ndarray, Tensor],
        Y: Union[np.ndarray, Tensor],
        score_norm: str = "l2",
        mode: str = "power",
        power: float = 1.0,
        alpha: float = 0.1,
        temperature: float = 0.05,
    ) -> np.ndarray:
        p = self.p_values(X, Y, score_norm=score_norm)
        return p_to_weight(p, mode=mode, power=power, alpha=alpha, temperature=temperature)
