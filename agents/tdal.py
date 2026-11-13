from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
CDC_GATE_SCOPE = "source-minibatch-quantile-v1"


def compute_cdc_batch_gate(d_cdc: Tensor, retention_ratio: float) -> Tensor:

    if d_cdc.ndim != 2 or d_cdc.shape[1] != 1 or d_cdc.shape[0] == 0:
        raise ValueError("CDC discrepancies must have nonempty shape (batch_size, 1)")
    ratio = float(np.clip(retention_ratio, 0.0, 1.0))
    if not np.isfinite(ratio):
        raise ValueError("CDC retention ratio must be finite")

    if ratio == 0.0:
        return torch.zeros_like(d_cdc)
    if ratio == 1.0:
        return torch.ones_like(d_cdc)
    threshold = torch.quantile(d_cdc.reshape(-1), ratio)
    return (d_cdc <= threshold).to(d_cdc.dtype)


def compute_cvc_soft_weights(d_cvc: np.ndarray, eta_v: float) -> np.ndarray:

    eta_v = float(eta_v)
    if eta_v < 0.0:
        raise ValueError("eta_v must be non-negative.")
    return np.exp(-eta_v * np.asarray(d_cvc, dtype=np.float32)).astype(np.float32)


from offline.array_buffer import ArrayReplayBuffer as OfflineReplayBuffer


def mlp(
    in_dim: int,
    out_dim: int,
    hidden: Tuple[int, ...],
    act=nn.ReLU,
    use_layernorm: bool = True,
) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class SquashedGaussianPolicy(nn.Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        hidden: Tuple[int, ...] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.net = mlp(s_dim, a_dim * 2, hidden, act=nn.ReLU)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mix = self.net(s)
        mean, log_std = torch.chunk(mix, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std, log_std

    def sample(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        mean, std, _ = self.forward(s)
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u)
        logp = dist.log_prob(u).sum(dim=-1, keepdim=True)
        correction = torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return a, logp - correction

    def log_prob(self, s: Tensor, a: Tensor) -> Tensor:
        mean, std, _ = self.forward(s)
        dist = torch.distributions.Normal(mean, std)
        a = a.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True)
        correction = torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return logp_u - correction

    @torch.no_grad()
    def act(self, s: np.ndarray, deterministic: bool = False) -> np.ndarray:
        st = torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(next(self.parameters()).device)
        mean, std, _ = self.forward(st)
        u = mean if deterministic else mean + std * torch.randn_like(std)
        a = torch.tanh(u)
        return a.squeeze(0).cpu().numpy()


class QNetwork(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.net = mlp(s_dim + a_dim, 1, hidden, act=nn.ReLU)

    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class TwinQ(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.q1 = QNetwork(s_dim, a_dim, hidden)
        self.q2 = QNetwork(s_dim, a_dim, hidden)

    def forward(self, s: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:
        return self.q1(s, a), self.q2(s, a)


class VNetwork(nn.Module):
    def __init__(self, s_dim: int, hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.net = mlp(s_dim, 1, hidden, act=nn.ReLU)

    def forward(self, s: Tensor) -> Tensor:
        return self.net(s)


def expectile_loss(diff: Tensor, expectile: float) -> Tensor:
    tau = float(expectile)
    w = torch.where(diff > 0, tau, 1.0 - tau)
    return w * diff.pow(2)


class StateEncoder(nn.Module):
    def __init__(self, s_dim: int, latent_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        self.net = mlp(s_dim, latent_dim, hidden, act=nn.ReLU)

    def forward(self, s: Tensor) -> Tensor:
        z = self.net(s)
        return F.normalize(z, dim=-1)


class StateActionEncoder(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, latent_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        self.net = mlp(s_dim + a_dim, latent_dim, hidden, act=nn.ReLU)

    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        z = self.net(torch.cat([s, a], dim=-1))
        return F.normalize(z, dim=-1)


class EnsembleGaussianDynamics(nn.Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        latent_dim: int,
        ensemble_size: int,
        hidden: Tuple[int, ...] = (256, 256, 256),
    ):
        super().__init__()
        self.s_dim = int(s_dim)
        self.ensemble_size = int(ensemble_size)
        self.models = nn.ModuleList(
            [mlp(latent_dim * 2, 2 * s_dim, hidden, act=nn.ReLU) for _ in range(self.ensemble_size)]
        )

    def forward(self, z_sa: Tensor, z_s: Tensor) -> Tuple[Tensor, Tensor]:
        inp = torch.cat([z_sa, z_s], dim=-1)
        means, logvars = [], []
        for m in self.models:
            out = m(inp)
            mean, logvar = torch.chunk(out, 2, dim=-1)
            logvar = torch.clamp(logvar, -10.0, 1.0)
            means.append(mean)
            logvars.append(logvar)
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)

    def predict_next_states(self, z_sa: Tensor, z_s: Tensor, s: Tensor) -> Tensor:
        means, _ = self.forward(z_sa, z_s)
        return s.unsqueeze(0) + means


@dataclass
class AgentConfig:
    s_dim: int
    a_dim: int
    buffer_capacity: int = 2_000_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    training_delay: int = 1


    alpha: Optional[float] = 0.05
    target_entropy: Optional[float] = None
    algo: str = "iql"
    cql_alpha: float = 1.0
    cql_n_actions: int = 10
    bc_coef: float = 0.0


    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    iql_max_weight: float = 100.0


    k_folds: int = 5
    dyn_hidden: Tuple[int, ...] = (256, 256, 256)
    dyn_epochs: int = 200
    dyn_batch_size: int = 256
    dyn_lr: float = 1e-3
    dyn_patience: int = 10
    dyn_train_val_ratio: float = 0.1
    cache_src_weights: bool = True


    weight_mode: str = "power"
    weight_power: float = 1.0
    weight_alpha: float = 0.1
    weight_temperature: float = 0.05


    xi: float = 0.75
    cdc_gate_scope: str = CDC_GATE_SCOPE
    latent_dim: int = 128
    contrastive_coef: float = 1.0
    dyn_nll_coef: float = 1.0

    value_temperature: float = 1.0
    dynamics_margin: float = 0.0


    static_value_steps: int = 20_000
    static_value_batch_size: int = 256
    static_value_lr: float = 3e-4
    static_value_tau: float = 0.005
    static_value_expectile: float = 0.7


    pseudo_stage_targets: Tuple[Tuple[float, ...], ...] = field(default_factory=tuple)
    pseudo_stage_pos_dims: Tuple[int, ...] = ()
    pseudo_stage_hit_radius: float = 0.05
    pseudo_stage_use_next_state: bool = True


class CCPWeightSACAgent:


    SOURCE_WEIGHT_REFERENCE = "frozen_target_v_successor"
    SOURCE_WEIGHT_FORMULA = "exp(-eta_v*d_cvc)"

    def __init__(self, cfg: AgentConfig):
        if cfg.cdc_gate_scope != CDC_GATE_SCOPE:
            raise ValueError(f"Unsupported CDC gate scope {cfg.cdc_gate_scope}; expected {CDC_GATE_SCOPE}")
        self.cfg = cfg
        self.device = cfg.device


        self.policy = SquashedGaussianPolicy(cfg.s_dim, cfg.a_dim).to(self.device)
        self.q = TwinQ(cfg.s_dim, cfg.a_dim).to(self.device)
        self.q_targ = TwinQ(cfg.s_dim, cfg.a_dim).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.v = VNetwork(cfg.s_dim).to(self.device)

        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.optim_q = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.optim_v = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)


        enc_hidden = cfg.dyn_hidden[:-1] if len(cfg.dyn_hidden) >= 2 else (256, 256)
        self.state_encoder = StateEncoder(cfg.s_dim, cfg.latent_dim, hidden=enc_hidden).to(self.device)
        self.sa_encoder = StateActionEncoder(cfg.s_dim, cfg.a_dim, cfg.latent_dim, hidden=enc_hidden).to(self.device)
        self.dynamics = EnsembleGaussianDynamics(
            s_dim=cfg.s_dim,
            a_dim=cfg.a_dim,
            latent_dim=cfg.latent_dim,
            ensemble_size=cfg.k_folds,
            hidden=cfg.dyn_hidden,
        ).to(self.device)
        self.optim_dynamics = torch.optim.Adam(
            list(self.state_encoder.parameters())
            + list(self.sa_encoder.parameters())
            + list(self.dynamics.parameters()),
            lr=cfg.dyn_lr,
        )


        self.static_q = TwinQ(cfg.s_dim, cfg.a_dim).to(self.device)
        self.static_q_targ = TwinQ(cfg.s_dim, cfg.a_dim).to(self.device)
        self.static_q_targ.load_state_dict(self.static_q.state_dict())
        self.static_v = VNetwork(cfg.s_dim).to(self.device)
        self.optim_static_q = torch.optim.Adam(self.static_q.parameters(), lr=cfg.static_value_lr)
        self.optim_static_v = torch.optim.Adam(self.static_v.parameters(), lr=cfg.static_value_lr)

        self.src_buffer = OfflineReplayBuffer(cfg.s_dim, cfg.a_dim, capacity=cfg.buffer_capacity, device=self.device)
        self.tar_buffer = OfflineReplayBuffer(cfg.s_dim, cfg.a_dim, capacity=cfg.buffer_capacity, device=self.device)

        self.training_count = 0
        self.loss_log: Dict[str, float] = {}
        self._dyn_ready = False
        self._static_v_ready = False
        self._static_src_ready = False


        self._src_d_dyn_cache: Optional[np.ndarray] = None
        self._src_d_val_cache: Optional[np.ndarray] = None
        self._src_stage_cache: Optional[np.ndarray] = None


    def load_source_dataset(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, done: np.ndarray, s2: np.ndarray) -> None:
        self._static_src_ready = False
        self._src_d_dyn_cache = None
        self._src_d_val_cache = None
        self._src_stage_cache = None
        self.src_buffer.load_from_arrays(s, a, r, done, s2)

    def load_target_dataset(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, done: np.ndarray, s2: np.ndarray) -> None:
        self.tar_buffer.load_from_arrays(s, a, r, done, s2)


    def _sample_target_batch(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        s, a, _, _, s2, _ = self.tar_buffer.sample(batch_size)
        return s, a, s2

    def _sample_negative_next_states(self, batch_size: int) -> Tensor:
        if len(self.src_buffer) >= batch_size:
            idx = np.random.randint(0, len(self.src_buffer), size=int(batch_size))
            arr = torch.from_numpy(self.src_buffer.s2[idx]).to(self.device)
            return arr
        _, _, _, _, s2, _ = self.tar_buffer.sample(batch_size)
        perm = torch.randperm(s2.shape[0], device=s2.device)
        return s2[perm]

    def _contrastive_logits(self, z_sa: Tensor, z_pos: Tensor, z_neg: Tensor) -> Tuple[Tensor, Tensor]:
        pos = torch.sum(z_sa * z_pos, dim=-1, keepdim=True)
        neg = torch.sum(z_sa * z_neg, dim=-1, keepdim=True)
        logits = torch.cat([pos, neg], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return logits, labels

    def train_dynamics(self, seed: int = 0) -> None:
        if len(self.tar_buffer) < self.cfg.dyn_batch_size:
            raise RuntimeError("Target buffer too small for dynamics training.")

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        best_loss = float("inf")
        bad_epochs = 0

        for epoch in range(int(self.cfg.dyn_epochs)):
            s, a, s2 = self._sample_target_batch(self.cfg.dyn_batch_size)
            s_neg = self._sample_negative_next_states(self.cfg.dyn_batch_size)

            z_s = self.state_encoder(s)
            z_sa = self.sa_encoder(s, a)
            z_s2 = self.state_encoder(s2)
            z_neg = self.state_encoder(s_neg)

            logits, labels = self._contrastive_logits(z_sa, z_s2, z_neg)
            contrastive_loss = F.cross_entropy(logits, labels)

            means, logvars = self.dynamics(z_sa, z_s)
            delta_target = (s2 - s).unsqueeze(0).expand_as(means)
            inv_var = torch.exp(-logvars)
            nll = 0.5 * ((delta_target - means).pow(2) * inv_var + logvars).mean()

            loss = self.cfg.contrastive_coef * contrastive_loss + self.cfg.dyn_nll_coef * nll

            self.optim_dynamics.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.state_encoder.parameters())
                + list(self.sa_encoder.parameters())
                + list(self.dynamics.parameters()),
                max_norm=10.0,
            )
            self.optim_dynamics.step()

            cur = float(loss.detach().cpu().item())
            self.loss_log["dynamics_total_loss"] = cur
            self.loss_log["dynamics_contrastive_loss"] = float(contrastive_loss.detach().cpu().item())
            self.loss_log["dynamics_nll_loss"] = float(nll.detach().cpu().item())

            if cur + 1e-6 < best_loss:
                best_loss = cur
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.cfg.dyn_patience):
                    break

        self._dyn_ready = True
        self.loss_log["dynamics_best_loss"] = float(best_loss)


    @torch.no_grad()
    def _soft_update_static(self) -> None:
        tau = float(self.cfg.static_value_tau)
        for p, pt in zip(self.static_q.parameters(), self.static_q_targ.parameters()):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def train_static_value_model(self, seed: int = 0) -> None:
        if len(self.tar_buffer) < max(2, int(self.cfg.static_value_batch_size)):
            raise RuntimeError("Target buffer too small for static target-only value training.")

        torch.manual_seed(int(seed) + 101)
        np.random.seed(int(seed) + 101)

        for step in range(1, int(self.cfg.static_value_steps) + 1):
            s, a, r, done, s2, _ = self.tar_buffer.sample(int(self.cfg.static_value_batch_size))


            with torch.no_grad():
                q1_t, q2_t = self.static_q_targ(s, a)
                q_t = torch.min(q1_t, q2_t).view(-1, 1)
            v = self.static_v(s).view(-1, 1)
            diff = q_t - v
            v_loss = expectile_loss(diff, self.cfg.static_value_expectile).mean()

            self.optim_static_v.zero_grad(set_to_none=True)
            v_loss.backward()
            self.optim_static_v.step()


            with torch.no_grad():
                td = r.view(-1, 1) + (1.0 - done.view(-1, 1)) * float(self.cfg.gamma) * self.static_v(s2).view(-1, 1)
            q1, q2 = self.static_q(s, a)
            q_loss = (
                F.mse_loss(q1.view(-1, 1), td, reduction="none")
                + F.mse_loss(q2.view(-1, 1), td, reduction="none")
            ).mean()

            self.optim_static_q.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.static_q.parameters(), max_norm=10.0)
            self.optim_static_q.step()
            self._soft_update_static()

            if step % 100 == 0 or step == int(self.cfg.static_value_steps):
                self.loss_log["static_v_loss"] = float(v_loss.detach().cpu().item())
                self.loss_log["static_q_loss"] = float(q_loss.detach().cpu().item())
                self.loss_log["static_value_step"] = float(step)

        for p in self.static_q.parameters():
            p.requires_grad_(False)
        for p in self.static_q_targ.parameters():
            p.requires_grad_(False)
        for p in self.static_v.parameters():
            p.requires_grad_(False)
        self.static_q.eval()
        self.static_q_targ.eval()
        self.static_v.eval()
        self._static_v_ready = True


    @torch.no_grad()
    def _compute_dynamics_discrepancy(self, s: Tensor, a: Tensor, s2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z_s = self.state_encoder(s)
        z_sa = self.sa_encoder(s, a)
        z_s2 = self.state_encoder(s2)

        real_score = torch.sum(z_sa * z_s2, dim=-1, keepdim=True)
        fake_next = self.dynamics.predict_next_states(z_sa, z_s, s)
        K, B, _ = fake_next.shape
        fake_flat = fake_next.reshape(K * B, -1)
        z_fake = self.state_encoder(fake_flat).reshape(K, B, -1)
        fake_scores = torch.sum(z_sa.unsqueeze(0) * z_fake, dim=-1)
        d_dyn = (real_score.squeeze(-1).unsqueeze(0) - fake_scores).abs().mean(dim=0, keepdim=False).unsqueeze(-1)
        return d_dyn, real_score, fake_scores.transpose(0, 1)

    @torch.no_grad()
    def _compute_value_discrepancy_static(self, s: Tensor, a: Tensor, s2: Tensor, *, unbiased: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        if not self._static_v_ready:
            raise RuntimeError("Static target-only value model is not ready.")

        z_s = self.state_encoder(s)
        z_sa = self.sa_encoder(s, a)
        fake_next = self.dynamics.predict_next_states(z_sa, z_s, s)


        v_obs = self.static_v(s2).view(-1, 1)
        K, B, _ = fake_next.shape
        if unbiased and K < 2:
            raise ValueError("CVC variance with denominator K-1 requires at least two ensemble members")
        fake_flat = fake_next.reshape(K * B, -1)
        v_fake = self.static_v(fake_flat).reshape(K, B, 1).transpose(0, 1)

        mu_v = v_fake.mean(dim=1)

        var_v = v_fake.var(dim=1, unbiased=unbiased)
        d_val = ((v_obs - mu_v).pow(2) / (var_v + 1e-6)).sqrt()
        return d_val, mu_v, var_v


    def _infer_pseudo_stage_ids_np(self, s: np.ndarray, s2: np.ndarray) -> Optional[np.ndarray]:
        targets_cfg = self.cfg.pseudo_stage_targets
        if not targets_cfg:
            return None
        pos_dims = tuple(int(x) for x in self.cfg.pseudo_stage_pos_dims)
        if len(pos_dims) == 0:
            return None
        if max(pos_dims) >= s.shape[1]:
            self.loss_log["pseudo_stage_enabled"] = 0.0
            self.loss_log["pseudo_stage_error"] = 1.0
            return None

        targets = np.asarray(targets_cfg, dtype=np.float32)
        base = s2 if bool(self.cfg.pseudo_stage_use_next_state) else s
        pos = np.asarray(base[:, pos_dims], dtype=np.float32)
        d = np.linalg.norm(pos[:, None, :] - targets[None, :, :], axis=-1)


        stage = np.argmin(d, axis=1).astype(np.int64)
        return stage


    @torch.no_grad()
    def _compute_source_reference_discrepancy(self, s, a, s2):
        return self._compute_value_discrepancy_static(s, a, s2)

    def _source_reference_weights(self, discrepancy):
        return compute_cvc_soft_weights(discrepancy, self.cfg.value_temperature)

    def precompute_static_source_weights(self, chunk_size: Optional[int] = None) -> None:
        if not self._dyn_ready:
            raise RuntimeError("Please call train_dynamics() first.")
        if not self._static_v_ready:
            raise RuntimeError("Please call train_static_value_model() first.")
        if len(self.src_buffer) == 0:
            raise RuntimeError("Source buffer is empty.")

        src_s, src_a, _, _, src_s2 = self.src_buffer.all()
        n = src_s.shape[0]
        chunk = int(chunk_size or max(self.cfg.dyn_batch_size, 1024))

        d_dyn_all: List[np.ndarray] = []
        d_val_all: List[np.ndarray] = []

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            s = torch.from_numpy(src_s[start:end]).to(self.device)
            a = torch.from_numpy(src_a[start:end]).to(self.device)
            s2 = torch.from_numpy(src_s2[start:end]).to(self.device)

            with torch.no_grad():
                d_dyn, _, _ = self._compute_dynamics_discrepancy(s, a, s2)
                d_val, _, _ = self._compute_source_reference_discrepancy(s, a, s2)
            d_dyn_all.append(d_dyn.cpu().numpy())
            d_val_all.append(d_val.cpu().numpy())

        d_dyn_np = np.concatenate(d_dyn_all, axis=0).reshape(-1, 1).astype(np.float32)
        d_val_np = np.concatenate(d_val_all, axis=0).reshape(-1, 1).astype(np.float32)

        if not np.all(np.isfinite(d_dyn_np)) or np.any(d_dyn_np < 0.0):
            raise FloatingPointError("CDC source discrepancies must be finite and nonnegative")


        soft = self._source_reference_weights(d_val_np)
        self.src_buffer.set_weights(soft)
        self._src_d_dyn_cache = d_dyn_np
        self._src_d_val_cache = d_val_np
        self._static_src_ready = True

        self.loss_log["cached_d_dyn_mean"] = float(d_dyn_np.mean())
        self.loss_log["cached_d_dyn_max"] = float(d_dyn_np.max())
        self.loss_log["cached_d_val_mean"] = float(d_val_np.mean())
        self.loss_log["cached_d_val_max"] = float(d_val_np.max())
        self.loss_log["cached_cvc_w_mean"] = float(soft.mean())
        self.loss_log["cached_cvc_w_max"] = float(soft.max())


        stage = self._infer_pseudo_stage_ids_np(src_s, src_s2)
        self._src_stage_cache = stage
        if stage is not None:
            num_stages = int(np.max(stage)) + 1
            self.loss_log["pseudo_stage_enabled"] = 1.0
            for k in range(num_stages):
                raw_mask = (stage == k)
                if raw_mask.any():
                    self.loss_log[f"pseudo_stage_raw_frac_{k}"] = float(raw_mask.mean())
                    self.loss_log[f"pseudo_stage_d_dyn_mean_{k}"] = float(d_dyn_np[raw_mask].mean())
                    self.loss_log[f"pseudo_stage_d_val_mean_{k}"] = float(d_val_np[raw_mask].mean())
                    self.loss_log[f"pseudo_stage_cvc_w_mean_{k}"] = float(soft[raw_mask].mean())
        else:
            self.loss_log["pseudo_stage_enabled"] = 0.0

    def get_source_weight_diagnostics(self, source_indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:

        if self._src_d_dyn_cache is None or self._src_d_val_cache is None:
            raise RuntimeError("Source diagnostics are not ready. Please run precompute_static_source_weights() first.")
        if self.src_buffer.w is None:
            raise RuntimeError("Source CVC weights are not ready. Please run precompute_static_source_weights() first.")
        indices = np.arange(len(self.src_buffer), dtype=np.int64) if source_indices is None else np.asarray(source_indices)
        if (indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= len(self.src_buffer))):
            raise ValueError("source_indices must be a nonempty one-dimensional batch of valid integer indices")
        d_dyn = self._src_d_dyn_cache[indices].reshape(-1).astype(np.float32)
        d_val = self._src_d_val_cache[indices].reshape(-1).astype(np.float32)
        soft = self.src_buffer.w[indices].reshape(-1).astype(np.float32)

        out = {
            "sample_index": indices.copy(),
            "d_dyn": d_dyn,
            "d_val": d_val,
            "cvc_weight": soft,
            "cdc_gate_scope": np.asarray([CDC_GATE_SCOPE]),
            "keep_ratio": np.asarray([float(self.cfg.xi)], dtype=np.float32),
            "value_temperature": np.asarray([float(self.cfg.value_temperature)], dtype=np.float32),
            "eta_v": np.asarray([float(self.cfg.value_temperature)], dtype=np.float32),
            "cvc_weight_formula": np.asarray([self.SOURCE_WEIGHT_FORMULA]),
            "source_weight_reference": np.asarray([self.SOURCE_WEIGHT_REFERENCE]),
        }
        if source_indices is not None:
            gate = compute_cdc_batch_gate(torch.from_numpy(d_dyn[:, None]), self.cfg.xi).numpy().reshape(-1)
            out["gate"] = gate
            out["src_w"] = gate * soft
        if self._src_stage_cache is not None:
            out["pseudo_stage"] = self._src_stage_cache[indices].reshape(-1).astype(np.int64)
        return out


    @torch.no_grad()
    def _soft_update(self) -> None:
        tau = float(self.cfg.tau)
        for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)

    @torch.no_grad()
    def _compute_td_target_iql(self, s2: Tensor, r: Tensor, done: Tensor) -> Tensor:
        r = r.view(-1, 1)
        done = done.view(-1, 1)
        v_next = self.v(s2).view(-1, 1)
        return r + (1.0 - done) * float(self.cfg.gamma) * v_next

    def train_ac(self) -> None:
        if not self._dyn_ready:
            raise RuntimeError("Please call train_dynamics() before actor-critic training.")
        if not self._static_v_ready:
            raise RuntimeError("Please call train_static_value_model() before actor-critic training.")
        if not self._static_src_ready:
            raise RuntimeError("Please call precompute_static_source_weights() before actor-critic training.")
        return self._train_ac_iql()

    def _train_ac_iql(self) -> None:
        bs = int(self.cfg.batch_size)
        if bs < 2:
            raise RuntimeError("batch_size too small.")
        bs_half = bs // 2
        if len(self.tar_buffer) < bs_half or len(self.src_buffer) < bs_half:
            raise RuntimeError("Buffers too small for training step.")
        if self._src_d_dyn_cache is None:
            raise RuntimeError("Source CDC discrepancy cache is missing.")

        tar_s, tar_a, tar_r, tar_done, tar_s2, _ = self.tar_buffer.sample(bs_half)

        src_idx = self.src_buffer.sample_indices(bs_half)
        src_s, src_a, src_r, src_done, src_s2, src_soft = self.src_buffer.get_batch_by_indices(src_idx)
        if src_soft is None:
            raise RuntimeError("Source CVC weight cache is missing.")
        src_d_dyn = torch.from_numpy(self._src_d_dyn_cache[src_idx]).to(self.device)
        src_gate = compute_cdc_batch_gate(src_d_dyn, self.cfg.xi)
        src_w = src_gate * src_soft

        num_src_batch = float(src_s.shape[0])


        with torch.no_grad():
            q1_tar_t, q2_tar_t = self.q_targ(tar_s, tar_a)
            q_tar_t = torch.min(q1_tar_t, q2_tar_t).view(-1, 1)
            q1_src_t, q2_src_t = self.q_targ(src_s, src_a)
            q_src_t = torch.min(q1_src_t, q2_src_t).view(-1, 1)

        v_tar = self.v(tar_s).view(-1, 1)
        diff_tar = q_tar_t - v_tar
        v_loss_tar = expectile_loss(diff_tar, self.cfg.iql_expectile).mean()

        v_src = self.v(src_s).view(-1, 1)
        diff_src = q_src_t - v_src
        v_loss_src_unreduced = expectile_loss(diff_src, self.cfg.iql_expectile)
        v_loss_src = (v_loss_src_unreduced * src_gate).sum() / num_src_batch

        v_loss = 0.5 * v_loss_tar + 0.5 * v_loss_src

        self.optim_v.zero_grad(set_to_none=True)
        v_loss.backward()
        self.optim_v.step()

        self.loss_log["loss_v_tar"] = float(v_loss_tar.detach().cpu().item())
        self.loss_log["loss_v_src_unweighted"] = float(v_loss_src.detach().cpu().item())
        self.loss_log["loss_v_total"] = float(v_loss.detach().cpu().item())


        with torch.no_grad():
            tar_td = self._compute_td_target_iql(tar_s2, tar_r, tar_done)
            src_td = self._compute_td_target_iql(src_s2, src_r, src_done)

        tar_q1, tar_q2 = self.q(tar_s, tar_a)
        src_q1, src_q2 = self.q(src_s, src_a)

        tar_err = F.mse_loss(tar_q1.view(-1, 1), tar_td.view(-1, 1), reduction="none") + \
                  F.mse_loss(tar_q2.view(-1, 1), tar_td.view(-1, 1), reduction="none")

        src_err = src_w * (
            F.mse_loss(src_q1.view(-1, 1), src_td.view(-1, 1), reduction="none") +
            F.mse_loss(src_q2.view(-1, 1), src_td.view(-1, 1), reduction="none")
        )

        q_loss = (tar_err.sum() + src_err.sum()) / float(tar_err.shape[0] + src_err.shape[0])

        self.optim_q.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optim_q.step()

        self.loss_log["loss_q_total"] = float(q_loss.detach().cpu().item())
        self.loss_log["loss_q_tar"] = float(tar_err.mean().detach().cpu().item())
        self.loss_log["loss_q_src_weighted"] = float(src_err.mean().detach().cpu().item())


        with torch.no_grad():
            q1_t2, q2_t2 = self.q_targ(tar_s, tar_a)
            q_t2 = torch.min(q1_t2, q2_t2).view(-1, 1)
            v_t2 = self.v(tar_s).view(-1, 1)
            adv_t = q_t2 - v_t2
            w_bc_t = torch.exp(float(self.cfg.iql_beta) * adv_t).clamp(max=float(self.cfg.iql_max_weight))

            q1_s2, q2_s2 = self.q_targ(src_s, src_a)
            q_s2 = torch.min(q1_s2, q2_s2).view(-1, 1)
            v_s2 = self.v(src_s).view(-1, 1)
            adv_s = q_s2 - v_s2
            w_bc_s = torch.exp(float(self.cfg.iql_beta) * adv_s).clamp(max=float(self.cfg.iql_max_weight))

        logp_t = self.policy.log_prob(tar_s, tar_a)
        pi_loss_tar = -(w_bc_t * logp_t).mean()

        logp_s = self.policy.log_prob(src_s, src_a)
        pi_loss_src_unreduced = -(w_bc_s * logp_s)
        pi_loss_src = (pi_loss_src_unreduced * src_gate).sum() / num_src_batch

        pi_loss = 0.5 * pi_loss_tar + 0.5 * pi_loss_src

        self.optim_policy.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optim_policy.step()

        self.loss_log["loss_pi"] = float(pi_loss.detach().cpu().item())
        self.loss_log["loss_pi_tar"] = float(pi_loss_tar.detach().cpu().item())
        self.loss_log["loss_pi_src_unweighted"] = float(pi_loss_src.detach().cpu().item())
        self.loss_log["iql_adv_mean"] = float(adv_t.mean().detach().cpu().item())
        self.loss_log["iql_w_bc_mean"] = float(w_bc_t.mean().detach().cpu().item())
        self.loss_log["iql_w_bc_max"] = float(w_bc_t.max().detach().cpu().item())
        self.loss_log["src_keep_frac_batch"] = float(src_gate.mean().detach().cpu().item())
        self.loss_log["src_keep_n_batch"] = float(src_gate.sum().detach().cpu().item())
        self.loss_log["src_w_mean_batch"] = float(src_w.mean().detach().cpu().item())

        self._soft_update()
        self.training_count += 1


    def state_dict(self) -> Dict[str, Dict]:
        return {
            "policy": self.policy.state_dict(),
            "q": self.q.state_dict(),
            "q_targ": self.q_targ.state_dict(),
            "v": self.v.state_dict(),
            "state_encoder": self.state_encoder.state_dict(),
            "sa_encoder": self.sa_encoder.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "static_q": self.static_q.state_dict(),
            "static_q_targ": self.static_q_targ.state_dict(),
            "static_v": self.static_v.state_dict(),
            "optim_policy": self.optim_policy.state_dict(),
            "optim_q": self.optim_q.state_dict(),
            "optim_v": self.optim_v.state_dict(),
            "optim_dynamics": self.optim_dynamics.state_dict(),
            "optim_static_q": self.optim_static_q.state_dict(),
            "optim_static_v": self.optim_static_v.state_dict(),
            "training_count": {"value": self.training_count},
            "cfg": dict(self.cfg.__dict__),
            "source_weight_reference": self.SOURCE_WEIGHT_REFERENCE,
        }

    def load_state_dict(self, state: Dict[str, Dict]) -> None:
        if state.get("cfg", {}).get("cdc_gate_scope") != CDC_GATE_SCOPE:
            raise ValueError("Unsupported CDC gate scope: expected source-minibatch metadata")
        if state.get("source_weight_reference", CCPWeightSACAgent.SOURCE_WEIGHT_REFERENCE) != self.SOURCE_WEIGHT_REFERENCE:
            raise ValueError("Source-weight reference does not match this TDAL variant")
        self.policy.load_state_dict(state["policy"])
        self.q.load_state_dict(state["q"])
        self.q_targ.load_state_dict(state["q_targ"])
        self.v.load_state_dict(state["v"])
        self.state_encoder.load_state_dict(state["state_encoder"])
        self.sa_encoder.load_state_dict(state["sa_encoder"])
        self.dynamics.load_state_dict(state["dynamics"])
        if "static_q" in state:
            self.static_q.load_state_dict(state["static_q"])
        if "static_q_targ" in state:
            self.static_q_targ.load_state_dict(state["static_q_targ"])
        if "static_v" in state:
            self.static_v.load_state_dict(state["static_v"])
        self.optim_policy.load_state_dict(state["optim_policy"])
        self.optim_q.load_state_dict(state["optim_q"])
        self.optim_v.load_state_dict(state["optim_v"])
        self.optim_dynamics.load_state_dict(state["optim_dynamics"])
        if "optim_static_q" in state:
            self.optim_static_q.load_state_dict(state["optim_static_q"])
        if "optim_static_v" in state:
            self.optim_static_v.load_state_dict(state["optim_static_v"])
        self.training_count = int(state.get("training_count", {}).get("value", 0))
        self._dyn_ready = True
        self._static_v_ready = True


class CDCQAgent(CCPWeightSACAgent):


    SOURCE_WEIGHT_REFERENCE = "frozen_target_q_current_state_same_action_kminus1"
    SOURCE_WEIGHT_FORMULA = "exp(-d_q/max(temperature,1e-6))"

    @torch.no_grad()
    def _compute_source_reference_discrepancy(self, s, a, s2):
        if not self._static_v_ready:
            raise RuntimeError("Please call train_static_value_model() first.")
        fake_next = self.dynamics.predict_next_states(
            self.sa_encoder(s, a), self.state_encoder(s), s
        )
        q1, q2 = self.static_q_targ(s, a)
        observed = torch.minimum(q1, q2).view(-1, 1)
        k, b, _ = fake_next.shape
        if k < 2:
            raise ValueError("Q-score variance with denominator K-1 requires at least two ensemble members")
        a_rep = a.unsqueeze(0).expand(k, -1, -1).reshape(k * b, -1)
        q1, q2 = self.static_q_targ(fake_next.reshape(k * b, -1), a_rep)
        values = torch.minimum(q1, q2).reshape(k, b, 1).transpose(0, 1)
        mu, var = values.mean(dim=1), values.var(dim=1, unbiased=True)
        discrepancy = ((observed - mu).square() / (var + 1e-6)).sqrt()
        return discrepancy, mu, var

    def _source_reference_weights(self, discrepancy):
        temperature = max(float(self.cfg.value_temperature), 1e-6)
        return np.exp(-np.asarray(discrepancy, dtype=np.float32) / temperature).astype(np.float32)

    def get_source_weight_diagnostics(self, source_indices=None):
        out = super().get_source_weight_diagnostics(source_indices)
        for previous, current in (("d_val", "d_q"), ("cvc_weight", "q_weight"),
                                  ("cvc_weight_formula", "q_weight_formula")):
            out[current] = out.pop(previous)
        out.pop("eta_v")
        return out
