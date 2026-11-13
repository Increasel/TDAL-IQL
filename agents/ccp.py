from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.dynamics import CrossConformalEnsembleDynamics, DynamicsTrainConfig

Tensor = torch.Tensor


from offline.array_buffer import ArrayReplayBuffer as OfflineReplayBuffer


def mlp(in_dim: int, out_dim: int, hidden: Tuple[int, ...], act=nn.ReLU, use_layernorm=True) -> nn.Sequential:
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
        self.s_dim = int(s_dim)
        self.a_dim = int(a_dim)
        self.net = mlp(self.s_dim, self.a_dim * 2, hidden, act=nn.ReLU)
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
        correction = torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        logp = logp - correction
        return a, logp


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
        if deterministic:
            u = mean
        else:
            u = mean + std * torch.randn_like(std)
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


    algo: str = "cql"


    cql_alpha: float = 1.0
    cql_n_actions: int = 10
    cql_use_random: bool = True


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
    dyn_include_reward: bool = False
    cp_score_ignore_reward: bool = True

    weight_mode: str = "power"
    weight_power: float = 1.0
    weight_alpha: float = 0.1
    weight_temperature: float = 0.05
    cache_src_weights: bool = True
    xi: float = 0.2

    normalize_src_weights: bool = True
    src_weight_clip: float = 1.0


class CCPWeightSACAgent:


    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.device = cfg.device


        self.policy = SquashedGaussianPolicy(cfg.s_dim, cfg.a_dim).to(self.device)
        self.q = TwinQ(cfg.s_dim, cfg.a_dim).to(self.device)
        self.q_targ = TwinQ(cfg.s_dim, cfg.a_dim).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())

        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.optim_q = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)


        self.v: Optional[VNetwork] = None
        self.optim_v: Optional[torch.optim.Optimizer] = None
        if cfg.algo.lower() == "iql":
            self.v = VNetwork(cfg.s_dim).to(self.device)
            self.optim_v = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)


        if cfg.algo.lower() == "iql":
            self.train_alpha = False
            self.alpha = torch.tensor(0.0, device=self.device)
            self.target_entropy = torch.tensor(-float(cfg.a_dim),
                                               device=self.device)
            self.optim_alpha = None

            self.log_alpha = None
        else:
            if cfg.alpha is None:
                self.train_alpha = True
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha = self.log_alpha.exp()
                target_ent = cfg.target_entropy if cfg.target_entropy is not None else -float(cfg.a_dim)
                self.target_entropy = torch.tensor(target_ent, device=self.device)
                self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
            else:
                self.train_alpha = False
                self.alpha = torch.tensor(float(cfg.alpha), device=self.device)
                self.target_entropy = torch.tensor(-float(cfg.a_dim), device=self.device)
                self.optim_alpha = None
                self.log_alpha = None


        input_dim = cfg.s_dim + cfg.a_dim
        output_dim = (1 + cfg.s_dim) if cfg.dyn_include_reward else cfg.s_dim
        self.dynamics = CrossConformalEnsembleDynamics(
            input_dim=input_dim,
            output_dim=output_dim,
            score_y_start_idx=(1 if (cfg.dyn_include_reward and cfg.cp_score_ignore_reward) else 0),
            k_folds=cfg.k_folds,
            hidden_dims=cfg.dyn_hidden,
        ).to(self.device)


        self.src_buffer = OfflineReplayBuffer(cfg.s_dim, cfg.a_dim, capacity=cfg.buffer_capacity, device=self.device)
        self.tar_buffer = OfflineReplayBuffer(cfg.s_dim, cfg.a_dim, capacity=cfg.buffer_capacity, device=self.device)


        self.training_count = 0
        self.loss_log: Dict[str, float] = {}


    def load_source_dataset(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, done: np.ndarray,
                            s2: np.ndarray) -> None:
        self.src_buffer.load_from_arrays(s, a, r, done, s2)

    def load_target_dataset(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, done: np.ndarray,
                            s2: np.ndarray) -> None:
        self.tar_buffer.load_from_arrays(s, a, r, done, s2)


    def train_dynamics(self, seed: int = 0) -> None:

        if len(self.tar_buffer) < self.cfg.dyn_batch_size:
            raise RuntimeError("Target buffer too small for dynamics training.")

        s, a, r, done, s2 = self.tar_buffer.all()
        x = np.concatenate([s, a], axis=-1).astype(np.float32)

        if self.cfg.dyn_include_reward:
            y = np.concatenate([r.reshape(-1, 1), (s2 - s)], axis=-1).astype(np.float32)
        else:
            y = (s2 - s).astype(np.float32)

        dyn_cfg = DynamicsTrainConfig(
            batch_size=self.cfg.dyn_batch_size,
            lr=self.cfg.dyn_lr,
            weight_decay=0.0,
            epochs=self.cfg.dyn_epochs,
            patience=self.cfg.dyn_patience,
            device=self.cfg.device,
            dtype="float32",
            verbose=True,
            seed=seed,
        )

        info = self.dynamics.fit_kfold(
            X=x,
            Y=y,
            cfg=dyn_cfg,
            loss_type="nll",
            score_norm="l2",
            refit_normalizer=False,
            train_val_ratio=self.cfg.dyn_train_val_ratio,
        )
        for k, v in info.items():
            self.loss_log[f"dynamics_{k}"] = float(v)


        if self.cfg.cache_src_weights:
            self._cache_source_weights()


    def _cache_source_weights(self) -> None:
        if len(self.src_buffer) == 0:
            return
        s, a, r, done, s2 = self.src_buffer.all()
        x = np.concatenate([s, a], axis=-1).astype(np.float32)

        if self.cfg.dyn_include_reward:
            y = np.concatenate([r.reshape(-1, 1), (s2 - s)], axis=-1).astype(np.float32)
        else:
            y = (s2 - s).astype(np.float32)

        w = self.dynamics.weights(
            X=x,
            Y=y,
            score_norm="l2",
            mode=self.cfg.weight_mode,
            power=self.cfg.weight_power,
            alpha=self.cfg.weight_alpha,
            temperature=self.cfg.weight_temperature,
        ).reshape(-1, 1).astype(np.float32)

        self.src_buffer.set_weights(w)
        self.loss_log["cached_src_weight_mean"] = float(w.mean())
        self.loss_log["cached_src_weight_min"] = float(w.min())
        self.loss_log["cached_src_weight_max"] = float(w.max())

        w = w.clip(0.0, 1.0)
        if self.cfg.normalize_src_weights:
            w_mean = float(w.mean())
            if w_mean > 1e-8:
                w = w / w_mean
            else:
                w = np.zeros_like(w)
        w = w.clip(max=self.cfg.src_weight_clip)

        self.loss_log["cached_src_w_norm_mean"] = float(w.mean())
        self.loss_log["cached_src_w_norm_min"] = float(w.min())
        self.loss_log["cached_src_w_norm_max"] = float(w.max())


    @torch.no_grad()
    def _soft_update(self) -> None:
        tau = float(self.cfg.tau)
        for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def _compute_td_target(self, s2: Tensor, r: Tensor, done: Tensor) -> Tensor:

        r = r.view(-1, 1)
        done = done.view(-1, 1)
        a2, logp2 = self.policy.sample(s2)
        q1_t, q2_t = self.q_targ(s2, a2)
        q_t = torch.min(q1_t, q2_t).view(-1, 1)
        logp2 = logp2.view(-1, 1)
        return r + (1.0 - done) * self.cfg.gamma * (q_t - self.alpha * logp2)


    def _compute_td_target_iql(self, s2: Tensor, r: Tensor, done: Tensor) -> Tensor:
        if self.v is None:
            raise RuntimeError("IQL requested but value network is not initialized.")
        r = r.view(-1, 1)
        done = done.view(-1, 1)
        with torch.no_grad():
            v2 = self.v(s2).view(-1, 1)
        return r + (1.0 - done) * self.cfg.gamma * v2

    def _cql_penalty(self, s: Tensor, a_data: Tensor) -> Tuple[Tensor, Tensor]:

        n = int(self.cfg.cql_n_actions)
        bs = s.shape[0]


        s_rep = s.unsqueeze(1).expand(bs, n, -1).reshape(bs * n, -1)


        with torch.no_grad():
            a_pi, _ = self.policy.sample(s_rep)
        q1_pi, q2_pi = self.q(s_rep, a_pi)
        q1_pi = q1_pi.view(bs, n, 1)
        q2_pi = q2_pi.view(bs, n, 1)

        q1_list = [q1_pi]
        q2_list = [q2_pi]


        if self.cfg.cql_use_random:
            a_rand = (2.0 * torch.rand_like(a_pi) - 1.0).clamp(-1, 1)
            q1_rand, q2_rand = self.q(s_rep, a_rand)
            q1_list.append(q1_rand.view(bs, n, 1))
            q2_list.append(q2_rand.view(bs, n, 1))


        q1_data, q2_data = self.q(s, a_data)
        q1_data = q1_data.view(bs, 1, 1)
        q2_data = q2_data.view(bs, 1, 1)


        q1_m = torch.cat(q1_list + [q1_data], dim=1)
        q2_m = torch.cat(q2_list + [q2_data], dim=1)


        cql1 = (torch.logsumexp(q1_m, dim=1) - q1_data.squeeze(1)).mean()
        cql2 = (torch.logsumexp(q2_m, dim=1) - q2_data.squeeze(1)).mean()
        return cql1, cql2

    def train_ac(self) -> None:


        if self.cfg.algo.lower() == "iql":
            return self._train_ac_iql()

        if len(self.tar_buffer) < self.cfg.batch_size or len(self.src_buffer) < self.cfg.batch_size:
            raise RuntimeError("Buffers too small for training step.")


        tar_s, tar_a, tar_r, tar_done, tar_s2, _ = self.tar_buffer.sample(self.cfg.batch_size)
        src_s, src_a, src_r, src_done, src_s2, src_w = self.src_buffer.sample(self.cfg.batch_size)


        if src_w is None:
            x = torch.cat([src_s, src_a], dim=-1).detach().cpu().numpy().astype(np.float32)

            if self.cfg.dyn_include_reward:
                y = torch.cat([src_r, (src_s2 - src_s)], dim=-1).detach().cpu().numpy().astype(np.float32)
            else:
                y = (src_s2 - src_s).detach().cpu().numpy().astype(np.float32)
            w_np = self.dynamics.weights(
                X=x,
                Y=y,
                score_norm="l2",
                mode=self.cfg.weight_mode,
                power=self.cfg.weight_power,
                alpha=self.cfg.weight_alpha,
                temperature=self.cfg.weight_temperature,
            ).reshape(-1, 1).astype(np.float32)
            src_w = torch.from_numpy(w_np).to(self.device)


        with torch.no_grad():
            tar_td = self._compute_td_target(tar_s2, tar_r, tar_done)
            src_td = self._compute_td_target(src_s2, src_r, src_done)

        tar_q1, tar_q2 = self.q(tar_s, tar_a)
        src_q1, src_q2 = self.q(src_s, src_a)


        tar_td = tar_td.view(-1, 1)
        src_td = src_td.view(-1, 1)
        tar_q1, tar_q2 = tar_q1.view(-1, 1), tar_q2.view(-1, 1)
        src_q1, src_q2 = src_q1.view(-1, 1), src_q2.view(-1, 1)
        if tar_q1.shape != tar_td.shape:
            raise RuntimeError(f"[shape] tar_q1 {tuple(tar_q1.shape)} vs tar_td {tuple(tar_td.shape)}")
        if src_q1.shape != src_td.shape:
            raise RuntimeError(f"[shape] src_q1 {tuple(src_q1.shape)} vs src_td {tuple(src_td.shape)}")


        tar_loss = F.huber_loss(tar_q1, tar_td, delta=10.0) + F.huber_loss(tar_q2, tar_td, delta=10.0)


        w = src_w.clamp(0.0, 1.0).detach()

        if self.cfg.normalize_src_weights:

            w_mean = float(w.mean().detach().cpu().item())

            if w_mean > 1e-8:
                w = w / w_mean
            else:
                w = torch.zeros_like(w)

        w = w.clamp(max=self.cfg.src_weight_clip)


        src_loss = (w * F.huber_loss(src_q1, src_td, delta=10.0, reduction='none')).mean() + \
                   (w * F.huber_loss(src_q2, src_td, delta=10.0, reduction='none')).mean()


        q_loss = tar_loss + src_loss


        if self.cfg.algo.lower() == "cql":
            cql1, cql2 = self._cql_penalty(tar_s, tar_a)
            q_loss = q_loss + self.cfg.cql_alpha * (cql1 + cql2)
            self.loss_log["cql1"] = float(cql1.detach().cpu().item())
            self.loss_log["cql2"] = float(cql2.detach().cpu().item())

        self.optim_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim_q.step()

        self.loss_log["loss_q_total"] = float(q_loss.detach().cpu().item())
        self.loss_log["loss_q_tar"] = float(tar_loss.detach().cpu().item())
        self.loss_log["loss_q_src_weighted"] = float(src_loss.detach().cpu().item())
        self.loss_log["src_w_mean"] = float(w.mean().detach().cpu().item())


        if self.training_count % self.cfg.training_delay == 0:

            a_pi, logp_pi = self.policy.sample(tar_s)
            q1_pi, q2_pi = self.q(tar_s, a_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            pi_loss = (self.alpha * logp_pi - q_pi).mean()


            if self.cfg.bc_coef > 0.0:

                bc_loss = F.mse_loss(a_pi, tar_a)

                a_mean, _, _ = self.policy.forward(tar_s)
                bc_loss = F.mse_loss(a_mean, tar_a)
                pi_loss = bc_loss


                self.loss_log["bc_loss"] = float(bc_loss.detach().cpu().item())

            self.optim_policy.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.optim_policy.step()
            self.loss_log["loss_pi"] = float(pi_loss.detach().cpu().item())


            if self.train_alpha and self.optim_alpha is not None:

                alpha_loss = (-(self.log_alpha.exp()) * (logp_pi.detach() + self.target_entropy)).mean()
                self.optim_alpha.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.optim_alpha.step()
                self.alpha = self.log_alpha.exp()
                self.loss_log["alpha"] = float(self.alpha.detach().cpu().item())
                self.loss_log["loss_alpha"] = float(alpha_loss.detach().cpu().item())


        self._soft_update()
        self.training_count += 1


    def _train_ac_iql(self) -> None:

        if self.v is None or self.optim_v is None:
            raise RuntimeError("IQL requested but value network/optimizer is missing.")
        if float(self.cfg.xi) != 0.0:
            raise ValueError("Shared IQL trains its selected dataset without conformal source filtering")
        bs = int(self.cfg.batch_size)
        if bs < 2 or bs % 2:
            raise RuntimeError("IQL batch_size must be positive and even.")
        half = bs // 2
        if len(self.tar_buffer) < half or len(self.src_buffer) < half:
            raise RuntimeError("Buffers too small for training step.")
        tar_s, tar_a, tar_r, tar_done, tar_s2, _ = self.tar_buffer.sample(half)
        self.src_buffer.sample_indices(half)
        self.loss_log["src_keep_n"] = 0.0
        self.loss_log["src_keep_frac"] = 0.0

        with torch.no_grad():
            q1, q2 = self.q_targ(tar_s, tar_a)
            q = torch.min(q1, q2).view(-1, 1)
        v = self.v(tar_s).view(-1, 1)
        v_loss = expectile_loss(q.detach() - v, self.cfg.iql_expectile).mean()
        self.optim_v.zero_grad(set_to_none=True)
        v_loss.backward()
        self.optim_v.step()
        self.loss_log["loss_v_tar"] = float(v_loss.detach().cpu().item())
        self.loss_log["loss_v_src_unweighted"] = 0.0
        self.loss_log["loss_v_total"] = float(v_loss.detach().cpu().item())

        with torch.no_grad():
            td = self._compute_td_target_iql(tar_s2, tar_r, tar_done).view(-1, 1)
        q1, q2 = self.q(tar_s, tar_a)
        error = F.mse_loss(q1.view(-1, 1), td, reduction="none") + \
                F.mse_loss(q2.view(-1, 1), td, reduction="none")
        q_loss = error.mean()
        self.optim_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim_q.step()
        self.loss_log["loss_q_total"] = float(q_loss.detach().cpu().item())
        self.loss_log["loss_q_tar"] = float(q_loss.detach().cpu().item())
        self.loss_log["loss_q_src_weighted"] = 0.0
        self.loss_log["src_w_mean"] = 0.0

        with torch.no_grad():
            q1, q2 = self.q_targ(tar_s, tar_a)
            advantage = torch.min(q1, q2).view(-1, 1) - self.v(tar_s).view(-1, 1)
            weight = torch.exp(self.cfg.iql_beta * advantage).clamp(max=self.cfg.iql_max_weight)
        loss = -(weight * self.policy.log_prob(tar_s, tar_a)).mean()
        self.optim_policy.zero_grad(set_to_none=True)
        loss.backward()
        self.optim_policy.step()
        self.loss_log["loss_pi"] = float(loss.detach().cpu().item())
        self.loss_log["loss_pi_tar"] = float(loss.detach().cpu().item())
        self.loss_log["loss_pi_src_unweighted"] = 0.0
        self.loss_log["iql_adv_mean"] = float(advantage.mean().detach().cpu().item())
        self.loss_log["iql_w_bc_mean"] = float(weight.mean().detach().cpu().item())
        self.loss_log["iql_w_bc_max"] = float(weight.max().detach().cpu().item())
        self._soft_update()
        self.training_count += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "q": self.q.state_dict(),
            "q_targ": self.q_targ.state_dict(),
            "alpha": float(self.alpha.detach().cpu().item()) if isinstance(self.alpha, torch.Tensor) else float(
                self.alpha),
            "train_alpha": bool(self.train_alpha),
            "log_alpha": self.log_alpha.detach().cpu() if getattr(self, "log_alpha", None) is not None else None,
            "dynamics": self.dynamics.state_dict(),

            "v": (self.v.state_dict() if self.v is not None else None),
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.policy.load_state_dict(sd["policy"])
        self.q.load_state_dict(sd["q"])
        self.q_targ.load_state_dict(sd["q_targ"])
        self.dynamics.load_state_dict(sd["dynamics"])


        if self.v is not None and sd.get("v") is not None:
            self.v.load_state_dict(sd["v"])

        if self.train_alpha and sd.get("log_alpha") is not None:
            self.log_alpha.data.copy_(sd["log_alpha"].to(self.device))
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(float(sd.get("alpha", self.alpha.item())), device=self.device)
