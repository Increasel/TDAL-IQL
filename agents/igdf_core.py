from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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

def merge_batch(source_batch: TensorBatch, target_batch: TensorBatch) -> TensorBatch:
    return [
        torch.cat((source_batch[0], target_batch[0]), dim=0),
        torch.cat((source_batch[1], target_batch[1]), dim=0),
        torch.cat((source_batch[2], target_batch[2]), dim=0),
        torch.cat((source_batch[3], target_batch[3]), dim=0),
        torch.cat((source_batch[4], target_batch[4]), dim=0),
    ]

def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Optional[Callable[[], nn.Module]] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP requires at least input and output dims")
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, max_action: float, hidden_dim: int = 256, n_hidden: int = 2, dropout: Optional[float] = None):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim], output_activation_fn=nn.Tanh)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = float(max_action)

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str):
        state_t = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state_t)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().numpy().flatten()

class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, max_action: float, hidden_dim: int = 256, n_hidden: int = 2, dropout: Optional[float] = None):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim], output_activation_fn=nn.Tanh, dropout=dropout)
        self.max_action = float(max_action)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str):
        state_t = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action = torch.clamp(self(state_t) * self.max_action, -self.max_action, self.max_action)
        return action.cpu().numpy().flatten()

class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)

class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
    ):
        self.max_action = float(max_action)
        self.qf = q_network

        import copy as _copy
        self.q_target = _copy.deepcopy(self.qf).requires_grad_(False).to(device)

        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = float(iql_tau)
        self.beta = float(beta)
        self.discount = float(discount)
        self.tau = float(tau)

        self.total_it = 0
        self.device = device

    def _update_v(self, observations: torch.Tensor, actions: torch.Tensor, log_dict: Dict) -> torch.Tensor:
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = float(v_loss.item())
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
        mask: torch.Tensor,
    ) -> None:
        qtarget = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)

        q_loss = sum(((mask.squeeze(-1)) * (q - qtarget) ** 2).mean() for q in qs) / len(qs)
        log_dict["q_loss"] = float(q_loss.item())

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(self, adv: torch.Tensor, observations: torch.Tensor, actions: torch.Tensor, log_dict: Dict) -> None:
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape mismatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = float(policy_loss.item())

        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch, mask: torch.Tensor) -> Dict[str, float]:
        self.total_it += 1
        obs, act, rew, next_obs, done = batch
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            next_v = self.vf(next_obs)
        adv = self._update_v(obs, act, log_dict)

        rew = rew.squeeze(-1)
        done = done.squeeze(-1)

        self._update_q(next_v, obs, act, rew, done, log_dict, mask)
        self._update_policy(adv, obs, act, log_dict)
        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        import copy as _copy
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = _copy.deepcopy(self.qf).requires_grad_(False).to(self.device)
        self.q_target.load_state_dict(state_dict["q_target"])

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = int(state_dict["total_it"])
