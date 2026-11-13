from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


TensorBatch = List[torch.Tensor]
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, float(expectile), 1.0 - float(expectile))
    return weight * diff.square()


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], output_tanh: bool = False, squeeze_output: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.ReLU()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.squeeze_output = bool(squeeze_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if self.squeeze_output else out


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP((state_dim, hidden_dim, hidden_dim, action_dim), output_tanh=True)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.max_action = float(max_action)

    def forward(self, states: torch.Tensor) -> Normal:
        mean = self.net(states)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
        action = self(state_t).mean * self.max_action
        return action.clamp(-self.max_action, self.max_action).cpu().numpy().reshape(-1)


class TwinQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        dims = (state_dim + action_dim, hidden_dim, hidden_dim, 1)
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_actions = torch.cat((states, actions), dim=-1)
        return self.q1(state_actions), self.q2(state_actions)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.minimum(*self.both(states, actions))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.v = MLP((state_dim, hidden_dim, hidden_dim, 1), squeeze_output=True)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.v(states)


class DynamicsHead(nn.Module):


    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = MLP((input_dim, hidden_dim, hidden_dim, hidden_dim, 2 * output_dim))
        self.min_logvar = nn.Parameter(torch.full((output_dim,), -10.0))
        self.max_logvar = nn.Parameter(torch.full((output_dim,), 0.5))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = torch.chunk(self.backbone(inputs), 2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar


@dataclass
class DynamicsConfig:
    ensemble_size: int = 7
    hidden_dim: int = 200
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    holdout_ratio: float = 0.1
    patience: int = 10


class TargetDynamicsEnsemble(nn.Module):


    def __init__(self, state_dim: int, action_dim: int, config: DynamicsConfig, device: str):
        super().__init__()
        self.config = config
        self.device_name = device
        self.output_dim = int(state_dim) + 1
        self.heads = nn.ModuleList(
            DynamicsHead(state_dim + action_dim, self.output_dim, int(config.hidden_dim))
            for _ in range(int(config.ensemble_size))
        )
        self.optimizers = [torch.optim.Adam(head.parameters(), lr=float(config.lr)) for head in self.heads]
        self.register_buffer("input_mean", torch.zeros(state_dim + action_dim))
        self.register_buffer("input_std", torch.ones(state_dim + action_dim))
        self.register_buffer("output_mean", torch.zeros(self.output_dim))
        self.register_buffer("output_std", torch.ones(self.output_dim))

    def fit(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, seed: int) -> Dict[str, float]:
        inputs = np.concatenate((states, actions), axis=-1).astype(np.float32)
        targets = np.concatenate((rewards.reshape(-1, 1), next_states - states), axis=-1).astype(np.float32)
        if inputs.shape[0] < 4:
            raise ValueError("VGDF dynamics requires at least four target transitions.")

        self.input_mean.copy_(torch.as_tensor(inputs.mean(axis=0), device=self.input_mean.device))
        self.input_std.copy_(torch.as_tensor(inputs.std(axis=0) + 1e-6, device=self.input_std.device))
        self.output_mean.copy_(torch.as_tensor(targets.mean(axis=0), device=self.output_mean.device))
        self.output_std.copy_(torch.as_tensor(targets.std(axis=0) + 1e-6, device=self.output_std.device))
        x = (inputs - self.input_mean.detach().cpu().numpy()) / self.input_std.detach().cpu().numpy()
        y = (targets - self.output_mean.detach().cpu().numpy()) / self.output_std.detach().cpu().numpy()

        rng = np.random.default_rng(int(seed))
        n = int(x.shape[0])
        holdout_n = max(1, int(round(n * float(self.config.holdout_ratio))))
        permutation = rng.permutation(n)
        holdout_idx, train_pool = permutation[:holdout_n], permutation[holdout_n:]
        if train_pool.size == 0:
            train_pool = holdout_idx
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device_name)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device_name)
        x_holdout, y_holdout = x_t[holdout_idx], y_t[holdout_idx]

        final_losses: List[float] = []
        for head_id, (head, optimizer) in enumerate(zip(self.heads, self.optimizers)):
            best_state = copy.deepcopy(head.state_dict())
            best_loss = float("inf")
            stale_epochs = 0
            bootstrap = rng.choice(train_pool, size=train_pool.size, replace=True)
            steps_per_epoch = max(1, int(np.ceil(bootstrap.size / float(self.config.batch_size))))
            for _epoch in range(int(self.config.epochs)):
                head.train()
                for _ in range(steps_per_epoch):
                    idx = rng.choice(bootstrap, size=int(self.config.batch_size), replace=True)
                    mean, logvar = head(x_t[idx])
                    inv_var = torch.exp(-logvar)
                    loss = ((y_t[idx] - mean).square() * inv_var + logvar).mean()
                    loss = loss + 1e-3 * (head.max_logvar.sum() - head.min_logvar.sum())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                head.eval()
                with torch.no_grad():
                    mean, logvar = head(x_holdout)
                    val_loss = float((((y_holdout - mean).square() * torch.exp(-logvar) + logvar).mean()).item())
                if val_loss < best_loss * 0.995:
                    best_loss = val_loss
                    best_state = copy.deepcopy(head.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(self.config.patience):
                        break
            head.load_state_dict(best_state)
            final_losses.append(best_loss)
        return {"dynamics_holdout_nll": float(np.mean(final_losses)), "dynamics_heads": float(len(final_losses))}

    @torch.no_grad()
    def predict_all(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((states, actions), dim=-1)
        inputs = (inputs - self.input_mean) / self.input_std
        means, variances = [], []
        for head in self.heads:
            mean, logvar = head(inputs)
            means.append(mean * self.output_std + self.output_mean)
            variances.append(torch.exp(logvar) * self.output_std.square())
        return torch.stack(means, dim=0), torch.stack(variances, dim=0)


class OfflineReplayBuffer:
    def __init__(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_states: np.ndarray, device: str):
        self.states = torch.as_tensor(states, dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        self.rewards = torch.as_tensor(rewards.reshape(-1, 1), dtype=torch.float32, device=device)
        self.dones = torch.as_tensor(dones.reshape(-1, 1), dtype=torch.float32, device=device)
        self.next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
        self.size = int(self.states.shape[0])

    def sample(self, batch_size: int) -> TensorBatch:
        idx = torch.randint(self.size, (int(batch_size),), device=self.states.device)
        return [self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]]


@dataclass
class VGDFIQLConfig:
    state_dim: int
    action_dim: int
    max_action: float
    device: str
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    likelihood_gate_quantile: float = 0.75

    start_gate_step: int = 20_000
    min_gate_std: float = 1e-3
    gate_scope: str = "q"


class VGDFIQLTrainer:


    def __init__(self, config: VGDFIQLConfig, dynamics: TargetDynamicsEnsemble):
        if config.gate_scope not in {"q", "all"}:
            raise ValueError("gate_scope must be either 'q' or 'all'.")
        self.config = config
        self.device = config.device
        self.dynamics = dynamics
        self.actor = GaussianPolicy(config.state_dim, config.action_dim, config.max_action).to(self.device)
        self.qf = TwinQ(config.state_dim, config.action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(self.device)
        self.vf = ValueFunction(config.state_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(config.lr))
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=float(config.lr))
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=float(config.lr))
        self.total_it = 0

    @torch.no_grad()
    def gate_source_batch(self, source_batch: TensorBatch, step: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        states, actions, rewards, next_states, dones = source_batch


        gate_active = int(step) > int(self.config.start_gate_step)
        if not gate_active:
            gate = torch.ones(states.shape[0], dtype=torch.bool, device=states.device)
            diagnostics = {
                "src_keep_ratio": 1.0,
                "fvp_log_likelihood_mean": float("nan"),
                "fvp_log_likelihood_min": float("nan"),
                "fvp_log_likelihood_max": float("nan"),
                "fvp_generated_td_mean": float("nan"),
                "fvp_generated_td_std": float("nan"),
                "fvp_observed_td_mean": float("nan"),
                "fvp_observed_generated_td_absdiff": float("nan"),
                "gate_threshold": float("nan"),
                "gate_active": 0.0,
            }
            return gate, diagnostics


        observed_next_actions = self.actor(next_states).sample()
        observed_next_q = self.q_target(next_states, observed_next_actions).unsqueeze(-1)
        observed_td = rewards + (1.0 - dones) * float(self.config.gamma) * observed_next_q

        pred_mean, pred_var = self.dynamics.predict_all(states, actions)


        pred_sample = pred_mean + pred_var.clamp_min(1e-8).sqrt() * torch.randn_like(pred_mean)
        predicted_rewards = pred_sample[..., :1]
        predicted_next_states = states.unsqueeze(0) + pred_sample[..., 1:]
        ensemble_n, batch_n, state_dim = predicted_next_states.shape
        flat_next_states = predicted_next_states.reshape(ensemble_n * batch_n, state_dim)
        imagined_actions = self.actor(flat_next_states).sample()
        predicted_q = self.q_target(flat_next_states, imagined_actions).reshape(ensemble_n, batch_n, 1)
        generated_td = predicted_rewards + float(self.config.gamma) * predicted_q
        td_mean = generated_td.mean(dim=0)
        td_std = generated_td.std(dim=0, unbiased=False).clamp_min(float(self.config.min_gate_std))
        log_likelihood = Normal(td_mean, td_std).log_prob(observed_td)

        threshold = torch.quantile(log_likelihood, float(self.config.likelihood_gate_quantile))
        gate = log_likelihood >= threshold

        diagnostics = {
            "src_keep_ratio": float(gate.float().mean().item()),
            "fvp_log_likelihood_mean": float(log_likelihood.mean().item()),
            "fvp_log_likelihood_min": float(log_likelihood.min().item()),
            "fvp_log_likelihood_max": float(log_likelihood.max().item()),
            "fvp_generated_td_mean": float(td_mean.mean().item()),
            "fvp_generated_td_std": float(td_std.mean().item()),
            "fvp_observed_td_mean": float(observed_td.mean().item()),
            "fvp_observed_generated_td_absdiff": float((observed_td - td_mean).abs().mean().item()),
            "gate_threshold": float(threshold.item()) if torch.isfinite(threshold) else float("nan"),
            "gate_active": float(gate_active),
        }
        return gate.squeeze(-1), diagnostics

    @staticmethod
    def _concat_batches(first_batch: TensorBatch, second_batch: TensorBatch) -> TensorBatch:
        return [torch.cat((first, second), dim=0) for first, second in zip(first_batch, second_batch)]

    def train_step(self, source_batch: TensorBatch, target_batch: TensorBatch, step: int) -> Dict[str, float]:
        self.total_it += 1
        source_gate, log = self.gate_source_batch(source_batch, step)
        source_keep_n = int(source_gate.sum().item())
        source_n = int(source_gate.numel())
        target_n = int(target_batch[0].shape[0])
        full_batch = self._concat_batches(source_batch, target_batch)
        states, actions, _rewards, _next_states, _dones = full_batch
        source_states, source_actions, source_rewards, source_next_states, source_dones = source_batch
        target_states, target_actions, target_rewards, target_next_states, target_dones = target_batch
        source_gate_weight = source_gate.to(dtype=source_states.dtype)
        gate_all_objectives = self.config.gate_scope == "all"

        if gate_all_objectives:
            with torch.no_grad():
                source_value_target = self.q_target(source_states, source_actions)
                target_value_target = self.q_target(target_states, target_actions)
            source_value_error = expectile_loss(
                source_value_target - self.vf(source_states), self.config.iql_expectile
            )
            target_value_error = expectile_loss(
                target_value_target - self.vf(target_states), self.config.iql_expectile
            )
            source_v_loss = (source_gate_weight * source_value_error).mean()
            target_v_loss = target_value_error.mean()
            v_loss = 0.5 * (target_v_loss + source_v_loss)
        else:

            with torch.no_grad():
                target_q = self.q_target(states, actions)
            values = self.vf(states)
            advantages = target_q - values
            v_loss = expectile_loss(advantages, self.config.iql_expectile).mean()
            source_v_loss = expectile_loss(
                advantages[:source_n], self.config.iql_expectile
            ).mean()
            target_v_loss = expectile_loss(
                advantages[source_n:], self.config.iql_expectile
            ).mean()
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()


        with torch.no_grad():
            target_td = target_rewards.squeeze(-1) + (1.0 - target_dones.squeeze(-1)) * float(self.config.gamma) * self.vf(target_next_states)
        target_q1, target_q2 = self.qf.both(target_states, target_actions)
        target_q_loss = 0.5 * ((target_q1 - target_td).square().mean() + (target_q2 - target_td).square().mean())

        with torch.no_grad():
            source_td = source_rewards.squeeze(-1) + (1.0 - source_dones.squeeze(-1)) * float(self.config.gamma) * self.vf(source_next_states)
        source_q1, source_q2 = self.qf.both(source_states, source_actions)
        source_q1_error = (source_q1 - source_td).square()
        source_q2_error = (source_q2 - source_td).square()
        source_gate_weight = source_gate_weight.to(dtype=source_q1_error.dtype)


        source_q_loss = 0.5 * (
            (source_gate_weight * source_q1_error).mean()
            + (source_gate_weight * source_q2_error).mean()
        )
        source_q_selected_loss = 0.5 * (
            source_q1_error[source_gate].mean() + source_q2_error[source_gate].mean()
        )
        q_loss = 0.5 * (target_q_loss + source_q_loss)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.config.tau)

        if gate_all_objectives:
            with torch.no_grad():
                source_policy_advantage = self.q_target(source_states, source_actions) - self.vf(source_states)
                target_policy_advantage = self.q_target(target_states, target_actions) - self.vf(target_states)
                source_adv_weight = torch.exp(
                    float(self.config.iql_beta) * source_policy_advantage
                ).clamp(max=EXP_ADV_MAX)
                target_adv_weight = torch.exp(
                    float(self.config.iql_beta) * target_policy_advantage
                ).clamp(max=EXP_ADV_MAX)
            source_policy_error = -(
                source_adv_weight * self.actor(source_states).log_prob(source_actions).sum(dim=-1)
            )
            target_policy_error = -(
                target_adv_weight * self.actor(target_states).log_prob(target_actions).sum(dim=-1)
            )
            source_policy_loss = (source_gate_weight * source_policy_error).mean()
            target_policy_loss = target_policy_error.mean()
            policy_loss = 0.5 * (target_policy_loss + source_policy_loss)
            policy_advantage = torch.cat((source_policy_advantage, target_policy_advantage), dim=0)
            adv_weight = torch.cat((source_adv_weight, target_adv_weight), dim=0)
        else:

            with torch.no_grad():
                policy_advantage = self.q_target(states, actions) - self.vf(states)
                adv_weight = torch.exp(float(self.config.iql_beta) * policy_advantage).clamp(max=EXP_ADV_MAX)
            policy_error = -(adv_weight * self.actor(states).log_prob(actions).sum(dim=-1))
            policy_loss = policy_error.mean()
            source_policy_loss = policy_error[:source_n].mean()
            target_policy_loss = policy_error[source_n:].mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()

        v_source_keep_n = source_keep_n if gate_all_objectives else source_n
        actor_source_keep_n = source_keep_n if gate_all_objectives else source_n

        log.update({
            "value_loss": float(v_loss.item()),
            "v_target_loss": float(target_v_loss.item()),
            "v_source_loss": float(source_v_loss.item()),
            "v_source_effective_weight": float(v_source_keep_n / source_n),
            "q_loss": float(q_loss.item()),
            "q_target_loss": float(target_q_loss.item()),
            "q_source_loss": float(source_q_loss.item()),
            "q_source_selected_loss": float(source_q_selected_loss.item()),
            "q_source_effective_weight": float(source_gate.float().mean().item()),
            "actor_loss": float(policy_loss.item()),
            "actor_target_loss": float(target_policy_loss.item()),
            "actor_source_loss": float(source_policy_loss.item()),
            "actor_source_effective_weight": float(actor_source_keep_n / source_n),
            "iql_adv_mean": float(policy_advantage.mean().item()),
            "iql_bc_weight_mean": float(adv_weight.mean().item()),
            "q_src_n_kept": float(source_keep_n),
            "q_src_keep_ratio": float(source_keep_n / source_n),
            "q_src_n_full": float(source_n),
            "q_target_n_full": float(target_n),
            "v_src_n_full": float(source_n),
            "v_src_n_kept": float(v_source_keep_n),
            "v_src_keep_ratio": float(v_source_keep_n / source_n),
            "v_target_n_full": float(target_n),
            "actor_src_n_full": float(source_n),
            "actor_src_n_kept": float(actor_source_keep_n),
            "actor_src_keep_ratio": float(actor_source_keep_n / source_n),
            "actor_target_n_full": float(target_n),
        })
        return log

    def state_dict(self) -> Dict[str, Any]:
        return {
            "trainer_config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
            "vf": self.vf.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "total_it": self.total_it,
        }
