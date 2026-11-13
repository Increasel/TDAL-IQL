import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from torch.nn.utils import clip_grad_norm_

def soft_update(net, target_net, tau):
    with torch.no_grad():
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden, action_dim)

        self.log_std_head = nn.Linear(hidden, action_dim)


        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.xavier_uniform_(self.log_std_head.weight)

    def forward(self, s):
        x = self.net(s)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)

        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self.forward(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)


        u = dist.rsample()
        a = torch.tanh(u)


        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return a, log_prob, torch.tanh(mu)


class QNetContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net(x)


@dataclass
class SACContinuousConfig:
    state_dim: int
    action_dim: int
    hidden: int = 256
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 1e-3
    target_entropy: float = None


class SACContinuous:


    def __init__(self, cfg: SACContinuousConfig, device="cpu"):
        self.device = torch.device(device)
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.action_dim = cfg.action_dim


        self.policy = PolicyNetContinuous(cfg.state_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.q1 = QNetContinuous(cfg.state_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.q2 = QNetContinuous(cfg.state_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.q1_target = QNetContinuous(cfg.state_dim, cfg.action_dim, cfg.hidden).to(self.device)
        self.q2_target = QNetContinuous(cfg.state_dim, cfg.action_dim, cfg.hidden).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)


        if cfg.target_entropy is None:

            self.target_entropy = -float(cfg.action_dim)
        else:
            self.target_entropy = cfg.target_entropy

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state, eval_mode=False):

        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if eval_mode:

            _, _, action = self.policy.sample(state)
        else:

            action, _, _ = self.policy.sample(state)

        return action.cpu().numpy()[0]

    def update(self, batch):

        s = torch.as_tensor(batch['s'], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch['a'], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(batch['r'], dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(batch['s2'], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch['d'], dtype=torch.float32, device=self.device)


        with torch.no_grad():

            next_action, next_log_prob, _ = self.policy.sample(s2)


            target_q1 = self.q1_target(s2, next_action)
            target_q2 = self.q2_target(s2, next_action)
            min_target_q = torch.min(target_q1, target_q2)


            target_q = r + (1 - d) * self.gamma * (min_target_q - self.alpha * next_log_prob)


        current_q1 = self.q1(s, a)
        current_q2 = self.q2(s, a)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        clip_grad_norm_(self.q1.parameters(), max_norm=5.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        clip_grad_norm_(self.q2.parameters(), max_norm=5.0)
        self.q2_opt.step()


        new_action, log_prob, _ = self.policy.sample(s)

        q1_pi = self.q1(s, new_action)
        q2_pi = self.q2(s, new_action)
        min_q_pi = torch.min(q1_pi, q2_pi)


        policy_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.policy_opt.step()


        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()


        self.alpha_opt.zero_grad()
        alpha_loss.backward()

        self.alpha_opt.step()


        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)

        return {
            "q1_loss": q1_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha.item()
        }

