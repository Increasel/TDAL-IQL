from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .darc_core import CriticQ, ValueV, GaussianPolicy, DarcDiscriminator, expectile_loss

@dataclass
class DARCConfig:
    s_dim: int
    a_dim: int
    device: str = "cpu"
    lr: float = 3e-4
    batch_size: int = 512
    gamma: float = .99
    tau: float = .005
    iql_expectile: float = .7
    iql_beta: float = 3.
    iql_max_weight: float = 100.
    normalize_states: bool = False
    max_action: float = 1.
    disc_lr: float = 3e-4
    input_noise: float = 1.
    classifier_loss_weight: float = 1.
    delta_r_scale: float = 1.
    delta_r_warmup: int = 0
    unnormalized_delta_r: bool = False
    use_importance_weights: bool = False
    iw_clip: float = 20.

class DARCIQL:
    def __init__(self, cfg):
        self.cfg, self.device = cfg, cfg.device
        self.q1 = CriticQ(cfg.s_dim, cfg.a_dim).to(cfg.device)
        self.q2 = CriticQ(cfg.s_dim, cfg.a_dim).to(cfg.device)
        self.q1_t = CriticQ(cfg.s_dim, cfg.a_dim).to(cfg.device)
        self.q2_t = CriticQ(cfg.s_dim, cfg.a_dim).to(cfg.device)
        self.q1_t.load_state_dict(self.q1.state_dict()); self.q2_t.load_state_dict(self.q2.state_dict())
        self.v = ValueV(cfg.s_dim).to(cfg.device)
        self.policy = GaussianPolicy(cfg.s_dim, cfg.a_dim, max_action=cfg.max_action).to(cfg.device)
        self.disc = DarcDiscriminator(cfg.s_dim, cfg.a_dim, cfg.input_noise).to(cfg.device)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=cfg.lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.opt_disc = torch.optim.Adam(self.disc.parameters(), lr=cfg.disc_lr)
        self.training_count = 0

    def update(self, source, target):
        cfg = self.cfg
        self.training_count += 1
        step = self.training_count
        s_s, a_s, r_s, s2_s, d_s = source
        s_t, a_t, r_t, s2_t, d_t = target
        bs_half = len(s_s)
        y = torch.cat([torch.zeros(bs_half, dtype=torch.long, device=cfg.device),
                       torch.ones(len(s_t), dtype=torch.long, device=cfg.device)])
        s_cat, a_cat, s2_cat = [torch.cat(v) for v in [(s_s,s_t), (a_s,a_t), (s2_s,s2_t)]]
        sa_logits, sas_logits = self.disc(s_cat, a_cat, s2_cat, training=True)
        sa_loss, sas_loss = F.cross_entropy(sa_logits, y), F.cross_entropy(sas_logits, y)
        disc_loss = cfg.classifier_loss_weight*(sa_loss+sas_loss)
        self.opt_disc.zero_grad(set_to_none=True); disc_loss.backward(); self.opt_disc.step()
        with torch.no_grad():
            sa_logits_s, sas_logits_s = self.disc(s_s, a_s, s2_s, training=False)
            log_sa, log_sas = F.log_softmax(sa_logits_s, dim=-1), F.log_softmax(sas_logits_s, dim=-1)
            if cfg.unnormalized_delta_r:
                delta_r = (log_sas[:,1]-log_sas[:,0]).unsqueeze(-1)
            else:
                delta_r = (log_sas[:,1]-log_sas[:,0]-log_sa[:,1]+log_sa[:,0]).unsqueeze(-1)
            warmup_scale = 0. if step < cfg.delta_r_warmup else 1.
            r_s_corr = r_s+cfg.delta_r_scale*warmup_scale*delta_r
            if cfg.use_importance_weights:
                sa_p, sas_p = self.disc.probs(s_s,a_s,s2_s)
                iw = ((sas_p[:,1]*sa_p[:,0])/(sas_p[:,0]*sa_p[:,1]+1e-8)).clamp(0.,cfg.iw_clip).unsqueeze(-1)
            else:
                iw = torch.ones_like(r_s_corr)
        s,a,r,d,s2,w = [torch.cat(v) for v in [(s_s,s_t),(a_s,a_t),(r_s_corr,r_t),(d_s,d_t),(s2_s,s2_t),(iw,torch.ones_like(r_t))]]
        with torch.no_grad():
            y_q = r+cfg.gamma*(1.-d)*self.v(s2)
        q_loss = (w*(self.q1(s,a)-y_q).pow(2)).mean()+(w*(self.q2(s,a)-y_q).pow(2)).mean()
        self.opt_q.zero_grad(set_to_none=True); q_loss.backward(); self.opt_q.step()
        with torch.no_grad():
            q_min = torch.minimum(self.q1(s,a),self.q2(s,a))
        v_loss = expectile_loss(q_min-self.v(s), cfg.iql_expectile)
        self.opt_v.zero_grad(set_to_none=True); v_loss.backward(); self.opt_v.step()
        with torch.no_grad():
            adv = q_min-self.v(s)
            weights_aw = torch.exp(cfg.iql_beta*adv).clamp(max=cfg.iql_max_weight)
        actor_loss = (-weights_aw*self.policy.log_prob_of_data_action(s,a)).mean()
        self.opt_pi.zero_grad(set_to_none=True); actor_loss.backward(); self.opt_pi.step()
        with torch.no_grad():
            for source_net, target_net in [(self.q1,self.q1_t),(self.q2,self.q2_t)]:
                for p, pt in zip(source_net.parameters(),target_net.parameters()):
                    pt.mul_(1.-cfg.tau).add_(p,alpha=cfg.tau)
        return {"loss_q_total":float(q_loss.item()), "loss_v_total":float(v_loss.item()),
                "loss_pi":float(actor_loss.item()), "loss_disc_total":float(disc_loss.item()),
                "delta_r_mean":float(delta_r.mean().item()), "src_reward_corr_mean":float(r_s_corr.mean().item())}
