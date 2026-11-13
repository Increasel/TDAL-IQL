from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from models.contrastive import ContrastiveInfo
from .igdf_core import GaussianPolicy, DeterministicPolicy, TwinQ, ValueFunction, ImplicitQLearning, merge_batch

@dataclass
class IGDFConfig:
    s_dim: int
    a_dim: int
    device: str = "cpu"
    batch_size: int = 512
    gamma: float = .99
    tau: float = .005
    lr: float = 3e-4
    iql_tau: float = .7
    iql_beta: float = 3.
    iql_deterministic: bool = False
    normalize_states: bool = False

    max_grad_steps: int = 200000
    repr_dim: int = 32
    ensemble_size: int = 1
    info_lr: float = 3e-4
    info_batch_size: int = 128
    info_steps: int = 7000
    repr_norm: bool = False
    repr_norm_temp: bool = False
    ortho_init: bool = False
    output_gain: float | None = None
    xi: float = .75
    alpha: float = 1.
    info_weight_clip: float = 100.
    source_weight_mode: str = "igdf_info"

class IGDFIQL:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.max_grad_steps < 1:
            raise ValueError("IGDF max_grad_steps must be positive")
        if cfg.ensemble_size != 1:
            raise ValueError("IGDF requires ensemble_size=1")
        self.info = ContrastiveInfo(state_dim=cfg.s_dim,action_dim=cfg.a_dim,repr_dim=cfg.repr_dim,
                                    ensemble_size=cfg.ensemble_size,repr_norm=cfg.repr_norm,
                                    repr_norm_temp=cfg.repr_norm_temp,ortho_init=cfg.ortho_init,
                                    output_gain=cfg.output_gain).to(cfg.device)
        self.info_opt = torch.optim.Adam(self.info.parameters(),lr=cfg.info_lr)
        self.qf = TwinQ(cfg.s_dim,cfg.a_dim).to(cfg.device)
        self.vf = ValueFunction(cfg.s_dim).to(cfg.device)
        cls = DeterministicPolicy if cfg.iql_deterministic else GaussianPolicy
        self.actor = cls(cfg.s_dim,cfg.a_dim,max_action=1.).to(cfg.device)
        self.trainer = ImplicitQLearning(1.,self.actor,torch.optim.Adam(self.actor.parameters(),lr=cfg.lr),
                         self.qf,torch.optim.Adam(self.qf.parameters(),lr=cfg.lr),self.vf,
                         torch.optim.Adam(self.vf.parameters(),lr=cfg.lr),iql_tau=cfg.iql_tau,
                         beta=cfg.iql_beta,max_steps=cfg.max_grad_steps,discount=cfg.gamma,tau=cfg.tau,device=cfg.device)
        self.prepared = False

    def prepare(self, source_buffer, target_buffer):
        cfg = self.cfg
        for _ in range(cfg.info_steps):
            s,a,_,s2,_ = target_buffer.sample(cfg.info_batch_size)
            _,_,_,negative,_ = source_buffer.sample(cfg.info_batch_size-1)
            candidates = torch.cat([s2.unsqueeze(1),negative.unsqueeze(0).expand(cfg.info_batch_size,-1,-1)],dim=1)
            logits = self.info(s.unsqueeze(1),a.unsqueeze(1),candidates).squeeze(1)
            labels = torch.zeros((cfg.info_batch_size,cfg.info_batch_size),device=cfg.device)
            labels[:,0] = 1.
            loss = F.binary_cross_entropy_with_logits(logits,labels).mean()
            self.info_opt.zero_grad(set_to_none=True); loss.backward(); self.info_opt.step()
        self.prepared = True

    def update(self, source, target, source_weights=None):
        if not self.prepared:
            raise RuntimeError("IGDF representation pretraining not complete")
        cfg = self.cfg
        s,a,r,s2,d = source
        with torch.no_grad():
            if cfg.repr_norm:
                score = torch.diag(self.info(s,a,s2)).reshape(-1,1)
            else:
                logits,sa_repr,ss_repr = self.info(s,a,s2,return_repr=True)
                score = torch.diag(logits).reshape(-1,1)/(torch.linalg.norm(sa_repr,dim=-1,keepdim=True)*torch.linalg.norm(ss_repr,dim=-1,keepdim=True)+1e-8)
            keep_n = int(round(len(s)*max(0.,min(1.,cfg.xi))))
            ids = torch.argsort(score[:,0])[-keep_n:] if keep_n else torch.zeros(0,device=cfg.device,dtype=torch.long)
            kept = [x[ids] for x in source]
            if cfg.source_weight_mode == "igdf_info":
                w = torch.exp(score[ids]*cfg.alpha).clamp(max=cfg.info_weight_clip)
            elif cfg.source_weight_mode == "none":
                w = torch.ones((keep_n,1),device=cfg.device)
            else:
                if source_weights is None:
                    raise RuntimeError("IGDF static V/Q soft weights have not been prepared")
                w = source_weights[ids]
            mask = torch.cat([w,torch.ones((len(target[0]),1),device=cfg.device)])
        log = self.trainer.train(merge_batch(kept,target),mask)
        return {**log,"src_keep_ratio":float(keep_n/len(s)),"src_w_mean":float(w.mean().item()) if keep_n else 0.}
