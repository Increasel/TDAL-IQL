from dataclasses import asdict
import copy
import numpy as np
import torch
from envs.schema import fingerprint
from offline.dataset import TransitionDataset
from . import ccp, tdal, vgdf
from .darc import DARCConfig, DARCIQL
from .igdf import IGDFConfig, IGDFIQL
from .sac import SACContinuous, SACContinuousConfig

ALGORITHMS = ("iql", "target_only_iql", "source_only_iql", "mixed_iql", "finetune_iql",
              "tdal", "cdc_only", "cvc_only", "cdc_q", "vgdf_iql", "vgdf_all_iql", "darc_iql",
              "igdf_iql", "igdf_no_weights", "igdf_static_v", "igdf_static_q",
              "ccp_sac", "ccp_cql", "offline_sac", "cql")
TDAL_ALGORITHMS = ("tdal", "cdc_only", "cvc_only", "cdc_q")
VGDF_ALGORITHMS = ("vgdf_iql", "vgdf_all_iql")
NORMALIZATION_VERSION = "source-target-state-zscore-v1"

def _cfg(cls, derived, defaults, options):
    forbidden = set(options) & {"s_dim", "a_dim", "state_dim", "action_dim"}
    if forbidden:
        raise ValueError(f"Dimensions must come from the environment/schema, not options: {forbidden}")
    return cls(**{**defaults, **options, **derived})

def shared_task(config):
    return {k:config[k] for k in ("version","physics","task","observation","reward")}

class TensorDatasetBuffer:
    def __init__(self, data, device, arrays=None):
        self.data, self.device = data, device
        s,a,r,d,s2 = data.core_arrays() if arrays is None else arrays
        self.values = [s,a,r,s2,d]

    def sample_indices(self, n):
        return np.random.randint(0,len(self.data),size=n)

    def get(self, ids):
        return [torch.as_tensor(x[ids],device=self.device) for x in self.values]

    def sample(self, n):
        return self.get(self.sample_indices(n))

@torch.no_grad()
def static_q_discrepancy(agent,s,a):
    fake_next = agent.dynamics.predict_next_states(agent.sa_encoder(s,a),agent.state_encoder(s),s)
    q1,q2 = agent.static_q_targ(s,a)
    observed = torch.minimum(q1,q2).view(-1,1)
    k,b,_ = fake_next.shape
    a_rep = a.unsqueeze(0).expand(k,-1,-1).reshape(k*b,-1)
    q1,q2 = agent.static_q_targ(fake_next.reshape(k*b,-1),a_rep)
    values = torch.minimum(q1,q2).reshape(k,b,1).transpose(0,1)
    mu,var = values.mean(dim=1),values.var(dim=1,unbiased=False)
    return (((observed-mu).pow(2)/(var+1e-6)).sqrt(),mu,var)

class OnlineSAC:
    algorithm = "online_sac"
    def __init__(self, env, options=None, device="cpu", seed=0):
        self.contract = env.contract()
        self.schema, self.seed = env.schema, int(seed)
        self.options = dict(options or {})
        self.device = device
        self.cfg = _cfg(SACContinuousConfig,{"state_dim":env.schema.obs_dim,"action_dim":env.schema.action_dim},{},self.options)
        self.core = SACContinuous(self.cfg,device=device)
        self.updates = 0

    def act(self, observation, deterministic=True):
        self.schema.validate_observation(observation)
        if np.asarray(observation).shape != (self.schema.obs_dim,):
            raise ValueError("act() expects one observation, not a batch")
        action = self.core.select_action(observation,eval_mode=deterministic).astype(np.float32)
        self.schema.validate_action(action)
        return action

    def update(self, batch):
        n = len(batch["s"])
        expected = {"s":(n,self.schema.obs_dim),"s2":(n,self.schema.obs_dim),"a":(n,self.schema.action_dim),"r":(n,1),"d":(n,1)}
        for key, shape in expected.items():
            if np.asarray(batch[key]).shape != shape or not np.all(np.isfinite(batch[key])):
                raise ValueError(f"SAC {key}: expected finite {shape}")
        if "terminated" not in batch or not np.array_equal(batch["d"],batch["terminated"]):
            raise ValueError("SAC batch d must equal terminated; do not combine truncation with termination")
        if np.any(np.abs(batch["a"])>1.) or not np.all((batch["d"]==0)|(batch["d"]==1)):
            raise ValueError("SAC batch action bounds or binary termination violated")
        self.updates += 1
        return self.core.update(batch)

    @property
    def resolved_config(self):
        return asdict(self.cfg)

class OfflineLearner:
    def __init__(self, algorithm, env, options=None, device="cpu", seed=0):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {algorithm}; available {ALGORITHMS}")
        self.algorithm, self.contract = algorithm, env.contract()
        self.schema, self.seed, self.device = env.schema, int(seed), device
        self.options = copy.deepcopy(options or {})
        options = copy.deepcopy(self.options)
        dims = {"s_dim":self.schema.obs_dim,"a_dim":self.schema.action_dim,"device":device}
        self.updates = 0
        self.ready = False
        self.dataset_refs = {}
        self.helper = None
        self.soft_weights = None
        self.observation_normalization = None
        if algorithm in TDAL_ALGORITHMS:
            defaults = {"batch_size":512}
            if algorithm == "cdc_only":
                defaults["value_temperature"] = 0.
            if algorithm == "cvc_only":
                defaults["xi"] = 1.
            self.cfg = _cfg(tdal.AgentConfig,dims,defaults,options)
            if self.cfg.k_folds < 2:
                raise ValueError("TDAL CVC variance with denominator K-1 requires k_folds>=2")
            core_class = tdal.CDCQAgent if algorithm == "cdc_q" else tdal.CCPWeightSACAgent
            self.core = core_class(self.cfg)
        elif algorithm in VGDF_ALGORITHMS:
            dyn_options = options.pop("dynamics",{})
            self.batch_size = int(options.pop("batch_size",512))
            self.dyn_cfg = vgdf.DynamicsConfig(**dyn_options)
            gate_scope = "all" if algorithm == "vgdf_all_iql" else "q"
            if "gate_scope" in options and options["gate_scope"] != gate_scope:
                raise ValueError("VGDF algorithm name and gate_scope disagree; select vgdf_iql or vgdf_all_iql explicitly")
            self.cfg = _cfg(vgdf.VGDFIQLConfig,{"state_dim":self.schema.obs_dim,"action_dim":self.schema.action_dim,
                            "max_action":1.,"device":device},{"gate_scope":gate_scope},options)
            dynamics = vgdf.TargetDynamicsEnsemble(self.schema.obs_dim,self.schema.action_dim,self.dyn_cfg,device).to(device)
            self.core = vgdf.VGDFIQLTrainer(self.cfg,dynamics)
        elif algorithm.startswith("igdf"):
            self.helper_options = options.pop("helper",{})
            default_mode = {"igdf_iql":"igdf_info","igdf_no_weights":"none", "igdf_static_v":"tcdw_static_v","igdf_static_q":"tcdw_static_q"}[algorithm]
            self.cfg = _cfg(IGDFConfig,dims,{"source_weight_mode":default_mode},options)
            self.core = IGDFIQL(self.cfg)
        elif algorithm == "darc_iql":
            self.cfg = _cfg(DARCConfig,dims,{},options)
            self.core = DARCIQL(self.cfg)
        else:
            backbone = "sac" if algorithm in {"offline_sac","ccp_sac"} else "cql" if algorithm in {"cql","ccp_cql"} else "iql"
            defaults = {"algo":backbone,"batch_size":512,"xi":.75}
            if algorithm in {"iql","target_only_iql","source_only_iql","mixed_iql","finetune_iql"}:
                defaults["xi"] = 0.
            if algorithm in {"offline_sac","cql"}:
                defaults["xi"] = 1.
            if backbone in {"sac","cql"}:
                defaults.update({"cql_alpha":5.,"bc_coef":1.})
            self.cfg = _cfg(ccp.AgentConfig,dims,defaults,options)
            self.core = ccp.CCPWeightSACAgent(self.cfg)
        if not hasattr(self,"batch_size"):
            self.batch_size = self.cfg.batch_size
        if self.batch_size<2 or self.batch_size%2:
            raise ValueError("Offline batch_size must be positive and even")
        if not isinstance(getattr(self.cfg,"normalize_states",False),bool):
            raise ValueError("normalize_states must be a boolean")

    @property
    def resolved_config(self):
        cfg = asdict(self.cfg)
        if self.algorithm in TDAL_ALGORITHMS:
            cfg["source_weight_reference"] = self.core.SOURCE_WEIGHT_REFERENCE
            cfg["source_weight_formula"] = self.core.SOURCE_WEIGHT_FORMULA
        if self.algorithm in VGDF_ALGORITHMS:
            cfg["dynamics"] = asdict(self.dyn_cfg)
            cfg["batch_size"] = self.batch_size
        return cfg

    def _load_core(self, source, target):
        self.core.load_source_dataset(*self._training_arrays(source))
        self.core.load_target_dataset(*self._training_arrays(target))

    def _fit_observation_normalization(self, source, target):
        self.observation_normalization = None
        if not getattr(self.cfg,"normalize_states",False):
            return

        states = np.concatenate([source.arrays["observations"],target.arrays["observations"]])
        epsilon = 1e-3 if self.algorithm.startswith("igdf") else 1e-6
        self.observation_normalization = {
            "version":NORMALIZATION_VERSION,"epsilon":epsilon,
            "mean":states.mean(axis=0).astype(np.float32),
            "std":(states.std(axis=0)+epsilon).astype(np.float32),
        }

    def _normalize_observations(self, observations):
        if not getattr(self.cfg,"normalize_states",False):
            return observations
        norm = self.observation_normalization
        if norm is None:
            raise RuntimeError("normalize_states requires fitted statistics from prepare() or a checkpoint")
        return (np.asarray(observations)-norm["mean"])/norm["std"]

    def _training_arrays(self, data):
        s,a,r,d,s2 = data.core_arrays()
        return self._normalize_observations(s),a,r,d,self._normalize_observations(s2)

    def _single_domain(self, data):
        arrays = data.core_arrays()
        self.core.tar_buffer.load_from_arrays(*arrays)
        self.core.src_buffer.load_from_arrays(*arrays,w=np.zeros((len(data),1),dtype=np.float32))
        self.core.cfg.xi = 0.

    def prepare(self, source: TransitionDataset, target: TransitionDataset):
        source.validate(self.contract); target.validate(self.contract)
        for label, dataset in [("source",source),("target",target)]:
            if shared_task(dataset.metadata["config"]) != shared_task(self.contract["config"]):
                raise ValueError(f"{label} dataset task/physics/reward differs from learner; only documented dynamics shifts may differ")
        if source.metadata["config"]["domain"] != "source" or target.metadata["config"]["domain"] != "target":
            raise ValueError("Source/target dataset roles do not match their recorded environment domains")
        if target.metadata["config_sha256"] != self.contract["config_sha256"]:
            raise ValueError("Target dataset config differs from requested target environment")
        self.source, self.target = source, target
        self._fit_observation_normalization(source,target)
        self.src_tensor = TensorDatasetBuffer(source,self.device,self._training_arrays(source))
        self.tar_tensor = TensorDatasetBuffer(target,self.device,self._training_arrays(target))
        self.dataset_refs = {label:{"config_sha256":d.metadata["config_sha256"],"transitions":len(d),
                                  "seed":d.metadata["seed"],"transition_sha256":dataset_digest(d)}
                             for label,d in [("source",source),("target",target)]}
        name = self.algorithm
        if name in {"iql","target_only_iql"}:
            self._single_domain(target)
        elif name in {"source_only_iql","finetune_iql"}:
            self._single_domain(source)
            self.finetune_phase = "source"
        elif name == "mixed_iql":
            arrays = [np.concatenate([s,t]) for s,t in zip(source.core_arrays(),target.core_arrays())]
            self.core.tar_buffer.load_from_arrays(*arrays)
            self.core.src_buffer.load_from_arrays(*arrays,w=np.zeros((len(source)+len(target),1),np.float32))
        elif name in TDAL_ALGORITHMS:
            self._load_core(source,target)
            self.core.train_dynamics(self.seed)
            self.core.train_static_value_model(self.seed)
            self.core.precompute_static_source_weights()
        elif name in VGDF_ALGORITHMS:
            s,a,r,d,s2 = target.core_arrays()
            self.core.dynamics.fit(s,a,r,s2,self.seed)
        elif name.startswith("igdf"):
            self.core.prepare(self.src_tensor,self.tar_tensor)
            if name in {"igdf_static_v","igdf_static_q"}:
                helper_cfg = _cfg(tdal.AgentConfig,{"s_dim":self.schema.obs_dim,"a_dim":self.schema.action_dim,"device":self.device},
                                  {"xi":1.,"gamma":self.cfg.gamma,"tau":self.cfg.tau,"lr":self.cfg.lr,
                                   "batch_size":self.cfg.batch_size,"iql_expectile":self.cfg.iql_tau,
                                   "iql_beta":self.cfg.iql_beta},self.helper_options)
                self.helper = tdal.CCPWeightSACAgent(helper_cfg)
                self.helper.load_source_dataset(*self._training_arrays(source))
                self.helper.load_target_dataset(*self._training_arrays(target))
                self.helper.train_dynamics(self.seed); self.helper.train_static_value_model(self.seed)
                with torch.no_grad():
                    values = []
                    for start in range(0,len(source),max(helper_cfg.dyn_batch_size,1024)):
                        s,a,_,s2,_ = self.src_tensor.get(np.arange(start,min(start+max(helper_cfg.dyn_batch_size,1024),len(source))))
                        d = static_q_discrepancy(self.helper,s,a)[0] if name.endswith("_q") else self.helper._compute_value_discrepancy_static(s,a,s2,unbiased=False)[0]

                        values.append(torch.exp(-d/max(helper_cfg.value_temperature,1e-6)).cpu().numpy())
                    self.soft_weights = np.concatenate(values).astype(np.float32)
        elif name.startswith("ccp"):
            self._load_core(source,target); self.core.train_dynamics(self.seed)
        elif name in {"offline_sac","cql"}:
            self._load_core(source,target)
            self.core.src_buffer.set_weights(np.ones((len(source),1),np.float32))
        self.ready = True

    def begin_target_finetuning(self):
        if self.algorithm != "finetune_iql" or not self.ready:
            raise RuntimeError("Only a prepared finetune_iql learner can switch phase")
        self._single_domain(self.target)
        self.finetune_phase = "target"

    def update(self):
        if not self.ready:
            raise RuntimeError("Call prepare() with validated datasets before updating")
        self.updates += 1
        if self.algorithm in VGDF_ALGORITHMS:
            return self.core.train_step(self.src_tensor.sample(self.batch_size//2),self.tar_tensor.sample(self.batch_size//2),self.updates)
        if self.algorithm.startswith("igdf"):
            ids = self.src_tensor.sample_indices(self.batch_size//2)
            w = None if self.soft_weights is None else torch.as_tensor(self.soft_weights[ids],device=self.device)
            return self.core.update(self.src_tensor.get(ids),self.tar_tensor.sample(self.batch_size//2),w)
        if self.algorithm == "darc_iql":
            return self.core.update(self.src_tensor.sample(self.batch_size//2),self.tar_tensor.sample(self.batch_size//2))
        self.core.train_ac()
        return dict(self.core.loss_log)

    def act(self, observation, deterministic=True):
        self.schema.validate_observation(observation)
        if np.asarray(observation).shape != (self.schema.obs_dim,):
            raise ValueError("act() expects one observation, not a batch")
        observation = self._normalize_observations(observation)
        if self.algorithm in VGDF_ALGORITHMS:
            action = self.core.actor.act(np.asarray(observation),self.device)
        elif self.algorithm.startswith("igdf"):
            actor = self.core.actor
            training = actor.training
            actor.train(not deterministic)
            action = actor.act(np.asarray(observation),self.device)
            actor.train(training)
        elif self.algorithm == "darc_iql":
            action = self.core.policy.act(np.asarray(observation),deterministic=deterministic)
        else:
            action = self.core.policy.act(np.asarray(observation),deterministic=deterministic)
        action = np.asarray(action,dtype=np.float32)
        self.schema.validate_action(action)
        return action

def dataset_digest(dataset):
    import hashlib
    h = hashlib.sha256()
    for key in sorted(dataset.arrays):
        x = np.ascontiguousarray(dataset.arrays[key])
        h.update(key.encode()); h.update(str(x.shape).encode()); h.update(str(x.dtype).encode()); h.update(x.tobytes())
    return h.hexdigest()
