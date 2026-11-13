from pathlib import Path
import random
import numpy as np
import torch
from offline.dataset import validate_contract
from .registry import OfflineLearner, OnlineSAC, shared_task, TDAL_ALGORITHMS, VGDF_ALGORITHMS, NORMALIZATION_VERSION
from .tdal import CCPWeightSACAgent, CDCQAgent
from .tdal import CDC_GATE_SCOPE

CHECKPOINT_VERSION = "auv-checkpoint-v1"

def _array_state(x):
    return {"dtype":str(x.dtype),"values":x.tolist()}

def _array_load(x):
    return np.asarray(x["values"],dtype=x["dtype"])


def _normalization_state(agent):
    norm = getattr(agent,"observation_normalization",None)
    if getattr(agent.cfg,"normalize_states",False) and norm is None:
        raise ValueError("Cannot save normalize_states=True before prepare() fits observation statistics")
    if norm is None:
        return None
    return {"version":norm["version"],"epsilon":norm["epsilon"],
            "mean":_array_state(norm["mean"]),"std":_array_state(norm["std"])}


def _restore_normalization(agent, saved):
    enabled = getattr(agent.cfg,"normalize_states",False)
    if not enabled:
        if saved is not None:
            raise ValueError("Checkpoint normalization metadata conflicts with normalize_states=False")
        return
    if not isinstance(saved,dict) or saved.get("version") != NORMALIZATION_VERSION:
        raise ValueError("Normalized checkpoint is missing compatible observation normalization statistics")
    try:
        mean,std = _array_load(saved["mean"]),_array_load(saved["std"])
    except (KeyError,TypeError,ValueError) as e:
        raise ValueError("Invalid checkpoint observation normalization statistics") from e
    epsilon = 1e-3 if agent.algorithm.startswith("igdf") else 1e-6
    shape = (agent.schema.obs_dim,)
    if (mean.shape != shape or std.shape != shape or mean.dtype != np.float32 or std.dtype != np.float32
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std<=0)
            or saved.get("epsilon") != epsilon):
        raise ValueError(f"Checkpoint normalization requires finite float32 mean/std shape {shape}, positive std and the original epsilon")
    agent.observation_normalization = {"version":NORMALIZATION_VERSION,"epsilon":epsilon,"mean":mean,"std":std}

def _module_state(module):
    out = {"weights":module.state_dict(),"training":module.training,
           "requires_grad":{k:p.requires_grad for k,p in module.named_parameters()},"extra":{}}
    for name in ("x_norm","y_norm"):
        norm = getattr(module,name,None)
        if norm is not None:
            out["extra"][name] = {k:None if getattr(norm,k) is None else _array_state(getattr(norm,k)) for k in ("mean","std")}
    for name in ("folds","train_pools","train_subpools","val_pools","calib_sorted_scores"):
        if hasattr(module,name):
            out["extra"][name] = [_array_state(x) for x in getattr(module,name)]
    for name in ("_fold_n","_fold_seed","_val_ratio"):
        if hasattr(module,name):
            out["extra"][name] = getattr(module,name)
    optimizer = getattr(module,"_optimizer",None)
    if optimizer is not None:
        out["extra"]["_optimizer"] = optimizer.state_dict()
    if hasattr(module,"optimizers"):
        out["extra"]["optimizers"] = [x.state_dict() for x in module.optimizers]
    return out

def _restore_module(module, state):
    module.load_state_dict(state["weights"],strict=True)
    module.train(state["training"])
    for name,p in module.named_parameters():
        p.requires_grad_(state["requires_grad"][name])
    for name,value in state["extra"].items():
        if name in {"x_norm","y_norm"}:
            for k,v in value.items():
                setattr(getattr(module,name),k,None if v is None else _array_load(v))
        elif name in {"folds","train_pools","train_subpools","val_pools","calib_sorted_scores"}:
            setattr(module,name,[_array_load(x) for x in value])
        elif name == "optimizers":
            for optimizer, saved in zip(module.optimizers,value):
                optimizer.load_state_dict(saved)
        elif name == "_optimizer":
            module._optimizer = torch.optim.Adam(module.parameters())
            module._optimizer.load_state_dict(value)
        else:
            setattr(module,name,value)

def _snapshot(obj):
    out = {"modules":{},"optimizers":{},"tensors":{},"scalars":{},"arrays":{},"nested":{}}
    for name,value in vars(obj).items():
        if isinstance(value,torch.nn.Module):
            out["modules"][name] = _module_state(value)
        elif isinstance(value,torch.optim.Optimizer):
            out["optimizers"][name] = value.state_dict()
        elif torch.is_tensor(value):
            out["tensors"][name] = value.detach().cpu().clone()
        elif isinstance(value,np.ndarray):
            out["arrays"][name] = _array_state(value)
        elif isinstance(value,(int,float,bool)) and name not in {"device"}:
            out["scalars"][name] = value
        elif name == "trainer":
            out["nested"][name] = _snapshot(value)
            if hasattr(value,"actor_lr_schedule"):
                out["nested"][name]["scheduler"] = value.actor_lr_schedule.state_dict()
    return out

def _restore(obj,state):
    for name,value in state["modules"].items():
        _restore_module(getattr(obj,name),value)
    for name,value in state["optimizers"].items():
        getattr(obj,name).load_state_dict(value)
    for name,value in state["tensors"].items():
        with torch.no_grad():
            getattr(obj,name).copy_(value.to(getattr(obj,name).device))
    for name,value in state["scalars"].items():
        setattr(obj,name,value)
    for name,value in state["arrays"].items():
        setattr(obj,name,_array_load(value))
    for name,value in state["nested"].items():
        child = getattr(obj,name)
        _restore(child,value)
        if "scheduler" in value:
            child.actor_lr_schedule.load_state_dict(value["scheduler"])

def save_checkpoint(agent, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np_rng = np.random.get_state()
    state = {"checkpoint_version":CHECKPOINT_VERSION,"contract":agent.contract,
             "algorithm":agent.algorithm,"options":agent.options,"resolved_config":agent.resolved_config,
             "observation_normalization":_normalization_state(agent),
             "seed":agent.seed,"updates":agent.updates,"dataset_refs":getattr(agent,"dataset_refs",{}),
             "core":_snapshot(agent.core),"extra":extra or {},
             "rng":{"numpy":[np_rng[0],np_rng[1].tolist(),np_rng[2],np_rng[3],np_rng[4]],
                    "torch":torch.get_rng_state(),"python":random.getstate()}}
    if torch.cuda.is_initialized():
        state["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    if getattr(agent,"helper",None) is not None:
        state["helper"] = _snapshot(agent.helper)
    if getattr(agent,"soft_weights",None) is not None:
        state["soft_weights"] = _array_state(agent.soft_weights)

    with path.open("xb") as f:
        torch.save(state,f)

def load_checkpoint(path, env, *, device="cpu", allow_dynamics_transfer=False, restore_rng=False):
    state = torch.load(path,map_location=device,weights_only=False)
    if not isinstance(state,dict) or state.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint format: expected auv-checkpoint-v1 with observation/action schema")
    validate_contract(state["contract"],env)
    if state["algorithm"] in TDAL_ALGORITHMS:
        if state.get("resolved_config", {}).get("cdc_gate_scope") != CDC_GATE_SCOPE:
            raise ValueError("Unsupported CDC gate scope: expected source-minibatch metadata")
        expected_reference = (CDCQAgent if state["algorithm"] == "cdc_q" else CCPWeightSACAgent).SOURCE_WEIGHT_REFERENCE
        actual_reference = state.get("resolved_config", {}).get("source_weight_reference", CCPWeightSACAgent.SOURCE_WEIGHT_REFERENCE)
        if actual_reference != expected_reference:
            raise ValueError("Checkpoint source-weight reference differs from its algorithm identity")
    if state["algorithm"] in VGDF_ALGORITHMS:
        expected_scope = "all" if state["algorithm"] == "vgdf_all_iql" else "q"
        if state.get("resolved_config", {}).get("gate_scope", "q") != expected_scope:
            raise ValueError("Checkpoint VGDF gate_scope differs from its algorithm identity")
    if state["contract"]["config_sha256"] != env.contract()["config_sha256"]:
        if not allow_dynamics_transfer:
            raise ValueError("Checkpoint environment config differs; explicitly enable dynamics transfer for a source policy used in a target environment")
        if shared_task(state["contract"]["config"]) != shared_task(env.contract()["config"]):
            raise ValueError("Dynamics transfer requires the same physical parameters, task and reward")
    cls = OnlineSAC if state["algorithm"] == "online_sac" else OfflineLearner
    if cls is OnlineSAC:
        agent = cls(env,options=state["options"],device=device,seed=state["seed"])
    else:
        agent = cls(state["algorithm"],env,options=state["options"],device=device,seed=state["seed"])
    _restore_normalization(agent,state.get("observation_normalization"))
    try:
        _restore(agent.core,state["core"])
    except (KeyError,RuntimeError,AttributeError) as e:
        raise ValueError(f"Checkpoint network/state incompatible with {env.schema.obs_dim} observation and {env.schema.action_dim} action dimensions: {e}") from e
    agent.updates = int(state["updates"])
    agent.dataset_refs = state.get("dataset_refs",{})
    agent.loaded_checkpoint = str(Path(path).resolve())

    if restore_rng:
        r = state["rng"]["numpy"]
        np.random.set_state((r[0],np.asarray(r[1],np.uint32),r[2],r[3],r[4]))
        torch.set_rng_state(state["rng"]["torch"].cpu())
        random.setstate(state["rng"]["python"])
        if "cuda" in state["rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return agent
