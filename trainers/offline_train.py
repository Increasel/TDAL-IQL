import argparse
from contextlib import contextmanager
import json
import random
import time
import numpy as np
import torch
from agents.registry import OfflineLearner,ALGORITHMS
from agents.checkpoint import save_checkpoint
from offline.dataset import TransitionDataset
from .common import build_env,set_seed,create_run_dir,write_json,read_options,project_path
from .evaluate import evaluate


@contextmanager
def evaluation_rng():

    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def finite_update(agent):
    log = agent.update()

    finite = {k:float(v) if np.isfinite(v) else None for k,v in log.items()}
    if any("loss" in k and v is None for k,v in finite.items()):
        raise FloatingPointError(f"Nonfinite {agent.algorithm} update: {log}")
    return finite


def progress(out, value):
    temporary = out/"progress.tmp"
    temporary.write_text(json.dumps(value,allow_nan=False,indent=2),encoding="utf-8")
    temporary.replace(out/"progress.json")

def run(args):
    if args.updates<1 or args.pretrain_updates<0:
        raise ValueError("Explicit positive update budget required")
    eval_steps = getattr(args,"eval_steps",[]) or []
    eval_seeds = getattr(args,"eval_seeds",[]) or []
    log_every = getattr(args,"log_every",1)
    purpose = getattr(args,"purpose","training")
    if (log_every<1 or len(set(eval_steps))!=len(eval_steps) or
            any(s<1 or s>args.updates for s in eval_steps) or (eval_steps and not eval_seeds)):
        raise ValueError("Invalid logging interval, evaluation steps or missing evaluation seeds")
    started = time.perf_counter()
    env = build_env(args.config); set_seed(args.seed)
    source = TransitionDataset.load(project_path(args.source),env)
    target = TransitionDataset.load(project_path(args.target),env)
    options = read_options(args.options)
    if args.algorithm.startswith("igdf"):

        options.setdefault("max_grad_steps",args.updates)
    agent = OfflineLearner(args.algorithm,env,options=options,device=args.device,seed=args.seed)
    out = create_run_dir(args.out)
    write_json(out/"run_config.json",{"purpose":purpose,"algorithm":args.algorithm,
        "seed":args.seed,"device":args.device,"main_updates":args.updates,
        "source_pretraining_updates":args.pretrain_updates if args.algorithm=="finetune_iql" else 0,
        "algorithm_config":agent.resolved_config,"contract":env.contract(),
        "source":str(project_path(args.source)),"target":str(project_path(args.target)),
        "eval_steps":eval_steps,"eval_seeds":eval_seeds,"log_every":log_every})
    progress(out,{"phase":"preparing","algorithm":args.algorithm,"total_gradient_steps":0})
    prepare_started = time.perf_counter()
    agent.prepare(source,target)
    preparation_s = time.perf_counter()-prepare_started
    print(json.dumps({"algorithm":args.algorithm,"phase":"prepared","preparation_s":preparation_s}),flush=True)
    logs, evaluations = [], []
    source_started = time.perf_counter()
    if args.algorithm == "finetune_iql":
        for step in range(1,args.pretrain_updates+1):
            log = finite_update(agent)
            if step==1 or step%log_every==0 or step==args.pretrain_updates:
                logs.append({"phase":"source","phase_step":step,"gradient_step":agent.updates,**log})
                progress(out,{"phase":"source","algorithm":args.algorithm,"phase_step":step,
                    "total_gradient_steps":agent.updates,"last_losses":log,"elapsed_s":time.perf_counter()-started})
        agent.begin_target_finetuning()
    source_pretraining_s = time.perf_counter()-source_started
    eval_env = build_env(args.config) if eval_steps else None
    main_started, evaluation_s = time.perf_counter(), 0.
    try:
        for step in range(1,args.updates+1):
            log = finite_update(agent)
            if step==1 or step%log_every==0 or step==args.updates:
                logs.append({"phase":"target" if args.algorithm=="finetune_iql" else "main",
                    "phase_step":step,"gradient_step":agent.updates,**log})
                row={"phase":"main","algorithm":args.algorithm,"main_step":step,
                    "total_gradient_steps":agent.updates,"last_losses":log,"elapsed_s":time.perf_counter()-started}
                progress(out,row)
                if log_every>1:
                    print(json.dumps(row,allow_nan=False),flush=True)
            if step in eval_steps:
                save_checkpoint(agent,out/f"policy_update_{step:06d}.pt",extra={"purpose":purpose,
                    "main_updates":step,"source_pretraining_updates":args.pretrain_updates if args.algorithm=="finetune_iql" else 0})
                eval_started = time.perf_counter()
                with evaluation_rng():
                    result=evaluate(agent,eval_env,eval_seeds)
                evaluation_s += time.perf_counter()-eval_started
                evaluation={"main_update":step,"total_gradient_steps":agent.updates,**result}
                evaluations.append(evaluation)
                with (out/"evaluations.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(evaluation,allow_nan=False)+"\n")
                print(json.dumps({"algorithm":args.algorithm,"evaluation_update":step,
                    "success_rate":result["success_rate"],"mean_return":result["mean_return"],
                    "mean_reached_targets":result["mean_reached_targets"]}),flush=True)
    finally:
        if eval_env is not None:
            eval_env.close()
    main_training_s = time.perf_counter()-main_started-evaluation_s
    obs,_ = env.reset(seed=args.seed+20000)
    action = agent.act(obs)
    save_checkpoint(agent,out/"policy.pt",extra={"purpose":purpose,"main_updates":args.updates,
        "source_pretraining_updates":args.pretrain_updates if args.algorithm=="finetune_iql" else 0})
    write_json(out/"summary.json",{"purpose":purpose,"algorithm":args.algorithm,
        "updates":agent.updates,"algorithm_config":agent.resolved_config,"contract":env.contract(),
        "main_updates":args.updates,"preparation_s":preparation_s,"source_pretraining_s":source_pretraining_s,
        "main_training_s":main_training_s,"evaluation_s":evaluation_s,"wall_time_s":time.perf_counter()-started,
        "dataset_refs":agent.dataset_refs,"action_example":action.tolist(),"update_log":logs,"evaluations":evaluations})
    progress(out,{"phase":"completed","algorithm":args.algorithm,"main_step":args.updates,
        "total_gradient_steps":agent.updates,"elapsed_s":time.perf_counter()-started})
    env.close()
    print(f"{args.algorithm}: {agent.updates} updates, action shape {action.shape}; {out}")
    return agent

def main(argv=None):
    p = argparse.ArgumentParser(description="Train an offline algorithm")
    p.add_argument("--algorithm",choices=ALGORITHMS,required=True)
    for name in ("config","source","target","out"):
        p.add_argument(f"--{name}",required=True)
    p.add_argument("--updates",type=int,required=True)
    p.add_argument("--pretrain-updates",type=int,default=0)
    p.add_argument("--options",default=""); p.add_argument("--device",default="cpu")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--eval-steps",nargs="+",type=int,default=[])
    p.add_argument("--eval-seeds",nargs="+",type=int,default=[])
    p.add_argument("--log-every",type=int,default=1)
    p.add_argument("--purpose",default="training")
    return run(p.parse_args(argv))

if __name__ == "__main__":
    main()
