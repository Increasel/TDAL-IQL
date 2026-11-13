import argparse
import json
import time
import numpy as np
import torch
from agents.registry import OnlineSAC
from agents.checkpoint import save_checkpoint
from offline.dataset import ReplayBuffer
from .common import build_env,set_seed,create_run_dir,write_json,read_options,project_path
from .evaluate import evaluate

def write_progress(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)

def run(args, *, evaluation_stop=None):
    if args.steps<1 or args.batch_size<2 or args.start_steps<0 or args.replay_capacity<args.batch_size or args.update_every<0:
        raise ValueError("Invalid SAC interaction/update budget")
    checkpoint_every = getattr(args,"checkpoint_every",0)
    eval_every = getattr(args,"eval_every",0)
    progress_every = getattr(args,"progress_every",0)
    log_every = getattr(args,"log_every",1)
    eval_seeds = getattr(args,"eval_seeds",None) or []
    purpose = getattr(args,"purpose","training")
    learning_reward_scale = getattr(args,"learning_reward_scale",1.)
    initial_alpha = getattr(args,"initial_alpha",1.)
    success_stop_evals = getattr(args,"stop_after_successful_evals",0)
    success_threshold = getattr(args,"success_rate_threshold",1.)
    if (not np.isfinite(learning_reward_scale) or learning_reward_scale<=0 or
            not np.isfinite(initial_alpha) or initial_alpha<=0 or success_stop_evals<0 or
            not 0<success_threshold<=1 or (success_stop_evals and not eval_every)):
        raise ValueError("Invalid SAC reward/entropy scale or success stopping settings")
    if min(checkpoint_every,eval_every,progress_every)<0 or log_every<1 or (eval_every and not eval_seeds):
        raise ValueError("Invalid progress/checkpoint/evaluation intervals or missing eval seeds")
    env = build_env(args.config)
    options = read_options(args.options)
    set_seed(args.seed)
    agent = OnlineSAC(env,options=options,device=args.device,seed=args.seed)
    with torch.no_grad():
        agent.core.log_alpha.fill_(float(np.log(initial_alpha)))
    initial_updates = agent.updates
    random_steps = args.start_steps
    out = create_run_dir(args.out)
    eval_env = build_env(args.config) if eval_every else None
    buffer = ReplayBuffer(min(args.steps,args.replay_capacity),env,seed=args.seed,
                          collection={"purpose":purpose,"policy":"online_sac_training_replay",
                                      "steps_requested":args.steps,"competence_validated":False})
    episode = 0
    obs,_ = env.reset(seed=args.seed)
    log = []
    episode_returns = []
    episode_return = 0.
    episode_start_step = 0
    evaluations = []
    started = time.perf_counter()
    interaction_config = {"steps":args.steps,"start_steps":args.start_steps,"batch_size":args.batch_size,
                          "update_every":args.update_every,"replay_capacity":args.replay_capacity,
                          "learning_reward_scale":learning_reward_scale,"initial_alpha":initial_alpha,
                          "random_action_steps":random_steps,
                          "stop_after_successful_evals":success_stop_evals,"success_rate_threshold":success_threshold}
    write_json(out/"run_config.json",{"purpose":purpose,"contract":env.contract(),"seed":args.seed,
               "algorithm_config":agent.resolved_config,"device":args.device,
               "interaction_config":interaction_config,"checkpoint_every":checkpoint_every,
               "eval_every":eval_every,"eval_seeds":eval_seeds,"progress_every":progress_every,"log_every":log_every})
    consecutive_successes = 0
    stop_reason = "interaction_budget"
    for step in range(args.steps):
        action = env.action_space.sample() if step<random_steps else agent.act(obs,deterministic=False)
        obs2,reward,terminated,truncated,info = env.step(action)
        episode_return += reward
        buffer.add(obs,action,reward,obs2,terminated,truncated,info,episode,args.seed+episode,
                   collector_cut=step==args.steps-1 and not (terminated or truncated))


        if len(buffer)>=max(args.start_steps,1):
            for _ in range(args.update_every):
                batch = buffer.sample(args.batch_size)
                batch["r"] = batch["r"]*learning_reward_scale
                losses = agent.update(batch)
                if not all(np.isfinite(v) for v in losses.values()):
                    raise FloatingPointError(f"Nonfinite SAC update: {losses}")
                if agent.updates%log_every==0 or agent.updates==initial_updates+1:
                    log.append({"interaction_step":step+1,"gradient_step":agent.updates,**losses})
        obs = obs2
        if terminated or truncated:
            episode_returns.append(episode_return)
            row={"episode":episode,"episode_seed":args.seed+episode,"interaction_step":step+1,
                 "steps":step+1-episode_start_step,"return":episode_return,
                 "success":info["success"],"reached_targets":info["reached_targets"],
                 "path_length_m":info["path_length_m"],"termination_reason":info["termination_reason"]}
            with (out/"episodes.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(row,allow_nan=False)+"\n")
            if step<args.steps-1:
                episode += 1; obs,_ = env.reset(seed=args.seed+episode)
                episode_return=0.;episode_start_step=step+1
        if checkpoint_every and (step+1)%checkpoint_every==0:
            save_checkpoint(agent,out/f"policy_step_{step+1:06d}.pt",extra={
                "interaction_steps":step+1,"purpose":purpose,"interaction_config":interaction_config})
        if eval_every and (step+1)%eval_every==0:


            cuda_devices = [agent.core.device.index or 0] if agent.core.device.type=="cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                result=evaluate(agent,eval_env,eval_seeds)
            evaluation={"interaction_step":step+1,"gradient_step":agent.updates,**result}
            evaluations.append(evaluation)
            with (out/"evaluations.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(evaluation,allow_nan=False)+"\n")
            print(json.dumps({"evaluation_step":step+1,"mean_return":result["mean_return"],
                  "success_rate":result["success_rate"],"mean_reached_targets":result["mean_reached_targets"]}),flush=True)
            consecutive_successes = consecutive_successes+1 if result["success_rate"]>=success_threshold else 0
            if success_stop_evals and consecutive_successes>=success_stop_evals:
                stop_reason = "consecutive_successful_evaluations"
            if evaluation_stop is not None:
                requested_reason = evaluation_stop(evaluation)
                if requested_reason:
                    stop_reason = str(requested_reason)
        stopping = stop_reason != "interaction_budget"
        if progress_every and ((step+1)%progress_every==0 or step+1==args.steps or stopping):
            progress={"interaction_step":step+1,"gradient_step":agent.updates,"episodes_completed":len(episode_returns),
                      "recent_mean_return":float(np.mean(episode_returns[-20:])) if episode_returns else None,
                      "last_losses":log[-1] if log else None,"elapsed_s":time.perf_counter()-started,
                      "latest_evaluation":evaluations[-1] if evaluations else None}
            write_progress(out/"progress.json",progress)
            print(json.dumps({k:v for k,v in progress.items() if k!="latest_evaluation"}),flush=True)
        if stopping:
            break
    completed_steps = step+1
    last_row = buffer.rows[(buffer.pointer-1)%len(buffer.rows)]
    if not (last_row["terminated"] or last_row["truncated"]):
        last_row["collector_cut"] = True
    save_checkpoint(agent,out/"policy.pt",extra={"interaction_steps":completed_steps,"purpose":purpose,
                                              "interaction_config":interaction_config})
    buffer.dataset().save(out/"replay.npz")
    summary = {"purpose":purpose,"steps":completed_steps,"updates":agent.updates,
               "updates_added":agent.updates-initial_updates,"initial_updates":initial_updates,
               "stop_reason":stop_reason,
               "schema_sha256":env.schema.hash,"observation_dim":env.schema.obs_dim,"action_dim":env.schema.action_dim,
               "config":env.config.to_dict(),"algorithm_config":agent.resolved_config,
               "interaction_config":interaction_config,"update_log":log,"evaluations":evaluations,
               "completed_episodes":len(episode_returns),"wall_time_s":time.perf_counter()-started}
    write_json(out/"summary.json",summary)
    env.close()
    if eval_env is not None:
        eval_env.close()
    print(f"SAC: {completed_steps} interactions, {agent.updates-initial_updates} new updates; {out}")
    return summary

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True); p.add_argument("--out",required=True)
    p.add_argument("--steps",type=int,required=True); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--start-steps",type=int,default=2000); p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--update-every",type=int,default=1,help="Gradient updates per eligible interaction; 0 disables updates")
    p.add_argument("--replay-capacity",type=int,default=200000)
    p.add_argument("--device",default="cpu"); p.add_argument("--options",default="")
    p.add_argument("--checkpoint-every",type=int,default=0)
    p.add_argument("--eval-every",type=int,default=0)
    p.add_argument("--eval-seeds",type=int,nargs="+")
    p.add_argument("--progress-every",type=int,default=0)
    p.add_argument("--log-every",type=int,default=1)
    p.add_argument("--purpose",default="training")
    p.add_argument("--learning-reward-scale",type=float,default=1.,help="Scale rewards for SAC updates only; saved replay and evaluation retain environment rewards")
    p.add_argument("--initial-alpha",type=float,default=1.)
    p.add_argument("--stop-after-successful-evals",type=int,default=0)
    p.add_argument("--success-rate-threshold",type=float,default=1.)
    return run(p.parse_args(argv))

if __name__ == "__main__":
    main()
