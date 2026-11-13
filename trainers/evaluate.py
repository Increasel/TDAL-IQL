import argparse
import numpy as np
from agents.checkpoint import load_checkpoint
from .common import build_env,project_path,write_json

def evaluate(agent, env, seeds):
    episodes = []
    for seed in seeds:
        obs,info = env.reset(seed=int(seed))
        reward_sum = 0.
        while True:
            obs,r,terminated,truncated,info = env.step(agent.act(obs,deterministic=True))
            reward_sum += r
            if terminated or truncated:
                break
        episodes.append({"seed":int(seed),"return":float(reward_sum),"steps":info["episode_step"],
             "time_s":info["time_s"],"path_length_m":info["path_length_m"],"success":info["success"],
             "mission_complete":info["mission_complete"],"reached_targets":info["reached_targets"],
             "collision_count":info["collision_count"],"had_collision":info["had_collision"],
             "terminated":info["terminated"],"truncated":info["truncated"],"termination_reason":info["termination_reason"]})
    successful = [e for e in episodes if e["success"]]
    return {"episodes":episodes,"episode_seeds":[int(s) for s in seeds],
            "success_rate":float(np.mean([e["success"] for e in episodes])),
            "mean_return":float(np.mean([e["return"] for e in episodes])),
            "mean_steps":float(np.mean([e["steps"] for e in episodes])),
            "mean_reached_targets":float(np.mean([e["reached_targets"] for e in episodes])),
            "successful_episode_count":len(successful),
            "successful_steps":float(np.mean([e["steps"] for e in successful])) if successful else None,
            "mean_path_length_m":float(np.mean([e["path_length_m"] for e in episodes])),
            "successful_path_length_m":float(np.mean([e["path_length_m"] for e in successful])) if successful else None,
            "endings":{r:sum(e["termination_reason"]==r for e in episodes) for r in sorted({e["termination_reason"] for e in episodes})},
            "collision_episode_count":sum(e["had_collision"] for e in episodes),
            "definition":"success = all targets reached with no collision anywhere in the episode and no timeout"}

def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a policy checkpoint and report episode metrics")
    for key in ("config","checkpoint","out"):
        p.add_argument(f"--{key}",required=True)
    p.add_argument("--seeds",nargs="+",type=int,required=True)
    p.add_argument("--allow-dynamics-transfer",action="store_true")
    p.add_argument("--device",default="cpu")
    p.add_argument("--purpose",default="evaluation")
    args = p.parse_args(argv)
    env = build_env(args.config)
    agent = load_checkpoint(project_path(args.checkpoint),env,device=args.device,allow_dynamics_transfer=args.allow_dynamics_transfer)
    result = evaluate(agent,env,args.seeds)
    write_json(args.out,{"purpose":args.purpose,"checkpoint":str(project_path(args.checkpoint)),"contract":env.contract(),**result})
    env.close()
    print(f"Evaluated {len(args.seeds)} episodes; {project_path(args.out)}")

if __name__ == "__main__":
    main()
