import argparse
from agents.checkpoint import load_checkpoint
from trainers.common import build_env,set_seed,project_path
from .collection import collect
from envs.currents import file_digest

def main(argv=None):
    p = argparse.ArgumentParser(description="Collect offline transitions from a configured environment")
    p.add_argument("--config",required=True); p.add_argument("--out",required=True)
    p.add_argument("--steps",type=int,required=True); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--policy",default=None,help="Policy checkpoint path; defaults to random actions")
    p.add_argument("--noise-std",type=float,default=0.)
    p.add_argument("--allow-dynamics-transfer",action="store_true")
    p.add_argument("--device",default="cpu")
    p.add_argument("--purpose",default="collection")
    args = p.parse_args(argv)
    env = build_env(args.config); set_seed(args.seed)
    policy = None if args.policy is None else load_checkpoint(project_path(args.policy),env,device=args.device,
                                                   allow_dynamics_transfer=args.allow_dynamics_transfer)
    data = collect(env,args.steps,args.seed,policy,args.noise_std,
                   policy_description={"kind":"random" if policy is None else "checkpoint","path":args.policy,
                     "sha256":None if args.policy is None else file_digest(project_path(args.policy)),
                     "algorithm":None if policy is None else policy.algorithm},purpose=args.purpose)
    data.save(project_path(args.out)); env.close()
    print(f"Collected {len(data)} transitions: {project_path(args.out)}")

if __name__ == "__main__":
    main()
