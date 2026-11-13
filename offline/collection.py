import numpy as np
import time
from .dataset import ReplayBuffer

def collect(env, steps, seed=0, policy=None, noise_std=0., policy_description=None,
            purpose="collection", progress_every=0, on_progress=None):
    if steps<1 or noise_std<0 or progress_every<0:
        raise ValueError("steps must be positive, noise_std nonnegative")
    buffer = ReplayBuffer(steps,env,seed=seed,collection={"purpose":purpose,
                    "steps_requested":int(steps),"policy":policy_description or ("random" if policy is None else "supplied_policy"),
                    "policy_mode":"random_actions" if policy is None else "deterministic_policy_plus_clipped_gaussian_noise",
                    "noise_std":float(noise_std),"competence_validated":False})
    rng = np.random.default_rng(seed)
    episode = 0
    completed, successful, started = 0, 0, time.perf_counter()
    obs,_ = env.reset(seed=seed)
    if policy is not None and hasattr(policy, "reset"):
        policy.reset(seed=seed)
    for step in range(steps):
        action = env.action_space.sample() if policy is None else policy.act(obs,deterministic=True)
        if noise_std:
            action = np.clip(action+rng.normal(0.,noise_std,size=env.schema.action_dim),-1.,1.).astype(np.float32)
        next_obs,reward,terminated,truncated,info = env.step(action)
        buffer.add(obs,action,reward,next_obs,terminated,truncated,info,episode,seed+episode,
                   collector_cut=step==steps-1 and not (terminated or truncated))
        obs = next_obs
        if terminated or truncated:
            completed += 1
            successful += int(info["success"])
        if (terminated or truncated) and step<steps-1:
            episode += 1
            obs,_ = env.reset(seed=seed+episode)
            if policy is not None and hasattr(policy, "reset"):
                policy.reset(seed=seed+episode)
        if on_progress is not None and progress_every and ((step+1)%progress_every==0 or step+1==steps):
            on_progress({"transitions":step+1,"requested":steps,"complete_episodes":completed,
                         "successful_episodes":successful,"elapsed_s":time.perf_counter()-started})
    return buffer.dataset()
