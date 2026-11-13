from pathlib import Path
import json
import random
import numpy as np
import torch
from envs import AUVEnv, EnvConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(path):
    p = Path(path).resolve()
    if not p.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Run artifacts must stay inside the project directory {PROJECT_ROOT}: {p}")
    return p

def create_run_dir(path):
    p = project_path(path)
    p.mkdir(parents=True,exist_ok=False)
    return p

def write_json(path, value):
    path = project_path(path)
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def set_seed(seed, threads=1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)

def read_options(value):
    if not value:
        return {}
    out = json.loads(value) if value.lstrip().startswith("{") else json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(out,dict):
        raise ValueError("Algorithm options must be a JSON object")
    return out

def build_env(path):
    return AUVEnv(EnvConfig.load(path))
