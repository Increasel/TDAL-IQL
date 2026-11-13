from .config import EnvConfig, make_config
from .auv import AUVEnv

def make_env(config=None, **kwargs):
    return AUVEnv(config if config is not None else make_config(**kwargs))
