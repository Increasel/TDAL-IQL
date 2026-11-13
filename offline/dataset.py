from pathlib import Path
import json
import numpy as np
from envs.config import ENV_VERSION, EnvConfig
from envs.schema import Schema, fingerprint
from .array_buffer import validate_arrays

FORMAT_VERSION = "auv-transitions-v2"
ARRAY_DTYPES = {"observations": np.float32, "actions": np.float32, "rewards": np.float32,
                "next_observations": np.float32, "terminated": np.bool_, "truncated": np.bool_,
                "episode_id": np.int64, "episode_seed": np.int64, "step": np.int64, "time_s": np.float64,
                "collision": np.bool_, "had_collision": np.bool_, "collision_count": np.int64,
                "success": np.bool_, "reached_targets": np.int64, "termination_reason": "U40",
                "collector_cut": np.bool_, "target_switched":np.bool_}

def validate_contract(meta, expected=None):
    if not isinstance(meta, dict) or meta.get("environment_version") != ENV_VERSION:
        raise ValueError(f"Unsupported environment version; expected {ENV_VERSION}")
    if fingerprint(meta.get("schema")) != meta.get("schema_sha256"):
        raise ValueError("Schema hash mismatch or missing schema metadata")
    if fingerprint(meta.get("config")) != meta.get("config_sha256"):
        raise ValueError("Environment config hash mismatch")
    cfg = EnvConfig.from_dict(meta["config"])
    schema = Schema(cfg)
    if schema.hash != meta["schema_sha256"]:
        raise ValueError("Schema does not match environment configuration")
    if expected is not None:
        expected = expected.contract() if hasattr(expected, "contract") else expected
        if expected.get("schema_sha256") != meta["schema_sha256"]:
            raise ValueError(f"Observation/action schema mismatch: file dimensions {schema.obs_dim}/{schema.action_dim}, "
                             f"requested {expected['schema']['observation_dim']}/{expected['schema']['action']['dim']}; no padding/cropping")
        if meta.get("assets",{}).get("bathymetry") != expected.get("assets",{}).get("bathymetry"):
            raise ValueError("Bathymetry asset identity differs from the requested environment")
        if meta.get("assets",{}).get("scene") != expected.get("assets",{}).get("scene"):
            raise ValueError("Scene layout identity differs from the requested environment")
        if meta["config_sha256"] == expected.get("config_sha256") and meta.get("assets") != expected.get("assets"):
            raise ValueError("External current/terrain asset identity changed despite identical config")
    return schema

class TransitionDataset:
    def __init__(self, arrays, metadata):
        self.arrays, self.metadata = arrays, metadata
        self.validate()

    def __len__(self):
        return len(self.arrays["observations"])

    def validate(self, expected=None):
        if self.metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported transition format; expected {FORMAT_VERSION}")
        schema = validate_contract(self.metadata, expected)
        if not isinstance(self.metadata.get("seed"), int) or "collection" not in self.metadata:
            raise ValueError("Dataset requires seed and collection metadata")
        a = self.arrays
        missing = set(ARRAY_DTYPES)-set(a)
        if missing:
            raise ValueError(f"Missing transition fields {sorted(missing)}")
        n = len(a["observations"])
        if not n:
            raise ValueError("Dataset is empty")
        for key, dtype in ARRAY_DTYPES.items():
            shape = (n, schema.obs_dim) if key in {"observations", "next_observations"} else (n, schema.action_dim) if key == "actions" else (n, 1)
            if a[key].shape != shape or a[key].dtype != np.dtype(dtype):
                raise ValueError(f"{key}: expected {shape} {np.dtype(dtype)}, got {a[key].shape} {a[key].dtype}; incompatible dataset")
        validate_arrays(a["observations"], a["actions"], a["rewards"], a["terminated"], a["next_observations"], schema.obs_dim, schema.action_dim)
        if np.any(a["terminated"] & a["truncated"]):
            raise ValueError("A transition cannot be both terminated and time-truncated in this format")
        if np.any(a["success"] & (~a["terminated"] | a["had_collision"])):
            raise ValueError("Success requires true termination and no collision anywhere in the episode")
        if np.any((a["termination_reason"] == "time_limit") != a["truncated"]):
            raise ValueError("time_limit reason and truncated flag disagree")
        allowed={"running","success","collision","out_of_bounds","mission_completed_with_collision","time_limit"}
        if not set(a["termination_reason"].reshape(-1)).issubset(allowed):
            raise ValueError("Unknown termination reason")
        if np.any((a["termination_reason"]=="running") != ~(a["terminated"]|a["truncated"])):
            raise ValueError("Running/terminal reasons disagree with ending flags")
        if np.any(a["collector_cut"] & (a["terminated"]|a["truncated"])):
            raise ValueError("collector_cut applies only to nonterminal, non-time-truncated transitions")
        if (not np.all(np.isfinite(a["time_s"])) or np.any(a["time_s"]<=0)
                or np.any(a["step"]<1) or np.any(a["episode_id"]<0) or np.any(a["collision_count"]<0)
                or np.any(a["reached_targets"]<0) or np.any(a["reached_targets"]>self.metadata["config"]["task"]["num_targets"])):
            raise ValueError("Invalid times, episode indices, collision or reached-target counts")
        schema.validate_observation(a["next_observations"])
        for key in ("observations", "next_observations"):
            x = a[key]
            low, high = schema.observation_space.low, schema.observation_space.high
            if np.any(x<low) or np.any(x>high):
                raise ValueError(f"{key} violates observation bounds")
            if np.any(np.abs(np.linalg.norm(x[:, schema.sl('quaternion')], axis=1)-1.)>1e-4):
                raise ValueError(f"{key} contains invalid unit quaternion")
        return self

    def core_arrays(self):
        a = self.arrays

        return a["observations"], a["actions"], a["rewards"], a["terminated"].astype(np.float32), a["next_observations"]

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            np.savez_compressed(f, **self.arrays, metadata_json=np.asarray(json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)))

    @classmethod
    def load(cls, path, expected=None):
        with np.load(path, allow_pickle=False) as f:
            if "metadata_json" not in f:
                raise ValueError("Dataset requires schema and version metadata")
            meta = json.loads(str(f["metadata_json"].item()))
            validate_contract(meta, expected)
            arrays = {k: np.array(f[k]) for k in f.files if k != "metadata_json"}
        out = cls(arrays, meta)
        return out.validate(expected)

class ReplayBuffer:
    def __init__(self, capacity, env, seed=0, collection=None):
        self.capacity = int(capacity)
        if self.capacity<1:
            raise ValueError("capacity must be positive")
        self.schema = env.schema
        self.metadata = {**env.contract(), "format_version": FORMAT_VERSION,
                         "seed": int(seed), "collection": collection or {"purpose": "interface_test"}}
        self.rng = np.random.default_rng(seed)
        self.rows = []
        self.pointer = 0

    def __len__(self):
        return len(self.rows)

    def add(self, obs, action, reward, next_obs, terminated, truncated, info, episode_id, episode_seed, collector_cut=False):
        self.schema.validate_observation(obs); self.schema.validate_observation(next_obs)
        self.schema.validate_action(action)
        if not np.isfinite(reward) or (terminated and truncated):
            raise ValueError("Invalid reward or ending flags")
        row = {"observations": np.asarray(obs).copy(), "actions": np.asarray(action).copy(), "rewards": reward,
               "next_observations": np.asarray(next_obs).copy(), "terminated": bool(terminated), "truncated": bool(truncated),
               "episode_id": episode_id, "episode_seed": episode_seed, "step": info["episode_step"], "time_s": info["time_s"],
               "collision": info["collision"], "had_collision": info["had_collision"], "collision_count": info["collision_count"],
               "success": info["success"], "reached_targets": info["reached_targets"], "termination_reason": info["termination_reason"],
               "collector_cut": collector_cut,"target_switched":info["target_switched"]}
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.pointer] = row
        self.pointer = (self.pointer+1) % self.capacity

    def sample(self, batch_size):
        if not self.rows:
            raise RuntimeError("Empty replay buffer")
        rows = [self.rows[i] for i in self.rng.integers(len(self.rows), size=int(batch_size))]
        get = lambda key: np.asarray([r[key] for r in rows], dtype=np.float32)
        return {"s": get("observations"), "a": get("actions"), "r": get("rewards").reshape(-1, 1),
                "s2": get("next_observations"), "d": get("terminated").reshape(-1, 1),
                "terminated": get("terminated").reshape(-1, 1), "truncated": get("truncated").reshape(-1, 1)}

    def dataset(self):
        if not self.rows:
            raise RuntimeError("Empty replay buffer")

        rows = self.rows if len(self.rows)<self.capacity else self.rows[self.pointer:]+self.rows[:self.pointer]
        arrays = {}
        for key, dtype in ARRAY_DTYPES.items():
            arrays[key] = np.asarray([r[key] for r in rows], dtype=dtype)
            if key not in {"observations", "actions", "next_observations"}:
                arrays[key] = arrays[key].reshape(-1, 1)
        return TransitionDataset(arrays, self.metadata)
