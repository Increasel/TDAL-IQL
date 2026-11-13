from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
from gymnasium.spaces import Box
from .config import SCHEMA_VERSION

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()

@dataclass
class Field:
    name: str
    start: int
    stop: int
    components: list
    frame: str
    physical_unit: str
    divisor: list
    lower: list
    upper: list
    group: str

class Schema:
    def __init__(self, config):
        self.config = config
        self.fields = []
        p, t, o = config.physics, config.task, config.observation
        def add(name, components, frame, unit, divisor, group, low=-np.inf, high=np.inf):
            n = len(components)
            start = self.fields[-1].stop if self.fields else 0
            def arr(v):
                return np.broadcast_to(np.asarray(v, dtype=float), (n,)).tolist()
            self.fields.append(Field(name, start, start+n, components, frame, unit,
                                     arr(divisor), arr(low), arr(high), group))
        xyz = ["x", "y", "z"]
        add("position", xyz, "NED (north,east,down)", "m", o.position_scale_m, "AP")
        add("quaternion", ["w", "x", "y", "z"], "body-to-NED unit quaternion", "1", 1., "AP", -1., 1.)
        add("linear_velocity", ["u", "v", "w"], "body, ground-relative", "m/s", o.linear_velocity_scale_m_s, "SV")
        add("angular_velocity", ["p", "q", "r"], "body", "rad/s", o.angular_velocity_scale_rad_s, "SV")
        add("target_relative", xyz, f"{o.target_relative_frame}, environment-selected current target", "m", o.target_scale_m, "RI")
        add("obstacle_risk", xyz, "NED, minimum-surface-clearance obstacle", "1", 1., "OR")
        if o.include_bottom_clearance:
            add("bottom_clearance", ["clearance"], "NED depth minus hull depth", "m",
                o.position_scale_m, "obstacle_geometry")
        self.obs_dim = self.fields[-1].stop
        self.action_components = ["X", "Y", "Z", "K", "M", "N"]
        self.action_dim = len(self.action_components)
        self.scales = np.concatenate([f.divisor for f in self.fields]).astype(np.float64)
        self.observation_space = Box(np.concatenate([f.lower for f in self.fields]).astype(np.float32),
                                     np.concatenate([f.upper for f in self.fields]).astype(np.float32), dtype=np.float32)
        self.action_space = Box(-1., 1., shape=(self.action_dim,), dtype=np.float32)

    def sl(self, name):
        for f in self.fields:
            if f.name == name:
                return slice(f.start, f.stop)
        raise KeyError(name)

    def indices(self, group):
        return np.array([i for f in self.fields if f.group == group for i in range(f.start, f.stop)], dtype=int)

    def decode(self, obs):
        obs = np.asarray(obs)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Observation dimension {obs.shape} incompatible with schema ({self.obs_dim},)")
        return obs * self.scales

    def encode(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (self.obs_dim,) or not np.all(np.isfinite(raw)):
            raise ValueError("Invalid physical observation")
        return (raw / self.scales).astype(np.float32)

    def validate_observation(self, obs):
        if np.asarray(obs).shape[-1:] != (self.obs_dim,) or not np.all(np.isfinite(obs)):
            raise ValueError(f"Expected finite observation dimension {self.obs_dim}; got {np.asarray(obs).shape}")

    def validate_action(self, action):
        a = np.asarray(action)
        if a.shape != (self.action_dim,) or not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1.):
            raise ValueError(f"Action must be finite ({self.action_dim},) in [-1,1]; got {a.shape}")
        return a.astype(np.float64)

    def to_dict(self):
        def finite_bounds(f):
            d = asdict(f)
            for key in ("lower", "upper"):
                d[key] = [v if np.isfinite(v) else ("-inf" if v < 0 else "+inf") for v in d[key]]
            return d
        return {"version": SCHEMA_VERSION, "dtype": "float32", "observation_dim": self.obs_dim,
                "observation": [finite_bounds(f) for f in self.fields],
                "normalization": "physical_value / divisor, no clipping; no target/obstacle slots or masks",
                "target_information": {"kind":"current target relative position only",
                                       "selection_owner":"environment", "zero_when":"mission complete"},
                "obstacle_information": {"kind":"three-dimensional nearest-surface risk vector",
                    "sensing_range_m":self.config.task.obstacle_detection_range_m,
                    "selection":"minimum center distance minus obstacle radius minus vehicle radius",
                    "formula":"risk=(1-clearance/R)*unit_direction when clearance<=R, else zero",
                    "direction_epsilon_m":1e-8,"frame":"NED","sonar_measurement_simulation":False},
                "action": {"dim": self.action_dim, "components": self.action_components, "frame": "body",
                           "range": [-1., 1.], "tau_max": list(self.config.physics.tau_max),
                           "tau_units": ["N"] * 3 + ["N m"] * 3,
                           "mapping": "tau_eff = G_deg @ (tau_max * a)"}}

    @property
    def hash(self):
        return fingerprint(self.to_dict())
