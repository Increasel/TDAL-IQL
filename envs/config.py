from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import copy
import json
import numpy as np

ENV_VERSION = "auv-sixdof-v5"
SCHEMA_VERSION = "auv-observation-v2"
ADS = {
    "mild": (1., .75, 1., .925, .85, .90),
    "moderate": (1., .50, 1., .85, .70, .80),
    "severe": (1., .25, 1., .775, .55, .70),
}
OCS = {
    "mild": (.42, -.175, -.63),
    "moderate": (1.2, -.5, -1.8),
    "severe": (1.8, -.75, -2.7),
}

@dataclass
class PhysicsConfig:

    mass_kg: float = 100.
    inertia_cg_kg_m2: list = field(default_factory=lambda: [8., 12., 15.])
    cg_m: list = field(default_factory=lambda: [0., 0., 0.])
    cb_m: list = field(default_factory=lambda: [0., 0., -.05])
    water_density_kg_m3: float = 1025.
    displaced_volume_m3: float = 100. / 1025.
    gravity_m_s2: float = 9.81

    added_mass: list = field(default_factory=lambda: [20., 22., 24., 2., 3., 4.])
    damping_linear: list = field(default_factory=lambda: [20., 30., 40., 5., 6., 7.])

    damping_quadratic: list = field(default_factory=lambda: [4., 2., 2., 2., 3., 4.])
    tau_max: list = field(default_factory=lambda: [200., 200., 250., 10., 10., 10.])

    dt_s: float = 1.
    integration_substeps: int = 10
    numerical_velocity_limit: float = 100.

@dataclass
class TaskConfig:

    workspace_m: list = field(default_factory=lambda: [1000., 1000., 1000.])
    scene: str = "paper_scale"
    num_targets: int = 3
    num_obstacles: int = 1
    vehicle_radius_m: float = 5.5
    target_radius_m: float = 25.
    obstacle_radius_m: list = field(default_factory=lambda: [22.5, 27.5])
    obstacle_detection_range_m: float = 50.
    max_episode_steps: int = 1000
    collision_terminates: bool = True
    boundary_terminates: bool = True
    visit_rule: str = "ordered"
    target_order: str = "provided"
    bathymetry_path: str | None = None
    target_distance_scale: float = 1.
    goal_distance_range_m: list | None = None

@dataclass
class CurrentConfig:
    kind: str = "uniform"
    vector_ned_m_s: list = field(default_factory=lambda: list(OCS["moderate"]))
    shear_rate_s_inv: float = .002
    shear_reference_depth_m: float = 500.
    temporal_amplitude_m_s: list = field(default_factory=lambda: [.3, .1, .2])
    temporal_period_s: float = 120.
    grid_path: str | None = None

@dataclass
class RewardConfig:


    alpha_dis: float = .02
    alpha_arr: float = 20.
    alpha_tim: float = .01
    alpha_obs: float = 20.
    alpha_reg: float = .05
    unsafe_clearance_m: float = 10.
    boundary_exit_cost: float = 1.
    overall_scale: float = 1.15

@dataclass
class ObservationConfig:

    position_scale_m: float = 1000.
    target_scale_m: float = 100.
    linear_velocity_scale_m_s: float = 5.
    angular_velocity_scale_rad_s: float = 1.
    include_bottom_clearance: bool = False
    target_relative_frame: str = "body"

@dataclass
class EnvConfig:
    version: str = ENV_VERSION
    domain: str = "source"
    shift: str = "ads"
    intensity: str = "moderate"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    current: CurrentConfig = field(default_factory=CurrentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def to_dict(self):
        data = asdict(self)

        if self.task.target_distance_scale == 1.:
            data["task"].pop("target_distance_scale")
        if self.task.goal_distance_range_m is None:
            data["task"].pop("goal_distance_range_m")
        if self.reward.boundary_exit_cost == 0.:
            data["reward"].pop("boundary_exit_cost")
        if self.reward.overall_scale == 1.:
            data["reward"].pop("overall_scale")
        if self.observation.target_relative_frame == "NED":
            data["observation"].pop("target_relative_frame")
        return data

    @classmethod
    def from_dict(cls, data):
        data = copy.deepcopy(data)
        for key, typ in [("physics", PhysicsConfig), ("task", TaskConfig),
                         ("current", CurrentConfig), ("reward", RewardConfig),
                         ("observation", ObservationConfig)]:
            if key in data:
                data[key] = typ(**data[key])
        out = cls(**data)
        out.validate()
        return out

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        out = cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for obj, attr in [(out.current, "grid_path"), (out.task, "bathymetry_path")]:
            value = getattr(obj, attr)
            if value and not Path(value).is_absolute():
                setattr(obj, attr, str((path.parent / value).resolve()))
        return out

    def validate(self):
        if self.version != ENV_VERSION:
            raise ValueError(f"Unsupported environment version {self.version}; expected {ENV_VERSION}")
        if self.domain not in {"source", "target"} or self.shift not in {"ads", "ocs", "polarity"}:
            raise ValueError("domain must be source/target; shift must be ads/ocs/polarity")
        if self.intensity not in ADS:
            raise ValueError("intensity must be mild/moderate/severe")
        p, t, o = self.physics, self.task, self.observation
        for obj in (p, t, self.current, self.reward, o):
            for f in fields(obj):
                v = getattr(obj, f.name)
                if isinstance(v, (int, float, list)) and not np.all(np.isfinite(v)):
                    raise ValueError(f"Nonfinite configuration {f.name}")
        for name, length in [("inertia_cg_kg_m2", 3), ("cg_m", 3), ("cb_m", 3),
                             ("added_mass", 6), ("damping_linear", 6),
                             ("damping_quadratic", 6), ("tau_max", 6)]:
            if np.asarray(getattr(p, name)).shape != (length,):
                raise ValueError(f"{name} must have {length} entries")
        if (p.mass_kg <= 0 or min(p.inertia_cg_kg_m2) <= 0 or min(p.added_mass) < 0
                or min(p.tau_max) <= 0 or min(p.damping_linear) < 0 or min(p.damping_quadratic) < 0
                or p.dt_s <= 0 or p.integration_substeps < 1 or int(p.integration_substeps) != p.integration_substeps
                or p.water_density_kg_m3 <= 0 or p.displaced_volume_m3 <= 0 or p.gravity_m_s2 <= 0):
            raise ValueError("Invalid mass/inertia/damping/actuation/integration parameters")
        if (not isinstance(t.num_targets,int) or not isinstance(t.num_obstacles,int)
                or t.num_targets<1 or t.num_obstacles<0):
            raise ValueError("num_targets must be a positive integer; num_obstacles must be a nonnegative integer")
        if (len(t.workspace_m) != 3 or min(t.workspace_m) <= 0 or t.max_episode_steps < 1
                or t.target_radius_m <= 0 or t.vehicle_radius_m <= 0
                or t.obstacle_detection_range_m <= 0 or t.visit_rule not in {"any_order", "ordered"}
                or t.target_order not in {"initial_distance", "provided"}
                or t.scene not in {"paper_scale", "random"}
                or len(t.obstacle_radius_m) != 2 or not 0 < t.obstacle_radius_m[0] <= t.obstacle_radius_m[1]):
            raise ValueError("Invalid task parameters")
        if any(getattr(o, name) <= 0 for name in ("position_scale_m","target_scale_m",
                                                "linear_velocity_scale_m_s","angular_velocity_scale_rad_s")):
            raise ValueError("All observation scales must be positive")
        if o.target_relative_frame not in {"NED","body"}:
            raise ValueError("target_relative_frame must be NED or body")
        if not 0 < t.target_distance_scale <= 1 or (t.scene != "paper_scale" and t.target_distance_scale != 1):
            raise ValueError("target_distance_scale must be in (0,1], and a curriculum requires paper_scale")
        if t.goal_distance_range_m is not None:
            if (t.scene!="random" or np.asarray(t.goal_distance_range_m).shape!=(2,) or
                    not t.target_radius_m<t.goal_distance_range_m[0]<=t.goal_distance_range_m[1]):
                raise ValueError("goal_distance_range_m requires a random scene and two distances above target_radius_m")
        if self.reward.boundary_exit_cost < 0:
            raise ValueError("boundary_exit_cost must be nonnegative")
        if self.reward.overall_scale <= 0:
            raise ValueError("overall_scale must be positive")
        if not isinstance(o.include_bottom_clearance,bool):
            raise ValueError("include_bottom_clearance must be boolean")
        if o.include_bottom_clearance and not t.bathymetry_path:
            raise ValueError("include_bottom_clearance requires bathymetry_path")
        if self.current.kind not in {"zero", "uniform", "spatial_shear", "temporal_shear", "grid"}:
            raise ValueError("Unknown current field")
        if (len(self.current.vector_ned_m_s) != 3 or len(self.current.temporal_amplitude_m_s) != 3
                or self.current.temporal_period_s <= 0 or self.reward.unsafe_clearance_m <= 0):
            raise ValueError("Invalid current/reward configuration")
        return self

    @property
    def degradation(self):
        if self.domain == "source" or self.shift == "ocs":
            return np.eye(6)
        if self.shift == "polarity":
            return np.diag([1., -1., 1., 1., 1., 1.])
        return np.diag(ADS[self.intensity])

def make_config(domain="source", shift="ads", intensity="moderate", **kwargs):
    cfg = EnvConfig(domain=domain, shift=shift, intensity=intensity, **kwargs)
    if shift == "ocs" and "current" not in kwargs:
        cfg.current.vector_ned_m_s = list(OCS[intensity])
    return cfg.validate()
