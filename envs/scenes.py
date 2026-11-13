import numpy as np
from .schema import fingerprint

SCENE_VERSION = "mission-layouts-v3"
START = [.34, -.72, -.44]
START_JITTER = .035
PROFILES = {
    (3, 1): {
        "targets": [[-.80, .80, -.30], [.80, -.80, .30], [.80, .80, .70]],
        "obstacles": [[.08, .08, .0, .055]],
    },
    (5, 3): {
        "targets": [[-.57, .52, -.32], [.48, -.53, -.14], [.48, .35, .12],
                    [-.31, -.44, .33], [.30, .35, .54]],
        "obstacles": [[-.01, -.02, -.37, .055], [.03, .08, -.23, .052], [.37, -.09, -.01, .050]],
    },
    (7, 5): {
        "targets": [[-.52, -.54, -.36], [-.64, .46, -.24], [.36, .70, -.08],
                    [.70, -.04, .08], [.0, -.62, .24], [-.64, .08, .38], [.08, .70, .56]],
        "obstacles": [[-.69, -.07, -.30, .052], [-.13, .70, -.15, .052], [.65, .34, .01, .050],
                      [.42, -.41, .16, .045], [-.39, -.36, .31, .045]],
    },
}


def normalized_to_ned(points, workspace_m):
    points = np.asarray(points, dtype=float)
    extent = np.asarray(workspace_m, dtype=float)
    return (points[..., [1, 0, 2]]*np.array([1., 1., -1.])+1.)*.5*extent


def _profile(task):
    key = (task.num_targets, task.num_obstacles)
    if key not in PROFILES:
        raise ValueError(f"No paper_scale layout for {key}; choose task.scene='random' for other mission sizes")
    if not np.all(np.asarray(task.workspace_m) == task.workspace_m[0]):
        raise ValueError("Normalized spherical geometry requires a cubic workspace; choose task.scene='random' otherwise")
    return PROFILES[key]


def sample_mission_scene(task, rng):
    profile = _profile(task)
    scale = task.workspace_m[0]/2.
    start = normalized_to_ned(np.asarray(START)+rng.uniform(-START_JITTER, START_JITTER, 3), task.workspace_m)
    targets = normalized_to_ned(profile["targets"], task.workspace_m)
    if task.target_distance_scale != 1.:
        anchor = normalized_to_ned(START, task.workspace_m)
        targets = anchor + task.target_distance_scale*(targets-anchor)
    raw_obstacles = np.asarray(profile["obstacles"])
    obstacles = np.column_stack([normalized_to_ned(raw_obstacles[:, :3], task.workspace_m), raw_obstacles[:, 3]*scale])
    return start, targets, obstacles


def scene_contract(task):
    if task.scene == "random":
        result = {"version": "random-free-space-v1", "selection": "explicit task.scene=random"}
        if task.goal_distance_range_m is not None:
            result["navigation_curriculum"] = {"goal_distance_range_m":list(task.goal_distance_range_m),
                "sampling":"uniform distance and isotropic direction from start/previous target, rejecting workspace/terrain/goal overlap",
                "final_task":False}
        return result
    profile = _profile(task)
    definition = {"start_enu": START, "start_jitter": START_JITTER, **profile}
    result = {"version": SCENE_VERSION, "profile": f"{task.num_targets}T{task.num_obstacles}O",
            "definition": definition, "definition_sha256": fingerprint(definition),
            "mapping": "N=(y+1)*L/2, E=(x+1)*L/2, D=(1-z)*L/2; radii*=L/2",
            "workspace_edge_m": task.workspace_m[0]}
    if task.target_distance_scale != 1.:
        result["curriculum"] = {"target_distance_scale": task.target_distance_scale,
            "anchor_ned_m": normalized_to_ned(START, task.workspace_m).tolist(),
            "mapping": "target = anchor + scale*(full_target-anchor); obstacles/radii/start distribution unchanged",
            "final_task": False}
    return result
