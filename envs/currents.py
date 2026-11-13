from pathlib import Path
import hashlib
import json
import numpy as np

def file_digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

class CurrentField:
    def __init__(self, config):
        self.cfg = config.current
        self.active = config.domain == "target" and config.shift == "ocs"
        self.asset = None
        if self.active and self.cfg.kind == "grid":
            path = Path(self.cfg.grid_path or "__missing_current_grid__")
            if not path.is_file():
                raise FileNotFoundError(f"SI current grid missing: {path}")
            from scipy.interpolate import RegularGridInterpolator
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata_json"].item()))
                if meta.get("frame") != "NED" or meta.get("velocity_unit") != "m/s" or meta.get("position_unit") != "m" or meta.get("time_unit") != "s" or not meta.get("provenance"):
                    raise ValueError("Current grid requires NED/m/s/m/s-time metadata and provenance")
                axes = [np.array(data[k], dtype=float) for k in ("time_s", "north_m", "east_m", "down_m")]
                vel = np.array(data["velocity_ned_m_s"], dtype=float)
            if vel.shape != tuple(len(a) for a in axes)+(3,) or not np.all(np.isfinite(vel)):
                raise ValueError("Invalid current-grid shape/values")
            if any(a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)) or np.any(np.diff(a) <= 0) for a in axes):
                raise ValueError("Current grid axes must have >=2 strictly increasing finite values")
            if ("mapping" in meta
                    and meta["mapping"].get("workspace_m") != list(config.task.workspace_m)):
                raise ValueError("Mapped HYCOM workspace differs from task.workspace_m; rebuild the asset explicitly")
            self.interpolator = RegularGridInterpolator(axes, vel, bounds_error=True)
            self.axes = axes
            self.sampling = meta.get("sampling", {})
            if (not isinstance(self.sampling, dict)
                    or set(self.sampling)-{"spatial_boundary", "time_boundary"}
                    or any(v not in {"error", "clamp"} for v in self.sampling.values())):
                raise ValueError("Grid sampling only accepts spatial_boundary/time_boundary = error or clamp")
            self.asset = {"path": str(path.resolve()), "sha256": file_digest(path), "metadata": meta}

    def _sample_grid(self, position, t):
        query = np.r_[t, position].astype(float)
        clamped = np.zeros(4, dtype=bool)
        for i, axis in enumerate(self.axes):
            key = "time_boundary" if i == 0 else "spatial_boundary"
            if self.sampling.get(key, "error") == "clamp":
                clamped[i] = query[i] < axis[0] or query[i] > axis[-1]
                query[i] = np.clip(query[i], axis[0], axis[-1])


        points = np.repeat(query[None, :], 9, axis=0)
        widths = np.empty(4)
        for i, axis in enumerate(self.axes):
            lo = int(np.clip(np.searchsorted(axis, query[i], side="right")-1, 0, len(axis)-2))
            points[1+2*i, i], points[2+2*i, i] = axis[lo:lo+2]
            widths[i] = axis[lo+1]-axis[lo]
        values = self.interpolator(points)
        partials = (values[2::2]-values[1::2])/widths[:, None]
        partials[clamped] = 0.
        return values[0], partials[1:].T, partials[0]

    def sample(self, position, t):
        c = self.cfg
        if not self.active or c.kind == "zero":
            return np.zeros(3), np.zeros((3, 3)), np.zeros(3)
        base = np.asarray(c.vector_ned_m_s, dtype=float).copy()
        jac = np.zeros((3, 3)); deriv = np.zeros(3)
        if c.kind == "spatial_shear":
            base[1] += c.shear_rate_s_inv*(position[2]-c.shear_reference_depth_m)
            jac[1, 2] = c.shear_rate_s_inv
        elif c.kind == "temporal_shear":
            freq = 2.*np.pi/c.temporal_period_s
            amplitude = np.asarray(c.temporal_amplitude_m_s)
            base += amplitude*np.sin(freq*t)
            deriv = freq*amplitude*np.cos(freq*t)
            base[1] += c.shear_rate_s_inv*(position[2]-c.shear_reference_depth_m)
            jac[1, 2] = c.shear_rate_s_inv
        elif c.kind == "grid":
            return self._sample_grid(position, t)
        return base, jac, deriv
