from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from envs.currents import file_digest

MAPPING_VERSION = "hycom-ned-mapping-v1"


def _clean(variable, name):
    values = variable[:]
    if np.ma.isMaskedArray(values) and np.ma.count_masked(values):
        raise ValueError(f"HYCOM {name} has missing cells; explicit preprocessing is required, no silent fill")
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"HYCOM {name} contains nonfinite values")
    return values


def convert(input_path, output_path, mapping):

    try:
        from netCDF4 import Dataset, num2date
    except ImportError as exc:
        raise ImportError("HYCOM NetCDF conversion requires: pip install netCDF4") from exc
    allowed = {"format_version", "workspace_m", "velocity_scale", "time_span_s", "sampling"}
    if not isinstance(mapping, dict) or set(mapping) != allowed or mapping.get("format_version") != MAPPING_VERSION:
        raise ValueError(f"Mapping must explicitly specify all fields of {MAPPING_VERSION}: {sorted(allowed)}")
    workspace = np.asarray(mapping["workspace_m"], dtype=float)
    scale, time_span = float(mapping["velocity_scale"]), mapping["time_span_s"]
    sampling = mapping["sampling"]
    if (workspace.shape != (3,) or not np.all(np.isfinite(workspace)) or np.any(workspace <= 0)
            or not np.isfinite(scale) or scale == 0
            or (time_span is not None and (not np.isfinite(time_span) or time_span <= 0))):
        raise ValueError("Mapping needs positive workspace/time span and a finite nonzero velocity scale")
    if (not isinstance(sampling, dict) or set(sampling) != {"spatial_boundary", "time_boundary"}
            or any(v not in {"error", "clamp"} for v in sampling.values())):
        raise ValueError("Mapping must explicitly choose error/clamp spatial and time boundaries")
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite current asset: {output_path}")
    with Dataset(input_path, mode="r") as nc:
        required = {"time", "depth", "lat", "lon", "water_u", "water_v"}
        if not required.issubset(nc.variables):
            raise ValueError(f"HYCOM NetCDF is missing {sorted(required-set(nc.variables))}")
        encoding = str(getattr(nc, "maddpg_current_encoding", ""))
        if encoding and encoding != "raw_hycom_horizontal_m_s_v1":
            raise ValueError(f"Unsupported HYCOM velocity encoding {encoding!r}; expected raw m/s data")
        axes = {name: _clean(nc[name], name) for name in ("time", "depth", "lat", "lon")}
        if any(a.ndim != 1 or len(a) < 2 or np.any(np.diff(a) <= 0) for a in axes.values()):
            raise ValueError("HYCOM axes must each have >=2 strictly increasing samples")
        for name, units in (("depth", {"m", "meters", "metres"}),
                            ("lat", {"degrees_north"}), ("lon", {"degrees_east"})):
            if getattr(nc[name], "units", "") not in units:
                raise ValueError(f"Unsupported HYCOM {name} coordinate units")
        if axes["depth"][0] != 0 or str(getattr(nc["depth"], "positive", "down")) != "down":
            raise ValueError("HYCOM depth must start at sea surface (0 m), positive down")
        velocity = []
        for name in ("water_u", "water_v"):
            var = nc[name]
            if var.dimensions != ("time", "depth", "lat", "lon"):
                raise ValueError(f"HYCOM {name} must have (time,depth,lat,lon) dimensions")
            if getattr(var, "units", "") not in {"m s-1", "m/s", "m s^-1"}:
                raise ValueError(f"HYCOM {name} must explicitly use m/s")
            velocity.append(_clean(var, name))
        if "water_w" in nc.variables:
            raise ValueError("This horizontal GOFS converter assumes no water_w; use an explicit 3-D converter for that product")
        time_units = getattr(nc["time"], "units", "")
        calendar = getattr(nc["time"], "calendar", "standard")
        dates = num2date(axes["time"], units=time_units, calendar=calendar)
        elapsed = np.asarray([(d-dates[0]).total_seconds() for d in dates], dtype=float)
        source = {"filename": input_path.name, "sha256": file_digest(input_path),
                  "dataset": str(getattr(nc, "source_dataset", "HYCOM horizontal GOFS subset")),
                  "url": str(getattr(nc, "source_url", "")),
                  "dataset_url": str(getattr(nc, "source_dataset_url", "")),
                  "encoding": encoding, "time_units": time_units, "calendar": calendar,
                  "time_origin": str(dates[0]), "time_end": str(dates[-1]),
                  "shape_T_D_lat_lon": list(velocity[0].shape),
                  "latitude_bounds_deg": axes["lat"][[0, -1]].tolist(),
                  "longitude_bounds_deg": axes["lon"][[0, -1]].tolist(),
                  "depth_bounds_m": axes["depth"][[0, -1]].tolist(),
                  "upstream_missing_cells_filled": int(getattr(nc, "horizontal_missing_cells_filled", 0)),
                  "upstream_note": str(getattr(nc, "note", ""))}
    def stretch(axis, length):
        return (axis-axis[0])/(axis[-1]-axis[0])*length
    ned = np.zeros((len(elapsed), len(axes["lat"]), len(axes["lon"]), len(axes["depth"]), 3), dtype=np.float32)

    ned[..., 0] = (velocity[1]*scale).transpose(0, 2, 3, 1)
    ned[..., 1] = (velocity[0]*scale).transpose(0, 2, 3, 1)
    if not np.all(np.isfinite(ned)):
        raise ValueError("HYCOM velocity cannot be represented as finite float32")
    speed = np.linalg.norm(ned.astype(float), axis=-1)
    native_span = float(elapsed[-1])
    metadata = {"format_version": "auv-current-grid-v1", "converter_version": "hycom-to-ned-v1",
                "frame": "NED", "velocity_unit": "m/s", "position_unit": "m", "time_unit": "s",
                "provenance": source, "mapping": mapping,
                "coordinate_mapping": {"north": "full latitude range -> [0, workspace_north_m]",
                                       "east": "full longitude range -> [0, workspace_east_m]",
                                       "down": "source surface-to-max-depth -> [0, workspace_down_m]",
                                       "type": "affine scene mapping",
                                       "velocity_ned": "velocity_scale * [water_v, water_u, 0]",
                                       "vertical_velocity": "zero; water_w absent",
                                       "native_time_span_s": native_span,
                                       "mapped_time_span_s": native_span if time_span is None else float(time_span),
                                       "time_origin": source["time_origin"]},
                "sampling": sampling,
                "speed_m_s": {"median": float(np.median(speed)), "p95": float(np.quantile(speed, .95)),
                              "max": float(speed.max())}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as f:
        np.savez_compressed(f, time_s=elapsed if time_span is None else elapsed/native_span*time_span,
                            north_m=stretch(axes["lat"], workspace[0]),
                            east_m=stretch(axes["lon"], workspace[1]),
                            down_m=stretch(axes["depth"], workspace[2]), velocity_ned_m_s=ned,
                            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw GOFS NetCDF (read-only)")
    parser.add_argument("--mapping", default=str(ROOT/"configs/hycom_mapping.json"))
    parser.add_argument("--out", required=True, help="New NPZ inside this project; existing files are protected")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"Converted assets must stay inside {ROOT}")
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    meta = convert(args.input, output, mapping)
    print(json.dumps({"output": str(output), "sha256": file_digest(output),
                      "source_sha256": meta["provenance"]["sha256"],
                      "speed_m_s": meta["speed_m_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
