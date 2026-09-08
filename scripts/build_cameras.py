"""
build_cameras.py

For every surviving instance (matte, non-metal, no glass, not transparent),
write a single cameras.json containing all 50 frames' camera info, already
converted from NeRF/instant-ngp convention to Mitsuba convention.

Output layout:
    mitsuba_scenes/instances/{instance_id}/cameras.json

Run this once as a batch step before rendering.
"""

import json
import numpy as np
from pathlib import Path

from explore_data import load_metadata, filter_matte_nonmetal

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "mitsuba_scenes" / "instances"


def nerf_to_mitsuba(transform_matrix):
    """
    Convert a NeRF/instant-ngp camera-to-world matrix (+X right, +Y up,
    -Z forward) to Mitsuba convention (+X left, +Y up, +Z forward).
    Flips X and Z axes.
    """
    T = np.array(transform_matrix)
    flip = np.diag([-1, 1, -1, 1])
    return T @ flip


def _to_native(obj):
    """Recursively convert numpy/pandas objects to plain Python types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def build_camera_json(df, instance_id):
    rows = df[df["instance_id"] == instance_id].sort_values("frame_id")
    cameras = []
    for _, row in rows.iterrows():
        T_mitsuba = nerf_to_mitsuba(row["transform_matrix"])
        cameras.append({
            "frame_id": int(row["frame_id"]),
            "transform_matrix": T_mitsuba.tolist(),
            "intrinsics": _to_native(row["intrinsics"]),
        })
    return cameras


def main():
    df = load_metadata()
    kept, _dropped = filter_matte_nonmetal(df)

    instance_ids = kept["instance_id"].unique()
    print(f"Building cameras.json for {len(instance_ids)} surviving instances...")

    for instance_id in instance_ids:
        cameras = build_camera_json(kept, instance_id)

        out_dir = OUTPUT_ROOT / instance_id
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "cameras.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cameras, f, indent=2)

    print(f"Done. Wrote {len(instance_ids)} cameras.json files under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
