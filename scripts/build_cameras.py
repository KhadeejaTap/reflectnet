"""
build_cameras.py

For every surviving instance (matte, non-metal, no glass, not transparent),
write a single cameras.json containing all 50 frames' camera info, already
converted from Blender's Z-up convention to Mitsuba's Y-up convention.

Camera data is read directly from each frame's camera.json inside the tar
shard (via 'shard_path' + 'camera_member' from the metadata parquet), since
parquet's own nested 'transform_matrix' column comes back as a ragged/object
numpy array that doesn't convert cleanly.

Output layout:
    mitsuba_scenes/instances/{instance_id}/cameras.json

Run this once as a batch step before rendering.
"""

import json
import tarfile
import numpy as np
from pathlib import Path
from functools import lru_cache

from explore_data import load_metadata, filter_matte_nonmetal

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "mitsuba_scenes" / "instances"


def blender_zup_to_mitsuba(transform_matrix):
    """
    Convert a Blender-native Z-up camera-to-world matrix to Mitsuba's
    Y-up convention while preserving a right-handed coordinate system.

    New axes are:
      new_X = old_X
      new_Y = old_Z
      new_Z = -old_Y
    """
    T = np.array(transform_matrix, dtype=np.float64)
    swap = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    return swap @ T


@lru_cache(maxsize=8)
def _open_tar(shard_path: str):
    """Cache open tarfile handles since many frames share the same shard."""
    full_path = REPO_ROOT / shard_path
    return tarfile.open(full_path)


def read_camera_json(shard_path: str, camera_member: str):
    tar = _open_tar(shard_path)
    f = tar.extractfile(camera_member)
    return json.load(f)


def build_camera_json(df, instance_id):
    rows = df[df["instance_id"] == instance_id].sort_values("frame_id")
    cameras = []
    for _, row in rows.iterrows():
        raw = read_camera_json(row["shard_path"], row["camera_member"])
        T_mitsuba = blender_zup_to_mitsuba(raw["transform_matrix"])
        cameras.append({
            "frame_id": int(row["frame_id"]),
            "transform_matrix": [[float(x) for x in r] for r in T_mitsuba],
            "intrinsics": raw["intrinsics"],
        })
    return cameras


def main():
    df = load_metadata()
    kept, _dropped = filter_matte_nonmetal(df)

    instance_ids = kept["instance_id"].unique()
    print(f"Building cameras.json for {len(instance_ids)} surviving instances...")

    for i, instance_id in enumerate(instance_ids):
        cameras = build_camera_json(kept, instance_id)

        out_dir = OUTPUT_ROOT / instance_id
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "cameras.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cameras, f, indent=2)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(instance_ids)} done")

    print(f"Done. Wrote {len(instance_ids)} cameras.json files under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
