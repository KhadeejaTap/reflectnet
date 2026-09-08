#!/usr/bin/env python3
"""Visualize all depth outputs in a ToF sample directory."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEPTH_SPECS = {
    "_tof.npy": ("meters", "Planar ToF Depth (m)"),
    "_tof_mm.npy": ("millimeters", "ToF Depth (m)"),
    "_depth_proj_mm.npy": ("millimeters", "Reprojected RGB Depth (m)"),
}


def find_depth_files(input_dir: Path):
    """Find supported depth arrays while excluding raw signal and mask files."""
    files = []
    for path in input_dir.rglob("*.npy"):
        if "vis" in path.parts:
            continue
        for suffix, unit_info in DEPTH_SPECS.items():
            if path.name.endswith(suffix):
                files.append((path, unit_info))
                break
    return sorted(files, key=lambda item: str(item[0]))


def save_heatmap(array_path: Path, output_path: Path, units: str, label: str, vmin: float, vmax: float):
    depth = np.asarray(np.load(array_path), dtype=np.float32)
    if depth.ndim != 2:
        print(f"Skipping {array_path}: expected a 2D depth map, got {depth.shape}")
        return False

    if units == "millimeters":
        depth = depth / 1000.0

    masked_depth = np.ma.masked_invalid(depth)
    masked_depth = np.ma.masked_less_equal(masked_depth, 0)

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7), dpi=150)
    image = plt.imshow(masked_depth, cmap=cmap, vmin=vmin, vmax=vmax)
    colorbar = plt.colorbar(image)
    colorbar.set_label(label)
    plt.title(f"{label}: {array_path.name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize correlated, millimeter, and reprojected ToF depth arrays."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Sample root or a directory containing ToF depth arrays.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-dir>/vis).",
    )
    parser.add_argument("--vmin", type=float, default=0.0, help="Minimum display depth in meters.")
    parser.add_argument("--vmax", type=float, default=8.33, help="Maximum display depth in meters.")
    args = parser.parse_args()

    output_dir = args.out_dir or args.input_dir / "vis"
    files = find_depth_files(args.input_dir)
    if not files:
        print(f"No supported depth arrays found in {args.input_dir}")
        return

    rendered = 0
    for source_path, (units, label) in files:
        relative_parent = source_path.parent.relative_to(args.input_dir)
        output_path = output_dir / relative_parent / f"{source_path.stem}.png"
        if save_heatmap(source_path, output_path, units, label, args.vmin, args.vmax):
            rendered += 1

    print(f"Visualized {rendered}/{len(files)} depth arrays.")
    print(f"PNGs saved to {output_dir}")


if __name__ == "__main__":
    main()
