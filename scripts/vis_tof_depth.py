#!/usr/bin/env python3
"""
Simple ToF Depth Visualizer (Turbo Heatmap)

Generates PNG heatmaps from a directory of .npy ToF depth maps.
Uses a consistent color scale (0 to 8.33m) for comparison.

USAGE:
    python3 scripts/diagnostics/vis_tof_depth.py --input-dir amp001 --out-dir amp001/vis
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Visualize ToF depth maps with Turbo heatmap.")
    parser.add_argument("--input-dir", type=str, required=True, help="Dir with frame_XXXX_tof.npy")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: input/vis)")
    parser.add_argument("--vmin", type=float, default=0.0, help="Min depth for scale (meters)")
    parser.add_argument("--vmax", type=float, default=8.33, help="Max depth for scale (meters)")
    args = parser.parse_args()

    indir = Path(args.input_dir)
    outdir = Path(args.out_dir) if args.out_dir else indir / "vis"
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob("*_tof.npy"))
    if not files:
        print(f"No *_tof.npy files found in {args.input_dir}")
        return

    print(f"Visualizing {len(files)} frames...")

    for fpath in files:
        # Load and handle invalid values
        depth = np.load(fpath)
        masked_depth = np.ma.masked_invalid(depth)
        masked_depth = np.ma.masked_less_equal(masked_depth, 0)

        # Plot
        plt.figure(figsize=(10, 7), dpi=150)
        cmap = plt.get_cmap("turbo").copy()
        cmap.set_bad(color="black")

        im = plt.imshow(masked_depth, cmap=cmap, vmin=args.vmin, vmax=args.vmax)
        cbar = plt.colorbar(im)
        cbar.set_label("Planar Depth (m)")

        plt.title(f"ToF Depth Heatmap: {fpath.name}")
        plt.axis("off")
        plt.tight_layout()

        out_path = outdir / f"{fpath.stem}.png"
        plt.savefig(out_path)
        plt.close()

    print(f"Done! PNGs saved to {outdir}")

if __name__ == "__main__":
    main()
