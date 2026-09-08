"""
CW-ToF Dual-Frequency Unwrapping and Planar Correction

This script processes raw transient phasor data (at 72 MHz and 90 MHz) to 
reconstruct metric depth maps. It handles both phase unwrapping and 
Euclidean-to-Planar (Z-depth) conversion.

LOGIC:
    1. Phase Unwrapping:
       Uses two frequencies (72/90 MHz) to extend the unambiguous range from 
       ~1.6m (at 90MHz) to ~8.33m. Supports 'beat' (synthetic wavelength) 
       and 'crt' (Chinese Remainder Theorem) methods.
       
    2. Planar Correction:
       ToF sensors measure Euclidean (ray) distance. This script uses a 
       precomputed Look-Up Table (LUT) or hardcoded ideal focal lengths 
       to convert these rays into planar Z-depth:
       Z = Euclidean_Depth / sqrt( ((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1 )

USAGE:
    # Single file
    python3 scripts/tof_processing/correlate_unwrapped_depth.py \
        --input path/to/transient.npy --out-dir output_folder --planar

    # Full directory
    python3 scripts/tof_processing/correlate_unwrapped_depth.py \
        --input-dir path/to/transient_dir --out-dir output_folder --planar

EXAMPLE:
    python3 scripts/tof_processing/correlate_unwrapped_depth.py \
        --input-dir flyingpixels/closer/renders_tof/transient \
        --out-dir processed_results --planar
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tof_io import (
    DEFAULT_TRANSIENT_INPUT,
    get_planar_lut_path,
    load_frequency_axis_hz,
    load_phasor_data,
    load_planar_lut,
    resolve_input_path,
    save_planar_lut,
    validate_frequency_axis_matches_data,
)
from tof_math import (
    compute_beat_depth,
    compute_crt_depth,
    convert_euclidean_to_planar,
    extract_phase_and_amplitude,
    generate_planar_correction_lut,
    select_bin_for_target,
)
from tof_outputs import build_prefix, save_unwrapped_outputs

# IDEAL CONSTANTS (Code-as-Calibration)
# Precise fx from Helios2 Ray calibration, used for both axes (square pixels)
IDEAL_F_TOF = 544.462653

def run(args: argparse.Namespace) -> None:
    input_file = resolve_input_path(args.input)
    data = load_phasor_data(input_file)
    freqs_hz = load_frequency_axis_hz(input_file)
    validate_frequency_axis_matches_data(freqs_hz, data)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Extract 72/90 MHz bins
    bin_90, f_90, _ = select_bin_for_target(freqs_hz, 90.0e6, args.tol)
    bin_72, f_72, _ = select_bin_for_target(freqs_hz, 72.0e6, args.tol)

    phi_90, amp_90 = extract_phase_and_amplitude(data, bin_90)
    phi_72, amp_72 = extract_phase_and_amplitude(data, bin_72)

    # 2. Compute unwrapped depth (coarse)
    if args.method == "beat":
        d_unwrapped, max_range = compute_beat_depth(phi_72, phi_90, f_72, f_90)
    else:  # crt
        d_unwrapped, max_range = compute_crt_depth(phi_72, phi_90, f_72, f_90)

    # 3. Optional: Convert Euclidean (Ray) depth to Planar (Z) depth
    # Mitsuba simulations often assume a perfectly centered principal point.
    if args.planar:
        h, w = d_unwrapped.shape
        # Prioritize the "ideal" LUT for simulation data, keyed by resolution.
        ideal_lut_path = get_planar_lut_path(w, h, "ideal")
        
        if os.path.exists(ideal_lut_path):
            lut = load_planar_lut(ideal_lut_path)
            print(f"✓ Loaded Ideal Planar LUT: {ideal_lut_path}")
        else:
            # Fallback: Generate it on the fly using hardcoded IDEAL_F_TOF
            # Scale focal length to match the current resolution (relative to base 640x480)
            base_width, base_height = 640, 480
            fx_scaled = IDEAL_F_TOF * (w / base_width)
            fy_scaled = IDEAL_F_TOF * (h / base_height)
            print(f"⚠ Ideal LUT not found. Generating from scaled fx={fx_scaled:.2f}, fy={fy_scaled:.2f}...")
            # Use scaled focal length for both axes to ensure square pixels and correct FOV
            lut = generate_planar_correction_lut(
                fx_scaled, fy_scaled, w, h, centered=True
            )
            save_planar_lut(ideal_lut_path, lut)
            
        d_unwrapped = convert_euclidean_to_planar(d_unwrapped, lut)
        print("✓ Converted Euclidean (Ray) depth to Planar (Z) depth (Ideal Constants)")

    # If user provided a max depth, use it instead of the computed unambiguous range.
    # Keep the original computed max_range available for metadata.
    effective_max_range = args.max_depth if getattr(args, "max_depth", None) is not None else max_range

    # 4. Masking
    cutoff_90 = args.amp_threshold * amp_90.max()
    cutoff_72 = args.amp_threshold * amp_72.max()
    
    # Combined mask: Amplitude threshold + Unambiguous range check
    valid_mask = (amp_90 >= cutoff_90) & (amp_72 >= cutoff_72)
    valid_mask &= (d_unwrapped >= 0.0) & (d_unwrapped <= max_range)
    total_pixels = valid_mask.size
    initial_valid_count = int(np.count_nonzero(valid_mask))
    initial_valid_pct = 100.0 * initial_valid_count / total_pixels

    depth_masked = d_unwrapped.copy()
    depth_masked[~valid_mask] = np.nan
    final_valid_count = int(np.count_nonzero(valid_mask))
    final_valid_pct = 100.0 * final_valid_count / total_pixels

    # 5. Save outputs
    prefix = build_prefix(args.out_dir, args.prefix, "depth_dual_unwrapped")
    meta = {
        "method": args.method,
        "input_file": input_file,
        "f1_hz": f_72,
        "f2_hz": f_90,
        "planar_correction": args.planar,
        "computed_unambiguous_range_m": max_range,
        "unambiguous_range_m": effective_max_range,
        "user_max_depth": args.max_depth,
        "amp_threshold": args.amp_threshold,
        "initial_valid_pixels": initial_valid_count,
        "initial_valid_percent": initial_valid_pct,
        "final_valid_pixels": final_valid_count,
        "final_valid_percent": final_valid_pct,
    }

    paths = save_unwrapped_outputs(prefix, depth_masked, valid_mask, meta)

    print(f"Dual-frequency unwrapping complete ({args.method} method)")
    print(f"Unambiguous range: {max_range:.4f} m")
    print(f"Initial validity: {initial_valid_count}/{total_pixels} ({initial_valid_pct:.2f}%)")
    print(f"Final validity: {final_valid_count}/{total_pixels} ({final_valid_pct:.2f}%)")
    print(f"Saved coarse depth: {paths['unwrapped']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate and unwrap 72/90 MHz phasor data.")
    
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single transient phasor .npy input.",
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing multiple transient .npy files.",
    )

    parser.add_argument("--out-dir", type=str, default="npy_renders", help="Output directory.")
    parser.add_argument("--method", choices=["beat", "crt"], default="beat", help="Unwrapping method.")
    parser.add_argument("--planar", action=argparse.BooleanOptionalAction, default=True, help="Convert Euclidean (Ray) depth to Planar (Z) depth using ideal constants (default: True).")
    parser.add_argument("--amp-threshold", type=float, default=0.01, help="Amplitude threshold ratio.")
    parser.add_argument("--max-depth", type=float, default=None, help="Optional maximum depth (meters). If set, overrides computed unambiguous range.")
    parser.add_argument("--tol", type=float, default=0.002)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    if args.input_dir:
        indir = Path(args.input_dir)
        # Find all .npy files that aren't frequencies or metadata
        files = sorted([
            f for f in indir.glob("*.npy") 
            if "_frequencies_hz" not in f.name and "meta" not in f.name
        ])
        
        if not files:
            print(f"No valid .npy files found in {args.input_dir}")
            return

        print(f"Batch Processing: Found {len(files)} files.")
        for i, fpath in enumerate(files):
            # Update args.input for each run
            args.input = str(fpath)
            # Use filename stem as prefix to prevent overwriting
            args.prefix = fpath.stem
            
            run(args)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                print(f"  Processed {i + 1}/{len(files)} frames...")
    else:
        # Fallback to default or explicit single input
        if args.input is None:
            args.input = DEFAULT_TRANSIENT_INPUT
        run(args)


if __name__ == "__main__":
    main()
