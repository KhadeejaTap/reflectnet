import sys

import numpy as np


C = 299_792_458.0


def select_bin_for_target(freqs_hz: np.ndarray, target_hz: float, tolerance_ratio: float) -> tuple[int, float, float]:
    idx = int(np.argmin(np.abs(freqs_hz - target_hz)))
    selected_hz = float(freqs_hz[idx])
    rel_err = abs(selected_hz - target_hz) / target_hz

    if rel_err > tolerance_ratio:
        print(
            "ERROR: Nearest frequency bin too far from requested target.\n"
            f"  target:   {target_hz/1e6:.6f} MHz\n"
            f"  selected: {selected_hz/1e6:.6f} MHz (bin {idx})\n"
            f"  rel err:  {100*rel_err:.3f}% > tolerance {100*tolerance_ratio:.3f}%"
        )
        sys.exit(1)

    return idx, selected_hz, rel_err


def extract_phase_and_amplitude(data: np.ndarray, bin_idx: int) -> tuple[np.ndarray, np.ndarray]:
    re_f = data[..., bin_idx, 0]
    im_f = data[..., bin_idx, 1]

    # REVERSIBLE FIX: Negate phase to fix depth inversion.
    # Mitsuba's transient renderer uses a negative phase shift for increasing distance (O.P.L. delay).
    # Without negation, small distances (e.g. 0.1m) wrap around to the back of the modulo range (e.g. 8.2m).
    #phase = np.arctan2(im_f, re_f) # Original (inverted)
    phase = -np.arctan2(im_f, re_f)  # Corrected (aligned)

    amplitude = np.sqrt(re_f**2 + im_f**2)
    return phase, amplitude


def compute_single_depth_wrapped(phase: np.ndarray, selected_hz: float) -> tuple[np.ndarray, float]:
    depth = (C * phase) / (4.0 * np.pi * selected_hz)
    max_range = C / (2.0 * selected_hz)
    depth_wrapped = np.mod(depth, max_range)
    return depth_wrapped, max_range


def compute_dual_depth_wrapped(
    phi_90: np.ndarray, phi_72: np.ndarray, selected_90_hz: float, selected_72_hz: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    phi_diff = np.arctan2(np.sin(phi_90 - phi_72), np.cos(phi_90 - phi_72))
    delta_f = abs(selected_90_hz - selected_72_hz)
    if delta_f <= 0.0:
        print("ERROR: delta_f <= 0. Selected frequencies are identical.")
        sys.exit(1)
    depth_dual = (C * phi_diff) / (4.0 * np.pi * delta_f)
    max_range_dual = C / (2.0 * delta_f)
    depth_dual_wrapped = np.mod(depth_dual, max_range_dual)
    return depth_dual_wrapped, phi_diff, delta_f, max_range_dual


def compute_beat_depth(
    phi1: np.ndarray, phi2: np.ndarray, f1: float, f2: float
) -> tuple[np.ndarray, float]:
    """
    Computes coarse depth using the beat frequency (synthetic wavelength) method.
    Typically f2 > f1.
    """
    delta_f = abs(f2 - f1)
    # Beat phase: phase difference mapped to (-pi, pi]
    phi_beat = np.arctan2(np.sin(phi2 - phi1), np.cos(phi2 - phi1))
    max_range_beat = C / (2.0 * delta_f)
    depth_beat = (C * phi_beat) / (4.0 * np.pi * delta_f)
    return np.mod(depth_beat, max_range_beat), max_range_beat


def compute_crt_depth(
    phi1: np.ndarray, phi2: np.ndarray, f1: float, f2: float
) -> tuple[np.ndarray, float]:
    """
    Computes depth using the Chinese Remainder Theorem (modular approach).
    Assumes f1 and f2 are in a simple integer ratio (e.g., 4:5 for 72:90 MHz).
    """
    # Unambiguous ranges for individual frequencies
    r1 = C / (2.0 * f1)
    r2 = C / (2.0 * f2)

    # Individual wrapped depths
    d1 = np.mod((C * phi1) / (4.0 * np.pi * f1), r1)
    d2 = np.mod((C * phi2) / (4.0 * np.pi * f2), r2)

    # For 72:90 MHz, ratio is 4:5.
    # The synthetic range is LCM(r1, r2) = 5*r2 = 4*r1 = ~8.33m.
    # We find d such that d = k1*r1 + d1 = k2*r2 + d2
    # Simple search for k1, k2 that minimizes the difference
    # Since ratio is small (4:5), we can just check the few possible integers.
    best_d = np.zeros_like(d1)
    min_error = np.full_like(d1, np.inf)

    # For 4:5 ratio, k1 is in [0, 3] and k2 is in [0, 4]
    for k1 in range(4):
        for k2 in range(5):
            cand_d1 = k1 * r1 + d1
            cand_d2 = k2 * r2 + d2
            err = np.abs(cand_d1 - cand_d2)
            mask = err < min_error
            min_error[mask] = err[mask]
            best_d[mask] = (cand_d1[mask] + cand_d2[mask]) / 2.0

    max_range_crt = 4.0 * r1  # or 5.0 * r2
    return np.mod(best_d, max_range_crt), max_range_crt


def refine_depth(
    d_coarse: np.ndarray, phi_high: np.ndarray, f_high: float
) -> np.ndarray:
    """
    Refines a coarse depth estimate using the high-frequency phase.
    """
    r_high = C / (2.0 * f_high)
    d_high_wrapped = np.mod((C * phi_high) / (4.0 * np.pi * f_high), r_high)

    # d_refined = k * r_high + d_high_wrapped
    k = np.round((d_coarse - d_high_wrapped) / r_high)
    return k * r_high + d_high_wrapped


def generate_planar_correction_lut(
    fx: float, fy: float, width: int, height: int, centered: bool = True, cx: float = None, cy: float = None
) -> np.ndarray:
    """
    Generates a per-pixel correction factor (LUT) to convert Euclidean depth to Planar depth.
    
    Logic:
        In a pinhole model, ray distance 'd' and planar depth 'z' are related by:
        d = z * sqrt(( (u-cx)/fx )^2 + ( (v-cy)/fy )^2 + 1)
        Therefore: z = d / CorrectionFactor
        
    Args:
        fx, fy: Focal lengths in pixels.
        width, height: Sensor dimensions.
        centered: If True, assumes principal point is at (width/2, height/2).
        cx, cy: Optional specific principal point if centered is False.
        
    Returns:
        np.ndarray: A 2D array of shape (height, width) containing the correction factors.
    """
    if centered:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
    elif cx is None or cy is None:
        raise ValueError("Principal point (cx, cy) must be provided if centered=False")

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Precompute the denominator term: sqrt( (x_norm)^2 + (y_norm)^2 + 1 )
    correction_lut = np.sqrt(
        ((u - cx) / fx) ** 2 + 
        ((v - cy) / fy) ** 2 + 
        1.0
    )
    return correction_lut


def convert_euclidean_to_planar(depth_euclidean: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Applies the correction LUT to convert Euclidean (ray) depth to Planar (Z) depth.
    
    Formula: z = d / correction_factor
    """
    return depth_euclidean / lut
