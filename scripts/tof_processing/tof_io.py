import os
import re
import sys

import numpy as np


TRANSIENT_BASENAME = "data_transient_phasor"
STEADY_BASENAME = "data_steady_phasor"
DEFAULT_TRANSIENT_INPUT = os.path.join("npy_renders", "transient", f"{TRANSIENT_BASENAME}.npy")


def frame_suffix(frame: int | None) -> str:
    return f"_{frame:04d}" if frame is not None else ""


def build_transient_output_path(out_dir: str, frame: int | None = None) -> str:
    suffix = frame_suffix(frame)
    return os.path.join(out_dir, "transient", f"{TRANSIENT_BASENAME}{suffix}.npy")


def build_steady_output_path(out_dir: str, frame: int | None = None) -> str:
    suffix = frame_suffix(frame)
    return os.path.join(out_dir, "steady", f"{STEADY_BASENAME}{suffix}.npy")


def _infer_frame_suffix_from_transient(transient_file: str) -> str:
    stem = os.path.splitext(os.path.basename(transient_file))[0]
    if not stem.startswith(TRANSIENT_BASENAME):
        return ""
    suffix = stem[len(TRANSIENT_BASENAME) :]
    if suffix == "":
        return ""
    if re.fullmatch(r"_\d{4}", suffix):
        return suffix
    return ""


def resolve_input_path(input_path: str | None) -> str:
    if input_path is not None:
        return input_path
    return DEFAULT_TRANSIENT_INPUT


def default_freq_sidecar(transient_file: str) -> str:
    if transient_file.endswith(".npy"):
        return transient_file[:-4] + "_frequencies_hz.npy"
    return transient_file + "_frequencies_hz.npy"


def legacy_freq_sidecar_candidates(transient_file: str) -> list[str]:
    frame_tag = _infer_frame_suffix_from_transient(transient_file)
    transient_dir = os.path.dirname(transient_file)
    run_root = os.path.dirname(transient_dir)

    candidates = [
        os.path.join(run_root, "frequencies", f"{TRANSIENT_BASENAME}_frequencies_hz{frame_tag}.npy"),
        os.path.join(transient_dir, f"{TRANSIENT_BASENAME}_frequencies_hz{frame_tag}.npy"),
    ]
    if frame_tag:
        candidates.append(os.path.join(run_root, "frequencies", f"{TRANSIENT_BASENAME}_frequencies_hz.npy"))

    deduped: list[str] = []
    for p in candidates:
        if p not in deduped:
            deduped.append(p)
    return deduped


def resolve_steady_path(transient_file: str, steady_override: str | None = None) -> str:
    if steady_override is not None:
        return steady_override

    frame_tag = _infer_frame_suffix_from_transient(transient_file)
    transient_dir = os.path.dirname(transient_file)
    run_root = os.path.dirname(transient_dir)
    canonical = os.path.join(run_root, "steady", f"{STEADY_BASENAME}{frame_tag}.npy")
    fallback = transient_file.replace("transient", "steady")

    for candidate in (canonical, fallback):
        if os.path.exists(candidate):
            return candidate
    return canonical


def load_phasor_data(path: str) -> np.ndarray:
    if not os.path.exists(path):
        print(f"ERROR: Input file not found: {path}")
        sys.exit(1)
    if not os.path.isfile(path):
        print(
            f"ERROR: --input must be a .npy file, but got a directory/path: {path}\n"
            "Example: --input original_math/transient/data_transient_phasor.npy"
        )
        sys.exit(1)
    if not path.endswith(".npy"):
        print(
            f"ERROR: --input must point to a .npy file: {path}\n"
            "Example: --input original_math/transient/data_transient_phasor.npy"
        )
        sys.exit(1)

    data = np.load(path).astype(np.float32)
    if data.ndim != 4 or data.shape[-1] != 2:
        print(f"ERROR: Unexpected data shape {data.shape}. Expected (H, W, F, 2).")
        sys.exit(1)
    return data


def load_frequency_axis_hz(transient_path: str) -> np.ndarray:
    sidecars = [default_freq_sidecar(transient_path), *legacy_freq_sidecar_candidates(transient_path)]
    sidecar = next((p for p in sidecars if os.path.exists(p)), None)
    if sidecar is None:
        expected = default_freq_sidecar(transient_path)
        print(
            f"ERROR: Frequency sidecar missing for transient file: {transient_path}\n"
            f"Expected sidecar: {expected}\n"
            "Re-render with render_cw_tof.py to generate it."
        )
        sys.exit(1)

    freqs_hz = np.load(sidecar).astype(np.float64)
    if freqs_hz.ndim != 1:
        print(f"ERROR: Frequency sidecar must be 1D. Got shape {freqs_hz.shape}.")
        sys.exit(1)
    return freqs_hz


def validate_frequency_axis_matches_data(freqs_hz: np.ndarray, data: np.ndarray) -> None:
    if freqs_hz.shape[0] != data.shape[2]:
        print(
            f"ERROR: Frequency axis length {freqs_hz.shape[0]} does not match data bins {data.shape[2]}."
        )
        sys.exit(1)


def load_calibration(filename: str) -> dict:
    """Loads camera calibration matrices from a YAML file using OpenCV."""
    import cv2
    from pathlib import Path
    
    if not Path(filename).exists():
        raise FileNotFoundError(f"Calibration file not found: {filename}")

    fs = cv2.FileStorage(filename, cv2.FileStorage_READ)
    calib = {
        "K_iToF": fs.getNode("K_iToF").mat(),
        "dist_iToF": fs.getNode("dist_iToF").mat(),
        "K_RGB": fs.getNode("K_RGB").mat(),
        "dist_RGB": fs.getNode("dist_RGB").mat(),
        "R": fs.getNode("R").mat(),
        "T": fs.getNode("T").mat(),
    }
    fs.release()
    return calib


def get_planar_lut_path(width: int, height: int, calib_path: str) -> str:
    """Generates a predictable path for the planar LUT based on resolution and calibration."""
    calib_stem = os.path.splitext(os.path.basename(calib_path))[0]
    return os.path.join(os.path.dirname(calib_path), f"lut_planar_{calib_stem}_{width}x{height}.npy")


def load_planar_lut(lut_path: str) -> np.ndarray | None:
    """Loads a precomputed planar correction LUT if it exists."""
    if os.path.exists(lut_path):
        return np.load(lut_path)
    return None


def save_planar_lut(lut_path: str, lut: np.ndarray) -> None:
    """Saves a computed planar correction LUT to disk."""
    lut_dir = os.path.dirname(lut_path)
    if lut_dir:
        os.makedirs(lut_dir, exist_ok=True)
    np.save(lut_path, lut)
    print(f"✓ Saved planar correction LUT: {lut_path}")
