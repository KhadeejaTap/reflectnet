import json
import os

import numpy as np


def build_prefix(out_dir: str, prefix: str | None, default_prefix: str) -> str:
    root = prefix if prefix is not None else default_prefix
    return os.path.join(out_dir, root)


def normalize_unwrapped_prefix(prefix: str) -> str:
    if prefix.endswith("_unwrapped"):
        return prefix[: -len("_unwrapped")]
    return prefix


def save_single_outputs(
    prefix: str,
    depth_wrapped: np.ndarray,
    depth_masked: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray,
    meta: dict,
) -> dict[str, str]:
    depth_path = f"{prefix}.npy"
    masked_path = f"{prefix}_masked.npy"
    amp_path = f"{prefix}_amplitude.npy"
    phase_path = f"{prefix}_phase.npy"
    meta_path = f"{prefix}_meta.json"

    np.save(depth_path, depth_wrapped.astype(np.float32))
    np.save(masked_path, depth_masked.astype(np.float32))
    np.save(amp_path, amplitude.astype(np.float32))
    np.save(phase_path, phase.astype(np.float32))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "depth": depth_path,
        "masked": masked_path,
        "amplitude": amp_path,
        "phase": phase_path,
        "meta": meta_path,
    }


def save_dual_outputs(
    prefix: str,
    depth_dual_wrapped: np.ndarray,
    depth_dual_masked: np.ndarray,
    phi_diff: np.ndarray,
    amp_90: np.ndarray,
    amp_72: np.ndarray,
    valid_mask: np.ndarray,
    meta: dict,
) -> dict[str, str]:
    depth_path = f"{prefix}.npy"
    masked_path = f"{prefix}_masked.npy"
    phase_diff_path = f"{prefix}_phase_diff.npy"
    amp90_path = f"{prefix}_amp90.npy"
    amp72_path = f"{prefix}_amp72.npy"
    valid_mask_path = f"{prefix}_valid_mask.npy"
    meta_path = f"{prefix}_meta.json"

    np.save(depth_path, depth_dual_wrapped.astype(np.float32))
    np.save(masked_path, depth_dual_masked.astype(np.float32))
    np.save(phase_diff_path, phi_diff.astype(np.float32))
    np.save(amp90_path, amp_90.astype(np.float32))
    np.save(amp72_path, amp_72.astype(np.float32))
    np.save(valid_mask_path, valid_mask.astype(np.uint8))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "depth": depth_path,
        "masked": masked_path,
        "phase_diff": phase_diff_path,
        "amp90": amp90_path,
        "amp72": amp72_path,
        "valid_mask": valid_mask_path,
        "meta": meta_path,
    }


def save_unwrapped_outputs(
    prefix: str,
    depth_unwrapped: np.ndarray,
    valid_mask: np.ndarray,
    meta: dict,
) -> dict[str, str]:
    output_prefix = normalize_unwrapped_prefix(prefix)
    unwrapped_path = f"{output_prefix}_tof.npy"
    valid_mask_path = f"{output_prefix}_valid_mask.npy"
    meta_path = f"{output_prefix}_meta.json"

    np.save(unwrapped_path, depth_unwrapped.astype(np.float32))
    np.save(valid_mask_path, valid_mask.astype(np.uint8))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "unwrapped": unwrapped_path,
        "valid_mask": valid_mask_path,
        "meta": meta_path,
    }
