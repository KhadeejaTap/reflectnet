"""
Render one instance's camera sequence with Mitsuba transient ToF rendering.

Usage:
    python scripts/render_instance.py <instance_id> <ply_dir> <output_dir>

The camera resolution and FOV are intentionally kept local to this renderer.
The transient sensor settings follow the working CW-ToF scene configuration.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mitsuba as mi

mi.set_variant("llvm_ad_mono")

import mitransient  # noqa: F401  # Registers transient Mitsuba plugins.

from patch_material import FLOOR_REFLECTANCE, REFLECTIVITY_MAP
from explore_data import load_metadata, filter_matte_nonmetal

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_ROOT = REPO_ROOT / "mitsuba_scenes" / "instances"
TOF_SCRIPTS = REPO_ROOT / "scripts" / "tof_processing"
REPROJECTION_SCRIPTS = REPO_ROOT / "scripts" / "reprojection"

FOV_DEGREES = 27.0
RES_X = 432
RES_Y = 320
SAMPLE_COUNT = 64

WL_MEAN = 3.747406
WL_SIGMA = 20.0
TEMPORAL_BINS = 512
BIN_WIDTH_OPL = 0.032529563585

SPOT_CUTOFF_ANGLE = 20.0
SPOT_BEAM_WIDTH = 13.0
SPOT_INTENSITY = 100.0


def build_tof_scene_dict(to_world_matrix, floor_ply, object_ply, object_reflectance):
    """Build the transient scene while preserving this renderer's camera setup."""
    # Scene dictionaries require a scalar transform during plugin construction.
    to_world = mi.ScalarTransform4f(np.asarray(to_world_matrix, dtype=np.float32))

    return {
        "type": "scene",
        "integrator": {"type": "transient_path", "max_depth": -1},
        "sensor": {
            "type": "perspective",
            "fov": FOV_DEGREES,
            "to_world": to_world,
            "sampler": {"type": "independent", "sample_count": SAMPLE_COUNT},
            "film": {
                "type": "phasor_hdr_film",
                "width": RES_X,
                "height": RES_Y,
                "wl_mean": WL_MEAN,
                "wl_sigma": WL_SIGMA,
                "temporal_bins": TEMPORAL_BINS,
                "bin_width_opl": BIN_WIDTH_OPL,
                "start_opl": 0.0,
                "rfilter": {"type": "box"},
            },
        },
        "vcsel_center": {
            "type": "spot",
            "to_world": to_world,
            "intensity": {"type": "rgb", "value": [SPOT_INTENSITY] * 3},
            "cutoff_angle": SPOT_CUTOFF_ANGLE,
            "beam_width": SPOT_BEAM_WIDTH,
        },
        "floor": {
            "type": "ply",
            "filename": str(floor_ply),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [FLOOR_REFLECTANCE] * 3},
            },
        },
        "object": {
            "type": "ply",
            "filename": str(object_ply),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [object_reflectance] * 3},
            },
        },
    }


def _parse_vector(value):
    return [float(part) for part in value.replace(",", " ").split()]


def build_tof_scene_dict_from_xml(xml_path, to_world_matrix):
    """Use XML shapes/materials and add the missing transient render components."""
    xml_path = Path(xml_path).resolve()
    root = ET.parse(xml_path).getroot()
    bsdfs = {}

    for bsdf in root.findall("bsdf"):
        diffuse = bsdf.find(".//bsdf[@type='diffuse']")
        material = diffuse if diffuse is not None else bsdf
        reflectance = material.find("rgb[@name='reflectance']")
        if reflectance is None:
            reflectance = material.find("rgb")
        if reflectance is None:
            raise ValueError(f"XML material {bsdf.get('id')!r} has no RGB reflectance.")
        bsdfs[bsdf.get("id")] = {
            "type": "diffuse",
            "reflectance": {
                "type": "rgb",
                "value": _parse_vector(reflectance.get("value", "")),
            },
        }

    shapes = {}
    for index, shape in enumerate(root.findall("shape")):
        filename = shape.find("string[@name='filename']")
        bsdf_ref = shape.find("ref[@name='bsdf']")
        if filename is None or bsdf_ref is None:
            raise ValueError(f"XML shape {shape.get('id')!r} is missing filename or BSDF ref.")
        bsdf_id = bsdf_ref.get("id")
        if bsdf_id not in bsdfs:
            raise ValueError(f"XML shape references unknown BSDF {bsdf_id!r}.")
        shape_key = shape.get("id") or shape.get("name") or f"shape_{index}"
        shape_dict = {
            "type": "ply",
            "filename": str((xml_path.parent / filename.get("value")).resolve()),
            "bsdf": bsdfs[bsdf_id],
        }
        face_normals = shape.find("boolean[@name='face_normals']")
        if face_normals is not None:
            shape_dict["face_normals"] = face_normals.get("value", "false").lower() == "true"
        shapes[shape_key] = shape_dict

    scene = build_tof_scene_dict(
        to_world_matrix,
        Path(shapes[next(iter(shapes))]["filename"]),
        Path(shapes[list(shapes)[-1]]["filename"]),
        0.0,
    )
    scene.pop("floor")
    scene.pop("object")
    scene.update(shapes)
    return scene


def find_scene_xml(ply_dir):
    ply_dir = Path(ply_dir)
    for candidate in (ply_dir.parent / "scene.xml", ply_dir.parent / "base.xml"):
        if candidate.exists():
            return candidate
    return None


def get_reflectivity_for_instance(instance_id):
    df = load_metadata()
    kept, _dropped = filter_matte_nonmetal(df)
    rows = kept[kept["instance_id"] == instance_id]
    if rows.empty:
        raise ValueError(f"Instance {instance_id!r} is not a surviving matte non-metal instance.")
    return REFLECTIVITY_MAP[rows.iloc[0]["reflectivity"]]


def _load_correlation_runner():
    sys.path.insert(0, str(TOF_SCRIPTS))
    from correlate_unwrapped_depth import run

    return run


def render_instance(
    instance_id,
    floor_ply,
    object_ply,
    output_dir,
    *,
    limit=None,
    correlate=True,
    reproject=True,
    spp=None,
    amp_threshold=0.05,
    scene_xml=None,
):
    cameras_path = INSTANCES_ROOT / instance_id / "cameras.json"
    with open(cameras_path, "r", encoding="utf-8") as f:
        cameras = json.load(f)

    if limit is not None:
        cameras = cameras[:limit]

    output_dir = Path(output_dir)
    transient_dir = output_dir / "transient"
    steady_dir = output_dir / "steady"
    frequency_dir = output_dir / "frequencies"
    transient_dir.mkdir(parents=True, exist_ok=True)
    steady_dir.mkdir(parents=True, exist_ok=True)
    frequency_dir.mkdir(parents=True, exist_ok=True)

    if scene_xml is not None:
        scene_dict = build_tof_scene_dict_from_xml(
            scene_xml, cameras[0]["transform_matrix"]
        )
        print(f"Loading geometry and materials from XML: {scene_xml}")
    else:
        object_reflectance = get_reflectivity_for_instance(instance_id)
        scene_dict = build_tof_scene_dict(
            cameras[0]["transform_matrix"], floor_ply, object_ply, object_reflectance
        )
    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)
    render_spp = spp or scene.sensors()[0].sampler().sample_count()

    frequencies_hz = (
        np.asarray(scene.sensors()[0].film().frequencies, dtype=np.float64).reshape(-1)
        * 299792458.0
    )

    correlate_run = _load_correlation_runner() if correlate else None
    project_depth_ideal = None
    if reproject:
        sys.path.insert(0, str(REPROJECTION_SCRIPTS))
        from ideal_projection import project_depth_ideal

    for camera in cameras:
        frame_id = int(camera["frame_id"])
        frame_name = f"frame_{frame_id:05d}"
        transform = mi.Transform4f(np.asarray(camera["transform_matrix"], dtype=np.float32))

        params["sensor.to_world"] = transform
        params["vcsel_center.to_world"] = transform
        params.update()

        steady, transient = mi.render(scene, spp=render_spp)
        transient_path = transient_dir / f"{frame_name}.npy"
        np.save(transient_path, np.asarray(transient))
        np.save(steady_dir / f"{frame_name}.npy", np.asarray(steady))
        np.save(frequency_dir / f"{frame_name}_frequencies_hz.npy", frequencies_hz)
        np.save(transient_dir / f"{frame_name}_frequencies_hz.npy", frequencies_hz)

        if correlate_run is not None:
            correlation_args = argparse.Namespace(
                input=str(transient_path),
                out_dir=str(output_dir / "processed"),
                method="beat",
                amp_threshold=amp_threshold,
                tol=0.002,
                planar=True,
                max_depth=None,
                prefix=frame_name,
            )
            correlate_run(correlation_args)

        if reproject:
            processed_dir = output_dir / "processed"
            depth_m = np.load(processed_dir / f"{frame_name}_tof.npy")
            valid_mask = np.load(processed_dir / f"{frame_name}_valid_mask.npy")
            depth_mm = depth_m.astype(np.float32) * 1000.0
            tof_mm_dir = output_dir / "tof_mm"
            tof_mm_dir.mkdir(parents=True, exist_ok=True)
            np.save(tof_mm_dir / f"{frame_name}_tof_mm.npy", depth_mm)
            projected = project_depth_ideal(
                depth_mm,
                valid_mask,
                quantize_mm=0.25,
                rgb_width=992,
                rgb_height=992,
            )
            reproject_dir = output_dir / "reprojected_rgb"
            reproject_dir.mkdir(parents=True, exist_ok=True)
            np.save(reproject_dir / f"{frame_name}_depth_proj_mm.npy", projected)

        print(f"Rendered {frame_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render one instance with transient ToF.")
    parser.add_argument("instance_id")
    parser.add_argument("ply_dir", type=Path)
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Output directory (default: <repository>/tof/<instance_id>).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--spp", type=int)
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=None,
        help="Geometry/material XML fragment; auto-detected beside ply_dir when present.",
    )
    parser.add_argument(
        "--amp-threshold",
        type=float,
        default=0.002,
        help="Correlation amplitude threshold ratio (default: 0.05).",
    )
    parser.add_argument("--no-correlate", action="store_true")
    parser.add_argument("--no-reproject", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or REPO_ROOT / "tof" / args.instance_id
    scene_xml = args.scene_xml or find_scene_xml(args.ply_dir)
    floor_ply = args.ply_dir / "bed_19_02_ground_plane.ply"
    object_ply = args.ply_dir / "bed_19_02.ply"
    render_instance(
        args.instance_id,
        floor_ply,
        object_ply,
        output_dir,
        limit=args.limit,
        correlate=not args.no_correlate,
        reproject=not args.no_reproject,
        spp=args.spp,
        amp_threshold=args.amp_threshold,
        scene_xml=scene_xml,
    )
