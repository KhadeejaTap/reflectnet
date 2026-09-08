"""
patch_material.py

Given a base Mitsuba XML (exported from Blender via MitsubaBlender) and an
instance's material info (from the metadata parquet / tags.json), patch the
<bsdf> reflectance values:
  - the floor/ground-plane BSDF gets a fixed reflectance (approximating
    polished marble)
  - all other BSDFs get a reflectance derived from the instance's
    'reflectivity' tag (weak/medium/strong)

This only touches material, not camera/sensor — camera patching is handled
separately per-frame.
"""

import xml.etree.ElementTree as ET


FLOOR_REFLECTANCE = 0.65  # fixed, approximates polished marble under a diffuse BSDF

REFLECTIVITY_MAP = {
    "weak": 0.2,
    "medium": 0.5,
    "strong": 0.8,
}


def _find_floor_bsdf_id(root):
    """
    Identify which bsdf id corresponds to the floor/ground plane by looking
    at <shape> blocks whose mesh filename contains 'ground_plane', then
    following that shape's <ref id="..."/> to find its bsdf id.
    """
    for shape in root.findall("shape"):
        filename_el = shape.find("string[@name='filename']")
        if filename_el is None:
            continue
        if "ground_plane" in filename_el.get("value", ""):
            ref_el = shape.find("ref[@name='bsdf']")
            if ref_el is not None:
                return ref_el.get("id")
    return None


def patch_material(xml_string: str, reflectivity: str) -> str:
    """
    Parse the given Mitsuba XML string, set reflectance values, and return
    the patched XML as a string.

    Args:
        xml_string: raw XML content of the exported base.xml
        reflectivity: one of 'weak', 'medium', 'strong' (from tags.json)

    Returns:
        Patched XML string.
    """
    if reflectivity not in REFLECTIVITY_MAP:
        raise ValueError(f"Unknown reflectivity value: {reflectivity!r}")

    object_reflectance = REFLECTIVITY_MAP[reflectivity]

    root = ET.fromstring(xml_string)
    floor_bsdf_id = _find_floor_bsdf_id(root)

    for bsdf in root.findall("bsdf"):
        bsdf_id = bsdf.get("id")
        rgb_el = bsdf.find("rgb[@name='reflectance']")
        if rgb_el is None:
            continue

        if bsdf_id == floor_bsdf_id:
            value = FLOOR_REFLECTANCE
        else:
            value = object_reflectance

        rgb_el.set("value", f"{value:.6f} {value:.6f} {value:.6f}")

    return ET.tostring(root, encoding="unicode")


if __name__ == "__main__":
    # Quick manual test using the base.xml you exported for bed_19_02.
    import sys

    if len(sys.argv) != 4:
        print("Usage: python patch_material.py <base.xml> <reflectivity> <output.xml>")
        sys.exit(1)

    base_xml_path, reflectivity, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(base_xml_path, "r", encoding="utf-8") as f:
        xml_string = f.read()

    patched = patch_material(xml_string, reflectivity)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"Wrote patched XML to {output_path}")
