"""
SVG icon -> positioned path data, fit into a world-space region.

Extracts every <path> element's `d` attribute from a vendored icon
SVG, concatenates them into one combined path (multiple subpaths in
one string — SVG handles this natively, and getTotalLength()/
getPointAtLength() treat it as one continuous length for the
stroke-reveal animation, which is what we want: the pen draws subpath
1, then subpath 2, in document order).

LIMITATION, stated plainly rather than silently mishandled: this only
handles <path> elements. Tabler/Phosphor/Lucide/Iconoir icons are
overwhelmingly path-based (that's WHY those libraries were picked —
see the original brainstorm), but a small number of icons use <circle>,
<rect>, or <line> primitives instead. Those elements are currently
SKIPPED, not converted — an icon that's ONLY primitives (no <path> at
all) will resolve to an empty path here. If that happens in practice,
either swap that concept_key to a different icon library's version
that IS path-based, or extend this module to convert primitives to
path equivalents (straightforward — a <circle> is two arc commands —
just not needed until a real case shows up).
"""
import re
import xml.etree.ElementTree as ET
from typing import Optional

SVG_NS = "{http://www.w3.org/2000/svg}"


def _parse_viewbox(svg_root) -> tuple:
    vb = svg_root.get("viewBox")
    if vb:
        parts = [float(p) for p in vb.replace(",", " ").split()]
        if len(parts) == 4:
            return tuple(parts)  # (min_x, min_y, width, height)
    w = float(re.sub(r"[^0-9.]", "", svg_root.get("width", "24")) or 24)
    h = float(re.sub(r"[^0-9.]", "", svg_root.get("height", "24")) or 24)
    return (0.0, 0.0, w, h)


def extract_paths(svg_path: str) -> Optional[str]:
    """Returns a combined `d` string in the SVG's OWN coordinate
    space (unscaled) — caller applies fit-to-region scaling. None if
    the file has no usable <path> elements (see LIMITATION above)."""
    try:
        tree = ET.parse(svg_path)
    except Exception as e:
        print(f"[svg_to_path] failed to parse {svg_path}: {e}")
        return None

    root = tree.getroot()
    ds = []
    for el in root.iter():
        tag = el.tag.replace(SVG_NS, "")
        if tag == "path":
            d = el.get("d")
            if d:
                ds.append(d)

    if not ds:
        print(f"[svg_to_path] no <path> elements found in {svg_path} — "
              f"see module docstring LIMITATION (primitive-only icon)")
        return None

    return " ".join(ds)


def icon_to_path_d(svg_path: str, region: dict, padding_ratio: float = 0.15) -> Optional[dict]:
    """Fits the icon's native viewBox into `region` (world-space
    {x,y,w,h}), preserving aspect ratio, centered, with padding.
    Returns {"d": "...", "transform_note": "already baked into d"} —
    like text_to_path, the scale/translate is applied to the actual
    path coordinates via a wrapper transform string, kept SEPARATE
    from the path `d` itself here (icons are simple enough this is
    cheap to do at render time via an SVG <g transform>, unlike glyph
    paths which get pre-baked in Python for the fit-before-layout
    reason explained in text_to_path.py).
    """
    d = extract_paths(svg_path)
    if d is None:
        return None

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception:
        return None

    vb_x, vb_y, vb_w, vb_h = _parse_viewbox(root)

    pad_x = region["w"] * padding_ratio
    pad_y = region["h"] * padding_ratio
    avail_w = region["w"] - 2 * pad_x
    avail_h = region["h"] - 2 * pad_y

    scale = min(avail_w / vb_w, avail_h / vb_h) if vb_w and vb_h else 1.0
    drawn_w, drawn_h = vb_w * scale, vb_h * scale

    offset_x = region["x"] + (region["w"] - drawn_w) / 2 - vb_x * scale
    offset_y = region["y"] + (region["h"] - drawn_h) / 2 - vb_y * scale

    transform = f"translate({offset_x:.2f}, {offset_y:.2f}) scale({scale:.4f})"

    return {"d": d, "transform": transform}
