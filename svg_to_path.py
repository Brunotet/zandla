"""
SVG icon -> positioned path data, fit into a world-space region.

IMPORTANT, confirmed via isolated testing (not assumed): stroke-dasharray
/ stroke-dashoffset patterns RESTART at every disconnected subpath (every
new "M" moveto command) — this is correct, spec'd SVG behavior, not a
bug in the browser. It means a single dasharray/dashoffset pair CANNOT
progressively reveal a multi-subpath icon (like a 6-stroke brain icon):
each short subpath fits entirely inside the "visible" portion of the
dash pattern regardless of the offset, so every subpath renders fully
solid immediately. An earlier version of this module concatenated all
subpaths into one `d` string assuming cumulative reveal would work —
it doesn't, for anything with more than one subpath. Confirmed via a
minimal isolated Playwright test before writing this fix.

The correct approach — and a better visual match for "a hand actually
drawing this icon" besides — is to keep each subpath SEPARATE and
reveal them one at a time, in sequence. That's what this module now
returns: a LIST of individual subpath `d` strings, not one merged
string. scene_template.html animates each list entry with its own
dasharray/dashoffset, in order.
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


def extract_paths(svg_path: str) -> Optional[list]:
    """Returns a LIST of `d` strings (one per VISIBLE <path> element
    found), in the SVG's OWN coordinate space (unscaled) — caller
    applies fit-to-region scaling. None if the file has no usable
    <path> elements.

    FIXED (confirmed via isolated test, not assumed): the previous
    version used root.iter(), which walks EVERY element in the tree
    including ones inside <defs>, <clipPath>, <mask>, and <symbol> —
    hidden DEFINITIONS, never directly rendered, just referenced by
    id elsewhere. A number of icons (Tabler included) use exactly
    this pattern — e.g. a clipPath containing a full bounding-box
    path. Walking the whole tree blindly pulled that phantom path in
    alongside the real visible strokes, producing extra/wrong
    geometry in the reveal. Now skips anything nested inside those
    container tags, at any depth.

    LIMITATION, stated plainly: this only handles <path> elements.
    Tabler/Phosphor/Lucide/Iconoir icons are overwhelmingly path-based
    (that's WHY those libraries were picked), but a small number of
    icons use <circle>, <rect>, or <line> primitives instead — those
    are currently SKIPPED, not converted. An icon that's ONLY
    primitives resolves to an empty list here. If that happens in
    practice, swap that concept_key to a different library's version,
    or extend this to convert primitives to path equivalents.
    """
    try:
        tree = ET.parse(svg_path)
    except Exception as e:
        print(f"[svg_to_path] failed to parse {svg_path}: {e}")
        return None

    root = tree.getroot()
    ds = []
    SKIP_CONTAINER_TAGS = {"defs", "clipPath", "mask", "symbol"}

    def _walk(el, hidden):
        tag = el.tag.replace(SVG_NS, "")
        if tag in SKIP_CONTAINER_TAGS:
            hidden = True
        if tag == "path" and not hidden:
            d = el.get("d")
            if d:
                ds.append(d)
        for child in el:
            _walk(child, hidden)

    _walk(root, False)

    if not ds:
        print(f"[svg_to_path] no <path> elements found in {svg_path} — "
              f"see module docstring LIMITATION (primitive-only icon)")
        return None

    return ds


def icon_to_path_d(svg_path: str, region: dict, padding_ratio: float = 0.15) -> Optional[dict]:
    """Fits the icon's native viewBox into `region` (world-space
    {x,y,w,h}), preserving aspect ratio, centered, with padding.
    Returns {"subpaths": ["d1", "d2", ...], "transform": "..."} — the
    SAME transform string applies to every subpath (they all share the
    icon's coordinate space), kept separate from the path `d` values
    themselves so the browser can apply it once per <path> element.
    """
    subpaths = extract_paths(svg_path)
    if subpaths is None:
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

    return {"subpaths": subpaths, "transform": transform, "scale": scale,
            "offset_x": offset_x, "offset_y": offset_y}
