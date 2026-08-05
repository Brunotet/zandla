"""
Text -> SVG path data, computed in Python via fonttools.

DESIGN DECISION: this runs server-side (during scene-program build),
NOT in the browser via opentype.js. Two reasons:
  1. Matches your standing rule — deterministic and cacheable over
     generative at render time. A path string computed once in Python
     is a plain value from then on; nothing font-related needs to
     load, parse, or race against anything else at render time.
  2. Precise control over sizing/fit BEFORE the camera/region layout
     is finalized — we need to know how wide the text is to decide
     the region size and camera framing, which is circular if the
     browser is the one computing glyph widths.

Font: Shadows Into Light (Google Fonts, OFL license — free, unlimited,
commercial use, no attribution required, vendored locally so there's
no runtime font-loading dependency at all). Swap FONT_PATH for a
different handwriting-style font if the aesthetic doesn't fit once
you see real output — nothing else here depends on which font it is.
"""
import os
from functools import lru_cache
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.misc.transform import Transform

FONT_PATH = os.path.join(os.path.dirname(__file__), "vendor", "fonts", "ShadowsIntoLight.ttf")


@lru_cache(maxsize=1)
def _load_font():
    font = TTFont(FONT_PATH)
    cmap = font.getBestCmap()
    glyph_set = font.getGlyphSet()
    hmtx = font["hmtx"]
    units_per_em = font["head"].unitsPerEm
    ascent = font["hhea"].ascent
    return {
        "cmap": cmap, "glyph_set": glyph_set, "hmtx": hmtx,
        "units_per_em": units_per_em, "ascent": ascent,
    }


def _glyph_name_for_char(cmap: dict, ch: str) -> str:
    cp = ord(ch)
    return cmap.get(cp, cmap.get(ord(" "), ".notdef"))


def text_advance_width(text: str, font_size: float) -> float:
    """Total width of `text` at `font_size`, font units converted to
    px — needed BEFORE path generation to decide how big to render
    (fit-to-region), avoiding a render/measure/re-render loop."""
    f = _load_font()
    scale = font_size / f["units_per_em"]
    total = 0
    for ch in text:
        if ch == " ":
            gname = _glyph_name_for_char(f["cmap"], " ")
        else:
            gname = _glyph_name_for_char(f["cmap"], ch)
        try:
            aw, _ = f["hmtx"][gname]
        except KeyError:
            aw = f["units_per_em"] * 0.5
        total += aw
    return total * scale


def fit_font_size(text: str, max_width: float, max_height: float,
                   min_size: float = 24, max_size: float = 160) -> float:
    """Binary-search-free direct fit: advance width scales linearly
    with font size, so we can solve for it in one shot rather than
    iterating."""
    f = _load_font()
    probe_size = 100.0
    probe_width = text_advance_width(text, probe_size)
    if probe_width <= 0:
        return min_size
    size_for_width = max_width / probe_width * probe_size
    size_for_height = max_height * 0.7  # leave room so ascenders/descenders don't clip the region
    size = min(size_for_width, size_for_height)
    return max(min_size, min(max_size, size))


def text_to_path_d(text: str, x: float, y: float, font_size: float) -> dict:
    """Returns {"d": "<svg path d string>", "width": total_px_width}.
    (x, y) is the TOP-LEFT of the text's bounding box in world-space —
    the function handles baseline math internally so callers don't
    need to think about font ascent/descent.

    Coordinate flip: font glyph coordinates are y-up (baseline at 0,
    ascenders positive). SVG screen space is y-down. TransformPen
    applies scale(s, -s) + translate so the OUTPUT path data is
    already in final screen coordinates — no transform attribute
    needed on the <path> element itself, which keeps getTotalLength()/
    getPointAtLength() (used for the stroke-reveal + hand-tracking
    animation) working in the same coordinate space as everything else
    on the board.
    """
    f = _load_font()
    scale = font_size / f["units_per_em"]
    baseline_y = y + f["ascent"] * scale  # top of bbox + ascent = baseline position

    d_parts = []
    cursor_x = x

    for ch in text:
        gname = _glyph_name_for_char(f["cmap"], ch)
        try:
            aw, _ = f["hmtx"][gname]
        except KeyError:
            aw = f["units_per_em"] * 0.5

        if ch != " ":
            transform = Transform(scale, 0, 0, -scale, cursor_x, baseline_y)
            svg_pen = SVGPathPen(f["glyph_set"])
            t_pen = TransformPen(svg_pen, transform)
            try:
                f["glyph_set"][gname].draw(t_pen)
                glyph_d = svg_pen.getCommands()
                if glyph_d:
                    d_parts.append(glyph_d)
            except Exception as e:
                print(f"[text_to_path] failed to draw glyph for '{ch}': {e}")

        cursor_x += aw * scale

    return {"d": " ".join(d_parts), "width": cursor_x - x}


if __name__ == "__main__":
    size = fit_font_size("the brain rewires itself", max_width=800, max_height=200)
    result = text_to_path_d("the brain rewires itself", x=100, y=100, font_size=size)
    print(f"font_size={size:.1f}, width={result['width']:.1f}px")
    print(f"path d (first 200 chars): {result['d'][:200]}...")
