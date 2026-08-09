"""
Text -> a list of real single-stroke pen paths, using Hershey font
data — the actual industry-standard single-stroke vector font format
(created 1967 at the US National Bureau of Standards, public domain,
used for pen plotters/engravers ever since specifically BECAUSE it's
genuine stroke-skeleton data, not filled letter shapes).

WHY THIS REPLACES THE EARLIER fontTools-BASED APPROACH:
Regular fonts (TTF/OTF, including "handwriting-styled" ones) ALWAYS
store letters as FILLED 2D outline shapes internally — that's the only
thing the format can represent, even when the letter is drawn to
LOOK like a thin pen stroke. Tracing the outline of a filled shape
with a thick stroke produces a blob; revealing it with a clip-wipe
produces a flat horizontal slide with no relationship to the letter's
actual shape — the hand can't trace something that doesn't exist in
the data. Hershey fonts solve this at the data level: each glyph is
literally a list of pen strokes (a real skeleton the letter is drawn
FROM), which is exactly the "hand actually writes each letter, moving
through the letter's real shape, lifting the pen between strokes"
effect this pipeline needs. A Hershey "A", for example, is genuinely
THREE separate pen strokes (two diagonals, one crossbar) — matching
how a hand actually draws it.

Output shape matches svg_to_path.py's icon output (a list of subpath
`d` strings) DELIBERATELY — write-mode and draw-mode now use the exact
same proven stroke-reveal renderer in scene_template.html, not two
separate techniques. One less thing to keep in sync, one fewer place
for a bug to hide.

Font data source: techninja/hersheytextjs (MIT wrapper around public-
domain Hershey vector font data). Vendored locally, no runtime fetch.
"""
import json
import os
import re
from functools import lru_cache

HERSHEY_JSON_PATH = os.path.join(os.path.dirname(__file__), "vendor", "fonts", "hershey-cursive.json")

# This font's own coordinate system spans roughly this Y range
# (measured directly from the vendored data, not assumed) — used as
# the reference "em size" for scaling to a target pixel font_size.
_NATIVE_Y_SPAN = 37.0


@lru_cache(maxsize=1)
def _load_font():
    with open(HERSHEY_JSON_PATH) as f:
        data = json.load(f)
    return data["chars"]  # list of 95, index = ord(char) - 33


def _char_data(ch: str, chars: list):
    idx = ord(ch) - 33
    if 0 <= idx < len(chars):
        return chars[idx]
    return None


def _split_subpaths(d: str) -> list:
    """A single Hershey character's `d` string can itself contain
    multiple disconnected strokes (multiple M commands) — e.g. 'A' is
    genuinely 3 separate pen strokes. Splits back into individual
    'M...' subpath strings so each becomes its own stroke-reveal
    <path> element (same reasoning as the icon multi-subpath fix)."""
    parts = re.split(r"(?=M)", d.strip())
    return [p.strip() for p in parts if p.strip()]


def text_advance_width(text: str, font_size: float) -> float:
    """Total width of `text` at `font_size` — needed BEFORE stroke
    generation to decide how big to render (fit-to-region)."""
    chars = _load_font()
    scale = font_size / _NATIVE_Y_SPAN
    total = 0.0
    for ch in text:
        if ch == " ":
            total += font_size * 0.5
            continue
        c = _char_data(ch, chars)
        advance = c["o"] if c else _NATIVE_Y_SPAN * 0.5
        total += advance * scale + font_size * 0.012
    return total


def fit_font_size(text: str, max_width: float, max_height: float,
                   min_size: float = 24, max_size: float = 160) -> float:
    probe_size = 100.0
    probe_width = text_advance_width(text, probe_size)
    if probe_width <= 0:
        return min_size
    size_for_width = max_width / probe_width * probe_size
    size_for_height = max_height * 0.75
    size = min(size_for_width, size_for_height)
    return max(min_size, min(max_size, size))


def text_to_strokes(text: str, x: float, y: float, font_size: float) -> dict:
    """Returns {"subpaths": ["d1", "d2", ...], "width": total_px_width,
    "word_groups": [{"word": str, "subpath_start": int, "subpath_end": int}, ...]}.

    (x, y) is the TOP-LEFT of the text's bounding box in world-space.
    subpaths are already translated/scaled into final world-space
    coordinates — no wrapper transform needed on the <path> elements
    (matches how text_to_path.py's old glyph paths worked, keeps
    getTotalLength()/getPointAtLength() in the same coordinate space
    as everything else).

    word_groups maps each word to a RANGE of subpath indices — this is
    what render_pipeline.py zips against real Chatterbox per-word
    timestamps, so multiple strokes within one word share that word's
    real speech duration proportionally (by individual stroke length),
    rather than every subpath getting equal time regardless of the
    word it belongs to.
    """
    chars = _load_font()
    scale = font_size / _NATIVE_Y_SPAN

    all_subpaths = []
    word_groups = []
    cursor_x = x
    current_word = ""
    current_word_start_idx = 0

    def _flush_word():
        if current_word:
            word_groups.append({
                "word": current_word,
                "subpath_start": current_word_start_idx,
                "subpath_end": len(all_subpaths),
            })

    for ch in text:
        if ch == " ":
            _flush_word()
            current_word = ""
            cursor_x += font_size * 0.5
            current_word_start_idx = len(all_subpaths)
            continue

        if not current_word:
            current_word_start_idx = len(all_subpaths)
        current_word += ch

        c = _char_data(ch, chars)
        if c is None:
            cursor_x += font_size * 0.5
            continue

        for raw_sub in _split_subpaths(c["d"]):
            # Parse "M x,y L x,y x,y ..." command tokens, scale + translate
            # every coordinate pair directly (this font's Y axis already
            # points down, same as SVG — no flip needed, unlike TTF glyphs).
            def _transform_coords(match):
                px = float(match.group(1))
                py = float(match.group(2))
                new_x = cursor_x + px * scale
                new_y = y + (py - (-3)) * scale  # shift so the font's own min-y sits near the top of the box
                return f"{new_x:.2f},{new_y:.2f}"

            transformed = re.sub(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", _transform_coords, raw_sub)
            all_subpaths.append(transformed)

        # Small extra breathing room between letters (matches the
        # reference library's "charSpacingAdjust" mechanism) — pure
        # advance-width spacing reads slightly crowded for cursive
        # connecting strokes; a touch of extra gap keeps it readable
        # without breaking the connected-cursive look.
        cursor_x += c["o"] * scale + font_size * 0.012

    _flush_word()

    return {
        "subpaths": all_subpaths,
        "width": cursor_x - x,
        "word_groups": word_groups,
    }


if __name__ == "__main__":
    size = fit_font_size("the brain rewires itself", max_width=800, max_height=200)
    result = text_to_strokes("the brain rewires itself", x=100, y=100, font_size=size)
    print(f"font_size={size:.1f}, width={result['width']:.1f}px, {len(result['subpaths'])} subpaths")
    print(f"word_groups: {result['word_groups']}")
    print(f"first subpath: {result['subpaths'][0]}")
