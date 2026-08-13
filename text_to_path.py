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
_NATIVE_Y_SPAN = 32.0


_COORD_RE = re.compile(r"(-?\d+\.?\d*),(-?\d+\.?\d*)")


def _real_max_x(d: str) -> float:
    """The glyph's declared 'o' (advance width) is NOT reliable on its
    own — measured directly against this font's data, several letters'
    actual ink (e.g. 'r', 'm') extends well past their own 'o' value
    (overhang). Advancing the cursor by 'o' alone under-spaces those
    letters relative to ones whose ink stays inside their 'o' box,
    which is what produced the "some letters packed, some letters
    apart" bug — it wasn't random, it was per-glyph overhang. Real
    max-x of the stroke coordinates is the actual right edge that must
    clear before the next letter starts."""
    xs = [float(m.group(1)) for m in _COORD_RE.finditer(d)]
    return max(xs) if xs else 0.0


@lru_cache(maxsize=1)
def _load_font():
    with open(HERSHEY_JSON_PATH) as f:
        data = json.load(f)
    chars = data["chars"]
    for c in chars:
        # Precomputed once here (not per-occurrence at render time) —
        # advance = whichever is bigger, the declared width or the
        # real ink extent, so overhang letters get their true width.
        c["_advance"] = max(c["o"], _real_max_x(c["d"]))
    return chars  # list of 95, index = ord(char) - 33


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
            total += font_size * 0.22
            continue
        c = _char_data(ch, chars)
        advance = c["_advance"] if c else _NATIVE_Y_SPAN * 0.5
        total += advance * scale + font_size * 0.05
    return total


def _wrap_lines(text: str, font_size: float, max_width: float) -> list:
    """Greedy word-wrap using the SAME advance-width measurement the
    renderer actually uses (text_advance_width), so a line that fits
    here is guaranteed to actually fit on screen at this font_size."""
    words = text.split(" ")
    lines = []
    current = []
    for w in words:
        trial = " ".join(current + [w])
        if not current or text_advance_width(trial, font_size) <= max_width:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def fit_font_size_wrapped(text: str, max_width: float, max_height: float,
                           min_size: float = 20, max_size: float = 160) -> float:
    """Finds the LARGEST font size where the text — WRAPPED across
    as many lines as needed — fits inside max_width x max_height.
    This is what fit_font_size (single-line only) was missing: for a
    long sentence, that function had no choice but to shrink all the
    way to its hard floor and then overflow past max_width anyway,
    since nothing ever wrapped it to a second line. Multi-line means
    the font can stay much bigger for the same sentence."""
    line_height_ratio = 1.35
    size = max_size
    step = 3
    while size > min_size:
        lines = _wrap_lines(text, size, max_width)
        total_h = len(lines) * size * line_height_ratio
        if total_h <= max_height:
            return size
        size -= step
    return min_size


def text_to_strokes_wrapped(text: str, x: float, y: float, font_size: float, max_width: float) -> dict:
    """Same return shape as text_to_strokes (subpaths + word_groups),
    but lays the text out across multiple lines instead of forcing
    everything onto one — word_groups indices stay contiguous and in
    reading order across the line break, so render_pipeline.py's
    per-word Chatterbox timing sync still zips correctly regardless
    of how many lines the sentence wrapped into."""
    lines = _wrap_lines(text, font_size, max_width)
    line_height = font_size * 1.35
    all_subpaths = []
    word_groups = []
    cursor_y = y
    for line in lines:
        result = text_to_strokes(line, x=x, y=cursor_y, font_size=font_size)
        offset = len(all_subpaths)
        for wg in result["word_groups"]:
            word_groups.append({
                "word": wg["word"],
                "subpath_start": wg["subpath_start"] + offset,
                "subpath_end": wg["subpath_end"] + offset,
            })
        all_subpaths.extend(result["subpaths"])
        cursor_y += line_height
    return {"subpaths": all_subpaths, "word_groups": word_groups, "line_count": len(lines)}


def fit_font_size(text: str, max_width: float, max_height: float,
                   min_size: float = 14, max_size: float = 160) -> float:
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
            cursor_x += font_size * 0.22
            current_word_start_idx = len(all_subpaths)
            continue

        if not current_word:
            current_word_start_idx = len(all_subpaths)
        current_word += ch

        c = _char_data(ch, chars)
        if c is None:
            cursor_x += font_size * 0.22
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

        # Advance by the REAL ink extent (c["_advance"], see
        # _load_font), not the raw declared "o" — "o" alone let
        # overhang-heavy letters (r, m, ...) collide into the next
        # letter while other letters got comparatively too much gap.
        # Small fixed breathing room on top is now much smaller (0.05
        # instead of 0.2) since the overhang correction already closes
        # most of the previous crowding-vs-gap inconsistency; this is
        # just a touch of visual air, not a fix for spacing itself.
        cursor_x += c["_advance"] * scale + font_size * 0.05

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
