"""
Main orchestrator for the hand-draw pipeline.

Flow per render request:
  1. Validate beats against the closed concept vocabulary (beat_schema.py) — hard-fail on unknown concept_key.
  2. Single Chatterbox call for the WHOLE script, timed onto every beat (voice_engine.words_for_beats).
  3. Resolve every beat's concept_key to a concrete asset (asset_resolver.resolve), checking
     concept-library.json first, then live CLIP-ranked search, caching new resolutions back
     into the channel's concept-library.json so the same keyword never re-searches.
  4. Lay out resolved content on the world-space board (simple flow layout — see _layout_board).
  5. Build a camera timeline across beats (camera.Camera), driven by each beat's requested action.
  6. Build gesture placements per beat (gesture_engine).
  7. Emit one JSON "scene program" that scene_template.html + Playwright consumes to actually
     render frames. This module does NOT touch Playwright directly — kept separate so the
     scene program can be inspected/debugged as plain JSON before ever opening a browser.

This mirrors the existing pipelines' shape (get_tts_safe -> assign_timing -> build_video) —
same separation of "figure out what happens when" from "actually draw pixels".
"""
import os
import json
from typing import List

import voice_engine
import asset_resolver
import historical_asset_resolver

# Channels whose beats should resolve through historical_asset_resolver
# (Wikimedia Commons / LOC / NASA / Internet Archive, CLIP-matched
# against a free-text concept_key) instead of asset_resolver's generic
# icon+stock pipeline. Every existing channel is untouched — this set
# is checked in ONE place (resolve_beat_asset, below) and everything
# else about the pipeline (layout, camera, gestures, timing) is shared
# as-is, unchanged, regardless of which channel is rendering.
HISTORICAL_CHANNELS = {"history"}
import gesture_engine
import text_to_path
import svg_to_path
from camera import Camera, CameraMove, get_frame_dims, get_board_dims, get_default_view, region_for_bbox, _fit_aspect
from beat_schema import validate_batch, load_vocabulary, GESTURE_FOR_MODE

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# How much bigger a pinched-in icon gets, relative to its original
# drawn size. Tune this single number if "enlarge" should read as
# more/less dramatic — nothing else needs to change.
ICON_ENLARGE_SCALE = 1.25  # NO LONGER USED — the zoom_in/zoom_out icon-enlarge effect (and
                           # its pinch hand) was removed per direct feedback. Left here rather
                           # than deleted so nothing else that might still reference this
                           # constant breaks; safe to remove for real once confirmed unused.
ICON_STROKE_TARGET_PX = 6.0  # numerator for scale-compensated stroke width (stroke_width =
                              # this / icon_path_info["scale"]) so the FINAL on-screen width
                              # lands near this many pixels regardless of the icon's fit-scale.
ROW_CONTENT_H = 230  # compact, FIXED row height for multi-row icon_word "items" — NOT
                      # stretched to fill the (intentionally oversized, for bleed-prevention)
                      # allocated region. That stretch was the actual root cause of "rows too
                      # far apart": each row became much taller than its content needed, with
                      # icon+word centered in mostly-dead space.
ROW_GAP_FIXED = 16   # small fixed gap between rows, not a percentage of region height.
                      # BOTH of these must be module-level, not local to one function — they're
                      # used by _layout_board() AND build_scene_program(), two separate
                      # functions. Defining them as locals inside only one of those (which is
                      # what broke the render) makes them invisible to the other.
                              # CONFIRMED BUG: this was 45.0 — a factor-of-10 mistake (meant to
                              # bump moderately from the original 5.0, wrote 45 instead of ~6).
                              # Verified in an actual browser render: 45px on a normal icon-sized
                              # shape renders as a thick black blob, exactly matching the
                              # screenshots; 6px renders as a clean line-icon weight.

# Camera pacing — NOT a fixed move duration. Real duration is derived
# per-move from (a) how far the camera actually has to travel between
# the previous target and this one, and (b) how much real slack exists
# before this beat's own content needs to be on screen (beat["start"]
# minus the moment the camera is actually free to move). These are
# just the clamps: a hop should never feel instant, a full-board jump
# should never crawl.
CAMERA_MIN_DURATION = 0.22
CAMERA_MAX_DURATION = 0.85
CAMERA_TRAVEL_SPEED = 2600.0  # world-units/second


def _camera_move_duration(prev_center, target_center, available_gap):
    """prev_center/target_center: (x, y) world-space camera centers.
    available_gap: seconds of real slack before this beat's content
    needs to be visible (beat['start'] - camera_free_at), or None if
    unknown. Distance-based pacing means a short hop between adjacent
    slots is quick and a jump across the board takes noticeably
    longer — proportional, not a flat number regardless of context."""
    if prev_center is None:
        return CAMERA_MIN_DURATION
    dx = target_center[0] - prev_center[0]
    dy = target_center[1] - prev_center[1]
    dist = (dx * dx + dy * dy) ** 0.5
    duration = max(CAMERA_MIN_DURATION, min(CAMERA_MAX_DURATION, dist / CAMERA_TRAVEL_SPEED))
    if available_gap is not None and available_gap > 0:
        # There's real slack (previous sentence finished early relative
        # to this one's start) — never take longer than that slack, so
        # the move still finishes before this beat's content needs to
        # be framed.
        duration = min(duration, max(CAMERA_MIN_DURATION, available_gap))
    return duration


def _region_center(region: dict) -> tuple:
    return (region["x"] + region["w"] / 2, region["y"] + region["h"] / 2)


def _center_text_x(label: str, font_size: float, region: dict) -> float:
    """Horizontally centers `label` within `region`, using the label's
    ACTUAL rendered width at font_size (text_to_path.text_advance_width)
    rather than an estimated left-padding fraction. fit_font_size picks
    whichever of width/height is the binding constraint — for a short
    caption in a tall-but-narrow box, height is very often the binding
    one, which means the rendered text ends up NARROWER than the box's
    usable width. A fixed left-padding formula has no way to know that,
    so the word visually reads as flush-left with empty space on the
    right instead of centered. Measuring the real width and centering
    against it fixes that regardless of which constraint won."""
    width = text_to_path.text_advance_width(label, font_size)
    return region["x"] + (region["w"] - width) / 2


def _layout_icon_grid(region: dict, n: int) -> list:
    """Lays out N pure icons (no word labels — see the 'icons' field on
    icon_word beats, PSYCHOLOGY CHANNEL ONLY) into a left/right grid, 2
    per row, per direct spec: N=2 -> one row, left+right. N=3 -> row 1
    left+right, row 2 has the lone 3rd icon CENTERED (not left-aligned).
    N=4 -> two full rows of left+right (a 2x2 grid). Generalizes the
    same way for any N (a trailing lone icon on the last row is always
    centered), though the visual planner is only ever expected to ask
    for 1-4. Verified against all 4 cases (1/2/3/4) before integration.

    Returns a list of N {x, y, w, h} icon regions, in the SAME order
    as the icons were given, top-to-bottom within a row pair, left
    before right.
    """
    if n < 1:
        raise ValueError(f"_layout_icon_grid requires at least 1 icon, got {n}")

    num_rows = (n + 1) // 2  # ceiling division without needing math.ceil
    row_h = region["h"] / num_rows
    col_w = region["w"] / 2

    boxes = []
    for i in range(n):
        row = i // 2
        col = i % 2
        is_last_row = (row == num_rows - 1)
        icons_in_this_row = n - row * 2  # only meaningful when is_last_row is True
        row_y = region["y"] + row * row_h

        if is_last_row and icons_in_this_row == 1:
            box_w = col_w
            box_x = region["x"] + (region["w"] - box_w) / 2
        else:
            box_w = col_w
            box_x = region["x"] + col * col_w

        boxes.append({"x": box_x, "y": row_y, "w": box_w, "h": row_h})
    return boxes


# How many trailing WORDS of a beat's real narration are held back as
# pure buffer — no visual is still drawing once the narrator reaches
# these words, so every beat ends on a clean, fully-drawn frame before
# the camera moves on. Word-based (not a flat second count) because a
# fixed time buffer means something different for a fast vs slow
# sentence; a word count means the same thing regardless of pace.
ICON_WORD_BUFFER_WORDS = 3
# Floor so a visual never becomes imperceptibly fast even if its real
# assigned words were spoken very quickly — a soft floor, not a hard
# guarantee (see _word_synced_slots' docstring for the one edge case
# where flooring can cause a small overlap with the next slot; this
# only matters for word durations far faster than any real narration
# pace this pipeline actually uses).
ICON_WORD_MIN_SLOT_DURATION = 0.35


def _word_synced_slots(beat_words: list, n_slots: int,
                        buffer_words: int = ICON_WORD_BUFFER_WORDS,
                        min_slot_duration: float = ICON_WORD_MIN_SLOT_DURATION):
    """Splits a beat's REAL per-word Chatterbox timestamps across
    n_slots visual elements (icon/word pairs — 2 for a single icon_word
    beat, 2*n_rows for a multi-row items beat), so each visual's
    on-screen draw duration matches how long the narrator actually took
    to say ITS share of the sentence, instead of a flat capped duration
    with no relationship to the sentence's real length or pace. This is
    the same real-timing approach 'write' mode already uses for letter
    strokes, applied here to icon/word visuals instead.

    Splits by WORD COUNT (not estimated syllable weight or character
    count) — a 5-word sentence split across 2 visuals gives one visual
    ~2-3 words and the other ~2-3 words, by direct request.

    The last `buffer_words` words are excluded entirely from the split
    — every visual finishes drawing before the narrator even reaches
    those trailing words, leaving a clean beat of "already drawn,
    holding" before the camera moves to the next beat.

    Returns a list of (start_t, end_t) absolute-seconds tuples (same
    timebase as beat['start']/beat['end']), one per slot in order, or
    None if there aren't enough real per-word timestamps to reliably do
    this (too few words for the number of visuals this beat needs) —
    callers fall back to the old fixed-duration scheme in that case,
    logged loudly, same "never silently do the wrong thing" rule used
    everywhere else in this pipeline.
    """
    if not beat_words:
        return None

    usable = beat_words[:-buffer_words] if buffer_words > 0 and len(beat_words) > buffer_words else beat_words
    if len(usable) < n_slots:
        return None

    n = len(usable)
    base = n // n_slots
    remainder = n % n_slots
    slots = []
    idx = 0
    for i in range(n_slots):
        count = base + (1 if i < remainder else 0)  # remainder words go to the FIRST groups
        group = usable[idx: idx + count]
        idx += count
        start_t, end_t = group[0]["start"], group[-1]["end"]
        if end_t - start_t < min_slot_duration:
            end_t = start_t + min_slot_duration
        slots.append((start_t, end_t))
    return slots


def _illustration_reveal(channel: str, asset_type: str, region: dict):
    """Decides how an illustration/photo enters (and exits) the frame.

    CHANGED per direct feedback, applied to the WHOLE pipeline (every
    channel, every asset_type) — previously only a real historical
    photo (asset_type="photo" on a HISTORICAL_CHANNELS channel) got a
    hand-less pop reveal; every other illustration/stock-photo fallback
    still got a "drag" hand mask-wipe. That mask_wipe/drag pathway is
    now unused — its code in scene_template.html's animateMaskWipe is
    left in place rather than deleted, in case this ever needs
    reverting, but nothing selects it anymore. Every illustration now
    pops in AND back out, no hand, regardless of channel or asset_type.

    Signature and return shape (mask_wipe_hand_dict_or_None,
    reveal_style_string) are UNCHANGED on purpose — every call site
    (single-beat and multi-row) keeps working with zero modification.
    """
    return None, "pop"


def _channel_library_path(channel: str) -> str:
    return os.path.join(REPO_ROOT, "channels", channel, "concept-library.json")


SHARED_LIBRARY_PATH = os.path.join(REPO_ROOT, "channels", "_shared", "concept-library.json")


def _load_library(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_library(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def resolve_beat_asset(beat: dict, channel: str, illustration_cache_dir: str, asset_type: str = "icon") -> dict:
    """Checks shared + channel library first (curated, no network),
    falls back to live asset_resolver.resolve(), and writes new
    resolutions back to the channel's own library so this is a
    one-time cost per concept, not per render.

    asset_type: "icon" (default) uses the normal icon+stock pipeline
    for EVERY channel, including history — a beat about an abstract
    idea ("mystery", "murder weapon", "money") still just draws a
    generic icon, same as any other channel. asset_type="photo" is
    what actually routes through historical_asset_resolver, and only
    does anything different when channel is also in HISTORICAL_CHANNELS
    — a beat can ask for a real photo, but only the history channel
    actually has that resolver wired up. This is a PER-BEAT choice
    (set by the Visual Planner), not a per-channel one — a single
    history script mixes real photos of the actual subject with
    generic icons for abstract narration beats, same as psychology
    already mixes icons with plain "write" beats."""
    concept_key = beat.get("concept_key")
    if not concept_key:
        return None

    shared_lib = _load_library(SHARED_LIBRARY_PATH)
    channel_lib_path = _channel_library_path(channel)
    channel_lib = _load_library(channel_lib_path)

    # Curated cache is keyed on concept_key alone — a photo query and an
    # icon concept are extremely unlikely to collide in practice (photo
    # queries are full descriptive phrases, icon concept_keys are 1-3
    # generic words), so no extra namespacing needed here.
    if concept_key in shared_lib:
        return shared_lib[concept_key]
    if concept_key in channel_lib:
        return channel_lib[concept_key]

    # No curated entry — live resolve, then persist. Only a "photo"
    # beat on a HISTORICAL_CHANNELS channel actually uses the
    # Commons/LOC/NASA/IA resolver; everything else (including
    # "icon" beats on the history channel itself) uses the same
    # generic icon+stock pipeline every other channel already uses.
    if asset_type == "photo" and channel in HISTORICAL_CHANNELS:
        resolved = historical_asset_resolver.resolve(concept_key, cache_dir=illustration_cache_dir)
        source_desc = "Wikimedia Commons/LOC/NASA/Internet Archive"
    else:
        resolved = asset_resolver.resolve(concept_key, cache_dir=illustration_cache_dir)
        source_desc = "icons or any stock source"
    if resolved is None:
        raise RuntimeError(
            f"No asset found for concept_key '{concept_key}' (beat_id={beat.get('beat_id')}) "
            f"across {source_desc} — cannot render this beat. Add a curated entry "
            f"to channels/{channel}/concept-library.json or channels/_shared/concept-library.json "
            f"if this concept should always resolve to something specific."
        )

    entry = {
        "channel_tags": [channel],
        "mode": beat.get("mode", "draw"),
        "asset_source": resolved.source,
        "asset_ref": {"path": resolved.data} if resolved.kind == "svg_path" else {"cached_path": resolved.data},
        "draw_style": resolved.draw_style,
        "auto_resolved": True,
    }
    channel_lib[concept_key] = entry
    _save_library(channel_lib_path, channel_lib)
    return entry


def resolve_icon_stroke_path(concept_key: str, channel: str, illustration_cache_dir: str,
                              icon_box: dict, beat_id, max_retry_candidates: int = 4):
    """Resolves `concept_key` to real, USABLE stroke-reveal path data —
    not just an asset reference. Added after a real production crash:
    the old flow trusted whichever single icon resolve_beat_asset/
    asset_resolver handed back and either used it or raised; there was
    no way to recover from a broken SVG file (fails to parse, or has
    no usable <path> data — see svg_to_path.py's documented LIMITATION
    on primitive-only icons) when a perfectly good sibling icon existed
    one CLIP-rank down, and no way to recover from a concept that
    simply has NO vendored icon at all (e.g. "faucet" — legitimately
    absent from all 4 libraries' filenames) without crashing the whole
    render.

    Tries, in order:
      1. The normal cached/resolved entry from resolve_beat_asset — the
         fast path, unchanged for the common case where it's already a
         working icon. Zero extra cost when nothing is broken.
      2. If that entry IS a real icon (draw_style=stroke_reveal) but
         its SVG FAILS to convert, retries against the next-best
         vendored candidates (asset_resolver.search_vendor_icon_candidates),
         up to max_retry_candidates total attempts — and if one works,
         SELF-HEALS the channel's concept-library.json cache so future
         renders skip straight to the working icon instead of
         re-discovering this every time.

    Returns (icon_path_info, asset_entry) on success, or (None,
    asset_entry) if no vendored icon can be made to work at all — this
    is NOT necessarily a hard failure; it just means "no real icon
    exists for this concept", and it's the CALLER's job to decide what
    that means for its context (fall back to a stock image, or raise).
    `asset_entry` is always returned (even on failure) so the caller
    can inspect it (e.g. its draw_style/asset_source) without a second
    resolve_beat_asset call.
    """
    asset_entry = resolve_beat_asset(
        {"concept_key": concept_key, "beat_id": beat_id}, channel, illustration_cache_dir, asset_type="icon",
    )
    if asset_entry.get("draw_style") != "stroke_reveal":
        return None, asset_entry  # no vendored icon at all for this concept — caller decides fallback

    svg_path = asset_entry["asset_ref"].get("path")
    icon_path_info = svg_to_path.icon_to_path_d(svg_path, icon_box, padding_ratio=0.08) if svg_path else None
    if icon_path_info is not None:
        return icon_path_info, asset_entry  # fast path: the cached/first icon just works

    print(f"[render_pipeline] beat_id={beat_id}: icon at {svg_path!r} for concept_key={concept_key!r} "
          f"failed to convert (broken SVG or no usable <path> data) — trying the next-best vendored "
          f"candidate(s)")

    candidates = asset_resolver.search_vendor_icon_candidates(concept_key)
    tried = {svg_path}
    for candidate_path in candidates:
        if candidate_path in tried:
            continue
        tried.add(candidate_path)
        candidate_info = svg_to_path.icon_to_path_d(candidate_path, icon_box, padding_ratio=0.08)
        if candidate_info is not None:
            print(f"[render_pipeline] beat_id={beat_id}: concept_key={concept_key!r} recovered using "
                  f"fallback icon {candidate_path!r} (attempt {len(tried)})")
            fixed_entry = dict(asset_entry)
            fixed_entry["asset_ref"] = {"path": candidate_path}
            channel_lib_path = _channel_library_path(channel)
            channel_lib = _load_library(channel_lib_path)
            channel_lib[concept_key] = fixed_entry
            _save_library(channel_lib_path, channel_lib)
            return candidate_info, fixed_entry
        if len(tried) >= max_retry_candidates:
            break

    print(f"[render_pipeline] beat_id={beat_id}: concept_key={concept_key!r} — ALL vendored candidates "
          f"tried ({len(tried)}) failed to convert. No usable icon for this concept.")
    return None, asset_entry


# ══════════════════════════════════════════════════════════════════
# World-space layout
# ══════════════════════════════════════════════════════════════════
def _layout_board(beats: List[dict], orientation: str = "landscape") -> dict:
    """Simple left-to-right, wrapping flow layout: each draw/write beat
    gets a slot on the world-space board in script order. This is
    intentionally the simplest thing that works — deterministic,
    debuggable, no packing algorithm. Replace with something smarter
    (e.g. grouping related concepts spatially) once real scripts show
    where a flow layout looks awkward, not before.

    Column count is derived from the board's own aspect ratio, not a
    fixed constant — a portrait board is much narrower, so packing 6
    columns across it would make every slot uncomfortably thin. Fewer,
    taller columns for portrait; more, wider ones for landscape.
    """
    board = get_board_dims(orientation)
    frame = get_frame_dims(orientation)
    target_aspect = frame["width"] / frame["height"]
    CONTENT_W, CONTENT_H = 420, 320  # unchanged content box size for a single-item slot
    MAX_ITEM_ROWS = 3  # icon_word "items" pairs: 2/4/6 entries -> 1/2/3 stacked rows
    MAX_ROWS_WITH_NUMBER = MAX_ITEM_ROWS + 1  # +1 for the optional standalone number row
                                               # (see beat_schema.py's "number" field) —
                                               # used ONLY to size the worst-case tall slot
                                               # generously enough; a beat without "number"
                                               # still gets exactly MAX_ITEM_ROWS worth.

    def _row_count_for_beat(b: dict) -> int:
        if b.get("mode") == "icon_word" and b.get("items"):
            content_rows = max(1, min(MAX_ITEM_ROWS, len(b["items"]) // 2))
            return content_rows + (1 if b.get("number") is not None else 0)
        if b.get("mode") == "icon_word" and b.get("icons"):
            # Pure icon-only grid (see _layout_icon_grid) — same ceiling
            # division as that function's own row math, so board slot
            # sizing actually matches what gets drawn. Capped at 4 icons
            # elsewhere, so this never exceeds 2 rows in practice; the
            # MAX_ITEM_ROWS cap is kept anyway for consistency/safety.
            content_rows = max(1, min(MAX_ITEM_ROWS, (len(b["icons"]) + 1) // 2))
            return content_rows + (1 if b.get("number") is not None else 0)
        return 1

    # SPACING BUG, FULLY FIXED THIS TIME — last pass only corrected
    # ROW spacing (vertical) and missed that COLUMN spacing (horizontal)
    # has the exact same problem, and is actually the more common case
    # to hit since most beat-to-beat moves advance to the next column,
    # not the next row. Confirmed by direct measurement: the camera's
    # real fitted footprint around one slot is 540px wide in portrait
    # and 782px wide in landscape — but columns were still spaced a
    # flat 500px apart in both. Both SLOT_W and SLOT_H now come from
    # what the camera will ACTUALLY frame (via the same _fit_aspect
    # math the camera itself uses), plus a safety margin, so no
    # adjacent slot in either direction is ever inside the same shot.
    fitted = _fit_aspect(region_for_bbox({"x": 0, "y": 0, "w": CONTENT_W, "h": CONTENT_H}, padding=60), target_aspect)
    SAFETY_MARGIN = 550  # increased again (was 350) per direct feedback that bleed was
                          # still visible — trusting the report over my own math this time
    SLOT_W = max(500, fitted["w"] + SAFETY_MARGIN)
    # SLOT_H sized for the TALLEST beat that can land in a row — a
    # multi-row icon_word beat (up to MAX_ROWS_WITH_NUMBER stacked
    # pairs, including the optional number row) needs up to 4x the
    # standard content height. Without this, a tall beat's camera
    # footprint would extend past a row spacing sized only for the
    # ordinary single-row case, bleeding into the row below — same
    # class of bug as the column one above, triggered by height instead
    # of width.
    _tall_content_h = ROW_CONTENT_H * MAX_ROWS_WITH_NUMBER + ROW_GAP_FIXED * (MAX_ROWS_WITH_NUMBER - 1)
    _tall_fitted = _fit_aspect(region_for_bbox({"x": 0, "y": 0, "w": CONTENT_W, "h": _tall_content_h}, padding=60), target_aspect)
    SLOT_H = max(400, _tall_fitted["h"] + SAFETY_MARGIN)

    # SECOND, SEPARATE SPACING BUG FOUND AND FIXED — the one above
    # (SLOT_W/SLOT_H sized from the camera's single-slot footprint)
    # does NOT cover this case: a beat that's part of a GROUP (shared
    # slot, sub-divided into up to 4 columns) near the EDGE of its
    # slot gets its own camera fit-growth centered on ITS OWN
    # off-center position, not the slot's center — this pushes its
    # fitted view past the slot boundary independent of SLOT_W. Proven
    # numerically: growing SLOT_W to even 3000+ converges to a fixed
    # ~20-unit overlap between the last item of one slot and the first
    # item of the next (4-item groups, the worst case) — it does NOT
    # go away no matter how large SLOT_W gets, because the overhang
    # comes from padding applied to a narrow sub-cell, not from slot
    # width itself. What DOES close it (verified): literal extra space
    # inserted directly BETWEEN slots, on top of SLOT_W.
    INTER_SLOT_GAP = 320  # increased again (was 180) alongside the safety margin bump above

    SLOT_PITCH_W = SLOT_W + INTER_SLOT_GAP
    SLOT_PITCH_H = SLOT_H + INTER_SLOT_GAP
    COLS = max(2, board["width"] // SLOT_PITCH_W)
    MARGIN = 100
    MAX_ITEMS_PER_SLOT = 4

    relevant = [b for b in beats if b["mode"] in ("draw", "write", "icon_word")]

    # Pass 1: chunk CONSECUTIVE beats sharing the same group_id into one
    # chunk. A beat with no group_id (or a group_id of None — the
    # default for every existing script) is always its own chunk of
    # size 1 — this is what keeps old scripts laying out EXACTLY as
    # before, unchanged. Only beats an upstream planner explicitly
    # tags with a shared group_id (e.g. "food [icon] + good [text]"
    # sharing one visual space) get packed together.
    chunks = []
    i = 0
    while i < len(relevant):
        gid = relevant[i].get("group_id")
        if gid is None:
            chunks.append([relevant[i]])
            i += 1
            continue
        j = i
        chunk = []
        while j < len(relevant) and relevant[j].get("group_id") == gid:
            chunk.append(relevant[j])
            j += 1
        chunks.append(chunk)
        i = j

    layout = {}
    for slot_i, chunk in enumerate(chunks):
        col = slot_i % COLS
        row = slot_i // COLS
        base_x = MARGIN + col * SLOT_PITCH_W
        base_y = MARGIN + row * SLOT_PITCH_H
        # Content height scales with the tallest beat in this chunk's
        # own row requirement (multi-row icon_word items) — a chunk is
        # almost always size 1 when items are used (grouping AND
        # multi-row items on the same beat would be unusual), but this
        # stays correct either way.
        chunk_rows = max(_row_count_for_beat(b) for b in chunk)
        full_w = CONTENT_W
        full_h = ROW_CONTENT_H * chunk_rows + ROW_GAP_FIXED * (chunk_rows - 1) if chunk_rows > 1 else CONTENT_H

        items = chunk[:MAX_ITEMS_PER_SLOT]
        n = len(items)
        # n==1 (the old, default case) gets the exact same rect as
        # before — no gap subtracted, so single-item slots are
        # unaffected byte-for-byte.
        gap = 20 if n > 1 else 0
        cell_w = full_w / n
        for k, b in enumerate(items):
            layout[b["beat_id"]] = {
                "x": base_x + k * cell_w, "y": base_y,
                "w": cell_w - gap, "h": full_h,
            }
        # Safety net, not an expected path: a group bigger than
        # MAX_ITEMS_PER_SLOT (planner should never emit this) reuses
        # the last cell instead of crashing render.
        for b in chunk[MAX_ITEMS_PER_SLOT:]:
            layout[b["beat_id"]] = layout[items[-1]["beat_id"]]
    return layout


# ══════════════════════════════════════════════════════════════════
# Listicle number-beat enforcement
# ══════════════════════════════════════════════════════════════════
import re

# Confirmed by direct inspection of real builder-node1 output: Gemini/the
# builder can silently drop 'number' entirely on a beat whose own sentence
# text plainly starts "One, ...", "Two, ...", etc. — 'items' comes through
# fine (icon+word pair, already correct), but with no 'number' key the
# render never draws the digit row at all, so the video shows only 2
# visuals instead of 3. A prompt fix and an items-trim can't catch a
# MISSING key, so this reads the ground truth straight from the beat's own
# text instead — the one field HARD RULE 1 in the planner prompt
# guarantees is never edited, so it's more reliable than trusting any
# upstream node remembered to set 'number' correctly.
_LISTICLE_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_LISTICLE_NUMBER_RE = re.compile(
    r"^\s*(" + "|".join(_LISTICLE_NUMBER_WORDS.keys()) + r")\s*,", re.IGNORECASE
)


def _detect_listicle_number(text: str):
    """Returns the integer a sentence's own listicle count word refers to
    (1 for a sentence starting 'One,', 2 for 'Two,', etc.) or None if the
    sentence doesn't start that way. Only ever looks at 'text' — never at
    'number', 'items', or anything else a planner/builder may have set."""
    if not text:
        return None
    m = _LISTICLE_NUMBER_RE.match(text)
    return _LISTICLE_NUMBER_WORDS[m.group(1).lower()] if m else None


def _normalize_listicle_number_beats(beats: List[dict]) -> None:
    """HARDCODE, not a suggestion, per direct feedback: a numbered listicle
    beat (a sentence whose own text starts "One, ...", "Two, ...", etc.)
    must always render with the number in its own row — already always
    centered full-width regardless of "layout" — plus content underneath
    it. Never missing.

    On the PSYCHOLOGY channel specifically, that content is now ALWAYS
    icon-only (see the channel-check below) — a prompt asking Gemini for
    "icons" instead of icon+word "items" is a request, not a guarantee,
    and this converts whatever shape the beat actually arrived in
    (icon+word "items", or a single concept_key/label) into pure
    "icons", dropping the word label(s) entirely, so a real mixture
    (regular sentences keep icon+word; every COUNTED item is icon-only)
    is actually guaranteed rather than hoped for.

    Three independent problems get fixed here, all confirmed against
    real pipeline output rather than assumed:
      1. 'number' can be present but too generous (an upstream planner
         gave it 4 or 6 items instead of 2) — trimmed to the first pair
         (non-psychology channels only — see above for psychology).
      2. 'number' can be MISSING ENTIRELY even though the sentence's own
         text is plainly a counted listicle item (confirmed: builder-node1
         output showed 'items' correct but no 'number' key at all on
         "One, ..."/"Two, ..." beats) — detected from the beat's own text
         and set here, regardless of whether Gemini or the builder node
         forgot it upstream.
      3. A numbered beat can arrive as icon+word 'items' (or a single
         concept_key/label) instead of icon-only 'icons' on the
         psychology channel — converted here, confirmed against a real
         render where every listicle item still showed icon+word.

    NOTE / caveat: this assumes any sentence starting "One,"/"Two,"/etc.
    followed by a comma is a listicle count word, which matches this
    channel's real scripts (confirmed) but would misfire on a sentence
    that coincidentally opens the same way for an unrelated reason (e.g.
    "One, however, disagreed...") — worth knowing if that phrasing ever
    shows up in a script.

    Mutates `beats` in place. Called once, before validate_beat, so
    validation only ever sees an already-correct beat — no error, no
    rejected batch, no manual re-run. Every beat whose text does NOT
    start with a listicle count word, and which had no 'number' set
    either, is completely untouched.
    """
    for beat in beats:
        if beat.get("mode") != "icon_word":
            continue

        detected_number = _detect_listicle_number(beat.get("text", ""))
        beat_number = beat.get("number")

        if beat_number is None and detected_number is None:
            continue  # not a listicle beat at all

        if beat.get("icons"):
            # NEW: pure icon-only grid beats (see _layout_icon_grid) are
            # already structurally complete on their own — no 'items'
            # pair needed, nothing to convert or trim here. Still apply
            # the same text-detected number override/fill logic below
            # (a numbered icon-grid beat deserves the same reliability
            # guarantee as a numbered items beat), just skip the
            # items-specific trimming/conversion that follows.
            if detected_number is not None and beat_number != detected_number:
                if beat_number is not None:
                    print(f"[render_pipeline] beat_id={beat.get('beat_id')}: sentence text says "
                          f"'{detected_number}' but beat had number={beat_number!r} — overriding to "
                          f"match the sentence's own text.")
                else:
                    print(f"[render_pipeline] beat_id={beat.get('beat_id')}: sentence text starts "
                          f"a listicle count ('{detected_number}') but 'number' was missing entirely "
                          f"— setting it from the sentence text.")
                beat["number"] = detected_number
            continue

        if detected_number is not None and beat_number != detected_number:
            if beat_number is not None:
                print(f"[render_pipeline] beat_id={beat.get('beat_id')}: sentence text says "
                      f"'{detected_number}' but beat had number={beat_number!r} — overriding to "
                      f"match the sentence's own text.")
            else:
                print(f"[render_pipeline] beat_id={beat.get('beat_id')}: sentence text starts "
                      f"a listicle count ('{detected_number}') but 'number' was missing entirely "
                      f"— setting it from the sentence text.")
            beat["number"] = detected_number
            beat_number = detected_number
        # else: planner explicitly set 'number' on a beat whose text
        # doesn't start a spelled-out count — left alone rather than
        # second-guessed, since that may be an intentional continuation.

        # NEW: force the icon-only listicle rule for real, per direct
        # feedback — a prompt asking Gemini for "icons" instead of
        # "items" is a request, not a guarantee, and this beat already
        # got PAST the "beat.get('icons')" early-continue above, which
        # means it does NOT have 'icons' yet. For the psychology
        # channel, every numbered beat's end-state must be icon-only —
        # so whatever shape it actually arrived in (icon+word 'items',
        # or a single concept_key/label) gets converted here, dropping
        # the word label(s) entirely, rather than rendering as icon+word
        # just because that's what the planner happened to output.
        if beat.get("channel") == "psychology":
            items = beat.get("items")
            concept_key, label = beat.get("concept_key"), beat.get("label")
            if items:
                converted = [pair["concept_key"] for pair in items
                             if pair.get("type") == "icon" and pair.get("concept_key")]
                if converted:
                    beat["icons"] = converted[:4]
                    beat.pop("items", None)
                    beat.pop("layout", None)  # was for icon+word pairing; meaningless for a pure icon grid
                    print(f"[render_pipeline] beat_id={beat.get('beat_id')}: numbered listicle beat "
                          f"arrived as icon+word 'items' — converted to icon-only 'icons' "
                          f"({len(beat['icons'])} icon(s)), dropping the word label(s), per this "
                          f"channel's icon-only listicle rule.")
                else:
                    print(f"[render_pipeline] beat_id={beat.get('beat_id')}: numbered listicle beat had "
                          f"'items' with no usable icon concept_key to convert — leaving 'number' set "
                          f"with nothing to draw.")
            elif concept_key:
                beat["icons"] = [concept_key]
                beat.pop("concept_key", None)
                beat.pop("label", None)
                beat.pop("layout", None)
                print(f"[render_pipeline] beat_id={beat.get('beat_id')}: numbered listicle beat used a "
                      f"single concept_key/label instead of icons — converted to a one-icon 'icons' "
                      f"list, dropping the word label, per this channel's icon-only listicle rule.")
            else:
                print(f"[render_pipeline] beat_id={beat.get('beat_id')}: looks like a numbered listicle "
                      f"item but has neither 'items', 'concept_key', nor 'icons' to draw — leaving "
                      f"'number' set with nothing to draw.")
            continue

        items = beat.get("items")
        if not items:
            concept_key, label = beat.get("concept_key"), beat.get("label")
            if concept_key and label:
                beat["items"] = [
                    {"type": "icon", "concept_key": concept_key},
                    {"type": "word", "label": label},
                ]
                beat.pop("concept_key", None)
                beat.pop("label", None)
                print(f"[render_pipeline] beat_id={beat.get('beat_id')}: numbered listicle beat "
                      f"used a single concept_key/label instead of items — converted to a "
                      f"one-pair items list so its number can render.")
            else:
                print(f"[render_pipeline] beat_id={beat.get('beat_id')}: beat_id={beat.get('beat_id')} "
                      f"looks like a numbered listicle item but has neither 'items' nor a "
                      f"concept_key/label to draw — leaving 'number' set with no icon/word pair.")
            continue

        if len(items) > 2:
            beat["items"] = items[:2]
            print(f"[render_pipeline] beat_id={beat.get('beat_id')}: numbered listicle beat had "
                  f"{len(items)} items — trimmed to the first icon,word pair (2) so it renders as "
                  f"exactly 3 visuals: number, icon, word.")


# ══════════════════════════════════════════════════════════════════
# Word-synced subtitles
# ══════════════════════════════════════════════════════════════════
# Per direct request: burned-in captions that pop in one word at a
# time, exactly on that word's own real Chatterbox timestamp, in
# screen-space (see camera.py's own note: "Fixed-layer UI (captions,
# logo, progress bar) does NOT go through this module... deliberately
# untouched by camera moves" — this was already the intended design,
# just never wired up). Built straight from `timing["words"]`, the
# same real per-word start/end times already used elsewhere in this
# file for stroke-reveal pacing — never estimated or re-derived.
#
# This is a caption of the RAW SPOKEN NARRATION, independent of
# whatever's drawn on the board at that moment — it doesn't read or
# depend on beats, concept_keys, or labels at all, so it can't be
# thrown off by anything upstream in the visual-planning side of the
# pipeline.
CAPTION_GROUP_SIZE = 1  # per direct feedback: ONE word visible on screen at a
                         # time, not several words together — this is the one
                         # number to change (e.g. back to 3-4) if that's ever
                         # wanted again; the grouping code below doesn't change.

def _build_captions(words: list, group_size: int = CAPTION_GROUP_SIZE) -> list:
    """Chunks Chatterbox's flat word list into caption groups of
    `group_size` consecutive words (currently 1 — see above). Each
    group keeps every word's own real start/end time (not just the
    group's overall start/end) so scene_template.html can pop each
    word in individually, exactly when it's actually spoken. A known
    simplification: grouping is purely by word count, so with a larger
    group_size a group could straddle a sentence boundary — moot at
    group_size=1, but worth knowing if this is ever turned back up.
    """
    groups = []
    for i in range(0, len(words), group_size):
        chunk = words[i:i + group_size]
        if not chunk:
            continue
        groups.append({
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "words": [{"text": w["word"], "start": w["start"], "end": w["end"]} for w in chunk],
        })
    return groups


# ══════════════════════════════════════════════════════════════════
# Scene program assembly
# ══════════════════════════════════════════════════════════════════
def build_scene_program(script_text: str, beats: List[dict], channel: str,
                         voice: str = None, illustration_cache_dir: str = "/tmp/illustration_cache",
                         orientation: str = "landscape") -> dict:
    vocab = load_vocabulary(SHARED_LIBRARY_PATH, _channel_library_path(channel))
    _normalize_listicle_number_beats(beats)
    # Beats can introduce NEW concept_keys not yet in any library — that's fine, they'll be
    # resolved and appended below. validate_batch's vocabulary check is for CATCHING TYPOS
    # against keys that already have curated intent, not for blocking anything new. So we
    # only hard-validate structural correctness here, not vocabulary membership yet.
    for i, beat in enumerate(beats):
        from beat_schema import validate_beat
        validate_beat(beat, i)

    timing = voice_engine.words_for_beats(script_text, beats, voice=voice)
    timed_beats = timing["beats"]

    frame = get_frame_dims(orientation)
    board = get_board_dims(orientation)
    board_layout = _layout_board(timed_beats, orientation=orientation)

    # Start the camera ALREADY FRAMED on the first beat's region rather
    # than the full board — a slow zoom-in ramp from a wide default view
    # meant text/icons looked tiny for the first ~1s of every render,
    # which read as broken rather than as an intentional establishing
    # shot. If the first beat has no region (e.g. starts with a talk
    # cutaway), the default wide view is still the right fallback.
    first_region = None
    for b in timed_beats:
        r = board_layout.get(b["beat_id"])
        if r:
            first_region = _fit_aspect(region_for_bbox(r, padding=60), frame["width"] / frame["height"])
            break

    cam = Camera(start_view=first_region, orientation=orientation)
    t_cursor = 0.0
    scene_beats = []

    # The camera is only allowed to START a move once the CURRENTLY
    # visible content has finished being drawn — not the instant the
    # previous camera move arrived (that mismatch was the actual bug:
    # a beat's own zoom-in only takes ~0.4s, but its text keeps
    # writing for several more seconds after that, and the OLD code
    # let the next beat's camera move start right after the zoom
    # arrived, drifting away mid-sentence). Updated to beat["end"]
    # after every beat with an active stroke-reveal or pinch gesture.
    camera_free_at = 0.0
    last_camera_center = _region_center(first_region) if first_region else None
    # BUG FIXED (historical): icon_group_id used to be generated purely
    # from concept_key (f"icon-{concept_key}"), with no per-beat
    # disambiguation. If two SEPARATE, unrelated beats picked the same
    # concept_key (e.g. two different sentences both drawing "brain"),
    # their paths landed in the SAME DOM group and both stayed visible
    # forever, overlapping. Each draw still gets a group id unique to
    # THAT beat (icon_group_id = f"icon-{beat_id}", set on beat_out
    # below) — that fix stands on its own and needs no extra tracking
    # dict now that nothing (zoom_in/zoom_out, point, relabel) ever
    # needs to look up "which group id is this concept_key's icon
    # right now" — those features are all removed.
    # NEW: sound effect cues — {"file": "...", "start": absolute_seconds}
    # collected as beats are processed, exported at the end for
    # run_render.py to mix into the final audio track. Purely additive:
    # if the sound files aren't present at render time, run_render.py
    # skips missing ones rather than failing the render.
    sound_cues = []

    def _apply_camera_move(beat_out_ref, beat_ref, region_ref, action, skip_transition=False):
        """Moves the camera toward region_ref (or the default wide view
        if region_ref is None, e.g. for 'swipe'). Duration is derived
        from actual travel distance + real available slack (see
        _camera_move_duration), not a flat constant.

        HAND REMOVED per direct feedback: this used to also attach a
        transition sweep-hand that passed across the screen on every
        beat-to-beat camera jump big enough to read as a cut. That hand
        is gone now — nothing visually covers the cut anymore. The
        woosh SOUND on a big-enough jump is UNCHANGED (kept exactly as
        before, same trigger condition, same timing) since only the
        hand was asked to go. This is a completely separate mechanism
        from the 'swipe' MODE's own dedicated wipe-hand (a full-board
        clear at a genuine topic shift, gesture_engine.scaled_swipe
        called directly in that mode's branch below) — that hand is
        untouched."""
        nonlocal camera_free_at, last_camera_center
        target_region = region_ref if region_ref is not None else cam.default_view
        target_center = _region_center(target_region)
        available_gap = beat_ref["start"] - camera_free_at
        duration = _camera_move_duration(last_camera_center, target_center, available_gap)

        cam.add(CameraMove(action=action, region=region_ref, duration=duration), start_t=camera_free_at)

        if last_camera_center is not None and not skip_transition:
            dx = target_center[0] - last_camera_center[0]
            dy = target_center[1] - last_camera_center[1]
            # Same "big enough to read as a cut" threshold as before —
            # only gates the SOUND now, not a hand.
            if (dx * dx + dy * dy) ** 0.5 > 150:
                sound_cues.append({"file": "woosh.mp3", "start": camera_free_at, "duration": duration})

        camera_free_at = beat_ref["end"]
        last_camera_center = target_center

    for beat in timed_beats:
        beat_out = {
            "beat_id": beat["beat_id"],
            "text": beat["text"],
            "mode": beat["mode"],
            "start": beat["start"],
            "end": beat["end"],
        }

        region = board_layout.get(beat["beat_id"])

        if beat["mode"] == "write" and region:
            # Text mode: REAL pen strokes via Hershey single-stroke font
            # data (see text_to_path.py docstring for why this replaced
            # the earlier filled-font approach) — computed deterministically
            # in Python, no font loading or generative step at render time.
            #
            # FIXED: was single-line only (fit_font_size) — for a long
            # sentence that function had no way to fit the width except
            # shrinking all the way to a hard 14px floor, and then STILL
            # overflowing past max_width because nothing ever wrapped it
            # to a second line. Now wraps across as many lines as needed
            # and picks the largest font that still fits vertically.
            pad = 0.12
            usable_w = region["w"] * (1 - 2 * pad)
            usable_h = region["h"] * (1 - 2 * pad)
            font_size = text_to_path.fit_font_size_wrapped(beat["text"], usable_w, usable_h)
            text_x = region["x"] + region["w"] * pad
            text_y = region["y"] + region["h"] * pad
            stroke_info = text_to_path.text_to_strokes_wrapped(
                beat["text"], x=text_x, y=text_y, font_size=font_size, max_width=usable_w
            )
            print(f"[render_pipeline] beat_id={beat['beat_id']} write mode: text={beat['text']!r} "
                  f"font_size={font_size:.1f} lines={stroke_info['line_count']} region={region} -> "
                  f"{len(stroke_info['subpaths'])} subpaths, {len(stroke_info['word_groups'])} words")

            beat_out["subpaths"] = stroke_info["subpaths"]
            # Stroke width proportional to font_size, NOT a fixed pixel
            # value — a constant stroke width was fine-looking at a big
            # font_size but became enormous relative to letters once a
            # smaller region (real script text, not a short test phrase)
            # forced font_size down to ~35px, causing adjacent pen
            # strokes to visually smear together into illegible mush.
            # This ratio keeps ink thickness looking natural regardless
            # of how big or small the text ends up rendering.
            beat_out["stroke_width"] = max(1.5, font_size * 0.045)
            beat_out["path_transform"] = None  # already baked into each subpath's coordinates
            beat_out["region"] = region

            # Real per-word Chatterbox timestamps drive reveal pace —
            # NOT a uniform slide. Each word maps to a RANGE of subpaths
            # (a word can be several pen strokes); that word's real
            # speech duration is split across just its own subpaths,
            # proportional to each stroke's own length, not divided
            # evenly across the whole sentence regardless of which word
            # is being spoken. Falls back to length-proportional across
            # the WHOLE beat (same technique as icons) if word counts
            # don't line up — logged, not silent, so a mismatch is
            # visible in the render log rather than just looking off.
            global_words = timing["words"]
            beat_words = [w for w in global_words if beat["start"] - 0.05 <= w["start"] < beat["end"] + 0.05]
            word_groups = stroke_info["word_groups"]

            # WRITING FINISHES EARLY, ON PURPOSE: previously the last
            # word's stroke finished at EXACTLY that word's own audio
            # end — zero buffer before the camera/swipe needed to move,
            # which is why the writing hand was still active right as
            # the swipe fired. All word timings are compressed by the
            # same ratio (rhythm relative to each other is preserved,
            # just the whole thing runs a bit faster) so the sentence
            # finishes being written roughly BUFFER_SECONDS before its
            # own audio ends — about 2-3 words' worth at typical pace.
            BUFFER_SECONDS = 1.1
            beat_duration = max(0.01, beat["end"] - beat["start"])
            compressed_duration = max(0.5, beat_duration - BUFFER_SECONDS)
            time_compression = compressed_duration / beat_duration
            # SOUND FIXED: now uses the ACTUAL compressed writing duration
            # (how long the hand really takes to finish), not the full
            # beat span — previously the sound file just played through
            # regardless of when writing actually stopped.
            sound_cues.append({"file": "wrighting.mp3", "start": beat["start"], "duration": compressed_duration})

            if len(beat_words) == len(word_groups) and len(beat_words) > 0:
                segment_durations = [None] * len(stroke_info["subpaths"])
                segment_delays = [None] * len(stroke_info["subpaths"])
                for gw, wg in zip(beat_words, word_groups):
                    word_start = max(0.0, (gw["start"] - beat["start"]) * time_compression)
                    word_duration = max(0.05, (gw["end"] - gw["start"]) * time_compression)
                    n_strokes = wg["subpath_end"] - wg["subpath_start"]
                    per_stroke = word_duration / max(1, n_strokes)
                    for i in range(wg["subpath_start"], wg["subpath_end"]):
                        segment_delays[i] = word_start + (i - wg["subpath_start"]) * per_stroke
                        segment_durations[i] = per_stroke
                beat_out["segment_durations"] = segment_durations
                beat_out["segment_delays"] = segment_delays
            else:
                print(f"[render_pipeline] beat_id={beat['beat_id']}: word count mismatch "
                      f"(chatterbox={len(beat_words)}, text_to_path={len(word_groups)}) "
                      f"— falling back to length-proportional reveal for this beat")
                beat_out["segment_durations"] = None
                beat_out["segment_delays"] = None
                # Fallback also needs to finish early — shrink the
                # window animateStrokeReveal treats as "beat.end" for
                # its length-proportional pacing (see totalDuration in
                # scene_template.html), same buffer amount.
                beat_out["end"] = beat["start"] + compressed_duration

            # NOTE: hand tracks the LIVE point on the stroke-reveal path every
            # frame, not a single static target — so we hand the template the
            # fully-resolved (already-scaled) hand data, not a pre-computed
            # placement. The template subtracts anchor_x/anchor_y from the
            # moving pen-tip point each frame. See scene_template.html.
            # target_height proportional to font_size — a real hand holding a
            # pen reads naturally scaled to the text height, not a fixed
            # constant, so it stays right-sized whether the region is huge or tiny.
            #
            # BUG FOUND (not just "needs more tuning"): earlier rounds only
            # bumped this multiplier, but font_size itself SHRINKS for long
            # sentences (fit_font_size forces it down to fit the region) —
            # so a bigger multiplier on a smaller font_size netted out to
            # almost no visible change (4.6*18.2=84 vs the previous
            # round's 4.0*20.2=81 — barely different). Clamping relative
            # to the REGION's own size (like icon-mode already correctly
            # does) fixes this for good: a floor so long sentences still
            # get a visibly-sized hand, a ceiling so short sentences with
            # a large font_size don't produce an absurdly oversized one.
            target_height = max(
                region["h"] * 0.40,
                min(font_size * 4.6, region["h"] * 0.85),
            )
            beat_out["hand"] = gesture_engine.scaled_hand("write", target_height=target_height).to_dict()

            _apply_camera_move(beat_out, beat, region, "zoom_in")

        elif beat["mode"] == "icon_word" and region and beat.get("icons"):
            # NEW: pure icon-only listicle items, per direct request —
            # PSYCHOLOGY CHANNEL ONLY (enforced below, not just left to
            # the prompt to get right, same "hardcode, don't rely on
            # upstream alone" discipline as every other channel-specific
            # behavior in this file). Replaces the icon+word row layout
            # for beats using "icons" instead of "items": no word
            # captions at all, just 1-4 icons arranged in a left/right
            # grid via _layout_icon_grid — one row of 2 for N=2, row 1
            # left+right + a centered lone icon on row 2 for N=3, a full
            # 2x2 grid for N=4. Mirrors the "items" branch's number
            # handling and word-synced timing exactly; the only real
            # difference is the grid layout and the absence of any word
            # visuals at all.
            if channel != "psychology":
                raise RuntimeError(
                    f"beat_id={beat['beat_id']}: 'icons' (icon-only grid) is only supported on the "
                    f"psychology channel per direct request — got channel={channel!r}. Use 'items' "
                    f"(icon,word pairs) for this channel instead."
                )

            icons = beat["icons"]
            if not isinstance(icons, list) or not (1 <= len(icons) <= 4):
                raise RuntimeError(
                    f"beat_id={beat['beat_id']}: 'icons' must be a list of 1-4 concept_keys — got {icons!r}"
                )

            row_gap = ROW_GAP_FIXED
            row_h = ROW_CONTENT_H

            beat_number = beat.get("number")
            has_number = beat_number is not None
            row_offset = 1 if has_number else 0

            global_words = timing["words"]
            beat_words = [w for w in global_words if beat["start"] - 0.05 <= w["start"] < beat["end"] + 0.05]

            number_start_t = number_end_t = None
            if has_number:
                if beat_words:
                    number_start_t, number_end_t = beat_words[0]["start"], beat_words[0]["end"]
                    if number_end_t - number_start_t < ICON_WORD_MIN_SLOT_DURATION:
                        number_end_t = number_start_t + ICON_WORD_MIN_SLOT_DURATION
                    content_words = [w for w in beat_words[1:] if w["start"] >= number_end_t - 1e-6]
                else:
                    print(f"[render_pipeline] beat_id={beat['beat_id']}: 'number' set but no real narration "
                          f"words found — skipping the number visual for this beat")
                    has_number = False
                    row_offset = 0
                    content_words = beat_words
            else:
                content_words = beat_words

            n_icons = len(icons)
            icon_grid_rows = (n_icons + 1) // 2
            word_slots = _word_synced_slots(content_words, n_icons)

            if word_slots is None:
                print(f"[render_pipeline] beat_id={beat['beat_id']}: not enough real narration words "
                      f"({len(content_words)}) for {n_icons} icons — falling back to fixed-proportion timing")
                _buffer = 1.1
                _beat_duration = max(0.01, beat["end"] - beat["start"])
                _usable = max(1.0, _beat_duration - _buffer)
                per_icon_duration = _usable / n_icons

            sub_visuals = []
            beat_out["region"] = region

            if has_number:
                number_row_region = {"x": region["x"], "y": region["y"], "w": region["w"], "h": row_h}
                number_str = str(beat_number)
                _num_pad = 0.12
                _num_usable_w = number_row_region["w"] * (1 - 2 * _num_pad)
                _num_usable_h = number_row_region["h"] * (1 - 2 * _num_pad)
                number_font_size = text_to_path.fit_font_size(number_str, _num_usable_w, _num_usable_h)
                number_x = _center_text_x(number_str, number_font_size, number_row_region)
                number_y = number_row_region["y"] + (number_row_region["h"] - number_font_size) / 2
                number_stroke_info = text_to_path.text_to_strokes(
                    number_str, x=number_x, y=number_y, font_size=number_font_size
                )
                sub_visuals.append({
                    "beat_id": f"{beat['beat_id']}-number",
                    "subpaths": number_stroke_info["subpaths"],
                    "stroke_width": max(1.5, number_font_size * 0.045),
                    "path_transform": None,
                    "start": number_start_t,
                    "end": number_end_t,
                    "min_reveal_duration": number_end_t - number_start_t,
                })
                sound_cues.append({"file": "wrighting.mp3", "start": number_start_t,
                                    "duration": number_end_t - number_start_t})

            grid_region = {
                "x": region["x"],
                "y": region["y"] + row_offset * (row_h + row_gap),
                "w": region["w"],
                "h": region["h"] - row_offset * (row_h + row_gap),
            }
            icon_boxes = _layout_icon_grid(grid_region, n_icons)

            illustration_items = []
            for i, concept_key_raw in enumerate(icons):
                concept_key = (concept_key_raw or "").strip()
                if not concept_key:
                    raise RuntimeError(f"beat_id={beat['beat_id']}: icons[{i}] is empty")

                if word_slots is not None:
                    icon_start_t, icon_end_t = word_slots[i]
                else:
                    _fallback_base = number_end_t if (has_number and number_end_t is not None) else beat["start"]
                    icon_start_t = _fallback_base + i * per_icon_duration
                    icon_end_t = icon_start_t + min(0.9, per_icon_duration * 0.9)
                sound_cues.append({"file": "drawing.mp3", "start": icon_start_t,
                                    "duration": icon_end_t - icon_start_t})

                # FIXED after a real production crash: this used to require
                # draw_style == "stroke_reveal" and raise the ENTIRE render
                # otherwise — but a concept_key can legitimately have NO
                # vendored icon match at all (e.g. "faucet" — genuinely
                # absent from all 4 libraries' filenames), which isn't a
                # bug, just reality, and shouldn't crash the video. This
                # now tries real icon resolution WITH retry-across-ranked-
                # candidates (see resolve_icon_stroke_path — also fixes the
                # separate case of a broken/unconvertable SVG file when a
                # working sibling icon exists), and only if truly NO
                # vendored icon works does it fall back to a small popping
                # image in the same grid slot instead of crashing.
                icon_path_info, item_asset_entry = resolve_icon_stroke_path(
                    concept_key, channel, illustration_cache_dir, icon_boxes[i], f"{beat['beat_id']}-icon{i}",
                )
                print(f"[render_pipeline] beat_id={beat['beat_id']} icon{i} concept_key={concept_key!r} "
                      f"-> resolved: source={item_asset_entry.get('asset_source')}")

                if icon_path_info is not None:
                    icon_group_id = f"icon-{beat['beat_id']}-grid{i}"
                    stroke_w = max(0.3, min(3.0, ICON_STROKE_TARGET_PX / icon_path_info["scale"]))
                    sub_visuals.append({
                        "beat_id": f"{beat['beat_id']}-grid{i}-icon",
                        "subpaths": icon_path_info["subpaths"],
                        "stroke_width": stroke_w,
                        "stroke_width_final": stroke_w * icon_path_info["scale"],
                        "path_transform": icon_path_info["transform"],
                        "path_offset_x": icon_path_info["offset_x"],
                        "path_offset_y": icon_path_info["offset_y"],
                        "path_scale": icon_path_info["scale"],
                        "icon_group_id": icon_group_id,
                        "start": icon_start_t,
                        "end": icon_end_t,
                        "min_reveal_duration": icon_end_t - icon_start_t,
                    })
                else:
                    # No vendored icon exists for this concept at all —
                    # fall back to the resolved stock image, shown as a
                    # small pop-in/out (see _illustration_reveal, applied
                    # pipeline-wide) inside this exact grid slot rather
                    # than failing the whole render over one missing icon.
                    cached_path = item_asset_entry.get("asset_ref", {}).get("cached_path")
                    if not cached_path:
                        raise RuntimeError(
                            f"beat_id={beat['beat_id']}: icons[{i}] concept_key={concept_key!r} has no "
                            f"vendored icon AND no stock image fallback either — nothing to draw for "
                            f"this grid slot. Add a curated entry to concept-library.json."
                        )
                    _, reveal_style = _illustration_reveal(channel, "icon", icon_boxes[i])
                    illustration_items.append({
                        "beat_id": f"{beat['beat_id']}-grid{i}-illus",
                        "illustration_path": cached_path,
                        "illustration_region": icon_boxes[i],
                        "illustration_start": icon_start_t,
                        "illustration_end": icon_end_t,
                        "illustration_reveal": reveal_style,
                    })

            beat_out["sub_visuals"] = sub_visuals
            if illustration_items:
                beat_out["illustration_items"] = illustration_items

            effective_rows = icon_grid_rows + row_offset
            target_height = (region["h"] / max(1, effective_rows)) * 0.85
            beat_out["hand"] = gesture_engine.scaled_hand("write", target_height=target_height).to_dict()

            _apply_camera_move(beat_out, beat, region, "zoom_in")

        elif beat["mode"] == "icon_word" and region and beat.get("items"):
            # REWRITTEN per direct feedback: items are no longer a
            # single horizontal strip of mixed icon/word cells — only
            # icon,word PAIRS are valid now (icon first, word second,
            # never word-word, icon-icon, or word-icon), each pair
            # forming its own ROW, stacked vertically. "icon-word" per
            # row, "icon-word" on the row below it, and so on for as
            # many pairs as the sentence has (capped at 3 rows / 6
            # items so a single beat's region doesn't get overcrowded).
            items = beat["items"][:6]
            if len(items) < 2 or len(items) % 2 != 0:
                raise RuntimeError(
                    f"beat_id={beat['beat_id']}: mode='icon_word' with 'items' needs an EVEN number of "
                    f"entries (2, 4, or 6) forming icon,word pairs — got {len(items)}"
                )
            pairs = []
            for i in range(0, len(items), 2):
                icon_item, word_item = items[i], items[i + 1]
                if icon_item.get("type", "icon") != "icon" or word_item.get("type") != "word":
                    raise RuntimeError(
                        f"beat_id={beat['beat_id']}: items[{i}]/items[{i+1}] must be an icon,word pair "
                        f"in that exact order — got types {icon_item.get('type')!r}, {word_item.get('type')!r}"
                    )
                pairs.append((icon_item, word_item))

            n_rows = len(pairs)
            row_gap = ROW_GAP_FIXED
            row_h = ROW_CONTENT_H

            # NEW: same "layout" idea as the single icon_word beat below —
            # optional, defaults to the original side-by-side split per row
            # so every existing script (which never sets this) renders
            # EXACTLY as before. "stacked" centers EVERY row's icon above
            # its word instead of icon-left/word-right — this is a
            # BEAT-level choice (applies uniformly to all rows in this
            # beat), not per-row, since a beat mixing both looks would be
            # visually inconsistent within one composition.
            items_layout = (beat.get("layout") or "side_by_side").strip().lower()

            # NEW, optional: "number" — for a numbered listicle sentence
            # ("One, you...", "Two, you..."), draws a big standalone
            # digit in its own row ABOVE the content rows, synced to
            # when the narrator actually says that number word. See
            # beat_schema.py's validation for the field itself.
            beat_number = beat.get("number")
            has_number = beat_number is not None
            row_offset = 1 if has_number else 0  # content rows shift down by one row-height

            # WORD-SYNCED TIMING (replaces the old fixed-duration-cap
            # scheme per direct feedback): each of this beat's 2*n_rows
            # content visuals (icon, word, icon, word, ...) gets a draw
            # duration equal to how long the narrator actually took to
            # say ITS share of the real sentence, not a flat capped
            # guess. Falls back to the old fixed/proportional scheme —
            # loudly logged, never silent — if the sentence has too few
            # real words for this many visuals to each get one.
            #
            # When this beat has a "number", its FIRST real word (e.g.
            # "One,") is carved out and given directly to the number
            # visual — not folded into the equal-split pool — since the
            # number should draw exactly while "One," is spoken, not
            # get an arbitrary equal share of the whole sentence. The
            # REST of the sentence's real timing then splits across the
            # content rows exactly as it would without a number.
            global_words = timing["words"]
            beat_words = [w for w in global_words if beat["start"] - 0.05 <= w["start"] < beat["end"] + 0.05]

            number_start_t = number_end_t = None
            if has_number:
                if beat_words:
                    number_start_t, number_end_t = beat_words[0]["start"], beat_words[0]["end"]
                    if number_end_t - number_start_t < ICON_WORD_MIN_SLOT_DURATION:
                        number_end_t = number_start_t + ICON_WORD_MIN_SLOT_DURATION
                    # Excludes by REAL start time (not just "drop the first
                    # word") — the min-duration floor above can push
                    # number_end_t slightly past when the next real word
                    # naturally starts, which would otherwise let the first
                    # content visual begin a few milliseconds before the
                    # number visual has actually finished revealing.
                    content_words = [w for w in beat_words[1:] if w["start"] >= number_end_t - 1e-6]
                else:
                    # No real narration words at all for this beat (shouldn't
                    # normally happen) — can't sync the number to anything,
                    # so skip it entirely rather than guess a placement.
                    print(f"[render_pipeline] beat_id={beat['beat_id']}: 'number' set but no real narration "
                          f"words found — skipping the number visual for this beat")
                    has_number = False
                    row_offset = 0
                    content_words = beat_words
            else:
                content_words = beat_words

            n_slots = n_rows * 2
            word_slots = _word_synced_slots(content_words, n_slots)

            if word_slots is None:
                print(f"[render_pipeline] beat_id={beat['beat_id']}: not enough real narration words "
                      f"({len(content_words)}) for {n_slots} visuals across {n_rows} row(s) — falling back "
                      f"to fixed-proportion timing for this beat")
                _buffer = 1.1
                _beat_duration = max(0.01, beat["end"] - beat["start"])
                _usable = max(1.0, _beat_duration - _buffer)
                per_row_duration = _usable / n_rows
                icon_duration = min(0.9, per_row_duration * 0.45)
                word_duration = min(0.9, per_row_duration * 0.45)

            sub_visuals = []
            illustration_items = []
            beat_out["region"] = region

            if has_number:
                number_row_region = {"x": region["x"], "y": region["y"], "w": region["w"], "h": row_h}
                number_str = str(beat_number)
                _num_pad = 0.12
                _num_usable_w = number_row_region["w"] * (1 - 2 * _num_pad)
                _num_usable_h = number_row_region["h"] * (1 - 2 * _num_pad)
                number_font_size = text_to_path.fit_font_size(number_str, _num_usable_w, _num_usable_h)
                number_x = _center_text_x(number_str, number_font_size, number_row_region)
                number_y = number_row_region["y"] + (number_row_region["h"] - number_font_size) / 2
                number_stroke_info = text_to_path.text_to_strokes(
                    number_str, x=number_x, y=number_y, font_size=number_font_size
                )
                sub_visuals.append({
                    "beat_id": f"{beat['beat_id']}-number",
                    "subpaths": number_stroke_info["subpaths"],
                    "stroke_width": max(1.5, number_font_size * 0.045),
                    "path_transform": None,
                    "start": number_start_t,
                    "end": number_end_t,
                    "min_reveal_duration": number_end_t - number_start_t,
                })
                sound_cues.append({"file": "wrighting.mp3", "start": number_start_t,
                                    "duration": number_end_t - number_start_t})

            for row_idx, (icon_item, word_item) in enumerate(pairs):
                row_y = region["y"] + (row_idx + row_offset) * (row_h + row_gap)
                row_region = {"x": region["x"], "y": row_y, "w": region["w"], "h": row_h}
                if items_layout == "stacked":
                    # Icon centered on top of THIS row, word centered
                    # directly below it — same proportions as the single
                    # icon_word beat's stacked layout, just applied per
                    # row within the row's own (shorter) height budget.
                    icon_h = row_region["h"] * 0.62
                    stack_gap = row_region["h"] * 0.06
                    word_h = row_region["h"] - icon_h - stack_gap
                    icon_w = row_region["w"] * 0.5  # narrower inset than the single-beat
                                                     # version since a row is already compact
                    icon_region = {
                        "x": row_region["x"] + (row_region["w"] - icon_w) / 2,
                        "y": row_region["y"],
                        "w": icon_w,
                        "h": icon_h,
                    }
                    word_region = {
                        "x": row_region["x"],
                        "y": row_region["y"] + icon_h + stack_gap,
                        "w": row_region["w"],
                        "h": word_h,
                    }
                else:
                    icon_w = row_region["w"] * 0.46  # increased from 0.42 — icons/images were reading small
                    gutter = row_region["w"] * 0.05
                    icon_region = {"x": row_region["x"], "y": row_region["y"], "w": icon_w, "h": row_region["h"]}
                    word_region = {
                        "x": row_region["x"] + icon_w + gutter, "y": row_region["y"],
                        "w": row_region["w"] - icon_w - gutter, "h": row_region["h"],
                    }
                if word_slots is not None:
                    icon_start_t, icon_end_t = word_slots[row_idx * 2]
                    word_start_t, word_end_t = word_slots[row_idx * 2 + 1]
                    icon_duration = icon_end_t - icon_start_t
                    word_duration = word_end_t - word_start_t
                else:
                    # If this beat has a number, its own reveal already
                    # claimed the time right at the start of the beat —
                    # content rows in the fallback path start after it
                    # finishes instead of at beat['start'], so they don't
                    # draw simultaneously with the number.
                    _fallback_row_base = number_end_t if (has_number and number_end_t is not None) else beat["start"]
                    row_start = _fallback_row_base + row_idx * per_row_duration
                    icon_start_t = row_start
                    icon_end_t = icon_start_t + icon_duration
                    word_start_t = icon_end_t
                    word_end_t = word_start_t + word_duration
                sound_cues.append({"file": "drawing.mp3", "start": icon_start_t, "duration": icon_duration})
                sound_cues.append({"file": "wrighting.mp3", "start": word_start_t, "duration": word_duration})

                concept_key = (icon_item.get("concept_key") or "").strip()
                if not concept_key:
                    raise RuntimeError(f"beat_id={beat['beat_id']}: items[{row_idx*2}] (icon) needs a 'concept_key'")
                # asset_type is per-ITEM (optional, defaults to "icon") — a
                # single multi-row beat can mix a generic icon in one row
                # with a real photo in another (e.g. row 1: "cult" icon +
                # "joined", row 2: an actual photo of the person + their name).
                item_asset_type = (icon_item.get("asset_type") or "icon").strip()
                item_asset_entry = resolve_beat_asset(
                    {"concept_key": concept_key, "beat_id": f"{beat['beat_id']}-row{row_idx}"},
                    channel, illustration_cache_dir, asset_type=item_asset_type,
                )
                print(f"[render_pipeline] beat_id={beat['beat_id']} row{row_idx} concept_key={concept_key!r} "
                      f"asset_type={item_asset_type!r} layout={items_layout!r} -> resolved: "
                      f"source={item_asset_entry.get('asset_source')}")

                if item_asset_entry.get("draw_style") == "stroke_reveal":
                    svg_path = item_asset_entry["asset_ref"].get("path")
                    # Reduced padding (was default 0.15) — icons were reading small; less
                    # internal padding means the icon fills more of its allotted box.
                    icon_path_info = svg_to_path.icon_to_path_d(svg_path, icon_region, padding_ratio=0.08) if svg_path else None
                    if icon_path_info is None:
                        raise RuntimeError(
                            f"beat_id={beat['beat_id']}: row{row_idx} concept_key={concept_key!r} resolved to an "
                            f"icon with no usable <path> data — pick a different concept_key for this row."
                        )
                    icon_group_id = f"icon-{beat['beat_id']}-row{row_idx}"
                    stroke_w = max(0.3, min(3.0, ICON_STROKE_TARGET_PX / icon_path_info["scale"]))
                    sub_visuals.append({
                        "beat_id": f"{beat['beat_id']}-row{row_idx}-icon",
                        "subpaths": icon_path_info["subpaths"],
                        "stroke_width": stroke_w,
                        "stroke_width_final": stroke_w * icon_path_info["scale"],
                        "path_transform": icon_path_info["transform"],
                        "path_offset_x": icon_path_info["offset_x"],
                        "path_offset_y": icon_path_info["offset_y"],
                        "path_scale": icon_path_info["scale"],
                        "icon_group_id": icon_group_id,
                        "start": icon_start_t,
                        "end": icon_end_t,
                        "min_reveal_duration": icon_duration,
                    })
                else:
                    # IMAGES SPECIFICALLY ENLARGED per direct feedback — a
                    # photo/illustration reads noticeably smaller/less bold
                    # than a bold vector icon line at the identical box size,
                    # so images get their own bigger region here, grown
                    # around the same center point rather than reusing the
                    # vector icon's box as-is.
                    _img_scale = 1.3
                    _cx = icon_region["x"] + icon_region["w"] / 2
                    _cy = icon_region["y"] + icon_region["h"] / 2
                    _grown_w = icon_region["w"] * _img_scale
                    _grown_h = icon_region["h"] * _img_scale
                    if items_layout == "stacked":
                        # Same reasoning as the single-beat stacked layout:
                        # a big image growing symmetrically around its
                        # center would eat into the word directly below
                        # it, so height growth is clamped to the icon's
                        # own row box; only width may grow.
                        _grown_h = icon_region["h"]
                        _grown_w = min(_grown_w, row_region["w"])
                    image_region = {
                        "x": _cx - _grown_w / 2,
                        "y": _cy - _grown_h / 2,
                        "w": _grown_w,
                        "h": _grown_h,
                    }
                    hand, reveal_style = _illustration_reveal(channel, item_asset_type, image_region)
                    illus_item = {
                        "beat_id": f"{beat['beat_id']}-row{row_idx}-illus",
                        "illustration_path": item_asset_entry["asset_ref"].get("cached_path"),
                        "illustration_region": image_region,
                        "illustration_start": icon_start_t,
                        "illustration_end": icon_end_t,
                        "illustration_reveal": reveal_style,
                    }
                    if hand:
                        illus_item["mask_wipe_hand"] = hand
                    illustration_items.append(illus_item)

                label = (word_item.get("label") or "").strip()
                if not label:
                    raise RuntimeError(f"beat_id={beat['beat_id']}: items[{row_idx*2+1}] (word) needs a non-empty 'label'")
                pad = 0.15
                usable_w = word_region["w"] * (1 - 2 * pad)
                usable_h = word_region["h"] * (1 - 2 * pad)
                font_size = text_to_path.fit_font_size(label, usable_w, usable_h)
                text_x = _center_text_x(label, font_size, word_region)
                text_y = word_region["y"] + (word_region["h"] - font_size) / 2
                stroke_info = text_to_path.text_to_strokes(label, x=text_x, y=text_y, font_size=font_size)
                sub_visuals.append({
                    "beat_id": f"{beat['beat_id']}-row{row_idx}-word",
                    "subpaths": stroke_info["subpaths"],
                    "stroke_width": max(1.5, font_size * 0.045),
                    "path_transform": None,
                    "start": word_start_t,
                    "end": word_end_t,
                    "min_reveal_duration": word_duration,
                })

            beat_out["sub_visuals"] = sub_visuals
            if illustration_items:
                beat_out["illustration_items"] = illustration_items
            # BUG FIXED (confirmed by direct comparison): this used a
            # 0.5 multiplier on top of the per-row height, while the
            # single icon_word beat below uses up to 0.85 of its own
            # (taller) CONTENT_H — for a 2-row beat that worked out to
            # roughly 119 world-units vs the single-beat's 128-272
            # range, a real, visible size mismatch, not a rounding
            # difference. Matching the single-beat's own upper
            # multiplier (0.85) here keeps the hand a consistent size
            # regardless of how many rows a beat has — it no longer
            # shrinks just because there's more content stacked below it.
            # effective_rows includes the number row (if present) since
            # region["h"] itself grew to include that row's height too —
            # dividing by n_rows alone here would OVERESTIMATE the hand.
            effective_rows = n_rows + row_offset
            target_height = (region["h"] / max(1, effective_rows)) * 0.85
            beat_out["hand"] = gesture_engine.scaled_hand("write", target_height=target_height).to_dict()

            _apply_camera_move(beat_out, beat, region, "zoom_in")

        elif beat["mode"] == "icon_word" and region:
            # NEW: "layout" picks how the icon and its word share this
            # beat's region. Optional, defaults to the original
            # side-by-side split so every existing script (which never
            # sets this field) renders EXACTLY as before, unchanged.
            #   "side_by_side" (default): icon on the left, word on the right.
            #   "stacked": icon centered on top, word centered directly below it.
            icon_layout = (beat.get("layout") or "side_by_side").strip().lower()

            if icon_layout == "stacked":
                # Icon takes the top ~62% of the region, word the
                # remainder below it, both horizontally centered. A
                # fixed fraction of region height (not a fixed pixel
                # count) so this scales correctly whether the region
                # is a normal single slot or a bigger grouped one.
                icon_h = region["h"] * 0.62
                stack_gap = region["h"] * 0.06
                text_h = region["h"] - icon_h - stack_gap
                icon_w = region["w"] * 0.8  # inset from the sides so a big icon never touches the slot edge
                icon_region = {
                    "x": region["x"] + (region["w"] - icon_w) / 2,
                    "y": region["y"],
                    "w": icon_w,
                    "h": icon_h,
                }
                text_region = {
                    "x": region["x"],
                    "y": region["y"] + icon_h + stack_gap,
                    "w": region["w"],
                    "h": text_h,
                }
            else:
                # Left ~46% of this beat's region for the icon, right side
                # for the short written label — NOT the full sentence, just
                # the label field (e.g. concept_key="food", label="good").
                # Increased from 0.42/0.06 — icon/image was reading small.
                gutter = region["w"] * 0.05
                icon_w = region["w"] * 0.46
                icon_region = {"x": region["x"], "y": region["y"], "w": icon_w, "h": region["h"]}
                text_region = {
                    "x": region["x"] + icon_w + gutter, "y": region["y"],
                    "w": region["w"] - icon_w - gutter, "h": region["h"],
                }
            label = beat.get("label") or beat["text"]

            asset_type = (beat.get("asset_type") or "icon").strip()
            asset_entry = resolve_beat_asset(beat, channel, illustration_cache_dir, asset_type=asset_type)
            beat_out["asset"] = asset_entry
            beat_out["region"] = region
            print(f"[render_pipeline] beat_id={beat['beat_id']} icon_word concept_key={beat.get('concept_key')!r} "
                  f"asset_type={asset_type!r} layout={icon_layout!r} label={label!r} -> resolved: "
                  f"source={asset_entry.get('asset_source')}")

            # WORD-SYNCED TIMING (replaces the old fixed-duration-cap
            # scheme per direct feedback): icon and label each get a
            # draw duration equal to how long the narrator actually
            # took to say their real share of the sentence — roughly
            # half the sentence's words each — instead of a flat capped
            # guess with no relationship to the sentence's real length
            # or pace. Falls back to the old fixed-proportion scheme —
            # loudly logged, never silent — if the sentence has too few
            # real words to split in two.
            global_words = timing["words"]
            beat_words = [w for w in global_words if beat["start"] - 0.05 <= w["start"] < beat["end"] + 0.05]
            word_slots = _word_synced_slots(beat_words, 2)

            if word_slots is not None:
                icon_start_t, icon_end_t = word_slots[0]
                label_start_t, label_end_t = word_slots[1]
                icon_duration = icon_end_t - icon_start_t
                label_duration = label_end_t - label_start_t
            else:
                print(f"[render_pipeline] beat_id={beat['beat_id']}: not enough real narration words "
                      f"({len(beat_words)}) to split between icon and label — falling back to "
                      f"fixed-proportion timing for this beat")
                _icon_word_buffer = 1.1
                _beat_duration = max(0.01, beat["end"] - beat["start"])
                _usable = max(1.0, _beat_duration - _icon_word_buffer)
                icon_duration = min(0.9, _usable * 0.45)
                label_duration = min(0.9, _usable * 0.45)
                icon_start_t = beat["start"]
                icon_end_t = icon_start_t + icon_duration
                label_start_t = icon_end_t
                label_end_t = label_start_t + label_duration
            sound_cues.append({"file": "drawing.mp3", "start": icon_start_t, "duration": icon_duration})
            sound_cues.append({"file": "wrighting.mp3", "start": label_start_t, "duration": label_duration})

            sub_visuals = []

            if asset_entry.get("draw_style") == "stroke_reveal":
                svg_path = asset_entry["asset_ref"].get("path")
                icon_path_info = svg_to_path.icon_to_path_d(svg_path, icon_region, padding_ratio=0.08) if svg_path else None
                if icon_path_info is None:
                    raise RuntimeError(
                        f"beat_id={beat['beat_id']}: icon_word concept_key resolved to an icon with "
                        f"no usable <path> data — pick a different concept_key for this beat."
                    )
                icon_group_id = f"icon-{beat['beat_id']}"
                sub_visuals.append({
                    "beat_id": f"{beat['beat_id']}-icon",
                    "subpaths": icon_path_info["subpaths"],
                    # THICKNESS: scale-compensated (not a flat constant)
                    # so it looks visually consistent across icons whose
                    # fit-scale differs — a truly fixed width only works
                    # WITH non-scaling-stroke, but that's now confirmed
                    # (via an actual browser test) to corrupt the
                    # dasharray reveal, so non-scaling-stroke is only
                    # applied AFTER the reveal finishes, not during it.
                    # Wide safety clamp, not a tight band — a tight clamp
                    # was what caused inconsistent results before.
                    "stroke_width": max(0.3, min(3.0, ICON_STROKE_TARGET_PX / icon_path_info["scale"])),
                    # BUG FOUND AND FIXED: switching to non-scaling-stroke
                    # after the reveal, WITHOUT also updating the numeric
                    # stroke-width value, meant the SAME number suddenly
                    # got interpreted in a different coordinate space —
                    # rendering much thinner right after finishing. This
                    # is the exact "thick while drawing, thin right after"
                    # bug. stroke_width_final is the equivalent absolute
                    # width, applied at the SAME moment non-scaling-stroke
                    # is — same visual thickness, no jump.
                    "stroke_width_final": max(0.3, min(3.0, ICON_STROKE_TARGET_PX / icon_path_info["scale"])) * icon_path_info["scale"],
                    "path_transform": icon_path_info["transform"],
                    "path_offset_x": icon_path_info["offset_x"],
                    "path_offset_y": icon_path_info["offset_y"],
                    "path_scale": icon_path_info["scale"],
                    "icon_group_id": icon_group_id,
                    "start": icon_start_t,
                    "end": icon_end_t,
                    "min_reveal_duration": icon_duration,
                })
            else:
                # mask_wipe illustration fallback — no pen-stroke reveal available for this
                # concept. FIXED: previously placed instantly with no reveal timing and no
                # hand at all — now wipes in during the icon's own half of the beat, with a
                # 'drag' hand tracking the reveal edge, matching the draw-mode fix.
                # IMAGES ENLARGED (same as the row-based branch) — a photo reads noticeably
                # smaller/less bold than a vector icon at the identical box size.
                _img_scale = 1.3
                _cx = icon_region["x"] + icon_region["w"] / 2
                _cy = icon_region["y"] + icon_region["h"] / 2
                _grown_w = icon_region["w"] * _img_scale
                _grown_h = icon_region["h"] * _img_scale
                if icon_layout == "stacked":
                    # A big image growing symmetrically around its center
                    # would eat into the word row directly below it — so
                    # for the stacked layout, height growth is clamped to
                    # the icon's own allotted box; only width is allowed
                    # to grow (and only up to the region's own width).
                    _grown_h = icon_region["h"]
                    _grown_w = min(_grown_w, region["w"])
                image_region = {
                    "x": _cx - _grown_w / 2,
                    "y": _cy - _grown_h / 2,
                    "w": _grown_w,
                    "h": _grown_h,
                }
                beat_out["illustration_path"] = asset_entry["asset_ref"].get("cached_path")
                beat_out["illustration_region"] = image_region
                beat_out["illustration_start"] = icon_start_t
                beat_out["illustration_end"] = icon_end_t
                hand, reveal_style = _illustration_reveal(channel, asset_type, image_region)
                beat_out["illustration_reveal"] = reveal_style
                if hand:
                    beat_out["mask_wipe_hand"] = hand

            pad = 0.15
            usable_w = text_region["w"] * (1 - 2 * pad)
            usable_h = text_region["h"] * (1 - 2 * pad)
            font_size = text_to_path.fit_font_size(label, usable_w, usable_h)
            text_x = _center_text_x(label, font_size, text_region)
            # VERTICALLY CENTERED, not top-anchored (region.y + h*pad) —
            # the icon (icon_to_path_d) fits/centers itself within its
            # own full-height region, so a top-anchored label sat well
            # above the icon's actual center. font_size directly
            # approximates the glyphs' total rendered height (Hershey
            # data spans ~_NATIVE_Y_SPAN units, which font_size is
            # scaled to) so this centers the label on the SAME
            # vertical midpoint the icon is centered on.
            text_y = text_region["y"] + (text_region["h"] - font_size) / 2
            stroke_info = text_to_path.text_to_strokes(label, x=text_x, y=text_y, font_size=font_size)
            label_group_id = f"label-{beat['beat_id']}"
            sub_visuals.append({
                "beat_id": f"{beat['beat_id']}-label",
                "subpaths": stroke_info["subpaths"],
                "stroke_width": max(1.5, font_size * 0.045),
                "path_transform": None,
                "icon_group_id": label_group_id,  # reuses the same generic
                # "parent this path under a named <g>" mechanism the
                # template already has for icons — works identically
                # for a label's paths, just under a different id prefix.
                "start": label_start_t,
                "end": label_end_t,
                "min_reveal_duration": label_duration,
            })

            beat_out["sub_visuals"] = sub_visuals
            target_height = max(region["h"] * 0.40, min(font_size * 4.6, region["h"] * 0.85))
            beat_out["hand"] = gesture_engine.scaled_hand("write", target_height=target_height).to_dict()

            _apply_camera_move(beat_out, beat, region, "zoom_in")

        elif beat["mode"] == "draw" and region:
            asset_type = (beat.get("asset_type") or "icon").strip()
            asset_entry = resolve_beat_asset(beat, channel, illustration_cache_dir, asset_type=asset_type)
            beat_out["asset"] = asset_entry
            beat_out["region"] = region
            print(f"[render_pipeline] beat_id={beat['beat_id']} concept_key={beat.get('concept_key')!r} "
                  f"asset_type={asset_type!r} -> resolved: source={asset_entry.get('asset_source')} "
                  f"ref={asset_entry.get('asset_ref')} draw_style={asset_entry.get('draw_style')}")

            if asset_entry.get("draw_style") == "stroke_reveal":
                svg_path = asset_entry["asset_ref"].get("path")
                icon_path_info = svg_to_path.icon_to_path_d(svg_path, region) if svg_path else None
                if icon_path_info is None:
                    raise RuntimeError(
                        f"beat_id={beat['beat_id']}: concept_key resolved to an icon with no usable "
                        f"<path> data (see svg_to_path.py LIMITATION) — pick a different icon for this "
                        f"concept_key or extend svg_to_path.py to handle primitive shapes."
                    )
                beat_out["subpaths"] = icon_path_info["subpaths"]
                # BUG FIXED (historical): was f"icon-{concept_key}" —
                # collided when two SEPARATE, unrelated beats picked the
                # same concept_key (e.g. two different sentences both
                # drawing "brain"), merging their strokes into the SAME
                # DOM group so both stayed visible forever, overlapping.
                # Unique per beat now — that's all this needs to do.
                icon_group_id = f"icon-{beat['beat_id']}"
                beat_out["icon_group_id"] = icon_group_id
                # THICKNESS: scale-compensated during the reveal (see
                # icon_word branch above for the full reasoning) —
                # non-scaling-stroke is applied only AFTER the reveal
                # completes.
                _icon_stroke_w = max(0.3, min(3.0, ICON_STROKE_TARGET_PX / icon_path_info["scale"]))
                beat_out["stroke_width"] = _icon_stroke_w
                # Equivalent absolute width for non-scaling-stroke,
                # applied at the same moment — fixes the thick-then-
                # thin jump right after the reveal finishes.
                beat_out["stroke_width_final"] = _icon_stroke_w * icon_path_info["scale"]
                beat_out["min_reveal_duration"] = 1.3
                sound_cues.append({"file": "drawing.mp3", "start": beat["start"], "duration": 1.3})
                beat_out["path_transform"] = icon_path_info["transform"]
                beat_out["path_offset_x"] = icon_path_info["offset_x"]
                beat_out["path_offset_y"] = icon_path_info["offset_y"]
                beat_out["path_scale"] = icon_path_info["scale"]
            else:
                # mask_wipe illustration — no path data, template reveals via clip-path sweep instead.
                # BUG FIXED: this branch never set a hand at all before — confirmed in the
                # code (the hand assignment below only fires when "subpaths" is present) —
                # which is exactly the hand-less icon in the screenshot. Now gets a 'drag'
                # gesture that tracks the reveal edge.
                # SECOND BUG FOUND AND FIXED HERE: this was the one mask_wipe case that
                # never got the fast-reveal timing fix (icon_word's mask_wipe got it
                # earlier, this plain "draw" one was missed) — it was spanning the ENTIRE
                # beat instead of a fast, fixed duration, matching the stroke_reveal
                # sibling case right above.
                _draw_reveal_duration = min(1.3, max(0.01, beat["end"] - beat["start"]))
                beat_out["illustration_path"] = asset_entry["asset_ref"].get("cached_path")
                beat_out["illustration_region"] = region
                beat_out["illustration_start"] = beat["start"]
                beat_out["illustration_end"] = beat["start"] + _draw_reveal_duration
                hand, reveal_style = _illustration_reveal(channel, asset_type, region)
                beat_out["illustration_reveal"] = reveal_style
                if hand:
                    beat_out["mask_wipe_hand"] = hand
                    sound_cues.append({"file": "drawing.mp3", "start": beat["start"], "duration": _draw_reveal_duration})

            if "subpaths" in beat_out or "path_d" in beat_out:
                # target_height proportional to region size for icons — a hand
                # tracing a small icon should be noticeably smaller than one
                # writing a full sentence, not a fixed constant either way.
                # Bumped up from 0.5 -> 0.68 (was reading as too small).
                # Bumped up again per feedback: 0.68 -> 0.85x region height.
                # Bumped again per feedback: 0.85 -> 0.95x region height.
                target_height = region["h"] * 0.95
                beat_out["hand"] = gesture_engine.scaled_hand("write", target_height=target_height).to_dict()

            _apply_camera_move(beat_out, beat, region, "zoom_in")

        elif beat["mode"] == "swipe":
            sweep = gesture_engine.scaled_swipe(
                direction=beat.get("swipe_direction", "ltr"),
                frame_width=frame["width"], target_height=280,
            )
            sweep["y"] = frame["height"] / 2 - sweep["anchor_y"]
            sweep["start"] = camera_free_at
            sweep["duration"] = max(0.4, beat["end"] - beat["start"])
            beat_out["hand_sweep"] = sweep
            sound_cues.append({"file": "woosh.mp3", "start": sweep["start"], "duration": sweep["duration"]})
            _apply_camera_move(beat_out, beat, None, "zoom_out", skip_transition=True)

        elif beat["mode"] == "talk":
            beat_out["cutaway_file"] = gesture_engine.cutaway_for_beat(beat["beat_id"])

        scene_beats.append(beat_out)

    return {
        "channel": channel,
        "audio_path": timing["audio_path"],
        "orientation": orientation,
        "frame": frame,
        "board": board,
        "camera_keyframes": json.loads(cam.gsap_keyframes_js()),
        "beats": scene_beats,
        "sound_cues": sound_cues,
        "captions": _build_captions(timing["words"]),
    }


if __name__ == "__main__":
    import sys
    print("This module builds a scene program dict — run via a channel-specific "
          "trigger script that supplies script_text + beats from n8n's extractor output. "
          "See EXAMPLE_BEAT in beat_schema.py for the expected shape of each beat.")
    sys.exit(0)
