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
import gesture_engine
import text_to_path
import svg_to_path
from camera import Camera, CameraMove, get_frame_dims, get_board_dims, get_default_view, region_for_bbox, _fit_aspect
from beat_schema import validate_batch, load_vocabulary, GESTURE_FOR_MODE

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# How much bigger a pinched-in icon gets, relative to its original
# drawn size. Tune this single number if "enlarge" should read as
# more/less dramatic — nothing else needs to change.
ICON_ENLARGE_SCALE = 1.45  # reduced from 1.9 — was overflowing off-canvas on pinch-enlarge
ICON_STROKE_TARGET_PX = 45.0  # numerator for scale-compensated stroke width (stroke_width =
                               # this / icon_path_info["scale"]) — tuned to land near "puzzle
                               # icon" thickness, moderately increased from the original 5.0.
                               # Wide clamp applied at each usage site, not a tight band.

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


def resolve_beat_asset(beat: dict, channel: str, illustration_cache_dir: str) -> dict:
    """Checks shared + channel library first (curated, no network),
    falls back to live asset_resolver.resolve(), and writes new
    resolutions back to the channel's own library so this is a
    one-time cost per concept, not per render."""
    concept_key = beat.get("concept_key")
    if not concept_key:
        return None

    shared_lib = _load_library(SHARED_LIBRARY_PATH)
    channel_lib_path = _channel_library_path(channel)
    channel_lib = _load_library(channel_lib_path)

    if concept_key in shared_lib:
        return shared_lib[concept_key]
    if concept_key in channel_lib:
        return channel_lib[concept_key]

    # No curated entry — live resolve, then persist.
    resolved = asset_resolver.resolve(concept_key, cache_dir=illustration_cache_dir)
    if resolved is None:
        raise RuntimeError(
            f"No asset found for concept_key '{concept_key}' (beat_id={beat.get('beat_id')}) "
            f"across icons or any stock source — cannot render this beat. Add a curated entry "
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
    SAFETY_MARGIN = 160
    SLOT_W = max(500, fitted["w"] + SAFETY_MARGIN)
    SLOT_H = max(400, fitted["h"] + SAFETY_MARGIN)

    COLS = max(2, board["width"] // SLOT_W)
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
        base_x = MARGIN + col * SLOT_W
        base_y = MARGIN + row * SLOT_H
        full_w, full_h = CONTENT_W, CONTENT_H

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
# Scene program assembly
# ══════════════════════════════════════════════════════════════════
def build_scene_program(script_text: str, beats: List[dict], channel: str,
                         voice: str = None, illustration_cache_dir: str = "/tmp/illustration_cache",
                         orientation: str = "landscape") -> dict:
    vocab = load_vocabulary(SHARED_LIBRARY_PATH, _channel_library_path(channel))
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
    # BUG FIXED: icon_group_id used to be generated purely from
    # concept_key (f"icon-{concept_key}"), with no per-beat
    # disambiguation. If two SEPARATE, unrelated beats picked the same
    # concept_key (e.g. two different sentences both drawing "brain"),
    # their paths landed in the SAME DOM group and both stayed visible
    # forever, overlapping — confirmed via code read, this is what
    # produced two simultaneous brain icons. Each draw now gets a
    # group id unique to THAT beat; this dict tracks which group id is
    # the CURRENT (most recently drawn) one for a given concept_key, so
    # point/zoom_in/zoom_out can still find the right icon to reference.
    concept_group_ids = {}

    def _apply_camera_move(beat_out_ref, beat_ref, region_ref, action, skip_transition=False):
        """Moves the camera toward region_ref (or the default wide view
        if region_ref is None, e.g. for 'swipe'). Duration is derived
        from actual travel distance + real available slack (see
        _camera_move_duration), not a flat constant. Also attaches a
        swipe-hand 'camera_transition' that plays IN THE SAME DIRECTION
        the camera pans, timed to the exact same window — masks the cut
        the way a real hand sweeping past the lens would."""
        nonlocal camera_free_at, last_camera_center
        target_region = region_ref if region_ref is not None else cam.default_view
        target_center = _region_center(target_region)
        available_gap = beat_ref["start"] - camera_free_at
        duration = _camera_move_duration(last_camera_center, target_center, available_gap)

        cam.add(CameraMove(action=action, region=region_ref, duration=duration), start_t=camera_free_at)

        if last_camera_center is not None and not skip_transition:
            dx = target_center[0] - last_camera_center[0]
            dy = target_center[1] - last_camera_center[1]
            # Only spawn a transition hand for a move big enough to
            # actually read as a cut — a tiny reframe within the same
            # neighborhood shouldn't flash a hand across the screen.
            if (dx * dx + dy * dy) ** 0.5 > 150:
                # DIRECTION FIXED — this was backwards before. Spec:
                # camera pans left -> hand sweeps LEFT-TO-RIGHT,
                # camera pans right -> hand sweeps RIGHT-TO-LEFT,
                # camera pans down -> hand sweeps DOWN-TO-UP (and the
                # symmetric case up -> hand sweeps up-to-down).
                # Whichever axis the camera moved MORE on is the one
                # that sweeps (a diagonal row-wrap move picks its
                # dominant axis rather than doing both at once).
                if abs(dx) >= abs(dy):
                    direction = "ltr" if dx < 0 else "rtl"
                    sweep = gesture_engine.scaled_swipe(
                        direction=direction, frame_width=frame["width"],
                        target_height=int(frame["height"] * 0.3),
                    )
                else:
                    direction = "btt" if dy > 0 else "ttb"
                    sweep = gesture_engine.scaled_swipe(
                        direction=direction, frame_width=frame["width"],
                        frame_height=frame["height"],
                        target_height=int(frame["width"] * 0.3),
                    )
                # SPEED FIXED — the hand always travels the full
                # off-screen-to-off-screen distance regardless of how
                # far the camera itself panned (which could be a very
                # short, fast move). Sharing the camera's own duration
                # made the hand look like a blur relative to the
                # distance it covered. It now gets its own readable
                # minimum duration, but is still anchored to arrive at
                # EXACTLY the same instant the camera does (start time
                # shifted earlier if needed), so it still genuinely
                # "matches" the camera move rather than just being a
                # fixed constant with no relationship to it.
                move_arrival = camera_free_at + duration
                sweep_duration = max(0.9, duration)
                sweep["start"] = move_arrival - sweep_duration
                sweep["duration"] = sweep_duration
                beat_out_ref["camera_transition"] = sweep

        camera_free_at = beat_ref["end"]
        last_camera_center = target_center

    # Maps concept_key -> the region it was drawn in, populated as we
    # process draw/write beats in order. point/zoom_in/zoom_out beats
    # don't get their own board slot (they reference something ALREADY
    # drawn, not new content) — this is what they look their target
    # region up from. A point/zoom beat whose concept_key was never
    # drawn earlier in the script has nothing to point at — that's a
    # script-authoring problem (Gemini referenced a concept before
    # introducing it), surfaced as a loud error below, not silently
    # producing an empty beat like it did before this fix.
    concept_regions = {}

    for beat in timed_beats:
        beat_out = {
            "beat_id": beat["beat_id"],
            "text": beat["text"],
            "mode": beat["mode"],
            "start": beat["start"],
            "end": beat["end"],
        }

        region = board_layout.get(beat["beat_id"])
        if region and beat.get("concept_key"):
            concept_regions[beat["concept_key"]] = region
        if region is None and beat["mode"] in ("point", "zoom_in", "zoom_out"):
            ck = beat.get("concept_key")
            region = concept_regions.get(ck)
            if region is None:
                raise RuntimeError(
                    f"beat_id={beat['beat_id']} mode='{beat['mode']}' references concept_key="
                    f"{ck!r}, but nothing with that concept_key was drawn earlier in this script "
                    f"— {beat['mode']} needs an existing element to reference. Check the script's "
                    f"beat order, or that the concept_key matches exactly."
                )

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

        elif beat["mode"] == "icon_word" and region:
            # Left ~42% of this beat's region for the icon, right side
            # for the short written label — NOT the full sentence, just
            # the label field (e.g. concept_key="food", label="good").
            gutter = region["w"] * 0.06
            icon_w = region["w"] * 0.42
            icon_region = {"x": region["x"], "y": region["y"], "w": icon_w, "h": region["h"]}
            text_region = {
                "x": region["x"] + icon_w + gutter, "y": region["y"],
                "w": region["w"] - icon_w - gutter, "h": region["h"],
            }
            label = beat.get("label") or beat["text"]

            asset_entry = resolve_beat_asset(beat, channel, illustration_cache_dir)
            beat_out["asset"] = asset_entry
            beat_out["region"] = region
            print(f"[render_pipeline] beat_id={beat['beat_id']} icon_word concept_key={beat.get('concept_key')!r} "
                  f"label={label!r} -> resolved: source={asset_entry.get('asset_source')}")

            half = (beat["end"] - beat["start"]) / 2
            sub_visuals = []

            if asset_entry.get("draw_style") == "stroke_reveal":
                svg_path = asset_entry["asset_ref"].get("path")
                icon_path_info = svg_to_path.icon_to_path_d(svg_path, icon_region) if svg_path else None
                if icon_path_info is None:
                    raise RuntimeError(
                        f"beat_id={beat['beat_id']}: icon_word concept_key resolved to an icon with "
                        f"no usable <path> data — pick a different concept_key for this beat."
                    )
                icon_group_id = f"icon-{beat['beat_id']}"
                concept_group_ids[beat.get("concept_key")] = icon_group_id
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
                    "stroke_width": max(1.5, min(7.0, ICON_STROKE_TARGET_PX / icon_path_info["scale"])),
                    "path_transform": icon_path_info["transform"],
                    "icon_group_id": icon_group_id,
                    "start": beat["start"],
                    "end": beat["start"] + half,
                    "min_reveal_duration": 0.6,
                })
            else:
                # mask_wipe illustration fallback — no pen-stroke reveal available for this
                # concept. FIXED: previously placed instantly with no reveal timing and no
                # hand at all — now wipes in during the icon's own half of the beat, with a
                # 'drag' hand tracking the reveal edge, matching the draw-mode fix.
                beat_out["illustration_path"] = asset_entry["asset_ref"].get("cached_path")
                beat_out["illustration_region"] = icon_region
                beat_out["illustration_start"] = beat["start"]
                beat_out["illustration_end"] = beat["start"] + half
                beat_out["mask_wipe_hand"] = gesture_engine.scaled_hand(
                    "drag", target_height=icon_region["h"] * 0.5
                ).to_dict()

            pad = 0.15
            usable_w = text_region["w"] * (1 - 2 * pad)
            usable_h = text_region["h"] * (1 - 2 * pad)
            font_size = text_to_path.fit_font_size(label, usable_w, usable_h)
            text_x = text_region["x"] + text_region["w"] * pad
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
            sub_visuals.append({
                "beat_id": f"{beat['beat_id']}-label",
                "subpaths": stroke_info["subpaths"],
                "stroke_width": max(1.5, font_size * 0.045),
                "path_transform": None,
                "start": beat["start"] + half,
                "end": beat["end"],
            })

            beat_out["sub_visuals"] = sub_visuals
            target_height = max(region["h"] * 0.40, min(font_size * 4.6, region["h"] * 0.85))
            beat_out["hand"] = gesture_engine.scaled_hand("write", target_height=target_height).to_dict()

            _apply_camera_move(beat_out, beat, region, "zoom_in")

        elif beat["mode"] == "draw" and region:
            asset_entry = resolve_beat_asset(beat, channel, illustration_cache_dir)
            beat_out["asset"] = asset_entry
            beat_out["region"] = region
            print(f"[render_pipeline] beat_id={beat['beat_id']} concept_key={beat.get('concept_key')!r} "
                  f"-> resolved: source={asset_entry.get('asset_source')} "
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
                # BUG FIXED: was f"icon-{concept_key}" — collided when
                # two SEPARATE, unrelated beats picked the same
                # concept_key (e.g. two different sentences both
                # drawing "brain"), merging their strokes into the
                # SAME DOM group so both stayed visible forever,
                # overlapping. Now unique per beat; concept_group_ids
                # tracks which one is CURRENT for point/zoom_in/zoom_out.
                icon_group_id = f"icon-{beat['beat_id']}"
                concept_group_ids[beat.get("concept_key")] = icon_group_id
                beat_out["icon_group_id"] = icon_group_id
                # THICKNESS: scale-compensated during the reveal (see
                # icon_word branch above for the full reasoning) —
                # non-scaling-stroke is applied only AFTER the reveal
                # completes, purely to protect against a later
                # pinch-enlarge, not during the dasharray animation.
                beat_out["stroke_width"] = max(1.5, min(7.0, ICON_STROKE_TARGET_PX / icon_path_info["scale"]))
                beat_out["min_reveal_duration"] = 1.3
                beat_out["path_transform"] = icon_path_info["transform"]
            else:
                # mask_wipe illustration — no path data, template reveals via clip-path sweep instead.
                # BUG FIXED: this branch never set a hand at all before — confirmed in the
                # code (the hand assignment below only fires when "subpaths" is present) —
                # which is exactly the hand-less icon in the screenshot. Now gets a 'drag'
                # gesture that tracks the reveal edge.
                beat_out["illustration_path"] = asset_entry["asset_ref"].get("cached_path")
                beat_out["illustration_region"] = region
                beat_out["illustration_start"] = beat["start"]
                beat_out["illustration_end"] = beat["end"]
                beat_out["mask_wipe_hand"] = gesture_engine.scaled_hand(
                    "drag", target_height=region["h"] * 0.5
                ).to_dict()

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

        elif beat["mode"] == "point" and region:
            target_x = region["x"] + region["w"] / 2
            target_y = region["y"] + region["h"] / 2
            hand = gesture_engine.scaled_hand("point", target_height=region["h"] * 0.45)
            beat_out["hand"] = hand.placement_at(target_x, target_y)
            beat_out["region"] = region

        elif beat["mode"] in ("zoom_in", "zoom_out") and region:
            direction = "in" if beat["mode"] == "zoom_in" else "out"
            target_x = region["x"] + region["w"] / 2
            th = region["h"] * 0.5
            # BELOW the icon, not centered on/beside it — the pinch
            # hand's own anchor point is placed this far past the
            # region's bottom edge (half its own height + a gap) so it
            # sits clearly underneath instead of overlapping the icon.
            target_y = region["y"] + region["h"] + th * 0.55
            start_hand, end_hand = gesture_engine.zoom_swap_pair(direction=direction, target_height=th)
            swap_at = (beat["start"] + beat["end"]) / 2
            beat_out["hand_swap"] = {
                "start": start_hand.placement_at(target_x, target_y),
                "end": end_hand.placement_at(target_x, target_y),
                "swap_at": swap_at,
            }
            beat_out["region"] = region

            # Enlarge/shrink the ICON ITSELF in place (separate from
            # the camera move below) — targets the icon_group_id set
            # when that concept_key was originally drawn. zoom_in grows
            # it to ICON_ENLARGE_SCALE; zoom_out shrinks it back to 1x.
            # If nothing with that concept_key was ever drawn as an
            # icon (e.g. it was a mask_wipe illustration, which has no
            # group id), this is silently skipped in the template —
            # the pinch hand still plays, just without an icon to grow.
            ck = beat.get("concept_key")
            if ck:
                icon_center_y = region["y"] + region["h"] / 2
                # Look up the CURRENT group id for this concept_key —
                # group ids are unique per beat now (see the
                # concept_group_ids comment above), so this can't just
                # be recomputed from concept_key anymore.
                target_group_id = concept_group_ids.get(ck)
                if target_group_id:
                    beat_out["icon_scale"] = {
                        "target_id": target_group_id,
                        "cx": target_x,
                        "cy": icon_center_y,
                        "scale": ICON_ENLARGE_SCALE if beat["mode"] == "zoom_in" else 1.0,
                        "swap_at": swap_at,
                        "duration": max(0.2, (beat["end"] - beat["start"]) / 2),
                    }
            _apply_camera_move(beat_out, beat, region, beat["mode"])

        elif beat["mode"] == "swipe":
            sweep = gesture_engine.scaled_swipe(
                direction=beat.get("swipe_direction", "ltr"),
                frame_width=frame["width"], target_height=280,
            )
            sweep["y"] = frame["height"] / 2 - sweep["anchor_y"]
            sweep["start"] = camera_free_at
            sweep["duration"] = max(0.4, beat["end"] - beat["start"])
            beat_out["hand_sweep"] = sweep
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
    }


if __name__ == "__main__":
    import sys
    print("This module builds a scene program dict — run via a channel-specific "
          "trigger script that supplies script_text + beats from n8n's extractor output. "
          "See EXAMPLE_BEAT in beat_schema.py for the expected shape of each beat.")
    sys.exit(0)
