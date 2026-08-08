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
from camera import Camera, CameraMove
from beat_schema import validate_batch, load_vocabulary, GESTURE_FOR_MODE

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


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
def _layout_board(beats: List[dict]) -> dict:
    """Simple left-to-right, wrapping flow layout: each draw/write beat
    gets a slot on the world-space board in script order. This is
    intentionally the simplest thing that works — deterministic,
    debuggable, no packing algorithm. Replace with something smarter
    (e.g. grouping related concepts spatially) once real scripts show
    where a flow layout looks awkward, not before.
    """
    SLOT_W, SLOT_H = 500, 400
    COLS = 6
    MARGIN = 100

    layout = {}
    slot_i = 0
    for beat in beats:
        if beat["mode"] not in ("draw", "write"):
            continue
        col = slot_i % COLS
        row = slot_i // COLS
        x = MARGIN + col * SLOT_W
        y = MARGIN + row * SLOT_H
        layout[beat["beat_id"]] = {"x": x, "y": y, "w": SLOT_W - 80, "h": SLOT_H - 80}
        slot_i += 1
    return layout


# ══════════════════════════════════════════════════════════════════
# Scene program assembly
# ══════════════════════════════════════════════════════════════════
def build_scene_program(script_text: str, beats: List[dict], channel: str,
                         voice: str = None, illustration_cache_dir: str = "/tmp/illustration_cache") -> dict:
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

    board_layout = _layout_board(timed_beats)

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
            from camera import region_for_bbox, _fit_aspect
            first_region = _fit_aspect(region_for_bbox(r, padding=60))
            break

    cam = Camera(start_view=first_region)
    t_cursor = 0.0
    scene_beats = []

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
            # Text mode: real glyph paths, computed deterministically in
            # Python (see text_to_path.py) — no font loading or generative
            # step happens in the browser at render time.
            pad = 0.15
            usable_w = region["w"] * (1 - 2 * pad)
            usable_h = region["h"] * (1 - 2 * pad)
            font_size = text_to_path.fit_font_size(beat["text"], usable_w, usable_h)
            path_info = text_to_path.text_to_path_d(
                beat["text"],
                x=region["x"] + region["w"] * pad,
                y=region["y"] + region["h"] * pad,
                font_size=font_size,
            )
            beat_out["path_d"] = path_info["d"]
            beat_out["path_transform"] = None  # already baked into path_d, no wrapper transform needed
            beat_out["region"] = region
            # Actual vertical center of the drawn text, NOT the region's
            # center — the region has padding above/below the text, so
            # tracking region.h/2 put the hand floating well below the
            # letters. ~0.35*font_size approximates the visual mid-height
            # of lowercase-heavy text (between baseline and x-height/ascender).
            text_top_y = region["y"] + region["h"] * pad
            beat_out["text_track_y"] = text_top_y + font_size * 0.35

            # Real per-word Chatterbox timestamps drive reveal pace —
            # NOT a uniform slide across the whole sentence. Slice the
            # global word list to this beat's time window, then zip
            # positionally against text_to_path's word_boundaries (same
            # word count expected since beat.text is a straight slice
            # of the full script). If counts don't match — e.g. a
            # tokenization mismatch between Chatterbox's words and a
            # plain whitespace split — fall back to a uniform reveal
            # rather than producing garbled/misaligned keyframes; this
            # is logged, not silent, so a mismatch is visible in the
            # render log instead of just looking subtly wrong.
            global_words = timing["words"]
            beat_words = [w for w in global_words if beat["start"] - 0.05 <= w["start"] < beat["end"] + 0.05]
            word_boundaries = path_info["word_boundaries"]

            if len(beat_words) == len(word_boundaries) and len(beat_words) > 0:
                beat_out["reveal_keyframes"] = [
                    {"t": max(0.0, gw["start"] - beat["start"]), "x_end": wb["x_end"]}
                    for gw, wb in zip(beat_words, word_boundaries)
                ]
            else:
                print(f"[render_pipeline] beat_id={beat['beat_id']}: word count mismatch "
                      f"(chatterbox={len(beat_words)}, text_to_path={len(word_boundaries)}) "
                      f"— falling back to uniform-speed reveal for this beat")
                beat_out["reveal_keyframes"] = None

            # NOTE: hand tracks the LIVE point on the stroke-reveal path every
            # frame, not a single static target — so we hand the template the
            # fully-resolved (already-scaled) hand data, not a pre-computed
            # placement. The template subtracts anchor_x/anchor_y from the
            # moving pen-tip point each frame. See scene_template.html.
            # target_height proportional to font_size — a real hand holding a
            # pen reads naturally at roughly 2.4x the text height; not a fixed
            # constant, so it stays right-sized whether the region is huge or tiny.
            # Bumped up from 2.4 -> 3.1x font_size (was reading as too small).
            target_height = font_size * 3.1
            beat_out["hand"] = gesture_engine.scaled_hand("write", target_height=target_height).to_dict()

            cam.add(CameraMove(action="zoom_in", region=region, duration=min(1.2, beat["end"] - beat["start"])),
                    start_t=beat["start"])

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
                beat_out["path_transform"] = icon_path_info["transform"]
            else:
                # mask_wipe illustration — no path data, template reveals via clip-path sweep instead.
                beat_out["illustration_path"] = asset_entry["asset_ref"].get("cached_path")

            if "subpaths" in beat_out or "path_d" in beat_out:
                # target_height proportional to region size for icons — a hand
                # tracing a small icon should be noticeably smaller than one
                # writing a full sentence, not a fixed constant either way.
                # Bumped up from 0.5 -> 0.68 (was reading as too small).
                target_height = region["h"] * 0.68
                beat_out["hand"] = gesture_engine.scaled_hand("write", target_height=target_height).to_dict()

            cam.add(CameraMove(action="zoom_in", region=region, duration=min(1.2, beat["end"] - beat["start"])),
                    start_t=beat["start"])

        elif beat["mode"] == "point" and region:
            target_x = region["x"] + region["w"] / 2
            target_y = region["y"] + region["h"] / 2
            hand = gesture_engine.scaled_hand("point", target_height=region["h"] * 0.45)
            beat_out["hand"] = hand.placement_at(target_x, target_y)
            beat_out["region"] = region

        elif beat["mode"] in ("zoom_in", "zoom_out") and region:
            direction = "in" if beat["mode"] == "zoom_in" else "out"
            target_x = region["x"] + region["w"] / 2
            target_y = region["y"] + region["h"] / 2
            th = region["h"] * 0.5
            start_hand, end_hand = gesture_engine.zoom_swap_pair(direction=direction, target_height=th)
            beat_out["hand_swap"] = {
                "start": start_hand.placement_at(target_x, target_y),
                "end": end_hand.placement_at(target_x, target_y),
                "swap_at": (beat["start"] + beat["end"]) / 2,
            }
            beat_out["region"] = region
            cam.add(CameraMove(action=beat["mode"], region=region,
                                duration=min(1.0, beat["end"] - beat["start"])),
                    start_t=beat["start"])

        elif beat["mode"] == "swipe":
            sweep = gesture_engine.scaled_swipe(
                direction=beat.get("swipe_direction", "ltr"),
                frame_width=1920, target_height=280,
            )
            sweep["y"] = 540 - sweep["anchor_y"]
            beat_out["hand_sweep"] = sweep
            cam.add(CameraMove(action="zoom_out", duration=0.6), start_t=beat["start"])

        elif beat["mode"] == "talk":
            beat_out["cutaway_file"] = gesture_engine.cutaway_for_beat(beat["beat_id"])

        scene_beats.append(beat_out)

    return {
        "channel": channel,
        "audio_path": timing["audio_path"],
        "board": {"width": 3840, "height": 2160},
        "camera_keyframes": json.loads(cam.gsap_keyframes_js()),
        "beats": scene_beats,
    }


if __name__ == "__main__":
    import sys
    print("This module builds a scene program dict — run via a channel-specific "
          "trigger script that supplies script_text + beats from n8n's extractor output. "
          "See EXAMPLE_BEAT in beat_schema.py for the expected shape of each beat.")
    sys.exit(0)
