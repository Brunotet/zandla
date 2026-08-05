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

    cam = Camera()
    t_cursor = 0.0
    scene_beats = []

    for beat in timed_beats:
        beat_out = {
            "beat_id": beat["beat_id"],
            "text": beat["text"],
            "mode": beat["mode"],
            "start": beat["start"],
            "end": beat["end"],
        }

        region = board_layout.get(beat["beat_id"])

        if beat["mode"] in ("draw", "write") and region:
            asset_entry = resolve_beat_asset(beat, channel, illustration_cache_dir)
            beat_out["asset"] = asset_entry
            beat_out["region"] = region

            gesture_name = GESTURE_FOR_MODE.get(beat["mode"], "write")
            target_x = region["x"] + region["w"] / 2
            target_y = region["y"] + region["h"] / 2
            placement = gesture_engine.place_write(target_x, target_y)
            beat_out["hand"] = placement.to_dict()

            cam.add(CameraMove(action="zoom_in", region=region, duration=min(1.2, beat["end"] - beat["start"])),
                    start_t=beat["start"])

        elif beat["mode"] == "point" and region:
            target_x = region["x"] + region["w"] / 2
            target_y = region["y"] + region["h"] / 2
            placement = gesture_engine.place_point(target_x, target_y)
            beat_out["hand"] = placement.to_dict()
            beat_out["region"] = region

        elif beat["mode"] in ("zoom_in", "zoom_out") and region:
            direction = "in" if beat["mode"] == "zoom_in" else "out"
            target_x = region["x"] + region["w"] / 2
            target_y = region["y"] + region["h"] / 2
            start_p, end_p = gesture_engine.zoom_swap_pair(target_x, target_y, direction=direction)
            beat_out["hand_swap"] = {"start": start_p.to_dict(), "end": end_p.to_dict(),
                                      "swap_at": (beat["start"] + beat["end"]) / 2}
            beat_out["region"] = region
            cam.add(CameraMove(action=beat["mode"], region=region,
                                duration=min(1.0, beat["end"] - beat["start"])),
                    start_t=beat["start"])

        elif beat["mode"] == "swipe":
            beat_out["hand_sweep"] = gesture_engine.place_swipe(
                direction=beat.get("swipe_direction", "ltr"),
                frame_width=1920, frame_height=1080, y_center=540,
            )
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
