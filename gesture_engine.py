"""
Gesture engine.

Turns a beat's `mode` + a target world-space point + a TARGET SIZE into
a fully-resolved hand placement: exact world-space width/height and
exact world-space anchor offset, already scaled. This is the
"bulletproof" version — every number the browser needs is computed
once, here, in Python. The browser does zero scaling math; it just
places what it's told. That's what makes this reliable: there is
exactly one place scale factors get applied, so there's no way for a
scale-the-hand step and a scale-the-anchor step to drift out of sync
with each other (that mismatch was the actual root cause of the
"pen tip way off" bug — the hand was being drawn at its raw native
pixel size regardless of how small the surrounding text/icon was).

Anchors are stored as NORMALIZED FRACTIONS (0-1) in hand-gestures.json
(anchor_frac), not raw pixels — see that file's _meta for why. This
module is the ONLY place anchor_frac gets multiplied out to real units.
"""
import json
import os
from dataclasses import dataclass

HANDS_JSON_PATH = os.path.join(os.path.dirname(__file__), "assets", "hands", "hand-gestures.json")

with open(HANDS_JSON_PATH) as f:
    _RAW = json.load(f)

GESTURES = {k: v for k, v in _RAW.items() if not k.startswith("_")}

# Default hand sizing — tunable in ONE place. target_height is in
# world-space units; call sites express their own target relative to
# something meaningful (font size, region size) rather than a fixed
# constant, so the hand stays proportionate to whatever it's touching.
DEFAULT_TARGET_HEIGHT = 220.0


@dataclass
class ScaledHand:
    """Fully-resolved hand placement — everything already in final
    world-space units. Nothing downstream needs to know the native
    pixel size or do any further scaling."""
    file: str
    w: float
    h: float
    anchor_x: float
    anchor_y: float

    def to_dict(self):
        return {"file": self.file, "w": self.w, "h": self.h,
                "anchor_x": self.anchor_x, "anchor_y": self.anchor_y}

    def placement_at(self, target_x: float, target_y: float, rotation: float = 0.0) -> dict:
        """For STATIC placements (point, cutaway-adjacent gestures) where
        the target doesn't move frame-to-frame — pre-computes the
        top-left corner so the anchor lands exactly on target. Beats
        with a moving target (write/draw stroke-reveal) use to_dict()
        instead and let the template compute top-left per frame."""
        return {
            "file": self.file, "w": self.w, "h": self.h,
            "x": target_x - self.anchor_x, "y": target_y - self.anchor_y,
            "rotation": rotation,
        }


def scaled_hand(gesture_name: str, target_height: float = DEFAULT_TARGET_HEIGHT) -> ScaledHand:
    """The single function everything else should call. Preserves the
    native aspect ratio, scales anchor_frac by the SAME factor as the
    image itself — by construction, not a second manual step."""
    g = GESTURES[gesture_name]
    native_w, native_h = g["native_size"]
    scale = target_height / native_h
    final_w = native_w * scale
    final_h = target_height

    if "anchor_frac" in g:
        anchor_x = g["anchor_frac"]["x"] * final_w
        anchor_y = g["anchor_frac"]["y"] * final_h
    else:
        anchor_x = anchor_y = 0.0

    return ScaledHand(file=g["file"], w=final_w, h=final_h, anchor_x=anchor_x, anchor_y=anchor_y)


def zoom_swap_pair(direction: str = "in", target_height: float = DEFAULT_TARGET_HEIGHT):
    """Returns (start_hand, end_hand) for the 2-frame pinch swap, both
    pre-scaled to the same target_height so the swap doesn't also
    change apparent hand size mid-motion."""
    if direction == "in":
        start_name, end_name = "pinch_in", "pinch_out"
    elif direction == "out":
        start_name, end_name = "pinch_out", "pinch_in"
    else:
        raise ValueError("direction must be 'in' or 'out'")
    return scaled_hand(start_name, target_height), scaled_hand(end_name, target_height)


def scaled_erase_zone(target_height: float = DEFAULT_TARGET_HEIGHT) -> dict:
    g = GESTURES["erase"]
    native_w, native_h = g["native_size"]
    scale = target_height / native_h
    final_w = native_w * scale
    zf = g["zone_frac"]
    return {
        "file": g["file"],
        "w": final_w, "h": target_height,
        "zone_cx": (zf["x"] + zf["w"] / 2) * final_w,
        "zone_cy": (zf["y"] + zf["h"] / 2) * target_height,
    }


def scaled_swipe(direction: str, frame_width: float, target_height: float = DEFAULT_TARGET_HEIGHT,
                  frame_height: float = None) -> dict:
    """Full-canvas clear or camera-transition sweep.
    direction: 'ltr'/'rtl' (horizontal) or 'ttb'/'btt' (vertical —
    requires frame_height, used when the camera pans between rows
    rather than between columns)."""
    g = GESTURES["swipe"]
    native_w, native_h = g["native_size"]
    scale = target_height / native_h
    final_w = native_w * scale
    af = g["anchor_frac"]

    if direction in ("ltr", "rtl"):
        anchor_y = af["y"] * target_height
        if direction == "ltr":
            start_x, end_x = -final_w, frame_width
        else:
            start_x, end_x = frame_width, -final_w
        return {
            "file": g["file"], "mirror": direction == "rtl", "axis": "x",
            "start_x": start_x, "end_x": end_x,
            "w": final_w, "h": target_height, "anchor_y": anchor_y,
        }

    if frame_height is None:
        raise ValueError("scaled_swipe: frame_height is required for vertical directions ('ttb'/'btt')")
    anchor_x = af.get("x", 0.5) * final_w
    if direction == "btt":
        start_y, end_y = frame_height, -target_height
    else:  # ttb
        start_y, end_y = -target_height, frame_height
    return {
        "file": g["file"], "mirror": False, "axis": "y",
        "start_y": start_y, "end_y": end_y,
        "w": final_w, "h": target_height, "anchor_x": anchor_x,
    }


def cutaway_for_beat(beat_index: int) -> str:
    """Alternates talk/talk2/shrug for visual variety across
    consecutive cutaway beats rather than always using the same pose."""
    options = ["talk", "talk2", "shrug"]
    return GESTURES[options[beat_index % len(options)]]["file"]
