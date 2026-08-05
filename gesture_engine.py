"""
Gesture engine.

Turns a beat's `mode` + a resolved world-space target point into a
concrete hand-placement instruction: which PNG, where to position its
top-left corner so the CALIBRATED ANCHOR (assets/hands/hand-gestures.json)
lands exactly on the target, and — for pinch pairs — the two-frame swap
sequence.

This module does no rendering itself; it hands render_pipeline.py a
plain dict the Playwright/GSAP template consumes directly.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

HANDS_JSON_PATH = os.path.join(os.path.dirname(__file__), "assets", "hands", "hand-gestures.json")

with open(HANDS_JSON_PATH) as f:
    _RAW = json.load(f)

GESTURES = {k: v for k, v in _RAW.items() if not k.startswith("_")}


@dataclass
class HandPlacement:
    file: str
    top_left_x: float   # world-space top-left corner for the PNG so its anchor lands on target
    top_left_y: float
    native_w: int
    native_h: int
    rotation_deg: float = 0.0

    def to_dict(self):
        return {
            "file": self.file, "x": self.top_left_x, "y": self.top_left_y,
            "w": self.native_w, "h": self.native_h, "rotation": self.rotation_deg,
        }


def _placement_for_point_gesture(gesture_name: str, target_x: float, target_y: float,
                                  rotation_deg: float = 0.0) -> HandPlacement:
    g = GESTURES[gesture_name]
    if g["type"] != "point":
        raise ValueError(f"gesture '{gesture_name}' is type '{g['type']}', not 'point'")
    anchor = g["anchor"]
    w, h = g["native_size"]
    return HandPlacement(
        file=g["file"],
        top_left_x=target_x - anchor["x"],
        top_left_y=target_y - anchor["y"],
        native_w=w, native_h=h,
        rotation_deg=rotation_deg,
    )


def place_write(target_x: float, target_y: float, path_angle_deg: float = 0.0) -> HandPlacement:
    """target = current point on the stroke-reveal path (see
    camera/render's getPointAtLength equivalent). path_angle_deg =
    tangent direction of the path at that point, so the hand leans
    into the stroke direction rather than staying axis-aligned."""
    return _placement_for_point_gesture("write", target_x, target_y, rotation_deg=path_angle_deg)


def place_point(target_x: float, target_y: float) -> HandPlacement:
    return _placement_for_point_gesture("point", target_x, target_y)


def place_drag(target_x: float, target_y: float) -> HandPlacement:
    return _placement_for_point_gesture("drag", target_x, target_y)


def zoom_swap_pair(target_x: float, target_y: float, direction: str = "in"):
    """Returns (start_placement, end_placement) for the 2-frame pinch
    swap. direction='in' => pinch_in (fingers closed) -> pinch_out
    (fingers spread), selling an enlarge. direction='out' => reverse."""
    if direction == "in":
        start_name, end_name = "pinch_in", "pinch_out"
    elif direction == "out":
        start_name, end_name = "pinch_out", "pinch_in"
    else:
        raise ValueError("direction must be 'in' or 'out'")
    return (
        _placement_for_point_gesture(start_name, target_x, target_y),
        _placement_for_point_gesture(end_name, target_x, target_y),
    )


def place_erase(region_center_x: float, region_center_y: float) -> dict:
    g = GESTURES["erase"]
    if g["type"] != "area":
        raise ValueError("erase gesture must be type 'area'")
    zone = g["zone"]
    zone_cx = zone["x"] + zone["w"] / 2
    zone_cy = zone["y"] + zone["h"] / 2
    w, h = g["native_size"]
    return {
        "file": g["file"],
        "x": region_center_x - zone_cx,
        "y": region_center_y - zone_cy,
        "w": w, "h": h,
        "sweep": True,
    }


def place_swipe(direction: str, frame_width: float, frame_height: float, y_center: float) -> dict:
    """Full-canvas clear. direction: 'ltr' or 'rtl'. Returns start/end
    x offsets for a sweep tween across the whole visible frame width;
    'rtl' mirrors the sprite (handled by the render template, not here
    — this just flags the intent)."""
    g = GESTURES["swipe"]
    w, h = g["native_size"]
    anchor = g["anchor"]
    if direction == "ltr":
        start_x, end_x = -w, frame_width
    else:
        start_x, end_x = frame_width, -w
    return {
        "file": g["file"], "mirror": direction == "rtl",
        "start_x": start_x, "end_x": end_x,
        "y": y_center - anchor["y"],
        "w": w, "h": h,
    }


def cutaway_for_beat(beat_index: int) -> str:
    """Alternates talk/talk2/shrug for visual variety across
    consecutive cutaway beats rather than always using the same pose."""
    options = ["talk", "talk2", "shrug"]
    return GESTURES[options[beat_index % len(options)]]["file"]
