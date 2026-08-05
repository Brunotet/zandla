"""
World-space camera for the hand-draw pipeline.

Design decision (locked in after brainstorming): the canvas is ONE large
fixed SVG coordinate space ("the board"). Content — icons, drawn text,
illustrations — gets placed once at a world-space (x, y) and never
moves. The "camera" is just which rectangular window into that space
is currently visible, expressed as an SVG viewBox string.

Why this over transforming the content itself: hand-gesture anchors
(assets/hands/hand-gestures.json) are calibrated once and stay valid
at ANY zoom level, because gesture positions are also world-space
coordinates — the camera window changing doesn't change where
anything "is", only what's currently in frame. One coordinate system,
used everywhere (gestures, icons, illustrations, camera).

Fixed-layer UI (captions, logo, progress bar) does NOT go through this
module — that's plain CSS position:fixed outside the SVG, deliberately
untouched by camera moves.

This module only computes the viewBox value per frame/tick. It doesn't
render anything itself — render_pipeline.py reads `Camera.state` (or
calls `Camera.value_at(t)` for a pre-baked timeline) and writes it into
the SVG element's viewBox attribute via Playwright.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import math

# Default board size — generous world-space canvas. Tune once you know
# how spread out a typical multi-beat script's content ends up being;
# this is a starting point, not a hard constraint (content can exceed
# it, the camera just won't have anywhere "wide" to pull back to).
BOARD_WIDTH = 3840
BOARD_HEIGHT = 2160

# Default fully-zoomed-out view — matches a 16:9 frame centered on the board.
DEFAULT_VIEW = {"x": 0, "y": 0, "w": 1920, "h": 1080}


@dataclass
class CameraKeyframe:
    t: float                 # seconds, absolute timeline position
    x: float
    y: float
    w: float
    h: float
    ease: str = "power2.inOut"


@dataclass
class CameraMove:
    """One beat's camera instruction, before being resolved into
    absolute keyframes. `region` is a world-space rect the camera
    should frame; `action` decides how tightly and how it transitions."""
    action: str               # "zoom_in" | "zoom_out" | "pan" | "hold"
    region: Optional[dict] = None      # {"x","y","w","h"} in world-space
    padding: float = 60.0     # world-space px of breathing room around region
    duration: float = 1.0
    ease: str = "power2.inOut"


def region_for_bbox(bbox: dict, padding: float = 60.0, min_size: float = 200.0) -> dict:
    """Turn a placed element's world-space bounding box into a camera
    region, with padding so the zoom doesn't crop right to the edge,
    and a minimum size so zooming into a tiny icon doesn't produce an
    absurdly tight/jittery frame."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    w = max(w, min_size)
    h = max(h, min_size)
    return {
        "x": x - padding,
        "y": y - padding,
        "w": w + padding * 2,
        "h": h + padding * 2,
    }


def _fit_aspect(region: dict, target_aspect: float = 16 / 9) -> dict:
    """Expands a region on its shorter axis so it matches the output
    aspect ratio without stretching (center-preserving)."""
    w, h = region["w"], region["h"]
    current_aspect = w / h if h else target_aspect
    cx = region["x"] + w / 2
    cy = region["y"] + h / 2

    if current_aspect > target_aspect:
        # too wide relative to target -> grow height
        new_h = w / target_aspect
        new_w = w
    else:
        new_w = h * target_aspect
        new_h = h

    return {"x": cx - new_w / 2, "y": cy - new_h / 2, "w": new_w, "h": new_h}


class Camera:
    """Accumulates a sequence of CameraMove instructions into absolute
    keyframes on a single timeline, starting from DEFAULT_VIEW."""

    def __init__(self, start_view: dict = None):
        self.keyframes: List[CameraKeyframe] = [
            CameraKeyframe(t=0.0, ease="none", **(start_view or DEFAULT_VIEW))
        ]

    def add(self, move: CameraMove, start_t: float):
        prev = self.keyframes[-1]

        if move.action == "hold":
            target = {"x": prev.x, "y": prev.y, "w": prev.w, "h": prev.h}
        elif move.action in ("zoom_in", "pan"):
            if not move.region:
                raise ValueError(f"camera action '{move.action}' requires a region")
            target = _fit_aspect(region_for_bbox(move.region, move.padding))
        elif move.action == "zoom_out":
            target = dict(DEFAULT_VIEW)
        else:
            raise ValueError(f"unknown camera action: {move.action}")

        self.keyframes.append(CameraKeyframe(
            t=start_t + move.duration, ease=move.ease, **target
        ))
        return self

    def value_at(self, t: float) -> dict:
        """Interpolated viewBox at time t. Used by the renderer both
        for a live GSAP-driven build (as start/end keyframe pairs it
        tweens between) and for a pure-Python frame-dump fallback."""
        kfs = self.keyframes
        if t <= kfs[0].t:
            k = kfs[0]
            return {"x": k.x, "y": k.y, "w": k.w, "h": k.h}
        for i in range(1, len(kfs)):
            if t <= kfs[i].t:
                a, b = kfs[i - 1], kfs[i]
                span = b.t - a.t
                progress = 0.0 if span <= 0 else (t - a.t) / span
                progress = _ease(progress, b.ease)
                return {
                    "x": a.x + (b.x - a.x) * progress,
                    "y": a.y + (b.y - a.y) * progress,
                    "w": a.w + (b.w - a.w) * progress,
                    "h": a.h + (b.h - a.h) * progress,
                }
        k = kfs[-1]
        return {"x": k.x, "y": k.y, "w": k.w, "h": k.h}

    def to_viewbox_str(self, view: dict) -> str:
        return f"{view['x']:.2f} {view['y']:.2f} {view['w']:.2f} {view['h']:.2f}"

    def gsap_keyframes_js(self) -> str:
        """Emits a JS array of {t,x,y,w,h,ease} for the Playwright page
        to consume directly with a GSAP timeline — avoids re-deriving
        interpolation logic in JS; Python computed the keyframes, JS
        just tweens between the ones adjacent to playback time."""
        import json
        return json.dumps([
            {"t": k.t, "x": k.x, "y": k.y, "w": k.w, "h": k.h, "ease": k.ease}
            for k in self.keyframes
        ])


def _ease(p: float, kind: str) -> float:
    p = max(0.0, min(1.0, p))
    if kind == "none" or kind == "linear":
        return p
    if kind == "power2.inOut":
        return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2
    if kind == "power2.out":
        return 1 - (1 - p) ** 2
    if kind == "power2.in":
        return p ** 2
    return p
