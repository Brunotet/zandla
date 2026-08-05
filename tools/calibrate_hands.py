"""
Anchor calibration helper for hand gesture PNGs.

Draws a labeled coordinate grid over each hand image so exact anchor
points (pen tip, pinch point, etc.) can be read off visually and
recorded in assets/hands/hand-gestures.json. This is a ONE-TIME build
step — never runs at render time, no runtime cost, no dependency on
this script after calibration is committed.

Usage:
    python tools/calibrate_hands.py
Outputs annotated copies into tools/calibration_preview/.
"""
import os
from PIL import Image, ImageDraw

HANDS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "hands")
OUT_DIR = os.path.join(os.path.dirname(__file__), "calibration_preview")
GRID_STEP = 50  # px between gridlines
LABEL_STEP = 100  # px between numeric labels (avoid clutter)


def annotate(path: str, out_path: str):
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for x in range(0, w, GRID_STEP):
        color = (255, 0, 0, 160) if x % LABEL_STEP == 0 else (255, 0, 0, 60)
        draw.line([(x, 0), (x, h)], fill=color, width=1)
        if x % LABEL_STEP == 0:
            draw.text((x + 2, 2), str(x), fill=(255, 0, 0, 255))

    for y in range(0, h, GRID_STEP):
        color = (0, 120, 255, 160) if y % LABEL_STEP == 0 else (0, 120, 255, 60)
        draw.line([(0, y), (w, y)], fill=color, width=1)
        if y % LABEL_STEP == 0:
            draw.text((2, y + 2), str(y), fill=(0, 120, 255, 255))

    combined = Image.alpha_composite(img, overlay)
    combined.save(out_path)


def bounding_box_info(path: str):
    """Reports the alpha bounding box + topmost/leftmost/rightmost
    non-transparent pixel — useful starting candidates for anchors
    like a pen tip or fingertip, which are almost always the topmost
    point of the hand's silhouette."""
    img = Image.open(path).convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if not bbox:
        return None
    px = alpha.load()
    x0, y0, x1, y1 = bbox
    topmost = None
    for y in range(y0, y1):
        for x in range(x0, x1):
            if px[x, y] > 10:
                topmost = (x, y)
                break
        if topmost:
            break
    return {"bbox": bbox, "topmost_point": topmost, "image_size": img.size}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname in sorted(os.listdir(HANDS_DIR)):
        if not fname.lower().endswith(".png"):
            continue
        src = os.path.join(HANDS_DIR, fname)
        dst = os.path.join(OUT_DIR, fname)
        annotate(src, dst)
        info = bounding_box_info(src)
        print(f"{fname}: {info}")
