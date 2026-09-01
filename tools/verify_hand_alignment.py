"""
Verify that anchor_frac actually lands where it should — visually,
automatically, every time. This is what makes the anchor system
"bulletproof" rather than "we eyeballed it once and hoped": run this
after calibrating (or re-calibrating) any hand PNG, and it draws a
bright red crosshair exactly where the code THINKS the anchor is,
directly onto a copy of the real image. If the crosshair isn't
sitting exactly on the pen tip / fingertip / contact point, the
anchor_frac value is wrong — fix it and re-run this until it matches.

This is a genuinely different guarantee than the earlier grid-overlay
calibration step: that was a human reading coordinates off a ruler
(error-prone). This is the ACTUAL VALUE STORED IN THE CONFIG, rendered
back onto the image, so what you're checking is exactly what the
renderer will use — no gap between "what I calibrated" and "what's
actually wired up."

Usage:
    python3 tools/verify_hand_alignment.py
    python3 tools/verify_hand_alignment.py --gesture write --height 300
Outputs annotated copies into tools/verification_preview/.
"""
import argparse
import json
import os
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANDS_DIR = os.path.join(REPO_ROOT, "assets", "hands")
HANDS_JSON = os.path.join(HANDS_DIR, "hand-gestures.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "verification_preview")


def draw_crosshair(draw: ImageDraw.Draw, x: float, y: float, size: int = 30, color=(255, 0, 0, 255)):
    draw.line([(x - size, y), (x + size, y)], fill=color, width=4)
    draw.line([(x, y - size), (x, y + size)], fill=color, width=4)
    draw.ellipse([x - 8, y - 8, x + 8, y + 8], outline=color, width=3)


def verify_gesture(gesture_name: str, config: dict, target_height: float = None) -> dict:
    entry = config[gesture_name]
    path = os.path.join(HANDS_DIR, entry["file"])
    if not os.path.exists(path):
        return {"gesture": gesture_name, "status": "MISSING_FILE", "path": path}

    img = Image.open(path).convert("RGBA")
    native_w, native_h = img.size

    declared_w, declared_h = entry.get("native_size", [native_w, native_h])
    size_mismatch = (declared_w, declared_h) != (native_w, native_h)

    draw = ImageDraw.Draw(img)
    results = {"gesture": gesture_name, "file": entry["file"],
               "actual_size": [native_w, native_h], "declared_size": [declared_w, declared_h],
               "size_mismatch": size_mismatch}

    if "anchor_frac" in entry:
        ax = entry["anchor_frac"]["x"] * native_w
        ay = entry["anchor_frac"]["y"] * native_h
        draw_crosshair(draw, ax, ay)
        results["anchor_px_in_native_image"] = [round(ax, 1), round(ay, 1)]

        # Also verify at a SCALED size (what actually gets rendered) —
        # confirms the scaling math independently of the native image.
        if target_height:
            import sys
            sys.path.insert(0, REPO_ROOT)
            import gesture_engine
            sh = gesture_engine.scaled_hand(gesture_name, target_height=target_height)
            results["scaled_at_height"] = target_height
            results["scaled_anchor"] = [round(sh.anchor_x, 1), round(sh.anchor_y, 1)]
            results["scaled_size"] = [round(sh.w, 1), round(sh.h, 1)]

    if "zone_frac" in entry:
        zf = entry["zone_frac"]
        x0, y0 = zf["x"] * native_w, zf["y"] * native_h
        x1, y1 = x0 + zf["w"] * native_w, y0 + zf["h"] * native_h
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=4)
        results["zone_px_in_native_image"] = [round(v, 1) for v in [x0, y0, x1, y1]]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{gesture_name}_verify.png")
    img.save(out_path)
    results["preview_path"] = out_path
    results["status"] = "OK"
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", default=None, help="Single gesture to verify (default: all)")
    parser.add_argument("--height", type=float, default=220.0, help="Target render height to test scaling against")
    args = parser.parse_args()

    with open(HANDS_JSON) as f:
        config = json.load(f)
    gestures = {k: v for k, v in config.items() if not k.startswith("_")}

    names = [args.gesture] if args.gesture else list(gestures.keys())

    print(f"{'gesture':<12} {'status':<14} {'size_ok':<8} {'anchor_px':<18} {'scaled_anchor':<18}")
    for name in names:
        if name not in gestures:
            print(f"{name}: not found in hand-gestures.json")
            continue
        r = verify_gesture(name, gestures, target_height=args.height)
        if r["status"] != "OK":
            print(f"{name:<12} {r['status']:<14} -- MISSING: {r.get('path')}")
            continue
        size_ok = "NO" if r["size_mismatch"] else "yes"
        anchor = r.get("anchor_px_in_native_image", r.get("zone_px_in_native_image", "-"))
        scaled = r.get("scaled_anchor", "-")
        print(f"{name:<12} {r['status']:<14} {size_ok:<8} {str(anchor):<18} {str(scaled):<18}")
        if r["size_mismatch"]:
            print(f"  ⚠ native_size in hand-gestures.json ({r['declared_size']}) doesn't match "
                  f"actual PNG size ({r['actual_size']}) — anchor_frac math will be WRONG until fixed")

    print(f"\nAnnotated images written to {OUT_DIR}/ — open each and confirm the red crosshair "
          f"sits exactly on the intended point (pen tip, fingertip, etc). If it doesn't, the "
          f"anchor_frac value in hand-gestures.json is wrong, not the code using it.")
