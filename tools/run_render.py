"""
CLI entrypoint: payload.json (script_text, beats, voice) -> scene
program -> Playwright renders scene_template.html frame-by-frame (or
screen-recorded, TBD — see note below) -> muxed with the Chatterbox
audio via ffmpeg -> output.mp4.

FIRST PASS — the scene program assembly (render_pipeline.py) is
complete and real. The Playwright capture step below is a working
skeleton: it loads the template, drives the GSAP timeline to
`duration` seconds, and screen-records via Playwright's built-in video
capture. This trades a bit of encode quality for a MUCH simpler first
version than a full frame-by-frame PNG dump + ffmpeg concat (which is
what the What's The Difference pipeline does, for CRF/FPS control).
Swap to frame-dump if/when render quality testing shows video capture
isn't crisp enough — flagged here rather than silently deciding for you.
"""
import argparse
import json
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import render_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.payload) as f:
        payload = json.load(f)

    scene = render_pipeline.build_scene_program(
        script_text=payload["script_text"],
        beats=payload["beats"],
        channel=args.channel,
        voice=payload.get("voice"),
        orientation=payload.get("orientation", "landscape"),
    )

    scene_path = "/tmp/scene_program.json"
    with open(scene_path, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"[run_render] scene program written to {scene_path} "
          f"({len(scene['beats'])} beats, orientation={scene['orientation']}, "
          f"frame={scene['frame']['width']}x{scene['frame']['height']})")

    duration = scene["beats"][-1]["end"] if scene["beats"] else 5.0
    video_no_audio = "/tmp/video_no_audio.webm"

    _capture_with_playwright(scene_path, duration, video_no_audio, scene["frame"])
    _mux_audio(video_no_audio, scene["audio_path"], args.out)
    print(f"[run_render] done -> {args.out}")


def _capture_with_playwright(scene_path: str, duration: float, out_path: str, frame: dict):
    from playwright.sync_api import sync_playwright

    template_path = os.path.join(os.path.dirname(__file__), "..", "scene_template.html")
    template_url = f"file://{os.path.abspath(template_path)}"

    with open(scene_path) as f:
        scene_json_str = f.read()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": frame["width"], "height": frame["height"]},
            record_video_dir=os.path.dirname(out_path),
            record_video_size={"width": frame["width"], "height": frame["height"]},
        )
        page = context.new_page()

        # Surface browser-side console/errors in the GitHub Actions log —
        # a silently-failing fetch() is exactly what produced a blank
        # first render with no indication anything was wrong. Never
        # again: any future JS error is now loud in the workflow log.
        page.on("console", lambda msg: print(f"[browser:{msg.type}] {msg.text}"))
        page.on("pageerror", lambda exc: print(f"[browser:pageerror] {exc}"))

        # Inject the scene data directly as a page global BEFORE
        # navigation, rather than having the page fetch() it — fetching
        # a file:// path from a file:// page hits Chromium's
        # cross-directory file-access sandboxing and fails silently.
        # add_init_script runs before any of the page's own scripts on
        # every navigation, so window.__SCENE_DATA__ is guaranteed to
        # already exist when scene_template.html's boot code runs.
        page.add_init_script(f"window.__SCENE_DATA__ = {scene_json_str};")

        page.goto(template_url)
        page.wait_for_timeout(int((duration + 1) * 1000))
        context.close()
        browser.close()

    # Playwright names the file itself — find and rename to the requested out_path.
    for fname in os.listdir(os.path.dirname(out_path)):
        if fname.endswith(".webm") and fname != os.path.basename(out_path):
            os.rename(os.path.join(os.path.dirname(out_path), fname), out_path)
            break


def _mux_audio(video_path: str, audio_path: str, out_path: str):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-c:v", "libx264", "-c:a", "aac", "-shortest", out_path,
    ], check=True, capture_output=True)


if __name__ == "__main__":
    main()
