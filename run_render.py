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

# Output quality/consistency knobs — one place to tune, applied to BOTH
# mux paths below (with sfx and narration-only fallback) so they never
# drift out of sync with each other.
VIDEO_CRF = "18"        # x264 CRF: lower = higher quality + bigger file. libx264's
                         # own default is 23 ("fine but visibly compressed" for a
                         # screen-recorded source); 18 is the commonly cited
                         # threshold for visually near-lossless output.
VIDEO_PRESET = "slow"   # better compression efficiency at the SAME CRF than the
                         # default "medium" — the cost is longer encode time, an
                         # easy trade to make running in CI rather than interactively.
OUTPUT_FPS = "60"       # Raised from 30 per direct feedback ("hand/camera movement
                         # has started being a bit choppy"). This forces a CONSISTENT
                         # output frame rate regardless of any minor timing variance in
                         # Playwright's own screen-capture (which isn't guaranteed
                         # constant-fps) — this is what actually addresses perceived
                         # motion "smoothness", separate from per-frame crispness
                         # (CRF/preset above). HONEST CAVEAT, not silently decided for
                         # you: this retimes/duplicates whatever Playwright actually
                         # captured — it cannot invent motion information Playwright's
                         # own capture didn't record. If the source capture itself is
                         # genuinely variable-frame-rate (a real possibility with
                         # screen-recording vs. a true frame-by-frame PNG dump — see
                         # this file's own top-of-file TBD note), raising this number
                         # helps but may not fully resolve choppiness on its own; the
                         # structural fix at that point would be switching to the
                         # frame-dump approach the docstring above already flags as a
                         # future option, which is a bigger architecture change than
                         # this one-line knob and shouldn't be done without asking first.
AUDIO_BITRATE = "192k"  # up from ffmpeg's own aac default (~128k).

# Per-file sfx volume — added because a single flat volume for every
# cue (the old behavior) meant bumping one sound's loudness (click.mp3,
# requested at 3.5) would have also blasted drawing/wrighting/woosh
# up to the same level, which was never asked for. SFX_VOLUME_DEFAULT
# preserves the exact previous behavior (2) for every file NOT listed
# here, so drawing.mp3/wrighting.mp3/woosh.mp3 are completely unchanged.
SFX_VOLUME_DEFAULT = 2
SFX_VOLUME_BY_FILE = {
    "click.mp3": 3.5,
}


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
    _mux_audio_with_sfx(video_no_audio, scene["audio_path"], scene.get("sound_cues", []), args.out)
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
        "-c:v", "libx264", "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET,
        "-pix_fmt", "yuv420p", "-r", OUTPUT_FPS,
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-shortest", out_path,
    ], check=True, capture_output=True)


# soundeffect/ lives at the repo root (sibling of tools/), matching
# drawing.mp3 / wrighting.mp3 / woosh.mp3 as uploaded.
SOUNDEFFECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "soundeffect")


def _mux_audio_with_sfx(video_path: str, narration_path: str, sound_cues: list, out_path: str):
    """Mixes drawing/writing/woosh/click sound-effect cues (from
    render_pipeline.py's sound_cues, each an absolute-time cue) into
    the narration track, muxed with the video. NEVER the reason a
    render fails — falls back to the plain, already-proven _mux_audio
    (narration only) at every possible failure point: no cues, no
    soundeffect/ folder, a missing individual file, or the ffmpeg
    mix itself erroring out for any reason.
    """
    if not sound_cues or not os.path.isdir(SOUNDEFFECT_DIR):
        print("[run_render] no sound cues or soundeffect/ folder missing — narration-only mux")
        _mux_audio(video_path, narration_path, out_path)
        return

    valid_cues = []
    for cue in sound_cues:
        sfx_path = os.path.join(SOUNDEFFECT_DIR, cue["file"])
        if os.path.isfile(sfx_path):
            valid_cues.append({
                "path": sfx_path, "start": max(0.0, cue["start"]),
                # Real duration of the visual action this sound accompanies —
                # if present, the clip gets TRIMMED to it so the sound stops
                # exactly when the drawing/writing/camera-move stops, instead
                # of playing the whole file through regardless.
                "duration": cue.get("duration"),
                "file": cue["file"],
            })
        else:
            print(f"[run_render] sound cue file not found, skipping: {sfx_path}")

    if not valid_cues:
        print("[run_render] no valid sound-effect files found — narration-only mux")
        _mux_audio(video_path, narration_path, out_path)
        return

    inputs = ["-i", video_path, "-i", narration_path]
    filter_parts = []
    mix_labels = ["[1:a]"]  # narration, kept at full volume
    for i, cue in enumerate(valid_cues):
        inputs += ["-i", cue["path"]]
        input_idx = i + 2  # 0=video, 1=narration, 2.. = sfx clips in order
        delay_ms = int(cue["start"] * 1000)
        label = f"[sfx{i}]"
        # Per-file volume (see SFX_VOLUME_BY_FILE above) — click.mp3 at
        # 3.5, every other cue file unchanged at the original flat 2.
        # NOTE (unchanged from before): values above 1.0 amplify beyond
        # the clip's original recorded level, not just "louder relative
        # to narration" — ffmpeg's volume filter doesn't limit/compress,
        # so if a cue's source audio already has hot peaks, amplifying
        # it can clip/distort on those peaks specifically. Narration
        # itself is untouched (still full level via normalize=0), so
        # this only ever affects the sfx layer. If click.mp3 sounds
        # harsh/crackly on a real render rather than just "loud", back
        # SFX_VOLUME_BY_FILE["click.mp3"] off (e.g. 2.5-3.0) rather than
        # pushing it higher.
        vol = SFX_VOLUME_BY_FILE.get(cue["file"], SFX_VOLUME_DEFAULT)
        trim = f"atrim=0:{cue['duration']:.3f}," if cue.get("duration") else ""
        filter_parts.append(f"[{input_idx}:a]{trim}adelay={delay_ms}|{delay_ms},volume={vol}{label}")
        mix_labels.append(label)

    mix_inputs = "".join(mix_labels)
    # normalize=0: amix normally divides overall volume by input count
    # to avoid clipping, which would make the narration itself quieter
    # every time a new sfx cue is added — normalize=0 keeps narration
    # at its own original level, with sfx already pre-attenuated above.
    filter_complex = (
        ";".join(filter_parts)
        + f";{mix_inputs}amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0:normalize=0[aout]"
    )

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "libx264", "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET,
        "-pix_fmt", "yuv420p", "-r", OUTPUT_FPS,
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-shortest", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[run_render] muxed with {len(valid_cues)} sound-effect cue(s)")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore")[-800:] if e.stderr else ""
        print(f"[run_render] sound-effect mix failed, falling back to narration-only mux: {stderr}")
        _mux_audio(video_path, narration_path, out_path)


if __name__ == "__main__":
    main()
