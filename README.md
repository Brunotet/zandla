# hand-draw-pipeline

Multi-channel hand-drawing/whiteboard render pipeline. n8n decides
*what* to say and *what* to draw; this repo decides *how it's drawn*
and renders the video. Same split as your other channels: n8n owns
script generation + orchestration, this repo owns rendering.

## Architecture at a glance

```
n8n (Gemini script gen, guarded by closed vocabulary)
  -> validates beats against concept-library vocabulary
  -> workflow_dispatch trigger to this repo's GitHub Action
       -> render_pipeline.build_scene_program()
            -> voice_engine: ONE Chatterbox call for the whole script,
               timestamps assigned per beat (reused pattern, not new)
            -> asset_resolver: concept_key -> icon | illustration,
               curated lookup first, CLIP-ranked live search fallback
            -> camera: world-space viewBox timeline across beats
            -> gesture_engine: hand placement per beat, anchor-calibrated
       -> scene_template.html + Playwright renders it
       -> R2 upload (presigned PUT) -> n8n resumeUrl
```

## What's real vs. what's a first pass

**Solid, tested logic (safe to build on):**
- `camera.py` — world-space viewBox camera, keyframe interpolation
- `asset_resolver.py` — CLIP-ranking is a direct reuse of the proven
  What's The Difference pipeline; the icon-search tier is new but
  follows the same never-block-the-render fallback pattern
- `voice_engine.py` — direct reuse of `get_tts_safe`/`assign_timing`,
  same Chatterbox contract as your other channels
- `gesture_engine.py` + `assets/hands/hand-gestures.json` — anchors for
  write/point/drag/pinch_in/pinch_out visually confirmed via grid
  overlay (see `tools/calibrate_hands.py`); erase/swipe anchors are a
  first read, worth double-checking on first real render
- `beat_schema.py` — validation + closed-vocabulary enforcement

**First-pass, needs work before a real render succeeds:**
- `scene_template.html` — mechanism (stroke-reveal, camera, hand
  tracking) is wired end-to-end, but text-to-SVG-path (actual letter
  drawing) is stubbed as a placeholder rectangle. Needs a real
  text-to-path step (e.g. opentype.js) before "write" mode draws
  actual words.
- `tools/upload_r2.py`, `tools/resume_n8n.py` — deliberately left as
  stubs. Port these directly from the tongue-twisters repo rather
  than re-deriving — that pipeline already solved R2 presigned-URL
  SigV4 upload and the resumeUrl callback correctly.
- `_layout_board()` in `render_pipeline.py` — simple grid flow layout.
  Works, but untested against a real multi-beat script; may need
  smarter placement once you see actual output.
- Vendored icon libraries (`vendor/icons/`) — empty. Needs
  `npm install @tabler/icons-webfont` (or equivalent per-library
  install) and an export step to flatten SVGs into these folders.

## Adding a new channel

1. `mkdir channels/<name>` + copy `channels/psychology/concept-library.json` as a starting template.
2. Point n8n's dispatch payload at `--channel <name>`.
3. Nothing else — `asset_resolver`, `camera`, `gesture_engine`, `voice_engine` are all channel-agnostic already.

## Calibrating new hand gestures

Run `python3 tools/calibrate_hands.py` after adding a PNG to
`assets/hands/` — it overlays a labeled coordinate grid so you can
read off the anchor point visually, same method used for the current
10 gestures. Add the result to `hand-gestures.json` by hand; this is a
one-time step per asset, never runs at render time.
