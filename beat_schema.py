"""
Beat schema + validation.

A "beat" is one unit of script + visual instruction — roughly one
sentence or short clause. This is what n8n's extractor node should
emit per beat (mirrors the BeatSegment pattern from the existing
pipelines), and what render_pipeline.py consumes.

Validation happens in n8n BEFORE the GitHub Actions render is
triggered (cheap, fast, fails the workflow immediately with a clear
error) AND again here at render time (belt-and-suspenders — matches
the existing "hard errors over silent fallbacks" rule). A beat
referencing a concept_key that resolves to nothing anywhere is a hard
failure, not a skipped beat — a silently dropped visual is worse than
a loud failure you can fix once and rerun.
"""
import json
import os
from typing import Optional

# "point" and "zoom_in"/"zoom_out" REMOVED per direct feedback — none
# of these gestures (pointing finger, pinch-zoom icon enlarge, or the
# plain camera-only zoom_in/zoom_out that replaced it) are wanted
# anymore. The only camera behavior beats drive now is the automatic
# reframe every draw/write/icon_word beat already gets via
# render_pipeline.py's own _apply_camera_move call — nothing a beat's
# own "mode" field needs to ask for separately.
VALID_MODES = {"write", "draw", "icon_word", "drag", "erase", "swipe", "talk"}
GESTURE_FOR_MODE = {
    "write": "write",
    "draw": "write",       # draw mode still uses the write/pen gesture, just targets an icon instead of text
    "icon_word": "write",  # icon + short label, same pen gesture as draw/write
    "drag": "drag",
    "erase": "erase",
    "swipe": "swipe",
    "talk": None,           # cutaway beats pick talk/talk2/shrug at render time for variety
}


class BeatValidationError(Exception):
    pass


def validate_beat(beat: dict, index: int) -> None:
    required = ["beat_id", "channel", "text", "mode"]
    for key in required:
        if key not in beat:
            raise BeatValidationError(f"beat[{index}] missing required field '{key}'")

    if beat["mode"] not in VALID_MODES:
        raise BeatValidationError(
            f"beat[{index}] (id={beat['beat_id']}) has invalid mode '{beat['mode']}', "
            f"must be one of {sorted(VALID_MODES)}"
        )

    has_items = beat["mode"] == "icon_word" and bool(beat.get("items"))
    # NEW: pure icon-only listicle items (no word labels), added per
    # direct feedback — PSYCHOLOGY CHANNEL ONLY (enforced in
    # render_pipeline.py, not here — this file has no notion of
    # "channel-specific", so the real gate lives where channel is
    # actually known). Mutually exclusive with 'items': a beat is
    # either icon,word pairs OR pure icons, never both.
    has_icons = beat["mode"] == "icon_word" and bool(beat.get("icons"))

    if has_items and has_icons:
        raise BeatValidationError(
            f"beat[{index}] (id={beat['beat_id']}) has BOTH 'items' and 'icons' — use one or the "
            f"other, not both."
        )

    # NEW: 'photos' — a draw beat can carry a 2-4 entry photo sequence
    # INSTEAD OF a concept_key (render_pipeline.py's history-channel
    # "photos" feature — already implemented there, this file just
    # never had the matching validation added, which is exactly what
    # produced a real render failure: the n8n builder already allowed
    # this shape, but this file still hard-required concept_key on
    # every 'draw' beat with no exception, so a valid photos-only beat
    # got rejected here even though render_pipeline.py knows how to
    # render it. Mirrors the n8n builder's own has_photos/hasPhotos
    # check exactly, so both validation layers agree on what's allowed.
    has_photos = beat["mode"] == "draw" and bool(beat.get("photos"))

    if beat["mode"] in ("draw", "drag") and "concept_key" not in beat and not has_photos:
        raise BeatValidationError(
            f"beat[{index}] (id={beat['beat_id']}) mode='{beat['mode']}' requires a concept_key"
            + (", or a 'photos' list of 2-4 entries" if beat["mode"] == "draw" else "")
        )

    if beat["mode"] == "draw" and "concept_key" in beat and has_photos:
        raise BeatValidationError(
            f"beat[{index}] (id={beat['beat_id']}) mode='draw' has BOTH concept_key and 'photos' — "
            f"use one or the other, not both."
        )

    if has_photos:
        photos = beat["photos"]
        if not isinstance(photos, list) or len(photos) < 2 or len(photos) > 4:
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) mode='draw' 'photos' needs 2-4 entries — got {photos!r}"
            )
        for p_idx, p in enumerate(photos):
            if not str(p or "").strip():
                raise BeatValidationError(
                    f"beat[{index}] (id={beat['beat_id']}) photos[{p_idx}] is empty"
                )

    if beat["mode"] == "icon_word" and not has_items and not has_icons:
        if "concept_key" not in beat:
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) mode='icon_word' requires a concept_key "
                f"(or an 'items' list of 2-4 icons/words, or an 'icons' list of 1-4 pure concept_keys)"
            )
        if not str(beat.get("label", "")).strip():
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) mode='icon_word' requires a non-empty 'label' "
                f"(the short word/phrase written beside the icon, e.g. concept_key='food', label='good')"
            )

    if beat["mode"] == "icon_word":
        # NEW, optional: "layout" picks how icon(s) and word(s) share
        # their space — "side_by_side" (default, icon left / word right)
        # or "stacked" (icon centered, word centered directly below it).
        # Applies to BOTH the single concept_key/label form AND the
        # multi-row "items" form (where it applies uniformly to every
        # row). Omit entirely for the default; only checked when
        # present, so every existing beat (which never sets this) is
        # unaffected.
        layout = beat.get("layout")
        if layout is not None and layout not in ("side_by_side", "stacked"):
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) mode='icon_word' has invalid 'layout' "
                f"{layout!r} — must be 'side_by_side' or 'stacked' (omit entirely for the default side_by_side)"
            )

    if has_items:
        items = beat["items"]
        if not isinstance(items, list) or len(items) < 2 or len(items) > 6 or len(items) % 2 != 0:
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) mode='icon_word' with 'items' needs an EVEN "
                f"number of entries (2, 4, or 6) forming icon,word pairs — got {items!r}"
            )
        for pair_idx in range(0, len(items), 2):
            icon_item, word_item = items[pair_idx], items[pair_idx + 1]
            icon_type = icon_item.get("type", "icon")
            word_type = word_item.get("type")
            if icon_type != "icon" or word_type != "word":
                raise BeatValidationError(
                    f"beat[{index}] (id={beat['beat_id']}) items[{pair_idx}]/items[{pair_idx+1}] must be "
                    f"an icon,word pair in that exact order (icon first, word second) — got types "
                    f"{icon_type!r}, {word_type!r}. Patterns like icon-word-word, word-word-icon, or "
                    f"word-icon-word are not valid — every pair is icon,word, forming one row."
                )
            if not str(icon_item.get("concept_key", "")).strip():
                raise BeatValidationError(
                    f"beat[{index}] (id={beat['beat_id']}) items[{pair_idx}] (icon) requires a concept_key"
                )
            if not str(word_item.get("label", "")).strip():
                raise BeatValidationError(
                    f"beat[{index}] (id={beat['beat_id']}) items[{pair_idx+1}] (word) requires a non-empty label"
                )

    if has_icons:
        icons = beat["icons"]
        if not isinstance(icons, list) or not (1 <= len(icons) <= 4):
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) 'icons' must be a list of 1-4 concept_keys "
                f"(pure icons, no word labels) — got {icons!r}"
            )
        for i, ck in enumerate(icons):
            if not str(ck or "").strip():
                raise BeatValidationError(
                    f"beat[{index}] (id={beat['beat_id']}) icons[{i}] is empty"
                )

        # NEW: "icon_layout" picks between the default left/right "grid"
        # (see _layout_icon_grid) and the "cluster" style — one icon
        # hand-drawn center, the rest popping around it (see
        # _layout_icon_cluster) — added per direct spec. Cheap early
        # check here matches render_pipeline.py's own belt-and-suspenders
        # check at render time, so a bad count fails fast in n8n
        # instead of burning a full GitHub Actions run first.
        icon_layout = beat.get("icon_layout")
        if icon_layout is not None and icon_layout not in ("grid", "cluster"):
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) 'icon_layout' must be 'grid' or 'cluster' "
                f"(omit entirely for the default 'grid') — got {icon_layout!r}"
            )
        if icon_layout == "cluster" and len(icons) not in (3, 4):
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) icon_layout='cluster' requires exactly 3 or 4 "
                f"icons — got {len(icons)}"
            )
    elif beat.get("icon_layout") is not None:
        raise BeatValidationError(
            f"beat[{index}] (id={beat['beat_id']}) sets 'icon_layout' but has no 'icons' — "
            f"'icon_layout' only applies to icon-only beats."
        )

    # NEW, optional: "number" draws a big standalone digit (1, 2, 3...)
    # in its own row ABOVE the items rows — for a numbered listicle
    # sentence ("One, you...", "Two, you..."), this puts an actual "1"/
    # "2" on screen synced to when the narrator says that number word,
    # with the sentence's real content still drawn as normal icon/word
    # rows below it. Only meaningful (and only rendered) on an
    # icon_word beat WITH items — a bare digit with nothing else on
    # screen isn't useful, so setting it anywhere else is a planner
    # mistake, surfaced loudly rather than silently ignored.
    if "number" in beat and beat["number"] is not None:
        if not has_items and not has_icons:
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) sets 'number' but mode='icon_word' has neither "
                f"'items' nor 'icons' — 'number' only makes sense alongside content to draw above (the "
                f"number draws in its own row above whichever content rows/grid the beat has). Add "
                f"'items' or 'icons', or drop 'number' for this beat."
            )
        if not isinstance(beat["number"], int) or isinstance(beat["number"], bool) or beat["number"] < 1:
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) 'number' must be a positive integer (1, 2, 3, ...) — "
                f"got {beat['number']!r}"
            )

    # NEW, PSYCHOLOGY CHANNEL ONLY, per direct request: a single lone
    # icon (mode='draw', or 'icon_word' with a single concept_key/label
    # and no 'items'/'icons') can only represent a genuinely SHORT
    # sentence — 4 words or fewer. A 5+ word sentence has too much
    # content for one icon to adequately carry; it must use multiple
    # icons instead ('items' or 'icons', 2+ pieces). This is a
    # structural guard, not a style choice — it doesn't say WHICH
    # multi-icon shape to use, just that a lone icon isn't enough here.
    if beat.get("channel") == "psychology":
        is_lone_icon = beat["mode"] == "draw" or (
            beat["mode"] == "icon_word" and not has_items and not has_icons
        )
        if is_lone_icon:
            word_count = len(beat.get("text", "").split())
            if word_count >= 5:
                raise BeatValidationError(
                    f"beat[{index}] (id={beat['beat_id']}) is a lone single icon (mode={beat['mode']!r}) "
                    f"for a {word_count}-word sentence — a single icon is only allowed for sentences of "
                    f"4 words or fewer on this channel. Use 'items' (icon+word pieces) or 'icons' "
                    f"(pure icons) with 2 or more entries instead."
                )

    if not beat["text"].strip():
        raise BeatValidationError(f"beat[{index}] (id={beat['beat_id']}) has empty text")


def validate_beats_against_vocabulary(beats: list, vocabulary: set) -> None:
    """Called AFTER validate_beat on every beat. `vocabulary` is the
    union of every concept_key across the shared + channel-specific
    concept-library.json files, generated at n8n-run time (see
    tools/build_vocabulary.py). A concept_key outside this set fails
    the whole batch immediately — this is the check that keeps Gemini
    constrained to a closed vocabulary rather than inventing draw
    targets that have no asset mapping."""
    for i, beat in enumerate(beats):
        ck = beat.get("concept_key")
        if ck and ck not in vocabulary:
            raise BeatValidationError(
                f"beat[{i}] (id={beat['beat_id']}) references unknown concept_key '{ck}' "
                f"— not present in any concept-library.json. Add it before rendering."
            )
        for item_idx, item in enumerate(beat.get("items") or []):
            item_ck = item.get("concept_key")
            if item_ck and item_ck not in vocabulary:
                raise BeatValidationError(
                    f"beat[{i}] (id={beat['beat_id']}) items[{item_idx}] references unknown concept_key "
                    f"'{item_ck}' — not present in any concept-library.json. Add it before rendering."
                )
        for icon_idx, ck in enumerate(beat.get("icons") or []):
            if ck and ck not in vocabulary:
                raise BeatValidationError(
                    f"beat[{i}] (id={beat['beat_id']}) icons[{icon_idx}] references unknown concept_key "
                    f"'{ck}' — not present in any concept-library.json. Add it before rendering."
                )


def validate_batch(beats: list, vocabulary: Optional[set] = None) -> None:
    for i, beat in enumerate(beats):
        validate_beat(beat, i)
    if vocabulary is not None:
        validate_beats_against_vocabulary(beats, vocabulary)


def load_vocabulary(*concept_library_paths: str) -> set:
    vocab = set()
    for path in concept_library_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for key in data.keys():
            if key.startswith("_"):
                continue
            vocab.add(key)
    return vocab


EXAMPLE_BEAT = {
    "beat_id": 4,
    "channel": "psychology",
    "text": "the brain rewires itself constantly",
    "mode": "draw",
    "concept_key": "brain",
    "camera": {"action": "zoom_in", "padding": 60, "duration": 1.0},
}
