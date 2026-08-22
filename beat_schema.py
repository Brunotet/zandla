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

# "point" REMOVED per direct feedback (the pointing-finger gesture is no
# longer wanted). zoom_in/zoom_out are kept as pure camera moves — see
# render_pipeline.py — so they no longer resolve to a hand gesture either.
VALID_MODES = {"write", "draw", "icon_word", "zoom_in", "zoom_out", "drag", "erase", "swipe", "talk"}
GESTURE_FOR_MODE = {
    "write": "write",
    "draw": "write",       # draw mode still uses the write/pen gesture, just targets an icon instead of text
    "icon_word": "write",  # icon + short label, same pen gesture as draw/write
    "zoom_in": None,        # pure camera move now — no hand gesture (icon-enlarge pinch removed)
    "zoom_out": None,       # pure camera move now — no hand gesture (icon-enlarge pinch removed)
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

    if beat["mode"] in ("draw", "zoom_in", "zoom_out", "drag") and "concept_key" not in beat:
        raise BeatValidationError(
            f"beat[{index}] (id={beat['beat_id']}) mode='{beat['mode']}' requires a concept_key"
        )

    if beat["mode"] == "icon_word" and not has_items:
        if "concept_key" not in beat:
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) mode='icon_word' requires a concept_key "
                f"(or an 'items' list of 2-4 icons/words instead)"
            )
        if not str(beat.get("label", "")).strip():
            raise BeatValidationError(
                f"beat[{index}] (id={beat['beat_id']}) mode='icon_word' requires a non-empty 'label' "
                f"(the short word/phrase written beside the icon, e.g. concept_key='food', label='good')"
            )
        # NEW, optional: "layout" picks how the icon and word share the
        # slot — "side_by_side" (default, icon left / word right) or
        # "stacked" (icon centered, word centered directly below it).
        # Omit entirely for the default; only checked when present so
        # every existing beat (which never sets this) is unaffected.
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
