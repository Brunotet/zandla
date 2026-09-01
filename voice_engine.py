"""
Voice engine for the hand-draw pipeline.

This is DELIBERATELY not a rewrite — it's the same get_tts() /
get_tts_safe() / assign_timing() pattern already proven in
render_pipeline.py (What's The Difference) and chatterbox_modal.py
(the Modal-side service), calling the same Chatterbox endpoint. The
only new piece is `words_for_beats()`, which adapts assign_timing's
segment-shaped input/output to this pipeline's beat schema so the
gesture engine and camera can read start/end times per beat without
re-deriving alignment logic.

Kept as a separate module (not copy-pasted per channel) so all
channels that use this repo share one TTS/timing implementation —
fix a bug once, every channel benefits.
"""
import os
import re
import base64
import tempfile
import difflib
import requests
from typing import Optional

MODAL_TTS_URL = os.environ.get("MODAL_TTS_URL") or (
    "https://simonmood123--chatterbox-tts-chatterboxservice-tts.modal.run"
)

_PUNCT_RE = re.compile(r"[^\w']")


def _norm_tok(tok: str) -> str:
    return _PUNCT_RE.sub("", tok).lower()


def get_tts(
    text: str,
    voice: str = None,
    exaggeration: float = 0.5,
    repetition_penalty: float = 1.5,
    speed: float = 1.0,
    exaggerations: list = None,
    pause_after_ms: list = None,
    strict_short_sentences: bool = False,
) -> dict:
    """Unchanged call shape from the existing pipelines — same payload,
    same endpoint, same response handling. Speed is applied server-side
    now (Chatterbox's own pitch-preserving TimeStretch), not via a
    local ffmpeg atempo pass, since chatterbox_modal.py already does
    this and returns rescaled word timestamps directly."""
    payload = {"text": text, "exaggeration": exaggeration, "repetition_penalty": repetition_penalty}
    if voice:
        payload["voice"] = voice
    if speed and speed != 1.0:
        payload["speed"] = speed
    if exaggerations:
        payload["exaggerations"] = exaggerations
    if pause_after_ms:
        payload["pause_after_ms"] = pause_after_ms
    if strict_short_sentences:
        payload["strict_short_sentences"] = True

    resp = requests.post(MODAL_TTS_URL, json=payload, timeout=240)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise ValueError(f"Chatterbox TTS error: {data['error']}")

    audio_bytes = base64.b64decode(data["audio_base64"])
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()

    words = data.get("words", [])
    return {"audio_path": tmp.name, "words": words, "duration": words[-1]["end"] if words else 10.0}


def _detect_repetition(script_text: str, words: list, min_ngram: int = 4) -> bool:
    expected = [_norm_tok(t) for t in script_text.split() if _norm_tok(t)]
    actual = [_norm_tok(w["word"]) for w in words]

    def ngrams(seq, n):
        return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]

    expected_counts = {}
    for g in ngrams(expected, min_ngram):
        expected_counts[g] = expected_counts.get(g, 0) + 1

    actual_counts = {}
    for g in ngrams(actual, min_ngram):
        actual_counts[g] = actual_counts.get(g, 0) + 1

    for g, count in actual_counts.items():
        if count > expected_counts.get(g, 0):
            return True
    return False


def get_tts_safe(
    text: str,
    voice: str = None,
    exaggeration: float = 0.5,
    repetition_penalty: float = 1.5,
    speed: float = 1.0,
    max_retries: int = 2,
    exaggerations: list = None,
    pause_after_ms: list = None,
    strict_short_sentences: bool = False,
) -> dict:
    """Identical retry-on-repetition-defect behavior to the existing
    pipelines — a bad take gets one full script regenerated (Modal's
    OWN internal per-sentence retry already handles most defects
    before this even sees them), not silently shipped."""
    last_result = None
    for attempt in range(max_retries + 1):
        result = get_tts(
            text, voice=voice, exaggeration=exaggeration,
            repetition_penalty=repetition_penalty, speed=speed,
            exaggerations=exaggerations, pause_after_ms=pause_after_ms,
            strict_short_sentences=strict_short_sentences,
        )
        if not _detect_repetition(text, result["words"]):
            return result
        print(f"[tts] repetition artifact detected on attempt {attempt + 1}, retrying...")
        if last_result:
            os.unlink(last_result["audio_path"])
        last_result = result

    print("[tts] repetition still present after retries — proceeding with last attempt")
    return last_result


def assign_timing(segments: list, words: list, debug: bool = False) -> list:
    """Unchanged from render_pipeline.py — difflib-based alignment of
    Whisper's flat word list back onto segment boundaries. `segments`
    here = this pipeline's beats, each needing a "text" key; returns
    the same list with "start"/"end" added."""
    expected_tokens = []
    for seg in segments:
        for tok in seg["text"].split():
            n = _norm_tok(tok)
            if n:
                expected_tokens.append(n)

    actual_tokens = [_norm_tok(w["word"]) for w in words]

    sm = difflib.SequenceMatcher(None, expected_tokens, actual_tokens, autojunk=False)
    expected_to_actual = [None] * len(expected_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                expected_to_actual[i1 + k] = j1 + k
        elif tag == "replace":
            span_e, span_a = i2 - i1, j2 - j1
            if span_a == 0:
                continue
            for k in range(span_e):
                j = j1 + min(k * span_a // span_e, span_a - 1)
                expected_to_actual[i1 + k] = j

    seg_ranges = []
    cursor = 0
    for seg in segments:
        n = len([t for t in seg["text"].split() if _norm_tok(t)])
        seg_ranges.append((cursor, cursor + n - 1) if n else (cursor, cursor - 1))
        cursor += n

    result = []
    last_end = 0.0
    for i, seg in enumerate(segments):
        lo, hi = seg_ranges[i]
        idxs = [expected_to_actual[k] for k in range(lo, hi + 1)
                if hi >= lo and expected_to_actual[k] is not None]

        if idxs:
            start = words[min(idxs)]["start"]
            end = words[max(idxs)]["end"]
            start = max(start, last_end)
            if end < start:
                end = start + 0.4
        else:
            start = last_end
            end = last_end + 0.4
            if debug:
                print(f"[timing] beat{i} '{seg['text'][:40]}' -> NO Whisper match, fallback {start:.2f}-{end:.2f}")

        result.append({**seg, "start": start, "end": end})
        last_end = end

    return result


def words_for_beats(script_text: str, beats: list, voice: str = None, **tts_kwargs) -> dict:
    """The one new function: runs the whole script through Chatterbox
    ONCE (matches the existing "single call, single Whisper pass"
    philosophy — never per-beat TTS calls, which would both cost more
    and break natural sentence-to-sentence prosody), then times every
    beat against it.

    beats: list of dicts, each MUST have a "text" key matching a
    contiguous slice of script_text (same contract as segments in the
    existing pipelines' extractor output).

    Returns {"audio_path", "words", "beats": [...with start/end...]}.
    """
    tts = get_tts_safe(script_text, voice=voice, **tts_kwargs)
    if not tts["words"]:
        raise ValueError("Chatterbox returned no word timestamps")

    timed_beats = assign_timing(beats, tts["words"], debug=os.environ.get("DEBUG_TIMING") == "1")

    return {
        "audio_path": tts["audio_path"],
        "words": tts["words"],
        "beats": timed_beats,
    }
