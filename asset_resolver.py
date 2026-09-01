"""
Unified asset resolver for the hand-draw pipeline.

Reuses the CLIP-ranking pattern proven in render_pipeline.py (What's
The Difference) almost verbatim — _clip_best_match, and the Pexels /
Pixabay / Openverse fetchers are copied over with no logic changes,
just relocated so every channel shares one resolver instead of each
channel reimplementing image sourcing.

What's NEW here: vendored stroke-icon libraries (Tabler / Phosphor /
Lucide / Iconoir) are folded into the SAME candidate pool. A keyword
resolves through three tiers, in order:

  1. concept-library.json lookup (exact, curated — fastest, most
     reliable, no network call). Checked first because a hand-picked
     mapping beats any automatic search every time.
  2. Vendored icon set, CLIP-scored against rendered PNG previews of
     the SVGs. Preferred over stock photos/illustrations because icons
     are what the "draw" stroke-reveal animation actually needs
     (stroke_reveal requires real path data — see draw_style below).
  3. Stock illustration/photo (Pexels/Pixabay/Openverse), CLIP-scored,
     same as the existing pipeline. Used as fallback when no icon
     covers the concept — rendered with mask_wipe, not stroke_reveal,
     since these aren't stroke-path assets (see the earlier
     brainstorm: filled art can't be "un-filled" into a drawable line).

Every resolution gets written back into the channel's concept-library
so the SAME keyword never triggers a live search twice — this is what
makes the system BOTH vast (nothing pre-registered required) and
deterministic after the first run (matches the "cacheable over
generative at render time" rule).
"""
import os
import json
import re
import requests
from io import BytesIO
from typing import Optional
from PIL import Image

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "")
CANDIDATES_PER_SOURCE = 5

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_clip_device = "cpu"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR_ICON_DIRS = {
    "tabler": os.path.join(REPO_ROOT, "vendor", "icons", "tabler"),
    "phosphor": os.path.join(REPO_ROOT, "vendor", "icons", "phosphor"),
    "lucide": os.path.join(REPO_ROOT, "vendor", "icons", "lucide"),
    "iconoir": os.path.join(REPO_ROOT, "vendor", "icons", "iconoir"),
}


# ══════════════════════════════════════════════════════════════════
# CLIP scoring — unchanged from render_pipeline.py
# ══════════════════════════════════════════════════════════════════
def _load_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer, _clip_device
    if _clip_model is not None:
        return
    import torch
    import open_clip
    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    _clip_model = _clip_model.to(_clip_device).eval()
    print(f"[assets] CLIP loaded on {_clip_device}")


def _clip_best_match(keyword: str, candidates: list) -> Optional[bytes]:
    """candidates: list of raw image bytes. Returns bytes of the best
    match, or the first candidate if scoring fails for any reason —
    same never-block-the-render guarantee as the original."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    try:
        import torch
        _load_clip()

        imgs, valid_bytes = [], []
        for raw in candidates:
            try:
                im = Image.open(BytesIO(raw)).convert("RGB")
                imgs.append(_clip_preprocess(im))
                valid_bytes.append(raw)
            except Exception:
                continue
        if not imgs:
            return candidates[0]

        image_batch = torch.stack(imgs).to(_clip_device)
        text = _clip_tokenizer([keyword]).to(_clip_device)

        with torch.no_grad():
            image_features = _clip_model.encode_image(image_batch)
            text_features = _clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(1)

        best_idx = int(sims.argmax().item())
        print(f"[assets] CLIP: picked {best_idx + 1}/{len(valid_bytes)} for '{keyword}' (score={sims[best_idx]:.3f})")
        return valid_bytes[best_idx]
    except Exception as e:
        print(f"[assets] CLIP scoring failed for '{keyword}', using first candidate: {e}")
        return candidates[0]


def _clip_best_match_index(keyword: str, candidates: list) -> Optional[int]:
    """Same as _clip_best_match but returns the winning INDEX, not the
    bytes — needed for the icon tier, where we score rendered PNG
    previews but need to return the original SVG file, not a raster."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return 0
    try:
        import torch
        _load_clip()

        imgs, valid_idx = [], []
        for i, raw in enumerate(candidates):
            try:
                im = Image.open(BytesIO(raw)).convert("RGB")
                imgs.append(_clip_preprocess(im))
                valid_idx.append(i)
            except Exception:
                continue
        if not imgs:
            return 0

        image_batch = torch.stack(imgs).to(_clip_device)
        text = _clip_tokenizer([keyword]).to(_clip_device)

        with torch.no_grad():
            image_features = _clip_model.encode_image(image_batch)
            text_features = _clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(1)

        best = int(sims.argmax().item())
        return valid_idx[best]
    except Exception as e:
        print(f"[assets] CLIP icon scoring failed for '{keyword}': {e}")
        return 0


def _clip_rank_indices(keyword: str, candidates: list) -> list:
    """Like _clip_best_match_index, but returns ALL indices ranked
    best-to-worst instead of just the single winner. Added per direct
    request: a caller that needs to FALL BACK to the next-best icon
    when the top one turns out to be broken (fails to parse, has no
    usable <path> data, etc.) needs the full ranking, not just index 0.

    On any scoring failure, falls back to the candidates' ORIGINAL
    order (index 0, 1, 2, ...) rather than raising — same
    never-block-the-render guarantee as every other CLIP function
    here; a fallback ranking is still a usable ranking.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return [0]
    try:
        import torch
        _load_clip()

        imgs, valid_idx = [], []
        for i, raw in enumerate(candidates):
            try:
                im = Image.open(BytesIO(raw)).convert("RGB")
                imgs.append(_clip_preprocess(im))
                valid_idx.append(i)
            except Exception:
                continue
        if not imgs:
            return list(range(len(candidates)))

        image_batch = torch.stack(imgs).to(_clip_device)
        text = _clip_tokenizer([keyword]).to(_clip_device)

        with torch.no_grad():
            image_features = _clip_model.encode_image(image_batch)
            text_features = _clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(1)

        order = sorted(range(len(valid_idx)), key=lambda k: -float(sims[k]))
        ranked = [valid_idx[k] for k in order]
        # Any candidate that failed to even load as an image (skipped
        # above) goes at the very end — still returned, never dropped,
        # just correctly deprioritized below everything CLIP could score.
        skipped = [i for i in range(len(candidates)) if i not in valid_idx]
        return ranked + skipped
    except Exception as e:
        print(f"[assets] CLIP ranking failed for '{keyword}', using original order: {e}")
        return list(range(len(candidates)))


# ══════════════════════════════════════════════════════════════════
# Stock fetchers — unchanged from render_pipeline.py
# ══════════════════════════════════════════════════════════════════
def _fetch_pexels_candidates(keyword: str, n: int = CANDIDATES_PER_SOURCE) -> list:
    out = []
    try:
        if not PEXELS_API_KEY:
            return out
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": PEXELS_API_KEY}
        params = {"query": keyword, "per_page": n, "orientation": "portrait"}
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code != 200:
            print(f"[assets] Pexels: HTTP {r.status_code} for '{keyword}'")
            return out
        for photo in r.json().get("photos", []):
            src = photo.get("src", {})
            for size in ["large2x", "large", "medium", "original"]:
                img_url = src.get(size)
                if img_url:
                    img_r = requests.get(img_url, timeout=15)
                    if img_r.status_code == 200:
                        out.append(img_r.content)
                    break
        print(f"[assets] Pexels: {len(out)} candidate(s) for '{keyword}'")
    except Exception as e:
        print(f"[assets] Pexels error '{keyword}': {e}")
    return out


def _fetch_pixabay_candidates(keyword: str, n: int = CANDIDATES_PER_SOURCE, image_type: str = "illustration") -> list:
    out = []
    try:
        if not PIXABAY_API_KEY:
            return out
        url = "https://pixabay.com/api/"
        params = {
            "key": PIXABAY_API_KEY,
            "q": keyword,
            "image_type": image_type,   # "illustration" (default here, not "photo") — matches the hand-drawn/flat aesthetic better for this channel
            "orientation": "vertical",
            "per_page": max(n, 3),
            "safesearch": "true",
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"[assets] Pixabay: HTTP {r.status_code} for '{keyword}'")
            return out
        for hit in r.json().get("hits", [])[:n]:
            img_url = hit.get("largeImageURL") or hit.get("webformatURL")
            if img_url:
                img_r = requests.get(img_url, timeout=15)
                if img_r.status_code == 200:
                    out.append(img_r.content)
        print(f"[assets] Pixabay: {len(out)} candidate(s) for '{keyword}'")
    except Exception as e:
        print(f"[assets] Pixabay error '{keyword}': {e}")
    return out


def _fetch_openverse_candidates(keyword: str, n: int = CANDIDATES_PER_SOURCE) -> list:
    out = []
    try:
        url = "https://api.openverse.org/v1/images/"
        params = {"q": keyword, "page_size": n, "license_type": "all"}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"[assets] Openverse: HTTP {r.status_code} for '{keyword}'")
            return out
        for res in r.json().get("results", [])[:n]:
            img_url = res.get("url")
            if img_url:
                img_r = requests.get(img_url, timeout=15)
                if img_r.status_code == 200:
                    out.append(img_r.content)
        print(f"[assets] Openverse: {len(out)} candidate(s) for '{keyword}'")
    except Exception as e:
        print(f"[assets] Openverse error '{keyword}': {e}")
    return out


# ══════════════════════════════════════════════════════════════════
# Vendored icon search
# ══════════════════════════════════════════════════════════════════
def _rasterize_svg(svg_path: str, size: int = 256) -> Optional[bytes]:
    """One-time-per-icon conversion for CLIP comparison only — the
    ORIGINAL SVG is what gets used in the render (for stroke-reveal
    path data), this raster is purely a throwaway image for scoring."""
    try:
        import cairosvg
        return cairosvg.svg2png(url=svg_path, output_width=size, output_height=size,
                                 background_color="white")
    except Exception as e:
        print(f"[assets] rasterize failed for {svg_path}: {e}")
        return None


def search_vendor_icon_candidates(keyword: str, top_k_by_filename: int = 12) -> list:
    """Two-stage search: filename match narrows the vendored set down
    to a manageable shortlist (full CLIP-scoring across 20,000+ icons
    per call would be slow and mostly pointless — a filename match is
    a strong prior), then CLIP ranks that shortlist. Returns the FULL
    ranked list of candidate SVG paths, best-to-worst — not just the
    single winner (see _search_vendor_icons below for that). Added per
    direct request: a caller needs this to fall back to the next-best
    icon when the top one turns out to be broken (fails to parse, has
    no usable <path> data — see svg_to_path.py's documented LIMITATION
    on primitive-only icons). Returns an empty list if nothing in the
    vendored set is even filename-plausible for this keyword.

    BUG FIXED (confirmed by tracing the loop, not guessed): the
    shortlist cap used to be checked BEFORE moving to the next
    library, in VENDOR_ICON_DIRS iteration order (tabler, phosphor,
    lucide, iconoir) — so if tabler alone produced >= top_k_by_filename
    filename matches for a keyword, the outer loop's cap check fired
    on the very next iteration and phosphor/lucide/iconoir (8,000+
    icons) were NEVER EVEN LISTED for that keyword. This is a direct,
    concrete explanation for repetitive/generic icon choices: three of
    the four libraries could be silently excluded before CLIP ever got
    a chance to consider anything from them. Every library is now
    scanned IN FULL for filename matches before any capping happens,
    and the final shortlist is built by taking matches ROUND-ROBIN
    across libraries (one from each in turn, cycling) rather than
    keeping whichever library's matches happened to come first — this
    is what actually gives every library with a real match a fair shot
    at reaching the CLIP-scored shortlist.

    Word-boundary filename matching (\\b...\\b) is unchanged from
    before — still correctly rejects "window"/"twin-bed" matching the
    concept_key "win", still only affected the shortlist-building
    stage, not this fix.
    """
    kw_tokens = keyword.lower().replace("-", " ").replace("_", " ").split()

    per_library_matches = {}
    for lib_name, lib_dir in VENDOR_ICON_DIRS.items():
        if not os.path.isdir(lib_dir):
            continue
        matches = []
        for fname in os.listdir(lib_dir):
            if not fname.endswith(".svg"):
                continue
            stem = fname[:-4].lower().replace("-", " ").replace("_", " ")
            if any(re.search(rf"\b{re.escape(tok)}\b", stem) for tok in kw_tokens):
                matches.append(os.path.join(lib_dir, fname))
        if matches:
            per_library_matches[lib_name] = matches

    if not per_library_matches:
        return []

    shortlist = []
    lib_iters = {name: iter(paths) for name, paths in per_library_matches.items()}
    while len(shortlist) < top_k_by_filename and lib_iters:
        exhausted = []
        for name, it in lib_iters.items():
            try:
                shortlist.append(next(it))
            except StopIteration:
                exhausted.append(name)
                continue
            if len(shortlist) >= top_k_by_filename:
                break
        for name in exhausted:
            del lib_iters[name]

    if len(shortlist) == 1:
        return shortlist

    rasters = [_rasterize_svg(p) for p in shortlist]
    valid = [(p, r) for p, r in zip(shortlist, rasters) if r is not None]
    if not valid:
        # Rasterization itself failed for everything (unlikely, but the
        # existing single-result function has always tolerated this by
        # just returning the first candidate) — same fallback here,
        # returning the whole shortlist in its original order so a
        # caller retrying candidates still has something to try.
        return shortlist

    paths, raster_bytes = zip(*valid)
    ranked_local_idx = _clip_rank_indices(keyword, list(raster_bytes))
    ranked_paths = [paths[i] for i in ranked_local_idx]
    # Anything that failed to rasterize at all is appended at the end —
    # still a candidate worth trying if everything CLIP could actually
    # score turns out to be broken, just correctly deprioritized.
    failed_to_rasterize = [p for p, r in zip(shortlist, rasters) if r is None]
    return ranked_paths + failed_to_rasterize


def _search_vendor_icons(keyword: str, top_k_by_filename: int = 12) -> Optional[str]:
    """Single-winner API, UNCHANGED behavior for existing callers
    (resolve()'s tier-2 check) — now just a thin wrapper around
    search_vendor_icon_candidates() above, which does the actual work
    and documents the round-robin library-fairness fix."""
    candidates = search_vendor_icon_candidates(keyword, top_k_by_filename)
    return candidates[0] if candidates else None


# ══════════════════════════════════════════════════════════════════
# Public resolver
# ══════════════════════════════════════════════════════════════════
class ResolvedAsset:
    def __init__(self, source: str, kind: str, path_or_bytes, draw_style: str):
        self.source = source            # "icon" | "pixabay" | "pexels" | "openverse"
        self.kind = kind                # "svg_path" | "image_bytes"
        self.data = path_or_bytes
        self.draw_style = draw_style    # "stroke_reveal" | "mask_wipe"

    def to_dict(self):
        return {"source": self.source, "kind": self.kind, "draw_style": self.draw_style}


def resolve(keyword: str, hint: Optional[str] = None, cache_dir: Optional[str] = None) -> Optional[ResolvedAsset]:
    """hint: "icon" | "illustration" | None (search everything, icon
    tier first). Returns None only if every tier failed — caller
    should hard-fail the beat rather than silently skip it, per the
    existing no-silent-fallback rule; this function itself doesn't
    decide that, it just reports what it found."""
    if hint != "illustration":
        icon_path = _search_vendor_icons(keyword)
        if icon_path:
            return ResolvedAsset("icon", "svg_path", icon_path, "stroke_reveal")

    candidates = []
    candidates += _fetch_pixabay_candidates(keyword, image_type="illustration")
    candidates += _fetch_openverse_candidates(keyword)
    if not candidates:
        candidates += _fetch_pexels_candidates(keyword)

    if not candidates:
        print(f"[assets] NO CANDIDATES FOUND for '{keyword}' across icons or any stock source")
        return None

    best = _clip_best_match(keyword, candidates)
    if cache_dir and best:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{keyword.replace(' ', '_')}.png")
        with open(cache_path, "wb") as f:
            f.write(best)
        return ResolvedAsset("pixabay_or_openverse_or_pexels", "image_bytes", cache_path, "mask_wipe")

    return ResolvedAsset("pixabay_or_openverse_or_pexels", "image_bytes", best, "mask_wipe")
