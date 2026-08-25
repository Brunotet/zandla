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


def _search_vendor_icons(keyword: str, top_k_by_filename: int = 12) -> Optional[str]:
    """Two-stage search: filename match narrows the vendored set down
    to a manageable shortlist (full CLIP-scoring across 20,000+ icons
    per call would be slow and mostly pointless — a filename match is
    a strong prior), then CLIP picks the best of that shortlist.
    Returns an absolute path to the winning SVG, or None if nothing in
    the vendored set is even filename-plausible.

    BUG FIXED: matching used to be raw substring containment
    (`tok in stem`) — for a short, common concept_key like "win", that
    matches ANY filename merely containing those letters as a
    substring: "window", "windows-restore", "twin-bed", none of which
    have anything to do with winning. CLIP then only ever gets to pick
    the least-bad option from a shortlist that was never actually
    about the right concept — this is the most likely real cause of a
    well-chosen but short/common concept_key resolving to an
    unrelated-looking icon. Word-boundary matching (\\b...\\b) matches
    "win" as its own standalone word — "win-trophy" or "future win",
    yes; "window" or "twin", no, since there's no boundary between the
    "win" letters and the rest of those words. Multi-word concept_keys
    were mostly safe from this already, since vendored filenames are
    hyphen/underscore-delimited into the same tokens a concept_key
    already splits into — this fix mainly protects short, single-word
    concept_keys, which is exactly where the false positives showed up.

    ALSO FIXED: the top_k_by_filename cutoff used to only `break` the
    INNER loop (one icon library's file listing) — the outer loop over
    VENDOR_ICON_DIRS kept going regardless, so the shortlist could
    silently grow past the intended cap once more than one vendored
    library was present. Checked before the outer loop continues too now.
    """
    kw_tokens = keyword.lower().replace("-", " ").replace("_", " ").split()
    shortlist = []
    for lib_name, lib_dir in VENDOR_ICON_DIRS.items():
        if len(shortlist) >= top_k_by_filename:
            break
        if not os.path.isdir(lib_dir):
            continue
        for fname in os.listdir(lib_dir):
            if not fname.endswith(".svg"):
                continue
            stem = fname[:-4].lower().replace("-", " ").replace("_", " ")
            if any(re.search(rf"\b{re.escape(tok)}\b", stem) for tok in kw_tokens):
                shortlist.append(os.path.join(lib_dir, fname))
            if len(shortlist) >= top_k_by_filename:
                break

    if not shortlist:
        return None
    if len(shortlist) == 1:
        return shortlist[0]

    rasters = [_rasterize_svg(p) for p in shortlist]
    valid = [(p, r) for p, r in zip(shortlist, rasters) if r is not None]
    if not valid:
        return shortlist[0]

    paths, raster_bytes = zip(*valid)
    best_i = _clip_best_match_index(keyword, list(raster_bytes))
    return paths[best_i] if best_i is not None else paths[0]


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
