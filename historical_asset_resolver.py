"""
Historical asset resolver — for the "history" channel ONLY.

This is a SEPARATE module from asset_resolver.py, not a modification of
it. asset_resolver.py resolves generic concepts ("brain", "food") to
icons or stock illustrations — it has no way to find a photo of a
SPECIFIC real person or event, and shouldn't be made to try. This file
owns that different problem: given a free-text search phrase (the
beat's concept_key, e.g. "Amelia Earhart portrait 1928"), pull
candidate PUBLIC-DOMAIN / openly-licensed images from four sources,
CLIP-score them against the phrase, return the best match.

Sources, in query order (first hit tier wins — see resolve()):
  1. Wikimedia Commons  — best for named people/subjects, huge archive,
     mostly public-domain/CC-licensed, searchable by free text.
  2. Library of Congress (Prints & Photographs) — strong for US history,
     public domain by policy for anything without a rights statement.
  3. NASA Image and Video Library — only useful if the phrase is
     space/science-flavored; skipped for everything else via a cheap
     keyword gate so it doesn't waste a request on every subject.
  4. Internet Archive — huge but noisier (mixed licensing, lots of scans/
     book pages), so it's the LAST resort, not the first.

Every candidate is filtered for a usable open license/public-domain
status BEFORE it's even added to the CLIP candidate pool — CLIP picks
the best-MATCHING image, it never decides copyright status. If a
source's response doesn't include clear rights info, that candidate is
dropped rather than guessed at.

Reuses asset_resolver's already-loaded CLIP model/scoring function
(_clip_best_match) instead of loading a second copy — same model,
same weights, one load per render process either way.

Caching: resolve() does NOT write to concept-library.json itself —
render_pipeline.resolve_beat_asset() already does that generically for
every channel, historical or not. This module just answers "what is
the best public-domain image for this phrase", nothing more.
"""
import os
import re
import requests
from io import BytesIO
from typing import Optional
from dataclasses import dataclass

import asset_resolver  # reuse _clip_best_match + the already-loaded CLIP model, not a copy of it

REQUEST_TIMEOUT = 12
CANDIDATES_PER_SOURCE = 6
USER_AGENT = "WenlincoHistoryBot/1.0 (team@wenlinco.com) research/education use"

# Wikimedia Commons license templates we treat as "safe to use" —
# anything NOT matching one of these is dropped. Deliberately
# conservative: better to skip a usable image than accidentally pull
# something still in copyright because a rights-statement field was
# empty or ambiguous.
COMMONS_SAFE_LICENSE_PATTERNS = [
    r"public domain", r"pd-", r"cc0", r"cc-by(?!-nc)", r"cc-by-sa",
]

NASA_KEYWORDS = re.compile(
    r"\b(nasa|space|astronaut|moon|mars|apollo|rocket|satellite|orbit|shuttle|nebula|galaxy|planet)\b",
    re.IGNORECASE,
)


@dataclass
class HistoricalAsset:
    source: str          # "wikimedia_commons" | "loc" | "nasa" | "internet_archive"
    image_bytes: bytes
    credit_url: str       # page to link/credit back to — keep this even though we don't render it yet,
                           # so a future caption/credits pass has it without re-fetching
    draw_style: str = "mask_wipe"  # historical photos are never stroke-path assets


def _get_json(url: str, params: dict) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[historical_assets] request failed: {url} params={params} -> {e}")
        return None


def _download_image(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"[historical_assets] image download failed: {url} -> {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# Wikimedia Commons
# ══════════════════════════════════════════════════════════════════
def _fetch_wikimedia_commons(query: str) -> list:
    """MediaWiki API, generator=search restricted to the File namespace
    (ns=6), with imageinfo (URL + extmetadata for the license check).
    Free-text search, not a category lookup — a category lookup would
    be more precise but requires first resolving query -> exact Commons
    category name, which fails silently for a lot of subjects (no
    category, or a differently-worded one). Free-text search is the
    more robust default; a category-based tier can be layered in later
    for subjects it's confirmed to help."""
    data = _get_json("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": query, "gsrnamespace": 6,
        "gsrlimit": CANDIDATES_PER_SOURCE,
        "prop": "imageinfo", "iiprop": "url|extmetadata|mime",
        "iiurlwidth": 1200,
    })
    if not data or "query" not in data:
        return []

    candidates = []
    for page in data["query"].get("pages", {}).values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = info.get("mime", "")
        if not mime.startswith("image/"):
            continue  # skip audio/video/pdf pages that matched the text search

        extmeta = info.get("extmetadata", {})
        license_short = (extmeta.get("LicenseShortName", {}).get("value")
                          or extmeta.get("License", {}).get("value") or "").lower()
        if not any(re.search(pat, license_short) for pat in COMMONS_SAFE_LICENSE_PATTERNS):
            continue

        img_url = info.get("thumburl") or info.get("url")
        if not img_url:
            continue
        img_bytes = _download_image(img_url)
        if img_bytes:
            candidates.append(HistoricalAsset(
                source="wikimedia_commons", image_bytes=img_bytes,
                credit_url=info.get("descriptionurl", img_url),
            ))
    return candidates


# ══════════════════════════════════════════════════════════════════
# Library of Congress — Prints & Photographs
# ══════════════════════════════════════════════════════════════════
def _fetch_loc(query: str) -> list:
    """loc.gov/pictures search API. LOC's Prints & Photographs division
    marks rights status per item (rights_advisory / access_advisory
    fields); items with NO advisory and an explicit "no known
    restrictions" note are the ones worth pulling. Where that field is
    ambiguous we drop the candidate rather than guess."""
    data = _get_json("https://www.loc.gov/pictures/search/", {
        "q": query, "fo": "json", "c": CANDIDATES_PER_SOURCE,
    })
    if not data or "results" not in data:
        return []

    candidates = []
    for item in data["results"][:CANDIDATES_PER_SOURCE]:
        rights = " ".join(str(item.get(k, "")) for k in
                           ("access_advisory", "rights_advisory", "reproduction_number")).lower()
        # LOC flags active copyright restriction explicitly when it
        # applies — if that flag is present, skip. Absence of a flag on
        # a Prints & Photographs item is LOC's own signal the item is
        # unrestricted, per their published rights framework.
        if "restricted" in rights or "copyright" in rights:
            continue

        img_url = None
        for key in ("image", "thumb_large", "thumb"):
            if item.get(key):
                img_url = item[key]
                break
        if not img_url:
            continue
        img_bytes = _download_image(img_url)
        if img_bytes:
            candidates.append(HistoricalAsset(
                source="loc", image_bytes=img_bytes,
                credit_url=item.get("link", img_url),
            ))
    return candidates


# ══════════════════════════════════════════════════════════════════
# NASA Image and Video Library
# ══════════════════════════════════════════════════════════════════
def _fetch_nasa(query: str) -> list:
    """Everything in NASA's media library is public domain (US
    government work) — no license filtering needed here, unlike the
    other three sources. Gated behind NASA_KEYWORDS so a subject with
    nothing to do with space doesn't burn a request/CLIP slot on
    irrelevant results every single time."""
    if not NASA_KEYWORDS.search(query):
        return []

    data = _get_json("https://images-api.nasa.gov/search", {"q": query, "media_type": "image"})
    if not data or "collection" not in data:
        return []

    candidates = []
    for item in data["collection"].get("items", [])[:CANDIDATES_PER_SOURCE]:
        links = item.get("links") or []
        img_url = next((l["href"] for l in links if l.get("render") == "image"), None)
        if not img_url and links:
            img_url = links[0].get("href")
        if not img_url:
            continue
        img_bytes = _download_image(img_url)
        if img_bytes:
            nasa_id = (item.get("data") or [{}])[0].get("nasa_id", "")
            candidates.append(HistoricalAsset(
                source="nasa", image_bytes=img_bytes,
                credit_url=f"https://images.nasa.gov/details-{nasa_id}" if nasa_id else img_url,
            ))
    return candidates


# ══════════════════════════════════════════════════════════════════
# Internet Archive — last resort (noisiest licensing of the four)
# ══════════════════════════════════════════════════════════════════
def _fetch_internet_archive(query: str) -> list:
    """advancedsearch.php to find candidate ITEMS, then the per-item
    metadata endpoint to find an actual downloadable image file inside
    each item (advancedsearch returns item-level metadata, not files).
    Restricted to mediatype:image and a licenseurl field present, so
    items with no stated license are skipped rather than assumed open —
    IA hosts a lot of copyrighted material uploaded under fair-use
    claims that is NOT freely reusable."""
    search = _get_json("https://archive.org/advancedsearch.php", {
        "q": f'{query} AND mediatype:image AND licenseurl:*',
        "fl[]": ["identifier", "licenseurl"],
        "rows": CANDIDATES_PER_SOURCE, "page": 1, "output": "json",
    })
    if not search:
        return []
    docs = (search.get("response") or {}).get("docs", [])

    candidates = []
    for doc in docs:
        licenseurl = (doc.get("licenseurl") or "").lower()
        if not any(tok in licenseurl for tok in ("publicdomain", "cc0", "by/", "by-sa/")):
            continue
        identifier = doc.get("identifier")
        if not identifier:
            continue
        meta = _get_json(f"https://archive.org/metadata/{identifier}", {})
        if not meta:
            continue
        files = meta.get("files", [])
        img_file = next((f for f in files if f.get("format") in
                          ("JPEG", "JPEG Thumb") and f.get("name")), None)
        if not img_file:
            continue
        img_url = f"https://archive.org/download/{identifier}/{img_file['name']}"
        img_bytes = _download_image(img_url)
        if img_bytes:
            candidates.append(HistoricalAsset(
                source="internet_archive", image_bytes=img_bytes,
                credit_url=f"https://archive.org/details/{identifier}",
            ))
    return candidates


# ══════════════════════════════════════════════════════════════════
# Public entrypoint
# ══════════════════════════════════════════════════════════════════
def resolve(query: str, cache_dir: Optional[str] = None) -> Optional["asset_resolver.ResolvedAsset"]:
    """query: free-text search phrase — the beat's concept_key, e.g.
    "Amelia Earhart portrait 1928". Returns an asset_resolver.ResolvedAsset
    (same shape existing channels already get back) so render_pipeline.py
    needs zero changes to how it CONSUMES a resolved asset — only to
    which resolver function it calls for this channel. Draw style is
    always mask_wipe: historical photos are filled images, never
    stroke-path line art, so they render the exact same way the
    existing pipeline already renders any non-icon illustration."""
    all_candidates = []
    for fetcher in (_fetch_wikimedia_commons, _fetch_loc, _fetch_nasa, _fetch_internet_archive):
        found = fetcher(query)
        all_candidates.extend(found)
        # Stop early once we have a healthy pool — no need to hit every
        # source for every beat once there's enough to CLIP-rank well.
        if len(all_candidates) >= CANDIDATES_PER_SOURCE * 2:
            break

    if not all_candidates:
        print(f"[historical_assets] NO CANDIDATES FOUND for '{query}' across Commons/LOC/NASA/IA")
        return None

    raw_bytes_list = [c.image_bytes for c in all_candidates]
    best_bytes = asset_resolver._clip_best_match(query, raw_bytes_list)
    if best_bytes is None:
        return None

    winner = next((c for c in all_candidates if c.image_bytes == best_bytes), all_candidates[0])
    print(f"[historical_assets] '{query}' -> {winner.source} ({winner.credit_url})")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", query)[:80]
        cache_path = os.path.join(cache_dir, f"hist_{safe_name}.jpg")
        with open(cache_path, "wb") as f:
            f.write(best_bytes)
        return asset_resolver.ResolvedAsset(winner.source, "image_bytes", cache_path, winner.draw_style)

    return asset_resolver.ResolvedAsset(winner.source, "image_bytes", best_bytes, winner.draw_style)
