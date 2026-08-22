"""
Historical story scout.

Runs in GitHub Actions, not n8n — deliberately. Two things it needs
are only available here: historical_asset_resolver's has_image_coverage()
(to reject a candidate BEFORE committing a script+render run to it),
and enough budget to try several candidates in a loop without n8n
babysitting each HTTP call individually.

Picking order (per your instruction — try date relevance first, fall
back to a themed pool, and forgotten/unusual/dark stories are
explicitly fine, doesn't need to be a famous person or event):
  1. Wikipedia's "On This Day" feed for TODAY's real calendar date —
     events/births/deaths, scored for "forgotten/unusual" flavor via a
     keyword heuristic, not a hard science.
  2. If nothing on today's date clears the bar (already used, or no
     image coverage), fall back to a themed pool of Wikipedia searches
     — unsolved mysteries, forgotten serial killers, hoaxes, obscure
     figures, etc.
  3. Whichever candidate survives BOTH the dedup check and the image-
     coverage check wins. Pulls the real Wikipedia extract as the
     factual grounding material the Script node uses downstream — this
     scout never invents facts, it only selects and retrieves them.
"""
import argparse
import json
import random
import re
import sys

import requests

import historical_asset_resolver as har

REQUEST_TIMEOUT = 12
USER_AGENT = har.USER_AGENT
MAX_CANDIDATES_TO_TRY = 8  # across both tiers combined, before giving up

# Lightweight scoring heuristic for "forgotten/unusual/dark" flavor —
# NOT a hard science, just biases candidate selection away from
# generic mainstream-history-textbook entries and toward the stuff
# that actually makes a good short-form hook. Real subject matter
# (serial killers, disappearances, hoaxes, unsolved cases) is
# explicitly fine per your direction, not something to filter out.
FLAVOR_KEYWORDS = re.compile(
    r"\b(murder|mysterious|mystery|disappear|vanish|hoax|unusual|bizarre|forgotten|"
    r"unsolved|curse|cursed|assassin|serial killer|cult|scandal|conspiracy|strange|"
    r"unexplained|execution|hanged|poison|haunted|massacre|riot|fraud|con artist|"
    r"impostor|imposter|kidnap|smuggl|outlaw|pirate|witch|occult|secret society)\b",
    re.IGNORECASE,
)

THEMED_QUERIES = [
    "unsolved historical murder mystery",
    "forgotten serial killer history",
    "bizarre historical hoax",
    "unexplained disappearance history",
    "strange historical urban legend",
    "obscure historical scandal",
    "cursed object history",
    "forgotten explorer disappeared",
    "historical conspiracy event",
    "unusual death historical figure",
    "forgotten con artist history",
    "historical cult tragedy",
]


def _get_json(url: str, params: dict = None):
    try:
        resp = requests.get(url, params=params or {}, timeout=REQUEST_TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[scout] request failed: {url} -> {e}")
        return None


def _flavor_score(text: str) -> int:
    return len(FLAVOR_KEYWORDS.findall(text or ""))


def _on_this_day_candidates(month: int, day: int) -> list:
    """Wikipedia's REST 'onthisday' feed — events/births/deaths for a
    specific date. Each candidate is one {title, year, extract,
    description, wiki_url} dict, scored for flavor."""
    data = _get_json(
        f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/all/{month:02d}/{day:02d}"
    )
    if not data:
        return []

    candidates = []
    for section in ("events", "deaths", "births"):
        for entry in data.get(section, []):
            pages = entry.get("pages") or []
            if not pages:
                continue
            page = pages[0]  # primary subject of this entry
            title = page.get("title")
            extract = page.get("extract") or entry.get("text") or ""
            if not title or not extract:
                continue
            candidates.append({
                "subject": title.replace("_", " "),
                "year": entry.get("year"),
                "extract": extract,
                "description": page.get("description", ""),
                "wiki_url": (page.get("content_urls", {}).get("desktop", {}) or {}).get("page", ""),
                "tier": "on_this_day",
                "score": _flavor_score(extract) + _flavor_score(page.get("description", "")),
            })
    return candidates


def _themed_pool_candidates() -> list:
    """Wikipedia full-text search across a themed query pool, tried in
    random order until one yields usable results — the themed queries
    themselves already bias toward forgotten/unusual subjects, so no
    separate flavor score is needed here (unlike on_this_day, which
    pulls from a broad, mostly-mainstream daily feed)."""
    candidates = []
    queries = random.sample(THEMED_QUERIES, len(THEMED_QUERIES))
    for query in queries:
        data = _get_json("https://en.wikipedia.org/w/api.php", {
            "action": "query", "format": "json",
            "list": "search", "srsearch": query, "srlimit": 8,
        })
        if not data:
            continue
        for hit in data.get("query", {}).get("search", []):
            title = hit.get("title")
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))  # strip <span class="searchmatch">
            if not title:
                continue
            candidates.append({
                "subject": title,
                "year": None,
                "extract": None,  # fetched properly later, once this candidate is actually chosen
                "description": snippet,
                "wiki_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "tier": "themed_pool",
                "score": _flavor_score(snippet) + 1,  # +1: themed query itself already selected for this
            })
        if len(candidates) >= MAX_CANDIDATES_TO_TRY:
            break
    return candidates


def _fetch_full_extract(title: str) -> str:
    """Real Wikipedia article extract (plain text, several paragraphs)
    for whichever candidate actually wins — the on_this_day feed's
    'extract' is often just one summary sentence, not enough factual
    material for a script writer to work from without either padding
    with filler or inventing details. This pulls the real thing."""
    data = _get_json("https://en.wikipedia.org/w/api.php", {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "exsectionformat": "plain",
        "titles": title, "exchars": 2000,
    })
    if not data:
        return ""
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "")
        if extract:
            return extract
    return ""


def scout(already_used: list) -> dict:
    already_used_norm = {u.strip().lower() for u in already_used if u.strip()}

    from datetime import date
    today = date.today()
    all_candidates = _on_this_day_candidates(today.month, today.day)
    all_candidates.sort(key=lambda c: c["score"], reverse=True)

    themed_candidates = None  # fetched lazily, only if on_this_day doesn't produce a winner

    tried = 0
    pools = [all_candidates]
    pool_idx = 0
    while tried < MAX_CANDIDATES_TO_TRY:
        if pool_idx >= len(pools[0]):
            if themed_candidates is None:
                print("[scout] on_this_day exhausted — falling back to themed pool")
                themed_candidates = _themed_pool_candidates()
                pools.append(themed_candidates)
            if len(pools) < 2 or pool_idx - len(pools[0]) >= len(pools[1]):
                break
            candidate = pools[1][pool_idx - len(pools[0])]
        else:
            candidate = pools[0][pool_idx]
        pool_idx += 1
        tried += 1

        subject_key = candidate["subject"].strip().lower()
        if subject_key in already_used_norm:
            print(f"[scout] skip (already used): {candidate['subject']}")
            continue

        if not har.has_image_coverage(candidate["subject"]):
            print(f"[scout] skip (no image coverage): {candidate['subject']}")
            continue

        # Winner — pull the real full extract now (cheap on-this-day
        # entries only had a one-line summary; themed-pool entries had
        # no extract fetched at all yet).
        extract = _fetch_full_extract(candidate["subject"])
        if not extract:
            print(f"[scout] skip (no fetchable Wikipedia extract): {candidate['subject']}")
            continue

        return {
            "found": True,
            "subject": candidate["subject"],
            "year": candidate.get("year"),
            "extract": extract,
            "description": candidate.get("description", ""),
            "wiki_url": candidate.get("wiki_url", ""),
            "source_tier": candidate["tier"],
            "candidates_tried": tried,
        }

    return {
        "found": False,
        "reason": f"exhausted {tried} candidates across on_this_day + themed pool — "
                  f"all either already used or had no usable image coverage",
        "candidates_tried": tried,
    }


if __name__ == "__main__":
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--already-used", default="",
                         help="Pipe-separated list of subjects already covered, e.g. "
                              "'Amelia Earhart|H.H. Holmes' — same convention as the psychology "
                              "channel's Guard node history compilation.")
    parser.add_argument("--output", default=None, help="Write result JSON to this path (also prints to stdout)")
    args = parser.parse_args()

    already_used = [s for s in args.already_used.split("|") if s.strip()] if args.already_used else []

    # CRASH-SAFE: always write SOMETHING to --output, even on an
    # unhandled exception — without this, a real bug (not just "no
    # candidates found today", which is an expected, valid outcome)
    # silently leaves no file behind, and the n8n callback step fails
    # with a confusing FileNotFoundError that tells you nothing about
    # what actually went wrong. Now the crash reason itself gets
    # written to the file and POSTed back to n8n, so you see the real
    # error immediately instead of having to dig through Actions logs.
    try:
        result = scout(already_used)
        exit_code = 0 if result.get("found") else 1
    except Exception as e:
        traceback.print_exc()  # still visible in the Actions log, not hidden
        result = {
            "found": False,
            "reason": f"scout CRASHED: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 1

    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    sys.exit(exit_code)
