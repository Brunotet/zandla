"""
Standalone smoke test for historical_asset_resolver.py.

Run this BEFORE wiring anything into n8n/GitHub Actions — it hits the
four sources directly, saves whatever CLIP picks for a few test
phrases into /tmp/historical_test/, and prints which source won each
one. Look at the saved images yourself; this only tells you the
pipeline runs end-to-end and licenses are being filtered, not that the
image is actually a great match — that's a judgment call for you.

Usage:
    pip install -r requirements.txt --break-system-packages   # if not already done
    python3 tools/test_historical_sources.py
    python3 tools/test_historical_sources.py --query "Nikola Tesla laboratory 1899"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import historical_asset_resolver as har

DEFAULT_TEST_PHRASES = [
    "Amelia Earhart portrait 1928",
    "Nikola Tesla laboratory",
    "Apollo 11 moon landing",
    "Frederick Douglass photograph",
]

OUT_DIR = "/tmp/historical_test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", help="Custom phrase to test (repeatable). "
                         "Defaults to a fixed set covering a person, a scientist, a space subject, "
                         "and a 19th-century figure, so all four sources get exercised at least once.")
    args = parser.parse_args()
    phrases = args.query if args.query else DEFAULT_TEST_PHRASES

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Testing {len(phrases)} phrase(s), saving results to {OUT_DIR}/\n")

    results = []
    for phrase in phrases:
        print(f"── {phrase!r} ──")
        resolved = har.resolve(phrase, cache_dir=OUT_DIR)
        if resolved is None:
            print("  NO RESULT — nothing usable found across any source\n")
            results.append((phrase, None, None))
            continue
        print(f"  -> source={resolved.source}  saved={resolved.data}\n")
        results.append((phrase, resolved.source, resolved.data))

    print("── Summary ──")
    hit = sum(1 for _, src, _ in results if src)
    print(f"{hit}/{len(results)} phrases resolved.")
    for phrase, src, path in results:
        status = f"{src} -> {path}" if src else "NO RESULT"
        print(f"  {phrase!r}: {status}")
    print(f"\nOpen the saved .jpg files in {OUT_DIR}/ and eyeball whether CLIP actually "
          f"picked something relevant — that's the real test.")


if __name__ == "__main__":
    main()
