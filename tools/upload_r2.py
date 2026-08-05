"""
Upload the rendered video to R2 via a presigned PUT URL.

IMPORTANT — this matches the fix from the earlier debugging round on
the tongue-twisters pipeline: the upload MUST be a raw binary PUT body
with an explicit Content-Type header, NOT a multipart/form-data
upload. A presigned URL's SigV4 signature is computed over the raw
request (including the exact Content-Type it was signed for) — sending
it as `files={...}` (multipart) wraps the bytes in form boundaries and
changes both the body and the effective content-type, which is exactly
the "binary multipart form type misconfiguration" that caused
signature-mismatch failures before. `requests.put(url, data=<bytes>)`
sends the raw body untouched, which is what a presigned PUT needs.

n8n is expected to set `presigned_put_url` in the dispatch payload
(pre-signed for a PUT, with whatever Content-Type it was signed
against — passed through here so this script never has to guess it).
"""
import argparse
import json
import mimetypes
import sys
import requests


def upload(file_path: str, presigned_url: str, content_type: str = None) -> None:
    if content_type is None:
        content_type, _ = mimetypes.guess_type(file_path)
        content_type = content_type or "video/mp4"

    with open(file_path, "rb") as f:
        body = f.read()

    resp = requests.put(
        presigned_url,
        data=body,                      # raw bytes — NOT files=... (multipart), see module docstring
        headers={"Content-Type": content_type},
        timeout=120,
    )

    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(
            f"R2 upload failed: HTTP {resp.status_code} — {resp.text[:500]}\n"
            f"If this is a signature mismatch, check that content_type here matches "
            f"exactly what the presigned URL was signed with on the n8n side."
        )

    print(f"[upload_r2] uploaded {len(body)} bytes -> R2 (HTTP {resp.status_code})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()

    with open(args.payload) as f:
        payload = json.load(f)

    presigned_url = payload.get("presigned_put_url")
    if not presigned_url:
        print("[upload_r2] payload missing 'presigned_put_url' — n8n must set this at dispatch time")
        sys.exit(1)

    content_type = payload.get("upload_content_type")  # optional override from n8n
    upload(args.file, presigned_url, content_type=content_type)
