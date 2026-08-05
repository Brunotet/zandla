"""
Resume the n8n execution that dispatched this render.

Matches the async trigger-and-callback architecture already proven on
tongue-twisters: n8n calls workflow_dispatch and then WAITS (via
$execution.resumeUrl, n8n's native "Wait for webhook" mechanism)
instead of polling. Once render + R2 upload succeed, this script POSTs
back to that resume URL so n8n's paused execution continues with the
uploaded video's info.

Kept deliberately simple — one POST, the body is whatever n8n's Wait
node expects downstream (the public R2 URL plus basic render
metadata). If your n8n workflow's resume node expects a different
body shape, adjust RESUME_BODY_KEYS below; the request mechanics
(POST to resume_url) don't change per channel.
"""
import argparse
import json
import sys
import requests


def resume(resume_url: str, body: dict) -> None:
    resp = requests.post(resume_url, json=body, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"n8n resume POST failed: HTTP {resp.status_code} — {resp.text[:500]}")
    print(f"[resume_n8n] resumed execution (HTTP {resp.status_code})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--public-url", default=None,
                         help="Public R2 URL of the uploaded video, if different from a derivable path")
    args = parser.parse_args()

    with open(args.payload) as f:
        payload = json.load(f)

    resume_url = payload.get("resume_url")
    if not resume_url:
        print("[resume_n8n] payload missing 'resume_url' — n8n must set $execution.resumeUrl at dispatch time")
        sys.exit(1)

    body = {
        "status": "success",
        "video_url": args.public_url or payload.get("public_video_url"),
        "channel": payload.get("channel"),
    }
    resume(resume_url, body)
