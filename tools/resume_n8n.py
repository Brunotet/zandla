"""
Callback to the n8n Wait node that dispatched this render.

Matches your EXISTING tongue-twisters convention exactly (seen in
your n8n screenshot: an If node checking `$json.body.status` equals
`"ok"`, fed by a Wait node) rather than inventing new field names —
POSTs {"status": "ok", "render_id": ..., "video_url": ..., "channel": ...}
to callback_url, which is n8n's Wait-node webhook URL, passed straight
through from the dispatch payload (see render.yml's decode step,
which merges it in from the `callback_url` workflow_dispatch input).

render_id is echoed back so n8n can correlate this callback with the
right waiting execution if multiple renders are ever in flight at once.
"""
import argparse
import json
import sys
import requests


def callback(callback_url: str, body: dict) -> None:
    resp = requests.post(callback_url, json=body, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"n8n callback POST failed: HTTP {resp.status_code} — {resp.text[:500]}")
    print(f"[resume_n8n] callback delivered (HTTP {resp.status_code})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--public-url", default=None,
                         help="Public R2 URL of the uploaded video, if different from a derivable path")
    args = parser.parse_args()

    with open(args.payload) as f:
        payload = json.load(f)

    callback_url = payload.get("callback_url")
    if not callback_url:
        print("[resume_n8n] payload missing 'callback_url' — n8n must pass this at dispatch time")
        sys.exit(1)

    body = {
        "status": "ok",
        "render_id": payload.get("render_id"),
        "video_url": args.public_url or payload.get("public_video_url"),
        "channel": payload.get("channel"),
    }
    callback(callback_url, body)
