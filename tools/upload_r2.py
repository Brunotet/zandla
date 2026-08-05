"""
NOT YET IMPLEMENTED — port directly from Brunotet's tongue-twisters
repo, which already solved this exact problem (R2 upload via
presigned URL, SigV4, binary multipart form type fix after the
earlier debugging round on that pipeline). Do not re-derive this from
scratch; copying the proven version avoids reintroducing the SigV2
rejection / content-type bugs already fixed there.

Expected shape once ported:
  python3 tools/upload_r2.py --file /tmp/output.mp4 --payload /tmp/payload.json
payload.json contains a "presigned_put_url" key (set by n8n at
dispatch time) that this script PUTs the file to directly.
"""
import argparse
import json
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()

    print("[upload_r2] STUB — port the working implementation from the "
          "tongue-twisters repo's R2 upload step before this workflow can run end-to-end.")
    sys.exit(1)
