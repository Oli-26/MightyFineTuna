#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Merge prefs.jsonl files from multiple machines into one combined file.
Dedupes on (ts, prompt, sha1(candidates)). Keeps first occurrence.

Usage:
  ./merge.py prefs.jsonl other-laptop/prefs.jsonl pc/prefs.jsonl > combined.jsonl
  ./merge.py *.jsonl --out combined.jsonl
  ./merge.py --stats combined.jsonl   # just report what's inside
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def fingerprint(rec: dict) -> str:
    cands = rec.get("candidates") or []
    h = hashlib.sha1()
    for c in cands:
        h.update((c or "").encode("utf-8", errors="replace"))
        h.update(b"\x00")
    ts = rec.get("ts", 0)
    prompt = rec.get("prompt", "")
    return f"{ts:.3f}|{prompt}|{h.hexdigest()[:16]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="jsonl files to merge")
    ap.add_argument("--out", "-o", default="-", help="output file (default stdout)")
    ap.add_argument("--stats", action="store_true", help="just print stats, don't write")
    ap.add_argument("--keep-obsolete", action="store_true", help="keep records flagged obsolete=true")
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    per_file = []
    for fp in args.files:
        path = Path(fp)
        if not path.exists():
            print(f"# missing: {fp}", file=sys.stderr); continue
        added = 0; dropped_dup = 0; dropped_obs = 0
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except Exception: continue
            if rec.get("obsolete") and not args.keep_obsolete:
                dropped_obs += 1; continue
            fp_key = fingerprint(rec)
            if fp_key in seen:
                dropped_dup += 1; continue
            seen[fp_key] = rec
            added += 1
        per_file.append((fp, added, dropped_dup, dropped_obs))

    # Stats
    print(f"# files merged: {len(args.files)}", file=sys.stderr)
    for fp, added, dup, obs in per_file:
        print(f"#   {fp}: +{added} (dup {dup}, obsolete {obs})", file=sys.stderr)
    print(f"# total unique records: {len(seen)}", file=sys.stderr)

    by_model = Counter(r.get("model", "?") for r in seen.values())
    by_host = Counter(r.get("host", "?") for r in seen.values())
    by_schema = Counter(r.get("schema", "?") for r in seen.values())
    print(f"# by schema: {dict(by_schema)}", file=sys.stderr)
    print(f"# by model: {dict(by_model)}", file=sys.stderr)
    print(f"# by host:  {dict(by_host)}", file=sys.stderr)

    if args.stats:
        return 0

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    try:
        # Stable order: by ts ascending
        for rec in sorted(seen.values(), key=lambda r: r.get("ts", 0)):
            out.write(json.dumps(rec) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
            print(f"# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
