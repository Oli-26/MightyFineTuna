#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""
Headless batch generation. Reads prompts.txt, calls picker /generate_one × N
per prompt, writes to pending.jsonl for later review in the UI.

Resumes automatically: skips prompts already present in pending.jsonl.

Usage:
  ./batch_gen.py                     # uses ./prompts.txt, ./pending.jsonl
  ./batch_gen.py --prompts other.txt
  ./batch_gen.py --url http://127.0.0.1:8000 --n 3
  ./batch_gen.py --no-skip           # don't skip; allow duplicate prompts
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path

import httpx


def load_prompts(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def already_done_prompts(pending: Path) -> set[str]:
    if not pending.exists():
        return set()
    seen = set()
    for line in pending.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seen.add(json.loads(line).get("prompt", ""))
        except Exception:
            pass
    return seen


async def gen_one(client: httpx.AsyncClient, url: str, prompt: str, index: int) -> dict:
    r = await client.post(
        f"{url}/generate_one",
        json={"prompt": prompt, "index": index},
        timeout=900.0,
    )
    r.raise_for_status()
    return r.json()


async def gen_batch(client: httpx.AsyncClient, url: str, prompt: str, n: int) -> list[dict]:
    # picker is np=1 typically, so requests serialize at llama-server. Fire all,
    # let server queue them. Catch per-task exceptions.
    tasks = [gen_one(client, url, prompt, i) for i in range(n)]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--prompts", default=str(here / "prompts.txt"))
    ap.add_argument("--out", default=str(here / "pending.jsonl"))
    ap.add_argument("--url", default=os.environ.get("PICKER_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--n", type=int, default=3, help="candidates per prompt")
    ap.add_argument("--no-skip", action="store_true", help="don't skip prompts already in pending.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many fresh prompts (0 = all)")
    args = ap.parse_args()

    prompts_path = Path(args.prompts)
    pending_path = Path(args.out)

    prompts = load_prompts(prompts_path)
    if not args.no_skip:
        done = already_done_prompts(pending_path)
        before = len(prompts)
        prompts = [p for p in prompts if p not in done]
        print(f"[batch] skipped {before - len(prompts)} prompts already in {pending_path.name}")

    if args.limit > 0:
        prompts = prompts[: args.limit]

    if not prompts:
        print("[batch] nothing to do")
        return 0

    print(f"[batch] {len(prompts)} prompts → {args.url}, n={args.n}")
    print(f"[batch] writing {pending_path}")
    print()

    started = time.time()
    async with httpx.AsyncClient() as client:
        # health check
        try:
            r = await client.get(f"{args.url}/health", timeout=5.0)
            r.raise_for_status()
        except Exception as e:
            print(f"[batch] picker not reachable at {args.url}: {e}", file=sys.stderr)
            return 1

        for i, prompt in enumerate(prompts, 1):
            t0 = time.time()
            print(f"[{i}/{len(prompts)}] {prompt!r}", flush=True)
            try:
                results = await gen_batch(client, args.url, prompt, args.n)
            except Exception as e:
                print(f"  ! batch error: {e}", file=sys.stderr)
                continue

            candidates: list[str] = []
            temps: list[float | None] = []
            suspect: list[bool] = []
            errored: list[bool] = []
            for r in results:
                if isinstance(r, Exception):
                    candidates.append("")
                    temps.append(None)
                    suspect.append(False)
                    errored.append(True)
                else:
                    candidates.append(r.get("code", ""))
                    temps.append(r.get("temp"))
                    suspect.append(bool(r.get("suspect")))
                    errored.append(bool(r.get("error")) or not r.get("code"))

            rec = {
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "schema": "pending-v1",
                "host": os.environ.get("HOST_TAG", socket.gethostname()),
                "prompt": prompt,
                "candidates": candidates,
                "temps": temps,
                "suspect": suspect,
                "errored": errored,
            }
            with pending_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            dt = time.time() - t0
            elapsed = time.time() - started
            print(f"  -> {rec['id'][:8]} chars={[len(c) for c in candidates]} ({dt:.1f}s, total {elapsed:.0f}s)")

    print(f"\n[batch] done. wrote {len(prompts)} records to {pending_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
