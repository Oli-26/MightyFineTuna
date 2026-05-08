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
import random
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


DEFAULT_TEMP_POOL = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
TEMP_TIERS = [(0.3, 0.5), (0.7, 0.9), (1.1, 1.3)]  # low / mid / high
TOP_P_POOL = [0.7, 0.85, 0.95, 1.0]
MIN_P_POOL = [0.0, 0.02, 0.05, 0.1]
MAX_TOKENS_POOL = [800, 1200, 1600]


def _sample_sampler(temp: float) -> dict:
    return {
        "temp": temp,
        "top_p": random.choice(TOP_P_POOL),
        "min_p": random.choice(MIN_P_POOL),
        "max_tokens": random.choice(MAX_TOKENS_POOL),
    }


async def gen_one(client: httpx.AsyncClient, url: str, prompt: str, sampler: dict) -> dict:
    r = await client.post(
        f"{url}/generate_one",
        json={"prompt": prompt, **sampler},
        timeout=900.0,
    )
    r.raise_for_status()
    return r.json()


async def fetch_temp_pool(client: httpx.AsyncClient, url: str) -> list[float]:
    try:
        r = await client.get(f"{url}/config", timeout=5.0)
        d = r.json()
        pool = d.get("temp_pool")
        if isinstance(pool, list) and pool:
            return [float(t) for t in pool]
    except Exception:
        pass
    return DEFAULT_TEMP_POOL


async def gen_batch(client: httpx.AsyncClient, url: str, prompt: str, n: int, pool: list[float]) -> list[dict]:
    # One temp from each tier (low/mid/high) when n=3, else random sample from pool.
    # Each candidate also gets random top_p / min_p / max_tokens for richer variance.
    if n == 3:
        temps = [random.uniform(*tier) for tier in TEMP_TIERS]
        random.shuffle(temps)  # decouple slot index from temp tier
    else:
        temps = random.sample(pool, k=min(n, len(pool)))
    samplers = [_sample_sampler(round(t, 3)) for t in temps]
    tasks = [gen_one(client, url, prompt, samplers[slot]) for slot in range(n)]
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

        pool = await fetch_temp_pool(client, args.url)
        print(f"[batch] temp pool: {pool}", flush=True)

        for i, prompt in enumerate(prompts, 1):
            t0 = time.time()
            print(f"[{i}/{len(prompts)}] {prompt!r}", flush=True)
            try:
                results = await gen_batch(client, args.url, prompt, args.n, pool)
            except Exception as e:
                print(f"  ! batch error: {e}", file=sys.stderr)
                continue

            candidates: list[str] = []
            temps: list[float | None] = []
            top_ps: list[float | None] = []
            min_ps: list[float | None] = []
            max_tokens_list: list[int | None] = []
            suspect: list[bool] = []
            errored: list[bool] = []
            for r in results:
                if isinstance(r, Exception):
                    candidates.append("")
                    temps.append(None)
                    top_ps.append(None)
                    min_ps.append(None)
                    max_tokens_list.append(None)
                    suspect.append(False)
                    errored.append(True)
                else:
                    candidates.append(r.get("code", ""))
                    temps.append(r.get("temp"))
                    top_ps.append(r.get("top_p"))
                    min_ps.append(r.get("min_p"))
                    max_tokens_list.append(r.get("max_tokens"))
                    suspect.append(bool(r.get("suspect")))
                    errored.append(bool(r.get("error")) or not r.get("code"))

            sp_ids = [r.get("system_prompt_id") if isinstance(r, dict) else None for r in results]
            rec = {
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "schema": "pending-v1",
                "host": os.environ.get("HOST_TAG", socket.gethostname()),
                "prompt": prompt,
                "candidates": candidates,
                "temps": temps,
                "top_ps": top_ps,
                "min_ps": min_ps,
                "max_tokens": max_tokens_list,
                "system_prompt_id": next((s for s in sp_ids if s is not None), None),
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
