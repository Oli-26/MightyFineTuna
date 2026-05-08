#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""
Blind eval pair generation.

Three modes:

1) Generate one side's responses (base or tuned), saved as side_<name>.jsonl:
     ./blind_eval.py --side base   --url http://127.0.0.1:8080 --prompts eval_prompts.txt
     ./blind_eval.py --side tuned  --url http://127.0.0.1:8080 --prompts eval_prompts.txt

   Run twice — once with the base model loaded in llama-server, once with the
   tuned model loaded. (URL points directly at llama-server, NOT the picker.)

2) Pair up two side files into eval_pairs.jsonl (consumed by /blind UI):
     ./blind_eval.py --pair side_base.jsonl side_tuned.jsonl

   Joins on prompt; only emits pairs where both sides succeeded.

3) Stats on what's been judged:
     ./blind_eval.py --stats
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).parent

SYSTEM_PROMPT = """You output a JavaScript snippet that draws on an HTML canvas.

Your code is inserted directly inside this wrapper:
    const ctx = canvas.getContext('2d');
    const W = 400, H = 400;
    try {
        // <-- YOUR CODE GOES HERE (executes immediately)
    } catch(e) { ... }

Rules — follow EXACTLY:
1. Write TOP-LEVEL STATEMENTS only. They execute immediately.
2. Do NOT wrap your code in `function foo() { ... }`. If you define a function, also CALL it on the next line.
3. Do NOT include placeholder comments like `// Your code here`. Write the actual drawing code.
4. Do NOT redeclare `ctx`, `W`, `H`, or `canvas`. Use them as-is.
5. Do NOT output prose, markdown fences (```), <script> tags, HTML, or `document.getElementById`.
6. No network, no external assets, no infinite loops.
7. Pixel-art style preferred: integer coords, blocky shapes, limited palette, fillRect.

Now produce the code for the user's prompt. Output JavaScript only.
"""

DEFAULT_TEMP = 0.6  # one fixed temp for fairness — both sides get the same


def load_prompts(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _strip_fences_and_redecl(code: str) -> str:
    import re
    fence_re = re.compile(r"```(?:javascript|js|html|jsx|typescript|ts)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    think_re = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
    redecl_re = re.compile(r"^\s*(?:const|let|var)\s+(?:ctx|W|H|canvas|c)\b")
    s = think_re.sub("", code).strip()
    matches = fence_re.findall(s)
    if matches:
        s = max(matches, key=len).strip()
    elif s.startswith("```"):
        nl = s.find("\n")
        if nl != -1: s = s[nl + 1 :]
        if s.endswith("```"): s = s[:-3]
        s = s.strip()
    return "\n".join(ln for ln in s.split("\n") if not redecl_re.match(ln)).strip()


async def gen_for_prompt(client: httpx.AsyncClient, url: str, prompt: str, temp: float, max_tokens: int) -> dict:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": True,
    }
    r = await client.post(f"{url}/v1/chat/completions", json=payload, timeout=900.0)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return {"raw": raw, "code": _strip_fences_and_redecl(raw)}


async def get_model_name(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(f"{url}/v1/models", timeout=5.0)
        d = r.json()
        for m in d.get("data", []):
            if m.get("id"): return m["id"]
        for m in d.get("models", []):
            cand = m.get("model") or m.get("name")
            if cand: return cand
    except Exception: pass
    return "unknown"


async def cmd_side(args) -> int:
    prompts = load_prompts(Path(args.prompts))
    if not prompts:
        print("no prompts", file=sys.stderr); return 1
    out_path = ROOT / f"side_{args.side}.jsonl"
    seen = set()
    if out_path.exists() and not args.overwrite:
        seen = {json.loads(l)["prompt"] for l in out_path.read_text().splitlines() if l.strip()}
        prompts = [p for p in prompts if p not in seen]
        print(f"[blind:{args.side}] resuming, {len(seen)} done, {len(prompts)} to go", file=sys.stderr)
    elif args.overwrite and out_path.exists():
        out_path.unlink()

    if not prompts:
        print(f"[blind:{args.side}] nothing to do"); return 0

    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{args.url}/health", timeout=5.0); r.raise_for_status()
        except Exception as e:
            print(f"[blind:{args.side}] server unreachable at {args.url}: {e}", file=sys.stderr); return 1
        model_name = await get_model_name(client, args.url)
        print(f"[blind:{args.side}] model: {model_name}", file=sys.stderr)

        for i, prompt in enumerate(prompts, 1):
            t0 = time.time()
            try:
                res = await gen_for_prompt(client, args.url, prompt, args.temp, args.max_tokens)
                rec = {
                    "ts": time.time(),
                    "side": args.side,
                    "model": model_name,
                    "prompt": prompt,
                    "code": res["code"],
                    "raw_len": len(res["raw"]),
                    "temp": args.temp,
                }
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{i}/{len(prompts)}] {prompt!r} -> {len(res['code'])} chars ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"[{i}/{len(prompts)}] ERROR: {e}", file=sys.stderr)
    print(f"[blind:{args.side}] wrote {out_path}", file=sys.stderr)
    return 0


def cmd_pair(args) -> int:
    base_recs = [json.loads(l) for l in Path(args.pair[0]).read_text().splitlines() if l.strip()]
    tuned_recs = [json.loads(l) for l in Path(args.pair[1]).read_text().splitlines() if l.strip()]

    by_prompt_base = {r["prompt"]: r for r in base_recs}
    by_prompt_tuned = {r["prompt"]: r for r in tuned_recs}

    common = sorted(set(by_prompt_base) & set(by_prompt_tuned))
    if not common:
        print("no common prompts between the two side files", file=sys.stderr); return 1

    out_path = ROOT / args.out
    n = 0; skipped = 0
    with out_path.open("w") as f:
        for prompt in common:
            b = by_prompt_base[prompt]; t = by_prompt_tuned[prompt]
            if not b.get("code") or not t.get("code"):
                skipped += 1; continue
            rec = {
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "schema": "blind-pair-v1",
                "prompt": prompt,
                "base":  {"code": b["code"], "model": b.get("model"), "temp": b.get("temp")},
                "tuned": {"code": t["code"], "model": t.get("model"), "temp": t.get("temp")},
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"[pair] wrote {n} pairs to {out_path} (skipped {skipped} for empty code)")
    return 0


def cmd_stats(args) -> int:
    judgments_path = ROOT / "eval_judgments.jsonl"
    pairs_path = ROOT / "eval_pairs.jsonl"
    if not judgments_path.exists():
        print("no judgments yet"); return 0
    judgments = [json.loads(l) for l in judgments_path.read_text().splitlines() if l.strip()]
    total = len(judgments)
    decided = [j for j in judgments if j.get("picked") in ("left", "right")]
    correct = sum(1 for j in decided if j.get("correct"))
    print(f"judgments: {total} ({len(decided)} decided, {total - len(decided)} unsure)")
    if decided:
        acc = correct / len(decided)
        # binomial p-value vs chance (50%) — z-approx
        n = len(decided); p_hat = acc
        se = (0.25 / n) ** 0.5
        z = (p_hat - 0.5) / se if se else 0
        print(f"correct: {correct}/{len(decided)} = {acc:.1%}  (z={z:+.2f} vs 50%)")
    if pairs_path.exists():
        pairs_total = sum(1 for l in pairs_path.read_text().splitlines() if l.strip())
        seen = {j.get("pair_id") for j in judgments}
        print(f"pairs: {len(seen)} judged / {pairs_total} total")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap.add_argument("--side", choices=["base", "tuned"], help="generate this side's responses")
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server URL")
    ap.add_argument("--prompts", default=str(ROOT / "eval_prompts.txt"))
    ap.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pair", nargs=2, metavar=("SIDE_BASE", "SIDE_TUNED"), help="pair two side files")
    ap.add_argument("--out", default="eval_pairs.jsonl")
    ap.add_argument("--stats", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.stats:
        return cmd_stats(args)
    if args.pair:
        return cmd_pair(args)
    if args.side:
        return asyncio.run(cmd_side(args))
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
