#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=0.45"]
# ///
"""
Hybrid auto-rater. Asks Claude to rank candidates in pending records.
- High/medium confidence → write rank-v1 to prefs.jsonl, mark reviewed
- Low confidence → leave alone (defers to human via picker UI)

Requires ANTHROPIC_API_KEY env var.

Usage:
  ./auto_rate.py                       # process all unreviewed pending
  ./auto_rate.py --limit 100           # cap at 100
  ./auto_rate.py --model claude-haiku-4-5  # default; Sonnet for higher quality
  ./auto_rate.py --dry-run             # don't write anything
"""
from __future__ import annotations
import argparse, glob, json, os, re, socket, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PREFS = ROOT / "prefs.jsonl"
REVIEWED = ROOT / "reviewed.txt"
HOST = socket.gethostname()

JUDGE_SYSTEM = """You are judging short JavaScript canvas-drawing snippets for a pixel-art project.

The target style is COMPOSITION-FIRST PIXEL ART:
- Integer coordinates, blocky fillRect (sparingly arc/ellipse)
- ALWAYS paints a full background first (sky/ground/mood) — never blank
- Limited coherent palette (~4-8 colors)
- Composes deliberately — foreground/midground/background, off-center placement
- Uses shading (darker shadows, lighter highlights) for depth
- Recognizable scene, not a single shape on a blank field

The wrapper provides: const ctx = canvas.getContext('2d'); const W=400, H=400; — code is inserted as top-level statements inside try{}.

You'll see 1-4 candidates per prompt. Rank best→worst (1=best). If two are clearly tied, give them the same rank (e.g. [1,1,3]). If you genuinely cannot tell which is best (close call), set confidence="low" and skip — a human will arbitrate.

Output STRICT JSON only, no prose, no markdown:
{"rankings": [1,2,3], "confidence": "high"|"medium"|"low", "reasoning": "<1 sentence>"}
"""


def load_pending() -> dict[str, dict]:
    out: dict[str, dict] = {}
    paths = []
    if (ROOT / "pending.jsonl").exists(): paths.append(str(ROOT / "pending.jsonl"))
    paths.extend(sorted(glob.glob(str(RESULTS / "*-pending.jsonl"))))
    for fp in paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            rid = r.get("id")
            if rid: out.setdefault(rid, r)
    return out


def reviewed_ids() -> set[str]:
    if not REVIEWED.exists(): return set()
    return {ln.strip() for ln in REVIEWED.read_text().splitlines() if ln.strip()}


def already_rated_ids() -> set[str]:
    """Pending IDs already in any rank-v1 record (auto-rated or skipped)."""
    out = set()
    rank_paths = sorted(glob.glob(str(RESULTS / "*-prefs.jsonl")))
    if PREFS.exists(): rank_paths.append(str(PREFS))
    for fp in rank_paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("schema") != "rank-v1": continue
            pid = r.get("pending_id")
            if pid: out.add(pid)
    return out


def build_user_msg(pend: dict) -> str:
    cands = pend.get("candidates") or []
    parts = [f"Prompt: {pend.get('prompt','?')}", ""]
    for i, code in enumerate(cands, 1):
        parts.append(f"--- Candidate {i} ---")
        parts.append(code or "(empty)")
        parts.append("")
    parts.append(f"Rank {len(cands)} candidates. Output strict JSON.")
    return "\n".join(parts)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr); return 1

    pending = load_pending()
    done_review = reviewed_ids()
    done_rate = already_rated_ids()
    todo_ids = [rid for rid in pending if rid not in done_review and rid not in done_rate]
    print(f"{len(pending)} pending; {len(done_review)} reviewed; {len(done_rate)} already rated;"
          f" {len(todo_ids)} to process", flush=True)
    if args.limit > 0: todo_ids = todo_ids[:args.limit]
    if not todo_ids:
        print("nothing to do"); return 0

    from anthropic import Anthropic
    client = Anthropic()
    model = args.model

    n_high = n_med = n_low = n_err = 0
    n_in_tok = n_out_tok = 0
    t_start = time.time()

    for i, rid in enumerate(todo_ids, 1):
        pend = pending[rid]
        n = len(pend.get("candidates") or [])
        if n < 2:
            print(f"[{i}/{len(todo_ids)}] {rid[:8]}: <2 cands, skip"); continue

        user_msg = build_user_msg(pend)

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=args.max_tokens,
                system=[{"type":"text","text":JUDGE_SYSTEM,
                         "cache_control":{"type":"ephemeral"}}],
                messages=[{"role":"user","content":user_msg}],
            )
            n_in_tok += resp.usage.input_tokens
            n_out_tok += resp.usage.output_tokens
            txt = resp.content[0].text if resp.content else ""
            parsed = parse_response(txt)
        except Exception as e:
            print(f"[{i}/{len(todo_ids)}] {rid[:8]}: API err {type(e).__name__}: {e}", file=sys.stderr)
            n_err += 1; continue

        if not parsed or "rankings" not in parsed or "confidence" not in parsed:
            print(f"[{i}/{len(todo_ids)}] {rid[:8]}: parse fail: {txt[:120]!r}", file=sys.stderr)
            n_err += 1; continue

        rankings = parsed["rankings"]
        confidence = (parsed.get("confidence") or "").lower()
        reasoning = parsed.get("reasoning", "")

        if confidence == "low":
            n_low += 1
            print(f"[{i}/{len(todo_ids)}] {rid[:8]} {pend['prompt'][:40]!r}: LOW (defer) — {reasoning[:80]}")
            continue

        if not (isinstance(rankings, list) and len(rankings) == n
                and all(isinstance(r, int) and 1 <= r <= n for r in rankings)):
            print(f"[{i}/{len(todo_ids)}] {rid[:8]}: bad rankings shape {rankings}", file=sys.stderr)
            n_err += 1; continue

        if confidence == "high": n_high += 1
        elif confidence == "medium": n_med += 1

        sp_id = pend.get("system_prompt_id")
        rec = {
            "ts": time.time(),
            "schema": "rank-v1",
            "model": pend.get("model") or "?",
            "host": HOST,
            "prompt": pend.get("prompt", ""),
            "rankings": rankings,
            "candidates": pend.get("candidates") or [],
            "errored": pend.get("errored") or [False] * n,
            "suspect": pend.get("suspect") or [False] * n,
            "temps": pend.get("temps"),
            "top_ps": pend.get("top_ps"),
            "min_ps": pend.get("min_ps"),
            "max_tokens": pend.get("max_tokens"),
            "system_prompt_ids": [sp_id] * n if sp_id else None,
            "pending_id": rid,
            "auto_rated": True,
            "auto_judge": model,
            "auto_confidence": confidence,
            "auto_reasoning": reasoning,
        }
        if not args.dry_run:
            with PREFS.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            with REVIEWED.open("a") as f:
                f.write(rid + "\n")
        marker = "[DRY]" if args.dry_run else ""
        print(f"[{i}/{len(todo_ids)}] {rid[:8]} {pend['prompt'][:40]!r}: "
              f"{confidence.upper()} ranks={rankings} {marker} — {reasoning[:60]}")

    elapsed = time.time() - t_start
    print(f"\n=== done in {elapsed/60:.1f} min ===")
    print(f"high: {n_high}, medium: {n_med}, low(deferred): {n_low}, errors: {n_err}")
    print(f"tokens: input {n_in_tok:,}, output {n_out_tok:,}")
    # rough cost estimate (haiku 4.5: $1/$5 per M; sonnet 4.6: $3/$15 per M)
    if "haiku" in model.lower():
        cost = n_in_tok/1e6*1.0 + n_out_tok/1e6*5.0
    elif "sonnet" in model.lower():
        cost = n_in_tok/1e6*3.0 + n_out_tok/1e6*15.0
    else:
        cost = None
    if cost is not None:
        print(f"approx cost: ${cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
