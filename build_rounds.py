#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Build blind-multi-v1 rounds: each round shows 4 random candidates (out of N
side files) for a single prompt. Balanced coverage — each candidate appears
roughly the same number of times across all rounds.

Usage:
  ./build_rounds.py sides_3b/side_*.jsonl --rounds-per-prompt 3 --out eval_pairs_3b.jsonl
"""
from __future__ import annotations
import argparse, json, random, time, uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sides", nargs="+", help="side jsonl files")
    ap.add_argument("--per-round", type=int, default=4, help="candidates per round (default 4)")
    ap.add_argument("--rounds-per-prompt", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_pairs.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load each side -> {prompt: rec}
    sides: list[tuple[str, dict[str, dict]]] = []
    for fp in args.sides:
        path = Path(fp)
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not recs: continue
        label = recs[0].get("side") or path.stem.replace("side_", "")
        sides.append((label, {r["prompt"]: r for r in recs}))

    if len(sides) < args.per_round:
        print(f"need ≥{args.per_round} sides, got {len(sides)}"); return 1

    # Common prompts
    common = set(sides[0][1])
    for _, m in sides[1:]: common &= set(m)
    prompts = sorted(common)
    if not prompts:
        print("no common prompts"); return 1
    print(f"[rounds] {len(sides)} sides, {len(prompts)} common prompts")

    # Build rounds with balanced coverage. For each prompt: K rounds, each picks
    # `per_round` distinct candidates. Across all rounds we want similar counts.
    appear: Counter = Counter()
    out_path = ROOT / args.out
    n = 0; skipped = 0
    with out_path.open("w") as f:
        for prompt in prompts:
            for _ in range(args.rounds_per_prompt):
                # Greedy: pick `per_round` sides with lowest appearance count
                # tie-break randomly to avoid deterministic ordering bias.
                ranked = sorted(range(len(sides)),
                                key=lambda i: (appear[sides[i][0]], rng.random()))
                picks = ranked[:args.per_round]
                rng.shuffle(picks)  # randomize slot order for blindness
                slots = []
                ok = True
                for idx in picks:
                    label, m = sides[idx]
                    rec = m.get(prompt)
                    if not rec or not rec.get("code"):
                        ok = False; break
                    slots.append({
                        "label": label,
                        "code": rec["code"],
                        "model": rec.get("model"),
                        "temp": rec.get("temp"),
                    })
                if not ok:
                    skipped += 1; continue
                round_rec = {
                    "id": str(uuid.uuid4()),
                    "ts": time.time(),
                    "schema": "blind-multi-v1",
                    "prompt": prompt,
                    "candidates": slots,
                }
                f.write(json.dumps(round_rec) + "\n")
                n += 1
                for s in slots: appear[s["label"]] += 1

    print(f"[rounds] wrote {n} rounds to {out_path.name} (skipped {skipped} for empty)")
    print(f"\nappearance counts (target = {n * args.per_round / len(sides):.1f}):")
    for label, c in sorted(appear.items(), key=lambda x: -x[1]):
        print(f"  {label:<25} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
