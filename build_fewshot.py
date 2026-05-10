#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Pick 3 strong rank-1 examples to graft into a new system prompt variant.

Selection:
  - Schema rank-v1, not obsolete, not flagged
  - Rank=1 candidate present, contains draw calls
  - Code length 600-1800 chars (long enough to be interesting, short enough to fit)
  - Diverse subjects (greedy: pick first, then maximize prompt-token-Jaccard distance)

Writes: fewshot_examples.json — list of {prompt, code, source_model}
Then prints suggested system_prompts.py addition for variant F.
"""
from __future__ import annotations
import glob, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT = ROOT / "fewshot_examples.json"

_DRAW_RE = re.compile(r"\b(?:fillRect|arc\(|fill\(|stroke\(|moveTo|lineTo)")
_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


def main():
    paths = sorted(glob.glob(str(RESULTS / "*-prefs.jsonl")))
    if (ROOT / "prefs.jsonl").exists():
        paths.append(str(ROOT / "prefs.jsonl"))

    candidates = []  # (prompt, code, model, score)
    for fp in paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("schema") != "rank-v1" or r.get("obsolete"): continue
            ranks = r.get("rankings") or []
            cands = r.get("candidates") or []
            errs = r.get("errored") or [False]*len(cands)
            sus = r.get("suspect") or [False]*len(cands)
            for i, rk in enumerate(ranks):
                if rk != 1: continue
                if errs[i] or sus[i]: continue
                code = cands[i] or ""
                if not _DRAW_RE.search(code): continue
                if not (600 <= len(code) <= 1800): continue
                # decisive-win bonus: bigger gap = stronger preference
                others = [x for j, x in enumerate(ranks) if j != i and x is not None]
                gap = (min(others) - 1) if others else 0
                score = gap * 100 - len(code) * 0.001  # prefer decisive + concise
                candidates.append((r["prompt"], code, r.get("model","?"), score))

    if not candidates:
        print("no usable rank-1 examples found"); return 1

    candidates.sort(key=lambda x: -x[3])
    print(f"considering top {min(50, len(candidates))} of {len(candidates)} eligible rank-1 codes")

    # Greedy diversity pick
    picked = [candidates[0]]
    picked_tokens = [set(_TOKEN_RE.findall(picked[0][0].lower()))]
    pool = candidates[1:50]
    while len(picked) < 3 and pool:
        best_idx, best_dist = None, -1.0
        for j, c in enumerate(pool):
            tk = set(_TOKEN_RE.findall(c[0].lower()))
            min_overlap = max(jaccard(tk, pt) for pt in picked_tokens)
            dist = 1 - min_overlap
            if dist > best_dist:
                best_dist = dist; best_idx = j
        picked.append(pool[best_idx])
        picked_tokens.append(set(_TOKEN_RE.findall(picked[-1][0].lower())))
        pool.pop(best_idx)

    out = [{"prompt": p, "code": c, "source_model": m} for (p,c,m,_) in picked]
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\npicked {len(picked)} examples → {OUT.name}")
    for i, ex in enumerate(picked):
        print(f"\n[{i+1}] prompt: {ex[0]!r}  ({len(ex[1])} chars, source={ex[2][:40]})")
        first_lines = ex[1].split("\n")[:3]
        for ln in first_lines: print(f"     | {ln}")


if __name__ == "__main__":
    main()
