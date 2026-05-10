#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Aggregate stats over collected rank-v1 prefs:
- per-model win rate / avg rank
- per-system-prompt-id win rate / avg rank
- per-temp-bin win rate (low <0.6, mid 0.6-0.95, high >0.95)
- per-top_p / min_p bin if present
- flag rates (suspect, errored)

Usage: ./analyze_prefs.py
"""
from __future__ import annotations
import glob, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def temp_bin(t: float | None) -> str:
    if t is None: return "?"
    if t < 0.6: return "low(<0.6)"
    if t < 0.95: return "mid"
    return "high(>=0.95)"


def topp_bin(p: float | None) -> str:
    if p is None: return "?"
    if p <= 0.7: return "low(<=0.7)"
    if p <= 0.9: return "mid"
    return "high(>0.9)"


def minp_bin(p: float | None) -> str:
    if p is None: return "?"
    if p == 0: return "0"
    if p < 0.05: return "tiny"
    return "high(>=0.05)"


def main():
    paths = sorted(glob.glob(str(RESULTS / "*-prefs.jsonl")))
    if (ROOT / "prefs.jsonl").exists():
        paths.append(str(ROOT / "prefs.jsonl"))
    print(f"reading {len(paths)} prefs files: {[Path(p).name for p in paths]}")

    n_recs = 0
    n_obsolete = 0
    n_no_rank = 0
    flagged_suspect = 0
    flagged_errored = 0
    n_cands_total = 0

    # group: dimension -> key -> list of ranks (1=best)
    by = {
        "model": defaultdict(list),
        "sp_id": defaultdict(list),
        "host": defaultdict(list),
        "temp_bin": defaultdict(list),
        "top_p_bin": defaultdict(list),
        "min_p_bin": defaultdict(list),
        "max_tokens": defaultdict(list),
    }
    # also: how often rank-1 came from each bucket (win count)
    wins = {k: Counter() for k in by}
    n_per_bucket = {k: Counter() for k in by}

    for fp in paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("schema") != "rank-v1": continue
            if r.get("obsolete"):
                n_obsolete += 1; continue
            ranks = r.get("rankings") or []
            if not ranks:
                n_no_rank += 1; continue
            n_recs += 1
            cands = r.get("candidates") or []
            errs = r.get("errored") or [False]*len(cands)
            sus = r.get("suspect") or [False]*len(cands)
            temps = r.get("temps") or [None]*len(cands)
            top_ps = r.get("top_ps") or [None]*len(cands)
            min_ps = r.get("min_ps") or [None]*len(cands)
            mxs = r.get("max_tokens") or [None]*len(cands)
            sp_ids = r.get("system_prompt_ids") or [r.get("system_prompt_id")]*len(cands)
            model = r.get("model") or "?"
            host = r.get("host") or "?"

            for i, rk in enumerate(ranks):
                if rk is None: continue
                if errs[i]: flagged_errored += 1
                if sus[i]: flagged_suspect += 1
                n_cands_total += 1

                # Drop flagged when computing rank quality (they pollute)
                if errs[i] or sus[i]: continue

                key_vals = {
                    "model": model,
                    "sp_id": sp_ids[i] if i < len(sp_ids) else None,
                    "host": host,
                    "temp_bin": temp_bin(temps[i] if i < len(temps) else None),
                    "top_p_bin": topp_bin(top_ps[i] if i < len(top_ps) else None),
                    "min_p_bin": minp_bin(min_ps[i] if i < len(min_ps) else None),
                    "max_tokens": str(mxs[i]) if i < len(mxs) else "?",
                }
                for dim, key in key_vals.items():
                    if key is None: key = "?"
                    by[dim][key].append(rk)
                    n_per_bucket[dim][key] += 1
                    if rk == 1:
                        wins[dim][key] += 1

    print(f"\n=== summary ===")
    print(f"records (with rankings): {n_recs}")
    print(f"obsolete skipped: {n_obsolete}, no-rank skipped: {n_no_rank}")
    print(f"candidate slots ranked: {n_cands_total}")
    print(f"flagged suspect: {flagged_suspect} ({flagged_suspect/n_cands_total*100:.1f}%)")
    print(f"flagged errored: {flagged_errored} ({flagged_errored/n_cands_total*100:.1f}%)")

    for dim in by:
        rows = []
        for key, rks in by[dim].items():
            n = len(rks)
            if n < 5: continue
            avg = statistics.mean(rks)
            w = wins[dim][key]
            wr = w / n
            rows.append((key, n, w, wr, avg))
        if not rows: continue
        rows.sort(key=lambda x: x[3], reverse=True)
        print(f"\n=== by {dim} === (sorted by win rate; min n=5)")
        print(f"  {'key':<55} {'n':>5}  {'wins':>5}  {'win%':>6}  {'avgRk':>6}")
        for key, n, w, wr, avg in rows:
            kdisp = (key[:53] + "..") if len(str(key)) > 55 else str(key)
            print(f"  {kdisp:<55} {n:>5}  {w:>5}  {wr*100:>5.1f}%  {avg:>6.2f}")


if __name__ == "__main__":
    main()
