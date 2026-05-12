"""Count all preference pairs across all data sources."""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).parent

def count_rankv1(paths, include_auto=False):
    """Old-style rank-v1 records. Derive pairs from rankings."""
    total_pairs = 0
    n_records = 0
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip(): continue
            try: r = json.loads(line)
            except: continue
            if r.get("schema") != "rank-v1": continue
            if r.get("obsolete"): continue
            if not include_auto and r.get("auto_rated"): continue
            n_records += 1
            cands = r.get("candidates") or []
            ranks = r.get("rankings") or []
            errs = r.get("errored") or [False]*len(cands)
            sus = r.get("suspect") or [False]*len(cands)
            valid = [i for i, rk in enumerate(ranks)
                     if rk is not None and not errs[i] and not sus[i] and cands[i]]
            for a in valid:
                for b in valid:
                    if ranks[a] < ranks[b]: total_pairs += 1
    return total_pairs, n_records

def count_blindjudge(paths):
    """New-style blind-judge-multi-v1. Derive pairs from per_label_rank."""
    total_pairs = 0
    n_records = 0
    for p in paths:
        if not Path(p).exists(): continue
        for line in Path(p).read_text().splitlines():
            if not line.strip(): continue
            try: r = json.loads(line)
            except: continue
            n_records += 1
            plr = r.get("per_label_rank") or {}
            ranks = list(plr.values())
            for a in ranks:
                for b in ranks:
                    if a < b: total_pairs += 1
    return total_pairs, n_records

# Old data
old_paths = sorted(glob.glob(str(ROOT / "results/*-prefs.jsonl")))
if (ROOT / "prefs.jsonl").exists():
    old_paths.append(str(ROOT / "prefs.jsonl"))
print("=== OLD rank-v1 data ===")
h_pairs, h_recs = count_rankv1(old_paths, include_auto=False)
a_pairs, a_recs = count_rankv1(old_paths, include_auto=True)
print(f"  human only: {h_pairs} pairs from {h_recs} records")
print(f"  + auto:     {a_pairs} pairs from {a_recs} records")
print(f"  files: {[Path(p).name for p in old_paths]}")

# New blind eval data
print("\n=== NEW blind-judge-multi-v1 data ===")
new_paths = [
    "eval_judgments_v2.jsonl",
    "eval_judgments_v3.jsonl",
    "eval_judgments_fresh.jsonl",
    "eval_judgments_intermodel.jsonl",
    "eval_judgments_gen5.jsonl",
    "eval_judgments_gen01.jsonl",
    "eval_judgments_preview.jsonl",
    "eval_judgments_v4_sanity.jsonl",
    "eval_judgments_v6_sanity.jsonl",
]
new_total = 0
for fname in new_paths:
    p = ROOT / fname
    if not p.exists():
        print(f"  {fname}: missing")
        continue
    pairs, recs = count_blindjudge([p])
    new_total += pairs
    print(f"  {fname}: {pairs} pairs from {recs} rounds")
print(f"  TOTAL new pairs: {new_total}")

print(f"\n=== GRAND TOTAL (human only + new) ===")
print(f"  {h_pairs + new_total} pairs")
print(f"\n=== GRAND TOTAL (incl auto + new) ===")
print(f"  {a_pairs + new_total} pairs")
