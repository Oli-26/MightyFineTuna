import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

p = Path(__file__).parent / "eval_judgments_v3.jsonl"
recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f"total judgments: {len(recs)}")

ranks_per_label = {}
wins = Counter()
top2 = Counter()
last = Counter()
for r in recs:
    plr = r.get("per_label_rank") or {}
    for lbl, rk in plr.items():
        ranks_per_label.setdefault(lbl, []).append(rk)
        if rk == 1: wins[lbl] += 1
        if rk <= 2: top2[lbl] += 1
        if rk == 4: last[lbl] += 1

print(f"\n{'label':<28} {'n':>3} {'mean':>6} {'std':>5} {'wins':>5} {'top-2':>6} {'last':>5}")
for lbl in sorted(ranks_per_label, key=lambda l: mean(ranks_per_label[l])):
    rs = ranks_per_label[lbl]
    m = mean(rs); s = stdev(rs) if len(rs) > 1 else 0.0
    print(f"{lbl:<28} {len(rs):>3} {m:>6.2f} {s:>5.2f} {wins[lbl]:>5} {top2[lbl]:>6} {last[lbl]:>5}")

# Aggregate per-temp (combine a+b rounds, keep wildcard separate)
print("\n=== aggregated by temp (a+b combined) ===")
agg = {}
for lbl, rs in ranks_per_label.items():
    if "-wild-" in lbl:
        key = "wild-t07"
    else:
        # 3b-dpo-v2-2ep-a-t05 -> t05
        key = lbl.split("-")[-1]
    agg.setdefault(key, []).extend(rs)

print(f"{'temp':<10} {'n':>4} {'mean':>6} {'std':>5}")
for k in sorted(agg, key=lambda x: mean(agg[x])):
    rs = agg[k]
    m = mean(rs); s = stdev(rs) if len(rs) > 1 else 0.0
    print(f"{k:<10} {len(rs):>4} {m:>6.2f} {s:>5.2f}")

total_pairs = sum(sum(1 for a in (r.get("per_label_rank") or {}).values() for b in (r.get("per_label_rank") or {}).values() if a < b) for r in recs)
print(f"\ntotal preference pairs derivable: {total_pairs}")
