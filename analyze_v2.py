import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

p = Path(__file__).parent / "eval_judgments_v2.jsonl"
recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f"total judgments: {len(recs)}")

ranks_per_temp = {}
wins = Counter()
top2 = Counter()
last = Counter()
for r in recs:
    plr = r.get("per_label_rank") or {}
    for lbl, rk in plr.items():
        temp = lbl.split("-t")[-1]
        ranks_per_temp.setdefault(temp, []).append(rk)
        if rk == 1: wins[temp] += 1
        if rk <= 2: top2[temp] += 1
        if rk == 4: last[temp] += 1

print(f"\n{'temp':<6} {'n':>3} {'mean':>6} {'std':>5} {'wins':>5} {'top-2':>6} {'last':>5}")
for temp in sorted(ranks_per_temp):
    rs = ranks_per_temp[temp]
    m = mean(rs); s = stdev(rs) if len(rs) > 1 else 0.0
    print(f"t{temp:<5} {len(rs):>3} {m:>6.2f} {s:>5.2f} {wins[temp]:>5} {top2[temp]:>6} {last[temp]:>5}")

total_pairs = 0
for r in recs:
    plr = r.get("per_label_rank") or {}
    if len(plr) >= 2: total_pairs += sum(1 for a in plr.values() for b in plr.values() if a < b)
print(f"\ntotal preference pairs derivable: {total_pairs}")
