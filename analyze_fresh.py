import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

p = Path(__file__).parent / "eval_judgments_fresh.jsonl"
recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f"rounds judged: {len(recs)}")

ranks = defaultdict(list)
wins = Counter(); top2 = Counter(); last = Counter()
for r in recs:
    for lbl, rk in (r.get("per_label_rank") or {}).items():
        ranks[lbl].append(rk)
        if rk == 1: wins[lbl] += 1
        if rk <= 2: top2[lbl] += 1
        if rk == 4: last[lbl] += 1

print(f"\n{'label':<10} {'n':>3} {'mean':>6} {'std':>5} {'wins':>5} {'top-2':>6} {'last':>5}")
for lbl in sorted(ranks, key=lambda l: mean(ranks[l])):
    rs = ranks[lbl]
    print(f"{lbl:<10} {len(rs):>3} {mean(rs):>6.2f} {stdev(rs):>5.2f} {wins[lbl]:>5} {top2[lbl]:>6} {last[lbl]:>5}")

print("\n=== aggregated by gen ===")
gen_ranks = defaultdict(list)
for lbl, rs in ranks.items():
    gen_ranks[lbl.split("-")[0]].extend(rs)
for gen in sorted(gen_ranks, key=lambda g: mean(gen_ranks[g])):
    rs = gen_ranks[gen]
    print(f"{gen:<6} n={len(rs)} mean={mean(rs):.2f} std={stdev(rs):.2f}")

# Head-to-head: per round, which gen's best beat which
gens_list = ["gen3", "gen4"]
h2h = {a: {b: 0 for b in gens_list} for a in gens_list}
ties = 0
for r in recs:
    plr = r.get("per_label_rank") or {}
    by_gen = {}
    for lbl, rk in plr.items():
        by_gen.setdefault(lbl.split("-")[0], []).append(rk)
    best3 = min(by_gen.get("gen3", [99]))
    best4 = min(by_gen.get("gen4", [99]))
    if best3 < best4: h2h["gen3"]["gen4"] += 1
    elif best4 < best3: h2h["gen4"]["gen3"] += 1
    else: ties += 1
print(f"\nhead-to-head (best-of-2 each):")
print(f"  gen3 best: {h2h['gen3']['gen4']}")
print(f"  gen4 best: {h2h['gen4']['gen3']}")
print(f"  tied:      {ties}")

# Per-temp: gen3-t05 vs gen4-t05 + gen3-t07 vs gen4-t07
print("\n=== same-temp matchups ===")
for t in ("t05", "t07"):
    a = f"gen3-{t}"; b = f"gen4-{t}"
    wins_a = wins_b = ties_t = 0
    for r in recs:
        plr = r.get("per_label_rank") or {}
        if a in plr and b in plr:
            if plr[a] < plr[b]: wins_a += 1
            elif plr[b] < plr[a]: wins_b += 1
            else: ties_t += 1
    print(f"  {t}: gen3 wins {wins_a}, gen4 wins {wins_b}, ties {ties_t}")

total_pairs = sum(sum(1 for a in (r.get("per_label_rank") or {}).values() for b in (r.get("per_label_rank") or {}).values() if a < b) for r in recs)
print(f"\ntotal derivable pairs: {total_pairs}")
