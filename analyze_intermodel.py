import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

p = Path(__file__).parent / "eval_judgments_intermodel.jsonl"
recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f"total rounds judged: {len(recs)}")

# Per-label stats
ranks_per_label = defaultdict(list)
wins = Counter(); top2 = Counter(); last = Counter()
for r in recs:
    plr = r.get("per_label_rank") or {}
    for lbl, rk in plr.items():
        ranks_per_label[lbl].append(rk)
        if rk == 1: wins[lbl] += 1
        if rk <= 2: top2[lbl] += 1
        if rk == 4: last[lbl] += 1

print(f"\n{'label':<12} {'n':>3} {'mean':>6} {'std':>5} {'wins':>5} {'top-2':>6} {'last':>5}")
for lbl in sorted(ranks_per_label, key=lambda l: mean(ranks_per_label[l])):
    rs = ranks_per_label[lbl]
    m = mean(rs); s = stdev(rs) if len(rs) > 1 else 0.0
    print(f"{lbl:<12} {len(rs):>3} {m:>6.2f} {s:>5.2f} {wins[lbl]:>5} {top2[lbl]:>6} {last[lbl]:>5}")

# Aggregate by generation (collapse temps)
print("\n=== aggregated by generation ===")
gen_agg = defaultdict(list)
for lbl, rs in ranks_per_label.items():
    gen = lbl.split("-")[0]  # gen0 / gen1 / gen2 / gen3
    gen_agg[gen].extend(rs)
gen_wins = Counter(); gen_top2 = Counter(); gen_last = Counter()
for r in recs:
    plr = r.get("per_label_rank") or {}
    for lbl, rk in plr.items():
        gen = lbl.split("-")[0]
        if rk == 1: gen_wins[gen] += 1
        if rk <= 2: gen_top2[gen] += 1
        if rk == 4: gen_last[gen] += 1

print(f"{'gen':<6} {'n':>4} {'mean':>6} {'std':>5} {'wins':>5} {'top-2':>6} {'last':>5}")
for gen in sorted(gen_agg, key=lambda g: mean(gen_agg[g])):
    rs = gen_agg[gen]
    m = mean(rs); s = stdev(rs) if len(rs) > 1 else 0.0
    print(f"{gen:<6} {len(rs):>4} {m:>6.2f} {s:>5.2f} {gen_wins[gen]:>5} {gen_top2[gen]:>6} {gen_last[gen]:>5}")

# Aggregate by temp (collapse gens)
print("\n=== aggregated by temp ===")
temp_agg = defaultdict(list)
for lbl, rs in ranks_per_label.items():
    temp = lbl.split("-")[1]
    temp_agg[temp].extend(rs)
print(f"{'temp':<6} {'n':>4} {'mean':>6} {'std':>5}")
for t in sorted(temp_agg, key=lambda x: mean(temp_agg[x])):
    rs = temp_agg[t]
    m = mean(rs); s = stdev(rs) if len(rs) > 1 else 0.0
    print(f"{t:<6} {len(rs):>4} {m:>6.2f} {s:>5.2f}")

# Head-to-head: for each pair of gens, count wins
print("\n=== head-to-head wins (row beats column) ===")
gens_list = ["gen0", "gen1", "gen2", "gen3"]
h2h = {a: {b: 0 for b in gens_list} for a in gens_list}
for r in recs:
    plr = r.get("per_label_rank") or {}
    by_gen = {}
    for lbl, rk in plr.items():
        gen = lbl.split("-")[0]
        by_gen.setdefault(gen, []).append(rk)
    # for each gen pair in this round, if one has at least one lower rank than the other...
    for a in gens_list:
        for b in gens_list:
            if a == b: continue
            if a not in by_gen or b not in by_gen: continue
            best_a = min(by_gen[a])
            best_b = min(by_gen[b])
            if best_a < best_b: h2h[a][b] += 1
print(f"{'':<6}" + "".join(f"{g:>7}" for g in gens_list))
for a in gens_list:
    print(f"{a:<6}" + "".join(f"{h2h[a][b]:>7}" if a != b else "{:>7}".format("-") for b in gens_list))

# Total preference pairs derivable
total_pairs = 0
for r in recs:
    plr = r.get("per_label_rank") or {}
    total_pairs += sum(1 for a in plr.values() for b in plr.values() if a < b)
print(f"\ntotal preference pairs derivable: {total_pairs}")
