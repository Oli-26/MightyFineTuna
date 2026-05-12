import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

p = Path(__file__).parent / "eval_judgments_gen5.jsonl"
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

print(f"\n{'label':<12} {'n':>3} {'mean':>6} {'std':>5} {'wins':>5} {'top-2':>6} {'last':>5}")
for lbl in sorted(ranks, key=lambda l: mean(ranks[l])):
    rs = ranks[lbl]
    s = stdev(rs) if len(rs) > 1 else 0.0
    print(f"{lbl:<12} {len(rs):>3} {mean(rs):>6.2f} {s:>5.2f} {wins[lbl]:>5} {top2[lbl]:>6} {last[lbl]:>5}")

print("\n=== aggregated by gen ===")
gen_agg = defaultdict(list)
for lbl, rs in ranks.items():
    gen_agg[lbl.split("-")[0]].extend(rs)
gen_wins = Counter(); gen_top2 = Counter(); gen_last = Counter()
for r in recs:
    for lbl, rk in (r.get("per_label_rank") or {}).items():
        gen = lbl.split("-")[0]
        if rk == 1: gen_wins[gen] += 1
        if rk <= 2: gen_top2[gen] += 1
        if rk == 4: gen_last[gen] += 1
print(f"{'gen':<6} {'n':>4} {'mean':>6} {'std':>5} {'wins':>5} {'top-2':>6} {'last':>5}")
for gen in sorted(gen_agg, key=lambda g: mean(gen_agg[g])):
    rs = gen_agg[gen]
    print(f"{gen:<6} {len(rs):>4} {mean(rs):>6.2f} {stdev(rs):>5.2f} {gen_wins[gen]:>5} {gen_top2[gen]:>6} {gen_last[gen]:>5}")

print("\n=== aggregated by temp ===")
temp_agg = defaultdict(list)
for lbl, rs in ranks.items():
    temp_agg[lbl.split("-")[1]].extend(rs)
for t in sorted(temp_agg, key=lambda x: mean(temp_agg[x])):
    rs = temp_agg[t]
    print(f"{t}: n={len(rs)} mean={mean(rs):.2f}")

# Head-to-head per round (each gen's best vs each gen's best)
gens_list = ["gen2", "gen3", "gen4", "gen5"]
h2h = {a: {b: 0 for b in gens_list} for a in gens_list}
for r in recs:
    by_gen = {}
    for lbl, rk in (r.get("per_label_rank") or {}).items():
        by_gen.setdefault(lbl.split("-")[0], []).append(rk)
    for a in gens_list:
        for b in gens_list:
            if a == b or a not in by_gen or b not in by_gen: continue
            if min(by_gen[a]) < min(by_gen[b]): h2h[a][b] += 1

print("\n=== head-to-head (row best beats column best, per round) ===")
print(f"{'':<6}" + "".join(f"{g:>7}" for g in gens_list))
for a in gens_list:
    row = ""
    for b in gens_list:
        row += "{:>7}".format("-" if a == b else h2h[a][b])
    print(f"{a:<6}{row}")

total_pairs = sum(sum(1 for a in (r.get("per_label_rank") or {}).values() for b in (r.get("per_label_rank") or {}).values() if a < b) for r in recs)
print(f"\ntotal derivable pairs: {total_pairs}")
