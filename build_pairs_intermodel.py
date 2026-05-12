"""
Build the gen4 training pairs jsonl from eval_judgments_intermodel.jsonl.
Uses the relabeled sides in sides_intermodel_tmp/ for code lookup.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TMP_DIR = ROOT / "sides_intermodel_tmp"
JUDGMENTS = ROOT / "eval_judgments_intermodel.jsonl"
OUT = ROOT / "training_pairs_intermodel.jsonl"

# Index: (label, prompt) -> code
idx = {}
for f in TMP_DIR.glob("*.jsonl"):
    label = f.stem  # "gen0-t05" etc
    for line in f.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        idx[(label, r["prompt"])] = r["code"]
print(f"indexed {len(idx)} (label,prompt) -> code entries")

pairs = []
missing = 0
for line in JUDGMENTS.read_text().splitlines():
    if not line.strip(): continue
    j = json.loads(line)
    prompt = j["prompt"]
    plr = j.get("per_label_rank") or {}
    labels = list(plr.keys())
    for a in labels:
        for b in labels:
            if plr[a] >= plr[b]: continue
            ca = idx.get((a, prompt)); cb = idx.get((b, prompt))
            if not ca or not cb: missing += 1; continue
            pairs.append({
                "prompt": prompt,
                "chosen": ca,
                "rejected": cb,
                "rank_chosen": plr[a],
                "rank_rejected": plr[b],
                "source": f"{a}>{b}",
            })

print(f"derivable pairs: {len(pairs)}  (missing-lookups: {missing})")
with OUT.open("w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"wrote {OUT}")

# Quick stats on which directions appear
from collections import Counter
direction_counts = Counter(p["source"].split(">")[0].split("-")[0] + ">" + p["source"].split(">")[1].split("-")[0] for p in pairs)
print("\npair direction counts (chosen-gen > rejected-gen):")
for d, c in sorted(direction_counts.items(), key=lambda x: -x[1]):
    print(f"  {d}: {c}")
