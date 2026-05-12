"""
Build a 4-way inter-model preference round file for gen2 vs gen3 vs gen4 vs gen5.
12 sides total: 4 gens x 3 temps. per_round=4, rounds-per-prompt=3 -> 120 rounds.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TMP = ROOT / "sides_gen5_eval_tmp"
TMP.mkdir(exist_ok=True)
for f in TMP.glob("*.jsonl"):
    f.unlink()

SOURCES = {
    # gen2: 3b-dpo-v2-2ep adapter outputs (round-a sides from v3 eval)
    "gen2-t05": ROOT / "sides_v3" / "side_v3a_t05_3b-dpo-v2-2ep.jsonl",
    "gen2-t07": ROOT / "sides_v3" / "side_v3a_t07_3b-dpo-v2-2ep.jsonl",
    "gen2-t09": ROOT / "sides_v3" / "side_v3a_t09_3b-dpo-v2-2ep.jsonl",
    # gen3: 3b-dpo-v3-2ep adapter outputs (v4 sanity)
    "gen3-t05": ROOT / "sides_v4" / "side_v4_t05_3b-dpo-v3-2ep.jsonl",
    "gen3-t07": ROOT / "sides_v4" / "side_v4_t07_3b-dpo-v3-2ep.jsonl",
    "gen3-t09": ROOT / "sides_v4" / "side_v4_t09_3b-dpo-v3-2ep.jsonl",
    # gen4: 3b-dpo-v4-2ep adapter outputs (gen4 sanity)
    "gen4-t05": ROOT / "sides_v5" / "side_v5_t05_3b-dpo-v4-2ep.jsonl",
    "gen4-t07": ROOT / "sides_v5" / "side_v5_t07_3b-dpo-v4-2ep.jsonl",
    "gen4-t09": ROOT / "sides_v5" / "side_v5_t09_3b-dpo-v4-2ep.jsonl",
    # gen5: 3b-dpo-v5-2ep adapter outputs (just generated)
    "gen5-t05": ROOT / "sides_v6" / "side_v6_t05_3b-dpo-v5-2ep.jsonl",
    "gen5-t07": ROOT / "sides_v6" / "side_v6_t07_3b-dpo-v5-2ep.jsonl",
    "gen5-t09": ROOT / "sides_v6" / "side_v6_t09_3b-dpo-v5-2ep.jsonl",
}

inputs = []
for label, src in SOURCES.items():
    if not src.exists():
        print(f"MISSING {src}", file=sys.stderr); sys.exit(1)
    out = TMP / f"{label}.jsonl"
    n = 0
    with out.open("w") as w:
        for line in src.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line); r["side"] = label
            w.write(json.dumps(r) + "\n"); n += 1
    print(f"{label}: {n} from {src.name}")
    inputs.append(str(out))

print(f"\n[invoke] build_rounds.py over {len(inputs)} files")
subprocess.check_call([
    sys.executable, str(ROOT / "build_rounds.py"),
    *inputs,
    "--per-round", "4",
    "--rounds-per-prompt", "3",
    "--out", "eval_pairs_gen5.jsonl",
], cwd=str(ROOT))
