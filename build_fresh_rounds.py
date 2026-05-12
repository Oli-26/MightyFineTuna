"""
Build head-to-head gen3 vs gen4 rounds on fresh prompts (eval_prompts_v3.txt).
4 sides: gen3-t05, gen3-t07, gen4-t05, gen4-t07 -> 40 rounds, per_round=4.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TMP = ROOT / "sides_fresh_tmp"
TMP.mkdir(exist_ok=True)
for f in TMP.glob("*.jsonl"):
    f.unlink()

SOURCES = {
    "gen3-t05": ROOT / "sides_fresh" / "side_gen3_t05_3b-dpo-v3-2ep.jsonl",
    "gen3-t07": ROOT / "sides_fresh" / "side_gen3_t07_3b-dpo-v3-2ep.jsonl",
    "gen4-t05": ROOT / "sides_fresh" / "side_gen4_t05_3b-dpo-v4-2ep.jsonl",
    "gen4-t07": ROOT / "sides_fresh" / "side_gen4_t07_3b-dpo-v4-2ep.jsonl",
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
    print(f"{label}: {n} recs")
    inputs.append(str(out))

subprocess.check_call([
    sys.executable, str(ROOT / "build_rounds.py"),
    *inputs,
    "--per-round", "4",
    "--rounds-per-prompt", "1",
    "--out", "eval_pairs_fresh.jsonl",
], cwd=str(ROOT))
