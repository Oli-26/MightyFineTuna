"""
Build a 4-way inter-model preference round file.
12 sides total: gen0/gen1/gen2/gen3 × 3 temps each (0.5/0.7/0.9).
per_round=4, rounds-per-prompt=3 -> 120 rounds, ~720 derivable pairs.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TMP_DIR = ROOT / "sides_intermodel_tmp"
TMP_DIR.mkdir(exist_ok=True)
# Clean tmp
for f in TMP_DIR.glob("*.jsonl"):
    f.unlink()

# Source files per (gen, temp). Use single representative temp variant per model.
SOURCES = {
    # gen0: parent_b_v2 SFT only
    ("gen0", "t05"): ROOT / "sides_gen01" / "side_gen0_t05_3b-parent_b_v2.jsonl",
    ("gen0", "t07"): ROOT / "sides_gen01" / "side_gen0_t07_3b-parent_b_v2.jsonl",
    ("gen0", "t09"): ROOT / "sides_gen01" / "side_gen0_t09_3b-parent_b_v2.jsonl",
    # gen1: + DPO v1 (standalone merged base)
    ("gen1", "t05"): ROOT / "sides_gen01" / "side_gen1_t05_none.jsonl",
    ("gen1", "t07"): ROOT / "sides_gen01" / "side_gen1_t07_none.jsonl",
    ("gen1", "t09"): ROOT / "sides_gen01" / "side_gen1_t09_none.jsonl",
    # gen2: + DPO v2 — use round-a sides from v3 eval
    ("gen2", "t05"): ROOT / "sides_v3" / "side_v3a_t05_3b-dpo-v2-2ep.jsonl",
    ("gen2", "t07"): ROOT / "sides_v3" / "side_v3a_t07_3b-dpo-v2-2ep.jsonl",
    ("gen2", "t09"): ROOT / "sides_v3" / "side_v3a_t09_3b-dpo-v2-2ep.jsonl",
    # gen3: + DPO v3 — use sanity-round sides from v4
    ("gen3", "t05"): ROOT / "sides_v4" / "side_v4_t05_3b-dpo-v3-2ep.jsonl",
    ("gen3", "t07"): ROOT / "sides_v4" / "side_v4_t07_3b-dpo-v3-2ep.jsonl",
    ("gen3", "t09"): ROOT / "sides_v4" / "side_v4_t09_3b-dpo-v3-2ep.jsonl",
}

inputs = []
for (gen, temp), src in SOURCES.items():
    if not src.exists():
        print(f"MISSING: {src}", file=sys.stderr); sys.exit(1)
    label = f"{gen}-{temp}"
    out = TMP_DIR / f"{label}.jsonl"
    n = 0
    with out.open("w") as w:
        for line in src.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            r["side"] = label  # unique label per (gen, temp)
            w.write(json.dumps(r) + "\n")
            n += 1
    print(f"{label}: {n} from {src.name}")
    inputs.append(str(out))

print(f"\n[invoke] build_rounds.py over {len(inputs)} files")
subprocess.check_call([
    sys.executable, str(ROOT / "build_rounds.py"),
    *inputs,
    "--per-round", "4",
    "--rounds-per-prompt", "3",
    "--out", "eval_pairs_intermodel.jsonl",
], cwd=str(ROOT))
