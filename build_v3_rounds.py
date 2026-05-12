"""
Rewrite the 'side' field of each v3 side file to include round + temp so that
build_rounds.py can distinguish the 9 sources, then call build_rounds inline.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SIDES_DIR = ROOT / "sides_v3"
TMP_DIR = ROOT / "sides_v3_relabeled"
TMP_DIR.mkdir(exist_ok=True)

# Map filename patterns to round-tag
NAME_RE = re.compile(r"side_(v3[ab]|v3_wild)_t(\d+)_3b-dpo-v2-2ep\.jsonl")

inputs = []
for f in sorted(SIDES_DIR.glob("*.jsonl")):
    m = NAME_RE.match(f.name)
    if not m:
        print(f"skip (no match): {f.name}", file=sys.stderr); continue
    round_id, temp_tag = m.group(1), m.group(2)
    short_round = round_id.replace("v3_", "").replace("v3", "")  # 'a', 'b', 'wild'
    if not short_round: short_round = "a"  # defensive
    label_suffix = f"{short_round}-t{temp_tag}"
    out_f = TMP_DIR / f.name
    n = 0
    with out_f.open("w") as w:
        for line in f.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            r["side"] = f"{r.get('side','3b-dpo-v2-2ep')}-{label_suffix}"
            w.write(json.dumps(r) + "\n")
            n += 1
    print(f"{f.name} -> side={r['side']} ({n} recs)")
    inputs.append(str(out_f))

print(f"\n[invoke] build_rounds.py over {len(inputs)} files")
cmd = [
    sys.executable, str(ROOT / "build_rounds.py"),
    *inputs,
    "--per-round", "4",
    "--rounds-per-prompt", "2",
    "--out", "eval_pairs_v3.jsonl",
]
subprocess.check_call(cmd, cwd=str(ROOT))
