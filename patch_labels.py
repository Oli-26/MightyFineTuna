import json
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_pairs_v2.jsonl")
if not src.is_absolute():
    src = Path(__file__).parent / src
out_lines = []
for line in src.read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    for c in r["candidates"]:
        t = c.get("temp")
        tag = f"t{int(round(t * 10)):02d}" if t is not None else "tNA"
        c["label"] = f"{c['label']}-{tag}"
    out_lines.append(json.dumps(r))
src.write_text("\n".join(out_lines) + "\n")
print(f"patched {len(out_lines)} rounds in {src.name}")
