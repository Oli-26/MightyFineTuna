"""
Build a combined preference-pairs jsonl for the next DPO round.

- Old data: results/*-prefs.jsonl + prefs.jsonl, human-rated only, gap>=2
- New v2 data: eval_judgments_v2.jsonl, lookup code in sides_v2/
- New v3 data: eval_judgments_v3.jsonl, lookup code in sides_v3/

Output: one pair per line as {prompt, chosen, rejected, rank_chosen, rank_rejected, source}
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PRIVATE_PREFS = ROOT / "prefs.jsonl"
OUT = ROOT / "training_pairs_v3.jsonl"


def fingerprint(rec: dict) -> str:
    cands = rec.get("candidates") or []
    h = hashlib.sha1()
    for c in cands:
        h.update((c or "").encode("utf-8", "replace")); h.update(b"\x00")
    return f"{rec.get('ts',0):.3f}|{rec.get('prompt','')}|{h.hexdigest()[:16]}"


def collect_old_pairs(min_gap: int = 2) -> list[dict]:
    """Collect human-rated old pairs with rank gap >= min_gap."""
    seen: dict[str, dict] = {}
    paths = sorted(glob.glob(str(RESULTS / "*-prefs.jsonl")))
    if PRIVATE_PREFS.exists():
        paths.append(str(PRIVATE_PREFS))
    for fp in paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("schema") != "rank-v1": continue
            if r.get("obsolete"): continue
            if r.get("auto_rated"): continue  # drop auto-rated
            seen.setdefault(fingerprint(r), r)

    pairs = []
    for rec in seen.values():
        cands = rec.get("candidates") or []
        ranks = rec.get("rankings") or []
        errs = rec.get("errored") or [False] * len(cands)
        sus = rec.get("suspect") or [False] * len(cands)
        valid = []
        for i, rk in enumerate(ranks):
            if rk is None: continue
            if errs[i] or sus[i]: continue
            if not cands[i]: continue
            valid.append((i, rk))
        for a_idx, a_rk in valid:
            for b_idx, b_rk in valid:
                if a_rk >= b_rk: continue
                gap = b_rk - a_rk
                if gap < min_gap: continue
                pairs.append({
                    "prompt": rec["prompt"],
                    "chosen": cands[a_idx],
                    "rejected": cands[b_idx],
                    "rank_chosen": a_rk,
                    "rank_rejected": b_rk,
                    "source": "old",
                })
    return pairs


def index_sides(side_dir: Path, label_to_path: dict[str, str]) -> dict[tuple[str, str], str]:
    """label -> {prompt -> code}"""
    out = {}
    for label, fname in label_to_path.items():
        fp = side_dir / fname
        if not fp.exists():
            print(f"WARNING: missing {fp}", file=sys.stderr); continue
        for line in fp.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            out[(label, r["prompt"])] = r["code"]
    return out


def collect_new_v2_pairs() -> list[dict]:
    jp = ROOT / "eval_judgments_v2.jsonl"
    if not jp.exists():
        print(f"no {jp.name}", file=sys.stderr); return []
    # v2 labels: 3b-dpo-v2-2ep-t05/t07/t09/t11
    label_to_path = {f"3b-dpo-v2-2ep-t{t:02d}": f"side_v2_t{t:02d}_3b-dpo-v2-2ep.jsonl"
                     for t in (5, 7, 9, 11)}
    idx = index_sides(ROOT / "sides_v2", label_to_path)
    pairs = []
    for line in jp.read_text().splitlines():
        if not line.strip(): continue
        j = json.loads(line)
        prompt = j["prompt"]
        plr = j.get("per_label_rank") or {}
        labels = list(plr.keys())
        for a in labels:
            for b in labels:
                if plr[a] >= plr[b]: continue
                ca = idx.get((a, prompt)); cb = idx.get((b, prompt))
                if not ca or not cb: continue
                pairs.append({
                    "prompt": prompt,
                    "chosen": ca,
                    "rejected": cb,
                    "rank_chosen": plr[a],
                    "rank_rejected": plr[b],
                    "source": "v2",
                })
    return pairs


def collect_new_v3_pairs() -> list[dict]:
    jp = ROOT / "eval_judgments_v3.jsonl"
    if not jp.exists():
        print(f"no {jp.name}", file=sys.stderr); return []
    # v3 labels: 3b-dpo-v2-2ep-{a|b|wild}-t{tt}
    label_to_path = {}
    for rnd in ("a", "b"):
        for t in (5, 7, 9, 11):
            lbl = f"3b-dpo-v2-2ep-{rnd}-t{t:02d}"
            label_to_path[lbl] = f"side_v3{rnd}_t{t:02d}_3b-dpo-v2-2ep.jsonl"
    label_to_path["3b-dpo-v2-2ep-wild-t07"] = "side_v3_wild_t07_3b-dpo-v2-2ep.jsonl"
    idx = index_sides(ROOT / "sides_v3", label_to_path)
    pairs = []
    for line in jp.read_text().splitlines():
        if not line.strip(): continue
        j = json.loads(line)
        prompt = j["prompt"]
        plr = j.get("per_label_rank") or {}
        labels = list(plr.keys())
        for a in labels:
            for b in labels:
                if plr[a] >= plr[b]: continue
                ca = idx.get((a, prompt)); cb = idx.get((b, prompt))
                if not ca or not cb: continue
                pairs.append({
                    "prompt": prompt,
                    "chosen": ca,
                    "rejected": cb,
                    "rank_chosen": plr[a],
                    "rank_rejected": plr[b],
                    "source": "v3",
                })
    return pairs


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gap", type=int, default=1)
    args = ap.parse_args()
    old = collect_old_pairs(min_gap=args.min_gap)
    v2 = collect_new_v2_pairs()
    v3 = collect_new_v3_pairs()
    total = old + v2 + v3
    new_count = len(v2) + len(v3)
    print(f"old (human, gap>=2): {len(old)}")
    print(f"new v2:              {len(v2)}")
    print(f"new v3:              {len(v3)}")
    print(f"total:               {len(total)}")
    print(f"new ratio:           {new_count / len(total) * 100:.1f}%")

    with OUT.open("w") as f:
        for p in total:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
