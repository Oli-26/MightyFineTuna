#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
For every id in reviewed.txt with no corresponding rank-v1 record
(matched by pending_id OR by prompt+candidates fingerprint), assume
it was skipped pre-patch and write a negative rank-v1 record:
all candidates tied at worst rank with suspect=True.

Idempotent — re-running won't double-write.
"""
from __future__ import annotations
import glob, hashlib, json, socket, time
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PREFS = ROOT / "prefs.jsonl"
REVIEWED = ROOT / "reviewed.txt"
HOST = socket.gethostname()


def cands_fp(cands: list[str]) -> str:
    h = hashlib.sha1()
    for c in cands: h.update((c or "").encode("utf-8", "replace")); h.update(b"\x00")
    return h.hexdigest()[:16]


def main() -> int:
    if not REVIEWED.exists():
        print("no reviewed.txt"); return 0
    reviewed_ids = {ln.strip() for ln in REVIEWED.read_text().splitlines() if ln.strip()}
    print(f"{len(reviewed_ids)} ids in reviewed.txt")

    # Load pending records (all sources)
    pending: dict[str, dict] = {}
    pending_paths = []
    if (ROOT / "pending.jsonl").exists(): pending_paths.append(str(ROOT / "pending.jsonl"))
    pending_paths.extend(sorted(glob.glob(str(RESULTS / "*-pending.jsonl"))))
    for fp in pending_paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            rid = r.get("id")
            if rid: pending.setdefault(rid, r)
    print(f"{len(pending)} pending records loaded from {len(pending_paths)} files")

    # Load rank-v1 coverage: by pending_id and by (prompt, candidates_fp)
    covered_ids: set[str] = set()
    covered_fps: set[tuple[str, str]] = set()
    rank_paths = sorted(glob.glob(str(RESULTS / "*-prefs.jsonl")))
    if PREFS.exists(): rank_paths.append(str(PREFS))
    for fp in rank_paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("schema") != "rank-v1": continue
            pid = r.get("pending_id")
            if pid: covered_ids.add(pid)
            covered_fps.add((r.get("prompt",""), cands_fp(r.get("candidates") or [])))
    print(f"{len(covered_ids)} rank-v1 records carry pending_id; {len(covered_fps)} unique (prompt, candsFP)")

    # For each reviewed id: write negative if pending exists AND not covered
    to_write = []
    no_pending = 0
    already_covered = 0
    for rid in reviewed_ids:
        pend = pending.get(rid)
        if not pend:
            no_pending += 1; continue
        if rid in covered_ids:
            already_covered += 1; continue
        fp = (pend.get("prompt",""), cands_fp(pend.get("candidates") or []))
        if fp in covered_fps:
            already_covered += 1; continue
        to_write.append(pend)

    print(f"\nplan: {len(to_write)} negative rank records to write")
    print(f"  skipped: {no_pending} (no pending rec found), {already_covered} (already covered)")

    if not to_write:
        print("nothing to do"); return 0

    n_added = 0
    with PREFS.open("a") as f:
        for pend in to_write:
            n = len(pend.get("candidates") or [])
            sp_id = pend.get("system_prompt_id")
            rec = {
                "ts": time.time(),
                "schema": "rank-v1",
                "model": pend.get("model") or "?",
                "host": HOST,
                "prompt": pend.get("prompt", ""),
                "rankings": [n] * n,
                "candidates": pend.get("candidates") or [],
                "errored": pend.get("errored") or [False] * n,
                "suspect": [True] * n,
                "temps": pend.get("temps"),
                "top_ps": pend.get("top_ps"),
                "min_ps": pend.get("min_ps"),
                "max_tokens": pend.get("max_tokens"),
                "system_prompt_ids": [sp_id] * n if sp_id else None,
                "pending_id": pend.get("id"),
                "skip_reason": "all_terrible_retroactive",
            }
            f.write(json.dumps(rec) + "\n")
            n_added += 1
    print(f"\nwrote {n_added} negative rank records to {PREFS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
