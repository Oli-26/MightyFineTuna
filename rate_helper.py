#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
In-session rating helper for Claude Code (no external API needed).

Modes:
  fetch N OUT      → write next N unreviewed pending records to OUT (json array)
                     each entry: {id, prompt, candidates: [str, ...], n}
  commit FILE      → ingest rankings, write rank-v1 records, mark reviewed
                     FILE format: [{id, rankings: [..], confidence: "high|medium|low",
                                    reasoning?: ""}, ...]
                     low confidence → skipped (not committed, not marked reviewed)
"""
from __future__ import annotations
import argparse, glob, json, socket, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PREFS = ROOT / "prefs.jsonl"
REVIEWED = ROOT / "reviewed.txt"
HOST = socket.gethostname()


def load_pending() -> dict[str, dict]:
    out: dict[str, dict] = {}
    paths = []
    if (ROOT / "pending.jsonl").exists(): paths.append(str(ROOT / "pending.jsonl"))
    paths.extend(sorted(glob.glob(str(RESULTS / "*-pending.jsonl"))))
    for fp in paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            rid = r.get("id")
            if rid: out.setdefault(rid, r)
    return out


def reviewed_ids() -> set[str]:
    if not REVIEWED.exists(): return set()
    return {ln.strip() for ln in REVIEWED.read_text().splitlines() if ln.strip()}


def already_rated_ids() -> set[str]:
    out = set()
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
            if pid: out.add(pid)
    return out


def cmd_fetch(n: int, out_path: Path) -> int:
    pending = load_pending()
    skip = reviewed_ids() | already_rated_ids()
    todo = [pending[rid] for rid in pending if rid not in skip][:n]
    payload = []
    for r in todo:
        payload.append({
            "id": r["id"],
            "prompt": r.get("prompt", ""),
            "candidates": r.get("candidates") or [],
            "n": len(r.get("candidates") or []),
        })
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"fetched {len(payload)} of {n} requested → {out_path}", file=sys.stderr)
    return 0


import re as _re
_DRAW_RE = _re.compile(r"\b(?:fillRect|strokeRect|arc\(|fill\(|stroke\(|moveTo|lineTo|ellipse\(|drawImage)")
_FILLSTYLE_RE = _re.compile(r"fillStyle\s*=")


_RECT_RE = _re.compile(r"\bfillRect\(")
_ARC_RE = _re.compile(r"\b(?:arc\(|ellipse\(|moveTo|lineTo|bezierCurveTo|quadraticCurveTo)")
_LOOP_RE = _re.compile(r"\bfor\s*\(|\bwhile\s*\(")
_PALETTE_RE = _re.compile(r"\bconst\s+\w+\s*=\s*['\"]#")
# Pathological: same fillRect with nearly-identical args repeated (e.g. y+=10 stripes filling whole canvas)
_REPEATED_RECT_RE = _re.compile(r"(fillRect\([^)]+\)\s*;\s*\n\s*){8,}")


def quality_score(code: str) -> tuple[str, str]:
    """Return (verdict, reason). Verdict in {'bad', 'ok', 'good'}."""
    if not code: return ("bad", "empty")
    L = len(code)
    if L < 200: return ("bad", f"too short ({L})")
    if L > 4000: return ("bad", f"likely pathological ({L})")
    if _REPEATED_RECT_RE.search(code): return ("bad", "repetitive fillRect spam")
    if not _DRAW_RE.search(code): return ("bad", "no draw calls")
    last = code.rstrip()[-30:].split("\n")[-1]
    if last and not (last.endswith("}") or last.endswith(";") or last.endswith(")")
                     or last.endswith(",") or last.endswith(":")):
        return ("bad", f"truncated: ...{last[-20:]!r}")
    n_fillstyle = len(_FILLSTYLE_RE.findall(code))
    if n_fillstyle < 2: return ("bad", f"only {n_fillstyle} fillStyle (no palette)")
    if 400 <= L <= 2500 and n_fillstyle >= 4: return ("good", f"L={L} fs={n_fillstyle}")
    return ("ok", f"L={L} fs={n_fillstyle}")


def effort_score(code: str) -> float:
    """Composite proxy for visual richness. Higher = more elaborate."""
    if not code: return 0.0
    L = len(code)
    n_fs = len(_FILLSTYLE_RE.findall(code))
    n_rect = len(_RECT_RE.findall(code))
    n_arc = len(_ARC_RE.findall(code))
    has_loop = 1.0 if _LOOP_RE.search(code) else 0.0
    has_palette = 1.0 if _PALETTE_RE.search(code) else 0.0
    has_both_shapes = 1.0 if (n_rect > 0 and n_arc > 0) else 0.5
    # length sweet spot 700-2000
    if 700 <= L <= 2000: l_score = 1.0
    elif 400 <= L < 700: l_score = 0.7
    elif 2000 < L <= 3000: l_score = 0.85
    else: l_score = 0.4
    fs_score = min(n_fs / 6.0, 1.5)
    draw_score = min((n_rect + n_arc) / 15.0, 1.5)
    return (l_score + fs_score + draw_score + has_loop * 0.3
            + has_palette * 0.2 + has_both_shapes * 0.4)


def ranks_from_scores(scores: list[float]) -> tuple[list[int], float]:
    """Convert scores to dense ranks (1=best). Return (ranks, gap_min_to_next).
    gap_min_to_next: smallest absolute gap between adjacent rank scores — close to 0 means tie."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: -scores[i])
    ranks = [0] * n
    for rank_pos, idx in enumerate(order):
        ranks[idx] = rank_pos + 1
    sorted_scores = sorted(scores, reverse=True)
    gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(n-1)]
    return ranks, min(gaps) if gaps else 0.0


def cmd_screen(n: int, unclear_out: Path) -> int:
    """Heuristic pre-screen: auto-rate clear-cut records, defer ambiguous to human.
    Records with 1 clear winner + ≥2 clear losers are auto-committed.
    Records where all candidates score equally are written to unclear_out for human review.
    All-bad records get tied-worst suspect rank (like skip)."""
    pending = load_pending()
    skip = reviewed_ids() | already_rated_ids()
    todo = [pending[rid] for rid in pending if rid not in skip][:n]
    if not todo: print("nothing to do"); return 0

    n_auto = 0
    n_all_bad = 0
    n_unclear = 0
    unclear_payload = []
    auto_recs = []
    auto_reviewed = []

    for pend in todo:
        cands = pend.get("candidates") or []
        n_cand = len(cands)
        if n_cand < 2: continue
        scores = [quality_score(c) for c in cands]
        verdicts = [s[0] for s in scores]
        n_good = sum(1 for v in verdicts if v == "good")
        n_ok = sum(1 for v in verdicts if v == "ok")
        n_bad = sum(1 for v in verdicts if v == "bad")

        if n_bad == n_cand:
            # All terrible → tied worst, suspect=true (like skip)
            sp_id = pend.get("system_prompt_id")
            rec = {
                "ts": time.time(), "schema": "rank-v1",
                "model": pend.get("model") or "?", "host": HOST,
                "prompt": pend.get("prompt", ""),
                "rankings": [n_cand] * n_cand,
                "candidates": cands,
                "errored": pend.get("errored") or [False] * n_cand,
                "suspect": [True] * n_cand,
                "temps": pend.get("temps"),
                "top_ps": pend.get("top_ps"),
                "min_ps": pend.get("min_ps"),
                "max_tokens": pend.get("max_tokens"),
                "system_prompt_ids": [sp_id] * n_cand if sp_id else None,
                "pending_id": pend["id"],
                "auto_rated": True,
                "auto_judge": "heuristic-allbad",
                "auto_confidence": "high",
                "auto_reasoning": "; ".join(f"c{i}={s[1]}" for i, s in enumerate(scores)),
                "skip_reason": "all_terrible_heuristic",
            }
            auto_recs.append(rec)
            auto_reviewed.append(pend["id"])
            n_all_bad += 1
            continue

        # Clear winner case: exactly 1 good/ok, others all bad
        if (n_good + n_ok) == 1 and n_bad >= 1:
            # Find non-bad index → rank 1, bad indices → rank n_cand
            ranks = []
            for v in verdicts:
                ranks.append(1 if v != "bad" else n_cand)
            sp_id = pend.get("system_prompt_id")
            rec = {
                "ts": time.time(), "schema": "rank-v1",
                "model": pend.get("model") or "?", "host": HOST,
                "prompt": pend.get("prompt", ""),
                "rankings": ranks,
                "candidates": cands,
                "errored": pend.get("errored") or [False] * n_cand,
                "suspect": [v == "bad" for v in verdicts],
                "temps": pend.get("temps"),
                "top_ps": pend.get("top_ps"),
                "min_ps": pend.get("min_ps"),
                "max_tokens": pend.get("max_tokens"),
                "system_prompt_ids": [sp_id] * n_cand if sp_id else None,
                "pending_id": pend["id"],
                "auto_rated": True,
                "auto_judge": "heuristic-clearwin",
                "auto_confidence": "high",
                "auto_reasoning": "; ".join(f"c{i}={s[0]}({s[1]})" for i, s in enumerate(scores)),
            }
            auto_recs.append(rec)
            auto_reviewed.append(pend["id"])
            n_auto += 1
            continue

        # All non-bad: rank by effort heuristic. If gap large enough → auto. Else defer.
        bad_idxs = [i for i, v in enumerate(verdicts) if v == "bad"]
        nonbad_idxs = [i for i in range(n_cand) if i not in bad_idxs]
        nonbad_scores = [effort_score(cands[i]) for i in nonbad_idxs]
        nb_ranks, gap = ranks_from_scores(nonbad_scores)
        score_range = (max(nonbad_scores) - min(nonbad_scores)) if len(nonbad_scores) > 1 else 0.0

        # Defer if too close (gap < 0.25 AND range < 0.5)
        if score_range < 0.5 and gap < 0.25 and len(nonbad_idxs) > 1:
            unclear_payload.append({
                "id": pend["id"],
                "prompt": pend.get("prompt", ""),
                "candidates": cands,
                "verdicts": verdicts,
                "verdict_reasons": [s[1] for s in scores],
                "effort_scores": [round(effort_score(c), 2) for c in cands],
                "n": n_cand,
            })
            n_unclear += 1
            continue

        # Auto-rate via effort: nonbad get nb_ranks (1..k); bad get rank n_cand
        ranks = [n_cand] * n_cand
        for pos, src_i in enumerate(nonbad_idxs):
            ranks[src_i] = nb_ranks[pos]
        sp_id = pend.get("system_prompt_id")
        rec = {
            "ts": time.time(), "schema": "rank-v1",
            "model": pend.get("model") or "?", "host": HOST,
            "prompt": pend.get("prompt", ""),
            "rankings": ranks,
            "candidates": cands,
            "errored": pend.get("errored") or [False] * n_cand,
            "suspect": [v == "bad" for v in verdicts],
            "temps": pend.get("temps"),
            "top_ps": pend.get("top_ps"),
            "min_ps": pend.get("min_ps"),
            "max_tokens": pend.get("max_tokens"),
            "system_prompt_ids": [sp_id] * n_cand if sp_id else None,
            "pending_id": pend["id"],
            "auto_rated": True,
            "auto_judge": "heuristic-effort",
            "auto_confidence": "medium",
            "auto_reasoning": "; ".join(
                f"c{i}={s[0]} eff={effort_score(cands[i]):.2f}"
                for i, s in enumerate(scores)),
        }
        auto_recs.append(rec)
        auto_reviewed.append(pend["id"])
        n_auto += 1

    # Commit auto-rates
    if auto_recs:
        with PREFS.open("a") as f:
            for r in auto_recs: f.write(json.dumps(r) + "\n")
        with REVIEWED.open("a") as f:
            for rid in auto_reviewed: f.write(rid + "\n")
    unclear_out.write_text(json.dumps(unclear_payload, indent=2))
    print(f"screened {len(todo)} records:")
    print(f"  auto-rated (clear winner): {n_auto}")
    print(f"  auto-skipped (all bad):    {n_all_bad}")
    print(f"  unclear → {unclear_out.name}: {n_unclear}")
    return 0


def cmd_commit(in_path: Path) -> int:
    pending = load_pending()
    try:
        rankings_input = json.loads(in_path.read_text())
    except Exception as e:
        print(f"failed to parse {in_path}: {e}", file=sys.stderr); return 1
    if not isinstance(rankings_input, list):
        print("expected JSON array", file=sys.stderr); return 1

    n_high = n_med = n_low = n_skip = 0
    with PREFS.open("a") as fout, REVIEWED.open("a") as freview:
        for entry in rankings_input:
            rid = entry.get("id")
            ranks = entry.get("rankings")
            conf = (entry.get("confidence") or "").lower()
            reasoning = entry.get("reasoning", "")
            if not rid or rid not in pending:
                print(f"  skip: unknown id {rid}", file=sys.stderr); n_skip += 1; continue
            pend = pending[rid]
            n = len(pend.get("candidates") or [])
            if conf == "low":
                n_low += 1
                continue
            if not (isinstance(ranks, list) and len(ranks) == n
                    and all(isinstance(r, int) and 1 <= r <= n for r in ranks)):
                print(f"  bad ranks for {rid}: {ranks}", file=sys.stderr); n_skip += 1; continue
            sp_id = pend.get("system_prompt_id")
            rec = {
                "ts": time.time(),
                "schema": "rank-v1",
                "model": pend.get("model") or "?",
                "host": HOST,
                "prompt": pend.get("prompt", ""),
                "rankings": ranks,
                "candidates": pend.get("candidates") or [],
                "errored": pend.get("errored") or [False] * n,
                "suspect": pend.get("suspect") or [False] * n,
                "temps": pend.get("temps"),
                "top_ps": pend.get("top_ps"),
                "min_ps": pend.get("min_ps"),
                "max_tokens": pend.get("max_tokens"),
                "system_prompt_ids": [sp_id] * n if sp_id else None,
                "pending_id": rid,
                "auto_rated": True,
                "auto_judge": "claude-opus-4-7-via-claude-code",
                "auto_confidence": conf,
                "auto_reasoning": reasoning,
            }
            fout.write(json.dumps(rec) + "\n")
            freview.write(rid + "\n")
            if conf == "high": n_high += 1
            elif conf == "medium": n_med += 1
    print(f"committed: high={n_high} medium={n_med} deferred(low)={n_low} skipped={n_skip}",
          file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sf = sub.add_parser("fetch")
    sf.add_argument("n", type=int)
    sf.add_argument("out")
    sc = sub.add_parser("commit")
    sc.add_argument("in_file")
    ss = sub.add_parser("screen")
    ss.add_argument("n", type=int)
    ss.add_argument("unclear_out")
    sft = sub.add_parser("force-effort", help="Effort-rank ALL remaining unreviewed pending; skips human judgment")
    sft.add_argument("n", type=int, default=10000)
    args = ap.parse_args()
    if args.cmd == "fetch":
        return cmd_fetch(args.n, Path(args.out))
    elif args.cmd == "commit":
        return cmd_commit(Path(args.in_file))
    elif args.cmd == "screen":
        return cmd_screen(args.n, Path(args.unclear_out))
    elif args.cmd == "force-effort":
        return cmd_force_effort(args.n)
    return 1


def cmd_force_effort(n: int) -> int:
    """Auto-rank all remaining unreviewed pending using effort heuristic.
    Bad candidates → rank n_cand. Non-bad ranked by effort_score."""
    pending = load_pending()
    skip = reviewed_ids() | already_rated_ids()
    todo = [pending[rid] for rid in pending if rid not in skip][:n]
    if not todo: print("nothing to do"); return 0

    n_committed = 0
    n_all_bad = 0
    auto_recs = []
    auto_reviewed = []

    for pend in todo:
        cands = pend.get("candidates") or []
        n_cand = len(cands)
        if n_cand < 2: continue
        scores = [quality_score(c) for c in cands]
        verdicts = [s[0] for s in scores]
        n_bad = sum(1 for v in verdicts if v == "bad")

        if n_bad == n_cand:
            ranks = [n_cand] * n_cand
            sus = [True] * n_cand
            judge = "force-effort-allbad"
            n_all_bad += 1
        else:
            efforts = [effort_score(c) for c in cands]
            # bad candidates get effort=-1 to push to bottom
            adj = [(-1.0 if verdicts[i] == "bad" else efforts[i]) for i in range(n_cand)]
            order = sorted(range(n_cand), key=lambda i: -adj[i])
            ranks = [0] * n_cand
            for pos, idx in enumerate(order):
                ranks[idx] = pos + 1
            sus = [v == "bad" for v in verdicts]
            judge = "force-effort"

        sp_id = pend.get("system_prompt_id")
        rec = {
            "ts": time.time(), "schema": "rank-v1",
            "model": pend.get("model") or "?", "host": HOST,
            "prompt": pend.get("prompt", ""),
            "rankings": ranks,
            "candidates": cands,
            "errored": pend.get("errored") or [False] * n_cand,
            "suspect": sus,
            "temps": pend.get("temps"),
            "top_ps": pend.get("top_ps"),
            "min_ps": pend.get("min_ps"),
            "max_tokens": pend.get("max_tokens"),
            "system_prompt_ids": [sp_id] * n_cand if sp_id else None,
            "pending_id": pend["id"],
            "auto_rated": True,
            "auto_judge": judge,
            "auto_confidence": "low",
            "auto_reasoning": "; ".join(
                f"c{i}={s[0]} eff={effort_score(cands[i]):.2f}"
                for i, s in enumerate(scores)),
        }
        auto_recs.append(rec)
        auto_reviewed.append(pend["id"])
        n_committed += 1

    if auto_recs:
        with PREFS.open("a") as f:
            for r in auto_recs: f.write(json.dumps(r) + "\n")
        with REVIEWED.open("a") as f:
            for rid in auto_reviewed: f.write(rid + "\n")
    print(f"force-effort committed {n_committed} (of which {n_all_bad} all-bad)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
