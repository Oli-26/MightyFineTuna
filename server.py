#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi", "uvicorn[standard]", "httpx"]
# ///
"""
Canvas picker: prompt -> N candidate canvas-JS -> user picks -> JSONL log.
Talks to a local llama-server (OpenAI-compatible) on LLAMA_URL (default :8080).
"""
import asyncio
import json
import os
import re
import socket
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
PREFS = ROOT / "prefs.jsonl"
PENDING = ROOT / "pending.jsonl"
REVIEWED = ROOT / "reviewed.txt"  # one pending id per line
EVAL_PAIRS = ROOT / os.environ.get("EVAL_PAIRS", "eval_pairs.jsonl")
EVAL_JUDGMENTS = ROOT / os.environ.get("EVAL_JUDGMENTS", "eval_judgments.jsonl")

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "3"))
# Wider pool than the fixed [0.4, 0.75, 1.05]. Each round samples N_CANDIDATES
# distinct temps from the pool — more variance per round → stronger preference signal.
TEMP_POOL = [float(t) for t in os.environ.get("TEMP_POOL", "0.3,0.5,0.7,0.9,1.1,1.3").split(",")]
TEMPS = TEMP_POOL[:N_CANDIDATES]  # legacy: still expose first N for /config
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1200"))
REQ_TIMEOUT = float(os.environ.get("REQ_TIMEOUT", "600"))
HOST_TAG = os.environ.get("HOST_TAG", socket.gethostname())

_HARD_RULES = """Hard rules — follow EXACTLY:
1. Write TOP-LEVEL STATEMENTS only. They execute immediately.
2. Do NOT wrap your code in `function foo() { ... }`. If you define a function, also CALL it on the next line.
3. Do NOT redeclare `ctx`, `W`, `H`, or `canvas`. Use them as-is.
4. Do NOT output prose, markdown fences (```), <script> tags, HTML, or `document.getElementById`.
5. No network, no external assets, no infinite loops.
"""

_WRAPPER_NOTE = """Your code is inserted directly inside this wrapper:
    const ctx = canvas.getContext('2d');
    const W = 400, H = 400;
    try {
        // <-- YOUR CODE GOES HERE (executes immediately)
    } catch(e) { ... }
"""

# 5 system-prompt variants for A/B comparison. Each batch round picks one via
# SYSTEM_PROMPT_ID env var; pending records tag which variant produced them.
SYSTEM_PROMPT_VARIANTS = {
    "A": f"""You output a JavaScript snippet that draws on an HTML canvas.

{_WRAPPER_NOTE}
{_HARD_RULES}
Style — composition-first pixel art:
- Integer coordinates; blocky shapes via fillRect (sparingly: arc/ellipse).
- Limited palette (~4-8 distinct colors). Pick a coherent palette for the subject.
- ALWAYS paint a full background first (sky, ground, or solid mood color) — never leave the canvas blank or mostly-empty.
- Compose deliberately: place the subject off-center where it helps; suggest foreground/midground/background with overlapping shapes.
- Use shading: a darker tone for shadow sides, a lighter tone for highlights, to give depth.
- Every prompt should produce a recognizable scene, not a single shape on a blank field.

Now produce the code for the user's prompt. Output JavaScript only.
""",

    "B": f"""You output a JavaScript snippet that draws on an HTML canvas.

{_WRAPPER_NOTE}
{_HARD_RULES}
Style — strict 8-bit retro pixel art on a 10-pixel grid:
- Use ONLY ctx.fillRect and ctx.fillStyle. No arc, no ellipse, no curves, no lineTo, no paths.
- ALL coordinates and sizes are multiples of 10. Snap to a 10-pixel grid (so the canvas is conceptually a 40×40 grid of 10×10 cells).
- Pick exactly 6 colors at the top of the code as constants (e.g. C1='#...', C2='#...', through C6).
- Always paint a background first using one of the constants — solid fill or two horizontal bands.
- Build the subject from chunky 10×10, 20×20, 30×30, or 40×40 rectangles. No detail finer than 10 pixels.

Now produce the code for the user's prompt. Output JavaScript only.
""",

    "C": f"""You output a JavaScript snippet that draws on an HTML canvas.

{_WRAPPER_NOTE}
{_HARD_RULES}
Style — palette-and-mood-led pixel art:
- BEFORE drawing, choose a 5-color palette suited to the subject's mood (warm sunset, cool moonlight, muted forest, vivid candy, somber rainy, dusty desert, etc).
- Declare the palette at the top using semantic constant names: SKY, GROUND, ACCENT, SHADOW, HIGHLIGHT (or similar suited to the subject).
- Reuse the palette consistently — don't introduce ad-hoc fillStyles mid-code.
- The first thing drawn establishes the mood (sky/ground/atmosphere); then layer the subject on top.
- Compose freely — fillRect preferred but use arc/ellipse where they serve the mood (sun, eyes, foliage).

Now produce the code for the user's prompt. Output JavaScript only.
""",

    "D": f"""You output a JavaScript snippet that draws on an HTML canvas.

{_WRAPPER_NOTE}
{_HARD_RULES}
Style — minimalist pixel art:
- Maximum 4 colors total.
- Maximum 30 draw calls. Do not exceed.
- Lots of negative space — fill no more than 40% of the canvas with subject shapes.
- Single subject, centered or rule-of-thirds.
- Solid backgrounds only, no gradients, no banding, no texture.
- No tiny details under 8 pixels.
- The image should read clearly when squinting from across a room.

Now produce the code for the user's prompt. Output JavaScript only.
""",

    "E": f"""You output a JavaScript snippet that draws on an HTML canvas.

{_WRAPPER_NOTE}
{_HARD_RULES}
Style — rich, detailed maximalist pixel art:
- Use 8-12 colors for variety. Define them as named constants up top.
- Layer the scene: distant background, mid-ground (terrain/water/sky elements), foreground subject, plus small decorative details (stars, leaves, sparkles, texture).
- Add at least 3 secondary elements beyond the main subject (e.g. for "a cat sitting", also draw a pillow, a window frame, ambient lighting, a small toy).
- Use shading: shadow tones on subject undersides, highlights where light catches.
- Aim for visual density — every region of the canvas should have something interesting.
- Use both rectangles and curves freely (arc, ellipse, paths) to build texture.

Now produce the code for the user's prompt. Output JavaScript only.
""",
}

SYSTEM_PROMPT_ID = os.environ.get("SYSTEM_PROMPT_ID", "A")
if SYSTEM_PROMPT_ID not in SYSTEM_PROMPT_VARIANTS:
    raise SystemExit(f"unknown SYSTEM_PROMPT_ID={SYSTEM_PROMPT_ID!r}, must be one of {list(SYSTEM_PROMPT_VARIANTS)}")
SYSTEM_PROMPT = SYSTEM_PROMPT_VARIANTS[SYSTEM_PROMPT_ID]

app = FastAPI()

_model_name_cache: str | None = None


async def _get_model_name(client: httpx.AsyncClient) -> str:
    global _model_name_cache
    if _model_name_cache:
        return _model_name_cache
    try:
        r = await client.get(f"{LLAMA_URL}/v1/models", timeout=3.0)
        data = r.json()
        # llama-server may return either OpenAI shape ({"data":[{"id":...}]})
        # or Ollama-ish ({"models":[{"name":..., "model":...}]}). Handle both.
        for m in data.get("data", []):
            if m.get("id"):
                _model_name_cache = m["id"]; return _model_name_cache
        for m in data.get("models", []):
            cand = m.get("model") or m.get("name")
            if cand:
                _model_name_cache = cand; return _model_name_cache
        return "unknown"
    except Exception:
        return "unknown"


class GenReq(BaseModel):
    prompt: str


class GenOneReq(BaseModel):
    prompt: str
    index: int | None = None  # legacy: maps into TEMPS[index]
    temp: float | None = None  # preferred: explicit temperature
    top_p: float | None = None
    min_p: float | None = None
    max_tokens: int | None = None


class PickReq(BaseModel):
    prompt: str
    chosen: int | None  # index of chosen candidate, or None for "none of these"
    candidates: list[str]


class RateReq(BaseModel):
    prompt: str
    rankings: list[int]  # 1 = best, larger = worse; ties allowed (e.g., two errored both at N)
    candidates: list[str]
    errored: list[bool]
    suspect: list[bool]
    temps: list[float] | None = None
    top_ps: list[float | None] | None = None
    min_ps: list[float | None] | None = None
    max_tokens: list[int | None] | None = None
    system_prompt_ids: list[str | None] | None = None
    pending_id: str | None = None  # if from review queue, mark this as reviewed


def _temps_for(n: int) -> list[float]:
    if n <= len(TEMPS):
        return TEMPS[:n]
    # extend by spreading
    return TEMPS + [round(0.6 + 0.1 * i, 2) for i in range(n - len(TEMPS))]


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:javascript|js|html|jsx|typescript|ts)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
# Drop lines that redeclare names already provided as formal parameters in the
# iframe wrapper (`ctx`, `W`, `H`, `canvas`, `c`). Such redeclarations cause
# "redeclaration of formal parameter" SyntaxErrors that the runtime catch can't catch.
_REDECL_RE = re.compile(r"^\s*(?:const|let|var)\s+(?:ctx|W|H|canvas|c)\b")
# Heuristic: code with NO drawing primitives almost certainly renders blank.
_DRAW_RE = re.compile(
    r"\b(?:fillRect|strokeRect|clearRect|fillText|strokeText|fill\(|stroke\(|"
    r"drawImage|putImageData|createImageData|getImageData|arc\(|rect\(|"
    r"moveTo|lineTo|bezierCurveTo|quadraticCurveTo|ellipse\()"
)


def _has_draw_calls(code: str) -> bool:
    return bool(_DRAW_RE.search(code))


def _strip_fences(s: str) -> str:
    # Strip <think>...</think> blocks (Qwen3, R1) before fence extraction.
    s = _THINK_RE.sub("", s).strip()
    matches = _FENCE_RE.findall(s)
    if matches:
        s = max(matches, key=len).strip()
    elif s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    return _sanitize(s)


def _sanitize(code: str) -> str:
    """Strip redeclarations of injected names; collapse leading blanks."""
    kept = [ln for ln in code.split("\n") if not _REDECL_RE.match(ln)]
    return "\n".join(kept).strip()


async def _one_completion(
    client: httpx.AsyncClient,
    prompt: str,
    temperature: float,
    top_p: float | None = None,
    min_p: float | None = None,
    max_tokens: int | None = None,
) -> str:
    payload: dict = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
        "stream": False,
        "cache_prompt": True,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if min_p is not None:
        payload["min_p"] = min_p
    r = await client.post(f"{LLAMA_URL}/v1/chat/completions", json=payload, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return _strip_fences(text)


@app.post("/generate_one")
async def generate_one(req: GenOneReq):
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    if req.temp is not None:
        temp = req.temp
    elif req.index is not None:
        temps = _temps_for(N_CANDIDATES)
        if req.index < 0 or req.index >= len(temps):
            raise HTTPException(400, f"index out of range 0..{len(temps) - 1}")
        temp = temps[req.index]
    else:
        raise HTTPException(400, "must provide either 'temp' or 'index'")
    async with httpx.AsyncClient() as client:
        try:
            code = await _one_completion(
                client, req.prompt, temp,
                top_p=req.top_p, min_p=req.min_p, max_tokens=req.max_tokens,
            )
        except Exception as e:
            return {"index": req.index, "temp": temp, "code": "", "error": f"{type(e).__name__}: {e}"}
    return {
        "index": req.index, "temp": temp, "code": code,
        "top_p": req.top_p, "min_p": req.min_p, "max_tokens": req.max_tokens,
        "system_prompt_id": SYSTEM_PROMPT_ID,
        "suspect": not _has_draw_calls(code),
    }


@app.get("/config")
async def config():
    return {
        "n": N_CANDIDATES,
        "temps": _temps_for(N_CANDIDATES),
        "temp_pool": TEMP_POOL,
        "system_prompt_id": SYSTEM_PROMPT_ID,
        "system_prompt_variants": list(SYSTEM_PROMPT_VARIANTS.keys()),
    }


@app.post("/generate")
async def generate(req: GenReq):
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    temps = _temps_for(N_CANDIDATES)
    async with httpx.AsyncClient() as client:
        try:
            results = await asyncio.gather(
                *[_one_completion(client, req.prompt, t) for t in temps],
                return_exceptions=True,
            )
        except Exception as e:
            raise HTTPException(502, f"llama-server error: {e}")
    candidates = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            candidates.append(f"// generation error: {type(r).__name__}: {r}\nctx.fillStyle='#400';ctx.fillRect(0,0,W,H);ctx.fillStyle='#f88';ctx.font='12px monospace';ctx.fillText('GEN ERROR',10,20);")
        else:
            candidates.append(r)
    return {"candidates": candidates, "temps": temps}


@app.post("/pick")
async def pick(req: PickReq):
    rec = {
        "ts": time.time(),
        "schema": "pick-v1",
        "prompt": req.prompt,
        "chosen": req.chosen,
        "candidates": req.candidates,
    }
    with PREFS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"ok": True, "logged": str(PREFS)}


def _find_pending(pending_id: str) -> dict | None:
    for rec in _iter_pending():
        if rec.get("id") == pending_id:
            return rec
    return None


@app.post("/rate")
async def rate(req: RateReq):
    n = len(req.candidates)
    if not (len(req.rankings) == n == len(req.errored) == len(req.suspect)):
        raise HTTPException(400, "rankings/errored/suspect length must match candidates")

    # Hydrate sampler/sp metadata from the original pending record if this came
    # from the review queue — pending records have per-candidate temps/top_p/min_p/
    # max_tokens/sp_id that the picker UI normally drops on submit.
    pend = _find_pending(req.pending_id) if req.pending_id else None
    temps = req.temps if req.temps is not None else (pend.get("temps") if pend else None)
    top_ps = req.top_ps if req.top_ps is not None else (pend.get("top_ps") if pend else None)
    min_ps = req.min_ps if req.min_ps is not None else (pend.get("min_ps") if pend else None)
    max_tokens = req.max_tokens if req.max_tokens is not None else (pend.get("max_tokens") if pend else None)
    sp_ids = req.system_prompt_ids
    if sp_ids is None:
        if pend:
            pend_sp = pend.get("system_prompt_id")
            if pend_sp is not None:
                sp_ids = [pend_sp] * n
        else:
            # direct gen path — picker used the live SYSTEM_PROMPT_ID for every cand
            sp_ids = [SYSTEM_PROMPT_ID] * n

    if pend:
        model_name = pend.get("model") or "?"
    else:
        async with httpx.AsyncClient() as client:
            model_name = await _get_model_name(client)

    rec = {
        "ts": time.time(),
        "schema": "rank-v1",
        "model": model_name,
        "host": HOST_TAG,
        "prompt": req.prompt,
        "rankings": req.rankings,
        "candidates": req.candidates,
        "errored": req.errored,
        "suspect": req.suspect,
        "temps": temps,
        "top_ps": top_ps,
        "min_ps": min_ps,
        "max_tokens": max_tokens,
        "system_prompt_ids": sp_ids,
        "pending_id": req.pending_id,
    }
    with PREFS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if req.pending_id:
        _mark_reviewed(req.pending_id)
    return {"ok": True}


def _reviewed_ids() -> set[str]:
    if not REVIEWED.exists():
        return set()
    return {ln.strip() for ln in REVIEWED.read_text().splitlines() if ln.strip()}


def _mark_reviewed(pending_id: str) -> None:
    with REVIEWED.open("a") as f:
        f.write(pending_id + "\n")


def _iter_pending() -> list[dict]:
    import glob
    paths = []
    if PENDING.exists():
        paths.append(str(PENDING))
    paths.extend(sorted(glob.glob(str(ROOT / "results" / "*-pending.jsonl"))))
    seen_ids: set[str] = set()
    out = []
    for fp in paths:
        for line in Path(fp).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rid = rec.get("id")
            if rid and rid in seen_ids:
                continue
            if rid:
                seen_ids.add(rid)
            out.append(rec)
    return out


@app.get("/pending/next")
async def pending_next():
    """Return the next unreviewed pending record + counts. Empty payload if queue done."""
    done = _reviewed_ids()
    pending = _iter_pending()
    remaining = [p for p in pending if p.get("id") not in done]
    if not remaining:
        return {"empty": True, "total": len(pending), "remaining": 0}
    return {
        "empty": False,
        "record": remaining[0],
        "total": len(pending),
        "remaining": len(remaining),
        "reviewed": len(pending) - len(remaining),
    }


@app.post("/pending/skip")
async def pending_skip(body: dict):
    """Mark a pending record reviewed AND log a negative rank-v1 record:
    all candidates tied at worst rank with suspect=True so train_sft.py skips
    them and analyze_prefs counts them as flagged. Treats 'skip' as 'all terrible'.
    """
    pid = body.get("id")
    if not pid:
        raise HTTPException(400, "missing id")
    pend = _find_pending(pid)
    if pend:
        n = len(pend.get("candidates") or [])
        rec = {
            "ts": time.time(),
            "schema": "rank-v1",
            "model": pend.get("model") or "?",
            "host": HOST_TAG,
            "prompt": pend.get("prompt", ""),
            "rankings": [n] * n,                  # everyone tied at worst rank
            "candidates": pend.get("candidates") or [],
            "errored": pend.get("errored") or [False] * n,
            "suspect": [True] * n,                # mark all as suspect → trainer drops
            "temps": pend.get("temps"),
            "top_ps": pend.get("top_ps"),
            "min_ps": pend.get("min_ps"),
            "max_tokens": pend.get("max_tokens"),
            "system_prompt_ids": [pend.get("system_prompt_id")] * n if pend.get("system_prompt_id") else None,
            "pending_id": pid,
            "skip_reason": "all_terrible",
        }
        with PREFS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    _mark_reviewed(pid)
    return {"ok": True, "logged_negative": bool(pend)}


@app.get("/history")
async def history():
    if not PREFS.exists():
        return {"records": []}
    out = []
    with PREFS.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return {"records": out[-50:]}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{LLAMA_URL}/health", timeout=3.0)
            return {"backend": r.status_code, "url": LLAMA_URL, "n": N_CANDIDATES}
    except Exception as e:
        return JSONResponse({"backend": "down", "url": LLAMA_URL, "error": str(e)}, status_code=503)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/blind")
async def blind_index():
    return FileResponse(STATIC / "blind.html")


# ---- Blind A/B eval ----

import secrets


def _load_pairs() -> list[dict]:
    if not EVAL_PAIRS.exists():
        return []
    return [json.loads(l) for l in EVAL_PAIRS.read_text().splitlines() if l.strip()]


def _judged_pair_ids() -> set[str]:
    if not EVAL_JUDGMENTS.exists():
        return set()
    return {
        json.loads(l).get("pair_id")
        for l in EVAL_JUDGMENTS.read_text().splitlines()
        if l.strip()
    }


@app.get("/blind/next")
async def blind_next(skip: str = ""):
    """Return one unjudged record. Handles 2-way ('blind-pair-v1') and N-way
    ('blind-multi-v1') schemas. Slots returned in random order; truth labels
    sent so the client can echo them back for scoring.
    """
    pairs = _load_pairs()
    if not pairs:
        return {"empty": True, "reason": "no eval_pairs.jsonl yet — run blind_eval.py to populate"}
    judged = _judged_pair_ids()
    skipped = set(s for s in skip.split(",") if s)
    remaining = [p for p in pairs if p["id"] not in judged and p["id"] not in skipped]
    if not remaining:
        return {"empty": True, "total": len(pairs), "reason": "all pairs judged or skipped"}
    pair = remaining[0]
    schema = pair.get("schema", "blind-pair-v1")

    if schema == "blind-multi-v1":
        # N-way: shuffle candidate order, send slots with hidden truth labels
        cands = pair["candidates"]
        n = len(cands)
        order = list(range(n))
        secrets.SystemRandom().shuffle(order)
        slots = [
            {"slot": i, "code": cands[order[i]]["code"], "_label": cands[order[i]]["label"]}
            for i in range(n)
        ]
        return {
            "empty": False,
            "schema": "blind-multi-v1",
            "pair_id": pair["id"],
            "prompt": pair["prompt"],
            "slots": slots,
            "n": n,
            "remaining": len(remaining),
            "total": len(pairs),
            "judged": len(judged),
        }

    # Legacy 2-way schema
    left_is = "base" if secrets.randbelow(2) == 0 else "tuned"
    right_is = "tuned" if left_is == "base" else "base"
    return {
        "empty": False,
        "schema": "blind-pair-v1",
        "pair_id": pair["id"],
        "prompt": pair["prompt"],
        "left_code":  pair[left_is]["code"],
        "right_code": pair[right_is]["code"],
        "_left_is": left_is,
        "_right_is": right_is,
        "remaining": len(remaining),
        "total": len(pairs),
        "judged": len(judged),
    }


class BlindJudgeReq(BaseModel):
    pair_id: str
    # 2-way fields:
    picked: str | None = None       # "left" | "right" | "unsure"
    left_is: str | None = None      # "base" or "tuned"
    # N-way fields:
    rankings: list[int] | None = None  # rank per slot (1=best, len=worst); ties allowed
    slot_labels: list[str] | None = None  # echoed: model label per slot index


@app.post("/blind/judge")
async def blind_judge(req: BlindJudgeReq):
    pair = next((p for p in _load_pairs() if p["id"] == req.pair_id), None)
    if not pair:
        raise HTTPException(404, f"pair {req.pair_id} not found")
    schema = pair.get("schema", "blind-pair-v1")

    if schema == "blind-multi-v1":
        if req.rankings is None or req.slot_labels is None:
            raise HTTPException(400, "rankings + slot_labels required for blind-multi-v1")
        if len(req.rankings) != len(pair["candidates"]) or len(req.slot_labels) != len(pair["candidates"]):
            raise HTTPException(400, "length mismatch")
        # Build per-label rank
        per_label = {req.slot_labels[i]: req.rankings[i] for i in range(len(req.rankings))}
        rec = {
            "ts": time.time(),
            "schema": "blind-judge-multi-v1",
            "pair_id": req.pair_id,
            "prompt": pair["prompt"],
            "rankings": req.rankings,
            "slot_labels": req.slot_labels,
            "per_label_rank": per_label,
            "models_per_label": {c["label"]: c.get("model") for c in pair["candidates"]},
        }
        with EVAL_JUDGMENTS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return {"ok": True, "per_label_rank": per_label, "models_per_label": rec["models_per_label"]}

    # Legacy 2-way
    if req.picked not in ("left", "right", "unsure"):
        raise HTTPException(400, "picked must be left/right/unsure")
    if req.left_is not in ("base", "tuned"):
        raise HTTPException(400, "left_is must be base/tuned")
    right_is = "tuned" if req.left_is == "base" else "base"
    picked_side: str | None = None
    correct: bool | None = None
    if req.picked == "left":
        picked_side = req.left_is
    elif req.picked == "right":
        picked_side = right_is
    if picked_side is not None:
        correct = picked_side == "tuned"
    rec = {
        "ts": time.time(),
        "schema": "blind-judge-v1",
        "pair_id": req.pair_id,
        "prompt": pair["prompt"],
        "left_is": req.left_is,
        "right_is": right_is,
        "picked": req.picked,
        "picked_side": picked_side,
        "correct": correct,
        "base_model": pair.get("base", {}).get("model"),
        "tuned_model": pair.get("tuned", {}).get("model"),
    }
    with EVAL_JUDGMENTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"ok": True, "correct": correct, "left_is": req.left_is, "right_is": req.right_is if hasattr(req, 'right_is') else right_is}


@app.get("/blind/stats")
async def blind_stats():
    if not EVAL_JUDGMENTS.exists():
        return {"total": 0, "decided": 0, "correct": 0, "accuracy": None}
    js = [json.loads(l) for l in EVAL_JUDGMENTS.read_text().splitlines() if l.strip()]

    # 2-way (legacy)
    twos = [j for j in js if j.get("schema", "blind-judge-v1") == "blind-judge-v1"]
    decided = [j for j in twos if j.get("picked") in ("left", "right")]
    correct = sum(1 for j in decided if j.get("correct"))
    n = len(decided)
    acc = correct / n if n else None
    z = None
    if n:
        se = (0.25 / n) ** 0.5
        z = (acc - 0.5) / se if se else 0

    # N-way (multi)
    multis = [j for j in js if j.get("schema") == "blind-judge-multi-v1"]
    win_count: dict[str, int] = {}
    rank_sum: dict[str, float] = {}
    n_count: dict[str, int] = {}
    for j in multis:
        for label, rank in (j.get("per_label_rank") or {}).items():
            n_count[label] = n_count.get(label, 0) + 1
            rank_sum[label] = rank_sum.get(label, 0.0) + rank
            if rank == 1:
                win_count[label] = win_count.get(label, 0) + 1
    multi_per_label = {
        label: {
            "rounds": n_count[label],
            "wins": win_count.get(label, 0),
            "win_rate": win_count.get(label, 0) / n_count[label] if n_count[label] else None,
            "avg_rank": rank_sum[label] / n_count[label] if n_count[label] else None,
        }
        for label in n_count
    }

    return {
        "total": len(js),
        "two_way": {
            "decided": n,
            "unsure": len(twos) - n,
            "correct": correct,
            "accuracy": acc,
            "z_vs_chance": z,
        },
        "multi_way": {
            "rounds": len(multis),
            "per_label": multi_per_label,
        },
        "pairs_total": len(_load_pairs()),
    }


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
