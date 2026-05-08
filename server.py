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

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "3"))
TEMPS = [0.4, 0.75, 1.05]  # one per candidate; pad/trim to N_CANDIDATES
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1200"))
REQ_TIMEOUT = float(os.environ.get("REQ_TIMEOUT", "600"))
HOST_TAG = os.environ.get("HOST_TAG", socket.gethostname())

SYSTEM_PROMPT = """You output a JavaScript snippet that draws on an HTML canvas.

Your code is inserted directly inside this wrapper:
    const ctx = canvas.getContext('2d');
    const W = 400, H = 400;
    try {
        // <-- YOUR CODE GOES HERE (executes immediately)
    } catch(e) { ... }

Rules — follow EXACTLY:
1. Write TOP-LEVEL STATEMENTS only. They execute immediately.
2. Do NOT wrap your code in `function foo() { ... }`. If you define a function, also CALL it on the next line.
3. Do NOT include placeholder comments like `// Your code here`. Write the actual drawing code.
4. Do NOT redeclare `ctx`, `W`, `H`, or `canvas`. Use them as-is.
5. Do NOT output prose, markdown fences (```), <script> tags, HTML, or `document.getElementById`.
6. No network, no external assets, no infinite loops.
7. Pixel-art style preferred: integer coords, blocky shapes, limited palette, fillRect.

Example of CORRECT output for "a red square":
    ctx.fillStyle = '#222';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#e44';
    ctx.fillRect(160, 160, 80, 80);

Now produce the code for the user's prompt. Output JavaScript only.
"""

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
    index: int  # which slot (0..N-1) — picks temperature


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


async def _one_completion(client: httpx.AsyncClient, prompt: str, temperature: float) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "cache_prompt": True,  # llama-server: reuse KV for matching prefix (the system prompt)
    }
    r = await client.post(f"{LLAMA_URL}/v1/chat/completions", json=payload, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return _strip_fences(text)


@app.post("/generate_one")
async def generate_one(req: GenOneReq):
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    temps = _temps_for(N_CANDIDATES)
    if req.index < 0 or req.index >= len(temps):
        raise HTTPException(400, f"index out of range 0..{len(temps) - 1}")
    temp = temps[req.index]
    async with httpx.AsyncClient() as client:
        try:
            code = await _one_completion(client, req.prompt, temp)
        except Exception as e:
            return {"index": req.index, "temp": temp, "code": "", "error": f"{type(e).__name__}: {e}"}
    return {"index": req.index, "temp": temp, "code": code, "suspect": not _has_draw_calls(code)}


@app.get("/config")
async def config():
    return {"n": N_CANDIDATES, "temps": _temps_for(N_CANDIDATES)}


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


@app.post("/rate")
async def rate(req: RateReq):
    n = len(req.candidates)
    if not (len(req.rankings) == n == len(req.errored) == len(req.suspect)):
        raise HTTPException(400, "rankings/errored/suspect length must match candidates")
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
        "temps": req.temps,
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
    if not PENDING.exists():
        return []
    out = []
    for line in PENDING.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
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
    pid = body.get("id")
    if not pid:
        raise HTTPException(400, "missing id")
    _mark_reviewed(pid)
    return {"ok": True}


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


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
