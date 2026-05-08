# MightyFineTuna

Small framework for collecting human preference data on **canvas-drawing code** — to fine-tune a smaller model that draws pixel-art-style images via JavaScript canvas API.

A local LLM produces N candidate canvas-JS snippets per prompt; you rank them best→worst; ranks land in `prefs.jsonl` as DPO/SFT-ready preference data.

## What's in here

| File | Purpose |
|---|---|
| `run.sh` | Boots `llama-server` (your model) + the FastAPI picker UI together. |
| `server.py` | FastAPI: serves UI, proxies to `llama-server`, logs ranks to `prefs.jsonl`. |
| `static/index.html` | Picker UI: side-by-side iframes, rank buttons, review-queue mode. |
| `batch_gen.py` | Headless: takes `prompts.txt`, generates 3 candidates per prompt, appends to `pending.jsonl` for offline review. |
| `merge.py` | Merges `prefs.jsonl` files from multiple machines/raters with dedup + stats. |
| `prompts.txt` | Seed prompt list — edit freely. |

## Hardware

Tested on:
- RTX 5070 Ti laptop (12GB VRAM) + 62GB RAM, Linux, Vulkan llama.cpp
- Should work on any rig that runs `llama-server` (CUDA/ROCm/Vulkan/Metal/CPU).

For 14B-class models you want ~10GB VRAM; for 80B+ MoE models you want CPU MoE offload (`MOE_CPU=1`) and ≥32GB RAM.

## Prereqs

1. **uv** (for inline-script Python deps):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **llama.cpp**: download a recent release with `llama-server` binary. Note the path.
3. **A GGUF model**. Recommended for the teacher/data-generator:
   - `Qwen3-Coder-Next-IQ4_XS.gguf` (~40GB, MoE — needs `MOE_CPU=1`)
   - `Qwen2.5-Coder-14B-Q4_K_M.gguf` (~9GB, dense — fits 12GB VRAM)
   - Or anything else strong at JS/HTML.

## Run

```bash
# Set your paths (or edit run.sh defaults)
export LLAMA_BIN=$HOME/path/to/llama.cpp/llama-server
export MODEL=$HOME/path/to/your-model.gguf

# For huge MoE models (e.g. Qwen3-Coder-Next), keep experts in RAM:
# export MOE_CPU=1 PARALLEL=1

# Optional: tag your records with your name (otherwise hostname is used)
# export HOST_TAG=oli

./run.sh
```

Open <http://127.0.0.1:8000>.

Type a prompt → "Generate" → 3 candidates render → click "1 best" / "2" / "3 worst" buttons (or click frames in order) → "Submit ranks". Errored / blank-canvas candidates are auto-marked worst.

## Batch generation (overnight)

Don't want to babysit? Generate a queue, rank later:

```bash
./batch_gen.py                   # uses ./prompts.txt → ./pending.jsonl
./batch_gen.py --limit 20        # cap to 20 fresh prompts
./batch_gen.py --url http://OTHER-MACHINE:8000   # use someone else's picker
```

Then in the UI, click **Review queue (X / Y)** — the picker walks you through each pending record, you rank, submit auto-loads the next.

`batch_gen.py` skips prompts already in `pending.jsonl` so you can rerun safely.

## Multi-rater / multi-machine

Each rank record is tagged with `host` (machine) and `model`. So friends with different rigs/models can run their own picker, send back their `prefs.jsonl`, and you merge:

```bash
./merge.py prefs.jsonl friend1-prefs.jsonl friend2-prefs.jsonl --out combined.jsonl
./merge.py --stats combined.jsonl   # report counts by model/host/schema
```

Dedup is on `(ts, prompt, sha1(candidates))`.

## Data schema

`prefs.jsonl` — append-only. Each line is one ranking event:

```jsonc
{
  "ts": 1778232661.34,
  "schema": "rank-v1",
  "model": "Qwen3-Coder-Next-IQ4_XS.gguf",
  "host": "my-laptop",
  "prompt": "a cute cat sitting",
  "rankings": [1, 2, 3],         // rank per candidate, 1=best
  "candidates": ["...js code...", "...", "..."],
  "errored":   [false, false, false],
  "suspect":   [false, false, false],   // no draw calls in code = blank canvas
  "temps":     [0.4, 0.75, 1.05],
  "pending_id": null              // set if rated from review queue
}
```

`pending.jsonl` — same shape minus rankings; the queue of unrated candidates.

`reviewed.txt` — one pending `id` per line, tracks what's been rated.

## Notes

- Model file paths are **not** committed. Set `MODEL` env var or edit `run.sh`.
- `prefs.jsonl` is **not** committed (private to each rater) — see `.gitignore`.
- Picker iframe sandbox uses `sandbox="allow-scripts"` only (no same-origin) — model output runs isolated.
