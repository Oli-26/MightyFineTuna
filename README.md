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

Each rank record is tagged with `host` (machine) and `model`. Rater workflow:

1. Collect ranks locally on your machine — they land in `prefs.jsonl` (gitignored).
2. When ready to share, copy to `results/<your-name>-prefs.jsonl` and commit:
   ```bash
   cp prefs.jsonl results/oli-laptop-prefs.jsonl
   git add results/oli-laptop-prefs.jsonl
   git commit -m "data: add oli-laptop ranks"
   git push
   ```
3. Pull others' contributions: `git pull` brings in `results/*.jsonl`.
4. Aggregate everything:
   ```bash
   ./merge.py results/*.jsonl --out combined.jsonl
   ./merge.py --stats results/*.jsonl   # counts by host/model/schema
   ```

Dedup is on `(ts, prompt, sha1(candidates))`. Root `prefs.jsonl` stays private (raw, possibly messy); `results/*.jsonl` is the clean shared pool.

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

## Blind A/B eval (after you've fine-tuned)

Once you have a tuned student, run a blind eval to verify the tune actually changed model behavior in a way you can detect.

1. Boot the **base** model in llama-server, generate one side:
   ```bash
   ./blind_eval.py --side base --url http://127.0.0.1:8080 --prompts eval_prompts.txt
   ```
2. Stop, boot the **tuned** model in llama-server, generate the other side:
   ```bash
   ./blind_eval.py --side tuned --url http://127.0.0.1:8080 --prompts eval_prompts.txt
   ```
3. Pair them up:
   ```bash
   ./blind_eval.py --pair side_base.jsonl side_tuned.jsonl
   ```
4. Open <http://127.0.0.1:8000/blind> — you'll see two canvases side by side per prompt and pick which is the tuned model. Truth is hidden until session end (10 pairs). Reveal shows accuracy + a verdict on signal strength.

`./blind_eval.py --stats` reports overall accuracy across all sessions.

`eval_prompts.txt` is intentionally **disjoint** from `prompts.txt` so you're testing generalization, not memorization of the training distribution.

## Notes

- Model file paths are **not** committed. Set `MODEL` env var or edit `run.sh`.
- `prefs.jsonl` is **not** committed (private to each rater) — see `.gitignore`.
- Picker iframe sandbox uses `sandbox="allow-scripts"` only (no same-origin) — model output runs isolated.
