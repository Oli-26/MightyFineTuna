#!/usr/bin/env bash
# Boot llama-server + FastAPI picker. Ctrl-C kills both.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp-bin/llama-b8931/llama-server}"
MODEL="${MODEL:-$HOME/models/qwen2.5-coder-14b-q4_k_m.gguf}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
APP_PORT="${PORT:-8000}"

# tune for 12GB VRAM (RTX 5070 Ti laptop). 14B Q4_K_M ~9GB; full GPU offload fits w/ headroom.
NGL="${NGL:-99}"          # all layers on GPU
CTX="${CTX:-4096}"
PARALLEL="${PARALLEL:-3}"  # 3 concurrent candidate gens; drop to 1 for huge MoE models

# MoE expert offload to CPU. Set MOE_CPU=1 for big MoE models that don't fit VRAM.
# Keeps attention/embeddings on GPU, experts in RAM. Critical for Qwen3-Next-80B-A3B.
MOE_CPU="${MOE_CPU:-}"

# Speculative decoding: 0.5B draft for 14B target (same Qwen2.5-Coder tokenizer).
# OFF by default — measured to HURT picker workload (creative code, low draft accept ~47%).
# Re-enable with DRAFT=$HOME/models/qwen2.5-coder-0.5b-instruct-q8_0.gguf for code-similar prompts.
DRAFT="${DRAFT:-}"
NGLD="${NGLD:-99}"          # draft layers on GPU
DRAFT_MAX="${DRAFT_MAX:-8}" # max tokens to draft per step
DRAFT_MIN="${DRAFT_MIN:-1}"
DRAFT_P_MIN="${DRAFT_P_MIN:-0.6}"  # abort draft if any token's prob < this

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "llama-server not found at $LLAMA_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$MOE_CPU" ]]; then
  EXTRA_ARGS+=( --cpu-moe )
  echo "[run] MoE expert offload to CPU enabled"
fi

# Boot llama-server in background
SPEC_ARGS=()
if [[ -n "$DRAFT" ]]; then
  if [[ ! -f "$DRAFT" ]]; then
    echo "draft model not found: $DRAFT" >&2
    exit 1
  fi
  SPEC_ARGS+=( -md "$DRAFT" -ngld "$NGLD" --draft-max "$DRAFT_MAX" --draft-min "$DRAFT_MIN" --draft-p-min "$DRAFT_P_MIN" )
  echo "[run] spec decoding ON: draft=$DRAFT max=$DRAFT_MAX p_min=$DRAFT_P_MIN"
else
  echo "[run] spec decoding OFF"
fi

echo "[run] starting llama-server ($MODEL) on :$LLAMA_PORT (ctx=$CTX np=$PARALLEL cache-reuse=256)"
"$LLAMA_BIN" \
  -m "$MODEL" \
  --host 127.0.0.1 --port "$LLAMA_PORT" \
  -ngl "$NGL" \
  -c "$CTX" \
  -np "$PARALLEL" \
  --cache-reuse 256 \
  --jinja \
  "${EXTRA_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  > "$HERE/llama.log" 2>&1 &
LLAMA_PID=$!
echo "[run] llama-server pid=$LLAMA_PID (logs: $HERE/llama.log)"

cleanup() {
  echo "[run] stopping..."
  kill "$LLAMA_PID" 2>/dev/null || true
  wait "$LLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for llama-server health
echo "[run] waiting for llama-server to come up..."
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1; then
    echo "[run] llama-server ready"
    break
  fi
  sleep 1
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "[run] llama-server died — last 40 lines of log:" >&2
    tail -40 "$HERE/llama.log" >&2
    exit 1
  fi
done

# Run FastAPI in foreground
echo "[run] starting picker on http://127.0.0.1:$APP_PORT"
LLAMA_URL="http://127.0.0.1:$LLAMA_PORT" PORT="$APP_PORT" \
  uv run "$HERE/server.py"
