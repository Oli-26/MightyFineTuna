#!/usr/bin/env bash
# Orchestrator: boots picker stack 5 times (one per system-prompt variant),
# runs batch_gen --no-skip --limit 200 under each. Pending records get tagged
# with system_prompt_id so we can A/B compare quality.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PATH="/c/Program Files/AMD/ROCm/7.1/bin:$PATH"
export LLAMA_BIN="${LLAMA_BIN:-C:/Users/Oli/Documents/llama.cpp/llama-server.exe}"
export MODEL="${MODEL:-C:/Users/Oli/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
export MOE_CPU=1
export N_CPU_MOE=15

VARIANTS="${VARIANTS:-A B C D E}"

kill_stack() {
  for p in $(tasklist 2>/dev/null | awk '/llama-server\.exe|^uv\.exe|^python\.exe/ {print $2}'); do
    cmd.exe /c "taskkill /F /PID $p" >/dev/null 2>&1 || true
  done
  sleep 2
}

wait_for_picker() {
  for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

for v in $VARIANTS; do
  echo
  echo "[orch] ============================================"
  echo "[orch] === variant $v starting ==="
  echo "[orch] ============================================"
  kill_stack

  SYSTEM_PROMPT_ID=$v ./run.sh > "llama.log" 2>&1 &
  RUN_PID=$!
  echo "[orch] run.sh pid=$RUN_PID, waiting for picker..."

  if wait_for_picker; then
    echo "[orch] picker ready"
    cfg=$(curl -fsS http://127.0.0.1:8000/config 2>/dev/null)
    echo "[orch] /config: $cfg"
    case "$cfg" in
      *"\"system_prompt_id\":\"$v\""*) echo "[orch] confirmed SYSTEM_PROMPT_ID=$v" ;;
      *) echo "[orch] WARNING: /config does not show variant $v — proceeding anyway" ;;
    esac

    echo "[orch] starting batch for variant $v"
    PYTHONUTF8=1 ./batch_gen.py --no-skip --limit 200 2>&1
    echo "[orch] variant $v batch done"
  else
    echo "[orch] picker did NOT come up under variant $v — skipping"
  fi
done

kill_stack
echo
echo "[orch] === all variants complete ==="
