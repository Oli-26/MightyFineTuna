#!/usr/bin/env bash
# Train 4 parent LoRAs on Qwen2.5-Coder-3B-Instruct with different data subsets.
# Sequential. ~10-15 min each. Output: checkpoints/3b-parent_{a,b,c,d}.
set -uo pipefail

cd "$(dirname "$0")"

BASE="Qwen/Qwen2.5-Coder-3B-Instruct"
COMMON=(--model "$BASE" --epochs 2 --lr 1e-4 --rank 16 --alpha 32 --batch 1 --accum 8)

run_parent() {
  local name="$1"; shift
  local logf="train_${name}.log"
  echo "=== START $name $(date +%H:%M:%S) ==="
  ./train_sft.py "${COMMON[@]}" --output "3b-${name}" "$@" 2>&1 | tee "$logf"
  echo "=== END $name $(date +%H:%M:%S) ==="
}

# parent_a: full data, spA (baseline composition)
run_parent "parent_a" --system-prompt-id A --max-seq 1024

# parent_b: full data, spF (few-shot grafted) — bigger prompt, max_seq up
run_parent "parent_b" --system-prompt-id F --max-seq 2048

# parent_c: Qwen3-Coder-Next teacher only (high-quality minority subset, ~114)
run_parent "parent_c" --system-prompt-id A --max-seq 1024 --filter-model "Qwen3-Coder-Next"

# parent_d: Qwen3-Coder-30B teacher only (large subset, ~855)
run_parent "parent_d" --system-prompt-id A --max-seq 1024 --filter-model "30B"

echo
echo "ALL DONE. Adapters in checkpoints/3b-parent_{a,b,c,d}"
