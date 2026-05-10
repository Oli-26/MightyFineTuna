#!/usr/bin/env bash
# Move checkpoints between laptop and PC. Adapter dirs are ~120MB each.
#
# Set REMOTE to the other machine's hostname/user. Examples:
#   ./sync_checkpoints.sh push                # send local checkpoints/ to PC
#   ./sync_checkpoints.sh pull                # fetch PC's checkpoints/ here
#   ./sync_checkpoints.sh push 3b-parent_b_v2 # only one adapter
#
# rsync skips identical files; safe to run repeatedly.

set -euo pipefail
REMOTE="${REMOTE:-oli@oli-rx7800xt.local}"   # override per machine
REPO_PATH="${REPO_PATH:-Documents/Code/canvas-picker}"
DIR="${2:-checkpoints/}"

cd "$(dirname "$0")"

case "${1:-}" in
  push)
    echo "[sync] $DIR  ->  $REMOTE:$REPO_PATH/"
    rsync -avz --partial --progress \
      --exclude '*-merged/' --exclude 'checkpoint-*/' \
      "$DIR" "$REMOTE:$REPO_PATH/$DIR"
    ;;
  pull)
    echo "[sync] $REMOTE:$REPO_PATH/$DIR  ->  $DIR"
    rsync -avz --partial --progress \
      --exclude '*-merged/' --exclude 'checkpoint-*/' \
      "$REMOTE:$REPO_PATH/$DIR" "$DIR"
    ;;
  *)
    echo "usage: $0 {push|pull} [path-under-repo]"
    echo "  REMOTE env var sets the other machine (default $REMOTE)"
    exit 1
    ;;
esac
echo "[sync] done"
