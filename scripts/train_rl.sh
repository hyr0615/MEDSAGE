#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-sagrpo}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  grpo) WEIGHTING=false ;;
  sagrpo) WEIGHTING=true ;;
  *) echo "Usage: bash scripts/train_rl.sh {grpo|sagrpo} [key=value ...]" >&2; exit 2 ;;
esac
: "${EASYR1_ROOT:?Set EASYR1_ROOT to the prepared EasyR1 working copy}"
: "${SFT_MODEL_PATH:?Set SFT_MODEL_PATH to the shared LVKC-SFT checkpoint}"
: "${RL_TRAIN_FILE:?Set RL_TRAIN_FILE to the frozen training JSONL}"
: "${RL_VAL_FILE:?Set RL_VAL_FILE to an authorized development JSONL, not benchmark test data}"
export MEDSAGE_ROOT="$ROOT"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$MODE}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse an existing output path: $OUTPUT_DIR" >&2
  exit 2
fi
export PYTHONPATH="$EASYR1_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec python -m verl.trainer.main "config=$ROOT/configs/rl.yaml" \
  "algorithm.stage_aware_grpo=$WEIGHTING" \
  "algorithm.sa_outcome_aware_credit=$WEIGHTING" \
  "trainer.experiment_name=$MODE" "$@"
