#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SFT_MODEL_PATH:?Set SFT_MODEL_PATH to the shared LVKC-SFT checkpoint}"
: "${DATA_ROOT:?Set DATA_ROOT to the LLaMA-Factory dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/dpo}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output path already exists: $OUTPUT_DIR" >&2; exit 2; }
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
cd "$ROOT"
exec llamafactory-cli train configs/dpo.yaml \
  "model_name_or_path=$SFT_MODEL_PATH" "ref_model=$SFT_MODEL_PATH" \
  "dataset_dir=$DATA_ROOT" "media_dir=$DATA_ROOT" "output_dir=$OUTPUT_DIR" "$@"
