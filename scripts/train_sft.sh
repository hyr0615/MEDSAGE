#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-VL-7B-Instruct}"
: "${DATA_ROOT:?Set DATA_ROOT to the LLaMA-Factory dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/lvkc_sft}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output path already exists: $OUTPUT_DIR" >&2; exit 2; }
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
cd "$ROOT"
exec llamafactory-cli train configs/sft.yaml \
  "model_name_or_path=$MODEL_PATH" "dataset_dir=$DATA_ROOT" "media_dir=$DATA_ROOT" \
  "output_dir=$OUTPUT_DIR" "$@"
