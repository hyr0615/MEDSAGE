#!/usr/bin/env python3
"""Generate reproducible same-prompt DPO pairs from frozen LVKC-SFT rollouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time
from collections import Counter

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dpo_core import (  # noqa: E402
    assert_unique_same_sample,
    is_forbidden_benchmark_test_row,
    select_same_prompt_pair,
    structurally_valid,
    to_sharegpt_pair,
    public_preference_record,
)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reward(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("medsage_existing_reward", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import reward: {path}")
    sys.path.insert(0, str(path.parent))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def build_messages(prompt: str) -> list[dict]:
    content = []
    for index, text in enumerate(prompt.split("<image>")):
        if index:
            content.append({"type": "image"})
        if text:
            content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--processor", required=True)
    p.add_argument("--reward", required=True)
    p.add_argument("--group-size", type=int, default=6)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-response-length", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=3072)
    p.add_argument("--max-pixels", type=int, default=262144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    input_path = pathlib.Path(args.input).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_all = [json.loads(line) for line in input_path.open(encoding="utf-8")]
    forbidden = [row for row in rows_all if is_forbidden_benchmark_test_row(row)]
    rows = [row for row in rows_all if not is_forbidden_benchmark_test_row(row)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no compliant RL rows remain")
    if any(is_forbidden_benchmark_test_row(row) for row in rows):
        raise RuntimeError("benchmark test labels reached DPO generation")

    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    processor.image_processor.max_pixels = args.max_pixels
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length,
        seed=args.seed,
    )
    scorer = load_reward(pathlib.Path(args.reward).resolve())
    pairs = []
    skip = Counter()
    started = time.time()
    raw_path = output_dir / "rollouts_and_rewards.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            requests = []
            for row in batch:
                messages = build_messages(row["prompt"])
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images = []
                for image_path in row.get("images") or []:
                    image = Image.open(image_path).convert("RGB")
                    if image.width * image.height > args.max_pixels:
                        scale = (args.max_pixels / (image.width * image.height)) ** 0.5
                        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
                    images.append(image)
                requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
            outputs = llm.generate(requests, sampling, use_tqdm=False)
            for row, output in zip(batch, outputs):
                rollouts = [candidate.text for candidate in output.outputs]
                reward_inputs = [{"response": text, "ground_truth": row["reward_ground_truth"]} for text in rollouts]
                scores = scorer(reward_inputs)
                pair = select_same_prompt_pair(row, rollouts, scorer)
                raw.write(json.dumps({"sample_id": row["sample_id"], "rollouts": rollouts, "scores": scores}, ensure_ascii=False) + "\n")
                if pair is None:
                    valid_count = sum(
                        1 for score in scores if structurally_valid(score, row["reward_ground_truth"])
                    )
                    skip["fewer_than_two_valid"] += valid_count < 2
                    skip["equal_reward_or_no_strict_pair"] += valid_count >= 2
                else:
                    pairs.append(to_sharegpt_pair(row, pair))
            print(json.dumps({"processed": min(offset + len(batch), len(rows)), "pairs": len(pairs), "skipped": dict(skip)}), flush=True)

    assert_unique_same_sample(pairs)
    pair_path = output_dir / "preferences.json"
    public_pairs = [public_preference_record(pair) for pair in pairs]
    pair_path.write_text(json.dumps(public_pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    margins = [row["chosen_reward"] - row["rejected_reward"] for row in pairs]
    manifest = {
        "status": "COMPLETE",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "model": args.model,
        "processor": args.processor,
        "reward_implementation": str(pathlib.Path(args.reward).resolve()),
        "reward_weights": {"accuracy": 0.6, "iou": 0.3, "format": 0.1},
        "seed": args.seed,
        "generation": {
            "group_size": args.group_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_response_length": args.max_response_length,
            "max_model_len": args.max_model_len,
            "max_pixels": args.max_pixels,
        },
        "source_rows": len(rows_all),
        "excluded_benchmark_test_rows": len(forbidden),
        "eligible_clean_rows": len(rows),
        "chosen_gt_rejected_pair_count": len(pairs),
        "skipped_equal_or_insufficient_valid_count": sum(skip.values()),
        "skip_reasons": dict(skip),
        "chosen_reward_mean": sum(x["chosen_reward"] for x in pairs) / len(pairs) if pairs else None,
        "rejected_reward_mean": sum(x["rejected_reward"] for x in pairs) / len(pairs) if pairs else None,
        "reward_margin_mean": sum(margins) / len(margins) if margins else None,
        "elapsed_seconds": time.time() - started,
        "preference_dataset": str(pair_path),
        "rollout_audit": str(raw_path),
    }
    (output_dir / "preference_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
