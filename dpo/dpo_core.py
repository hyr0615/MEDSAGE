"""Pure utilities for the controlled MEDSAGE sequence-level DPO baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass(frozen=True)
class PairSelection:
    chosen: str
    rejected: str
    chosen_score: dict[str, float]
    rejected_score: dict[str, float]


def parse_ground_truth(value: str | dict[str, Any]) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else value


def is_forbidden_benchmark_test_row(row: dict[str, Any]) -> bool:
    """Reject labels injected from the five benchmark test partitions."""
    return (
        row.get("source") == "BENCHMARK_TEST_ORACLE"
        or row.get("contamination_stage") == "RL"
        or bool(row.get("benchmark"))
    )


def structurally_valid(score: dict[str, float], ground_truth: str | dict[str, Any]) -> bool:
    """Apply exactly the validity conditions exposed by the existing reward."""
    gt = parse_ground_truth(ground_truth)
    if float(score.get("format", 0.0)) != 1.0:
        return False
    conclusion_required = bool(gt.get("full_LVKC")) or gt.get("task_type") in {"full_LVKC", "C"}
    if conclusion_required and float(score.get("explicit_conclusion", 0.0)) != 1.0:
        return False
    availability = gt.get("stage_availability") or {}
    localization_required = (
        bool(gt.get("full_LVKC"))
        or gt.get("task_type") in {"full_LVKC", "L"}
        or bool(availability.get("L"))
    )
    if localization_required and float(score.get("bbox_count_match", 0.0)) != 1.0:
        return False
    return True


def select_same_prompt_pair(
    row: dict[str, Any],
    rollouts: list[str],
    scorer: Callable[[list[dict[str, Any]]], list[dict[str, float]]],
) -> PairSelection | None:
    if is_forbidden_benchmark_test_row(row):
        raise ValueError(f"benchmark-test label is forbidden: {row.get('sample_id')}")
    inputs = [{"response": text, "ground_truth": row["reward_ground_truth"]} for text in rollouts]
    scores = scorer(inputs)
    if len(scores) != len(rollouts):
        raise ValueError("reward returned a different number of scores")
    valid = [
        (text, score)
        for text, score in zip(rollouts, scores)
        if text and structurally_valid(score, row["reward_ground_truth"])
    ]
    if len(valid) < 2:
        return None
    chosen_text, chosen_score = max(valid, key=lambda x: float(x[1]["overall"]))
    rejected_text, rejected_score = min(valid, key=lambda x: float(x[1]["overall"]))
    if not float(chosen_score["overall"]) > float(rejected_score["overall"]):
        return None
    return PairSelection(chosen_text, rejected_text, chosen_score, rejected_score)


def to_sharegpt_pair(row: dict[str, Any], pair: PairSelection) -> dict[str, Any]:
    """Create LLaMAFactory's multimodal pairwise schema and retain provenance."""
    return {
        "sample_id": row["sample_id"],
        "images": list(row.get("images") or []),
        "question": row["prompt"],
        "conversations": [{"from": "human", "value": row["prompt"]}],
        "chosen": {"from": "gpt", "value": pair.chosen},
        "rejected": {"from": "gpt", "value": pair.rejected},
        "chosen_reward": float(pair.chosen_score["overall"]),
        "rejected_reward": float(pair.rejected_score["overall"]),
        "chosen_reward_components": pair.chosen_score,
        "rejected_reward_components": pair.rejected_score,
        "source": row.get("source"),
        "modality": row.get("modality"),
    }


def response_mask(labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """True exactly on response labels; prompt and padding are ignored."""
    return labels.ne(ignore_index)


def public_preference_record(pair: dict[str, Any]) -> dict[str, Any]:
    """Export standard ShareGPT fields only; keep scoring metadata internal."""
    return {key: pair[key] for key in ('conversations', 'chosen', 'rejected', 'images')}


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Sum causal-token log probabilities over response tokens only."""
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = response_mask(shifted_labels, ignore_index)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = shifted_logits.log_softmax(-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(-1)


def standard_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    return -F.logsigmoid(beta * (policy_margin - reference_margin))


def assert_unique_same_sample(rows: Iterable[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"duplicate preference pair for sample_id={sample_id}")
        seen.add(sample_id)
        if not row.get("chosen") or not row.get("rejected"):
            raise ValueError(f"incomplete pair for sample_id={sample_id}")
