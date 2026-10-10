"""Outcome-aware V/K/C allocation using the shared trajectory advantage."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def build_stage_similarity_matrix(
    stage_texts: Sequence[str],
    similarity_fn: Callable[[str, str], float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build pairwise similarities and a nonempty-stage mask."""
    group_size = len(stage_texts)
    similarities = torch.zeros((group_size, group_size), dtype=torch.float32)
    valid = torch.tensor([bool((text or "").strip()) for text in stage_texts], dtype=torch.bool)
    for row in range(group_size):
        if valid[row]:
            similarities[row, row] = 1.0
        for col in range(row + 1, group_size):
            if not valid[row] or not valid[col]:
                continue
            value = float(similarity_fn(stage_texts[row], stage_texts[col]))
            if not torch.isfinite(torch.tensor(value)):
                value = 0.0
            value = max(0.0, min(1.0, value))
            similarities[row, col] = value
            similarities[col, row] = value
    return similarities, valid


def disagreement_from_similarity(similarities: torch.Tensor, valid: torch.Tensor) -> float:
    """Return 1 - mean pairwise similarity over valid, non-self pairs."""
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() < 2:
        return 0.0
    submatrix = similarities[valid_indices][:, valid_indices]
    pair_mask = torch.triu(torch.ones_like(submatrix, dtype=torch.bool), diagonal=1)
    pair_values = submatrix[pair_mask]
    if pair_values.numel() == 0:
        return 0.0
    disagreement = 1.0 - float(pair_values.mean().item())
    return max(0.0, min(1.0, disagreement))


def compute_local_outcome_credit(
    similarities: torch.Tensor,
    valid: torch.Tensor,
    correctness: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute leave-one-out success-minus-failure similarity per rollout.

    Returns ``(credit, fallback_mask)`` with shape ``[G]``.  A fallback entry
    has exactly zero credit because its current stage is invalid or either
    leave-one-out outcome peer set is empty.
    """
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("similarities must have shape [G, G]")
    group_size = similarities.shape[0]
    valid = valid.to(dtype=torch.bool, device=similarities.device)
    correctness = correctness.to(dtype=torch.bool, device=similarities.device)
    if valid.shape != (group_size,) or correctness.shape != (group_size,):
        raise ValueError("valid and correctness must have shape [G]")

    credits = torch.zeros(group_size, dtype=torch.float32, device=similarities.device)
    fallback = torch.ones(group_size, dtype=torch.bool, device=similarities.device)
    all_indices = torch.arange(group_size, device=similarities.device)
    for row in range(group_size):
        if not bool(valid[row].item()):
            continue
        peer = all_indices != row
        success = peer & valid & correctness
        failure = peer & valid & ~correctness
        if not bool(success.any().item()) or not bool(failure.any().item()):
            continue
        sim_plus = similarities[row, success].mean()
        sim_minus = similarities[row, failure].mean()
        value = torch.clamp(sim_plus - sim_minus, min=-1.0, max=1.0)
        if not bool(torch.isfinite(value).item()):
            continue
        credits[row] = value
        fallback[row] = False
    return credits, fallback


def adjust_vkc_disagreement(
    base_vkc: torch.Tensor,
    local_vk_credit: torch.Tensor,
    global_advantage: torch.Tensor,
) -> torch.Tensor:
    """Apply sign-aware local calibration and leave conclusion unchanged."""
    if base_vkc.shape != (3,):
        raise ValueError("base_vkc must have shape [3] for V/K/C")
    if local_vk_credit.ndim != 2 or local_vk_credit.shape[1] != 2:
        raise ValueError("local_vk_credit must have shape [G, 2] for V/K")
    if global_advantage.shape != (local_vk_credit.shape[0],):
        raise ValueError("global_advantage must have shape [G]")
    if not torch.isfinite(base_vkc).all() or not torch.isfinite(local_vk_credit).all():
        raise ValueError("disagreement and local credit must be finite")
    if not torch.isfinite(global_advantage).all():
        raise ValueError("global advantage must be finite")

    adjusted = base_vkc.unsqueeze(0).expand(local_vk_credit.shape[0], -1).clone()
    adjusted[:, :2] = torch.clamp(
        base_vkc[:2].unsqueeze(0)
        * (1.0 + torch.sign(global_advantage).unsqueeze(1) * local_vk_credit),
        min=0.0,
        max=1.0,
    )
    return adjusted


def bounded_stage_allocation(
    vkc_disagreement: torch.Tensor,
    gamma: float,
    mix: float,
    lower: float,
    upper: float,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Compute bounded V/K/C multipliers with a row sum of three."""
    if vkc_disagreement.ndim != 2 or vkc_disagreement.shape[1] != 3:
        raise ValueError("vkc_disagreement must have shape [G, 3]")
    if not 0.0 <= mix <= 1.0:
        raise ValueError(f"mix must be in [0, 1], got {mix}")
    if not 0.0 < lower <= 1.0 <= upper:
        raise ValueError(f"bounds must satisfy 0 < lower <= 1 <= upper, got {lower}, {upper}")
    if not torch.isfinite(vkc_disagreement).all():
        raise ValueError("vkc_disagreement must be finite")

    dynamic = torch.softmax(vkc_disagreement.to(dtype=torch.float32) * gamma, dim=-1) * 3.0
    multipliers = torch.clamp(1.0 + mix * (dynamic - 1.0), min=lower, max=upper)
    for row in range(multipliers.shape[0]):
        for _ in range(4):
            correction = 3.0 - float(multipliers[row].sum().item())
            if abs(correction) < tolerance:
                break
            eligible = multipliers[row] < upper if correction > 0 else multipliers[row] > lower
            eligible_count = int(eligible.sum().item())
            if eligible_count == 0:
                break
            multipliers[row, eligible] += correction / eligible_count
            multipliers[row] = torch.clamp(multipliers[row], min=lower, max=upper)
    if not torch.isfinite(multipliers).all():
        raise RuntimeError("non-finite stage multiplier")
    return multipliers
