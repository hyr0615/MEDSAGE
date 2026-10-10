from __future__ import annotations

import torch

from verl.trainer.sa_outcome_credit import (
    adjust_vkc_disagreement,
    bounded_stage_allocation,
    build_stage_similarity_matrix,
    compute_local_outcome_credit,
    disagreement_from_similarity,
)


GAMMA = 1.25
MIX = 0.25
LOWER = 0.85
UPPER = 1.15


def exact_similarity(left: str, right: str) -> float:
    return float(left == right)


def matrix_and_valid(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    return build_stage_similarity_matrix(texts, exact_similarity)


def weights(base: torch.Tensor, credit: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    adjusted = adjust_vkc_disagreement(base, credit, advantage)
    return bounded_stage_allocation(adjusted, GAMMA, MIX, LOWER, UPPER)


def test_1_all_correct_falls_back_and_exactly_matches_legacy_weights() -> None:
    similarity, valid = matrix_and_valid(["a", "a", "b", "b"])
    credit, fallback = compute_local_outcome_credit(similarity, valid, torch.ones(4, dtype=torch.bool))
    assert torch.equal(credit, torch.zeros_like(credit))
    assert bool(fallback.all())
    base = torch.tensor([0.5, 0.2, 0.4])
    advantage = torch.tensor([1.0, -1.0, 0.0, 0.5])
    new = weights(base, torch.stack([credit, credit], dim=1), advantage)
    legacy = bounded_stage_allocation(base.expand(4, -1), GAMMA, MIX, LOWER, UPPER)
    assert torch.equal(new, legacy)


def test_2_all_incorrect_falls_back_and_exactly_matches_legacy_weights() -> None:
    similarity, valid = matrix_and_valid(["a", "a", "b", "b"])
    credit, fallback = compute_local_outcome_credit(similarity, valid, torch.zeros(4, dtype=torch.bool))
    assert torch.equal(credit, torch.zeros_like(credit))
    assert bool(fallback.all())
    base = torch.tensor([0.5, 0.2, 0.4])
    advantage = torch.tensor([1.0, -1.0, 0.0, 0.5])
    new = weights(base, torch.stack([credit, credit], dim=1), advantage)
    legacy = bounded_stage_allocation(base.expand(4, -1), GAMMA, MIX, LOWER, UPPER)
    assert torch.equal(new, legacy)


def test_3_success_and_failure_clusters_have_opposite_credit() -> None:
    similarity, valid = matrix_and_valid(["success", "success", "failure", "failure"])
    correctness = torch.tensor([True, True, False, False])
    credit, fallback = compute_local_outcome_credit(similarity, valid, correctness)
    assert not bool(fallback.any())
    assert bool((credit[:2] > 0).all())
    assert bool((credit[2:] < 0).all())
    assert torch.equal(credit, torch.tensor([1.0, 1.0, -1.0, -1.0]))


def test_4_positive_advantage_adjusts_in_expected_direction() -> None:
    base = torch.tensor([0.4, 0.6, 0.2])
    credit = torch.tensor([[0.5, -0.5]])
    adjusted = adjust_vkc_disagreement(base, credit, torch.tensor([1.0]))
    assert adjusted[0, 0] >= base[0]
    assert adjusted[0, 1] <= base[1]
    assert adjusted[0, 2] == base[2]


def test_5_negative_advantage_adjusts_in_expected_direction() -> None:
    base = torch.tensor([0.4, 0.6, 0.2])
    credit = torch.tensor([[0.5, -0.5]])
    adjusted = adjust_vkc_disagreement(base, credit, torch.tensor([-1.0]))
    assert adjusted[0, 0] <= base[0]
    assert adjusted[0, 1] >= base[1]
    assert adjusted[0, 2] == base[2]


def test_6_invalid_or_empty_stage_uses_zero_credit_without_error() -> None:
    similarity, valid = matrix_and_valid(["success", "", "failure", "failure"])
    correctness = torch.tensor([True, True, False, False])
    credit, fallback = compute_local_outcome_credit(similarity, valid, correctness)
    assert credit[1] == 0.0
    assert bool(fallback[1])
    assert torch.isfinite(credit).all()
    assert 0.0 <= disagreement_from_similarity(similarity, valid) <= 1.0


def test_7_weights_are_bounded_and_sum_to_three_per_rollout() -> None:
    base = torch.tensor([1.0, 0.8, 0.1])
    credit = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.2, -0.7], [0.0, 0.0]])
    advantage = torch.tensor([2.0, 1.0, -3.0, 0.0])
    multiplier = weights(base, credit, advantage)
    assert bool((multiplier >= LOWER - 1e-7).all())
    assert bool((multiplier <= UPPER + 1e-7).all())
    assert torch.allclose(multiplier.sum(dim=1), torch.full((4,), 3.0), atol=1e-6, rtol=0.0)


def test_8_effective_advantage_always_preserves_global_sign() -> None:
    base = torch.tensor([0.9, 0.4, 0.2])
    credit = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.3, 0.3]])
    advantage = torch.tensor([1.5, -0.7, 0.0])
    multiplier = weights(base, credit, advantage)
    effective = multiplier * advantage.unsqueeze(1)
    assert torch.equal(torch.sign(effective), torch.sign(advantage).unsqueeze(1).expand_as(effective))

