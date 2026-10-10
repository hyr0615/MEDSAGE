"""Use the same answer rules for trajectory and stage-C correctness."""

from __future__ import annotations

try:
    from .accuracy_reward import accuracy_reward, explicit_mc_choice
except ImportError:
    from accuracy_reward import accuracy_reward, explicit_mc_choice


def stage_c_correctness(response: str, ground_truth: str) -> float:
    return accuracy_reward(response, ground_truth)


def mc_final_binary_correct(response: str, ground_truth: str) -> bool:
    return stage_c_correctness(response, ground_truth) == 1.0
