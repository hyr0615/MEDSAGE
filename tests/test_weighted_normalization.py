"""Gradient checks for weighted microbatch reduction."""
import torch
from verl.trainer.core_algos import compute_policy_loss


def gradient(weights):
    weights = torch.tensor(weights, dtype=torch.float64)
    log_probs = torch.zeros_like(weights, requires_grad=True)
    loss, _ = compute_policy_loss(
        old_log_probs=torch.zeros_like(weights), log_probs=log_probs,
        advantages=torch.ones_like(weights), response_mask=torch.ones_like(weights),
        clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_dual=0.0,
        loss_type='default', loss_avg_mode='token', token_loss_weights=weights,
    )
    loss.backward()
    return log_probs.grad


def test_uniform_single_response_weight_scales_gradient():
    assert torch.allclose(gradient([[1, 1]]) * 2, gradient([[2, 2]]), atol=1e-8, rtol=0)


def test_stage_weights_change_within_response_gradient():
    result = gradient([[0.85, 1.15]])
    assert torch.allclose(result[0, 0] / result[0, 1], torch.tensor(0.85 / 1.15, dtype=torch.float64))


def test_separate_microbatches_match_joint_weighting():
    together = gradient([[1, 1], [2, 2]])
    separately = torch.cat([gradient([[1, 1]]), gradient([[2, 2]])]) / 2
    assert torch.allclose(together, separately, atol=1e-8, rtol=0)
