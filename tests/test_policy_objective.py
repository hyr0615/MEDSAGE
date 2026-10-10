"""Check the documented clipped objective against CPU values and gradients."""
import math

import pytest
import torch

from verl.trainer.core_algos import compute_policy_loss


@pytest.mark.parametrize('weighted', [False, True])
def test_documented_token_objective_matches_loss_and_gradient(weighted):
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]], dtype=torch.float64)
    old = torch.full_like(mask, -5)
    delta = torch.tensor([[math.log(.5), math.log(1.4), math.log(1.1)],
                          [math.log(10), math.log(.6), 0]], dtype=torch.float64)
    advantage = torch.tensor([[1., 1., 1.], [-1., -1., -1.]], dtype=torch.float64)
    weights = torch.tensor([[1., 1.2, .9], [2., 1.7, 1.]], dtype=torch.float64) if weighted else None
    actual_logp = (old + delta).requires_grad_()
    actual, _ = compute_policy_loss(old, actual_logp, advantage, mask,
                                     .2, .2, 0., 'default', 'token', weights)
    expected_logp = (old + delta).requires_grad_()
    ratio = (expected_logp - old).exp()
    clipped = ratio.clamp(.8, 1.2)
    token_loss = -torch.minimum(advantage * ratio, advantage * clipped)
    weighted_mask = mask if weights is None else mask * weights
    expected = (weighted_mask * token_loss).sum() / (mask.sum() + 1e-8)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_logp.grad, expected_logp.grad)


@pytest.mark.parametrize('ratio,advantage,expected', [(1.4, 1., -1.2), (10., -1., 10.), (.5, -1., .8)])
def test_configured_symmetric_clip_without_dual_clip(ratio, advantage, expected):
    logp = torch.tensor([[math.log(ratio)]], dtype=torch.float64)
    loss, _ = compute_policy_loss(torch.zeros_like(logp), logp, torch.full_like(logp, advantage),
                                  torch.ones_like(logp), .2, .2, 0., 'default', 'token')
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_old_policy_advantage_and_weights_are_detached():
    new = torch.tensor([[-2.]], requires_grad=True)
    old = torch.tensor([[-2.]], requires_grad=True)
    adv = torch.tensor([[1.]], requires_grad=True)
    weights = torch.tensor([[1.5]], requires_grad=True)
    loss, _ = compute_policy_loss(old, new, adv, torch.ones_like(new),
                                  .2, .2, 0., 'default', 'token', weights)
    loss.backward()
    assert new.grad is not None
    assert old.grad is adv.grad is weights.grad is None


def test_default_ratio_is_not_capped_at_exp_20():
    new = torch.tensor([[21.]], dtype=torch.float64, requires_grad=True)
    loss, _ = compute_policy_loss(torch.zeros_like(new), new, -torch.ones_like(new),
                                  torch.ones_like(new), .2, .2, 0., 'default', 'token')
    assert loss.item() == pytest.approx(math.exp(21), rel=1e-7)


def test_nonfinite_ratio_fails_explicitly():
    new = torch.tensor([[1000.]], requires_grad=True)
    with pytest.raises(FloatingPointError, match='Non-finite policy ratio'):
        compute_policy_loss(torch.zeros_like(new), new, -torch.ones_like(new),
                            torch.ones_like(new), .2, .2, 0., 'default', 'token')
