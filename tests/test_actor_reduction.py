"""Exercise actor gradient accumulation with CPU tensors and stub collectives."""
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl.trainer.core_algos import average_loss, compute_kl, compute_policy_loss


class Batch:
    def __init__(self, tensors):
        self.batch = tensors
        self.non_tensor_batch = {}
        self.meta_info = {'temperature': 1.0}

    def select(self, keys, non_tensor_keys):
        return Batch({key: self.batch[key] for key in keys})

    def split(self, size):
        count = self.batch['responses'].shape[0]
        return [Batch({key: value[i:i+size] for key, value in self.batch.items()})
                for i in range(0, count, size)]


def actor_update(total_tokens):
    path = Path(__file__).resolve().parents[1] / 'easyr1/verl/workers/actor/dp_actor.py'
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'update_policy')
    method.decorator_list = []
    scope = {'torch': torch, 'DataProto': Batch, 'Any': object, 'defaultdict': defaultdict,
             'compute_policy_loss': compute_policy_loss, 'compute_kl': compute_kl,
             'average_loss': average_loss,
             'tqdm': lambda items, **kwargs: items,
             'dist': SimpleNamespace(all_reduce=lambda t, **kw: t.fill_(total_tokens),
                                     ReduceOp=SimpleNamespace(SUM=0)),
             'append_to_dict': lambda dest, source: [dest[k].append(v) for k, v in source.items()]}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actor-update>', 'exec'), scope)
    return scope['update_policy']


@pytest.mark.parametrize('world_size,micro_size', [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2)])
def test_actor_partitioning_matches_global_masked_objective(world_size, micro_size):
    mask = torch.tensor([[1., 1., 1.], [1., 0., 0.], [1., 1., 0.], [1., 1., 1.]], dtype=torch.float64)
    old = torch.full_like(mask, -2.)
    ref = old - .1
    advantage = torch.tensor([[1.], [-1.], [.5], [-.5]], dtype=torch.float64).expand_as(mask)
    weight = torch.tensor([[1.], [2.], [1.3], [1.7]], dtype=torch.float64).expand_as(mask)
    values = old + torch.tensor([[.1, -.1, .3], [.2, 0, 0], [-.3, .1, 0], [.1, .1, -.1]])
    accumulated = torch.zeros_like(values)
    total_tokens = mask.sum().item()
    update = actor_update(total_tokens)
    rank_size = 4 // world_size
    for rank in range(world_size):
        module = torch.nn.Module()
        module.logp = torch.nn.Parameter(values.clone())
        indices = torch.arange(rank * rank_size, (rank + 1) * rank_size)
        batch = Batch({'input_ids': indices[:, None], 'attention_mask': mask[indices],
                       'position_ids': mask[indices], 'responses': torch.zeros_like(mask[indices]),
                       'response_mask': mask[indices], 'old_log_probs': old[indices],
                       'ref_log_probs': ref[indices], 'advantages': advantage[indices],
                       'sa_token_weights': weight[indices]})
        actor = SimpleNamespace(actor_module=module, rank=rank, world_size=world_size,
            config=SimpleNamespace(global_batch_size_per_device=rank_size, ppo_epochs=1,
                dynamic_batching=False, micro_batch_size_per_device_for_update=micro_size,
                clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_dual=0., loss_type='default',
                loss_avg_mode='token', use_kl_loss=True, kl_penalty='low_var_kl', kl_coef=.03),
            _forward_micro_batch=lambda inputs, temperature: module.logp[inputs['input_ids'][:, 0]],
            _optimizer_step=lambda: module.logp.grad.norm())
        update(actor, batch)
        accumulated += module.logp.grad / world_size
    expected_logp = values.clone().requires_grad_()
    pg, _ = compute_policy_loss(old, expected_logp, advantage, mask, .2, .2, 0., 'default', 'token', weight)
    kl = average_loss(compute_kl(expected_logp, ref, 'low_var_kl'), mask, 'token')
    (pg + .03 * kl).backward()
    torch.testing.assert_close(accumulated, expected_logp.grad, atol=1e-7, rtol=1e-6)
